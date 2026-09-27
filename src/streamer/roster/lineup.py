"""Lineup optimisation that maximises the probability of winning the matchup.

Maximising expected points is the wrong objective in a head-to-head week. If
you are the underdog you want variance -- a boom-or-bust play beats a steady
one when only a boom wins -- and if you are the favourite you want to choke it
off. So every candidate lineup is scored by Monte Carlo against the
opponent's lineup, and the one with the highest **P(win)** is recommended.

What makes the variance real rather than cosmetic is how the draws are made:

* **Correlated.** Players in the same game move together -- a quarterback
  and his receivers (+0.3), a quarterback and the defence facing him (-0.44),
  both quarterbacks in a shootout (+0.18). Stacking raises your spread;
  starting the receiver who catches your opponent's quarterback's passes
  lowers the spread of the *difference*. Measured 2021-2025; see
  :mod:`streamer.roster.outcome`.
* **Skewed.** A projection of 6 has a floor near zero and a long right tail;
  the draws follow the measured shape by position and level, not a normal.
* **Injury tags are a coin flip.** A Questionable player either plays (full
  distribution) or scores zero -- a genuinely high-variance play.

Tested on 2024-25 data: player "types" (touchdown-dependent, deep threat,
target hog) did not predict weekly spread beyond position and projection
level, so no such labels are used.

The search is exhaustive over valid slot assignments (with a light candidate
prune per slot), and every lineup is evaluated on one shared sample matrix.
The best candidates are then re-scored on a *fresh* sample, so a lineup that
only won by luck on the first draw cannot be recommended, and a lineup that
gives up projected points is only recommended when it wins more by a margin
the simulation can actually resolve.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from ..league.model import SLOT_ELIGIBILITY, PlayerRow
from . import outcome

#: Positions that can legitimately score below zero.
NEGATIVE_OK = ("DST", "K")

#: How many extra candidates beyond the slot count to consider per slot.
PRUNE_EXTRA = 3

#: Lineups re-scored on the confirmation sample.
CONFIRM_TOP = 30

#: The smallest P(win) edge worth giving up projected points for. Below it,
#: the higher-projected lineup is recommended.
MIN_WIN_EDGE = 0.005

#: Draws shipped to the page for the lineup editor (5,000 keeps P(win) to
#: about +-0.7 points while the page stays small).
EDITOR_SIMS = 5000

#: Correlation beyond which a pair is worth naming in an explanation.
NOTABLE_CORR = 0.15


@dataclass
class LineupResult:
    """One evaluated lineup."""

    starters: dict[str, list[PlayerRow]]        # slot -> players
    expected: float
    sd: float
    win_probability: float

    @property
    def player_ids(self) -> frozenset[str]:
        return frozenset(p.player_id for ps in self.starters.values() for p in ps)

    def flat(self) -> list[tuple[str, PlayerRow]]:
        return [(slot, p) for slot, ps in self.starters.items() for p in ps]


@dataclass
class Optimisation:
    """Everything the CLI and page need from one optimisation."""

    best_win: LineupResult
    best_ev: LineupResult
    current: LineupResult | None
    opponent: LineupResult | None
    changes: list[tuple[str, PlayerRow | None, PlayerRow]] = field(default_factory=list)
    n_lineups: int = 0
    n_sims: int = 0
    #: Why the recommendation starts someone projected below a benched player.
    reasons: list[str] = field(default_factory=list)
    #: Player id -> (15th, 85th percentile) of his simulated score.
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: A slice of the confirmation draws, for trying lineups by hand on the
    #: page: my non-IR players' scores (columns in ``sample_ids`` order) and
    #: the opponent's total, draw by draw.
    sample_ids: list[str] = field(default_factory=list)
    samples: np.ndarray | None = None
    opp_samples: np.ndarray | None = None

    @property
    def win_gain(self) -> float:
        if self.current is None:
            return 0.0
        return self.best_win.win_probability - self.current.win_probability


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def pair_correlation(a: PlayerRow, b: PlayerRow) -> float:
    """Residual correlation of two players' scores this week (0 if unrelated)."""
    if not (a.team and b.team and a.role and b.role) or a.player_id == b.player_id:
        return 0.0
    if a.team == b.team:
        return outcome.correlation(True, a.role, b.role)
    if a.nfl_opponent and a.nfl_opponent == b.team:
        return outcome.correlation(False, a.role, b.role)
    return 0.0


def correlation_matrix(players: list[PlayerRow]) -> np.ndarray:
    """Pairwise correlations, repaired to the nearest valid matrix if needed."""
    n = len(players)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = pair_correlation(players[i], players[j])
            if r:
                corr[i, j] = corr[j, i] = r
    w, v = np.linalg.eigh(corr)
    if w.min() < 1e-6:
        corr = (v * np.clip(w, 1e-6, None)) @ v.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    return corr


def _if_plays(p: PlayerRow) -> tuple[float, float, float]:
    """(mean, sd, chance he plays) -- the distribution *given* he suits up.

    ``projection`` already folds a Questionable tag in as a mixture; the
    simulation undoes that and flips the coin instead, which is what makes a
    questionable player the high-variance play he really is.
    """
    mean = float(p.projection or 0.0)
    sd = float(p.projection_sd or 0.0)
    q = float(p.play_probability if p.play_probability is not None else 1.0)
    if p.is_out or q <= 0.0:
        return 0.0, 0.0, 0.0
    if q < 1.0 and p.outcome_sd is not None and mean > 0:
        return mean / q, float(p.outcome_sd), q
    return mean, sd, 1.0


def sample_points(players: list[PlayerRow], n_sims: int, rng: np.random.Generator,
                  correlated: bool = True) -> np.ndarray:
    """``(n_sims, n_players)`` simulated scores.

    Correlated standard normals (a Gaussian copula) are mapped through each
    player's measured outcome shape, scaled by his projection and spread,
    floored at zero for skill positions (a receiver cannot score -4) but not
    for D/ST and K, and zeroed when an injury tag's coin comes up "sits".
    """
    n = len(players)
    if not n:
        return np.zeros((n_sims, 0))
    z = rng.standard_normal((n_sims, n))
    if correlated and n > 1:
        corr = correlation_matrix(players)
        try:
            root = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(corr)
            root = v * np.sqrt(np.clip(w, 0.0, None))
        z = z @ root.T
    u = norm.cdf(z)
    out = np.empty((n_sims, n))
    for j, p in enumerate(players):
        if p.actual_points is not None:
            out[:, j] = float(p.actual_points)      # the game is over
            continue
        mean, sd, q = _if_plays(p)
        if q <= 0.0:
            out[:, j] = 0.0
            continue
        probs, zq = outcome.shape(p.position, mean)
        out[:, j] = mean + max(sd, 1e-6) * np.interp(u[:, j], probs, zq)
        if p.position not in NEGATIVE_OK:
            out[:, j] = np.maximum(out[:, j], 0.0)
        if q < 1.0:
            out[:, j] *= rng.random(n_sims) < q
    return out


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
def _eligible(player: PlayerRow, slot: str) -> bool:
    if player.is_out:
        return False
    if player.eligible_slots and slot in player.eligible_slots:
        return True
    return player.position in SLOT_ELIGIBILITY.get(slot, ())


def enumerate_lineups(
    roster: list[PlayerRow], slots: dict[str, int], prune_extra: int = PRUNE_EXTRA
) -> list[dict[str, list[PlayerRow]]]:
    """All distinct valid lineups (as slot -> players), dedicated slots first.

    Two lineups that start the same set of players in different slot
    arrangements score identically, so they are collapsed to one.
    """
    ordered = sorted(
        slots.items(),
        key=lambda kv: (len(SLOT_ELIGIBILITY.get(kv[0], (kv[0],))), kv[0]),
    )
    ranked = sorted(roster, key=lambda p: -p.week_value)

    results: list[dict[str, list[PlayerRow]]] = []
    seen: set[frozenset[str]] = set()

    def recurse(i: int, used: set[str], chosen: dict[str, list[PlayerRow]]) -> None:
        if i == len(ordered):
            key = frozenset(used)
            if key not in seen:
                seen.add(key)
                results.append({k: list(v) for k, v in chosen.items()})
            return
        slot, count = ordered[i]
        pool = [p for p in ranked if p.player_id not in used and _eligible(p, slot)]
        pool = pool[: count + prune_extra]
        if len(pool) < count:
            # Cannot fill the slot; start what we can (an empty slot scores 0).
            for combo in itertools.combinations(pool, len(pool)):
                chosen[slot] = list(combo)
                recurse(i + 1, used | {p.player_id for p in combo}, chosen)
            chosen.pop(slot, None)
            return
        for combo in itertools.combinations(pool, count):
            chosen[slot] = list(combo)
            recurse(i + 1, used | {p.player_id for p in combo}, chosen)
        chosen.pop(slot, None)

    recurse(0, set(), {})
    return results


def best_by_key(
    roster: list[PlayerRow], slots: dict[str, int], key
) -> LineupResult:
    """The lineup that maximises ``sum(key(player))``, filled greedily.

    Dedicated slots are filled before flex ones, so a flex-eligible player is
    only spent on the flex when nothing else wants them. Exact for the usual
    one-flex structure. Used for "what would the market start?", where there
    is a single number per player and no distribution to simulate.
    """
    ordered = sorted(slots.items(),
                     key=lambda kv: (len(SLOT_ELIGIBILITY.get(kv[0], (kv[0],))), kv[0]))
    ranked = sorted(roster, key=key, reverse=True)
    used: set[str] = set()
    starters: dict[str, list[PlayerRow]] = {}
    total = 0.0
    for slot, count in ordered:
        picked: list[PlayerRow] = []
        for p in ranked:
            if len(picked) >= count:
                break
            if p.player_id in used or not _eligible(p, slot):
                continue
            picked.append(p)
            used.add(p.player_id)
            total += float(key(p))
        starters[slot] = picked
    return LineupResult(starters=starters, expected=round(total, 2), sd=0.0, win_probability=0.0)


def current_lineup(roster: list[PlayerRow], slots: dict[str, int]) -> dict[str, list[PlayerRow]] | None:
    """The lineup as currently set on the platform, if it is complete enough to score."""
    out: dict[str, list[PlayerRow]] = {s: [] for s in slots}
    for p in roster:
        if p.starting and p.slot in out:
            out[p.slot].append(p)
    if not any(out.values()):
        return None
    return out


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
def _totals(
    lineups: list[dict[str, list[PlayerRow]]], samples: np.ndarray, index: dict[str, int]
) -> np.ndarray:
    """``(n_lineups, n_sims)`` totals via one indicator-matrix multiply."""
    if not lineups:
        return np.zeros((0, samples.shape[0]))
    ind = np.zeros((len(lineups), samples.shape[1]))
    for i, lineup in enumerate(lineups):
        for ps in lineup.values():
            for p in ps:
                ind[i, index[p.player_id]] = 1.0
    return ind @ samples.T


def _win_rates(lineups, samples, index, opp_total, chunk: int = 256) -> np.ndarray:
    """P(win) per lineup, in chunks so thousands of lineups stay in memory."""
    out = np.empty(len(lineups))
    for lo in range(0, len(lineups), chunk):
        t = _totals(lineups[lo:lo + chunk], samples, index)
        out[lo:lo + chunk] = (t > opp_total).mean(axis=1) + 0.5 * (t == opp_total).mean(axis=1)
    return out


def _projected(lineup: dict[str, list[PlayerRow]]) -> float:
    """Projected total, with final scores in place of projections."""
    return float(sum(p.week_value for ps in lineup.values() for p in ps))


def _with_locks(roster: list[PlayerRow], slots: dict[str, int]) -> list[dict[str, list[PlayerRow]]]:
    """Every valid lineup that respects kickoffs: a starter whose game has
    begun keeps his slot, and a benched one cannot come in."""
    fixed: dict[str, list[PlayerRow]] = {}
    for p in roster:
        if p.locked and p.starting and p.slot in slots:
            fixed.setdefault(p.slot, []).append(p)
    free_slots = {s: n - len(fixed.get(s, [])) for s, n in slots.items()}
    free_slots = {s: n for s, n in free_slots.items() if n > 0}
    free = [p for p in roster if not p.locked]
    lineups = enumerate_lineups(free, free_slots) if free_slots else [{}]
    out = []
    for lu in lineups:
        merged = {s: list(fixed.get(s, [])) + list(lu.get(s, [])) for s in slots}
        out.append(merged)
    return out


def optimise(
    roster: list[PlayerRow],
    slots: dict[str, int],
    opponent_roster: list[PlayerRow] | None,
    n_sims: int = 20000,
    seed: int = 7,
    min_edge: float = MIN_WIN_EDGE,
) -> Optimisation:
    """Find the lineup that maximises P(win) against the opponent."""
    rng = np.random.default_rng(seed)
    # A player parked in an IR slot cannot be started without a roster move
    # (activation needs a free spot), so the lineup is chosen from the rest.
    roster = [p for p in roster if not p.in_ir_slot]
    if opponent_roster:
        opponent_roster = [p for p in opponent_roster if not p.in_ir_slot]
    mine = _with_locks(roster, slots)
    if not mine:
        raise ValueError("no valid lineup can be built from this roster")

    # The opponent is assumed to start their best expected-points lineup,
    # unless they have already set one, in which case that is what we face.
    opp_lineup: dict[str, list[PlayerRow]] | None = None
    if opponent_roster:
        opp_lineup = current_lineup(opponent_roster, slots)
        if opp_lineup is None or sum(len(v) for v in opp_lineup.values()) < sum(slots.values()):
            opp_lineups = enumerate_lineups(opponent_roster, slots)
            if opp_lineups:
                opp_lineup = max(opp_lineups, key=_projected)

    everyone = list(roster) + (list(opponent_roster) if opponent_roster else [])
    index = {p.player_id: i for i, p in enumerate(everyone)}

    # Select on one sample, confirm on a fresh one: picking the best of
    # thousands of lineups on the same draws that score them flatters
    # whichever got lucky.
    select = sample_points(everyone, n_sims, rng)
    opp_sel = _totals([opp_lineup], select, index)[0] if opp_lineup is not None else np.zeros(n_sims)
    wins_sel = _win_rates(mine, select, index, opp_sel)
    projected = np.array([_projected(lu) for lu in mine])
    ev_best = int(np.argmax(projected))
    top = list(np.argsort(-wins_sel)[:CONFIRM_TOP])
    if ev_best not in top:
        top.append(ev_best)

    confirm = sample_points(everyone, n_sims, rng)
    opp_total = _totals([opp_lineup], confirm, index)[0] if opp_lineup is not None else np.zeros(n_sims)
    cand = [mine[i] for i in top]
    totals = _totals(cand, confirm, index)
    win_ind = (totals > opp_total) + 0.5 * (totals == opp_total)
    wins = win_ind.mean(axis=1)
    leader = int(np.argmax(wins))
    # Anything the simulation cannot tell apart from the leader is a tie, and
    # ties go to projected points: never give up points for noise.
    tied = []
    for k in range(len(cand)):
        se = float(np.std(win_ind[leader] - win_ind[k]) / np.sqrt(n_sims))
        if wins[k] >= wins[leader] - max(min_edge, 2.0 * se):
            tied.append(k)
    pick = max(tied, key=lambda k: (round(projected[top[k]], 6), wins[k]))
    ev_k = top.index(ev_best)

    def result(k: int) -> LineupResult:
        return LineupResult(starters=cand[k], expected=float(projected[top[k]]),
                            sd=float(totals[k].std()), win_probability=float(wins[k]))

    best_win = result(pick)
    best_ev = result(ev_k)

    current = None
    cur = current_lineup(roster, slots)
    if cur is not None:
        t = _totals([cur], confirm, index)[0]
        current = LineupResult(
            starters=cur, expected=_projected(cur), sd=float(t.std()),
            win_probability=float((t > opp_total).mean() + 0.5 * (t == opp_total).mean()),
        )

    opponent = None
    if opp_lineup is not None:
        opponent = LineupResult(starters=opp_lineup, expected=_projected(opp_lineup),
                                sd=float(opp_total.std()), win_probability=1.0 - best_win.win_probability)

    ranges = {p.player_id: (float(np.percentile(confirm[:, index[p.player_id]], 15)),
                            float(np.percentile(confirm[:, index[p.player_id]], 85)))
              for p in everyone}
    reasons = explain(best_win, best_ev, opp_lineup, confirm, index, opp_total, ranges)

    keep = min(EDITOR_SIMS, n_sims)
    ids = [p.player_id for p in roster]
    return Optimisation(
        best_win=best_win, best_ev=best_ev, current=current, opponent=opponent,
        changes=diff_lineups(current.starters if current else None, best_win.starters),
        n_lineups=len(mine), n_sims=n_sims, reasons=reasons, ranges=ranges,
        sample_ids=ids, samples=confirm[:keep, [index[i] for i in ids]],
        opp_samples=opp_total[:keep],
    )


def _strongest(pool: list[PlayerRow], who: PlayerRow, sign: int,
               skip: tuple[str, ...]) -> tuple[float, PlayerRow | None]:
    """The pool member most correlated with ``who`` in direction ``sign``."""
    scored = [(pair_correlation(who, m), m) for m in pool if m.player_id not in skip]
    scored = [x for x in scored if sign * x[0] >= NOTABLE_CORR]
    return max(scored, key=lambda x: abs(x[0]), default=(0.0, None))


def explain(
    chosen: LineupResult,
    by_points: LineupResult,
    opp_lineup: dict[str, list[PlayerRow]] | None,
    samples: np.ndarray,
    index: dict[str, int],
    opp_total: np.ndarray,
    ranges: dict[str, tuple[float, float]],
) -> list[str]:
    """One sentence per player started below a benched one's projection."""
    if chosen.player_ids == by_points.player_ids:
        return []
    started = [p for _s, p in chosen.flat() if p.player_id not in by_points.player_ids]
    benched = [p for _s, p in by_points.flat() if p.player_id not in chosen.player_ids]
    mates = [p for _s, p in chosen.flat()]
    theirs = [p for ps in (opp_lineup or {}).values() for p in ps]
    underdog = by_points.win_probability < 0.5
    base = _totals([by_points.starters], samples, index)[0]
    out = []
    for s in started:
        if not benched:
            break
        b = min(benched, key=lambda x: (x.position != s.position, abs((x.projection or 0) - (s.projection or 0))))
        benched.remove(b)
        swapped = base - samples[:, index[b.player_id]] + samples[:, index[s.player_id]]
        p_swap = float((swapped > opp_total).mean() + 0.5 * (swapped == opp_total).mean())

        skip = (s.player_id, b.player_id)
        why = ""
        if underdog:
            r, m = _strongest(mates, s, +1, skip)
            if m is not None:
                why = f"he stacks with your {m.name} (they rise and fall together), which widens your range"
            else:
                r, m = _strongest(theirs, s, -1, skip)
                if m is not None:
                    why = f"he tends to do well when your opponent's {m.name} does badly"
        else:
            r, m = _strongest(theirs, s, +1, skip)
            if m is not None:
                why = f"he rises and falls with your opponent's {m.name}, which protects your lead"
            else:
                r, m = _strongest(mates, b, +1, skip)
                if m is not None:
                    why = f"benching {b.name} breaks up his stack with your {m.name}, which steadies your score"
        if not why:
            lo_s, hi_s = ranges.get(s.player_id, (0, 0))
            lo_b, hi_b = ranges.get(b.player_id, (0, 0))
            if s.play_probability < 1.0 and underdog:
                why = f"his {s.status.lower() or 'injury'} tag makes him boom-or-bust, and you need a boom"
            elif underdog:
                why = f"more upside: {lo_s:.0f}-{hi_s:.0f} against {lo_b:.0f}-{hi_b:.0f}"
            else:
                why = f"a steadier floor: {lo_s:.0f}-{hi_s:.0f} against {lo_b:.0f}-{hi_b:.0f}"
        lead = "you are the underdog" if underdog else "you are favoured"
        out.append(
            f"Start {s.name} over {b.name} ({b.name} projects "
            f"{b.week_value - s.week_value:.1f} more): {lead}, and {why}. "
            f"P(win) {by_points.win_probability:.1%} -> {p_swap:.1%}."
        )
    return out


def diff_lineups(
    current: dict[str, list[PlayerRow]] | None, target: dict[str, list[PlayerRow]]
) -> list[tuple[str, PlayerRow | None, PlayerRow]]:
    """``(slot, benched_or_None, started)`` for every change from current to target."""
    if current is None:
        return [(slot, None, p) for slot, ps in target.items() for p in ps]
    cur_ids = {p.player_id for ps in current.values() for p in ps}
    tgt_ids = {p.player_id for ps in target.values() for p in ps}
    benched = [p for ps in current.values() for p in ps if p.player_id not in tgt_ids]
    changes: list[tuple[str, PlayerRow | None, PlayerRow]] = []
    for slot, ps in target.items():
        for p in ps:
            if p.player_id in cur_ids:
                continue
            # Pair with a benched player of a compatible position where possible.
            partner = next((b for b in benched if b.position == p.position), None)
            if partner is None and benched:
                partner = benched[0]
            if partner is not None:
                benched.remove(partner)
            changes.append((slot, partner, p))
    return changes
