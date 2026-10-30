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
from datetime import datetime, timedelta

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

#: P(win) as drawn is a little too sure of the favourite: on 100 replayed
#: leagues of 2022-2025 with defences and kickers (scripts/grade_league_odds.py),
#: favourites won 59.1% against 61.4% predicted, because the projected gap
#: between two lineups holds up only about 0.83 of the way (who starts is
#: picked on the projections). ``calibrated`` maps p to Phi(k Phi^-1(p)):
#: k picked on three seasons and scored on the fourth was 0.76-0.82, better
#: in all four (Brier 0.2373 -> 0.2366). Monotone, so it never changes which
#: lineup is best. DECISIONS.md, "Defences and kickers in the replay".
WIN_SHRINK = 0.78


def calibrated(p):
    """P(win) as it happens, from P(win) as drawn (:data:`WIN_SHRINK`)."""
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out = norm.cdf(WIN_SHRINK * norm.ppf(q))
    return float(out) if np.ndim(out) == 0 else out


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
class Fallback:
    """If a questionable starter is ruled out: when you will know (his
    inactives, about 90 minutes before his game) and who could take his slot
    then -- ``options`` as (player, how), best first, the first being the one
    the win odds count on when there is room for him."""

    starter: PlayerRow
    slot: str
    deadline: datetime
    options: list[tuple[PlayerRow, str]] = field(default_factory=list)
    #: The replacement the win odds count on (None: none in time).
    priced: PlayerRow | None = None
    #: For an option who cannot play the slot himself: the flex starter who
    #: moves into it and the flex slot the option takes (player id -> (y, slot)).
    moves: dict[str, tuple[PlayerRow, str]] = field(default_factory=dict)


#: Replacements listed per questionable starter.
FALLBACKS_SHOWN = 4


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
    #: The opponent's starters (ids) and, for the same draws, each one's score:
    #: what the page's live scores condition on, player by player.
    opp_ids: list[str] = field(default_factory=list)
    opp_player_samples: np.ndarray | None = None
    #: Benched player id -> why the P(win) lineup sits him despite projecting
    #: more than a starter he could replace.
    bench_reasons: dict[str, str] = field(default_factory=dict)
    #: For each questionable starter in the recommended lineup: who could
    #: replace him if he is ruled out, best first (:class:`Fallback`).
    fallbacks: list[Fallback] = field(default_factory=list)
    #: The opponent's bench (not in an IR slot), and what their lineup is
    #: assumed to change: a starter already out swapped for their best
    #: healthy bench player, a questionable one replaced if he sits.
    opp_bench: list[PlayerRow] = field(default_factory=list)
    opp_notes: list[str] = field(default_factory=list)
    #: Moves within the recommended lineup that put the latest kickoff in a
    #: flex slot: (flex slot, player moved in, the slot the other one takes,
    #: player moved out of the flex).
    flex_moves: list[tuple[str, PlayerRow, str, PlayerRow]] = field(default_factory=list)

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
    if (a.nfl_opponent and a.nfl_opponent == b.team) or (b.nfl_opponent and b.nfl_opponent == a.team):
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
                  correlated: bool = True, sits: np.ndarray | None = None) -> np.ndarray:
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
            plays = rng.random(n_sims) < q
            out[:, j] *= plays
            if sits is not None:
                sits[:, j] = ~plays
    return out


# ---------------------------------------------------------------------------
# Late swaps
# ---------------------------------------------------------------------------
#: Inactives are announced this long before a game, so a bench player whose
#: game starts no earlier than that can still replace a starter ruled out.
INACTIVES_LEAD = timedelta(minutes=90)


def _kick(p: PlayerRow) -> datetime | None:
    try:
        return datetime.fromisoformat(p.kickoff) if p.kickoff else None
    except ValueError:
        return None


@dataclass
class Wire:
    """Free agents a starter ruled out at the last minute could be replaced
    with, and the room on your roster to add one then: an empty bench spot,
    or an empty IR slot the ruled-out starter moves into (a player tagged out
    can go there). Once his inactives are out, your bench players whose games
    have begun can no longer be dropped, so a drop is never counted on."""

    players: list[PlayerRow] = field(default_factory=list)
    open_spots: int = 0
    ir_open: int = 0

    def without(self, p: PlayerRow) -> Wire:
        """The wire after ``p`` is added to your roster now (he takes an open
        spot if there is one)."""
        return Wire([w for w in self.players if w.player_id != p.player_id],
                    max(self.open_spots - 1, 0), self.ir_open)


#: How a ruled-out starter is replaced: from the bench, or by adding a free
#: agent to an open spot or after moving the starter to IR.
HOW_BENCH, HOW_OPEN, HOW_IR = "bench", "open spot", "IR slot"


def _in_time(b: PlayerRow, kx: datetime) -> bool:
    kb = _kick(b)
    return kb is not None and kb >= kx - INACTIVES_LEAD


def _reach(c: PlayerRow, slot: str, x: PlayerRow, lineup: dict[str, list[PlayerRow]],
           kx: datetime) -> tuple[PlayerRow, str] | None | bool:
    """How ``c`` can take ``x``'s place in ``slot`` when ``x`` is ruled out:
    True straight in (he is eligible for the slot); or (y, flex slot) -- a
    flex starter ``y`` who can play ``slot`` moves into it and ``c`` takes
    his flex spot, which needs ``y``'s own game not begun by then (a player
    whose game has kicked off is locked in place). None when neither."""
    if _eligible(c, slot):
        return True
    for fs, ps in lineup.items():
        if fs == slot or len(SLOT_ELIGIBILITY.get(fs, (fs,))) < 2 or not _eligible(c, fs):
            continue
        for y in ps:
            if y is not x and not y.locked and _eligible(y, slot) and _in_time(y, kx):
                return (y, fs)
    return None


def swap_plan(lineup: dict[str, list[PlayerRow]], roster: list[PlayerRow],
              wire: Wire | None = None) -> list[tuple[PlayerRow, PlayerRow, str]]:
    """(questionable starter, who replaces him, how) for each starter who may
    sit and has a replacement you could still put in when his inactives come
    out: a bench player eligible for his slot, not out, whose own game has not
    kicked off by then -- or, where the roster has room then, a free agent
    like that. The better of the two by this week's projection; one
    replacement per starter, each used once."""
    started = {p.player_id for ps in lineup.values() for p in ps}
    bench = sorted((p for p in roster if p.player_id not in started and not p.locked and not p.in_ir_slot),
                   key=lambda p: -p.week_value)
    have = {p.player_id for p in roster}
    free = sorted((p for p in (wire.players if wire else [])
                   if p.player_id not in have and not p.locked and not p.is_out and not p.in_ir_slot),
                  key=lambda p: -p.week_value)
    open_spots, ir_open = (wire.open_spots, wire.ir_open) if wire else (0, 0)
    used: set[str] = set()
    out = []
    for slot, ps in lineup.items():
        for x in ps:
            q = float(x.play_probability if x.play_probability is not None else 1.0)
            kx = _kick(x)
            if x.locked or q >= 1.0 or q <= 0.0 or kx is None:
                continue
            ok = lambda c, kx=kx, slot=slot, x=x: (c.player_id not in used and _in_time(c, kx)  # noqa: E731
                                                   and _reach(c, slot, x, lineup, kx) is not None)
            b = next((c for c in bench if ok(c)), None)
            w = next((c for c in free if ok(c)), None) if (open_spots or ir_open) else None
            if w is not None and (b is None or w.week_value > b.week_value):
                how = HOW_OPEN if open_spots else HOW_IR
                if open_spots:
                    open_spots -= 1
                else:
                    ir_open -= 1
                out.append((x, w, how))
                used.add(w.player_id)
            elif b is not None:
                out.append((x, b, HOW_BENCH))
                used.add(b.player_id)
    return out


def swap_pairs(lineup: dict[str, list[PlayerRow]], roster: list[PlayerRow],
               wire: Wire | None = None) -> list[tuple[PlayerRow, PlayerRow]]:
    """(questionable starter, the player who would replace him): :func:`swap_plan`
    without the how."""
    return [(x, b) for x, b, _how in swap_plan(lineup, roster, wire)]


def _swap_totals(lineups, samples, sits, index, roster, wire: Wire | None = None) -> np.ndarray:
    """``(n_lineups, n_sims)`` points each lineup gains from swapping in a
    bench player (or a free agent, where the roster has room) wherever a
    questionable starter sat."""
    out = np.zeros((len(lineups), samples.shape[0]))
    if sits is None or not sits.any():
        return out
    cache: dict[tuple[str, str], np.ndarray] = {}
    for i, lu in enumerate(lineups):
        for x, b in swap_pairs(lu, roster, wire):
            key = (x.player_id, b.player_id)
            if key not in cache:
                cache[key] = sits[:, index[x.player_id]] * samples[:, index[b.player_id]]
            out[i] += cache[key]
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


def _win_rates(lineups, samples, index, opp_total, chunk: int = 256, sits=None, roster=None,
               wire: Wire | None = None) -> np.ndarray:
    """P(win) per lineup, in chunks so thousands of lineups stay in memory."""
    out = np.empty(len(lineups))
    for lo in range(0, len(lineups), chunk):
        t = _totals(lineups[lo:lo + chunk], samples, index)
        if sits is not None and roster is not None:
            t = t + _swap_totals(lineups[lo:lo + chunk], samples, sits, index, roster, wire)
        out[lo:lo + chunk] = (t > opp_total).mean(axis=1) + 0.5 * (t == opp_total).mean(axis=1)
    return calibrated(out)


def _projected(lineup: dict[str, list[PlayerRow]]) -> float:
    """Projected total, with final scores in place of projections."""
    return float(sum(p.week_value for ps in lineup.values() for p in ps))


def expected_with_swaps(lineup: dict[str, list[PlayerRow]], roster: list[PlayerRow]) -> float:
    """Projected points counting the late swap: a questionable starter's
    projection already weighs in that he may sit, and when he does his
    bench replacement (:func:`swap_pairs`) scores instead."""
    gain = sum((1.0 - float(x.play_probability)) * b.week_value for x, b in swap_pairs(lineup, roster)
               if x.play_probability is not None)
    return _projected(lineup) + gain


#: Lineups re-ranked with their late swaps counted (the rest cannot catch up:
#: a swap adds at most a bench player's points times the chance of a sit).
SWAP_RERANK = 200


def best_expected(lineups: list[dict[str, list[PlayerRow]]], roster: list[PlayerRow]
                  ) -> dict[str, list[PlayerRow]]:
    """The lineup with the most expected points, late swaps counted: starting
    a questionable player with a good backup behind him can beat benching
    him for a slightly higher sure projection."""
    top = sorted(lineups, key=_projected, reverse=True)[:SWAP_RERANK]
    return max(top, key=lambda lu: expected_with_swaps(lu, roster))


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


def opponent_fixes(lineup: dict[str, list[PlayerRow]], roster: list[PlayerRow]
                   ) -> tuple[dict[str, list[PlayerRow]], list[str]]:
    """The opponent's set lineup with every starter already out (ruled out,
    on bye) swapped for his best healthy bench player whose game has not
    started -- the fix nearly every manager makes. Questionable starters are
    left in; :func:`_swap_totals` replaces them in the draws they sit."""
    started = {q.player_id for ps in lineup.values() for q in ps}
    bench = sorted((q for q in roster if q.player_id not in started and not q.locked and not q.in_ir_slot
                    and not q.is_out), key=lambda q: -q.week_value)
    out = {slot: list(ps) for slot, ps in lineup.items()}
    notes = []
    for slot, ps in out.items():
        for i, x in enumerate(ps):
            if not x.is_out or x.locked or x.actual_points is not None:
                continue
            b = next((c for c in bench if _eligible(c, slot)), None)
            if b is None:
                continue
            bench.remove(b)
            ps[i] = b
            why = "on bye" if x.on_bye else "out"
            notes.append(f"{x.name} is {why}: assumed replaced by {b.name} ({b.week_value:.1f})")
    return out, notes


def _flex_slot(slot: str) -> bool:
    return len(SLOT_ELIGIBILITY.get(slot, (slot,))) > 1


def flex_last(lineup: dict[str, list[PlayerRow]]
              ) -> tuple[dict[str, list[PlayerRow]], list[tuple[str, PlayerRow, str, PlayerRow]]]:
    """The same starters, arranged so each flex slot holds the latest kickoff
    among the starters at its occupant's position: if someone is ruled out
    late, the flex can still take a player from any later game (a dedicated
    slot only its own position). Only players whose games have not started
    move."""
    out = {slot: list(ps) for slot, ps in lineup.items()}
    moves = []
    never = datetime.max.replace(tzinfo=None)

    def when(q: PlayerRow):
        k = _kick(q)
        return k.replace(tzinfo=None) if k is not None else None

    for fslot in [sl for sl in out if _flex_slot(sl)]:
        for i, x in enumerate(out[fslot]):
            kx = when(x)
            if x.locked or kx is None:
                continue
            best = None
            for slot, ps in out.items():
                if _flex_slot(slot) or x.position not in SLOT_ELIGIBILITY.get(slot, (slot,)):
                    continue
                for j, y in enumerate(ps):
                    ky = when(y)
                    if y.locked or ky is None or y.position != x.position or ky <= kx:
                        continue
                    if best is None or ky > (when(best[2]) or never):
                        best = (slot, j, y)
            if best is not None:
                slot, j, y = best
                out[fslot][i], out[slot][j] = y, x
                moves.append((fslot, y, slot, x))
    return out, moves


def optimise(
    roster: list[PlayerRow],
    slots: dict[str, int],
    opponent_roster: list[PlayerRow] | None,
    n_sims: int = 20000,
    seed: int = 7,
    min_edge: float = MIN_WIN_EDGE,
    opponent_mode: str = "current",
    wire: Wire | None = None,
) -> Optimisation:
    """Find the lineup that maximises P(win) against the opponent.

    ``opponent_mode``: "current" faces the lineup the opponent has set (their
    best expected one when none is set); "ideal" always faces their best
    expected lineup, late swaps counted (:func:`best_expected`). ``wire``:
    free agents who could replace a starter ruled out at the last minute,
    and the room to add one (:class:`Wire`)."""
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
        opp_lineup = current_lineup(opponent_roster, slots) if opponent_mode == "current" else None
        if opp_lineup is None or sum(len(v) for v in opp_lineup.values()) < sum(slots.values()):
            # Their best: respecting kickoffs when it is the ideal asked for.
            opp_lineups = (_with_locks(opponent_roster, slots) if opponent_mode == "ideal"
                           else enumerate_lineups(opponent_roster, slots))
            if opp_lineups:
                opp_lineup = best_expected(opp_lineups, opponent_roster)
    opp_notes: list[str] = []
    opp_bench: list[PlayerRow] = []
    if opp_lineup is not None and opponent_roster:
        opp_lineup, opp_notes = opponent_fixes(opp_lineup, opponent_roster)
        started = {p.player_id for ps in opp_lineup.values() for p in ps}
        opp_bench = sorted((p for p in opponent_roster if p.player_id not in started),
                           key=lambda p: -p.week_value)

    everyone = list(roster) + (list(opponent_roster) if opponent_roster else [])
    if wire is not None:
        seen = {p.player_id for p in everyone}
        wire = Wire([w for w in wire.players if w.player_id not in seen], wire.open_spots, wire.ir_open)
        everyone += wire.players
    index = {p.player_id: i for i, p in enumerate(everyone)}

    # Select on one sample, confirm on a fresh one: picking the best of
    # thousands of lineups on the same draws that score them flatters
    # whichever got lucky.
    # Which questionable starters sat in each draw: a starter ruled out in
    # time is replaced from the bench (swap_pairs) -- yours, and the
    # opponent's from his own bench.
    sits_sel = np.zeros((n_sims, len(everyone)), dtype=bool)
    select = sample_points(everyone, n_sims, rng, sits=sits_sel)
    opp_sel = (_totals([opp_lineup], select, index)[0]
               + _swap_totals([opp_lineup], select, sits_sel, index, opponent_roster)[0]
               if opp_lineup is not None else np.zeros(n_sims))
    wins_sel = _win_rates(mine, select, index, opp_sel, sits=sits_sel, roster=roster, wire=wire)
    projected = np.array([_projected(lu) for lu in mine])
    ev_best = int(np.argmax(projected))
    top = list(np.argsort(-wins_sel)[:CONFIRM_TOP])
    if ev_best not in top:
        top.append(ev_best)

    sits = np.zeros((n_sims, len(everyone)), dtype=bool)
    confirm = sample_points(everyone, n_sims, rng, sits=sits)
    opp_total = (_totals([opp_lineup], confirm, index)[0]
                 + _swap_totals([opp_lineup], confirm, sits, index, opponent_roster)[0]
                 if opp_lineup is not None else np.zeros(n_sims))
    cand = [mine[i] for i in top]
    totals = _totals(cand, confirm, index) + _swap_totals(cand, confirm, sits, index, roster, wire)
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
    # Same starters, same totals: only which slot each one sits in.
    arranged, flex_moves = flex_last(best_win.starters)
    best_win = LineupResult(starters=arranged, expected=best_win.expected, sd=best_win.sd,
                            win_probability=best_win.win_probability)

    current = None
    cur = current_lineup(roster, slots)
    if cur is not None:
        t = _totals([cur], confirm, index)[0] + _swap_totals([cur], confirm, sits, index, roster, wire)[0]
        current = LineupResult(
            starters=cur, expected=_projected(cur), sd=float(t.std()),
            win_probability=calibrated((t > opp_total).mean() + 0.5 * (t == opp_total).mean()),
        )

    opponent = None
    if opp_lineup is not None:
        opponent = LineupResult(starters=opp_lineup, expected=_projected(opp_lineup),
                                sd=float(opp_total.std()), win_probability=1.0 - best_win.win_probability)

    ranges = {p.player_id: (float(np.percentile(confirm[:, index[p.player_id]], 15)),
                            float(np.percentile(confirm[:, index[p.player_id]], 85)))
              for p in everyone}
    bench_reasons: dict[str, str] = {}
    reasons = explain(best_win, best_ev, opp_lineup, confirm, index, opp_total, ranges, bench_reasons,
                      sits=sits, roster=roster, wire=wire)
    for x, b, how in swap_plan(best_win.starters, roster, wire):
        add = ("" if how == HOW_BENCH else
               " (a free agent: add him to your open roster spot)" if how == HOW_OPEN else
               f" (a free agent: move {x.name} to your empty IR slot once he is marked out, then add him)")
        reasons.append(f"if {x.name} is ruled out ({x.play_probability:.0%} to play; inactives come out about "
                       f"90 minutes before his game), start {b.name} in his place{add} -- the win odds count "
                       "on that swap")
    fallbacks = fallback_plans(best_win.starters, roster, wire)

    keep = min(EDITOR_SIMS, n_sims)
    ids = [p.player_id for p in roster]
    opp_ids = [p.player_id for ps in (opp_lineup or {}).values() for p in ps if p.player_id in index]
    return Optimisation(
        best_win=best_win, best_ev=best_ev, current=current, opponent=opponent,
        changes=diff_lineups(current.starters if current else None, best_win.starters),
        n_lineups=len(mine), n_sims=n_sims, reasons=reasons, ranges=ranges, bench_reasons=bench_reasons,
        sample_ids=ids, samples=confirm[:keep, [index[i] for i in ids]],
        opp_samples=opp_total[:keep], fallbacks=fallbacks,
        opp_ids=opp_ids, opp_player_samples=confirm[:keep, [index[i] for i in opp_ids]] if opp_ids else None,
        opp_bench=opp_bench, opp_notes=opp_notes, flex_moves=flex_moves,
    )


def fallback_plans(lineup: dict[str, list[PlayerRow]], roster: list[PlayerRow],
                   wire: Wire | None = None) -> list[Fallback]:
    """Each questionable starter's replacements if he is ruled out, best
    first: bench players and free agents whose games have not begun by his
    inactives (a free agent only where the roster will have room), and which
    of them the win odds count on (:func:`swap_plan`)."""
    priced = {x.player_id: b for x, b, _how in swap_plan(lineup, roster, wire)}
    started = {p.player_id for ps in lineup.values() for p in ps}
    have = {p.player_id for p in roster}
    room = HOW_OPEN if wire and wire.open_spots else HOW_IR if wire and wire.ir_open else None
    out = []
    for slot, ps in lineup.items():
        for x in ps:
            q = float(x.play_probability if x.play_probability is not None else 1.0)
            kx = _kick(x)
            if x.locked or q >= 1.0 or q <= 0.0 or kx is None:
                continue
            moves: dict[str, tuple[PlayerRow, str]] = {}

            def fits(c, kx=kx, slot=slot, x=x, moves=moves):
                if not _in_time(c, kx) or c.locked:
                    return False
                how = _reach(c, slot, x, lineup, kx)
                if isinstance(how, tuple):
                    moves[c.player_id] = how
                return how is not None

            opts = [(c, HOW_BENCH) for c in roster
                    if c.player_id not in started and not c.in_ir_slot and fits(c)]
            if room is not None:
                opts += [(c, room) for c in wire.players
                         if c.player_id not in have and not c.is_out and not c.in_ir_slot and fits(c)]
            opts.sort(key=lambda o: -o[0].week_value)
            shown = opts[:FALLBACKS_SHOWN]
            out.append(Fallback(starter=x, slot=slot, deadline=kx - INACTIVES_LEAD,
                                options=shown, priced=priced.get(x.player_id),
                                moves={c.player_id: moves[c.player_id] for c, _h in shown if c.player_id in moves}))
    return out


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
    bench_reasons: dict[str, str] | None = None,
    sits: np.ndarray | None = None,
    roster: list[PlayerRow] | None = None,
    wire: Wire | None = None,
) -> list[str]:
    """One sentence per player started below a benched one's projection.
    ``bench_reasons`` is filled with the same sentence keyed by the benched
    player. Each swap is priced as the lineups were chosen: with the late
    swap a questionable starter allows (``sits``, ``roster``, ``wire``) --
    without it, a starter ruled out read as a zero and the sentence said the
    chosen lineup was the worse one."""
    if chosen.player_ids == by_points.player_ids:
        return []
    started = [p for _s, p in chosen.flat() if p.player_id not in by_points.player_ids]
    benched = [p for _s, p in by_points.flat() if p.player_id not in chosen.player_ids]
    mates = [p for _s, p in chosen.flat()]
    theirs = [p for ps in (opp_lineup or {}).values() for p in ps]
    underdog = by_points.win_probability < 0.5
    out = []
    for s in started:
        if not benched:
            break
        b = min(benched, key=lambda x: (x.position != s.position, abs((x.projection or 0) - (s.projection or 0))))
        benched.remove(b)
        lu = {slot: [s if q.player_id == b.player_id else q for q in ps] for slot, ps in by_points.starters.items()}
        swapped = _totals([lu], samples, index)[0]
        cover = None
        if roster is not None:
            swapped = swapped + _swap_totals([lu], samples, sits, index, roster, wire)[0]
            cover = next((c for x, c in swap_pairs(lu, roster, wire) if x.player_id == s.player_id), None)
        p_swap = calibrated((swapped > opp_total).mean() + 0.5 * (swapped == opp_total).mean())

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
            if s.play_probability < 1.0 and cover is not None:
                why = (f"he is {s.play_probability:.0%} to play, but his game is late enough that if he is ruled "
                       f"out you can still start {cover.name} in his place, so the downside is covered")
            elif s.play_probability < 1.0 and underdog:
                why = f"his {s.status.lower() or 'injury'} tag makes him boom-or-bust, and you need a boom"
            elif underdog:
                why = f"more upside: {lo_s:.0f}-{hi_s:.0f} against {lo_b:.0f}-{hi_b:.0f}"
            else:
                why = f"a steadier floor: {lo_s:.0f}-{hi_s:.0f} against {lo_b:.0f}-{hi_b:.0f}"
        lead = "you are the underdog" if underdog else "you are favoured"
        if bench_reasons is not None:
            bench_reasons[b.player_id] = (
                f"{s.name} starts instead: {lead}, and {why} "
                f"(P(win) {by_points.win_probability:.1%} with {b.name}, {p_swap:.1%} with {s.name})")
        out.append(
            f"Start {s.name} over {b.name} ({b.name} projects "
            f"{b.week_value - s.week_value:.1f} more): {lead}, and {why}. "
            f"P(win) {by_points.win_probability:.1%} -> {p_swap:.1%}."
        )
    return out


@dataclass
class StreamOption:
    """One D/ST or kicker scored in your lineup against this week's opponent."""

    player: PlayerRow
    win_probability: float
    current: bool
    links: list[str] = field(default_factory=list)
    #: 15th and 85th percentile of his simulated score (the same range the
    #: lineup tables show), None when not drawn.
    low: float | None = None
    high: float | None = None


def _links(p: PlayerRow, mine: list[PlayerRow], theirs: list[PlayerRow]) -> list[str]:
    """How a unit's score ties to players in this matchup, strongest first."""
    out = []
    for pool, whose in ((mine, "your"), (theirs, "their")):
        for q in pool:
            r = pair_correlation(p, q)
            if abs(r) >= NOTABLE_CORR:
                verb = "moves with" if r > 0 else "works against"
                out.append((abs(r), f"{verb} {whose} {q.name} ({r:+.2f})"))
    return [t for _r, t in sorted(out, reverse=True)]


def stream_options(
    opt: Optimisation,
    position: str,
    candidates: list[PlayerRow],
    n_sims: int = 20000,
    seed: int = 11,
) -> list[StreamOption]:
    """P(win) with each candidate D/ST or kicker in place of the current one.

    The rest of the recommended lineup and the opponent's stay fixed, and all
    candidates are scored on the same correlated draws, so the ranking
    carries the matchup. A defence facing your own quarterback (-0.44) is a
    hedge -- it damps your total's swings, which an underdog does not want
    and a favourite does; one facing theirs works the other way round. A
    kicker tied to your stack adds variance.
    """
    if opt.opponent is None:
        return []
    slot_players = [(slot, p) for slot, p in opt.best_win.flat() if p.position == position]
    if slot_players:
        _slot, current = slot_players[0]
        if current.locked:
            return []
    elif position in opt.best_win.starters:
        # The slot is empty (no unit rostered, or only one on bye): it scores
        # zero, so it is shown as "yours" and every unit is priced against it.
        current = PlayerRow(player_id=f"__empty_{position}", name="(empty slot)", position=position,
                            projection=0.0, slot=position, play_probability=0.0)
    else:
        return []
    mine = [p for _s, p in opt.best_win.flat()]
    theirs = [p for _s, p in opt.opponent.flat()]
    if current not in mine:
        mine = mine + [current]
    pool = [current] + [c for c in candidates
                        if c.player_id != current.player_id and not c.locked and not c.is_out
                        and c.position == position and c.projection is not None]
    everyone = mine + theirs + [c for c in pool if c.player_id != current.player_id]
    index = {p.player_id: i for i, p in enumerate(everyone)}
    rng = np.random.default_rng(seed)
    samples = sample_points(everyone, n_sims, rng)
    base = sum(samples[:, index[p.player_id]] for p in mine) - samples[:, index[current.player_id]]
    opp = sum(samples[:, index[p.player_id]] for p in theirs)
    out = []
    for c in pool:
        t = base + samples[:, index[c.player_id]]
        pw = calibrated((t > opp).mean() + 0.5 * (t == opp).mean())
        others = [m for m in mine if m.player_id != current.player_id]
        col = samples[:, index[c.player_id]]
        out.append(StreamOption(player=c, win_probability=pw, current=c.player_id == current.player_id,
                                links=_links(c, others, theirs),
                                low=float(np.percentile(col, 15)), high=float(np.percentile(col, 85))))
    out.sort(key=lambda o: -o.win_probability)
    return out


@dataclass
class PickupOption:
    """A free agent added for this week, and what he does to P(win)."""

    player: PlayerRow
    win_probability: float          # your best lineup with him in it
    base: float                     # your best lineup as it stands, on the same draws
    noise: float                    # standard error of the difference
    lineup: LineupResult
    #: Changes from your best lineup as it stands: (slot, benched, started).
    changes: list[tuple[str, PlayerRow | None, PlayerRow]] = field(default_factory=list)
    drop: PlayerRow | None = None
    #: A pickup who helps from the bench: the questionable starter he would
    #: replace if that starter is ruled out (None: he starts, or does not).
    cover: PlayerRow | None = None
    #: Title odds of the whole move (this week and the rest of the season,
    #: the drop included), when the season engine priced it.
    title_gain: float | None = None
    title_noise: float = 0.0
    #: The part of ``title_gain`` that is this week: the P(win) gain times
    #: what this game swings your title odds (title odds if you win it,
    #: less if you lose it). The rest is the season after, the drop's lost
    #: weeks included.
    week_title: float | None = None
    #: Title odds lost on later claims by spending waiver priority on him
    #: (0 for a free agent, who needs no claim).
    claim_cost: float = 0.0

    @property
    def rest_title(self) -> float | None:
        if self.title_gain is None or self.week_title is None:
            return None
        return self.title_gain - self.week_title

    @property
    def title_net(self) -> float | None:
        """The whole move in title odds, after the waiver priority it spends."""
        return None if self.title_gain is None else self.title_gain - self.claim_cost

    @property
    def gain(self) -> float:
        return self.win_probability - self.base

    @property
    def moves_text(self) -> str:
        """"start A over B; start C over D" -- every lineup change he brings,
        or, for cover, what he is held for."""
        if not self.changes and self.cover is not None:
            return f"hold him on the bench to replace {self.cover.name} if he is ruled out"
        return "; ".join(f"start {s.name} at {slot}" + (f" over {b.name}" if b is not None else " (an open slot)")
                         for slot, b, s in self.changes)

    @property
    def clear(self) -> bool:
        """Better by more than the simulation can blur."""
        return self.gain >= max(MIN_WIN_EDGE, 2.0 * self.noise)


def pickup_options(
    opt: Optimisation,
    roster: list[PlayerRow],
    slots: dict[str, int],
    candidates: list[PlayerRow],
    n_sims: int = 20000,
    seed: int = 13,
    wire: Wire | None = None,
) -> list[PickupOption]:
    """P(win) this week with each free agent on the roster.

    Your best lineup is rebuilt around each candidate, or kept as it is with
    him on the bench as cover for a questionable starter (a late game, so he
    can come in if that starter is ruled out), and scored against the
    opponent's lineup on draws shared by every candidate and by your lineup
    as it stands -- late swaps counted on both sides, from the bench and the
    wire (``wire``) -- so the differences are the players, not the luck of
    the draw. The lineup is chosen on one sample and scored on a fresh one,
    as :func:`optimise` does. Who to drop is left to the caller: this week it
    is anyone who would not start.
    """
    if opt.opponent is None:
        return []
    roster = [p for p in roster if not p.in_ir_slot]
    have = {p.player_id for p in roster}
    pool = [c for c in candidates if c.player_id not in have and not c.locked and not c.is_out
            and not c.in_ir_slot and c.projection is not None]
    if not pool:
        return []
    theirs = [p for _s, p in opt.opponent.flat()]
    everyone, seen = [], set()
    for p in roster + theirs + pool + (wire.players if wire else []):
        if p.player_id not in seen:
            seen.add(p.player_id)
            everyone.append(p)
    index = {p.player_id: i for i, p in enumerate(everyone)}
    rng = np.random.default_rng(seed)
    sits_sel = np.zeros((n_sims, len(everyone)), dtype=bool)
    sits = np.zeros((n_sims, len(everyone)), dtype=bool)
    select = sample_points(everyone, n_sims, rng, sits=sits_sel)
    confirm = sample_points(everyone, n_sims, rng, sits=sits)
    opp_sel = _totals([opt.opponent.starters], select, index)[0]
    opp_conf = _totals([opt.opponent.starters], confirm, index)[0]
    base = [opt.best_win.starters]
    base_tot = _totals(base, confirm, index)[0] + _swap_totals(base, confirm, sits, index, roster, wire)[0]
    base_ind = (base_tot > opp_conf) + 0.5 * (base_tot == opp_conf)
    out = []
    for c in pool:
        mine, rest = roster + [c], (wire.without(c) if wire else None)
        lineups = [lu for lu in _with_locks(mine, slots)
                   if any(p.player_id == c.player_id for ps in lu.values() for p in ps)]
        # Or keep the lineup and hold him as cover (only worth it if a swap uses him).
        cover = next((x for x, b in swap_pairs(opt.best_win.starters, mine, rest)
                      if b.player_id == c.player_id), None)
        if cover is not None:
            lineups.append(opt.best_win.starters)
        if not lineups:
            continue
        best = lineups[int(np.argmax(_win_rates(lineups, select, index, opp_sel, sits=sits_sel,
                                                roster=mine, wire=rest)))]
        tot = _totals([best], confirm, index)[0] + _swap_totals([best], confirm, sits, index, mine, rest)[0]
        ind = (tot > opp_conf) + 0.5 * (tot == opp_conf)
        diff = ind - base_ind
        held = best is opt.best_win.starters
        out.append(PickupOption(
            player=c, win_probability=calibrated(ind.mean()), base=calibrated(base_ind.mean()),
            noise=float(diff.std() / np.sqrt(n_sims)),
            lineup=LineupResult(starters=best, expected=_projected(best), sd=float(tot.std()),
                                win_probability=calibrated(ind.mean())),
            changes=[] if held else diff_lineups(opt.best_win.starters, best),
            cover=cover if held else None))
    out.sort(key=lambda o: -o.gain)
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
