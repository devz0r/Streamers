"""Lineup optimisation that maximises the probability of winning the matchup.

Maximising expected points is the wrong objective in a head-to-head week. If
you are the underdog you want variance -- a boom-or-bust receiver beats a
steady one when only a boom wins -- and if you are the favourite you want to
choke it off. So every candidate lineup is scored by Monte Carlo against the
opponent's lineup, and the one with the highest **P(win)** is recommended,
with the expected-points-maximising lineup shown alongside so the trade-off is
visible.

The search is exhaustive over valid slot assignments (with a light candidate
prune per slot), and every lineup is evaluated on one shared sample matrix, so
thousands of lineups cost a single matrix multiply.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from ..league.model import SLOT_ELIGIBILITY, PlayerRow

#: Positions that can legitimately score below zero.
NEGATIVE_OK = ("DST", "K")

#: How many extra candidates beyond the slot count to consider per slot.
PRUNE_EXTRA = 3


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

    @property
    def win_gain(self) -> float:
        if self.current is None:
            return 0.0
        return self.best_win.win_probability - self.current.win_probability


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_points(players: list[PlayerRow], n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """``(n_sims, n_players)`` simulated scores.

    Normal around the projection with the calibrated sd, floored at zero for
    skill positions (a receiver cannot score -4) but not for D/ST and K.
    """
    if not players:
        return np.zeros((n_sims, 0))
    means = np.array([float(p.projection or 0.0) for p in players])
    sds = np.array([float(p.projection_sd or 0.0) for p in players])
    out = rng.normal(means, np.maximum(sds, 1e-6), size=(n_sims, len(players)))
    floor = np.array([0 if p.position in NEGATIVE_OK else 1 for p in players], dtype=bool)
    out[:, floor] = np.maximum(out[:, floor], 0.0)
    # An unavailable player scores exactly nothing.
    out[:, [p.is_out for p in players]] = 0.0
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
    ranked = sorted(roster, key=lambda p: -(p.projection or 0.0))

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


def optimise(
    roster: list[PlayerRow],
    slots: dict[str, int],
    opponent_roster: list[PlayerRow] | None,
    n_sims: int = 20000,
    seed: int = 7,
) -> Optimisation:
    """Find the lineup that maximises P(win) against the opponent."""
    rng = np.random.default_rng(seed)
    # A player parked in an IR slot cannot be started without a roster move
    # (activation needs a free spot), so the lineup is chosen from the rest.
    roster = [p for p in roster if not p.in_ir_slot]
    if opponent_roster:
        opponent_roster = [p for p in opponent_roster if not p.in_ir_slot]
    mine = enumerate_lineups(roster, slots)
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
                opp_lineup = max(
                    opp_lineups,
                    key=lambda lu: sum(p.projection or 0.0 for ps in lu.values() for p in ps),
                )

    everyone = list(roster) + (list(opponent_roster) if opponent_roster else [])
    index = {p.player_id: i for i, p in enumerate(everyone)}
    samples = sample_points(everyone, n_sims, rng)

    my_totals = _totals(mine, samples, index)
    if opp_lineup is not None:
        opp_total = _totals([opp_lineup], samples, index)[0]
    else:
        opp_total = np.zeros(n_sims)

    wins = (my_totals > opp_total[None, :]).mean(axis=1) + 0.5 * (my_totals == opp_total[None, :]).mean(axis=1)
    evs = my_totals.mean(axis=1)
    sds = my_totals.std(axis=1)

    def result(i: int) -> LineupResult:
        return LineupResult(starters=mine[i], expected=float(evs[i]), sd=float(sds[i]),
                            win_probability=float(wins[i]))

    best_win = result(int(np.argmax(wins)))
    best_ev = result(int(np.argmax(evs)))

    current = None
    cur = current_lineup(roster, slots)
    if cur is not None:
        t = _totals([cur], samples, index)[0]
        current = LineupResult(
            starters=cur, expected=float(t.mean()), sd=float(t.std()),
            win_probability=float((t > opp_total).mean() + 0.5 * (t == opp_total).mean()),
        )

    opponent = None
    if opp_lineup is not None:
        opponent = LineupResult(starters=opp_lineup, expected=float(opp_total.mean()),
                                sd=float(opp_total.std()), win_probability=1.0 - best_win.win_probability)

    return Optimisation(
        best_win=best_win, best_ev=best_ev, current=current, opponent=opponent,
        changes=diff_lineups(current.starters if current else None, best_win.starters),
        n_lineups=len(mine), n_sims=n_sims,
    )


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
