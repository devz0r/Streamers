"""Waiver-wire recommendations: who to add, who to drop, and why.

A pickup is worth exactly what it does to the roster, so every candidate move
is an (add, drop) pair scored by the change in *roster value*:

* the expected points of the best starting lineup the roster can field, plus
* a small credit for bench depth -- the first and second bench body at RB/WR
  (and the first at TE/QB) measured over **replacement level**, the best free
  agent left on the wire at that position.

That second term is what stops the engine from loving a backup quarterback: a
16-point QB on the bench of a one-QB league is worth almost nothing when a
15-point QB is always on the wire, whereas an RB who would start over your
flex is worth every point of the difference.

Two horizons are blended by how much season is left:

* **next week** -- this week's projection, zero if on bye or out;
* **rest of season** -- the per-game projection with no matchup adjustment,
  zero for anyone on injured reserve or suspended.

Early in the season the rest-of-season half dominates (a bye next week is not
a reason to skip a real asset); late in the season next week is what matters.
Moves are assigned greedily and re-scored after each one, so the list reads as
a sequence that can all be made together, each priced on top of the last.

Players sitting in an IR slot are neither drop candidates (dropping them frees
no bench spot) nor part of the lineup (activating them is a roster move).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..league.model import SLOT_ELIGIBILITY, LeagueSnapshot, PlayerRow

#: Positions a roster must keep at least this many of.
MIN_KEEP = {"QB": 1, "K": 1, "DST": 1, "TE": 1}

#: Credit for bench depth over replacement level, by bench rank at the
#: position. Roughly the chance that body is needed in the lineup in a given
#: week (injury or bye to one of the starters ahead of it).
BENCH_WEIGHTS: dict[str, tuple[float, ...]] = {
    "RB": (0.25, 0.10),
    "WR": (0.25, 0.10),
    "TE": (0.10,),
    "QB": (0.10,),
    "K": (),
    "DST": (),
}

#: Free agents considered per position, by their better horizon value.
POOL_PER_POSITION = 30
#: Pickups re-scored after each accepted move.
RECHECK = 40

Key = Callable[[PlayerRow], float]


@dataclass
class Move:
    add: PlayerRow
    drop: PlayerRow | None
    next_week_gain: float
    ros_gain: float
    score: float
    tag: str
    reason: str


def horizon_weight(week: int, max_week: int = 18) -> float:
    """Weight on next week vs rest of season, rising as the season runs out."""
    frac = min(max(week, 1), max_week) / max_week
    return 0.35 + 0.5 * frac


def _next_week(p: PlayerRow) -> float:
    if p.is_out:
        return 0.0
    return float(p.projection or 0.0)


def _ros(p: PlayerRow) -> float:
    if p.is_long_term_out:
        return 0.0
    base = p.ros_value if p.ros_value is not None else (p.projection or 0.0)
    return float(base)


def value(p: PlayerRow, w_next: float) -> float:
    """A player's own blended value, ignoring the roster around them."""
    return w_next * _next_week(p) + (1.0 - w_next) * _ros(p)


# ---------------------------------------------------------------------------
# Roster value
# ---------------------------------------------------------------------------
def _eligible(p: PlayerRow, slot: str) -> bool:
    if p.eligible_slots and slot in p.eligible_slots:
        return True
    return p.position in SLOT_ELIGIBILITY.get(slot, ())


def best_lineup(players: list[PlayerRow], slots: dict[str, int], key: Key) -> tuple[set[str], float]:
    """Greedy best lineup on ``key``: dedicated slots first, flex slots last.

    Exact for the usual one-flex structure; a close approximation otherwise.
    Returns the starters' ids and their total.
    """
    ordered = sorted(
        slots.items(),
        key=lambda kv: (len(SLOT_ELIGIBILITY.get(kv[0], (kv[0],))), kv[0]),
    )
    ranked = sorted(players, key=key, reverse=True)
    used: set[str] = set()
    total = 0.0
    for slot, count in ordered:
        taken = 0
        for p in ranked:
            if taken >= count:
                break
            if p.player_id in used or not _eligible(p, slot):
                continue
            used.add(p.player_id)
            total += key(p)
            taken += 1
    return used, total


def roster_value(
    players: list[PlayerRow], slots: dict[str, int], key: Key, replacement: dict[str, float]
) -> float:
    """Best-lineup points plus replacement-level bench depth."""
    starters, total = best_lineup(players, slots, key)
    bench: dict[str, list[float]] = {}
    for p in players:
        if p.player_id not in starters:
            bench.setdefault(p.position, []).append(key(p))
    for pos, vals in bench.items():
        vals.sort(reverse=True)
        level = replacement.get(pos, 0.0)
        for w, v in zip(BENCH_WEIGHTS.get(pos, ()), vals):
            total += w * max(v - level, 0.0)
    return total


def _replacement_levels(pool: list[PlayerRow], key: Key) -> dict[str, list[float]]:
    """Top two free-agent values per position, so a candidate can be measured
    against the best *other* body on the wire."""
    out: dict[str, list[float]] = {}
    for p in pool:
        out.setdefault(p.position, []).append(key(p))
    return {pos: sorted(v, reverse=True)[:2] for pos, v in out.items()}


def _level_excluding(levels: dict[str, list[float]], fa: PlayerRow, key: Key) -> dict[str, float]:
    out = {pos: (v[0] if v else 0.0) for pos, v in levels.items()}
    top = levels.get(fa.position, [])
    if top and abs(top[0] - key(fa)) < 1e-9:
        out[fa.position] = top[1] if len(top) > 1 else 0.0
    return out


def droppable(roster: list[PlayerRow], w_next: float, position: str) -> list[PlayerRow]:
    """Roster players that could be dropped to make room for ``position``.

    Never an IR-slot player (that frees no spot), never below the minimum at a
    core position unless the pickup plays that same position, and never the
    best player at a position for a pickup at another.
    """
    active = [p for p in roster if not p.in_ir_slot]
    counts: dict[str, int] = {}
    for p in active:
        counts[p.position] = counts.get(p.position, 0) + 1
    best_at: dict[str, str] = {}
    for p in sorted(active, key=lambda x: -value(x, w_next)):
        best_at.setdefault(p.position, p.player_id)

    out = []
    for p in active:
        if p.player_id == best_at.get(p.position) and p.position != position:
            continue
        need = MIN_KEEP.get(p.position, 0)
        if p.position != position and counts.get(p.position, 0) - 1 < need:
            continue
        out.append(p)
    return sorted(out, key=lambda x: value(x, w_next))


def _has_value(p: PlayerRow) -> bool:
    return p.projection is not None or p.ros_value is not None


def recommend(
    snapshot: LeagueSnapshot,
    min_gain: float = 0.5,
    max_moves: int = 12,
    max_week: int = 18,
) -> list[Move]:
    """Ranked add/drop moves for the user's team, each priced on top of the last."""
    me = snapshot.my_team
    slots = snapshot.starting_slots
    w_next = horizon_weight(snapshot.week, max_week)
    roster = [p for p in me.roster if _has_value(p) and not p.in_ir_slot]
    if not roster:
        return []

    pool_all = [fa for fa in snapshot.free_agents if _has_value(fa)]
    by_pos: dict[str, list[PlayerRow]] = {}
    for fa in pool_all:
        by_pos.setdefault(fa.position, []).append(fa)
    pool: list[PlayerRow] = []
    for fas in by_pos.values():
        fas.sort(key=lambda p: -max(_next_week(p), _ros(p)))
        pool.extend(fas[:POOL_PER_POSITION])
    lv_next = _replacement_levels(pool_all, _next_week)
    lv_ros = _replacement_levels(pool_all, _ros)

    def score_add(state: list[PlayerRow], fa: PlayerRow) -> Move | None:
        rep_next = _level_excluding(lv_next, fa, _next_week)
        rep_ros = _level_excluding(lv_ros, fa, _ros)
        base_next = roster_value(state, slots, _next_week, rep_next)
        base_ros = roster_value(state, slots, _ros, rep_ros)
        best: Move | None = None
        for drop in droppable(state, w_next, fa.position):
            trial = [p for p in state if p.player_id != drop.player_id] + [fa]
            next_gain = roster_value(trial, slots, _next_week, rep_next) - base_next
            ros_gain = roster_value(trial, slots, _ros, rep_ros) - base_ros
            score = w_next * next_gain + (1.0 - w_next) * ros_gain
            # On a tie, prefer dropping at the pickup's own position (the old
            # defence for the new one) over some other zero-value body.
            better = best is None or score > best.score + 1e-9
            tie_same_pos = (best is not None and abs(score - best.score) <= 1e-9
                            and drop.position == fa.position and best.drop.position != fa.position)
            if better or tie_same_pos:
                tag, reason = _explain(fa, drop, next_gain, ros_gain)
                best = Move(add=fa, drop=drop, next_week_gain=next_gain,
                            ros_gain=ros_gain, score=score, tag=tag, reason=reason)
        return best

    state = list(roster)
    candidates = pool
    out: list[Move] = []
    while len(out) < max_moves and candidates:
        scored = [m for m in (score_add(state, fa) for fa in candidates) if m is not None]
        scored.sort(key=lambda m: -m.score)
        scored = [m for m in scored if m.score >= min_gain]
        if not scored:
            break
        pick = scored[0]
        out.append(pick)
        state = [p for p in state if p.player_id != pick.drop.player_id] + [pick.add]
        candidates = [m.add for m in scored[1:RECHECK + 1]]
    return out


def _explain(fa: PlayerRow, drop: PlayerRow, next_gain: float, ros_gain: float) -> tuple[str, str]:
    bits = []
    if fa.position in ("DST", "K"):
        tag = "stream"
        bits.append(f"{fa.position} streamer, {_next_week(fa):.1f} projected this week")
        if fa.projection_source and "hold" in fa.projection_source:
            bits.append("favourable next week too")
    elif next_gain > 0 and ros_gain > 0:
        tag = "upgrade"
        bits.append(f"+{next_gain:.1f} this week and +{ros_gain:.1f}/game rest of season")
    elif next_gain > ros_gain:
        tag = "stream"
        bits.append(f"+{next_gain:.1f} this week over {drop.name}")
    else:
        tag = "stash"
        bits.append(f"+{ros_gain:.1f}/game rest of season; worth holding through this week")
    if fa.on_bye:
        bits.append("on bye this week")
    if drop.on_bye:
        bits.append(f"{drop.name} is on bye")
    elif drop.is_out:
        bits.append(f"{drop.name} is {drop.status or 'out'}")
    if fa.percent_owned is not None and fa.percent_owned < 25:
        bits.append(f"{fa.percent_owned:.0f}% rostered")
    return tag, "; ".join(bits)


def drop_watch(snapshot: LeagueSnapshot, n: int = 4, max_week: int = 18) -> list[PlayerRow]:
    """The roster spots whose loss would cost the least, for pruning before waivers."""
    me = snapshot.my_team
    slots = snapshot.starting_slots
    w_next = horizon_weight(snapshot.week, max_week)
    roster = [p for p in me.roster if _has_value(p) and not p.in_ir_slot]
    pool = [fa for fa in snapshot.free_agents if _has_value(fa)]
    rep_next = {pos: (v[0] if v else 0.0) for pos, v in _replacement_levels(pool, _next_week).items()}
    rep_ros = {pos: (v[0] if v else 0.0) for pos, v in _replacement_levels(pool, _ros).items()}
    base_next = roster_value(roster, slots, _next_week, rep_next)
    base_ros = roster_value(roster, slots, _ros, rep_ros)

    def loss(p: PlayerRow) -> float:
        rest = [q for q in roster if q.player_id != p.player_id]
        d_next = base_next - roster_value(rest, slots, _next_week, rep_next)
        d_ros = base_ros - roster_value(rest, slots, _ros, rep_ros)
        return w_next * d_next + (1.0 - w_next) * d_ros

    # Ties (several spots that cost nothing) go to the least valuable body.
    return sorted(roster, key=lambda p: (round(loss(p), 6), value(p, w_next)))[:n]


def ir_notes(snapshot: LeagueSnapshot) -> list[str]:
    """Players in an IR slot who are no longer long-term out and need a roster move."""
    notes = []
    for p in snapshot.my_team.roster:
        if p.in_ir_slot and not p.is_long_term_out:
            notes.append(
                f"{p.name} is in your IR slot but listed {p.status or 'healthy'}; the "
                "platform treats the roster as invalid until they are moved to the bench "
                "(needs a free spot) or dropped"
            )
    return notes
