"""Waiver-wire recommendations: who to add, who to drop, and why.

Every candidate move is an (add, drop) pair scored by *marginal* value -- what
the pickup is worth over the player it displaces -- on two horizons blended by
how much season is left:

* **next week** -- this week's projection, zero if on bye or out;
* **rest of season** -- the per-game projection with no matchup adjustment.

Early in the season the rest-of-season half dominates (a bye next week is not
a reason to skip a real asset); late in the season next week is what matters.
Moves are tagged so the reasoning is visible: an *upgrade* is better on both
horizons, a *stream* is a next-week play, a *stash* is a rest-of-season hold.

D/ST and K come from the streaming model, which already knows about two-week
holds, so a defence that is good this week *and* next is preferred over one
that is good this week only.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..league.model import LeagueSnapshot, PlayerRow

#: Positions a roster must keep at least this many of.
MIN_KEEP = {"QB": 1, "K": 1, "DST": 1, "TE": 1}


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
    if p.status in ("IR", "IR-R", "PUP", "PUP-R", "SUSP", "NFI"):
        return 0.0
    base = p.ros_value if p.ros_value is not None else (p.projection or 0.0)
    return float(base)


def value(p: PlayerRow, w_next: float) -> float:
    return w_next * _next_week(p) + (1.0 - w_next) * _ros(p)


def droppable(roster: list[PlayerRow], w_next: float, position: str) -> list[PlayerRow]:
    """Roster players that could be dropped to make room for ``position``.

    Never drops below the minimum at a core position unless the pickup plays
    that same position, and never drops the best player at any position.
    """
    counts: dict[str, int] = {}
    for p in roster:
        counts[p.position] = counts.get(p.position, 0) + 1
    best_at: dict[str, str] = {}
    for p in sorted(roster, key=lambda x: -value(x, w_next)):
        best_at.setdefault(p.position, p.player_id)

    out = []
    for p in roster:
        if p.player_id == best_at.get(p.position) and p.position != position:
            continue
        need = MIN_KEEP.get(p.position, 0)
        if p.position != position and counts.get(p.position, 0) - 1 < need:
            continue
        out.append(p)
    return sorted(out, key=lambda x: value(x, w_next))


def recommend(
    snapshot: LeagueSnapshot,
    min_gain: float = 1.0,
    max_moves: int = 12,
    max_week: int = 18,
) -> list[Move]:
    """Ranked add/drop moves for the user's team."""
    me = snapshot.my_team
    w_next = horizon_weight(snapshot.week, max_week)
    roster = [p for p in me.roster if p.projection is not None or p.ros_value is not None]

    def build(fa: PlayerRow, taken: set[str]) -> Move | None:
        candidates = [c for c in droppable(roster, w_next, fa.position) if c.player_id not in taken]
        if not candidates:
            return None
        # The best drop is the lowest-value player we are allowed to lose.
        drop = candidates[0]
        next_gain = _next_week(fa) - _next_week(drop)
        ros_gain = _ros(fa) - _ros(drop)
        score = w_next * next_gain + (1.0 - w_next) * ros_gain
        if score < min_gain:
            return None
        tag, reason = _explain(fa, drop, next_gain, ros_gain)
        return Move(add=fa, drop=drop, next_week_gain=next_gain,
                    ros_gain=ros_gain, score=score, tag=tag, reason=reason)

    pool = [fa for fa in snapshot.free_agents if fa.projection is not None or fa.ros_value is not None]
    # Rank pickups by their best-case move, then assign drops greedily so the
    # list reads as a sequence of moves that can all be made together: once a
    # roster spot is spent on one pickup, the next pickup is priced against
    # the next-cheapest spot (and drops out if that no longer clears the bar).
    first_pass = [(fa, build(fa, set())) for fa in pool]
    first_pass = [(fa, m) for fa, m in first_pass if m is not None]
    first_pass.sort(key=lambda fm: -fm[1].score)

    taken: set[str] = set()
    out: list[Move] = []
    for fa, _ in first_pass:
        m = build(fa, taken)
        if m is None:
            continue
        taken.add(m.drop.player_id)
        out.append(m)
        if len(out) >= max_moves:
            break
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
    """The user's lowest-value roster spots, for pruning before waivers run."""
    w_next = horizon_weight(snapshot.week, max_week)
    roster = [p for p in snapshot.my_team.roster if p.projection is not None or p.ros_value is not None]
    return sorted(roster, key=lambda p: value(p, w_next))[:n]
