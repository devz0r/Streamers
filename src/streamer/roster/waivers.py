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

Rest of season also prices **upside**. A bench player is worth what he adds
when his role grows past one of your starters, and nothing when it does not
-- an option, so a player whose projection could move a lot (a rookie, a
back one injury from the job) is worth more than a steady one with the same
mean. How far projections move over a month was measured 2021-2025 by
position and level: typically 1.5-2.5 points per game, a quarter wider for
rookies, whose projections also tend to rise (+0.4 points per game over a
month against veterans).

Players sitting in an IR slot are neither drop candidates (dropping them frees
no bench spot) nor part of the lineup (activating them is a roster move).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from scipy.stats import norm

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

#: Positions whose bench bodies carry upside (their role can grow into a
#: starting one). Kickers and defences are streamed week to week instead.
UPSIDE_POSITIONS = ("QB", "RB", "WR", "TE")

#: Share of games a lead back misses (2021-2025: 8.5% of the games after one
#: he played with 12+ expected points). What a handcuff's contingency is worth.
LEAD_BACK_MISS_RATE = 0.085

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
    upside_gain: float = 0.0


def horizon_weight(week: int, max_week: int = 18) -> float:
    """Weight on next week vs rest of season, rising as the season runs out."""
    frac = min(max(week, 1), max_week) / max_week
    return 0.35 + 0.5 * frac


def _next_week(p: PlayerRow) -> float:
    if p.is_out:
        return 0.0
    return float(p.projection or 0.0)


def _next_week_model(p: PlayerRow) -> float:
    """This week, on our model alone -- the basis both sides of a move share
    when the free agents carry no platform projection."""
    if p.is_out:
        return 0.0
    own = p.model_projection if p.model_projection is not None else p.projection
    return float(own or 0.0)


def next_week_basis(snapshot: LeagueSnapshot):
    """The next-week valuation that prices roster and wire on the same footing.

    Rostered players carry the platform's projection blended into ours; if the
    free agents do not (Yahoo's player list does not reliably say which stat
    its points column shows, so it is not trusted), a waiver move would be
    comparing a blended number against a model-only one. In that case every
    player is valued on the model alone. Rest-of-season value is model-only
    already, for everyone.
    """
    roster = [p for p in snapshot.my_team.roster if p.projection is not None]
    wire = [p for p in snapshot.free_agents if p.projection is not None]
    if not roster or not wire:
        return _next_week
    roster_cover = sum(p.platform_projection is not None for p in roster) / len(roster)
    wire_cover = sum(p.platform_projection is not None for p in wire) / len(wire)
    if roster_cover >= 0.5 and wire_cover < 0.5:
        return _next_week_model
    return _next_week


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


def lineup_slots(players: list[PlayerRow], slots: dict[str, int], key: Key) -> dict[str, str]:
    """Greedy best lineup on ``key``: player id -> the slot he fills.

    Dedicated slots first, flex slots last; exact for the usual one-flex
    structure and a close approximation otherwise.
    """
    ordered = sorted(
        slots.items(),
        key=lambda kv: (len(SLOT_ELIGIBILITY.get(kv[0], (kv[0],))), kv[0]),
    )
    ranked = sorted(players, key=key, reverse=True)
    assigned: dict[str, str] = {}
    for slot, count in ordered:
        taken = 0
        for p in ranked:
            if taken >= count:
                break
            if p.player_id in assigned or not _eligible(p, slot):
                continue
            assigned[p.player_id] = slot
            taken += 1
    return assigned


def best_lineup(players: list[PlayerRow], slots: dict[str, int], key: Key) -> tuple[set[str], float]:
    """The best lineup's starters and their total on ``key``."""
    assigned = lineup_slots(players, slots, key)
    return set(assigned), sum(key(p) for p in players if p.player_id in assigned)


def upside(mean: float, spread: float, bar: float) -> float:
    """Expected points a game by which a player's role outgrows ``bar``.

    ``E[max(0, V - bar)]`` for V ~ Normal(mean, spread): the option value of
    holding someone who starts for you only if he gets better.
    """
    if spread <= 0:
        return max(mean - bar, 0.0)
    d = (mean - bar) / spread
    return float(spread * (norm.pdf(d) + d * norm.cdf(d)))


def roster_value(
    players: list[PlayerRow], slots: dict[str, int], key: Key, replacement: dict[str, float],
    with_upside: bool = False,
) -> float:
    """Best-lineup points plus replacement-level bench depth, plus -- for the
    rest-of-season horizon -- the upside of bench players whose role could
    grow past the weakest starter they are eligible to replace."""
    assigned = lineup_slots(players, slots, key)
    starters = set(assigned)
    total = sum(key(p) for p in players if p.player_id in assigned)
    bench: dict[str, list[float]] = {}
    for p in players:
        if p.player_id not in starters:
            bench.setdefault(p.position, []).append(key(p))
    for pos, vals in bench.items():
        vals.sort(reverse=True)
        level = replacement.get(pos, 0.0)
        for w, v in zip(BENCH_WEIGHTS.get(pos, ()), vals):
            total += w * max(v - level, 0.0)
    if with_upside:
        by_id = {p.player_id: p for p in players}
        for p in players:
            if p.player_id in starters or p.position not in UPSIDE_POSITIONS or p.ros_sd is None:
                continue
            # He can only take the job of a starter whose slot he can fill.
            rivals = [key(by_id[i]) for i, slot in assigned.items() if _eligible(p, slot)]
            if rivals:
                total += upside(key(p), float(p.ros_sd), min(rivals))
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
    nxt = next_week_basis(snapshot)
    roster = [p for p in me.roster if _has_value(p) and not p.in_ir_slot]
    if not roster:
        return []

    pool_all = [fa for fa in snapshot.free_agents if _has_value(fa)]
    by_pos: dict[str, list[PlayerRow]] = {}
    for fa in pool_all:
        by_pos.setdefault(fa.position, []).append(fa)
    pool: list[PlayerRow] = []
    for fas in by_pos.values():
        fas.sort(key=lambda p: -max(nxt(p), _ros(p)))
        pool.extend(fas[:POOL_PER_POSITION])
    lv_next = _replacement_levels(pool_all, nxt)
    lv_ros = _replacement_levels(pool_all, _ros)

    def score_add(state: list[PlayerRow], fa: PlayerRow) -> Move | None:
        rep_next = _level_excluding(lv_next, fa, nxt)
        rep_ros = _level_excluding(lv_ros, fa, _ros)
        base_next = roster_value(state, slots, nxt, rep_next)
        base_ros = roster_value(state, slots, _ros, rep_ros, with_upside=True)
        base_ros_flat = roster_value(state, slots, _ros, rep_ros)
        best: Move | None = None
        starting_now, _ = best_lineup(state, slots, nxt)
        for drop in droppable(state, w_next, fa.position):
            trial = [p for p in state if p.player_id != drop.player_id] + [fa]
            next_gain = roster_value(trial, slots, nxt, rep_next) - base_next
            ros_gain = roster_value(trial, slots, _ros, rep_ros, with_upside=True) - base_ros
            flat_gain = roster_value(trial, slots, _ros, rep_ros) - base_ros_flat
            score = w_next * next_gain + (1.0 - w_next) * ros_gain
            # Who the pickup would actually start over this week, if anyone:
            # "add a QB, drop a WR, +3.5" is baffling until it says the QB is
            # taking the starting quarterback's job.
            starting_after, _ = best_lineup(trial, slots, nxt)
            benched = [p for p in state if p.player_id in starting_now
                       and p.player_id not in starting_after and p.player_id != drop.player_id]
            starts_over = benched[0] if fa.player_id in starting_after and benched else None
            # On a tie, prefer dropping at the pickup's own position (the old
            # defence for the new one) over some other zero-value body.
            better = best is None or score > best.score + 1e-9
            tie_same_pos = (best is not None and abs(score - best.score) <= 1e-9
                            and drop.position == fa.position and best.drop.position != fa.position)
            if better or tie_same_pos:
                up = (1.0 - w_next) * (ros_gain - flat_gain)
                tag, reason = _explain(fa, drop, next_gain, ros_gain, nxt, starts_over, up)
                best = Move(add=fa, drop=drop, next_week_gain=next_gain,
                            ros_gain=ros_gain, score=score, tag=tag, reason=reason, upside_gain=up)
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


def _explain(fa: PlayerRow, drop: PlayerRow, next_gain: float, ros_gain: float,
             nxt=_next_week, starts_over: PlayerRow | None = None,
             upside_gain: float = 0.0) -> tuple[str, str]:
    bits = list(fa.signals)
    if fa.position in ("DST", "K"):
        tag = "stream"
        bits.append(f"{fa.position} streamer, {nxt(fa):.1f} projected this week")
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
    if starts_over is not None:
        bits.append(f"would start over {starts_over.name} "
                    f"({nxt(fa):.1f} vs {nxt(starts_over):.1f} projected)")
    if upside_gain >= 0.1:
        bits.append(f"+{upside_gain:.1f} of that is upside, the chance his projection climbs past your starter"
                    + (f"; {_youth(fa)}" if _youth(fa) else ""))
    elif _youth(fa):
        bits.append(_youth(fa))
    if fa.on_bye:
        bits.append("on bye this week")
    if drop.on_bye:
        bits.append(f"{drop.name} is on bye")
    elif drop.is_out:
        bits.append(f"{drop.name} is {drop.status or 'out'}")
    if fa.percent_owned is not None and fa.percent_owned < 25:
        bits.append(f"{fa.percent_owned:.0f}% rostered")
    return tag, "; ".join(bits)


def _youth(p: PlayerRow) -> str:
    if p.experience == 0:
        return "rookie, and rookies' roles tend to grow"
    if p.experience == 1:
        return "second-year player"
    return ""


@dataclass
class Stash:
    """A free agent worth a spare bench spot for what he could become."""

    player: PlayerRow
    ceiling: float                 # his per-game projection a month out, 85th percentile
    value: float                   # upside to your lineup, points a game
    reasons: list[str] = field(default_factory=list)


def handcuffs(snapshot: LeagueSnapshot) -> dict[str, tuple[PlayerRow, float]]:
    """Free-agent backs next in line behind a healthy lead back: player id ->
    (the starter, the extra points a game he would inherit if the starter
    misses time). History: the next man up averaged 13.9 points in the first
    game of an absence, against 10.7 for backs projected the same."""
    everyone = snapshot.all_players()
    leads = {p.team: p for p in everyone
             if p.position == "RB" and p.role == "RB1" and not p.is_out and (p.ros_value or 0) >= 12.0}
    out = {}
    for fa in snapshot.free_agents:
        if fa.position != "RB" or fa.role != "RB2" or fa.team not in leads:
            continue
        lead = leads[fa.team]
        gain = 0.35 * max(float(lead.ros_value or 0) - float(fa.ros_value or 0), 0.0)
        if gain >= 1.5:
            out[fa.player_id] = (lead, gain)
    return out


def stashes(snapshot: LeagueSnapshot, moves: list[Move], n: int = 3) -> list[Stash]:
    """Lottery tickets: free agents whose value could jump, even though their
    projection does not yet clear the bar -- a back behind a lead back, a
    rookie whose role is growing, a player whose opportunity just rose.

    Ranked by upside to *your* lineup: the expected points a game by which he
    could outgrow the weakest starter he is eligible to replace, plus, for a
    handcuff, the share of games a lead back misses times what the job is
    worth. A backup quarterback in a one-QB league scores near zero here
    however good he is, because he would have to beat your starter.
    """
    taken = {m.add.player_id for m in moves}
    cuffs = handcuffs(snapshot)
    roster = [p for p in snapshot.my_team.roster if _has_value(p) and not p.in_ir_slot]
    assigned = lineup_slots(roster, snapshot.starting_slots, _ros)
    by_id = {p.player_id: p for p in roster}
    out = []
    for fa in snapshot.free_agents:
        if (fa.player_id in taken or fa.position not in UPSIDE_POSITIONS or fa.is_long_term_out
                or fa.ros_value is None or fa.ros_sd is None or not fa.team):
            continue
        reasons = [s for s in fa.signals if not s.startswith("opportunity down")]
        rivals = [_ros(by_id[i]) for i, slot in assigned.items() if _eligible(fa, slot)]
        if not rivals:
            continue
        bar = min(rivals)
        ros, spread = float(fa.ros_value), float(fa.ros_sd)
        value = upside(ros, spread, bar)
        ceiling = ros + 1.04 * spread
        if fa.player_id in cuffs:
            lead, gain = cuffs[fa.player_id]
            reasons.append(f"handcuff to {lead.name}: about +{gain:.1f} a game if he misses time")
            value += LEAD_BACK_MISS_RATE * max(ros + gain - bar, 0.0)
            ceiling = max(ceiling, ros + gain)
        if fa.experience == 0:
            reasons.append(_youth(fa))
        if not reasons or value < 0.05:
            continue
        out.append(Stash(player=fa, ceiling=ceiling, value=value, reasons=reasons))
    out.sort(key=lambda s: -s.value)
    return out[:n]


def drop_watch(snapshot: LeagueSnapshot, n: int = 4, max_week: int = 18) -> list[PlayerRow]:
    """The roster spots whose loss would cost the least, for pruning before waivers."""
    me = snapshot.my_team
    slots = snapshot.starting_slots
    w_next = horizon_weight(snapshot.week, max_week)
    nxt = next_week_basis(snapshot)
    roster = [p for p in me.roster if _has_value(p) and not p.in_ir_slot]
    pool = [fa for fa in snapshot.free_agents if _has_value(fa)]
    rep_next = {pos: (v[0] if v else 0.0) for pos, v in _replacement_levels(pool, nxt).items()}
    rep_ros = {pos: (v[0] if v else 0.0) for pos, v in _replacement_levels(pool, _ros).items()}
    base_next = roster_value(roster, slots, nxt, rep_next)
    base_ros = roster_value(roster, slots, _ros, rep_ros, with_upside=True)

    def loss(p: PlayerRow) -> float:
        rest = [q for q in roster if q.player_id != p.player_id]
        d_next = base_next - roster_value(rest, slots, nxt, rep_next)
        d_ros = base_ros - roster_value(rest, slots, _ros, rep_ros, with_upside=True)
        return w_next * d_next + (1.0 - w_next) * d_ros

    # Ties (several spots that cost nothing) go to the least valuable body.
    return sorted(roster, key=lambda p: (round(loss(p), 6), value(p, w_next)))[:n]


def ir_notes(snapshot: LeagueSnapshot) -> list[str]:
    """Players filling an IR slot that their status does not entitle them to.

    An OUT designation is enough for the IR slot in most leagues, so only a
    player who is active, questionable or day-to-day is flagged.
    """
    notes = []
    for p in snapshot.my_team.roster:
        if not p.ir_slot_is_valid:
            notes.append(
                f"{p.name} is in your IR slot but listed {p.status or 'healthy'}; the "
                "platform treats the roster as invalid until they are moved to the bench "
                "(needs a free spot) or dropped"
            )
    return notes
