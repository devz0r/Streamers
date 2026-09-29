"""Several waiver moves at once, priced together in title odds.

Single moves are priced one at a time, and moves interact: two backs
chasing one lineup spot are worth less together than apart, while cutting
two dead roster spots for a starter and his handcuff can be worth more.
A plan is a set of two or three (add, drop) pairs made together.

Every combination of the most promising free agents and your weakest
players is screened on roster value (best lineup, bench depth over
replacement, bench upside -- the waiver engine's measure), keeping one
drop-set per group of adds. The best plans are then priced in P(title) on
the title engine's simulated seasons, against standing pat and against the
best single move, with paired noise. A plan that beats the best single
move by more than twice that noise is recommended; one that beats it by
less is shown as a close call (the single move gets most of the gain).

Claims go in the order of their single-move value. On a rolling waiver list
the first claim that succeeds spends your priority; the rest are processed
behind it and land when nobody ahead of you wants the same player. Players
not on waivers (free agents) are added at once and cost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..league.model import PlayerRow
from .season import replacement_levels
from .title_moves import DROP_TIE, MIN_GAIN, TitleEngine
from .waivers import MIN_KEEP, _ros, roster_value

SKILL = ("QB", "RB", "WR", "TE")

#: Free agents and roster players considered, and plans priced.
ADD_POOL = 10
DROP_POOL = 8
PRICE_PLANS = 12
SIZES = (2, 3)


@dataclass
class Plan:
    moves: list[tuple[PlayerRow, PlayerRow]]          # (add, drop), in claim order
    proxy: float                                       # roster value change, points a game
    p_base: float = 0.0
    p_plan: float = 0.0
    p_single: float = 0.0                              # the best single move's P(title)
    single_gains: dict[str, float] = field(default_factory=dict)   # add id -> gain alone
    noise: float = 0.0
    #: "plan" when it beats the best single move by more than the noise,
    #: "lean" when it beats it by less (a close call).
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)
    #: Drops swapped to keep a player the market values: (kept, dropped instead).
    kept: list[tuple[PlayerRow, PlayerRow]] = field(default_factory=list)
    #: Plans one player different and about as good: (add in, add out, P(title)).
    alternatives: list[tuple[PlayerRow, PlayerRow, float]] = field(default_factory=list)

    @property
    def gain(self) -> float:
        return self.p_plan - self.p_base

    @property
    def over_single(self) -> float:
        return self.p_plan - max(self.p_single, self.p_base)

    @property
    def summed(self) -> float:
        return sum(self.single_gains.values())

    @property
    def adds(self) -> list[PlayerRow]:
        return [a for a, _d in self.moves]

    @property
    def drops(self) -> list[PlayerRow]:
        return [d for _a, d in self.moves]


def _counts(roster: list[PlayerRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in roster:
        if not p.in_ir_slot:
            counts[p.position] = counts.get(p.position, 0) + 1
    return counts


def _keeps_minimums(roster: list[PlayerRow], before: list[PlayerRow]) -> bool:
    """No position falls below its minimum -- or below what the roster had,
    if it already had fewer."""
    now, was = _counts(roster), _counts(before)
    return all(now.get(pos, 0) >= min(n, was.get(pos, 0)) for pos, n in MIN_KEEP.items())


#: A drop the market still values this much above our projection (points a
#: game) is worth shopping in a trade before cutting.
TRADE_VALUE_NOTE = 1.0


class PlanFinder:
    def __init__(self, engine: TitleEngine, singles: list | None = None, seen: dict | None = None):
        self.engine = engine
        self.seen = seen or {}
        self.snapshot = engine.snapshot
        self.me = engine.me
        self.slots = self.snapshot.starting_slots
        self.replacement = replacement_levels(self.snapshot)
        self.singles = singles or []
        self._won_cache: dict[tuple, np.ndarray] = {}

    def value(self, roster: list[PlayerRow]) -> float:
        return roster_value(roster, self.slots, _ros, self.replacement, with_upside=True)

    def _won(self, roster: list[PlayerRow]) -> np.ndarray:
        key = tuple(sorted(p.player_id for p in roster))
        if key not in self._won_cache:
            self._won_cache[key] = self.engine._won([p.player_id for p in roster])
        return self._won_cache[key]

    # -- screening ----------------------------------------------------------
    def pools(self) -> tuple[list[PlayerRow], list[PlayerRow], dict[str, tuple[float, PlayerRow]]]:
        roster = self.me.roster
        base = self.value(roster)
        drops = sorted((p for p in roster if p.position in SKILL and not p.in_ir_slot),
                       key=lambda p: _ros(p) + float(p.ros_sd or 0.0))[:DROP_POOL]
        best_alone: dict[str, tuple[float, PlayerRow]] = {}
        for x in self.engine.candidates:
            for y in drops:
                new = [p for p in roster if p is not y] + [x]
                if not _keeps_minimums(new, roster):
                    continue
                g = self.value(new) - base
                if x.player_id not in best_alone or g > best_alone[x.player_id][0]:
                    best_alone[x.player_id] = (g, y)
        adds = sorted((x for x in self.engine.candidates if best_alone.get(x.player_id, (0.0,))[0] > 0),
                      key=lambda x: -best_alone[x.player_id][0])[:ADD_POOL]
        return adds, drops, best_alone

    def screen(self) -> list[Plan]:
        roster = self.me.roster
        base = self.value(roster)
        adds, drops, best_alone = self.pools()
        best_single = max((g for g, _y in best_alone.values()), default=0.0)
        by_adds: dict[tuple, Plan] = {}
        for k in SIZES:
            for add_set in combinations(adds, k):
                for drop_set in combinations(drops, k):
                    new = [p for p in roster if p not in drop_set] + list(add_set)
                    if not _keeps_minimums(new, roster):
                        continue
                    g = self.value(new) - base
                    if g <= best_single:
                        continue                       # no better than one move on paper
                    key = tuple(sorted(p.player_id for p in add_set))
                    if key in by_adds and by_adds[key].proxy >= g:
                        continue
                    by_adds[key] = Plan(moves=self._pair(list(add_set), list(drop_set)), proxy=g)
        return sorted(by_adds.values(), key=lambda p: -p.proxy)

    def _pair(self, adds: list[PlayerRow], drops: list[PlayerRow]) -> list[tuple[PlayerRow, PlayerRow]]:
        """Match each add to a drop for display: the best add replaces the
        weakest drop, and so on down (the roster that results is the same)."""
        adds = sorted(adds, key=lambda p: -_ros(p))
        drops = sorted(drops, key=lambda p: _ros(p) + float(p.ros_sd or 0.0))
        return list(zip(adds, drops))

    # -- pricing -----------------------------------------------------------
    def find(self, n: int = 3) -> list[Plan]:
        screened = self.screen()[:PRICE_PLANS]
        if not screened:
            return []
        roster = self.me.roster
        _adds, _drops, best_alone = self.pools()
        base_won = self.engine.base_won
        p_base = float(base_won.mean())

        def alone(x: PlayerRow) -> np.ndarray:
            y = best_alone[x.player_id][1]
            return self._won([p for p in roster if p is not y] + [x])

        # The best single move, on the same seasons: the engine's top move
        # if it priced one, else the best single among the plans' adds.
        singles = {x.player_id: alone(x) for plan in screened for x in plan.adds}
        for m in self.singles:
            singles.setdefault(m.add.player_id, self._won([p for p in roster if p is not m.drop] + [m.add]))
        best_single = max(singles.values(), key=lambda w: w.mean()) if singles else base_won
        ref = best_single if best_single.mean() >= p_base else base_won

        out = []
        for plan in screened:
            won = self._won([p for p in roster if p not in plan.drops] + plan.adds)
            plan.p_base, plan.p_plan = p_base, float(won.mean())
            plan.p_single = float(best_single.mean())
            plan.single_gains = {x.player_id: float(singles[x.player_id].mean()) - p_base for x in plan.adds}
            plan.noise = float((won - ref).std() / np.sqrt(len(won)))
            # Claim order: the move worth most on its own goes first.
            plan.moves.sort(key=lambda m: -plan.single_gains.get(m[0].player_id, 0.0))
            if plan.over_single >= max(MIN_GAIN, 2.0 * plan.noise):
                plan.verdict = "plan"
            elif plan.over_single > 0 and plan.gain >= max(MIN_GAIN, 2.0 * plan.noise):
                plan.verdict = "lean"
            else:
                continue
            out.append(plan)
        out.sort(key=lambda p: (p.verdict != "plan", -p.p_plan))
        # Distinct plans: one sharing all but one pickup with a better plan is
        # listed on it as an alternative, not as a plan of its own.
        kept: list[Plan] = []
        for plan in out:
            ids = {x.player_id for x in plan.adds}
            twin = next((k for k in kept if len(ids - {x.player_id for x in k.adds}) <= 1), None)
            if twin is not None:
                k_ids = {x.player_id for x in twin.adds}
                if len(ids) == len(k_ids) and len(ids - k_ids) == 1 \
                        and plan.p_plan >= twin.p_plan - 2.0 * twin.noise:
                    new = next(x for x in plan.adds if x.player_id not in k_ids)
                    old = next(x for x in twin.adds if x.player_id not in ids)
                    twin.alternatives.append((new, old, plan.p_plan))
                continue
            kept.append(plan)
            if len(kept) >= n:
                break
        _adds, pool, _best = self.pools()
        for plan in kept:
            self._keep_assets(plan, pool)
            plan.reasons = self._reasons(plan)
        return kept

    def market_value(self, p: PlayerRow) -> float:
        s = self.seen.get(p.player_id)
        return float(s.value) if s is not None else float(self.engine.market_value(p))

    def _keep_assets(self, plan: Plan, pool: list[PlayerRow]) -> None:
        """Where another drop is as good for the title (within noise), cut the
        player the market values least instead: a player others still rate
        can be traded, and cutting him hands him to a rival for nothing."""
        roster = self.me.roster
        for d in sorted(plan.drops, key=lambda q: -self.market_value(q)):
            best = None
            for r in pool:
                if r in plan.drops or self.market_value(r) >= self.market_value(d):
                    continue
                drops = [q for q in plan.drops if q is not d] + [r]
                new = [q for q in roster if q not in drops] + plan.adds
                if not _keeps_minimums(new, roster):
                    continue
                p = float(self._won(new).mean())
                if p >= plan.p_plan - DROP_TIE and (best is None or self.market_value(r) < self.market_value(best[0])):
                    best = (r, p)
            if best is not None:
                r, p = best
                plan.kept.append((d, r))
                plan.moves = self._pair(plan.adds, [q for q in plan.drops if q is not d] + [r])
                plan.moves.sort(key=lambda m: -plan.single_gains.get(m[0].player_id, 0.0))
                plan.p_plan = p

    def _reasons(self, plan: Plan) -> list[str]:
        pts = lambda v: f"{v * 100:+.1f}"            # noqa: E731
        bits = [f"{pts(plan.over_single)} over the best single move"]
        if plan.verdict == "lean":
            bits.append(f"that edge is inside the simulation's noise (+-{2 * plan.noise * 100:.2f}): "
                        "the single move gets most of it")
        tolerance = 2.0 * plan.noise
        if plan.gain > plan.summed + tolerance:
            bits.append(f"worth more together: {pts(plan.gain)} against {pts(plan.summed)} priced one at a time")
        elif plan.gain < plan.summed - tolerance:
            bits.append(f"they overlap: {pts(plan.gain)} together against {pts(plan.summed)} priced one at a time")
        first = plan.moves[0][0]
        rank = self.engine.rank
        if self.engine.priority_waivers:
            bits.append(f"claim {first.name} first: on a rolling list the first claim spends your "
                        f"#{rank} priority, and the rest land if nobody ahead of you wants them")
        for new, old, p in plan.alternatives[:2]:
            bits.append(f"about as good: {new.name} instead of {old.name} (title {p:.1%})")
        for kept, instead in plan.kept:
            bits.append(f"keeps {kept.name}, whom the market still values at {self.market_value(kept):.1f} a game "
                        f"(tradeable, and a rival would claim him): dropping {instead.name} instead is as good "
                        "for your title odds")
        for d in plan.drops:
            s = self.seen.get(d.player_id)
            if s is not None and s.value >= _ros(d) + TRADE_VALUE_NOTE:
                bits.append(f"shop {d.name} in a trade before dropping him: the market sees "
                            f"{s.value:.1f} a game, we project {_ros(d):.1f}")
        for x in plan.adds:
            if x.is_out or x.is_long_term_out:
                bits.append(f"{x.name} is listed {x.status or 'out'}: a stash for when he is back")
        for x in plan.adds:
            if x.usage:
                bits.append(f"{x.name}: {x.usage}")
        return bits
