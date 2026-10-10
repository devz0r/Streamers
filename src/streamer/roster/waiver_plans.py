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
from .title_moves import MIN_GAIN, TOSS_UP_SE, TitleEngine, drop_choice, toss_up_text, toss_ups
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
    #: Toss-up drops: a drop's id -> every player the simulation cannot tell
    #: from him as that drop, with the plan's P(title) each way, best first.
    drop_options: dict[str, list[tuple[PlayerRow, float]]] = field(default_factory=dict)
    #: Plans one player different and about as good: (add in, add out, P(title)).
    alternatives: list[tuple[PlayerRow, PlayerRow, float]] = field(default_factory=list)
    #: Per add: (chance another manager claims him, chance one ahead of you does).
    claim_odds: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Per add: title odds your waiver priority is worth on him (rolling lists).
    priority_worth: dict[str, float] = field(default_factory=dict)

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


def claims_text(plan: Plan) -> str:
    """The adds, in claim order."""
    return ", ".join(a.name for a, _d in plan.moves)


def drops_text(plan: Plan) -> str:
    """The drops as a set: the clear ones by name, a toss-up as "one of A
    (4.8%) or B (4.7%)". Which add is paired with which drop does not
    matter -- the roster that results is the same."""
    options = getattr(plan, "drop_options", None) or {}
    drops = [d for _a, d in plan.moves]
    fixed = [d.name for d in drops if d.player_id not in options]
    open_ = [f"one of {drop_choice(d, options[d.player_id])}" for d in drops if d.player_id in options]
    parts = fixed + open_
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


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
        self._worth: dict[str, float] | None = None

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
                       key=self._cheap)[:DROP_POOL]
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
        drops = sorted(drops, key=self._cheap)
        return list(zip(adds, drops))

    def _cheap(self, p: PlayerRow) -> tuple[float, float]:
        """Cheapest to let go first: what he is worth to your title, then his
        season value plus how far it could move."""
        if self._worth is None:
            self._worth = self.engine.worth_mean()
        return self._worth.get(p.player_id, 0.0), _ros(p) + float(p.ros_sd or 0.0)

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
            singles.setdefault(m.add.player_id, self.engine._won_move(m) if getattr(m, "how", "")
                               else self._won([p for p in roster if p is not m.drop] + [m.add]))
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
            self._drop_options(plan, pool)
            self._claim_order(plan)
            plan.reasons = self._reasons(plan)
        return kept

    def _claim_order(self, plan: Plan) -> None:
        """On a rolling list your first successful claim spends your
        priority, and every claim after it is processed from the back of the
        order: you get those players only if nobody else claims them. So the
        first claim goes where the priority is worth most -- the player you
        would otherwise lose to someone ahead of you or behind -- and a
        player nobody else wants goes last (he clears anyway). Elsewhere
        (FAAB, an order reset by standings) the most valuable goes first."""
        plan.claim_odds = {x.player_id: self.engine.claim_odds(x) for x in plan.adds}
        # Each add's value on its own carries the simulation's noise (+-0.3
        # points), and every add is in the plan because it helps: halfway to
        # the plan's average, so how contested he is decides a close call.
        raw = {x.player_id: max(plan.single_gains.get(x.player_id, 0.0), 0.0) for x in plan.adds}
        mean = sum(raw.values()) / max(len(raw), 1)
        value = {pid: 0.5 * v + 0.5 * mean for pid, v in raw.items()}
        if self.engine.priority_waivers:
            def priority_worth(x: PlayerRow) -> float:
                anyone, ahead = plan.claim_odds[x.player_id]
                return value[x.player_id] * ((1.0 - ahead) - (1.0 - anyone))

            plan.priority_worth = {x.player_id: priority_worth(x) for x in plan.adds}
            first = max(plan.adds, key=lambda x: (priority_worth(x), value[x.player_id]))
            rest = sorted((x for x in plan.adds if x is not first),
                          key=lambda x: -value[x.player_id] * (1.0 - plan.claim_odds[x.player_id][0]))
            order = [first] + rest
        else:
            order = sorted(plan.adds, key=lambda x: -value[x.player_id])
        plan.moves.sort(key=lambda m: order.index(m[0]))

    def market_value(self, p: PlayerRow) -> float:
        s = self.seen.get(p.player_id)
        return float(s.value) if s is not None else float(self.engine.market_value(p))

    def _drop_options(self, plan: Plan, pool: list[PlayerRow]) -> None:
        """Each drop against every other bench player in its place, on the
        same seasons. The best becomes the drop; any the simulation cannot
        tell from it are offered beside it with their title odds -- a
        toss-up is the manager's call (a name he can still trade, a hunch
        about a role). Each alternative is offered once, where it is best."""
        roster = self.me.roster

        def priced(d: PlayerRow) -> list[tuple[PlayerRow, np.ndarray]]:
            out = [(d, self._won([q for q in roster if q not in plan.drops] + plan.adds))]
            for r in pool:
                if r in plan.drops:
                    continue
                drops = [q for q in plan.drops if q is not d] + [r]
                new = [q for q in roster if q not in drops] + plan.adds
                if _keeps_minimums(new, roster):
                    out.append((r, self._won(new)))
            return out

        # A drop is changed only for one clearly better (beyond the noise):
        # taking whichever of eight noisy estimates came out highest would
        # overstate the plan. Then the toss-ups around the finished plan.
        for d in list(plan.drops):
            options = priced(d)
            mine = options[0][1]
            best, best_w = max(options, key=lambda t: float(t[1].mean()))
            diff = best_w - mine
            if best is not d and float(diff.mean()) > TOSS_UP_SE * float(diff.std()) / np.sqrt(len(diff)):
                plan.moves = self._pair(plan.adds, [q for q in plan.drops if q is not d] + [best])
                plan.moves.sort(key=lambda m: -plan.single_gains.get(m[0].player_id, 0.0))
                plan.p_plan = float(best_w.mean())
        found: dict[str, list[tuple[PlayerRow, float]]] = {}
        for d in plan.drops:
            options = toss_ups(priced(d), self.market_value)
            if len(options) > 1 and any(q is d for q, _v in options):
                found[d.player_id] = options
        # A bench player tied for two drops is offered where he is worth most.
        where: dict[str, tuple[str, float]] = {}
        for drop_id, options in found.items():
            for q, v in options:
                if q.player_id != drop_id and (q.player_id not in where or v > where[q.player_id][1]):
                    where[q.player_id] = (drop_id, v)
        for drop_id, options in found.items():
            kept = [(q, v) for q, v in options if q.player_id == drop_id or where[q.player_id][0] == drop_id]
            if len(kept) > 1:
                plan.drop_options[drop_id] = kept

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
        rank = self.engine.rank
        if self.engine.priority_waivers and plan.claim_odds:
            first = plan.moves[0][0]
            odds = [f"{a.name} {plan.claim_odds[a.player_id][0]:.0%} (your priority worth "
                    f"{plan.priority_worth.get(a.player_id, 0.0) * 100:.2f} on him)" for a, _d in plan.moves]
            bits.append(f"claim {first.name} first: your first successful claim spends your #{rank} priority and "
                        "the rest are processed from the back of the order, so it goes on the player you would "
                        "otherwise lose; chance another manager claims each this week: " + ", ".join(odds))
        for new, old, p in plan.alternatives[:2]:
            bits.append(f"about as good: {new.name} instead of {old.name} (title {p:.1%})")
        for options in plan.drop_options.values():
            bits.append(toss_up_text(options, self.market_value))
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
