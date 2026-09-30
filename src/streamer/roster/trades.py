"""Trades: what they do to your title odds, and whether the other side says yes.

Two questions, answered separately because they are about different people:

1. **Does it help you?** Priced on the season simulator, for both teams at
   once, on the title engine's simulated seasons: your P(title) with the new
   roster against as it stands. Our projections, our futures.
2. **Will he say yes?** He cannot see this tool. He judges on what he can
   see -- the platform's projection, a player's name and last season, how
   he has started this one, whether he is hurt -- blended in the proportions
   the market was measured to use (:mod:`streamer.roster.perception`). His
   perceived gain is the change in his roster value on that blend (best
   lineup plus bench depth, the way he would set it), plus a consolidation
   effect in uneven deals (the side getting a clearly best player feels it
   won), turned into a rough chance he accepts and scaled by how engaged he
   is. His own players are valued at least as high as his lineup says he
   rates them: a player he starts over a healthy bench option is worth at
   least that bench option to him, whatever the market says.

The gap between the two views is the point. A player the market rates above
our projection -- a big name in a shrinking role, a hot start on thin
opportunity -- is worth more to sell than to hold; one it rates below ours is
worth buying. Trades that exploit that gap help you *and* look good to him.

Three lists from the same candidates:

* **Top trades** -- by expected gain: chance he accepts x title odds gained.
* **Best for you** -- by title odds gained, among offers with at least a
  15% chance.
* **Most likely yes** -- by chance he accepts, among offers that clearly
  raise your title odds.

Searched: every 1-for-1, 2-for-1 and 1-for-2 of skill players with every
team, screened on roster value (ours for you, the perceived blend for him),
and the best screened offers for each list priced in title odds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..league.model import LeagueSnapshot, PlayerRow, TeamRow
from . import perception
from .season import SeasonModel, replacement_levels
from .waivers import MIN_KEEP, _eligible, _ros, lineup_slots, roster_value

SKILL = ("QB", "RB", "WR", "TE")

#: Title-odds gains below this (in probability) are not worth a trade.
MIN_GAIN = 0.003
#: Screening floors: your roster value (points a game, our projections) and
#: his chance of accepting.
MY_MIN_VALUE = 0.2
MIN_ACCEPT = 0.05
#: "Best for you" only lists offers with at least this chance of a yes.
BEST_MIN_ACCEPT = 0.15
#: Offers priced in the simulator for each list, and per partner in each.
PRICE_EACH = 24
PRICE_PER_TEAM = 5
#: Gaps between the market's view and ours worth pointing out (points a game).
GAP_NOTE = 1.0
#: How much better the best player in an uneven deal must be than the best on
#: the other side before the consolidation effect is full. A player barely
#: ahead is not "the best player in the deal" to anyone: Xavier Worthy, 0.4 a
#: game ahead of Kenyon Sadiq in the market's eyes, earned the full effect and
#: a 62% chance for a two-for-one his owner would laugh at. About the spread
#: of managers' opinions of one player (the scale of the rostership fit, 1.7
#: points a game).
CONSOLIDATION_GAP = 1.7
#: Bench slots, for reading a manager's lineup.
BENCH_SLOTS = ("BN", "BE")


def own_view(team: TeamRow, market) -> tuple[dict[str, float], dict[str, str]]:
    """How a manager values his own players, where his lineup says more than
    the market does.

    Starting one player over a healthy bench player who could fill that slot
    says he rates the starter at least as high, so the starter is raised to
    the bench player's market value. Only raised: a lineup can be a week old
    and set before the news, so a bench player is never marked down for
    sitting (a backup promoted by a Monday injury still sits on a bench set
    the week before). Returns player id -> his value, for the
    players it moves, and player id -> the bench player he starts him over.
    """
    values: dict[str, float] = {}
    over: dict[str, str] = {}
    bench = [p for p in team.roster if p.slot in BENCH_SLOTS and p.position in SKILL
             and not p.is_out and not p.is_questionable and not p.unsigned]
    for s in team.roster:
        if not s.starting or s.position not in SKILL or s.is_out:
            continue
        rivals = [b for b in bench if _eligible(b, s.slot) and market(b) > market(s)]
        if rivals:
            top = max(rivals, key=market)
            values[s.player_id], over[s.player_id] = market(top), top.name
    return values, over


@dataclass
class Trade:
    partner: TeamRow
    give: list[PlayerRow]
    get: list[PlayerRow]
    their_drop: PlayerRow | None
    my_drop: PlayerRow | None
    my_value: float           # your roster value change, points a game, our projections
    my_lineup: float          # your starting lineup change, our projections
    their_seen: float         # his roster value change as he sees it
    consolidation: float      # +1 he gets a clearly best player of an uneven deal, -1 he gives him up
    p_accept: float
    p_base: float = 0.0
    p_new: float = 0.0
    their_base: float = 0.0
    their_new: float = 0.0
    noise: float = 0.0
    why_you: list[str] = field(default_factory=list)
    why_them: list[str] = field(default_factory=list)

    @property
    def gain(self) -> float:
        return self.p_new - self.p_base

    @property
    def their_gain(self) -> float:
        return self.their_new - self.their_base

    @property
    def expected(self) -> float:
        return self.p_accept * self.gain

    @property
    def win_win(self) -> bool:
        return self.their_gain >= 0

    @property
    def key(self) -> tuple:
        return (self.partner.team_id, tuple(sorted(p.player_id for p in self.give)),
                tuple(sorted(p.player_id for p in self.get)))


@dataclass
class TradeBoard:
    top: list[Trade]
    best: list[Trade]
    likely: list[Trade]
    screened: int = 0
    priced: int = 0

    def __bool__(self) -> bool:
        return bool(self.top or self.best or self.likely)


def _tradeable(team: TeamRow) -> list[PlayerRow]:
    return [p for p in team.roster
            if p.position in SKILL and not p.in_ir_slot and (p.ros_value is not None or p.projection)]


def _keeps_minimums(roster: list[PlayerRow]) -> bool:
    counts: dict[str, int] = {}
    for p in roster:
        if not p.in_ir_slot:
            counts[p.position] = counts.get(p.position, 0) + 1
    return all(counts.get(pos, 0) >= n for pos, n in MIN_KEEP.items() if pos in SKILL)


def _make_room(roster: list[PlayerRow], extra: int, keep: set[str], key) -> tuple[list[PlayerRow], PlayerRow | None]:
    """Drop ``extra`` of the least valuable bench players on ``key`` (never one just received)."""
    dropped = None
    for _ in range(extra):
        options = [p for p in roster if p.player_id not in keep and not p.in_ir_slot]
        options = [p for p in options if _keeps_minimums([q for q in roster if q is not p])]
        if not options:
            break
        worst = min(options, key=key)
        roster = [p for p in roster if p is not worst]
        dropped = dropped or worst
    return roster, dropped


class TradeFinder:
    """Screen every small trade, then price the promising ones in title odds.

    ``model`` is a :class:`SeasonModel` already built for the snapshot (the
    title engine's), so trades and waiver moves are priced on the same
    simulated seasons. ``history`` (weekly points, :func:`load_history`)
    gives the other managers' view of each player; without it their view
    falls back to the platform projection.
    """

    def __init__(self, snapshot: LeagueSnapshot, model: SeasonModel, history=None,
                 seen: dict[str, perception.Perceived] | None = None):
        self.snapshot, self.model = snapshot, model
        self.slots = snapshot.starting_slots
        self.replacement = replacement_levels(snapshot)
        self.seen = seen if seen is not None else perception.for_snapshot(snapshot, history)
        self.seen_replacement = self._seen_replacement()
        self.own = {t.team_id: own_view(t, self.theirs) for t in snapshot.teams if not t.is_mine}
        self.me = snapshot.my_team
        self.mine = model.team_index[self.me.team_id]
        weeks = max(int(snapshot.week) - 1, 1)
        self.engaged = {t.team_id: perception.engagement(t.acquisitions, weeks) for t in snapshot.teams}
        base = model.odds()
        self.base_champ = base.champion
        self.base_p = base.p_title

    # -- the two views of a player ----------------------------------------
    def ours(self, p: PlayerRow) -> float:
        return _ros(p)

    def theirs(self, p: PlayerRow) -> float:
        s = self.seen.get(p.player_id)
        return s.value if s is not None else _ros(p)

    def his(self, team: TeamRow):
        """A manager's view: the market's, raised where his lineup rates his
        own players higher."""
        own = self.own.get(team.team_id, ({}, {}))[0]
        return lambda p: own.get(p.player_id, self.theirs(p))

    def _seen_replacement(self) -> dict[str, float]:
        out = {}
        for pos in SKILL:
            vals = sorted((self.theirs(p) for p in self.snapshot.free_agents
                           if p.position == pos and not p.is_long_term_out and not p.unsigned),
                          reverse=True)
            if vals:
                out[pos] = vals[min(1, len(vals) - 1)]
        return out

    def my_roster_value(self, roster: list[PlayerRow]) -> float:
        return roster_value(roster, self.slots, self.ours, self.replacement, with_upside=True)

    def seen_roster_value(self, roster: list[PlayerRow], key=None) -> float:
        return roster_value(roster, self.slots, key or self.theirs, self.seen_replacement)

    def _lineup(self, roster: list[PlayerRow], key) -> float:
        assigned = lineup_slots(roster, self.slots, key)
        return sum(key(p) for p in roster if p.player_id in assigned)

    # -- screening --------------------------------------------------------
    def screen(self) -> list[Trade]:
        me = self.me
        mine = _tradeable(me)
        my0 = self.my_roster_value(me.roster)
        my_l0 = self._lineup(me.roster, self.ours)
        out: list[Trade] = []
        for team in self.snapshot.teams:
            if team.is_mine or not team.roster:
                continue
            theirs = _tradeable(team)
            their0 = self.seen_roster_value(team.roster, self.his(team))
            for n_give, n_get in ((1, 1), (2, 1), (1, 2)):
                for give in combinations(mine, n_give):
                    for get in combinations(theirs, n_get):
                        t = self._screen_one(team, list(give), list(get), my0, my_l0, their0)
                        if t is not None:
                            out.append(t)
        return out

    def _screen_one(self, team, give, get, my0, my_l0, their0):
        his = self.his(team)
        give_ids = {p.player_id for p in give}
        get_ids = {p.player_id for p in get}
        my_new = [p for p in self.me.roster if p.player_id not in give_ids] + get
        their_new = [p for p in team.roster if p.player_id not in get_ids] + give
        my_drop = their_drop = None
        if len(get) > len(give):
            my_new, my_drop = _make_room(my_new, len(get) - len(give), get_ids,
                                         lambda p: self.ours(p) + float(p.ros_sd or 0.0))
        if len(give) > len(get):
            their_new, their_drop = _make_room(their_new, len(give) - len(get), give_ids, his)
        if not (_keeps_minimums(my_new) and _keeps_minimums(their_new)):
            return None
        dv_me = self.my_roster_value(my_new) - my0
        if dv_me < MY_MIN_VALUE:
            return None
        seen_gain = self.seen_roster_value(their_new, his) - their0
        consolidation = self._consolidation(give, get, his)
        p = perception.p_accept(seen_gain, consolidation, self.engaged.get(team.team_id, 0.8))
        if p < MIN_ACCEPT:
            return None
        return Trade(partner=team, give=give, get=get, their_drop=their_drop, my_drop=my_drop,
                     my_value=dv_me, my_lineup=self._lineup(my_new, self.ours) - my_l0,
                     their_seen=seen_gain, consolidation=consolidation, p_accept=p)

    @staticmethod
    def _consolidation(give, get, his) -> float:
        """+1 when he receives the smaller side of an uneven deal and it holds
        a clearly best player, -1 when he gives that side up, in between as
        the best player is less clearly best, 0 for an even count."""
        if len(give) == len(get):
            return 0.0
        he_gets_fewer = len(give) < len(get)
        fewer, more = (give, get) if he_gets_fewer else (get, give)
        lead = max(map(his, fewer)) - max(map(his, more))
        if lead <= 0:
            return 0.0
        clear = min(lead / CONSOLIDATION_GAP, 1.0)
        return clear if he_gets_fewer else -clear

    # -- pricing ----------------------------------------------------------
    def _rosters(self, t: Trade) -> tuple[list[str], list[str]]:
        give_ids = {p.player_id for p in t.give}
        get_ids = {p.player_id for p in t.get}
        mine = [p.player_id for p in self.me.roster if p.player_id not in give_ids] + list(get_ids)
        theirs = [p.player_id for p in t.partner.roster if p.player_id not in get_ids] + list(give_ids)
        if t.my_drop is not None:
            mine = [i for i in mine if i != t.my_drop.player_id]
        if t.their_drop is not None:
            theirs = [i for i in theirs if i != t.their_drop.player_id]
        return mine, theirs

    def price(self, t: Trade) -> Trade:
        mine, theirs = self._rosters(t)
        m = self.model
        odds = m.odds(override={self.me.team_id: m.team_scores(mine, owner=self.me.team_id),
                                t.partner.team_id: m.team_scores(theirs, owner=t.partner.team_id)})
        k = m.team_index[t.partner.team_id]
        won_new = (odds.champion == self.mine).astype(float)
        won_base = (self.base_champ == self.mine).astype(float)
        t.p_base, t.p_new = float(self.base_p[self.mine]), float(odds.p_title[self.mine])
        t.their_base, t.their_new = float(self.base_p[k]), float(odds.p_title[k])
        t.noise = float((won_new - won_base).std() / np.sqrt(len(won_new)))
        return t

    @staticmethod
    def _shortlist(trades: list[Trade], key, n: int, per_team: int) -> list[Trade]:
        out, per = [], {}
        for t in sorted(trades, key=key, reverse=True):
            if per.get(t.partner.team_id, 0) >= per_team:
                continue
            per[t.partner.team_id] = per.get(t.partner.team_id, 0) + 1
            out.append(t)
            if len(out) >= n:
                break
        return out

    @staticmethod
    def _distinct(trades: list[Trade], key, n: int, per_team: int = 2) -> list[Trade]:
        """Best first, one version of each deal: a variant that adds a throw-in
        or swaps the add-on for the same headline player is left out."""
        out, per, used = [], {}, set()
        for t in sorted(trades, key=key, reverse=True):
            got = (t.partner.team_id, "get", tuple(sorted(p.player_id for p in t.get)))
            gave = (t.partner.team_id, "give", tuple(sorted(p.player_id for p in t.give)))
            if got in used or gave in used or per.get(t.partner.team_id, 0) >= per_team:
                continue
            used |= {got, gave}
            per[t.partner.team_id] = per.get(t.partner.team_id, 0) + 1
            out.append(t)
            if len(out) >= n:
                break
        return out

    def find(self, n: int = 5) -> TradeBoard:
        screened = self.screen()
        pools = [
            self._shortlist(screened, lambda t: t.p_accept * t.my_value, PRICE_EACH, PRICE_PER_TEAM),
            self._shortlist([t for t in screened if t.p_accept >= BEST_MIN_ACCEPT],
                            lambda t: t.my_value, PRICE_EACH, PRICE_PER_TEAM),
            self._shortlist([t for t in screened if t.my_value >= 2 * MY_MIN_VALUE],
                            lambda t: t.p_accept, PRICE_EACH, PRICE_PER_TEAM),
        ]
        chosen: dict[tuple, Trade] = {}
        for pool in pools:
            for t in pool:
                chosen.setdefault(t.key, t)
        priced = [self.price(t) for t in chosen.values()]
        good = [t for t in priced if t.gain >= max(MIN_GAIN, 2.0 * t.noise)]
        for t in good:
            self._explain(t)
        return TradeBoard(
            top=self._distinct(good, lambda t: t.expected, n),
            best=self._distinct([t for t in good if t.p_accept >= BEST_MIN_ACCEPT], lambda t: t.gain, n),
            likely=self._distinct(good, lambda t: t.p_accept, n),
            screened=len(screened), priced=len(priced))

    # -- reasons ----------------------------------------------------------
    def _explain(self, t: Trade) -> None:
        you = [f"your lineup {t.my_lineup:+.1f} a game on our projections"]
        for p in t.give:
            gap = self.theirs(p) - self.ours(p)
            if gap >= GAP_NOTE:
                you.append(f"sell high: the market sees {p.name} at {self.theirs(p):.1f} a game, "
                           f"we project {self.ours(p):.1f}")
        for p in t.get:
            gap = self.ours(p) - self.theirs(p)
            if gap >= GAP_NOTE:
                you.append(f"buy low: we project {p.name} at {self.ours(p):.1f}, "
                           f"the market sees {self.theirs(p):.1f}")
            you.extend(f"{p.name}: {s}" for s in p.signals[:1])
        if t.my_drop is not None:
            you.append(f"you would drop {t.my_drop.name} to make room")
        elif len(t.get) < len(t.give):
            you.append("opens a roster spot for a pickup")
        t.why_you = you

        them = [f"as he would see it, his roster {t.their_seen:+.1f} a game"]
        his = self.his(t.partner)
        over = self.own.get(t.partner.team_id, ({}, {}))[1]
        for p in t.get:
            if p.player_id in over and his(p) >= self.theirs(p) + 0.3:
                them.append(f"he starts {p.name} over {over[p.player_id]}, so he rates him above the "
                            f"market's {self.theirs(p):.1f} a game")
        old = t.partner.roster
        new_ids = ({p.player_id for p in old} - {p.player_id for p in t.get}) | {p.player_id for p in t.give}
        if t.their_drop is not None:
            new_ids.discard(t.their_drop.player_id)
        by_id = {p.player_id: p for p in old + t.give}
        new = [by_id[i] for i in new_ids]
        start_old = set(lineup_slots(old, self.slots, his))
        start_new = set(lineup_slots(new, self.slots, his))
        benched = [by_id[i] for i in start_old & new_ids if i not in start_new]
        for p in t.give:
            if p.player_id in start_new and benched:
                worst = min(benched, key=his)
                them.append(f"{p.name} would start for him over {worst.name}")
                break
        if t.consolidation >= 0.5:
            them.append("he gets the best player in the deal")
        elif t.consolidation <= -0.5:
            them.append("he gives up the best player in the deal, a harder sell")
        for p in t.give:
            s = self.seen.get(p.player_id)
            if s is None:
                continue
            if s.last_ppg is not None and s.last_games >= 6 and s.last_ppg >= self.ours(p) + 2:
                them.append(f"name value: {p.name} scored {s.last_ppg:.1f} a game last season")
            if s.now_ppg is not None and s.now_games >= 2 and s.now_ppg >= s.platform + 3:
                them.append(f"{p.name} is averaging {s.now_ppg:.1f} this season")
        for p in t.get:
            s = self.seen.get(p.player_id)
            if p.is_out or p.status:
                them.append(f"{p.name} is listed {p.status or 'out'}")
            elif s is not None and s.now_ppg is not None and s.now_games >= 2 and s.now_ppg <= s.platform - 3:
                them.append(f"{p.name} has averaged only {s.now_ppg:.1f} so far")
        acq, engaged = t.partner.acquisitions, self.engaged.get(t.partner.team_id, 0.8)
        if acq is not None and engaged < 0.75:
            them.append(f"he rarely makes moves ({acq} pickups), so any offer is a long shot")
        elif acq is not None and engaged >= 0.95:
            them.append(f"an active manager ({acq} pickups)")
        if t.win_win:
            them.append(f"his title odds rise too ({t.their_base:.1%} to {t.their_new:.1%})")
        if t.their_drop is not None:
            them.append(f"he would drop {t.their_drop.name} to make room")
        t.why_them = them


def acceptance_label(p: float) -> str:
    return "likely" if p >= 0.5 else "possible" if p >= 0.25 else "long shot"
