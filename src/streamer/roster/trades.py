"""Trades that raise your title odds and that the other manager should take.

A trade has to clear two bars, and they are different bars:

1. **They say yes.** A manager judges a trade by what he can see: does his
   roster get better on projections? Measured as his roster value -- best
   lineup, bench depth over replacement level, and the upside of bench
   players who could grow into a starting job -- on per-game projections.
   Only trades that raise it (or leave it level) are offered. That is a
   trade he should accept on the numbers, not one that relies on him
   misreading them.
2. **You get closer to the title.** Priced on the season simulator, for
   both teams, on the same simulated seasons: your P(title) with the new
   roster against as it stands, and his.

Trades that pass the first bar mostly exist because of fit: a team deep at
one position and thin at another values the same players differently from
you. When both title odds rise the trade is a **win-win**; when only yours
does, he still gains on paper -- his lineup is better -- but the simulation
says you gain more from it.

Searched: every 1-for-1, 2-for-1 and 1-for-2 of skill players with every
team, screened on roster value (cheap), and the best screened offers priced
in title odds (each costs a pass through every simulated season). A team
receiving two players for one drops his least valuable bench player; a team
receiving one for two gets an open roster spot, which is left open here
(the simulator fills a hole at replacement level).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..league.model import LeagueSnapshot, PlayerRow, TeamRow
from .season import SeasonModel, replacement_levels
from .waivers import MIN_KEEP, _ros, lineup_slots, roster_value

SKILL = ("QB", "RB", "WR", "TE")

#: Screened offers priced in the simulator, overall and per partner.
PRICE_TOP = 36
PRICE_PER_TEAM = 6
#: Title-odds gains below this (in probability) are not worth a trade.
MIN_GAIN = 0.003
#: What the other manager must see in it: his roster value up by at least
#: this much a game, and his starting lineup no worse than this. A trade
#: that leaves him level gives him no reason to say yes.
THEIR_MIN_VALUE = 0.25
THEIR_MIN_LINEUP = -0.25


@dataclass
class Trade:
    partner: TeamRow
    give: list[PlayerRow]
    get: list[PlayerRow]
    #: What the partner would drop to make room, if he receives more players.
    their_drop: PlayerRow | None
    my_value: float          # roster-value change, points a game (with depth and upside)
    their_value: float
    my_lineup: float         # starting-lineup change, points a game
    their_lineup: float
    p_base: float = 0.0
    p_new: float = 0.0
    their_base: float = 0.0
    their_new: float = 0.0
    noise: float = 0.0
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)

    @property
    def gain(self) -> float:
        return self.p_new - self.p_base

    @property
    def their_gain(self) -> float:
        return self.their_new - self.their_base


def _tradeable(team: TeamRow) -> list[PlayerRow]:
    return [p for p in team.roster
            if p.position in SKILL and not p.in_ir_slot and (p.ros_value is not None or p.projection)]


def _keeps_minimums(roster: list[PlayerRow]) -> bool:
    counts: dict[str, int] = {}
    for p in roster:
        if not p.in_ir_slot:
            counts[p.position] = counts.get(p.position, 0) + 1
    return all(counts.get(pos, 0) >= n for pos, n in MIN_KEEP.items() if pos in SKILL)


def _make_room(roster: list[PlayerRow], extra: int, keep: set[str]) -> tuple[list[PlayerRow], PlayerRow | None]:
    """Drop ``extra`` of the least valuable bench players (not ones just received)."""
    dropped = None
    for _ in range(extra):
        options = [p for p in roster if p.player_id not in keep and not p.in_ir_slot]
        options = [p for p in options if _keeps_minimums([q for q in roster if q is not p])]
        if not options:
            break
        worst = min(options, key=lambda p: _ros(p) + float(p.ros_sd or 0.0))
        roster = [p for p in roster if p is not worst]
        dropped = dropped or worst
    return roster, dropped


def _lineup(players: list[PlayerRow], slots: dict[str, int]) -> float:
    assigned = lineup_slots(players, slots, _ros)
    return sum(_ros(p) for p in players if p.player_id in assigned)


class TradeFinder:
    """Screen every small trade on roster value, then price the best in title odds.

    ``model`` is a :class:`SeasonModel` already built for the snapshot (the
    title engine's), so trades and waiver moves are priced on the same
    simulated seasons.
    """

    def __init__(self, snapshot: LeagueSnapshot, model: SeasonModel):
        self.snapshot, self.model = snapshot, model
        self.slots = snapshot.starting_slots
        self.replacement = replacement_levels(snapshot)
        self.me = snapshot.my_team
        self.mine = model.team_index[self.me.team_id]
        base = model.odds()
        self.base_champ = base.champion
        self.base_p = base.p_title

    def value(self, roster: list[PlayerRow]) -> float:
        return roster_value(roster, self.slots, _ros, self.replacement, with_upside=True)

    # -- screening --------------------------------------------------------
    def screen(self) -> list[Trade]:
        me = self.me
        mine = _tradeable(me)
        my_value0 = self.value(me.roster)
        my_lineup0 = _lineup(me.roster, self.slots)
        out: list[Trade] = []
        for team in self.snapshot.teams:
            if team.is_mine or not team.roster:
                continue
            theirs = _tradeable(team)
            their_value0 = self.value(team.roster)
            their_lineup0 = _lineup(team.roster, self.slots)
            shapes = [(1, 1), (2, 1), (1, 2)]
            for n_give, n_get in shapes:
                for give in combinations(mine, n_give):
                    for get in combinations(theirs, n_get):
                        t = self._screen_one(team, list(give), list(get), my_value0, my_lineup0,
                                             their_value0, their_lineup0)
                        if t is not None:
                            out.append(t)
        return out

    def _screen_one(self, team, give, get, my_value0, my_lineup0, their_value0, their_lineup0):
        give_ids = {p.player_id for p in give}
        get_ids = {p.player_id for p in get}
        my_new = [p for p in self.me.roster if p.player_id not in give_ids] + get
        their_new = [p for p in team.roster if p.player_id not in get_ids] + give
        if not _keeps_minimums(my_new):
            return None
        their_drop = None
        extra = len(give) - len(get)
        if extra > 0:
            their_new, their_drop = _make_room(their_new, extra, give_ids)
        if not _keeps_minimums(their_new):
            return None
        # A roster spot you open is left open; one they open is too.
        extra_mine = len(get) - len(give)
        if extra_mine > 0:
            my_new, _dropped = _make_room(my_new, extra_mine, get_ids)
        dv_them = self.value(their_new) - their_value0
        if dv_them < THEIR_MIN_VALUE:
            return None                                    # nothing in it for him
        dl_them = _lineup(their_new, self.slots) - their_lineup0
        if dl_them < THEIR_MIN_LINEUP:
            return None
        dv_me = self.value(my_new) - my_value0
        if dv_me <= 0:
            return None
        return Trade(partner=team, give=give, get=get, their_drop=their_drop,
                     my_value=dv_me, their_value=dv_them,
                     my_lineup=_lineup(my_new, self.slots) - my_lineup0,
                     their_lineup=dl_them)

    # -- pricing ----------------------------------------------------------
    def _rosters(self, t: Trade) -> tuple[list[str], list[str]]:
        give_ids = {p.player_id for p in t.give}
        get_ids = {p.player_id for p in t.get}
        mine = [p.player_id for p in self.me.roster if p.player_id not in give_ids] + list(get_ids)
        theirs = [p.player_id for p in t.partner.roster if p.player_id not in get_ids] + list(give_ids)
        if t.their_drop is not None:
            theirs = [i for i in theirs if i != t.their_drop.player_id]
        return mine, theirs

    def price(self, t: Trade) -> Trade:
        mine, theirs = self._rosters(t)
        m = self.model
        odds = m.odds(override={self.me.team_id: m.team_scores(mine),
                                t.partner.team_id: m.team_scores(theirs)})
        k = m.team_index[t.partner.team_id]
        won_new = (odds.champion == self.mine).astype(float)
        won_base = (self.base_champ == self.mine).astype(float)
        t.p_base, t.p_new = float(self.base_p[self.mine]), float(odds.p_title[self.mine])
        t.their_base, t.their_new = float(self.base_p[k]), float(odds.p_title[k])
        t.noise = float((won_new - won_base).std() / np.sqrt(len(won_new)))
        return t

    def find(self, n: int = 5, per_team: int = 2) -> list[Trade]:
        screened = sorted(self.screen(), key=lambda t: -t.my_value)
        chosen: list[Trade] = []
        count: dict[str, int] = {}
        seen: set[tuple] = set()
        for t in screened:
            key = (t.partner.team_id, tuple(sorted(p.player_id for p in t.give)),
                   tuple(sorted(p.player_id for p in t.get)))
            if key in seen or count.get(t.partner.team_id, 0) >= PRICE_PER_TEAM:
                continue
            seen.add(key)
            count[t.partner.team_id] = count.get(t.partner.team_id, 0) + 1
            chosen.append(t)
            if len(chosen) >= PRICE_TOP:
                break
        priced = [self.price(t) for t in chosen]
        good = [t for t in priced if t.gain >= max(MIN_GAIN, 2.0 * t.noise)]
        for t in good:
            t.verdict = "win-win" if t.their_gain >= 0 else "offer"
            t.reasons = self._reasons(t)
        good.sort(key=lambda t: -t.gain)
        out, per, used = [], {}, set()
        for t in good:
            # One version of each deal: a 2-for-1 that only adds a throw-in
            # to a better 1-for-1 for the same player is left out.
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

    def _reasons(self, t: Trade) -> list[str]:
        bits = [f"your lineup {t.my_lineup:+.1f} pts a game, theirs {t.their_lineup:+.1f} "
                f"(roster value with depth: you {t.my_value:+.1f}, them {t.their_value:+.1f})"]
        if t.their_gain >= 0:
            bits.append(f"their title odds rise too ({t.their_base:.1%} to {t.their_new:.1%})")
        else:
            bits.append(f"their title odds {t.their_base:.1%} to {t.their_new:.1%}: they gain on "
                        "projections, you gain more in the title race")
        if t.their_drop is not None:
            bits.append(f"they would drop {t.their_drop.name} to make room")
        if len(t.get) < len(t.give):
            bits.append("opens a roster spot for a pickup")
        for p in t.get:
            bits.extend(f"{p.name}: {s}" for s in p.signals[:1])
        return bits
