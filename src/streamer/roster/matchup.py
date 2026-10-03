"""The weekly matchup report: where you stand and what would move it."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config, get_config
from ..league.model import LeagueSnapshot, PlayerRow
from .lineup import (
    Optimisation,
    PickupOption,
    StreamOption,
    optimise,
    pickup_options,
    stream_options,
)


@dataclass
class MatchupReport:
    snapshot: LeagueSnapshot
    optimisation: Optimisation
    opponent_name: str = ""
    notes: list[str] = field(default_factory=list)
    #: D/ST and kicker options ranked by this week's P(win) (position -> list).
    streams: dict[str, list[StreamOption]] = field(default_factory=dict)
    #: Free agents who raise this week's P(win), best first.
    pickups: list[PickupOption] = field(default_factory=list)
    #: Rest-of-season odds for every team (None until the league's schedule
    #: and playoff format have been read).
    season: object | None = None
    #: Waiver moves valued by title odds (None when the league structure
    #: has not been read, and the roster-value engine is used instead).
    title_moves: list | None = None
    #: Upside plays priced but not worth a move: (player, title-odds gain, why priced).
    passed: list | None = None
    waiver_rank: int | None = None
    #: Trades that raise your title odds and pass the other manager's bar
    #: What each remaining game is worth (a :class:`Stakes`; None when the
    #: league structure has not been read).
    stakes: object | None = None
    #: How the projections have done, graded on results (a Scorecard).
    scorecard: object | None = None
    #: Waiver plans -- several moves priced together (None when the league
    #: structure has not been read).
    plans: list | None = None
    #: Trade views (a :class:`TradeBoard`; None when the league structure
    #: has not been read).
    trades: object | None = None
    trade_eval: dict | None = None
    #: Your players valued for the rest of the season (:mod:`roster_view`).
    roster_values: list | None = None

    @property
    def win_probability(self) -> float:
        return self.optimisation.best_win.win_probability

    @property
    def current_win_probability(self) -> float | None:
        cur = self.optimisation.current
        return cur.win_probability if cur else None

    def verdict(self) -> str:
        p = self.win_probability
        if p >= 0.65:
            return "favoured"
        if p >= 0.55:
            return "slight favourite"
        if p > 0.45:
            return "coin flip"
        if p > 0.35:
            return "slight underdog"
        return "underdog"

    def swing_players(self, n: int = 3) -> list[PlayerRow]:
        """Starters whose variance most decides the week."""
        starters = [p for _s, p in self.optimisation.best_win.flat()]
        return sorted(starters, key=lambda p: -(p.projection_sd or 0.0))[:n]


def build_report(snapshot: LeagueSnapshot, cfg: Config | None = None) -> MatchupReport:
    """Optimise the user's lineup against this week's opponent."""
    cfg = cfg or get_config()
    me = snapshot.my_team
    opp = snapshot.opponent
    notes: list[str] = []
    if opp is None:
        notes.append("no matchup found in the snapshot; optimising for expected points only")
    result = optimise(
        me.roster, snapshot.starting_slots,
        opp.roster if opp else None,
        n_sims=int(cfg.raw["roster"]["sims"]),
    )
    if result.current is None:
        notes.append("no lineup is currently set on the platform")
    from .waivers import ir_notes

    notes.extend(ir_notes(snapshot))
    streams: dict[str, list[StreamOption]] = {}
    for pos in ("DST", "K"):
        pool = [p for p in snapshot.free_agents if p.position == pos and p.projection is not None]
        pool += [p for p in me.roster if p.position == pos and not p.in_ir_slot]
        pool = sorted(pool, key=lambda p: -(p.projection or 0.0))[:12]
        try:
            streams[pos] = stream_options(result, pos, pool, n_sims=int(cfg.raw["roster"]["sims"]))
        except Exception:  # noqa: BLE001 - a bonus table never blocks the report
            streams[pos] = []
    try:
        pickups = this_week_pickups(snapshot, result, n_sims=int(cfg.raw["roster"]["sims"]))
    except Exception:  # noqa: BLE001 - a bonus table never blocks the report
        pickups = []
    return MatchupReport(
        snapshot=snapshot, optimisation=result,
        opponent_name=opp.name if opp else "", notes=notes, streams=streams, pickups=pickups,
    )


#: Free agents tried per position: the best projections this week that come
#: within ``PICKUP_REACH`` points of a starter they could replace.
PICKUPS_PER_POSITION = 6
PICKUP_REACH = 3.0
PICKUPS_SHOWN = 6


def this_week_pickups(snapshot: LeagueSnapshot, opt: Optimisation, n_sims: int = 20000) -> list[PickupOption]:
    """Free agents who would raise your P(win) this week, each with the
    player he would start over and the cheapest one to drop.

    The drop is the bench player with the least rest-of-season value (as the
    waiver engine ranks them) who would not start this week, never one whose
    game has kicked off, never one that leaves a position short; the season
    engine re-prices the move with its own pick of drops
    (:func:`price_pickups`)."""
    from ..league.model import SLOT_ELIGIBILITY
    from .title_moves import _droppable

    me = snapshot.my_team
    pool = []
    for pos in ("QB", "RB", "WR", "TE"):
        # The weakest starter he could replace sets the bar he has to come near.
        bar = [p.week_value for slot, p in opt.best_win.flat() if pos in SLOT_ELIGIBILITY.get(slot, (slot,))]
        floor = min(bar) - PICKUP_REACH if bar else 0.0
        at = sorted((p for p in snapshot.free_agents
                     if p.position == pos and p.team and not p.on_bye and not p.is_out and not p.locked
                     and p.projection is not None and p.week_value >= floor),
                    key=lambda p: -p.week_value)
        pool.extend(at[:PICKUPS_PER_POSITION])
    options = pickup_options(opt, me.roster, snapshot.starting_slots, pool, n_sims=n_sims)
    out = []
    for o in options:
        if o.gain <= 0:
            continue
        starting = {p.player_id for _s, p in o.lineup.flat()}
        drops = [p for p in _droppable(me.roster, o.player) if p.player_id not in starting and not p.locked]
        o.drop = drops[0] if drops else None
        out.append(o)
    return out[:PICKUPS_SHOWN]


def price_pickups(report: MatchupReport, engine) -> None:
    """Title odds of each pickup as a whole move -- this week and the rest of
    the season, the drop included -- on the season engine (which must have
    simulated the pickups: ``TitleEngine(..., extra=...)``). Of the few
    drops cheapest to your title who would not start this week, the one that
    keeps the most title odds is taken."""
    import numpy as np

    from .title_moves import _droppable

    me = report.snapshot.my_team
    worth = engine.worth_mean() if report.pickups else {}
    for o in report.pickups:
        if o.player.player_id not in engine.model.pid:
            continue
        starting = {p.player_id for _s, p in o.lineup.flat()}
        drops = [p for p in _droppable(me.roster, o.player, worth)
                 if p.player_id not in starting and not p.locked][: engine.n_drops]
        best = None
        for d in drops:
            won = engine.value_now(o.player, d, per_sim=True)
            if best is None or won.mean() > best[1].mean():
                best = (d, won)
        if best is None:
            continue
        o.drop = best[0]
        o.title_gain = float(best[1].mean()) - engine.base
        o.title_noise = float(np.std(best[1] - engine.base_won) / np.sqrt(len(best[1])))
