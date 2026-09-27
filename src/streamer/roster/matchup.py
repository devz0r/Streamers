"""The weekly matchup report: where you stand and what would move it."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config, get_config
from ..league.model import LeagueSnapshot, PlayerRow
from .lineup import Optimisation, StreamOption, optimise, stream_options


@dataclass
class MatchupReport:
    snapshot: LeagueSnapshot
    optimisation: Optimisation
    opponent_name: str = ""
    notes: list[str] = field(default_factory=list)
    #: D/ST and kicker options ranked by this week's P(win) (position -> list).
    streams: dict[str, list[StreamOption]] = field(default_factory=dict)

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
    return MatchupReport(
        snapshot=snapshot, optimisation=result,
        opponent_name=opp.name if opp else "", notes=notes, streams=streams,
    )
