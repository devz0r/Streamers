"""Attach sportsbook-implied points to a roster, and start the lineup they imply.

Two projections disagreeing is information. Ours leans on trailing production
and opportunity; the market's leans on everything a book knows, including
beat-reporter noise about a snap count that has not shown up in a box score
yet. Where they disagree the market is usually, though not always, right --
so the panel shows both and says which players the disagreement is about.

Props do not cover everybody. Kickers and defences are rarely posted, deep
bench players never are, and a player the books have not priced simply has no
Vegas number -- which is shown as a blank rather than a zero, because "no
market" is not "no points".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import Config, get_config
from ..league.model import LeagueSnapshot, PlayerRow
from .players import normalize_name

log = logging.getLogger(__name__)


@dataclass
class VegasReport:
    """What the props pull produced, for the page and the CLI."""

    matched: int = 0
    unmatched: list[str] = field(default_factory=list)
    events: int = 0
    credits_remaining: int | None = None
    warnings: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def is_usable(self) -> bool:
        return self.matched > 0


def teams_on(snapshot: LeagueSnapshot, include_opponent: bool = False) -> set[str]:
    """NFL teams worth paying for props on.

    Only players who could start for *you*: props are billed per market per
    event, the opponent's implied points are never shown, and P(win) scores
    both sides with our own projections. Players already ruled out for the
    week cost credits and change nothing.
    """
    rows = [p for p in snapshot.my_team.roster
            if not p.in_ir_slot and not p.on_bye and not p.is_out]
    if include_opponent and snapshot.opponent is not None:
        rows += list(snapshot.opponent.roster)
    return {p.team for p in rows if p.team}


def attach(
    snapshot: LeagueSnapshot,
    cfg: Config | None = None,
    allow_network: bool = True,
    payload=None,
) -> VegasReport:
    """Fetch props for the matchup's teams and attach them to the players.

    ``payload`` injects a payload instead of calling out, for tests.
    """
    from ..data.props import fetch_props, props_to_points

    cfg = cfg or get_config()
    report = VegasReport()
    if payload is not None:
        points = props_to_points(payload, cfg)
    elif not allow_network:
        report.warnings.append("player props skipped: --offline")
        return report
    else:
        result = fetch_props(cfg, teams=teams_on(snapshot))
        report.events = result.events
        report.credits_remaining = result.credits_remaining
        report.warnings.extend(result.warnings)
        report.description = result.describe()
        snapshot._vegas_credits = result.credits_remaining
        if not result.is_usable:
            return report
        from ..data.props import consensus, implied_means, to_fantasy_points

        points = to_fantasy_points(consensus(implied_means(result.frame, cfg), cfg), cfg)

    if points.empty:
        return report
    by_name = {normalize_name(r.player): r for r in points.itertuples()}
    everyone = list(snapshot.all_players())
    for p in everyone:
        row = by_name.get(normalize_name(p.name))
        if row is None:
            continue
        p.vegas_points = float(row.vegas_points)
        p.vegas_stats = dict(row.stats)
        p.vegas_books = int(row.books)
        report.matched += 1
    report.unmatched = sorted(
        {p.name for p in snapshot.my_team.roster
         if p.vegas_points is None and p.position not in ("K", "DST") and not p.on_bye}
    )
    return report


def vegas_lineup(snapshot: LeagueSnapshot, cfg: Config | None = None):
    """The lineup the market's numbers would start, and what it projects.

    Falls back to our own projection for players with no posted prop, because
    a lineup that benches every kicker is not a lineup.
    """
    from .lineup import best_by_key

    cfg = cfg or get_config()
    roster = [p for p in snapshot.my_team.roster if not p.in_ir_slot]

    def key(p: PlayerRow) -> float:
        if p.is_out:
            return 0.0
        if p.vegas_points is not None:
            return float(p.vegas_points)
        return float(p.projection or 0.0)

    return best_by_key(roster, snapshot.starting_slots, key)


def disagreements(snapshot: LeagueSnapshot, n: int = 3) -> list[tuple[PlayerRow, float]]:
    """Players our projection and the market disagree about most, biggest first."""
    gaps = []
    for p in snapshot.my_team.roster:
        if p.vegas_points is None or p.projection is None or p.in_ir_slot or p.is_out:
            continue
        gaps.append((p, float(p.vegas_points) - float(p.projection)))
    gaps.sort(key=lambda g: -abs(g[1]))
    return gaps[:n]
