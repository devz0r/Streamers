"""Sleeper's weekly projection on every skill player it covers.

One more second forecast for the week, blended last
(:func:`streamer.roster.vegas.blend_market`). Matching is by the nflverse id
where Sleeper has one, otherwise by name, team and position; DECISIONS.md, "Sleeper projections".
"""

from __future__ import annotations

import logging

from ..config import Config
from ..data import sleeper
from ..league.model import LeagueSnapshot

log = logging.getLogger(__name__)
SKILL = ("QB", "RB", "WR", "TE")


def attach(snapshot: LeagueSnapshot, cfg: Config, allow_network: bool = True) -> int:
    """Set ``sleeper_projection`` on every skill player with one, matched by
    nflverse id or else by name, team and position. Returns how many matched
    (0 with no data)."""
    rows = sleeper.projection_rows(cfg, int(snapshot.season), int(snapshot.week), allow_network=allow_network)
    if not rows:
        return 0
    skill = [p for p in snapshot.all_players() if p.position in SKILL]
    got = sleeper.match(rows, ((id(p), p.nfl_id, p.name, p.team, p.position) for p in skill))
    for p in skill:
        if id(p) in got:
            p.sleeper_projection = got[id(p)]
    log.info("Sleeper projections matched %d of %d skill players", len(got), len(skill))
    return len(got)
