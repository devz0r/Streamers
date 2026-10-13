"""Sleeper's weekly projection on every skill player it covers.

One more second forecast for the week, blended last
(:func:`streamer.roster.vegas.blend_market`). Matching is by the nflverse id
the platform rows already carry; DECISIONS.md, "Sleeper projections".
"""

from __future__ import annotations

import logging

from ..config import Config
from ..data import sleeper
from ..league.model import LeagueSnapshot

log = logging.getLogger(__name__)
SKILL = ("QB", "RB", "WR", "TE")


def attach(snapshot: LeagueSnapshot, cfg: Config, allow_network: bool = True) -> int:
    """Set ``sleeper_projection`` on every skill player with one. Returns how
    many matched (0 with no data)."""
    proj = sleeper.projections(cfg, int(snapshot.season), int(snapshot.week), allow_network=allow_network)
    if not proj:
        return 0
    n = 0
    for p in snapshot.all_players():
        if p.position in SKILL and p.nfl_id and str(p.nfl_id) in proj:
            p.sleeper_projection = proj[str(p.nfl_id)]
            n += 1
    log.info("Sleeper projections matched %d players", n)
    return n
