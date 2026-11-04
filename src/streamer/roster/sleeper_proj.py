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
    people = [(id(p), p.nfl_id, p.name, p.team, p.position) for p in skill]
    got = sleeper.match(rows, people)
    # Sleeper's points are on the standard sheet; the league's may differ
    # (Yahoo: 6 a touchdown pass, -1 an interception).
    from ..scoring import rescore_feed

    sheet = cfg._profile.get("offense_scoring")
    stats = sleeper.match(rows, people, field="stats")
    dropped = 0
    for p in skill:
        if id(p) in got:
            pts = rescore_feed(got[id(p)], stats.get(id(p)), sheet, p.position)
            dropped += pts is None
            p.sleeper_projection = pts
    if dropped:
        log.info("Sleeper projections left out for %d players whose stats could not be rescored", dropped)
    log.info("Sleeper projections matched %d of %d skill players", len(got), len(skill))
    return len(got)
