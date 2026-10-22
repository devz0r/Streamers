"""The league's most valuable players, for the rest of the season.

Ranked on value over replacement: the points a player is simulated to score
above a waiver pickup at his position, summed over every remaining week he
plays (this week through week 17). The weeks he misses -- a bye, an injury,
a reported timeline -- count nothing: you would play the pickup either way.
The simulation is the one that prices the title odds (:class:`SeasonModel`),
so injuries, timelines, byes and the next man up are already in it.

Replacement is :func:`season.replacement_levels`: the second-best free agent's
season value a game at each position (the best for D/ST and K, streamed).
Points over a pickup make positions comparable on one list, where raw points
would put every quarterback first. Upside is the same value in his best
tenth of simulated seasons (the 90th percentile): a breakout, a bigger role,
health. Also carried for sorting: season value a
game, this week's projection, the 10th-90th percentile of points a week, and
what the rest of the league sees him worth (the trade model's market view).
"""

from __future__ import annotations

import numpy as np

from ..league.model import LeagueSnapshot
from .season import replacement_levels

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


def build(snapshot: LeagueSnapshot, model, seen: dict | None = None) -> dict | None:
    """Every simulated player, rostered or free, as rows for the page."""
    futures = getattr(model, "futures", None)
    if futures is None:
        return None
    repl = replacement_levels(snapshot)
    weeks = np.array(model.weeks)
    keep = weeks >= int(snapshot.week)
    if not keep.any():
        return None
    owner = {p.player_id: t for t in snapshot.teams for p in t.roster}
    free = {p.player_id: p for p in snapshot.free_agents}
    mine = snapshot.my_team.team_id if snapshot.my_team else None
    rows = []
    for pid, j in model.pid.items():
        team = owner.get(pid)
        p = next((q for q in team.roster if q.player_id == pid), None) if team else free.get(pid)
        if p is None or p.position not in POSITIONS:
            continue
        scores = futures.scores[:, j, :][:, keep]
        played = futures.played[:, j, :][:, keep] if futures.played is not None else scores > 0
        level = float(repl.get(p.position, 0.0))
        vor = (scores - level * played).sum(axis=1)
        per_week = scores.mean(axis=1)
        s = (seen or {}).get(pid)
        rows.append({
            "name": p.name, "pos": p.position, "team": p.team or "", "status": p.status or "",
            "owner": team.name if team else "FA", "mine": bool(team and team.team_id == mine),
            "vor": round(float(vor.mean()), 1),
            "up": round(float(np.quantile(vor, 0.9)), 1),
            "ros": round(float(p.ros_value if p.ros_value is not None else (p.projection or 0.0)), 1),
            "proj": round(float(p.week_value), 1),
            "games": round(float(played.sum(axis=1).mean()), 1),
            "lo": round(float(np.quantile(per_week, 0.1)), 1), "hi": round(float(np.quantile(per_week, 0.9)), 1),
            "mkt": round(float(s.value), 1) if s is not None else None,
        })
    if not rows:
        return None
    rows.sort(key=lambda r: -r["vor"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {"rows": rows, "repl": {k: round(v, 1) for k, v in repl.items()},
            "weeks": [int(weeks[keep][0]), int(weeks[keep][-1])]}
