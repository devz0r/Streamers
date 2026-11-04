"""A defence's and a kicker's level for the rest of the season.

The rankings' ``baseline_points`` is a Vegas-only projection of *this week's*
game; used as the season value it carried this week's matchup into every
later week (a defence facing a weak offence read as elite all season). The
level is now a blend of that line and the unit's points so far this season,
shrunk toward the league's average unit:

    level = line * baseline + (1 - line) * (points so far + shrink * mean) / (games + shrink)

Fitted on two of 2023-2025 and scored on the third (rest-of-season points a
game from checkpoints at weeks 3-10, ``scripts/fit_unit_level.py``): D/ST
line 0.2, shrink 48 -- defences barely differ over a season -- and K line
0.7, shrink 8 -- a kicker's team total is a lasting trait of his offence.
DECISIONS.md, "A season level for defences and kickers".
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)

DEFAULTS = {"DST": {"line": 0.2, "shrink": 48.0}, "K": {"line": 0.7, "shrink": 8.0}}


def conf(cfg: Config) -> dict:
    raw = (cfg.raw.get("roster") or {}).get("unit_level") or {}
    return {pos: {**DEFAULTS[pos], **(raw.get(pos) or {})} for pos in DEFAULTS}


def season_points(cfg: Config, season: int, week: int) -> dict[str, tuple[dict[str, tuple[float, int]], float]]:
    """Position -> ({team: (points so far, games)}, league mean a game), from
    the scored rankings (``results/<league>/history.parquet``, on this
    league's sheet), weeks before ``week``."""
    from ..calibrate import history_path

    path = history_path(cfg)
    if not path.exists():
        return {}
    h = pd.read_parquet(path, columns=["season", "week", "position", "team", "actual_points"])
    h = h[(h["season"] == season) & (h["week"] < week) & h["actual_points"].notna()]
    out = {}
    for pos, g in h.groupby("position"):
        if pos not in DEFAULTS or g.empty:
            continue
        g = g.drop_duplicates(["week", "team"])
        agg = g.groupby("team")["actual_points"].agg(["sum", "count"])
        out[pos] = ({t: (float(r["sum"]), int(r["count"])) for t, r in agg.iterrows()},
                    float(g["actual_points"].mean()))
    return out


def level(position: str, team: str, baseline: float, points: dict, cfg_level: dict) -> float:
    """The unit's season level a game (``baseline`` alone when nothing is scored yet)."""
    got = points.get(position)
    if not got:
        return baseline
    by_team, mean = got
    total, n = by_team.get(team, (0.0, 0))
    c = cfg_level[position]
    shrunk = (total + c["shrink"] * mean) / (n + c["shrink"])
    return c["line"] * baseline + (1.0 - c["line"]) * shrunk
