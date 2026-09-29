"""Where our rest-of-season projection should defer to the market.

Our projection moves fast on opportunity (a 2.5-game half-life), and the
market -- a player's name value from last season plus this season so far --
moves slowly. Tested on 2022-2025 (weeks 3-13, 8,000 player-weeks, rest-of-
season points a game), our projection beat the name-value view overall and
on big disagreements (3.9 against 5.8 average miss at 3.5+ points apart), so
by default it stands. Two early-season cases were the exception, out of
sample as well as in:

* **a receiver we mark up in weeks 3-6** -- a few games of targets; the truth
  landed about a quarter of the way from the market to us;
* **a quarterback we mark down in weeks 3-6** -- a slow start; the market
  was closer.

Only those cells are adjusted: the projection becomes
``market + k * (ours - market)``. ``scripts/fit_market_calibration.py``
refits them into ``market_calibration.json``, keeping a cell only when it
beats our projection on seasons it was not fitted on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MODEL_PATH = Path(__file__).with_name("market_calibration.json")

#: How the name-value view is built -- the same measured proportions as the
#: perception model, without the platform projection (not kept historically).
REP_WEIGHT, SEASON_WEIGHT = 0.46, 0.13
REP_PRIOR, SEASON_PRIOR, PREV_WEIGHT = 4.0, 2.0, 0.5


def load() -> dict:
    try:
        return json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"cells": []}


def name_value(ours: float, last: tuple[float, int] | None, prev: tuple[float, int] | None,
               now: tuple[float, int] | None) -> float:
    """Reputation (last season, the one before at half weight) and this season
    so far, each shrunk toward our projection for thin samples."""
    gl = last[1] if last else 0
    gp = (prev[1] if prev else 0) * PREV_WEIGHT
    rep = ((last[0] * gl if last else 0.0) + (prev[0] * gp if prev else 0.0) + REP_PRIOR * ours) \
        / (gl + gp + REP_PRIOR)
    gn = now[1] if now else 0
    cur = ((now[0] * gn if now else 0.0) + SEASON_PRIOR * ours) / (gn + SEASON_PRIOR)
    return (REP_WEIGHT * rep + SEASON_WEIGHT * cur) / (REP_WEIGHT + SEASON_WEIGHT)


def season_lines(history: pd.DataFrame | None, season: int, week: int) -> dict[tuple[str, int], tuple[float, int]]:
    """(nflverse id, season) -> (points a game, games): the two seasons before,
    and this one up to (not including) ``week``."""
    if history is None or history.empty:
        return {}
    h = history[history["season"].between(season - 2, season)]
    h = h[(h["season"] < season) | (h["week"] < week)]
    g = h.groupby(["player_id", "season"])["fantasy_points_ppr"].agg(["mean", "count"])
    return {(str(pid), int(s)): (float(r["mean"]), int(r["count"])) for (pid, s), r in g.iterrows()}


@dataclass
class Adjustment:
    value: float
    k: float
    market: float
    note: str


def calibrate(position: str, week: int, ours: float, nfl_id: str | None, season: int,
              lines: dict, conf: dict | None = None) -> Adjustment | None:
    """The adjusted rest-of-season value, or None when no measured cell applies."""
    if not nfl_id:
        return None
    last = lines.get((str(nfl_id), season - 1))
    if last is None or last[1] < 4:
        return None                       # no name to have a market value
    market = name_value(ours, last, lines.get((str(nfl_id), season - 2)), lines.get((str(nfl_id), season)))
    sign = "up" if ours >= market else "down"
    for cell in (conf or load()).get("cells", []):
        lo, hi = cell["weeks"]
        if cell["position"] == position and cell["sign"] == sign and lo <= week <= hi:
            k = float(cell["k"])
            return Adjustment(value=market + k * (ours - market), k=k, market=market,
                              note=cell.get("note", ""))
    return None
