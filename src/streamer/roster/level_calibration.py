"""The top of a back's or receiver's weekly number, brought back to what such numbers scored.

Over every player the model's weekly projection is calibrated: a point more
projected is a point more scored. Among the players a lineup actually chooses
between it is not. 2022-2025, walk-forward, starter-calibre backs and
receivers (the top 30 and 36 a week): a point more projected was 0.81 and
0.77 more scored. The curve bends: from about 12 points up each projected
point is worth about three quarters of one (a top-fifth back projected 18.9
scored 18.1), and below it a little more than one (9.7 scored 10.7). A
lineup, a claim or a trade leans on exactly those differences, so rosters
built on the numbers came out weaker than the numbers said, and the favourite
in a matchup won less often than drawn -- which the win chance was then
pulled toward 50% to cover.

This is that bend, measured and taken out at the source: per position, a line
with one knot at ``knot`` points,

    calibrated = a + b * projected + c * max(projected - knot, 0)

fitted on the model's if-he-plays number for players with no injury tag
(``scripts/fit_level_calibration.py``, each season held out in turn; the
numbers are in ``level_calibration.json``). Below 3 points, where nothing was
fitted, the number is scaled so the map stays continuous. Tight ends and
quarterbacks are left alone: a tight end's fit was not stable from season to
season, and a starting quarterback's differences hold up as projected.

It is applied to the model's own number before the platform's, the experts'
and the market's are blended in.

The season value bends the same way (a back valued at 22 a game scores about
20.8), and the season simulator reads every player's level through the same
kind of line fitted on season values (``season``): on replayed leagues that
took P(playoffs) and P(title) closer to what happened. Only the simulator
reads it; the season value shown, and what trades and claims are priced on,
is unchanged. DECISIONS.md, "The top of the weekly number".
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("level_calibration.json")
#: Below this the map was not fitted; the number is scaled to meet it.
FLOOR = 3.0
#: Set by the fitting script so it sees the projection (and the simulator the
#: season value) as it is before this step.
OFF = False
SEASON_OFF = False
#: The league being worked on reads season values through the line
#: (``season_levels`` in its profile, default on): set by :func:`use_profile`.
#: Replayed in Yahoo's format it did not hold (P(playoffs) Brier 0.18093 ->
#: 0.18129, better in 2 of 4 seasons), so Yahoo's is off.
_SEASON_ON = True


def use_profile(cfg) -> None:
    global _SEASON_ON
    prof = (getattr(cfg, "profiles", None) or {}).get(getattr(cfg, "profile", None)) or {}
    _SEASON_ON = bool(prof.get("season_levels", True))


@functools.lru_cache(maxsize=1)
def load() -> dict:
    try:
        return json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def line(coef: dict, knot: float, x: float) -> float:
    return float(coef["a"]) + float(coef["b"]) * x + float(coef["c"]) * max(x - knot, 0.0)


def apply(position: str, mean: float, conf: dict | None = None) -> float:
    """``mean`` (the model's if-he-plays weekly number) calibrated for ``position``;
    unchanged for a position with no fit."""
    if mean is None or mean <= 0 or (OFF and conf is None):
        return mean
    conf = load() if conf is None else conf
    coef = (conf.get("positions") or {}).get(position)
    if not coef:
        return mean
    return _mapped(coef, float(conf.get("knot", 12.0)), float(mean))


def _mapped(coef: dict | None, knot: float, value: float) -> float:
    if not coef or value <= 0:
        return value
    if value >= FLOOR:
        return max(line(coef, knot, float(value)), 0.0)
    return float(value) * max(line(coef, knot, FLOOR), 0.0) / FLOOR


def season(position: str, value: float, conf: dict | None = None) -> float:
    """A season value as the season simulator reads it."""
    if value is None or (conf is None and (SEASON_OFF or not _SEASON_ON)):
        return value
    conf = load() if conf is None else conf
    return _mapped((conf.get("season") or {}).get(position), float(conf.get("knot", 12.0)), float(value))


def season_level(p, conf: dict | None = None) -> float:
    """A player's level for the season simulator: his season value through
    :func:`season` (this week's projection, already calibrated, when he has none)."""
    if p.ros_value is None:
        return float(p.projection or 0.0)
    return season(p.position, float(p.ros_value), conf)
