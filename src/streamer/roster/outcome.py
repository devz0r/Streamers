"""The shape of a player's week, and how players move together.

Three things measured from 2021-2025 history and stored in
``outcome_model.json`` (refit with ``scripts/fit_outcome_model.py``):

* **shape** -- the distribution of (actual - projection) / sd by position and
  projection level. Fantasy scores are skewed: a low projection has a hard
  floor near zero and a long right tail, which a normal curve gets wrong in
  exactly the place an underdog cares about.
* **correlation** -- between the residuals of players in the same game, by
  role: a quarterback and his WR1 (+0.34), a quarterback and the defence
  facing him (-0.44), the two quarterbacks in a shootout (+0.18). Everything
  else is close to zero and treated as zero.
* **drift** -- how far a player's per-game projection moves over the next
  four games (his role firming up or falling away), by position and level,
  wider for rookies and second-year players. Waivers price upside from it.

The lineup simulator draws correlated normals, maps them through the shape,
and scales by each player's projection and spread (a Gaussian copula).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).with_name("outcome_model.json")

#: Projection levels the shape and drift tables are keyed on (same edges as
#: the sd calibration in :mod:`streamer.roster.projections`).
LEVELS = (0.0, 5.0, 8.0, 12.0, 16.0, 20.0, 99.0)

#: Roles the correlation table knows. Deeper depth-chart players borrow the
#: nearest one (an RB3 behaves like an RB2).
ROLES = ("QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "K", "DST")


def level(mean: float) -> int:
    return int(np.digitize([float(mean)], LEVELS[1:-1])[0])


@lru_cache(maxsize=1)
def load() -> dict:
    try:
        return json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def canonical_role(role: str) -> str:
    """Map a depth-chart role onto one the correlation table knows."""
    if role in ROLES:
        return role
    if role.startswith("RB"):
        return "RB2"
    if role.startswith("WR"):
        return "WR3"
    if role.startswith("TE"):
        return "TE"
    if role.startswith("QB"):
        return "QB"
    return ""


def correlation(same_team: bool, role_a: str, role_b: str) -> float:
    """Residual correlation between two players in the same game."""
    a, b = canonical_role(role_a), canonical_role(role_b)
    if not a or not b:
        return 0.0
    table = load().get("correlation", {}).get("same" if same_team else "opp", {})
    return float(table.get(f"{a}|{b}", table.get(f"{b}|{a}", 0.0)))


def shape(position: str, mean: float) -> tuple[np.ndarray, np.ndarray]:
    """``(probabilities, standardised quantiles)`` for a player's week.

    Falls back to the normal curve when the table has nothing for the
    position, so an absent model file degrades to the old behaviour.
    """
    shapes = load().get("shape", {})
    key = position if position in ("DST", "K") else f"{position}|{level(mean)}"
    entry = shapes.get(key) or shapes.get(f"{position}|all")
    if entry:
        return np.asarray(entry["p"], dtype=float), np.asarray(entry["z"], dtype=float)
    from scipy.stats import norm

    p = np.linspace(0.005, 0.995, 41)
    return p, norm.ppf(p)


def drift_sd(position: str, mean: float, experience: int | None) -> float:
    """Spread of the change in his per-game projection over the next month."""
    drift = load().get("drift", {})
    base = drift.get("sd", {}).get(f"{position}|{level(mean)}")
    if base is None:
        base = drift.get("sd", {}).get(f"{position}|all", 1.8)
    mult = drift.get("youth_multiplier", {})
    if experience == 0:
        base *= float(mult.get("rookie", 1.0))
    elif experience == 1:
        base *= float(mult.get("second", 1.0))
    return float(base)


def youth_drift(experience: int | None) -> float:
    """Average rise in a young player's per-game projection over a month,
    relative to a veteran's -- roles grow for rookies."""
    means = load().get("drift", {}).get("youth_mean", {})
    if experience == 0:
        return float(means.get("rookie", 0.0))
    if experience == 1:
        return float(means.get("second", 0.0))
    return 0.0
