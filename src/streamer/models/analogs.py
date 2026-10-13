"""Similar past matchups: the mean score of a unit's nearest earlier games.

Subvertadown shows each kicker and defence the most similar historical
matchups and what they scored. As a forecast that is a nearest-neighbour
mean: standardise a few pre-game features over every completed game in the
pool, take the ``k`` closest to the game being projected, average what they
scored. For kickers, half that and half the model beat the model alone in
every season 2023-2026 (scripts/fit_analogs.py); for defences it lost in all
of them. Config ``model.analogs``; DECISIONS.md, "Similar past matchups".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def settings(cfg: Config, position: str) -> dict | None:
    """The analog settings for ``position``, or None when it does not use them."""
    conf = ((cfg.model.get("analogs") or {}).get(position)) or None
    if not conf or float(conf.get("weight", 0.0)) <= 0.0:
        return None
    return conf


def analog_mean(pool: pd.DataFrame, slate: pd.DataFrame, columns: list[str], k: int,
                target: str = "fantasy_points") -> np.ndarray:
    """Mean ``target`` of each slate row's ``k`` nearest pool rows (NaN where a
    slate row lacks a matching feature or the pool is too small)."""
    out = np.full(len(slate), np.nan)
    pool = pool[pool[target].notna()].dropna(subset=columns)
    if len(pool) < k or slate.empty:
        return out
    mu, sd = pool[columns].mean(), pool[columns].std().replace(0.0, 1.0).fillna(1.0)
    a = ((pool[columns] - mu) / sd).to_numpy(float)
    ok = slate[columns].notna().all(axis=1).to_numpy()
    if not ok.any():
        return out
    b = ((slate.loc[ok, columns] - mu) / sd).to_numpy(float)
    dist = ((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)
    nearest = np.argpartition(dist, k - 1, axis=1)[:, :k]
    out[ok] = pool[target].to_numpy(float)[nearest].mean(axis=1)
    return out


def blend(model_points: np.ndarray, analog: np.ndarray, weight: float) -> np.ndarray:
    """``(1 - weight) * model + weight * analog``; the model alone where no analog."""
    analog = np.asarray(analog, float)
    model_points = np.asarray(model_points, float)
    return np.where(np.isfinite(analog), (1.0 - weight) * model_points + weight * analog, model_points)
