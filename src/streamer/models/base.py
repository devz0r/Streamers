"""Shared estimator plumbing: design matrices, sample weights, residual bands.

The two position models differ in structure but share everything here, so the
walk-forward backtest, the weekly refit and the live projection all go through
one code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from ..config import Config, get_config


@dataclass
class DesignMatrix:
    """Standardised features plus the statistics needed to reapply them."""

    columns: list[str]
    means: np.ndarray
    scales: np.ndarray
    multipliers: np.ndarray

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = _numeric_block(frame, self.columns)
        return (raw - self.means) / self.scales * self.multipliers


def _numeric_block(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Columns as a float matrix, tolerating missing ones.

    A column absent from ``frame`` becomes a zero column rather than an error:
    after standardisation zero is the training mean, so an unavailable factor
    contributes nothing instead of taking the whole projection down.
    """
    if not columns:
        return np.zeros((len(frame), 0))
    blocks = []
    for column in columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        else:
            values = np.zeros(len(frame), dtype=float)
        blocks.append(values)
    return np.nan_to_num(np.column_stack(blocks), nan=0.0, posinf=0.0, neginf=0.0)


def fit_design(
    frame: pd.DataFrame, columns: list[str], multipliers: dict[str, float] | None = None
) -> DesignMatrix:
    """Standardise ``columns`` and bake in per-factor ledger multipliers."""
    raw = _numeric_block(frame, columns)
    means = raw.mean(axis=0)
    scales = raw.std(axis=0)
    scales[scales < 1e-8] = 1.0
    mult = np.array([float((multipliers or {}).get(c, 1.0)) for c in columns])
    return DesignMatrix(columns=list(columns), means=means, scales=scales, multipliers=mult)


def sample_weights(
    frame: pd.DataFrame, current_season: int, cfg: Config | None = None
) -> np.ndarray:
    """Recency weights: current season up-weighted, older seasons decayed.

    ``model.current_season_weight`` (default 2x) is the required "current-season
    games count double" rule; ``model.season_decay`` fades older seasons on top
    of it so 2021 does not out-vote 2025.
    """
    cfg = cfg or get_config()
    conf = cfg.model
    seasons = pd.to_numeric(frame["season"], errors="coerce").fillna(current_season).to_numpy()
    age = np.maximum(0, current_season - seasons)
    weights = np.power(float(conf["season_decay"]), age)
    weights = np.where(seasons >= current_season, weights * float(conf["current_season_weight"]), weights)
    return weights


def ridge_alpha(cfg: Config, position: str | None = None) -> float:
    """Ridge penalty for ``position``.

    A value written by ``streamer backtest --tune --save`` wins over the
    ``config.yaml`` seed, so a re-tune takes effect without editing config.
    """
    conf = cfg.model
    if position:
        try:
            from ..pipeline import tuned_alpha_for

            tuned = tuned_alpha_for(position, cfg)
        except Exception:  # noqa: BLE001 - a missing/garbled file is not fatal
            tuned = None
        if tuned is not None:
            return float(tuned)
        by_position = conf.get("ridge_alpha_by_position") or {}
        if position in by_position:
            return float(by_position[position])
    return float(conf["ridge_alpha"])


def make_estimator(kind: str, cfg: Config | None = None, position: str | None = None,
                   alpha: float | None = None):
    """Instantiate one of the configured candidate estimators."""
    cfg = cfg or get_config()
    conf = cfg.model
    if kind == "ridge":
        return Ridge(alpha=float(alpha) if alpha is not None else ridge_alpha(cfg, position))
    if kind == "gbm":
        g = conf["gbm"]
        return HistGradientBoostingRegressor(
            max_iter=int(g["max_iter"]),
            learning_rate=float(g["learning_rate"]),
            max_depth=int(g["max_depth"]),
            min_samples_leaf=int(g["min_samples_leaf"]),
            l2_regularization=float(g["l2_regularization"]),
            random_state=17,
        )
    raise ValueError(f"unknown estimator: {kind!r}")


@dataclass
class ResidualBands:
    """Empirical predictive spread, conditioned on the prediction level.

    Fantasy scoring is heteroscedastic -- a projected 11-point DST has a much
    wider realistic range than a projected 4-point one -- so residuals are
    bucketed by predicted value rather than pooled.
    """

    edges: np.ndarray
    residuals: list[np.ndarray]
    pooled: np.ndarray = field(default_factory=lambda: np.zeros(1))

    @classmethod
    def fit(cls, predictions: np.ndarray, actuals: np.ndarray, n_buckets: int = 4) -> ResidualBands:
        pred = np.asarray(predictions, dtype=float)
        act = np.asarray(actuals, dtype=float)
        mask = np.isfinite(pred) & np.isfinite(act)
        pred, act = pred[mask], act[mask]
        if len(pred) < 40:
            resid = act - pred if len(pred) else np.zeros(1)
            return cls(edges=np.array([-np.inf, np.inf]), residuals=[resid], pooled=resid)
        qs = np.linspace(0, 1, n_buckets + 1)[1:-1]
        edges = np.concatenate([[-np.inf], np.quantile(pred, qs), [np.inf]])
        buckets = []
        for i in range(len(edges) - 1):
            sel = (pred >= edges[i]) & (pred < edges[i + 1])
            resid = act[sel] - pred[sel]
            buckets.append(resid if len(resid) >= 10 else act - pred)
        return cls(edges=edges, residuals=buckets, pooled=act - pred)

    def _bucket(self, value: float) -> np.ndarray:
        idx = int(np.searchsorted(self.edges, value, side="right") - 1)
        idx = max(0, min(idx, len(self.residuals) - 1))
        resid = self.residuals[idx]
        return resid if len(resid) else self.pooled

    def quantile(self, predictions: np.ndarray, q: float) -> np.ndarray:
        return np.array([p + np.quantile(self._bucket(p), q) for p in np.asarray(predictions, float)])

    def sample(self, predictions: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
        """``(draws, n)`` matrix of simulated outcomes."""
        preds = np.asarray(predictions, float)
        out = np.empty((draws, len(preds)))
        for j, p in enumerate(preds):
            resid = self._bucket(p)
            out[:, j] = p + rng.choice(resid, size=draws, replace=True)
        return out


def top_n_probability(
    predictions: np.ndarray,
    bands: ResidualBands,
    cutoff: int,
    draws: int = 4000,
    seed: int = 11,
) -> np.ndarray:
    """P(finishing inside the top ``cutoff``) for each unit on a slate."""
    preds = np.asarray(predictions, float)
    if len(preds) == 0:
        return preds
    rng = np.random.default_rng(seed)
    sims = bands.sample(preds, draws, rng)
    # Rank descending within each simulated week; ties broken by draw order.
    order = np.argsort(-sims, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(sims.shape[0])[:, None]
    ranks[rows, order] = np.arange(sims.shape[1])[None, :]
    return (ranks < cutoff).mean(axis=0)


def spearman(a, b) -> float:
    """Spearman rank correlation, NaN-safe, returning ``nan`` for degenerate input."""
    a = pd.Series(np.asarray(a, dtype=float))
    b = pd.Series(np.asarray(b, dtype=float))
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return float("nan")
    a, b = a[mask], b[mask]
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))
