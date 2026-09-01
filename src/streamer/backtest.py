"""Strict walk-forward backtesting.

The rule is absolute: to project season *S* week *W*, the model may see only
games that finished before it -- every earlier week of *S*, plus every prior
season. Nothing is fitted on data from the week being predicted, and the
feature builder itself is leak-free (see :mod:`streamer.features.rolling`), so
the numbers here are what the tool would actually have produced live.

Every run reports, per season and per position:

* **MAE** against realised fantasy points
* **rank correlation** (Spearman), averaged over weeks -- the metric that
  matters for streaming, since you only need the ordering
* **top-5 hit rate**: how often the five recommended units finished top-12
* the same three for the naive Vegas-only baseline

The full model beating the baseline on rank correlation for *both* positions is
the project's definition of done.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, get_config
from .features.build import DST_FEATURES, KICKER_FEATURES
from .models.base import spearman
from .models.ledger import multipliers_for
from .models.positions import BASELINE_COLUMN, DstModel, KickerModel, VegasBaseline

log = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Per-row predictions plus the aggregate scorecard."""

    predictions: pd.DataFrame
    summary: pd.DataFrame
    weekly: pd.DataFrame = field(default_factory=pd.DataFrame)

    def beats_baseline(self) -> dict[str, bool]:
        out = {}
        for position, grp in self.summary.groupby("position"):
            model = float(grp["rank_corr"].mean())
            base = float(grp["baseline_rank_corr"].mean())
            out[str(position)] = bool(model > base)
        return out


def _model_for(position: str):
    return KickerModel if position == "K" else DstModel


def _features_for(position: str) -> list[str]:
    return list(KICKER_FEATURES if position == "K" else DST_FEATURES)


def walk_forward(
    features: pd.DataFrame,
    position: str,
    seasons: list[int],
    cfg: Config | None = None,
    kind: str = "ridge",
    two_stage: bool = False,
    min_train_rows: int | None = None,
    max_week: int | None = None,
    alpha: float | None = None,
    use_ledger: bool = True,
) -> pd.DataFrame:
    """Predict every week of ``seasons`` using only strictly earlier games.

    ``use_ledger`` recomputes the factor-correlation multipliers at every step,
    exactly as the weekly refit does, so the backtest measures the shipped
    model rather than a plain ridge.
    """
    cfg = cfg or get_config()
    min_train_rows = min_train_rows or int(cfg.model["min_train_rows"])
    max_week = max_week or int(cfg.season["max_regular_season_week"])
    target = "fantasy_points"
    model_cls = _model_for(position)
    baseline_col = BASELINE_COLUMN[position]

    played = features[features[target].notna()].copy()
    played = played.sort_values(["season", "week"]).reset_index(drop=True)
    out: list[pd.DataFrame] = []

    for season in sorted(seasons):
        weeks = sorted(played.loc[played["season"] == season, "week"].unique())
        for week in weeks:
            if week > max_week:
                continue
            train = played[
                (played["season"] < season)
                | ((played["season"] == season) & (played["week"] < week))
            ]
            if len(train) < min_train_rows:
                continue
            test = played[(played["season"] == season) & (played["week"] == week)]
            if test.empty:
                continue
            # The multiplier's signal prior must be measured against the
            # quantity the core regression actually fits: for the two-stage
            # D/ST model that is the big-play component, not the total, since
            # the tier half is handled separately.
            fit_target = (
                DstModel.BIG_PLAY_TARGET if (position == "DST" and two_stage) else target
            )
            multipliers = (
                multipliers_for(
                    train, _features_for(position), fit_target, position, season, week, cfg
                )
                if use_ledger
                else {}
            )
            try:
                if position == "DST":
                    model = model_cls.fit(
                        train, cfg, kind=kind, current_season=season,
                        multipliers=multipliers, two_stage=two_stage, alpha=alpha,
                        compute_weights=False,
                    )
                else:
                    model = model_cls.fit(
                        train, cfg, kind=kind, current_season=season,
                        multipliers=multipliers, alpha=alpha, compute_weights=False,
                    )
                projection = model.predict(test)
            except Exception as exc:  # noqa: BLE001 - one bad week must not kill a run
                log.warning("%s %s wk%s failed: %s", position, season, week, exc)
                continue

            baseline = VegasBaseline.fit(train, baseline_col, target)
            frame = projection.frame.copy()
            frame["baseline_points"] = baseline.predict(test)
            frame["position"] = position
            frame["estimator"] = kind
            out.append(frame)

    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def score_predictions(
    predictions: pd.DataFrame, cfg: Config | None = None, top_k: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse per-row predictions into weekly and per-season scorecards."""
    cfg = cfg or get_config()
    cutoff = cfg.startable_rank
    rows = []
    for (position, season, week), grp in predictions.groupby(["position", "season", "week"]):
        grp = grp.dropna(subset=["fantasy_points"])
        if len(grp) < 5:
            continue
        actual_rank = grp["fantasy_points"].rank(ascending=False, method="min")
        model_order = grp["expected_points"].rank(ascending=False, method="first")
        base_order = grp["baseline_points"].rank(ascending=False, method="first")
        rows.append(
            {
                "position": position,
                "season": int(season),
                "week": int(week),
                "n": len(grp),
                "mae": float((grp["expected_points"] - grp["fantasy_points"]).abs().mean()),
                "baseline_mae": float((grp["baseline_points"] - grp["fantasy_points"]).abs().mean()),
                "rank_corr": spearman(grp["expected_points"], grp["fantasy_points"]),
                "baseline_rank_corr": spearman(grp["baseline_points"], grp["fantasy_points"]),
                "top5_hit_rate": float((actual_rank[model_order <= top_k] <= cutoff).mean()),
                "baseline_top5_hit_rate": float((actual_rank[base_order <= top_k] <= cutoff).mean()),
                "top5_mean_points": float(grp.loc[model_order <= top_k, "fantasy_points"].mean()),
                "slate_mean_points": float(grp["fantasy_points"].mean()),
            }
        )
    weekly = pd.DataFrame(rows)
    if weekly.empty:
        return weekly, weekly

    summary = (
        weekly.groupby(["position", "season"])
        .agg(
            weeks=("week", "nunique"),
            mae=("mae", "mean"),
            baseline_mae=("baseline_mae", "mean"),
            rank_corr=("rank_corr", "mean"),
            baseline_rank_corr=("baseline_rank_corr", "mean"),
            top5_hit_rate=("top5_hit_rate", "mean"),
            baseline_top5_hit_rate=("baseline_top5_hit_rate", "mean"),
            top5_mean_points=("top5_mean_points", "mean"),
            slate_mean_points=("slate_mean_points", "mean"),
        )
        .reset_index()
    )
    summary["rank_corr_edge"] = summary["rank_corr"] - summary["baseline_rank_corr"]
    summary["mae_edge"] = summary["baseline_mae"] - summary["mae"]
    return weekly, summary


def run_backtest(
    kicker_features: pd.DataFrame,
    dst_features: pd.DataFrame,
    seasons: list[int],
    cfg: Config | None = None,
    kinds: list[str] | None = None,
    tune: bool = False,
) -> dict[str, WalkForwardResult]:
    """Backtest every candidate estimator for both positions.

    With ``tune`` the ridge penalty is also swept over ``model.ridge_alpha_grid``
    walk-forward, so the shipped hyper-parameter is chosen by out-of-sample
    performance rather than by hand.
    """
    cfg = cfg or get_config()
    kinds = kinds or list(cfg.model["candidates"])
    results: dict[str, WalkForwardResult] = {}
    for position, features in (("K", kicker_features), ("DST", dst_features)):
        for kind in kinds:
            alphas = (
                [float(a) for a in cfg.model["ridge_alpha_grid"]]
                if (tune and kind == "ridge")
                else [None]
            )
            for alpha in alphas:
                preds = walk_forward(features, position, seasons, cfg, kind=kind, alpha=alpha)
                if preds.empty:
                    continue
                weekly, summary = score_predictions(preds, cfg)
                label = f"{position}:{kind}" + (f":a{alpha:g}" if alpha is not None else "")
                results[label] = WalkForwardResult(preds, summary, weekly)
    return results


def select_best(results: dict[str, WalkForwardResult]) -> dict[str, dict[str, object]]:
    """Pick the configuration per position by mean rank correlation, MAE as tiebreak."""
    best: dict[str, tuple[float, float, dict[str, object]]] = {}
    for key, result in results.items():
        parts = key.split(":")
        position, kind = parts[0], parts[1]
        alpha = float(parts[2][1:]) if len(parts) > 2 else None
        if result.summary.empty:
            continue
        rc = float(result.summary["rank_corr"].mean())
        mae = float(result.summary["mae"].mean())
        current = best.get(position)
        choice: dict[str, object] = {"estimator": kind, "rank_corr": rc, "mae": mae}
        if alpha is not None:
            choice["ridge_alpha"] = alpha
        if current is None or rc > current[0] or (np.isclose(rc, current[0]) and mae < current[1]):
            best[position] = (rc, mae, choice)
    return {position: choice for position, (_rc, _mae, choice) in best.items()}
