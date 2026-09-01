"""The Kicker and D/ST projection models.

Both share the same shape -- a Vegas anchor plus a regularised adjustment
stage (:class:`_AnchoredCore`) -- and both follow the same contract: ``fit`` on
completed games, ``predict`` on a slate.

**Kicker.** Weekly kicker scoring is close to irreducible noise; the strongest
single public factor correlates about 0.12 with actual points. The job is
therefore to squeeze the Vegas signal and the team's field-goal *generation*
profile without overfitting to kicker identity, which is exactly what a heavily
penalised adjustment stage does.

**D/ST** additionally fits a :class:`~streamer.models.tiers.TierModel`, because
the points-allowed component is a step function of a quantity Vegas already
prices and therefore has to be handled as a *distribution*: a defence projected
to allow 20.5 points is a mixture, not a single tier. The tier model always
runs -- it is what produces ``P(shutout)``, ``P(under 14)`` and the per-tier
probabilities -- but whether it also feeds the point projection is a structural
choice the backtest makes, not an assumption:

``two_stage=False`` (the default, and what the 2023-2025 backtest selected)
    Project total fantasy points directly; the tier model supplies
    probabilities only.
``two_stage=True``
    Project the big-play component only, and add the tier model's expected
    points. Better MAE, worse ranking -- see DECISIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

from ..config import Config, get_config
from ..features.build import DST_FEATURES, KICKER_FEATURES, VEGAS_ANCHOR
from .base import (
    DesignMatrix,
    ResidualBands,
    fit_design,
    make_estimator,
    sample_weights,
    top_n_probability,
)
from .tiers import TierModel

#: Factors whose blended correlation collapses this far below their historical
#: value are dropped from the tree model's design matrix entirely -- the
#: mechanism by which the ledger moves weight for an estimator that has no
#: coefficients to scale.
LEDGER_DROP_THRESHOLD = 0.3


@dataclass
class Projection:
    """A model's output for one slate."""

    frame: pd.DataFrame
    feature_weights: dict[str, float]
    estimator_kind: str


@dataclass
class _AnchoredCore:
    """A Vegas anchor plus a regularised adjustment stage.

    Stage 1 is a lightly-regularised ridge on the Vegas block alone -- the
    honest baseline anybody could build. Stage 2 fits every factor against what
    stage 1 leaves behind, heavily regularised and scaled by the factor
    ledger's multipliers, so the adjustments can only move the projection as far
    as their measured signal justifies. As the stage-2 penalty rises the model
    degrades gracefully back to the Vegas baseline rather than falling apart,
    which is what makes it reliably *beat* that baseline out of sample.
    """

    kind: str
    anchor_columns: list[str]
    anchor: DesignMatrix
    anchor_model: object
    design: DesignMatrix
    estimator: object
    bands: ResidualBands
    columns: list[str]
    weights: dict[str, float] = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        base = np.asarray(self.anchor_model.predict(self.anchor.transform(frame)), dtype=float)
        adjust = np.asarray(self.estimator.predict(self.design.transform(frame)), dtype=float)
        return base + adjust

    def anchor_only(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.anchor_model.predict(self.anchor.transform(frame)), dtype=float)


def _fit_core(
    train: pd.DataFrame,
    columns: list[str],
    target: str,
    kind: str,
    cfg: Config,
    current_season: int,
    multipliers: dict[str, float] | None = None,
    position: str | None = None,
    alpha: float | None = None,
    compute_weights: bool = True,
) -> _AnchoredCore:
    multipliers = multipliers or {}
    anchor_columns = [c for c in VEGAS_ANCHOR.get(position or "", ()) if c in train.columns]
    if not anchor_columns:
        anchor_columns = [columns[0]] if columns else []

    y = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
    w = sample_weights(train, current_season, cfg)

    # -- stage 1: the Vegas anchor -----------------------------------------
    anchor_design = fit_design(train, anchor_columns, None)
    anchor_model = Ridge(alpha=float(cfg.model["anchor_alpha"]))
    anchor_model.fit(anchor_design.transform(train), y, sample_weight=w)
    base = np.asarray(anchor_model.predict(anchor_design.transform(train)), dtype=float)
    residual = y - base

    # -- stage 2: adjustments ----------------------------------------------
    if kind == "gbm":
        # Trees are invariant to monotone feature scaling, so the ledger acts on
        # them by pruning factors whose signal has collapsed.
        columns = [c for c in columns if multipliers.get(c, 1.0) >= LEDGER_DROP_THRESHOLD] or columns
        design = fit_design(train, columns, None)
    else:
        design = fit_design(train, columns, multipliers)

    x = design.transform(train)
    estimator = make_estimator(kind, cfg, position=position, alpha=alpha)
    estimator.fit(x, residual, sample_weight=w)
    fitted = base + np.asarray(estimator.predict(x), dtype=float)

    if kind == "ridge":
        # Report the *total* standardised influence of each factor: its anchor
        # coefficient (if it is in the Vegas block) plus its adjustment weight.
        weights = dict(zip(columns, np.asarray(estimator.coef_, dtype=float)))
        for name, coef in zip(anchor_columns, np.asarray(anchor_model.coef_, dtype=float)):
            weights[name] = weights.get(name, 0.0) + float(coef)
    elif compute_weights:
        # Permutation importance costs more than the fit itself, so the
        # backtest -- which refits once per week and never reads the weights --
        # skips it.
        weights = _tree_importances(estimator, x, residual, columns)
    else:
        weights = {}

    return _AnchoredCore(
        kind=kind,
        anchor_columns=anchor_columns,
        anchor=anchor_design,
        anchor_model=anchor_model,
        design=design,
        estimator=estimator,
        bands=ResidualBands.fit(fitted, y),
        columns=list(columns),
        weights=weights,
    )


def _tree_importances(estimator, x, y, columns: list[str]) -> dict[str, float]:
    """Permutation importance, so a tree model reports comparable factor weights."""
    n = min(len(x), 1500)
    try:
        result = permutation_importance(
            estimator, x[:n], y[:n], n_repeats=3, random_state=7, scoring="neg_mean_absolute_error"
        )
        return dict(zip(columns, np.asarray(result.importances_mean, dtype=float)))
    except Exception:  # noqa: BLE001 - importance is diagnostic, never load-bearing
        return {c: float("nan") for c in columns}


# ---------------------------------------------------------------------------
# Kicker
# ---------------------------------------------------------------------------
@dataclass
class KickerModel:
    core: _AnchoredCore
    cfg: Config

    FEATURES = KICKER_FEATURES
    TARGET = "fantasy_points"
    POSITION = "K"

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        cfg: Config | None = None,
        kind: str = "ridge",
        current_season: int | None = None,
        multipliers: dict[str, float] | None = None,
        alpha: float | None = None,
        compute_weights: bool = True,
    ) -> KickerModel:
        cfg = cfg or get_config()
        train = train[train[cls.TARGET].notna()]
        if train.empty:
            raise ValueError("no completed kicker games to train on")
        current_season = current_season or int(train["season"].max())
        if multipliers is None:
            from .ledger import multipliers_for

            multipliers = multipliers_for(
                train, list(cls.FEATURES), cls.TARGET, cls.POSITION,
                int(current_season), int(train["week"].max()) + 1, cfg,
            )
        core = _fit_core(
            train, list(cls.FEATURES), cls.TARGET, kind, cfg, current_season, multipliers,
            position=cls.POSITION, alpha=alpha, compute_weights=compute_weights,
        )
        return cls(core=core, cfg=cfg)

    def predict(self, slate: pd.DataFrame) -> Projection:
        cfg = self.cfg
        preds = self.core.predict(slate)
        out = slate.copy()
        out["expected_points"] = preds
        out["floor"] = np.maximum(
            0.0, self.core.bands.quantile(preds, float(cfg.model["floor_quantile"]))
        )
        out["ceiling"] = self.core.bands.quantile(preds, float(cfg.model["ceiling_quantile"]))
        out["p_top12"] = top_n_probability(
            preds, self.core.bands, cfg.startable_rank, int(cfg.model["sim_draws"])
        )
        return Projection(frame=out, feature_weights=self.core.weights, estimator_kind=self.core.kind)


# ---------------------------------------------------------------------------
# D/ST
# ---------------------------------------------------------------------------
@dataclass
class DstModel:
    core: _AnchoredCore
    tier_model: TierModel
    cfg: Config
    two_stage: bool = False

    FEATURES = DST_FEATURES
    TARGET = "fantasy_points"
    BIG_PLAY_TARGET = "big_play_points"
    POSITION = "DST"

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        cfg: Config | None = None,
        kind: str = "ridge",
        current_season: int | None = None,
        multipliers: dict[str, float] | None = None,
        two_stage: bool = False,
        alpha: float | None = None,
        compute_weights: bool = True,
    ) -> DstModel:
        cfg = cfg or get_config()
        train = train[train[cls.TARGET].notna()]
        if train.empty:
            raise ValueError("no completed D/ST games to train on")
        current_season = current_season or int(train["season"].max())
        target = cls.BIG_PLAY_TARGET if two_stage else cls.TARGET
        if multipliers is None:
            # Keep production identical to what the backtest measured: weight the
            # core regression against the target it actually fits.
            from .ledger import multipliers_for

            multipliers = multipliers_for(
                train, list(cls.FEATURES), target, cls.POSITION,
                int(current_season), int(train["week"].max()) + 1, cfg,
            )
        core = _fit_core(
            train, list(cls.FEATURES), target, kind, cfg, current_season, multipliers,
            position=cls.POSITION, alpha=alpha, compute_weights=compute_weights,
        )
        tier_model = TierModel.fit(train, cfg, sample_weights(train, current_season, cfg))
        if two_stage:
            # The residual band must describe the *total*, not the big-play part
            # alone, or the floor/ceiling and P(top-12) will be far too tight.
            total_fitted = core.predict(train) + tier_model.expected_tier_points(train)
            core.bands = ResidualBands.fit(
                total_fitted, pd.to_numeric(train[cls.TARGET], errors="coerce").to_numpy(float)
            )
        return cls(core=core, tier_model=tier_model, cfg=cfg, two_stage=two_stage)

    def predict(self, slate: pd.DataFrame) -> Projection:
        cfg = self.cfg
        base = self.core.predict(slate)
        out = slate.copy()
        probs = self.tier_model.tier_probabilities(slate)
        tier_values = np.asarray(self.tier_model.scoring.tier_values, dtype=float)
        out["expected_points_allowed"] = self.tier_model.expected_points_allowed(slate)
        out["expected_tier_points"] = probs @ tier_values
        out["p_shutout"] = probs[:, 0]
        # P(holding the opponent under 14), the practical "good week" threshold.
        under_14 = [i for i, (_lo, hi, _v) in enumerate(self.tier_model.scoring.points_allowed_tiers)
                    if hi is not None and hi <= 13]
        out["p_under_14"] = probs[:, under_14].sum(axis=1) if under_14 else 0.0
        for i, label in enumerate(self.tier_model.scoring.tier_labels):
            out[f"p_pa_{label}"] = probs[:, i]

        if self.two_stage:
            out["expected_big_play_points"] = base
            preds = base + out["expected_tier_points"].to_numpy()
        else:
            out["expected_big_play_points"] = base - out["expected_tier_points"].to_numpy()
            preds = base
        out["expected_points"] = preds

        out["floor"] = self.core.bands.quantile(preds, float(cfg.model["floor_quantile"]))
        out["ceiling"] = self.core.bands.quantile(preds, float(cfg.model["ceiling_quantile"]))
        out["p_top12"] = top_n_probability(
            preds, self.core.bands, cfg.startable_rank, int(cfg.model["sim_draws"])
        )
        return Projection(frame=out, feature_weights=self.core.weights, estimator_kind=self.core.kind)


# ---------------------------------------------------------------------------
# Naive Vegas-only baselines (the bar the full model must clear)
# ---------------------------------------------------------------------------
@dataclass
class VegasBaseline:
    """Ordinary least squares on the single Vegas number for the position.

    This is the honest benchmark: anybody can rank kickers by team implied
    total and defenses by opponent implied total in a spreadsheet. Everything
    the full model adds has to beat it.
    """

    column: str
    slope: float
    intercept: float
    sign: float

    @classmethod
    def fit(cls, train: pd.DataFrame, column: str, target: str = "fantasy_points") -> VegasBaseline:
        x = pd.to_numeric(train[column], errors="coerce")
        y = pd.to_numeric(train[target], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 10:
            return cls(column=column, slope=0.0, intercept=float(y[mask].mean() or 0.0), sign=1.0)
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        return cls(column=column, slope=float(slope), intercept=float(intercept),
                   sign=float(np.sign(slope) or 1.0))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = pd.to_numeric(frame[self.column], errors="coerce").fillna(
            pd.to_numeric(frame[self.column], errors="coerce").mean()
        )
        return np.asarray(self.slope * x + self.intercept, dtype=float)


#: Which Vegas number anchors each position's baseline.
BASELINE_COLUMN = {"K": "team_implied_total", "DST": "opp_implied_total"}
