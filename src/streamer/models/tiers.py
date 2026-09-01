"""Ladder components as probability distributions.

Two step-function ladders can contribute to a D/ST score: **points allowed**
(every profile) and **total yards allowed** (ESPN only). Both are handled here
by the same :class:`LadderModel`, and for the same reason.

A ladder is a step function, so ``E[tier(x)] != tier(E[x])``. A defence
projected to allow 20.5 points is not "0 points"; under ESPN's ladder it is a
mixture that is mostly 0 and -1 with real mass on +3 and +1. Collapsing to a
point estimate first throws that away and biases every projection toward the
middle of the ladder.

The model is two-stage:

1. A ridge regression predicts the *mean* of the underlying quantity -- points
   from the opponent's implied total (dominant), the defence's own
   points-per-drive prior, pace and home field; yards mostly from volume.
2. The spread around that mean comes from historical residuals, bucketed by
   predicted level because high-scoring (and high-yardage) spots are also more
   variable. Sampling those residuals gives a full distribution, which is
   binned onto the ladder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ..config import Config, get_config
from ..scoring import DstScoring
from .base import ResidualBands, _numeric_block, ridge_alpha

#: Inputs to the mean points-allowed regression.
TIER_FEATURES: tuple[str, ...] = (
    "opp_implied_total",
    "total_line",
    "team_spread",
    "is_home",
    "def_points_allowed_per_drive",
    "opp_points_per_drive",
    "opp_drives_per_game",
    "opp_plays_per_game",
)

#: Inputs to the mean yards-allowed regression. Volume matters far more here
#: than the betting market does: yards accumulate with plays, and a defence
#: that concedes efficiency concedes yardage whatever the scoreboard says.
YARDS_FEATURES: tuple[str, ...] = (
    "def_yards_allowed_per_game",
    "opp_plays_per_game",
    "opp_drives_per_game",
    "opp_dropbacks_per_game",
    "opp_points_per_drive",
    "opp_pass_rate",
    "opp_neutral_plays_per_game",
    "opp_implied_total",
    "total_line",
    "team_spread",
    "is_home",
    "def_points_allowed_per_drive",
)

#: Per-ladder wiring: which column holds the realised quantity, and which
#: features predict its mean.
LADDERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "points": ("points_allowed", TIER_FEATURES),
    "yards": ("yards_allowed", YARDS_FEATURES),
}


@dataclass
class LadderModel:
    """Distribution over a realised quantity, collapsed onto a scoring ladder."""

    scoring: DstScoring
    ladder: str
    target: str
    columns: list[str]
    means: np.ndarray
    scales: np.ndarray
    coef: np.ndarray
    intercept: float
    bands: ResidualBands
    draws: int = 4000
    seed: int = 5

    # -- construction ------------------------------------------------------
    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        cfg: Config | None = None,
        sample_weight: np.ndarray | None = None,
        ladder: str = "points",
    ) -> LadderModel:
        cfg = cfg or get_config()
        scoring = DstScoring.from_config(cfg)
        if ladder not in LADDERS:
            raise ValueError(f"unknown ladder: {ladder!r}")
        target, feature_names = LADDERS[ladder]

        if frame.empty or target not in frame.columns:
            raise ValueError(f"cannot fit the {ladder} ladder without a {target} column")
        columns = [c for c in feature_names if c in frame.columns]
        mask = frame[target].notna()
        train = frame[mask]
        if train.empty:
            raise ValueError(f"cannot fit the {ladder} ladder without realised {target}")

        raw = _numeric_block(train, columns)
        means = raw.mean(axis=0)
        scales = raw.std(axis=0)
        scales[scales < 1e-8] = 1.0
        x = (raw - means) / scales
        y = pd.to_numeric(train[target], errors="coerce").to_numpy(float)

        weight = None
        if sample_weight is not None:
            weight = np.asarray(sample_weight, float)[mask.to_numpy()]

        model = Ridge(alpha=ridge_alpha(cfg, "DST"))
        model.fit(x, y, sample_weight=weight)
        fitted = model.predict(x)
        return cls(
            scoring=scoring,
            ladder=ladder,
            target=target,
            columns=columns,
            means=means,
            scales=scales,
            coef=model.coef_,
            intercept=float(model.intercept_),
            bands=ResidualBands.fit(fitted, y),
            draws=int(cfg.model["sim_draws"]),
        )

    # -- ladder wiring -----------------------------------------------------
    @property
    def tiers(self) -> tuple[tuple[int, int | None, float], ...]:
        tiers = (
            self.scoring.points_allowed_tiers
            if self.ladder == "points"
            else self.scoring.yards_allowed_tiers
        )
        if tiers is None:
            raise ValueError("this profile does not score yards allowed")
        return tiers

    @property
    def tier_values(self) -> np.ndarray:
        return np.asarray([pts for _lo, _hi, pts in self.tiers], dtype=float)

    @property
    def tier_labels(self) -> tuple[str, ...]:
        return (
            self.scoring.tier_labels
            if self.ladder == "points"
            else self.scoring.yards_tier_labels
        )

    # -- prediction --------------------------------------------------------
    def expected_value(self, frame: pd.DataFrame) -> np.ndarray:
        """Mean of the underlying quantity (points or yards allowed)."""
        raw = _numeric_block(frame, self.columns)
        x = (raw - self.means) / self.scales
        return x @ self.coef + self.intercept

    def _bin(self, sims: np.ndarray) -> np.ndarray:
        """Bucket simulated values onto the ladder, returning tier indices."""
        edges = np.asarray([hi for _lo, hi, _pts in self.tiers[:-1]], dtype=float)
        sims = np.clip(np.rint(sims), 0, None)
        return np.clip(np.searchsorted(edges, sims, side="left"), 0, len(self.tiers) - 1)

    def tier_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        """``(n_rows, n_tiers)`` matrix of ladder probabilities."""
        mu = self.expected_value(frame)
        rng = np.random.default_rng(self.seed)
        idx = self._bin(self.bands.sample(mu, self.draws, rng))
        n_tiers = len(self.tiers)
        probs = np.zeros((idx.shape[1], n_tiers))
        for t in range(n_tiers):
            probs[:, t] = (idx == t).mean(axis=0)
        return probs

    def expected_tier_points(self, frame: pd.DataFrame) -> np.ndarray:
        return self.tier_probabilities(frame) @ self.tier_values

    def sample_tier_points(
        self, frame: pd.DataFrame, draws: int, rng: np.random.Generator
    ) -> np.ndarray:
        """``(draws, n_rows)`` simulated ladder points."""
        mu = self.expected_value(frame)
        return self.tier_values[self._bin(self.bands.sample(mu, draws, rng))]


@dataclass
class TierModel(LadderModel):
    """The points-allowed ladder, with the shorthand the D/ST model reads."""

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        cfg: Config | None = None,
        sample_weight: np.ndarray | None = None,
        ladder: str = "points",
    ) -> TierModel:
        built = LadderModel.fit(frame, cfg, sample_weight, ladder)
        return cls(**built.__dict__)

    def expected_points_allowed(self, frame: pd.DataFrame) -> np.ndarray:
        return self.expected_value(frame)

    def shutout_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return self.tier_probabilities(frame)[:, 0]
