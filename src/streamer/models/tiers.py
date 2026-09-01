"""Points-allowed tiers as a probability distribution.

The spec is explicit that the tier component must be a distribution, not a
point estimate, and the reason is arithmetic: the ESPN ladder is a step
function, so ``E[tier(PA)] != tier(E[PA])``. A defense projected to allow 20.5
points is not "0 points"; it is a mixture that is mostly 0 and -1 with a real
tail into +3 and +4.

The model is two-stage:

1. A ridge regression predicts *mean* points allowed from the opponent's
   implied total (dominant), the defense's own points-per-drive prior, pace and
   home field.
2. The spread around that mean is taken from historical residuals, bucketed by
   predicted level because high-scoring spots are also more variable. Sampling
   those residuals gives a full distribution over points allowed, which is
   binned into the ESPN ladder.
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


@dataclass
class TierModel:
    """Distribution over opponent points, collapsed onto the ESPN tier ladder."""

    scoring: DstScoring
    columns: list[str]
    means: np.ndarray
    scales: np.ndarray
    coef: np.ndarray
    intercept: float
    bands: ResidualBands
    draws: int = 4000
    seed: int = 5

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        cfg: Config | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "TierModel":
        cfg = cfg or get_config()
        scoring = DstScoring.from_config(cfg)
        columns = [c for c in TIER_FEATURES if c in frame.columns]
        train = frame[frame["points_allowed"].notna()]
        if train.empty:
            raise ValueError("cannot fit a tier model without realised points allowed")
        raw = _numeric_block(train, columns)
        means = raw.mean(axis=0)
        scales = raw.std(axis=0)
        scales[scales < 1e-8] = 1.0
        x = (raw - means) / scales
        y = pd.to_numeric(train["points_allowed"], errors="coerce").to_numpy(float)
        model = Ridge(alpha=ridge_alpha(cfg, "DST"))
        weight = None if sample_weight is None else np.asarray(sample_weight, float)[
            frame["points_allowed"].notna().to_numpy()
        ]
        model.fit(x, y, sample_weight=weight)
        fitted = model.predict(x)
        return cls(
            scoring=scoring,
            columns=columns,
            means=means,
            scales=scales,
            coef=model.coef_,
            intercept=float(model.intercept_),
            bands=ResidualBands.fit(fitted, y),
            draws=int(cfg.model["sim_draws"]),
        )

    # -- prediction --------------------------------------------------------
    def expected_points_allowed(self, frame: pd.DataFrame) -> np.ndarray:
        raw = _numeric_block(frame, self.columns)
        x = (raw - self.means) / self.scales
        return x @ self.coef + self.intercept

    def tier_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        """``(n_rows, n_tiers)`` matrix of points-allowed tier probabilities."""
        mu = self.expected_points_allowed(frame)
        rng = np.random.default_rng(self.seed)
        sims = self.bands.sample(mu, self.draws, rng)
        sims = np.clip(np.rint(sims), 0, None)
        n_tiers = len(self.scoring.points_allowed_tiers)
        edges = [t[1] for t in self.scoring.points_allowed_tiers[:-1]]
        # Bucket by the tier ladder's upper bounds.
        idx = np.searchsorted(np.asarray(edges, dtype=float), sims, side="left")
        idx = np.clip(idx, 0, n_tiers - 1)
        probs = np.zeros((sims.shape[1], n_tiers))
        for t in range(n_tiers):
            probs[:, t] = (idx == t).mean(axis=0)
        return probs

    def expected_tier_points(self, frame: pd.DataFrame) -> np.ndarray:
        probs = self.tier_probabilities(frame)
        return probs @ np.asarray(self.scoring.tier_values, dtype=float)

    def shutout_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return self.tier_probabilities(frame)[:, 0]

    def sample_tier_points(self, frame: pd.DataFrame, draws: int, rng: np.random.Generator) -> np.ndarray:
        """``(draws, n_rows)`` simulated tier points, for the P(top-12) sim."""
        mu = self.expected_points_allowed(frame)
        sims = self.bands.sample(mu, draws, rng)
        sims = np.clip(np.rint(sims), 0, None)
        edges = [t[1] for t in self.scoring.points_allowed_tiers[:-1]]
        idx = np.clip(
            np.searchsorted(np.asarray(edges, dtype=float), sims, side="left"),
            0,
            len(self.scoring.points_allowed_tiers) - 1,
        )
        return np.asarray(self.scoring.tier_values, dtype=float)[idx]
