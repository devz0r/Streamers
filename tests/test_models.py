"""Model behaviour: tier probabilities, the ledger, and the anchored core.

These assert the *properties* that make the projections trustworthy -- the tier
distribution really is a distribution, the ledger really does move weight when
a factor's correlation moves, and the model really does degrade to the Vegas
baseline rather than to nonsense.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from streamer.models.base import ResidualBands, sample_weights, spearman, top_n_probability
from streamer.models.ledger import FactorLedger


@pytest.fixture
def synthetic_dst(cfg):
    """A DST-shaped frame where the truth is a known function of the features."""
    rng = np.random.default_rng(7)
    n = 900
    seasons = rng.choice([2022, 2023, 2024], n)
    weeks = rng.integers(1, 18, n)
    opp_implied = rng.normal(22, 4, n)
    sack_rate = rng.normal(0.07, 0.02, n)
    frame = pd.DataFrame(
        {
            "season": seasons,
            "week": weeks,
            "game_id": [f"g{i}" for i in range(n)],
            "team": rng.choice(list("ABCDEFGH"), n),
            "opponent": rng.choice(list("IJKLMNOP"), n),
            "opp_implied_total": opp_implied,
            "total_line": opp_implied * 2 + rng.normal(0, 2, n),
            "team_spread": rng.normal(0, 5, n),
            "is_home": rng.integers(0, 2, n),
            "opp_sack_rate_allowed": sack_rate,
            "def_sack_rate": rng.normal(0.07, 0.015, n),
            "def_points_allowed_per_drive": rng.normal(2.0, 0.4, n),
            "opp_points_per_drive": rng.normal(2.0, 0.4, n),
            "opp_drives_per_game": rng.normal(11, 1, n),
            "opp_plays_per_game": rng.normal(63, 5, n),
        }
    )
    frame["points_allowed"] = np.clip(
        rng.poisson(np.clip(opp_implied, 3, None)), 0, None
    )
    frame["fantasy_points"] = (
        14 - 0.4 * opp_implied + 60 * sack_rate + rng.normal(0, 4, n)
    )
    frame["big_play_points"] = frame["fantasy_points"] - 3.0
    return frame


# ---------------------------------------------------------------------------
# Tier model
# ---------------------------------------------------------------------------
def test_tier_probabilities_form_a_distribution(cfg, synthetic_dst):
    from streamer.models.tiers import TierModel

    model = TierModel.fit(synthetic_dst, cfg)
    probs = model.tier_probabilities(synthetic_dst.head(30))
    assert probs.shape == (30, len(cfg.dst_scoring["points_allowed_tiers"]))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()


def test_a_low_implied_total_shifts_mass_to_the_good_tiers(cfg, synthetic_dst):
    from streamer.models.tiers import TierModel

    model = TierModel.fit(synthetic_dst, cfg)
    slate = synthetic_dst.head(2).copy()
    slate.loc[slate.index[0], "opp_implied_total"] = 13.0
    slate.loc[slate.index[1], "opp_implied_total"] = 32.0
    probs = model.tier_probabilities(slate)
    # Tiers 0-2 are "held them under 14", the tiers worth 3 points or more.
    assert probs[0, :3].sum() > probs[1, :3].sum()
    assert model.expected_tier_points(slate)[0] > model.expected_tier_points(slate)[1]


def test_expected_points_allowed_tracks_the_implied_total(cfg, synthetic_dst):
    from streamer.models.tiers import TierModel

    model = TierModel.fit(synthetic_dst, cfg)
    slate = synthetic_dst.head(2).copy()
    slate.loc[slate.index[0], "opp_implied_total"] = 15.0
    slate.loc[slate.index[1], "opp_implied_total"] = 30.0
    predicted = model.expected_points_allowed(slate)
    assert predicted[1] > predicted[0]


def test_tier_model_needs_realised_results(cfg, synthetic_dst):
    from streamer.models.tiers import TierModel

    blank = synthetic_dst.copy()
    blank["points_allowed"] = np.nan
    with pytest.raises(ValueError):
        TierModel.fit(blank, cfg)


# ---------------------------------------------------------------------------
# Residual bands and P(top-12)
# ---------------------------------------------------------------------------
def test_residual_bands_bracket_the_prediction():
    rng = np.random.default_rng(3)
    pred = rng.normal(8, 3, 500)
    actual = pred + rng.normal(0, 4, 500)
    bands = ResidualBands.fit(pred, actual)
    lo = bands.quantile(np.array([8.0]), 0.15)[0]
    hi = bands.quantile(np.array([8.0]), 0.85)[0]
    assert lo < 8.0 < hi


def test_top_n_probability_is_bounded_and_ordered():
    rng = np.random.default_rng(5)
    pred = rng.normal(8, 3, 200)
    bands = ResidualBands.fit(pred, pred + rng.normal(0, 4, 200))
    slate = np.array([15.0, 10.0, 8.0, 3.0])
    probs = top_n_probability(slate, bands, cutoff=2, draws=2000)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert probs[0] > probs[-1]


def test_top_n_probability_handles_an_empty_slate():
    bands = ResidualBands.fit(np.array([1.0]), np.array([1.0]))
    assert len(top_n_probability(np.array([]), bands, cutoff=12)) == 0


# ---------------------------------------------------------------------------
# Sample weighting
# ---------------------------------------------------------------------------
def test_current_season_is_weighted_up(cfg):
    frame = pd.DataFrame({"season": [2024, 2025, 2026]})
    weights = sample_weights(frame, current_season=2026, cfg=cfg)
    assert weights[2] > weights[1] > weights[0]
    # The current season carries the configured multiplier over an undecayed row.
    assert weights[2] == pytest.approx(float(cfg.model["current_season_weight"]))


def test_older_seasons_decay(cfg):
    frame = pd.DataFrame({"season": [2021, 2022]})
    weights = sample_weights(frame, current_season=2026, cfg=cfg)
    assert weights[0] < weights[1]


# ---------------------------------------------------------------------------
# Factor ledger
# ---------------------------------------------------------------------------
def test_ledger_reports_all_three_windows(cfg, synthetic_dst):
    ledger = FactorLedger.compute(
        synthetic_dst, ["opp_implied_total", "opp_sack_rate_allowed"],
        "fantasy_points", "DST", asof_season=2024, asof_week=10, cfg=cfg,
    )
    assert set(ledger.frame["factor"]) == {"opp_implied_total", "opp_sack_rate_allowed"}
    for column in ("r_hist", "r_current", "r_trailing", "r_blend", "multiplier"):
        assert column in ledger.frame.columns
    # The known-strong factors must come out with the right signs.
    row = ledger.frame.set_index("factor").loc["opp_implied_total"]
    assert row["r_hist"] < 0
    assert ledger.frame.set_index("factor").loc["opp_sack_rate_allowed"]["r_hist"] > 0


def test_blend_shrinks_toward_the_historical_prior(cfg, synthetic_dst):
    """Early in a season the historical value must dominate."""
    early = FactorLedger.compute(
        synthetic_dst, ["opp_implied_total"], "fantasy_points", "DST",
        asof_season=2024, asof_week=2, cfg=cfg,
    ).frame.iloc[0]
    late = FactorLedger.compute(
        synthetic_dst, ["opp_implied_total"], "fantasy_points", "DST",
        asof_season=2024, asof_week=17, cfg=cfg,
    ).frame.iloc[0]
    assert abs(early["r_blend"] - early["r_hist"]) <= abs(late["r_blend"] - late["r_hist"])


def test_a_factor_whose_signal_dies_loses_weight(cfg, synthetic_dst):
    """The requirement: a meta shift must move weight automatically."""
    rng = np.random.default_rng(11)
    broken = synthetic_dst.copy()
    current = broken["season"] == 2024
    # Sever the sack-rate relationship for the current season only.
    broken.loc[current, "opp_sack_rate_allowed"] = rng.normal(
        0.07, 0.02, int(current.sum())
    )
    intact = FactorLedger.compute(
        synthetic_dst, ["opp_implied_total", "opp_sack_rate_allowed"], "fantasy_points",
        "DST", asof_season=2024, asof_week=17, cfg=cfg,
    ).frame.set_index("factor")
    shifted = FactorLedger.compute(
        broken, ["opp_implied_total", "opp_sack_rate_allowed"], "fantasy_points",
        "DST", asof_season=2024, asof_week=17, cfg=cfg,
    ).frame.set_index("factor")
    assert (
        shifted.loc["opp_sack_rate_allowed", "multiplier"]
        < intact.loc["opp_sack_rate_allowed", "multiplier"]
    )


def test_ledger_records_weight_movement(cfg, synthetic_dst):
    factors = ["opp_implied_total", "opp_sack_rate_allowed"]
    first = FactorLedger.compute(
        synthetic_dst, factors, "fantasy_points", "DST", 2024, 8, cfg
    )
    first.attach_model_weights({"opp_implied_total": -1.0, "opp_sack_rate_allowed": 0.5})
    second = FactorLedger.compute(
        synthetic_dst, factors, "fantasy_points", "DST", 2024, 9, cfg, previous=first.frame
    )
    second.attach_model_weights({"opp_implied_total": -1.4, "opp_sack_rate_allowed": 0.9})
    moves = second.frame.set_index("factor")["weight_delta"]
    assert moves["opp_implied_total"] == pytest.approx(-0.4)
    assert moves["opp_sack_rate_allowed"] == pytest.approx(0.4)
    assert not second.top_movers(2).empty


def test_ledger_survives_a_factor_with_no_data(cfg, synthetic_dst):
    frame = synthetic_dst.copy()
    frame["dead_factor"] = np.nan
    ledger = FactorLedger.compute(
        frame, ["opp_implied_total", "dead_factor"], "fantasy_points", "DST", 2024, 10, cfg
    )
    assert len(ledger.frame) == 2
    assert np.isfinite(ledger.multipliers["dead_factor"])


# ---------------------------------------------------------------------------
# The anchored model
# ---------------------------------------------------------------------------
def test_dst_model_projects_and_reports_probabilities(cfg, synthetic_dst):
    from streamer.models.positions import DstModel

    train = synthetic_dst[synthetic_dst["season"] < 2024]
    test = synthetic_dst[synthetic_dst["season"] == 2024].head(20)
    model = DstModel.fit(train, cfg, current_season=2024)
    out = model.predict(test).frame
    for column in ("expected_points", "floor", "ceiling", "p_top12",
                   "expected_points_allowed", "expected_tier_points", "p_shutout"):
        assert column in out.columns
        assert out[column].notna().all()
    assert (out["floor"] <= out["expected_points"]).all()
    assert (out["ceiling"] >= out["expected_points"]).all()


def test_model_recovers_a_known_relationship(cfg, synthetic_dst):
    """Truth is 14 - 0.4 * implied + 60 * sack_rate; ranking must reflect it."""
    from streamer.models.positions import DstModel

    train = synthetic_dst[synthetic_dst["season"] < 2024]
    test = synthetic_dst[synthetic_dst["season"] == 2024]
    model = DstModel.fit(train, cfg, current_season=2024)
    out = model.predict(test).frame
    assert spearman(out["expected_points"], out["fantasy_points"]) > 0.3


def test_vegas_baseline_is_a_sane_reference(cfg, synthetic_dst):
    from streamer.models.positions import VegasBaseline

    baseline = VegasBaseline.fit(synthetic_dst, "opp_implied_total")
    assert baseline.slope < 0  # more implied points for the offence = worse D/ST
    preds = baseline.predict(synthetic_dst.head(10))
    assert np.isfinite(preds).all()


def test_fitting_with_no_completed_games_raises(cfg, synthetic_dst):
    from streamer.models.positions import DstModel

    blank = synthetic_dst.copy()
    blank["fantasy_points"] = np.nan
    with pytest.raises(ValueError):
        DstModel.fit(blank, cfg)


def test_spearman_is_nan_safe():
    assert np.isnan(spearman([1, 2], [1, 2]))            # too few points
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))      # degenerate
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
