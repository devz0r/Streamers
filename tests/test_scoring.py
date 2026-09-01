"""Exact scoring tests.

The ESPN D/ST points-allowed ladder is the single place where an off-by-one
silently corrupts every projection and every backtest, so every tier boundary
is asserted explicitly rather than sampled.
"""

from __future__ import annotations

import pytest

from streamer.config import Config
from streamer.scoring import (
    FG_BUCKETS,
    DstScoring,
    DstStatLine,
    KickerScoring,
    KickerStatLine,
    fg_bucket,
    tier_probabilities_from_samples,
)

# ---------------------------------------------------------------------------
# D/ST points-allowed tiers
# ---------------------------------------------------------------------------
#: Every boundary of the ESPN ladder, including both sides of each step.
TIER_CASES = [
    (0, 5.0),
    (1, 4.0), (3, 4.0), (6, 4.0),
    (7, 3.0), (10, 3.0), (13, 3.0),
    (14, 1.0), (16, 1.0), (17, 1.0),
    (18, 0.0), (20, 0.0), (21, 0.0),
    (22, -1.0), (25, -1.0), (27, -1.0),
    (28, -3.0), (31, -3.0), (34, -3.0),
    (35, -5.0), (40, -5.0), (45, -5.0),
    (46, -6.0), (59, -6.0), (100, -6.0),
]


@pytest.mark.parametrize("points_allowed,expected", TIER_CASES)
def test_points_allowed_tier_boundaries(cfg, points_allowed, expected):
    assert DstScoring.from_config(cfg).points_allowed_points(points_allowed) == expected


def test_tier_ladder_is_contiguous_and_covers_everything(cfg):
    scoring = DstScoring.from_config(cfg)
    # Walking 0..120 must never raise and must never skip a value.
    values = [scoring.points_allowed_points(pa) for pa in range(0, 121)]
    assert len(values) == 121
    # Tier points are monotonically non-increasing as points allowed rises.
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_negative_points_allowed_is_rejected(cfg):
    with pytest.raises(ValueError):
        DstScoring.from_config(cfg).points_allowed_points(-1)


def test_tier_ladder_validation_rejects_gaps(cfg):
    broken = dict(cfg.raw["dst_scoring"])
    broken["points_allowed_tiers"] = [[0, 0, 5.0], [2, None, 4.0]]  # 1 is unreachable
    bad = Config(raw={**cfg.raw, "dst_scoring": broken}, root=cfg.root)
    with pytest.raises(ValueError, match="contiguous"):
        DstScoring.from_config(bad)


def test_tier_ladder_validation_requires_open_top(cfg):
    broken = dict(cfg.raw["dst_scoring"])
    broken["points_allowed_tiers"] = [[0, 0, 5.0], [1, 40, 4.0]]
    bad = Config(raw={**cfg.raw, "dst_scoring": broken}, root=cfg.root)
    with pytest.raises(ValueError, match="open-ended"):
        DstScoring.from_config(bad)


# ---------------------------------------------------------------------------
# D/ST full stat lines
# ---------------------------------------------------------------------------
def test_dst_scores_a_realistic_line(cfg):
    scoring = DstScoring.from_config(cfg)
    # 10 allowed (3) + 4 sacks (4) + 2 INT (4) + 1 FR (2) = 13
    line = DstStatLine(points_allowed=10, sacks=4, interceptions=2, fumble_recoveries=1)
    assert scoring.score(line) == pytest.approx(13.0)


def test_dst_scores_every_component(cfg):
    scoring = DstScoring.from_config(cfg)
    line = DstStatLine(
        points_allowed=0, sacks=1, interceptions=1, fumble_recoveries=1, safeties=1,
        defensive_tds=1, return_tds=1, blocked_kicks=1, blocked_kick_tds=1,
        extra_points_returned=1,
    )
    # 5 + 1 + 2 + 2 + 2 + 6 + 6 + 2 + 6 + 2
    assert scoring.score(line) == pytest.approx(34.0)


def test_score_without_tier_strips_only_the_tier(cfg):
    scoring = DstScoring.from_config(cfg)
    line = DstStatLine(points_allowed=3, sacks=2, interceptions=1)
    assert scoring.score(line) == pytest.approx(4.0 + 2.0 + 2.0)
    assert scoring.score_without_tier(line) == pytest.approx(4.0)


def test_expected_tier_points_is_not_the_tier_of_the_expected_value(cfg):
    """The whole reason tiers are modelled as a distribution."""
    scoring = DstScoring.from_config(cfg)
    # A 50/50 shot at a shutout or 30 allowed averages 15 points allowed, whose
    # tier is worth 1.0 -- but the expectation of the tiers is (5 + -3)/2 = 1.0.
    # Use an asymmetric case where the two genuinely differ.
    probs = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]  # shutout or 46+
    assert scoring.expected_tier_points(probs) == pytest.approx((5.0 + -6.0) / 2)
    assert scoring.points_allowed_points(23) == -1.0  # tier of the mean, quite different


def test_expected_tier_points_rejects_wrong_length(cfg):
    with pytest.raises(ValueError):
        DstScoring.from_config(cfg).expected_tier_points([1.0, 0.0])


def test_tier_probabilities_from_samples(cfg):
    scoring = DstScoring.from_config(cfg)
    probs = tier_probabilities_from_samples([0, 0, 10, 50], scoring)
    assert probs[0] == pytest.approx(0.5)   # two shutouts
    assert probs[2] == pytest.approx(0.25)  # 7-13
    assert probs[-1] == pytest.approx(0.25)  # 46+
    assert sum(probs) == pytest.approx(1.0)


def test_tier_labels_and_values_line_up(cfg):
    scoring = DstScoring.from_config(cfg)
    assert scoring.tier_labels[0] == "0"
    assert scoring.tier_labels[1] == "1-6"
    assert scoring.tier_labels[-1] == "46+"
    assert len(scoring.tier_labels) == len(scoring.tier_values)


# ---------------------------------------------------------------------------
# Kicker
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "distance,bucket",
    [(20, "0_39"), (39, "0_39"), (40, "40_49"), (49, "40_49"), (50, "50_plus"), (66, "50_plus")],
)
def test_fg_bucket_boundaries(distance, bucket):
    assert fg_bucket(distance) == bucket


def test_kicker_scores_each_bucket(cfg):
    scoring = KickerScoring.from_config(cfg)
    line = KickerStatLine(fg_made_0_39=1, fg_made_40_49=1, fg_made_50_plus=1, pat_made=2)
    assert scoring.score(line) == pytest.approx(3 + 4 + 5 + 2)


def test_kicker_misses_are_penalised(cfg):
    scoring = KickerScoring.from_config(cfg)
    line = KickerStatLine(fg_made_0_39=2, fg_missed_40_49=1, pat_made=3, pat_missed=1)
    assert scoring.score(line) == pytest.approx(6 - 1 + 3 - 1)


def test_kicker_stat_line_totals():
    line = KickerStatLine(fg_made_0_39=1, fg_made_50_plus=2, fg_missed_0_39=1)
    assert line.fg_made == 3
    assert line.fg_attempted == 4


def test_kicker_expected_points_matches_hand_arithmetic(cfg):
    scoring = KickerScoring.from_config(cfg)
    expected = scoring.expected_points(
        expected_fg_attempts={"0_39": 1.0, "40_49": 1.0, "50_plus": 0.0},
        make_prob={"0_39": 1.0, "40_49": 0.5, "50_plus": 0.0},
        expected_pat_attempts=2.0,
        pat_make_prob=1.0,
    )
    # 1 * 3 + 1 * (0.5 * 4 + 0.5 * -1) + 2 * 1
    assert expected == pytest.approx(3 + 1.5 + 2)


def test_scoring_reads_from_config_not_code(cfg):
    """Changing config.yaml must change the scoring, with no code edit."""
    tweaked = dict(cfg.raw["kicker_scoring"])
    tweaked["fg_50_plus"] = 10.0
    other = Config(raw={**cfg.raw, "kicker_scoring": tweaked}, root=cfg.root)
    line = KickerStatLine(fg_made_50_plus=1)
    assert KickerScoring.from_config(cfg).score(line) == 5.0
    assert KickerScoring.from_config(other).score(line) == 10.0


def test_fg_buckets_registry_is_contiguous():
    assert [b[0] for b in FG_BUCKETS] == ["0_39", "40_49", "50_plus"]
    assert FG_BUCKETS[0][2] == 39 and FG_BUCKETS[1][1] == 40
    assert FG_BUCKETS[-1][2] is None
