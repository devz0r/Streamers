"""Exact scoring tests for every configured league profile.

The tier ladders are where an off-by-one silently corrupts every projection,
every backtest and every published ranking, so both profiles' ladders are
asserted at every boundary rather than sampled. Everything is driven from
``config.yaml`` -- if a scoring value changes there, these tests are what
notices.
"""

from __future__ import annotations

import pytest

from streamer.config import Config, load_config
from streamer.scoring import (
    FG_BUCKETS,
    FG_FEATURE_BUCKETS,
    DstScoring,
    DstStatLine,
    KickerScoring,
    KickerStatLine,
    fg_bucket,
    tier_probabilities_from_samples,
)

PROFILES = ("espn", "yahoo")


@pytest.fixture(params=PROFILES)
def any_profile(request, cfg):
    """Each configured profile in turn."""
    return cfg.for_profile(request.param)


@pytest.fixture
def espn(cfg):
    return cfg.for_profile("espn")


@pytest.fixture
def yahoo(cfg):
    return cfg.for_profile("yahoo")


# ---------------------------------------------------------------------------
# Points-allowed ladders, boundary by boundary
# ---------------------------------------------------------------------------
ESPN_PA = [
    (0, 6.0),
    (1, 4.0), (3, 4.0), (6, 4.0),
    (7, 3.0), (10, 3.0), (13, 3.0),
    (14, 1.0), (16, 1.0), (17, 1.0),
    (18, 0.0), (23, 0.0), (27, 0.0),
    (28, -1.0), (31, -1.0), (34, -1.0),
    (35, -3.0), (40, -3.0), (45, -3.0),
    (46, -4.0), (59, -4.0), (100, -4.0),
]

YAHOO_PA = [
    (0, 10.0),
    (1, 7.0), (3, 7.0), (6, 7.0),
    (7, 4.0), (10, 4.0), (13, 4.0),
    (14, 1.0), (17, 1.0), (20, 1.0),
    (21, 0.0), (24, 0.0), (27, 0.0),
    (28, -1.0), (31, -1.0), (34, -1.0),
    (35, -4.0), (49, -4.0), (100, -4.0),
]


@pytest.mark.parametrize("points_allowed,expected", ESPN_PA)
def test_espn_points_allowed_ladder(espn, points_allowed, expected):
    assert DstScoring.from_config(espn).points_allowed_points(points_allowed) == expected


@pytest.mark.parametrize("points_allowed,expected", YAHOO_PA)
def test_yahoo_points_allowed_ladder(yahoo, points_allowed, expected):
    assert DstScoring.from_config(yahoo).points_allowed_points(points_allowed) == expected


def test_the_two_profiles_really_do_differ(espn, yahoo):
    """Guards against both profiles accidentally reading the same block."""
    e, y = DstScoring.from_config(espn), DstScoring.from_config(yahoo)
    assert e.points_allowed_points(0) == 6.0 and y.points_allowed_points(0) == 10.0
    assert e.sack == 2.5 and y.sack == 1.0
    assert e.fourth_down_stop == 0.0 and y.fourth_down_stop == 1.0
    assert e.scores_yards_allowed and not y.scores_yards_allowed


def test_ladders_are_monotone_and_total(any_profile):
    scoring = DstScoring.from_config(any_profile)
    values = [scoring.points_allowed_points(pa) for pa in range(0, 121)]
    assert len(values) == 121
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_negative_points_allowed_is_rejected(any_profile):
    with pytest.raises(ValueError):
        DstScoring.from_config(any_profile).points_allowed_points(-1)


# ---------------------------------------------------------------------------
# Yards-allowed ladder (ESPN only)
# ---------------------------------------------------------------------------
ESPN_YARDS = [
    (0, 5.0), (99, 5.0),
    (100, 3.0), (199, 3.0),
    (200, 2.0), (299, 2.0),
    (300, 1.0), (349, 1.0),
    (350, -1.0), (399, -1.0),
    (400, -2.0), (449, -2.0),
    (450, -3.0), (499, -3.0),
    (500, -4.0), (549, -4.0),
    (550, -5.0), (700, -5.0),
]


@pytest.mark.parametrize("yards,expected", ESPN_YARDS)
def test_espn_yards_allowed_ladder(espn, yards, expected):
    assert DstScoring.from_config(espn).yards_allowed_points(yards) == expected


def test_yahoo_ignores_yards_allowed(yahoo):
    scoring = DstScoring.from_config(yahoo)
    assert scoring.yards_allowed_tiers is None
    # Scoring a line must not blow up just because yards are present.
    assert scoring.yards_allowed_points(450) == 0.0
    assert scoring.yards_tier_values == ()
    with pytest.raises(ValueError):
        scoring.yards_tier_index(300)


def test_yards_ladder_is_monotone(espn):
    scoring = DstScoring.from_config(espn)
    values = [scoring.yards_allowed_points(y) for y in range(0, 800, 10)]
    assert all(a >= b for a, b in zip(values, values[1:]))


# ---------------------------------------------------------------------------
# Ladder validation
# ---------------------------------------------------------------------------
def _with_dst(cfg: Config, **overrides) -> Config:
    raw = {k: dict(v) for k, v in cfg.raw["profiles"].items()}
    raw[cfg.profile] = {**raw[cfg.profile], "dst_scoring": {
        **cfg.dst_scoring, **overrides
    }}
    return Config(raw={**cfg.raw, "profiles": raw}, root=cfg.root, profile=cfg.profile)


def test_ladder_validation_rejects_gaps(espn):
    bad = _with_dst(espn, points_allowed_tiers=[[0, 0, 5.0], [2, None, 4.0]])
    with pytest.raises(ValueError, match="contiguous"):
        DstScoring.from_config(bad)


def test_ladder_validation_requires_an_open_top(espn):
    bad = _with_dst(espn, points_allowed_tiers=[[0, 0, 5.0], [1, 40, 4.0]])
    with pytest.raises(ValueError, match="open-ended"):
        DstScoring.from_config(bad)


def test_yards_ladder_is_validated_too(espn):
    bad = _with_dst(espn, yards_allowed_tiers=[[0, 99, 5.0], [200, None, 1.0]])
    with pytest.raises(ValueError, match="contiguous"):
        DstScoring.from_config(bad)


# ---------------------------------------------------------------------------
# Full D/ST stat lines
# ---------------------------------------------------------------------------
def test_espn_scores_a_realistic_line(espn):
    scoring = DstScoring.from_config(espn)
    line = DstStatLine(points_allowed=10, yards_allowed=310, sacks=4,
                       interceptions=2, fumble_recoveries=1, fourth_down_stops=2)
    # PA 3 + yards 1 + 4 sacks at 2.5 + 2 INT at 2.5 + 1 FR at 2; stops score 0.
    assert scoring.score(line) == pytest.approx(3 + 1 + 10 + 5 + 2)


def test_yahoo_scores_the_same_line_differently(yahoo):
    scoring = DstScoring.from_config(yahoo)
    line = DstStatLine(points_allowed=10, yards_allowed=310, sacks=4,
                       interceptions=2, fumble_recoveries=1, fourth_down_stops=2)
    # PA 4 + no yards + 4 sacks at 1 + 2 INT at 2 + 1 FR at 2 + 2 stops at 1.
    assert scoring.score(line) == pytest.approx(4 + 4 + 4 + 2 + 2)


def test_every_component_is_scored(any_profile):
    scoring = DstScoring.from_config(any_profile)
    line = DstStatLine(
        points_allowed=0, yards_allowed=50, sacks=1, interceptions=1,
        fumble_recoveries=1, safeties=1, one_point_safeties=1, defensive_tds=1,
        return_tds=1, blocked_kicks=1, blocked_kick_tds=1, extra_points_returned=1,
        fourth_down_stops=1,
    )
    expected = (
        scoring.points_allowed_points(0) + scoring.yards_allowed_points(50)
        + scoring.sack + scoring.interception + scoring.fumble_recovery
        + scoring.safety + scoring.one_point_safety + scoring.defensive_td
        + scoring.return_td + scoring.blocked_kick + scoring.blocked_kick_td
        + scoring.extra_point_returned + scoring.fourth_down_stop
    )
    assert scoring.score(line) == pytest.approx(expected)


def test_big_play_component_excludes_both_ladders(any_profile):
    scoring = DstScoring.from_config(any_profile)
    line = DstStatLine(points_allowed=3, yards_allowed=280, sacks=2, interceptions=1)
    ladders = scoring.points_allowed_points(3) + scoring.yards_allowed_points(280)
    assert scoring.score(line) - scoring.score_big_plays(line) == pytest.approx(ladders)
    assert scoring.score_big_plays(line) == pytest.approx(
        2 * scoring.sack + scoring.interception
    )


# ---------------------------------------------------------------------------
# Tier probabilities
# ---------------------------------------------------------------------------
def test_expected_tier_points_is_not_the_tier_of_the_expected_value(espn):
    """Why the ladder has to be modelled as a distribution."""
    scoring = DstScoring.from_config(espn)
    probs = [0.5] + [0.0] * 6 + [0.5]   # a shutout or a 46+ blowout
    assert scoring.expected_tier_points(probs) == pytest.approx((6.0 + -4.0) / 2)
    # The tier of the average outcome (23 points) is a different number.
    assert scoring.points_allowed_points(23) == 0.0


def test_expected_tier_points_rejects_a_wrong_length(any_profile):
    with pytest.raises(ValueError):
        DstScoring.from_config(any_profile).expected_tier_points([1.0, 0.0])


def test_tier_probabilities_from_samples(espn):
    scoring = DstScoring.from_config(espn)
    probs = tier_probabilities_from_samples([0, 0, 10, 50], scoring)
    assert probs[0] == pytest.approx(0.5)
    assert probs[2] == pytest.approx(0.25)
    assert probs[-1] == pytest.approx(0.25)
    assert sum(probs) == pytest.approx(1.0)


def test_tier_probabilities_from_samples_on_the_yards_ladder(espn):
    scoring = DstScoring.from_config(espn)
    probs = tier_probabilities_from_samples([50, 250, 250, 600], scoring, ladder="yards")
    assert probs[0] == pytest.approx(0.25)
    assert probs[2] == pytest.approx(0.5)
    assert probs[-1] == pytest.approx(0.25)


def test_expected_yards_tier_points(espn, yahoo):
    espn_scoring = DstScoring.from_config(espn)
    probs = [0.0] * 9
    probs[2] = 1.0  # certain to allow 200-299
    assert espn_scoring.expected_yards_tier_points(probs) == pytest.approx(2.0)
    # A profile without the ladder contributes nothing rather than raising.
    assert DstScoring.from_config(yahoo).expected_yards_tier_points([]) == 0.0


def test_tier_labels_line_up_with_values(any_profile):
    scoring = DstScoring.from_config(any_profile)
    assert len(scoring.tier_labels) == len(scoring.tier_values)
    assert scoring.tier_labels[0] == "0"
    assert scoring.tier_labels[-1].endswith("+")


# ---------------------------------------------------------------------------
# Kickers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "distance,bucket",
    [(20, "0_39"), (39, "0_39"), (40, "40_49"), (49, "40_49"),
     (50, "50_59"), (59, "50_59"), (60, "60_plus"), (66, "60_plus")],
)
def test_fg_scoring_bucket_boundaries(distance, bucket):
    assert fg_bucket(distance) == bucket


def test_espn_pays_a_premium_for_sixty_yards(espn):
    scoring = KickerScoring.from_config(espn)
    line = KickerStatLine(fg_made_0_39=1, fg_made_40_49=1, fg_made_50_59=1,
                          fg_made_60_plus=1, pat_made=2)
    assert scoring.score(line) == pytest.approx(3 + 4 + 5 + 6 + 2)


def test_yahoo_pays_the_same_for_fifty_and_sixty(yahoo):
    scoring = KickerScoring.from_config(yahoo)
    line = KickerStatLine(fg_made_0_39=1, fg_made_40_49=1, fg_made_50_59=1,
                          fg_made_60_plus=1, pat_made=2)
    assert scoring.score(line) == pytest.approx(3 + 4 + 5 + 5 + 2)


def test_espn_penalises_misses_and_yahoo_does_not(espn, yahoo):
    line = KickerStatLine(fg_made_0_39=2, fg_missed_40_49=1, fg_missed_50_59=1,
                          pat_made=3, pat_missed=1)
    # ESPN: 6 made, two misses at -1, 3 PATs, one missed PAT at -0.5.
    assert KickerScoring.from_config(espn).score(line) == pytest.approx(6 - 2 + 3 - 0.5)
    # Yahoo: misses are free.
    assert KickerScoring.from_config(yahoo).score(line) == pytest.approx(6 + 3)


def test_kicker_stat_line_totals():
    line = KickerStatLine(fg_made_0_39=1, fg_made_50_59=2, fg_made_60_plus=1,
                          fg_missed_0_39=1)
    assert line.fg_made == 4
    assert line.fg_missed == 1
    assert line.fg_attempted == 5


def test_kicker_expected_points_matches_hand_arithmetic(espn):
    scoring = KickerScoring.from_config(espn)
    expected = scoring.expected_points(
        expected_fg_attempts={"0_39": 1.0, "40_49": 1.0},
        make_prob={"0_39": 1.0, "40_49": 0.5},
        expected_pat_attempts=2.0,
        pat_make_prob=1.0,
    )
    # 1 * 3 + 1 * (0.5 * 4 + 0.5 * -1) + 2 * 1
    assert expected == pytest.approx(3 + 1.5 + 2)


# ---------------------------------------------------------------------------
# Bucket registries
# ---------------------------------------------------------------------------
def test_scoring_buckets_are_contiguous_and_open_topped():
    assert [b[0] for b in FG_BUCKETS] == ["0_39", "40_49", "50_59", "60_plus"]
    for (_n, _lo, hi), (_n2, lo2, _hi2) in zip(FG_BUCKETS, FG_BUCKETS[1:]):
        assert hi is not None and lo2 == hi + 1
    assert FG_BUCKETS[-1][2] is None


def test_feature_buckets_cover_every_scoring_bucket():
    """The coarse modelling buckets must partition the fine scoring ones."""
    covered = [b for _name, buckets in FG_FEATURE_BUCKETS for b in buckets]
    assert sorted(covered) == sorted(name for name, _lo, _hi in FG_BUCKETS)
    assert len(covered) == len(set(covered))


# ---------------------------------------------------------------------------
# Config-driven, not code-driven
# ---------------------------------------------------------------------------
def test_changing_config_changes_the_scoring(espn):
    tweaked = {k: dict(v) for k, v in espn.raw["profiles"].items()}
    tweaked["espn"] = {**tweaked["espn"],
                       "kicker_scoring": {**espn.kicker_scoring, "fg_60_plus": 99.0}}
    other = Config(raw={**espn.raw, "profiles": tweaked}, root=espn.root, profile="espn")
    line = KickerStatLine(fg_made_60_plus=1)
    assert KickerScoring.from_config(espn).score(line) == 6.0
    assert KickerScoring.from_config(other).score(line) == 99.0


def test_unknown_profile_is_rejected(cfg):
    with pytest.raises(ValueError, match="unknown scoring profile"):
        cfg.for_profile("draftkings")


def test_every_profile_loads(cfg):
    for name in cfg.profile_names:
        profile = cfg.for_profile(name)
        assert DstScoring.from_config(profile)
        assert KickerScoring.from_config(profile)
        assert profile.startable_rank > 0
        assert profile.league["teams"] > 0


def test_config_file_and_fixture_agree():
    """The fixture must not drift from the real config.yaml."""
    assert set(load_config().profile_names) == {"espn", "yahoo"}
