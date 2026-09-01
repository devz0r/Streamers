"""Feature builders and the leak-free rolling machinery.

The single most important property here is that no feature for game *g* can
see game *g*'s own result. A leak would make the backtest look excellent and
the live tool useless, and it would be invisible without a test like this.
"""

from __future__ import annotations

import pandas as pd
import pytest

from streamer.features.rolling import (
    GAME_DECAY,
    RateSpec,
    add_shrunk_rate,
    bayesian_update,
    decayed_per_game,
)


def _series(n: int, values: list[float], team: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024] * n,
            "week": list(range(1, n + 1)),
            "game_id": [f"g{i}" for i in range(1, n + 1)],
            "team": [team] * n,
            "num": values,
            "den": [1.0] * n,
        }
    )


def test_first_game_uses_no_own_history():
    frame = _series(3, [10.0, 0.0, 0.0])
    out = add_shrunk_rate(frame, RateSpec("rate", "num", "den", 6.0), ["team"])
    # Row 0 has zero prior observations of its own.
    assert out.loc[0, "rate__n"] == 0.0


def test_a_rate_never_sees_its_own_row():
    """A huge value in the final game must not move that game's own feature."""
    quiet = _series(6, [1.0] * 6)
    spike = quiet.copy()
    spike.loc[5, "num"] = 1000.0
    spec = RateSpec("rate", "num", "den", 6.0)
    a = add_shrunk_rate(quiet, spec, ["team"]).loc[5, "rate"]
    b = add_shrunk_rate(spike, spec, ["team"]).loc[5, "rate"]
    assert a == pytest.approx(b)


def test_history_does_move_later_rows():
    quiet = _series(6, [1.0] * 6)
    spike = quiet.copy()
    spike.loc[0, "num"] = 100.0
    spec = RateSpec("rate", "num", "den", 6.0)
    a = add_shrunk_rate(quiet, spec, ["team"]).loc[5, "rate"]
    b = add_shrunk_rate(spike, spec, ["team"]).loc[5, "rate"]
    assert b > a


def test_shrinkage_pulls_small_samples_toward_the_league():
    """One extreme game should not produce an extreme rate."""
    frame = pd.concat(
        [_series(8, [1.0] * 8, team=t) for t in ("AAA", "BBB", "CCC")], ignore_index=True
    )
    frame.loc[frame["team"] == "AAA", "num"] = 9.0
    frame = frame.sort_values(["week", "team"]).reset_index(drop=True)
    out = add_shrunk_rate(frame, RateSpec("rate", "num", "den", 6.0), ["team"])
    aaa_second = out[(out["team"] == "AAA")].iloc[1]["rate"]
    # AAA's raw rate is 9.0, but after one game it must sit far below that.
    assert 1.0 < aaa_second < 5.0


def test_more_evidence_moves_a_team_further_from_the_prior():
    frame = pd.concat(
        [_series(12, [1.0] * 12, team=t) for t in ("AAA", "BBB", "CCC")], ignore_index=True
    )
    frame.loc[frame["team"] == "AAA", "num"] = 9.0
    frame = frame.sort_values(["week", "team"]).reset_index(drop=True)
    out = add_shrunk_rate(frame, RateSpec("rate", "num", "den", 6.0), ["team"])
    aaa = out[out["team"] == "AAA"].reset_index(drop=True)
    assert aaa.loc[10, "rate"] > aaa.loc[2, "rate"]


def test_future_rows_receive_priors_but_do_not_contribute():
    """An unplayed game must observe history without polluting it."""
    frame = _series(4, [5.0, 5.0, 0.0, 0.0])
    frame["is_future"] = [0, 0, 1, 0]
    spec = RateSpec("rate", "num", "den", 6.0)
    out = add_shrunk_rate(frame, spec, ["team"])
    # Rows 2 (future) and 3 both see exactly the two completed games.
    assert out.loc[2, "rate__n"] == pytest.approx(out.loc[3, "rate__n"])


def test_season_boundary_decays_the_carryover():
    frame = _series(6, [5.0] * 6)
    frame.loc[3:, "season"] = 2025
    out = add_shrunk_rate(frame, RateSpec("rate", "num", "den", 6.0), ["team"])
    # The first row of the new season carries less accumulated evidence than
    # the last row of the old one.
    assert out.loc[3, "rate__n"] < out.loc[2, "rate__n"] + 1.0


def test_recent_games_weigh_more_than_old_ones():
    early = _series(10, [9.0] + [1.0] * 9)
    late = _series(10, [1.0] * 9 + [9.0])
    spec = RateSpec("rate", "num", "den", 6.0)
    # Compare the state each series hands to a hypothetical 11th game.
    a = add_shrunk_rate(early, spec, ["team"])
    b = add_shrunk_rate(late, spec, ["team"])
    assert b.loc[9, "rate"] < a.loc[9, "rate"]  # the spike is still in b's future
    assert GAME_DECAY < 1.0


def test_decayed_per_game_is_also_leak_free():
    quiet = _series(6, [2.0] * 6)
    spike = quiet.copy()
    spike.loc[5, "num"] = 99.0
    a = decayed_per_game(quiet, "num", ["team"], "avg").loc[5, "avg"]
    b = decayed_per_game(spike, "num", ["team"], "avg").loc[5, "avg"]
    assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# Bayesian updating
# ---------------------------------------------------------------------------
def test_bayesian_update_with_no_evidence_returns_the_prior():
    assert bayesian_update(2.5, 8.0, 0.0, 0.0) == pytest.approx(2.5)


def test_bayesian_update_moves_gradually():
    prior, strength = 2.0, 8.0
    after_one = bayesian_update(prior, strength, 6.0, 1.0)
    after_eight = bayesian_update(prior, strength, 48.0, 8.0)
    assert prior < after_one < after_eight < 6.0
    # One game of evidence against a 8-game prior moves ~1/9th of the way.
    assert after_one == pytest.approx((2.0 * 8 + 6.0) / 9)


def test_bayesian_update_converges_with_overwhelming_evidence():
    assert bayesian_update(2.0, 8.0, 6.0 * 500, 500.0) == pytest.approx(6.0, abs=0.1)


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------
def test_kicker_features_include_every_registered_column(cfg, toy_pbp, toy_games):
    from streamer.features.build import KICKER_FEATURES, build_kicker_features

    out = build_kicker_features(toy_pbp, toy_games, cfg)
    assert not out.empty
    for column in KICKER_FEATURES:
        assert column in out.columns, column
        assert out[column].notna().all(), column


def test_dst_features_include_every_registered_column(cfg, toy_pbp, toy_games):
    from streamer.features.build import DST_FEATURES, build_dst_features

    out = build_dst_features(toy_pbp, toy_games, cfg)
    assert not out.empty
    for column in DST_FEATURES:
        assert column in out.columns, column
        assert out[column].notna().all(), column


def test_dst_features_carry_the_targets(cfg, toy_pbp, toy_games):
    from streamer.features.build import build_dst_features

    out = build_dst_features(toy_pbp, toy_games, cfg)
    for column in ("fantasy_points", "big_play_points", "tier_points", "points_allowed"):
        assert column in out.columns


def test_implied_totals_are_derived_correctly(cfg, toy_games):
    # NYJ: total 44, spread -2 (2-point underdog at home) -> 21.0 implied.
    nyj = toy_games[toy_games["team"] == "NYJ"].iloc[0]
    assert nyj["team_implied_total"] == pytest.approx(21.0)
    assert nyj["opp_implied_total"] == pytest.approx(23.0)
    assert nyj["team_implied_total"] + nyj["opp_implied_total"] == pytest.approx(44.0)


def test_every_feature_has_a_human_label():
    from streamer.features.build import DST_FEATURES, FACTOR_LABELS, KICKER_FEATURES

    for column in set(KICKER_FEATURES) | set(DST_FEATURES):
        assert column in FACTOR_LABELS, f"{column} needs an entry in FACTOR_LABELS"


def test_vegas_anchor_columns_are_real_features():
    from streamer.features.build import DST_FEATURES, KICKER_FEATURES, VEGAS_ANCHOR

    assert set(VEGAS_ANCHOR["K"]).issubset(set(KICKER_FEATURES))
    assert set(VEGAS_ANCHOR["DST"]).issubset(set(DST_FEATURES))


# ---------------------------------------------------------------------------
# Placeholder rows for unplayed games
# ---------------------------------------------------------------------------
def _future_slate(toy_games):
    """Week 2 fixtures for the same teams, which have not been played."""
    nxt = toy_games[["season", "game_id", "team", "opponent"]].drop_duplicates().copy()
    nxt["week"] = 2
    # Both teams in a game must keep sharing one game_id, or the opponent
    # join silently produces nothing.
    nxt["game_id"] = nxt["game_id"].str.replace("_01_", "_02_", regex=False)
    return nxt[["season", "week", "game_id", "team", "opponent"]]


def _games_with_upcoming(toy_games):
    """The played week plus a scheduled, unplayed week 2.

    Real runs always have the upcoming fixtures (with lines, without scores) in
    the games frame, because that is what the schedule feed carries.
    """
    upcoming = toy_games.copy()
    upcoming["week"] = 2
    upcoming["game_id"] = upcoming["game_id"].str.replace("_01_", "_02_", regex=False)
    upcoming["team_score"] = float("nan")
    upcoming["opp_score"] = float("nan")
    return pd.concat([toy_games, upcoming], ignore_index=True)


def test_unplayed_kicker_rows_have_no_target(cfg, toy_pbp, toy_games):
    """A placeholder row padded with 0.0 would read as a real scoreless game.

    That is the bug this guards: it would both contaminate training and let
    `update` "score" a week that has not happened.
    """
    from streamer.features.build import build_kicker_features

    future = _future_slate(toy_games)
    kickers = future.merge(
        pd.DataFrame({"team": ["NYJ", "BUF", "DEN", "KC"],
                      "player_id": ["K1", "K2", "K3", "K4"],
                      "player_name": ["T.Boot", "A.Leg", "B.Toe", "C.Foot"]}),
        on="team", how="inner",
    )
    out = build_kicker_features(toy_pbp, _games_with_upcoming(toy_games), cfg, future, kickers)
    upcoming = out[out["week"] == 2]
    assert not upcoming.empty
    assert upcoming["fantasy_points"].isna().all()


def test_unplayed_dst_rows_have_no_target(cfg, toy_pbp, toy_games):
    from streamer.features.build import build_dst_features

    out = build_dst_features(
        toy_pbp, _games_with_upcoming(toy_games), cfg, _future_slate(toy_games)
    )
    upcoming = out[out["week"] == 2]
    assert not upcoming.empty
    assert upcoming["fantasy_points"].isna().all()


def test_completed_only_rejects_placeholder_rows():
    """Second line of defence, independent of how the target got filled."""
    from streamer.pipeline import _completed_only

    frame = pd.DataFrame({
        "fantasy_points": [7.0, 0.0, float("nan"), 4.0],
        "is_future": [0, 1, 1, 0],
    })
    out = _completed_only(frame)
    assert len(out) == 2
    assert sorted(out["fantasy_points"]) == [4.0, 7.0]


def test_unplayed_rows_still_receive_priors(cfg, toy_pbp, toy_games):
    """Dropping the target must not mean dropping the features."""
    from streamer.features.build import DST_FEATURES, build_dst_features

    out = build_dst_features(
        toy_pbp, _games_with_upcoming(toy_games), cfg, _future_slate(toy_games)
    )
    upcoming = out[out["week"] == 2]
    for column in DST_FEATURES:
        assert upcoming[column].notna().all(), column
