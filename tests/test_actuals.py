"""Deriving realised stat lines from play-by-play.

These lock in the attribution rules that are easy to get subtly wrong:
kickoff vs punt ``posteam`` conventions, blocked kicks counting as misses, and
the fact that several distinct events all roll up into one stat.
"""

from __future__ import annotations

import pytest

from streamer.actuals import dst_game_lines, kicker_game_lines


def test_kicker_line_buckets_every_attempt(cfg, toy_pbp):
    lines = kicker_game_lines(toy_pbp, cfg)
    nyj = lines[lines["team"] == "NYJ"].iloc[0]
    assert nyj["fg_made_0_39"] == 1      # 30 yards
    assert nyj["fg_made_40_49"] == 1     # 45 yards
    assert nyj["fg_made_50_plus"] == 1   # 55 yards
    # The 42-yard miss and the blocked 48-yarder both land in 40-49.
    assert nyj["fg_missed_40_49"] == 2
    assert nyj["fg_missed_0_39"] == 0
    assert nyj["fg_missed_50_plus"] == 0
    assert nyj["pat_made"] == 1
    assert nyj["pat_missed"] == 1


def test_blocked_field_goal_counts_as_a_miss(cfg, toy_pbp):
    lines = kicker_game_lines(toy_pbp, cfg)
    nyj = lines[lines["team"] == "NYJ"].iloc[0]
    # Four attempts scored plus the blocked 48-yarder.
    assert nyj["fg_att"] == 5
    assert nyj["fg_made"] == 3
    assert nyj["fg_missed"] == 2


def test_kicker_fantasy_points_match_the_scoring_rules(cfg, toy_pbp):
    lines = kicker_game_lines(toy_pbp, cfg)
    nyj = lines[lines["team"] == "NYJ"].iloc[0]
    # 3 + 4 + 5 made, two 40-49 misses at -1, one PAT made, one missed.
    expected = 3 + 4 + 5 + (-1) + (-1) + 1 + (-1)
    assert nyj["fantasy_points"] == pytest.approx(expected)


def test_dst_counts_every_event_type(cfg, toy_pbp, toy_games):
    lines = dst_game_lines(toy_pbp, toy_games, cfg)
    buf = lines[lines["team"] == "BUF"].iloc[0]
    assert buf["sacks"] == 1
    assert buf["interceptions"] == 1
    assert buf["fumble_recoveries"] == 1
    assert buf["safeties"] == 1
    assert buf["blocked_kicks"] == 1        # the blocked field goal
    assert buf["points_allowed"] == 13
    # tier(13) = 3, plus 1 + 2 + 2 + 2 + 2
    assert buf["fantasy_points"] == pytest.approx(3 + 1 + 2 + 2 + 2 + 2)


def test_kickoff_return_td_is_credited_to_the_receiving_team(cfg, toy_pbp, toy_games):
    """On a kickoff, nflverse sets posteam to the RECEIVING team."""
    lines = dst_game_lines(toy_pbp, toy_games, cfg)
    den = lines[lines["team"] == "DEN"].iloc[0]
    # One defensive TD (pick six), one punt return TD, one kickoff return TD.
    assert den["defensive_tds"] == 1
    assert den["return_tds"] == 2


def test_defensive_and_return_tds_are_worth_six_each(cfg, toy_pbp, toy_games):
    lines = dst_game_lines(toy_pbp, toy_games, cfg)
    den = lines[lines["team"] == "DEN"].iloc[0]
    assert den["points_allowed"] == 0
    # shutout (5) + 3 TDs (18) + the INT on the pick-six (2)
    assert den["fantasy_points"] == pytest.approx(5 + 18 + 2)


def test_offense_does_not_get_credited_with_its_own_touchdowns(cfg, toy_pbp, toy_games):
    lines = dst_game_lines(toy_pbp, toy_games, cfg)
    kc = lines[lines["team"] == "KC"].iloc[0]
    assert kc["defensive_tds"] == 0
    assert kc["return_tds"] == 0


def test_multiple_blocked_kick_types_sum_into_one_stat(cfg, toy_pbp, toy_games):
    """Blocked punts, FGs and PATs are three code paths and one column."""
    pbp = toy_pbp.copy()
    extra = pbp.iloc[[0]].copy()
    extra["play_type"] = "punt"
    extra["punt_blocked"] = 1.0
    extra["field_goal_attempt"] = 0.0
    extra["field_goal_result"] = None
    extra["posteam"], extra["defteam"] = "NYJ", "BUF"
    pbp = __import__("pandas").concat([pbp, extra], ignore_index=True)
    lines = dst_game_lines(pbp, toy_games, cfg)
    buf = lines[lines["team"] == "BUF"].iloc[0]
    assert buf["blocked_kicks"] == 2


def test_points_allowed_exclusion_is_configurable(cfg, toy_pbp, toy_games):
    from dataclasses import replace

    tweaked = dict(cfg.raw["dst_scoring"])
    tweaked["points_allowed_excludes_opponent_dst_st"] = True
    other = replace(cfg, raw={**cfg.raw, "dst_scoring": tweaked})
    default = dst_game_lines(toy_pbp, toy_games, cfg)
    excluded = dst_game_lines(toy_pbp, toy_games, other)
    kc_default = default[default["team"] == "KC"].iloc[0]["points_allowed"]
    kc_excluded = excluded[excluded["team"] == "KC"].iloc[0]["points_allowed"]
    # DEN scored 21, all of it on defense/special teams, so excluding it drops
    # what KC's D/ST is charged with.
    assert kc_default == 21
    assert kc_excluded < kc_default


def test_empty_input_returns_empty_frames(cfg, toy_pbp, toy_games):
    empty = toy_pbp.iloc[0:0]
    assert kicker_game_lines(empty, cfg).empty
    assert dst_game_lines(empty, toy_games.iloc[0:0], cfg).empty
