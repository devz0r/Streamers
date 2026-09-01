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
    assert nyj["fg_made_50_59"] == 1     # 55 yards
    assert nyj["fg_made_60_plus"] == 0
    # The 42-yard miss and the blocked 48-yarder both land in 40-49.
    assert nyj["fg_missed_40_49"] == 2
    assert nyj["fg_missed_0_39"] == 0
    assert nyj["fg_missed_50_59"] == 0
    assert nyj["fg_missed_60_plus"] == 0
    assert nyj["pat_made"] == 1
    assert nyj["pat_missed"] == 1


def test_blocked_field_goal_counts_as_a_miss(cfg, toy_pbp):
    lines = kicker_game_lines(toy_pbp, cfg)
    nyj = lines[lines["team"] == "NYJ"].iloc[0]
    # Four attempts scored plus the blocked 48-yarder.
    assert nyj["fg_att"] == 5
    assert nyj["fg_made"] == 3
    assert nyj["fg_missed"] == 2


def test_kicker_fantasy_points_follow_the_active_profile(cfg, toy_pbp):
    """The same kicking line is worth different amounts in the two leagues."""
    espn = kicker_game_lines(toy_pbp, cfg.for_profile("espn"))
    espn_nyj = espn[espn["team"] == "NYJ"].iloc[0]
    # 3 + 4 + 5 made, two 40-49 misses at -1, one PAT, one missed PAT at -0.5.
    assert espn_nyj["fantasy_points"] == pytest.approx(3 + 4 + 5 - 1 - 1 + 1 - 0.5)

    yahoo = kicker_game_lines(toy_pbp, cfg.for_profile("yahoo"))
    yahoo_nyj = yahoo[yahoo["team"] == "NYJ"].iloc[0]
    # Yahoo penalises neither the missed field goals nor the missed PAT.
    assert yahoo_nyj["fantasy_points"] == pytest.approx(3 + 4 + 5 + 1)


def test_dst_counts_every_event_type(cfg, toy_pbp, toy_games):
    lines = dst_game_lines(toy_pbp, toy_games, cfg)
    buf = lines[lines["team"] == "BUF"].iloc[0]
    assert buf["sacks"] == 1
    assert buf["interceptions"] == 1
    assert buf["fumble_recoveries"] == 1
    assert buf["safeties"] == 1
    assert buf["blocked_kicks"] == 1          # the blocked field goal
    assert buf["fourth_down_stops"] == 1      # NYJ went for it and failed
    assert buf["points_allowed"] == 13


def test_dst_total_matches_the_scoring_rules(cfg, toy_pbp, toy_games):
    """Attribution is tested here; the arithmetic is tested in test_scoring."""
    from streamer.scoring import DstScoring, DstStatLine

    for profile in cfg.profile_names:
        bound = cfg.for_profile(profile)
        scoring = DstScoring.from_config(bound)
        lines = dst_game_lines(toy_pbp, toy_games, bound)
        buf = lines[lines["team"] == "BUF"].iloc[0]
        expected = scoring.score(DstStatLine(
            points_allowed=int(buf["points_allowed"]),
            yards_allowed=int(buf["yards_allowed"]),
            sacks=buf["sacks"], interceptions=buf["interceptions"],
            fumble_recoveries=buf["fumble_recoveries"], safeties=buf["safeties"],
            one_point_safeties=buf["one_point_safeties"],
            defensive_tds=buf["defensive_tds"], return_tds=buf["return_tds"],
            blocked_kicks=buf["blocked_kicks"],
            blocked_kick_tds=buf["blocked_kick_tds"],
            extra_points_returned=buf["extra_points_returned"],
            fourth_down_stops=buf["fourth_down_stops"],
        ))
        assert buf["fantasy_points"] == pytest.approx(expected), profile


def test_the_two_profiles_score_the_same_defense_differently(cfg, toy_pbp, toy_games):
    espn = dst_game_lines(toy_pbp, toy_games, cfg.for_profile("espn"))
    yahoo = dst_game_lines(toy_pbp, toy_games, cfg.for_profile("yahoo"))
    espn_buf = espn[espn["team"] == "BUF"].iloc[0]
    yahoo_buf = yahoo[yahoo["team"] == "BUF"].iloc[0]
    # Same counting stats, different sheets.
    assert espn_buf["sacks"] == yahoo_buf["sacks"]
    assert espn_buf["fantasy_points"] != yahoo_buf["fantasy_points"]


def test_yards_allowed_is_the_opponents_scrimmage_yards(cfg, toy_pbp, toy_games):
    lines = dst_game_lines(toy_pbp, toy_games, cfg)
    nyj = lines[lines["team"] == "NYJ"].iloc[0]
    # BUF ran three punt-drive plays at 10 yards plus a 5-yard scoring play.
    assert nyj["yards_allowed"] == 35


def test_sack_yardage_counts_against_the_offense(cfg, toy_pbp, toy_games):
    """A sack is negative yardage, so it reduces the yards the defense allows."""
    lines = dst_game_lines(toy_pbp, toy_games, cfg)
    buf = lines[lines["team"] == "BUF"].iloc[0]
    # NYJ's scrimmage plays include a -7 sack, so its total is below the raw sum.
    assert buf["yards_allowed"] < 35


def test_fourth_down_stop_needs_a_conversion_attempt(cfg, toy_pbp, toy_games):
    """Punts and field goals on fourth down are not stops."""
    lines = dst_game_lines(toy_pbp, toy_games, cfg)
    nyj = lines[lines["team"] == "NYJ"].iloc[0]
    assert nyj["fourth_down_stops"] == 0


def test_kickoff_return_td_is_credited_to_the_receiving_team(cfg, toy_pbp, toy_games):
    """On a kickoff, nflverse sets posteam to the RECEIVING team."""
    lines = dst_game_lines(toy_pbp, toy_games, cfg)
    den = lines[lines["team"] == "DEN"].iloc[0]
    # One defensive TD (pick six), one punt return TD, one kickoff return TD.
    assert den["defensive_tds"] == 1
    assert den["return_tds"] == 2


def test_offense_does_not_get_credited_with_its_own_touchdowns(cfg, toy_pbp, toy_games):
    lines = dst_game_lines(toy_pbp, toy_games, cfg)
    kc = lines[lines["team"] == "KC"].iloc[0]
    assert kc["defensive_tds"] == 0
    assert kc["return_tds"] == 0


def test_multiple_blocked_kick_types_sum_into_one_stat(cfg, toy_pbp, toy_games):
    """Blocked punts, FGs and PATs are three code paths and one column."""
    import pandas as pd

    pbp = toy_pbp.copy()
    extra = pbp.iloc[[0]].copy()
    extra["play_type"] = "punt"
    extra["punt_blocked"] = 1.0
    extra["field_goal_attempt"] = 0.0
    extra["field_goal_result"] = None
    extra["posteam"], extra["defteam"] = "NYJ", "BUF"
    pbp = pd.concat([pbp, extra], ignore_index=True)
    lines = dst_game_lines(pbp, toy_games, cfg)
    buf = lines[lines["team"] == "BUF"].iloc[0]
    assert buf["blocked_kicks"] == 2


def test_points_allowed_exclusion_is_configurable(cfg, toy_pbp, toy_games):
    from streamer.config import Config

    profiles = {k: dict(v) for k, v in cfg.raw["profiles"].items()}
    profiles[cfg.profile] = {
        **profiles[cfg.profile],
        "dst_scoring": {**cfg.dst_scoring, "points_allowed_excludes_opponent_dst_st": True},
    }
    other = Config(raw={**cfg.raw, "profiles": profiles}, root=cfg.root, profile=cfg.profile)

    default = dst_game_lines(toy_pbp, toy_games, cfg)
    excluded = dst_game_lines(toy_pbp, toy_games, other)
    kc_default = default[default["team"] == "KC"].iloc[0]["points_allowed"]
    kc_excluded = excluded[excluded["team"] == "KC"].iloc[0]["points_allowed"]
    # DEN scored 21, all of it on defense/special teams.
    assert kc_default == 21
    assert kc_excluded < kc_default


def test_empty_input_returns_empty_frames(cfg, toy_pbp, toy_games):
    empty = toy_pbp.iloc[0:0]
    assert kicker_game_lines(empty, cfg).empty
    assert dst_game_lines(empty, toy_games.iloc[0:0], cfg).empty
