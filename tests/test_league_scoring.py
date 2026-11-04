"""Every league's fantasy points on its own sheet.

The Yahoo league here pays 6 a touchdown pass and -1 an interception; nflverse
(and ESPN here) pays 4 and -2. Checked against Yahoo's own points."""

from __future__ import annotations

import pandas as pd

from streamer.config import get_config
from streamer.scoring import (
    EXPECTED_COLUMNS,
    STAT_COLUMNS,
    offense_deltas,
    rescore,
    rescore_feed,
    scoring_tag,
)


def test_each_league_reads_its_own_sheet():
    espn = get_config().for_profile("espn")._profile["offense_scoring"]
    yahoo = get_config().for_profile("yahoo")._profile["offense_scoring"]
    assert offense_deltas(espn) == {} and scoring_tag(espn) == ""
    assert offense_deltas(yahoo) == {"pass_td": 2.0, "interception": 1.0}
    assert scoring_tag(yahoo) and scoring_tag(yahoo) != scoring_tag(espn)


def test_yahoo_points_match_the_app_for_josh_allen_and_joe_burrow():
    yahoo = get_config().for_profile("yahoo")._profile["offense_scoring"]
    # nflverse's standard points and the stats behind them; Yahoo's app.
    games = pd.DataFrame([
        # standard, pass TD, INT, Yahoo app
        (26.56, 2, 1, 31.56),     # Allen week 8
        (17.46, 0, 2, 19.46),     # Allen week 3
        (39.48, 5, 3, 52.48),     # Burrow week 8
        (24.60, 4, 0, 32.60),     # Burrow week 5
    ], columns=["fantasy_points_ppr", "passing_tds", "passing_interceptions", "app"])
    got = rescore(games, yahoo, "fantasy_points_ppr", STAT_COLUMNS)
    assert (got - games["app"]).abs().max() < 1e-9
    espn = get_config().for_profile("espn")._profile["offense_scoring"]
    assert (rescore(games, espn, "fantasy_points_ppr", STAT_COLUMNS) == games["fantasy_points_ppr"]).all()


def test_expected_points_follow_the_sheet_and_missing_stays_missing():
    yahoo = get_config().for_profile("yahoo")._profile["offense_scoring"]
    opp = pd.DataFrame({"total_fantasy_points_exp": [20.0, None], "pass_touchdown_exp": [1.5, 1.0],
                        "pass_interception_exp": [0.8, 0.5]})
    got = rescore(opp, yahoo, "total_fantasy_points_exp", EXPECTED_COLUMNS)
    assert abs(got[0] - (20.0 + 2 * 1.5 + 0.8)) < 1e-9 and pd.isna(got[1])


def test_a_feeds_projection_is_rescored_or_left_out():
    yahoo = get_config().for_profile("yahoo")._profile["offense_scoring"]
    assert rescore_feed(20.0, {"pass_td": 1.6, "pass_int": 0.8}, yahoo, "QB") == 20.0 + 3.2 + 0.8
    assert rescore_feed(20.0, {"pass_tds": 1.6, "pass_ints": 0.8}, yahoo, "QB") == 20.0 + 3.2 + 0.8
    assert rescore_feed(20.0, {}, yahoo, "QB") is None            # cannot rescore a passer: no number
    assert rescore_feed(12.0, {}, yahoo, "RB") == 12.0             # a back throws no touchdowns
    assert rescore_feed(20.0, {}, {"pass_td": 4.0}, "QB") == 20.0  # the standard sheet: as sent


def test_the_refresh_flags_points_that_differ_from_the_platforms(caplog):
    from types import SimpleNamespace as NS

    from streamer.roster import scoring_check

    hist = pd.DataFrame([{"player_id": f"q{i}", "season": 2026, "week": w, "fantasy_points_ppr": 20.0}
                         for i in range(6) for w in (1, 2, 3)])
    players = [NS(nfl_id=f"q{i}", position="QB", season_avg=24.0) for i in range(6)]
    snap = NS(season=2026, profile="yahoo", all_players=lambda: players)
    assert scoring_check.check(snap, hist) == {"QB": 4.0}
    assert "differ from the platform's" in caplog.text
    for p in players:
        p.season_avg = 20.0
    assert scoring_check.check(snap, hist) == {"QB": 0.0}


def test_yahoos_scoring_modifiers_are_read_from_its_settings():
    from streamer.league.yahoo import scoring_from_settings

    raw = {"fantasy_content": {"league": [{"league_key": "x"}, {"settings": [{
        "stat_categories": {"stats": [{"stat": {"stat_id": 5, "display_name": "Pass TD", "position_type": "O"}},
                                      {"stat": {"stat_id": 6, "display_name": "Int", "position_type": "O"}},
                                      {"stat": {"stat_id": 4, "display_name": "Pass Yds", "position_type": "O"}},
                                      {"stat": {"stat_id": 11, "display_name": "Rec", "position_type": "O"}}]},
        "stat_modifiers": {"stats": [{"stat": {"stat_id": 5, "value": "6"}}, {"stat": {"stat_id": 6, "value": "-1"}},
                                     {"stat": {"stat_id": 4, "value": "0.04"}}, {"stat": {"stat_id": 11, "value": "1"}},
                                     {"stat": {"stat_id": 99, "value": "3"}}]}}]}]}}
    assert scoring_from_settings(raw) == {"pass_td": 6.0, "interception": -1.0, "pass_yards": 0.04, "reception": 1.0}
    assert scoring_from_settings({}) == {} and scoring_from_settings(None) == {}


def test_the_refresh_flags_a_config_sheet_that_differs_from_yahoos(caplog):
    from types import SimpleNamespace as NS

    from streamer.roster import scoring_check

    snap = NS(profile="yahoo", extra={"platform_scoring": {"pass_td": 6.0, "interception": -1.0, "reception": 1.0}})
    assert scoring_check.check_sheet(snap, {"pass_td": 4.0, "interception": -1.0, "reception": 1.0}) == [
        "pass_td: platform 6, config 4"]
    assert "differs from offense_scoring" in caplog.text
    assert scoring_check.check_sheet(snap, {"pass_td": 6.0, "interception": -1.0, "reception": 1.0}) == []
    assert scoring_check.check_sheet(NS(profile="espn", extra={}), {"pass_td": 4.0}) == []     # nothing reported


def test_kicker_and_defence_sheets_are_the_leagues_own():
    """Pinned to the scoring sheets of the two leagues, line by line (ESPN's
    sack is 2 here, not its default 2.5)."""
    e, y = get_config().for_profile("espn"), get_config().for_profile("yahoo")
    assert e.kicker_scoring == {"fg_0_39": 3.0, "fg_40_49": 4.0, "fg_50_59": 5.0, "fg_60_plus": 6.0,
                                "fg_missed_0_39": -1.0, "fg_missed_40_49": -1.0, "fg_missed_50_59": -1.0,
                                "fg_missed_60_plus": -1.0, "pat_made": 1.0, "pat_missed": -0.5}
    assert y.kicker_scoring == {"fg_0_39": 3.0, "fg_40_49": 4.0, "fg_50_59": 5.0, "fg_60_plus": 5.0,
                                "fg_missed_0_39": 0.0, "fg_missed_40_49": 0.0, "fg_missed_50_59": 0.0,
                                "fg_missed_60_plus": 0.0, "pat_made": 1.0, "pat_missed": 0.0}
    ed, yd = e.dst_scoring, y.dst_scoring
    assert (ed["sack"], ed["interception"], ed["fumble_recovery"], ed["safety"], ed["one_point_safety"],
            ed["blocked_kick"], ed["fumble_lost"]) == (2.0, 2.5, 2.0, 2.0, 1.0, 2.0, -2.0)
    assert [list(t) for t in ed["points_allowed_tiers"]] == [[0, 0, 6.0], [1, 6, 4.0], [7, 13, 3.0], [14, 17, 1.0],
                                                           [18, 27, 0.0], [28, 34, -1.0], [35, 45, -3.0],
                                                           [46, None, -4.0]]
    assert [t[2] for t in ed["yards_allowed_tiers"]] == [5.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0]
    assert (yd["sack"], yd["interception"], yd["fumble_recovery"], yd["safety"], yd["blocked_kick"],
            yd["fourth_down_stop"], yd["extra_point_returned"]) == (1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0)
    assert [list(t) for t in yd["points_allowed_tiers"]] == [[0, 0, 10.0], [1, 6, 7.0], [7, 13, 4.0], [14, 20, 1.0],
                                                           [21, 27, 0.0], [28, 34, -1.0], [35, None, -4.0]]
    assert yd["yards_allowed_tiers"] is None


def test_a_platform_average_of_zero_is_missing_not_a_mismatch(caplog):
    from types import SimpleNamespace as NS

    from streamer.roster import scoring_check

    hist = pd.DataFrame([{"player_id": f"q{i}", "season": 2026, "week": w, "fantasy_points_ppr": 20.0}
                         for i in range(8) for w in (1, 2, 3)])
    players = ([NS(nfl_id=f"q{i}", position="QB", season_avg=20.0) for i in range(5)]
               + [NS(nfl_id=f"q{i}", position="QB", season_avg=0.0) for i in range(5, 8)])
    snap = NS(season=2026, profile="espn", all_players=lambda: players)
    assert scoring_check.check(snap, hist) == {"QB": 0.0}
    assert "differ" not in caplog.text


_YAHOO_SETTINGS = """<html><body>
<table><tr><th>Stat</th><th>League Value</th><th>Yahoo Default Value</th></tr>
<tr><td>Passing Yards</td><td>25 yards per point</td></tr>
<tr><td>Passing Touchdowns</td><td>6</td><td>4</td></tr>
<tr><td>Interceptions</td><td>-1</td></tr>
<tr><td>Receptions</td><td>1</td><td>0.5</td></tr>
<tr><td>Field Goals 0-19 Yards</td><td>3</td></tr>
<tr><td>Field Goals 50+ Yards</td><td>5</td></tr>
<tr><td>Point After Attempt Made</td><td>1</td></tr>
<tr><td>Sack</td><td>1</td></tr>
<tr><td>Points Allowed 0 points</td><td>10</td></tr>
<tr><td>Points Allowed 1-6 points</td><td>7</td></tr>
<tr><td>Points Allowed 35+ points</td><td>-4</td></tr>
<tr><td>4th Down Stops (Yahoo Default: 0)</td><td>1</td><td>0</td></tr>
<tr><td>Extra Point Returned</td><td>2</td></tr>
<tr><td>Roster Positions</td><td>QB, WR</td></tr>
</table></body></html>"""


def test_yahoos_settings_page_scoring_is_read_and_matches_the_config(caplog):
    import logging
    from types import SimpleNamespace as NS

    from streamer.league.yahoo_web import parse_scoring
    from streamer.roster import scoring_check

    sheet = parse_scoring(_YAHOO_SETTINGS)
    assert sheet == {"pass_yards": 0.04, "pass_td": 6.0, "interception": -1.0, "reception": 1.0,
                     "k.fg_0_19": 3.0, "k.fg_50_plus": 5.0, "k.pat_made": 1.0, "dst.sack": 1.0,
                     "dst.pa_0_0": 10.0, "dst.pa_1_6": 7.0, "dst.pa_35_plus": -4.0,
                     "dst.fourth_down_stop": 1.0, "dst.extra_point_returned": 2.0}
    profile = get_config().for_profile("yahoo")._profile
    caplog.set_level(logging.INFO)
    assert scoring_check.check_sheet(NS(profile="yahoo", rules={"platform_scoring": sheet}, extra={}), profile) == []
    assert "matches" in caplog.text
    changed = dict(sheet, **{"dst.sack": 2.0, "k.fg_50_plus": 6.0})
    assert scoring_check.check_sheet(NS(profile="yahoo", rules={"platform_scoring": changed}, extra={}), profile) == [
        "dst.sack: platform 2, config 1", "k.fg_50_plus: platform 6, config 5"]


def test_every_yahoo_sheet_line_is_covered_by_the_config():
    from streamer.roster.scoring_check import flatten

    flat = flatten(get_config().for_profile("yahoo")._profile)
    assert {k: flat[k] for k in ("k.fg_0_19", "k.fg_20_29", "k.fg_30_39", "k.fg_40_49", "k.fg_50_plus",
                                 "dst.pa_14_20", "dst.pa_21_27", "dst.return_td")} == {
        "k.fg_0_19": 3.0, "k.fg_20_29": 3.0, "k.fg_30_39": 3.0, "k.fg_40_49": 4.0, "k.fg_50_plus": 5.0,
        "dst.pa_14_20": 1.0, "dst.pa_21_27": 0.0, "dst.return_td": 6.0}


def test_without_a_header_the_first_value_after_the_label_is_the_leagues():
    from streamer.league.yahoo_web import parse_scoring

    html = "<table><tr><td>Passing Touchdowns</td><td>6</td><td>4</td></tr></table>"
    assert parse_scoring(html) == {"pass_td": 6.0}


def test_a_kickers_game_log_points_follow_each_leagues_sheet():
    """Harrison Butker, 2026 weeks 2 and 6, by hand: ESPN scores a miss -1 and a
    missed extra point -0.5; Yahoo scores neither, and a 50+ kick 5 on both."""
    from streamer.roster.game_log import _kicker_points

    cols = ["position", "fantasy_points_ppr", "fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49",
            "fg_made_50_59", "fg_made_60_", "fg_missed_0_19", "fg_missed_20_29", "fg_missed_30_39",
            "fg_missed_40_49", "fg_missed_50_59", "fg_missed_60_", "pat_made", "pat_missed"]
    wk2 = ["K", 0.0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0]     # 46 28 31 40 made, 51 missed, 3 XP
    wk6 = ["K", 0.0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1]     # 44 50 30 made, 52 missed, 1 of 2 XP
    df = pd.DataFrame([wk2, wk6, ["QB"] + [0.0] * 15], columns=cols)
    e = _kicker_points(df, get_config().for_profile("espn"))
    y = _kicker_points(df, get_config().for_profile("yahoo"))
    assert list(e) == [16.0, 11.5, 0.0] and list(y) == [17.0, 13.0, 0.0]
