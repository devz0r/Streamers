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
