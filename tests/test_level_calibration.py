"""The top of a back's or receiver's weekly number is read as what such numbers scored."""

import numpy as np

from streamer.roster import level_calibration as lc

CONF = {"knot": 12.0, "positions": {"RB": {"a": -1.2, "b": 1.16, "c": -0.34}}}


def test_the_top_comes_down_the_middle_goes_up_a_little_and_the_order_never_changes():
    xs = np.arange(0.0, 30.01, 0.25)
    ys = np.array([lc.apply("RB", float(x), CONF) for x in xs])
    assert (np.diff(ys) >= 0).all()                           # a higher number never reads lower
    assert lc.apply("RB", 22.0, CONF) < 22.0 and lc.apply("RB", 12.0, CONF) > 12.0
    assert abs(lc.apply("RB", 12.0, CONF) - (-1.2 + 1.16 * 12)) < 1e-9
    assert abs(lc.apply("RB", 22.0, CONF) - (-1.2 + 1.16 * 22 - 0.34 * 10)) < 1e-9


def test_it_is_continuous_where_the_fit_stops_and_leaves_other_positions_alone():
    assert abs(lc.apply("RB", 3.0, CONF) - lc.apply("RB", 2.999, CONF)) < 0.01
    assert lc.apply("RB", 0.0, CONF) == 0.0 and 0 < lc.apply("RB", 1.5, CONF) < 1.5
    assert lc.apply("QB", 22.0, CONF) == 22.0 and lc.apply("TE", 9.0, CONF) == 9.0


def test_the_shipped_fit_is_monotone_and_modest():
    conf = lc.load()
    assert set(conf["positions"]) == {"RB", "WR"} and conf["knot"] == 12.0
    for pos in conf["positions"]:
        ys = [lc.apply(pos, float(x)) for x in np.arange(0.0, 35.01, 0.5)]
        assert (np.diff(ys) >= 0).all()
        assert abs(lc.apply(pos, 12.0) - 12.0) < 1.5 and 0.6 < (lc.apply(pos, 25.0) - lc.apply(pos, 15.0)) / 10 < 1.0


def test_the_fitting_script_can_switch_it_off(monkeypatch):
    monkeypatch.setattr(lc, "OFF", True)
    monkeypatch.setattr(lc, "SEASON_OFF", True)
    assert lc.apply("RB", 22.0) == 22.0 and lc.season("RB", 22.0) == 22.0
    assert lc.apply("RB", 22.0, CONF) < 22.0                  # a fit handed in is still applied (the grading does)


def test_the_season_simulator_reads_a_calibrated_level_and_the_row_keeps_its_value():
    from streamer.league.model import PlayerRow

    p = PlayerRow(player_id="1", name="A", position="WR", team="KC", ros_value=22.0, projection=20.0)
    level = lc.season_level(p)
    assert level < 22.0 and p.ros_value == 22.0
    assert lc.season_level(PlayerRow(player_id="2", name="B", position="QB", team="KC", ros_value=22.0)) == 22.0
    assert lc.season_level(PlayerRow(player_id="3", name="C", position="WR", team="KC", ros_value=None, projection=9.0)) == 9.0


def test_each_league_prices_p_win_with_its_own_calibration():
    from scipy.stats import norm

    from streamer.config import get_config
    from streamer.roster import lineup

    try:
        assert lineup.use_profile(get_config().for_profile("yahoo")) == 1.0
        assert lineup.calibrated(0.7) == 0.7 or abs(lineup.calibrated(0.7) - 0.7) < 1e-9
        assert lineup.use_profile(get_config().for_profile("espn")) == lineup.WIN_SHRINK
        assert abs(lineup.calibrated(0.7) - float(norm.cdf(lineup.WIN_SHRINK * norm.ppf(0.7)))) < 1e-9
    finally:
        lineup.use_profile(get_config().for_profile("espn"))


def test_yahoo_reads_season_values_as_they_are():
    from streamer.config import get_config
    from streamer.league.model import PlayerRow
    from streamer.roster import lineup

    p = PlayerRow(player_id="1", name="A", position="WR", team="KC", ros_value=22.0, projection=20.0)
    try:
        lineup.use_profile(get_config().for_profile("yahoo"))
        assert lc.season_level(p) == 22.0
        lineup.use_profile(get_config().for_profile("espn"))
        assert lc.season_level(p) < 22.0
    finally:
        lineup.use_profile(get_config().for_profile("espn"))
