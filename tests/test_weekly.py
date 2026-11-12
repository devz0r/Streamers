"""Week-by-week expected points for the player card: read off the season
simulation, summing to the season value, a bye a bye, this week's row the
weekly projection untouched."""

from types import SimpleNamespace as NS

import pytest

from streamer.league.model import PlayerRow
from streamer.roster import weekly
from streamer.roster.futures import simulate

WEEKS = list(range(10, 18))


def _p(pid, ros=12.0, status="", team="LA"):
    return PlayerRow(player_id=pid, name=pid, position="WR", team=team, status=status, slot="BN", projection=10.0,
                     projection_sd=5.0, outcome_sd=5.0, ros_value=ros)


def _model(players, byes=None):
    fut = simulate(players, weeks=WEEKS, byes=byes or {}, n_sims=2000, seed=4, current=0)
    return NS(futures=fut, lag=0, players=players, weeks=WEEKS, snapshot=None)


def test_future_weeks_sum_to_the_season_value_times_expected_games(monkeypatch):
    healthy = _p("a", ros=12.0)
    monkeypatch.setattr(weekly, "_opponents", lambda m, c: {"LA": {w: "SF" for w in WEEKS if w != 12}})
    got = weekly.build(_model([healthy], byes={"LA": {12}}), None, None)["a"]
    rows = got["w"]
    assert rows[WEEKS.index(12)] == [12, "bye"]
    future = [r for r in rows[1:] if len(r) > 2]
    games = sum(r[3] for r in future)
    assert sum(r[2] for r in future) == pytest.approx(12.0 * games, rel=0.02)     # rows add up to the season value
    assert got["now"] == 0 and got["tot"] == pytest.approx(sum(r[2] for r in rows if len(r) > 2), abs=0.2)


def test_this_weeks_row_is_not_rescaled_and_an_out_player_shows_no_chance_of_playing(monkeypatch):
    monkeypatch.setattr(weekly, "_opponents", lambda m, c: {})
    hurt = _p("o", ros=12.0, status="OUT")
    hurt.games_missed = 0
    hurt.reported_back = [13, 13]
    got = weekly.build(_model([hurt]), None, None)["o"]["w"]
    assert all(r[3] == 0.0 for r in got[:3])                    # weeks 10-12: out in every simulated season
    assert got[3][3] > 0.85 and got[3][2] > 0                   # back in 13


def test_a_player_with_no_value_and_no_games_is_left_out(monkeypatch):
    monkeypatch.setattr(weekly, "_opponents", lambda m, c: {})
    gone = _p("g", ros=12.0, status="OUT")
    gone.reported_back = [99, 99]
    assert weekly.build(_model([gone]), None, None) == {}
