"""Trades: offered only when the other side gains on projections, priced in title odds."""

from __future__ import annotations

import pytest

from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow
from streamer.roster import season as season_mod
from streamer.roster.season import SeasonModel
from streamer.roster.trades import TradeFinder


def _p(pid, pos, v):
    return PlayerRow(player_id=pid, name=pid, position=pos, team="KC", projection=v, projection_sd=6.0,
                     outcome_sd=6.0, ros_value=v, ros_sd=2.0,
                     eligible_slots=[pos, "FLEX"] if pos != "QB" else ["QB"])


def _team(tid, spec, mine=False):
    roster = [_p(f"{tid}-{pos}{i}", pos, v) for pos, vals in spec.items() for i, v in enumerate(vals)]
    return TeamRow(team_id=tid, name=f"Team {tid}", roster=roster, is_mine=mine, wins=1, losses=1)


def _league():
    # You: four good backs, weak receivers. Team 1: the mirror image.
    me = _team("0", {"QB": [18], "RB": [16, 15, 14, 13], "WR": [8, 7, 6], "TE": [8]}, mine=True)
    mirror = _team("1", {"QB": [18], "RB": [8, 7, 6], "WR": [16, 15, 14, 13], "TE": [8]})
    others = [_team(str(i), {"QB": [18], "RB": [12, 11, 7], "WR": [12, 11, 7], "TE": [8]}) for i in (2, 3)]
    teams = [me, mirror, *others]
    ids = [t.team_id for t in teams]
    schedule = [[w, a, b] for w in range(3, 11) for a, b in ((ids[0], ids[1 + w % 3]),
                                                             tuple(i for i in ids[1:] if i != ids[1 + w % 3]))]
    fa = [_p("fa-rb", "RB", 5), _p("fa-rb2", "RB", 4.5), _p("fa-wr", "WR", 5), _p("fa-wr2", "WR", 4.5),
          _p("fa-qb", "QB", 12), _p("fa-qb2", "QB", 11), _p("fa-te", "TE", 4), _p("fa-te2", "TE", 3.5)]
    return LeagueSnapshot(
        platform="espn", profile="espn", league_id="1", league_name="L", season=2026, week=3,
        slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}, bench_size=4, teams=teams,
        free_agents=fa, matchup=None, synced_at="",
        rules={"regular_season_weeks": 10, "playoff_teams": 2, "playoff_round_weeks": 1,
               "schedule": schedule})


@pytest.fixture(autouse=True)
def _no_byes(monkeypatch):
    monkeypatch.setattr(season_mod, "nfl_byes", lambda *a, **k: {})


@pytest.fixture
def finder(cfg):
    snap = _league()
    return TradeFinder(snap, SeasonModel(snap, cfg, n_sims=1500))


def test_surplus_for_need_is_found_and_both_sides_gain_on_projections(finder):
    trades = finder.find(5)
    assert trades, "a back-for-receiver swap with the mirror team should be on offer"
    best = trades[0]
    assert best.partner.team_id == "1"
    assert "RB" in {p.position for p in best.give} and "WR" in {p.position for p in best.get}
    assert best.gain > 0 and best.p_new > best.p_base
    assert best.their_value >= 0.25 and best.their_lineup >= -0.25


def test_nothing_is_offered_that_the_other_side_would_refuse(finder):
    for t in finder.screen():
        assert t.their_value >= 0.25 and t.their_lineup >= -0.25 and t.my_value > 0


def test_a_trade_never_leaves_either_side_without_a_quarterback(finder):
    for t in finder.screen():
        mine = [p for p in finder.me.roster if p not in t.give] + t.get
        theirs = [p for p in t.partner.roster if p not in t.get] + t.give
        assert any(p.position == "QB" for p in mine) and any(p.position == "QB" for p in theirs)


def test_the_page_names_both_sides_of_each_trade(finder):
    from types import SimpleNamespace

    from streamer.roster.page import _trades

    trades = finder.find(3)
    html = _trades(SimpleNamespace(trades=trades))
    assert "Trades, by title odds" in html and "Team 1" in html
    assert all(p.name in html for t in trades for p in t.give + t.get)
    assert _trades(SimpleNamespace(trades=None)) == ""
    assert "No trade" in _trades(SimpleNamespace(trades=[]))
