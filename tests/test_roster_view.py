"""Your roster, valued for the rest of the season, and its tab on the page."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from streamer.roster import season as season_mod
from streamer.roster.matchup import MatchupReport
from streamer.roster.page import _roster
from streamer.roster.roster_view import roster_values
from streamer.roster.title_moves import TitleEngine
from test_season import _league


@pytest.fixture(autouse=True)
def _no_byes(monkeypatch):
    monkeypatch.setattr(season_mod, "nfl_byes", lambda *a, **k: {})


def test_every_player_is_valued_on_the_same_seasons_most_valuable_first(cfg):
    snap = _league([1.0, 1.1, 1.0, 1.05, 0.95, 1.0])
    me = snap.my_team
    star = max(me.roster, key=lambda p: p.ros_value or 0)
    engine = TitleEngine(snap, cfg, n_sims=1500, candidates=4)
    seen = {star.player_id: SimpleNamespace(value=(star.ros_value or 0) + 3.0)}
    vals = roster_values(engine, seen=seen)

    assert {r.player.player_id for r in vals} == {p.player_id for p in me.roster}
    assert [r.title for r in vals] == sorted((r.title for r in vals), reverse=True)
    by = {r.player.player_id: r for r in vals}
    assert vals[0].player is star and by[star.player_id].title > 0
    for r in vals:
        assert r.low <= r.per_week <= r.high and r.noise >= 0
    assert by[star.player_id].market == pytest.approx((star.ros_value or 0) + 3.0)
    assert all(r.market is None for r in vals if r.player is not star)


def test_the_roster_tab_reads_the_values_and_links_to_the_moves(cfg):
    snap = _league([1.0, 1.1, 1.0, 1.05])
    engine = TitleEngine(snap, cfg, n_sims=800, candidates=4)
    star = max(snap.my_team.roster, key=lambda p: p.ros_value or 0)
    vals = roster_values(engine, seen={star.player_id: SimpleNamespace(value=(star.ros_value or 0) + 3.0)})
    report = SimpleNamespace(roster_values=vals)
    html = _roster(report, "espn")
    assert html.count("<td class='unit'><b>") == len(vals)
    assert "worth more traded than held" in html                 # the market sees him higher
    assert 'for="sec-espn-waivers"' in html and 'for="sec-espn-trades"' in html
    assert _roster(MatchupReport.__new__(MatchupReport), "espn") == ""


def _p(pid, pos, ros, ir=False):
    from streamer.league.model import PlayerRow

    return PlayerRow(player_id=pid, name=pid, position=pos, team="SEA", ros_value=ros, ros_sd=1.0,
                     slot="IR" if ir else "BE")


def test_drops_are_tried_cheapest_to_the_title_and_always_the_one_he_replaces():
    """A third quarterback projects well and is worth nothing to your title:
    by season value he was never tried as a drop."""
    from streamer.roster.title_moves import drop_candidates

    roster = [_p("qb1", "QB", 18), _p("qb2", "QB", 16), _p("qb3", "QB", 15.7), _p("rb1", "RB", 14),
              _p("rb2", "RB", 8.0), _p("wr1", "WR", 12), _p("wr2", "WR", 7.5), _p("te", "TE", 8.3),
              _p("k", "K", 8.0)]
    worth = {"qb1": 0.02, "qb2": 0.004, "qb3": 0.002, "rb1": 0.04, "rb2": 0.009, "wr1": 0.03, "wr2": 0.008,
             "te": 0.007, "k": 0.0}
    add = _p("new", "RB", 9.0)
    assert [p.player_id for p in drop_candidates(roster, add, 2)] == ["wr2", "rb2"]
    # The only tight end is kept, and the back he replaces is always tried.
    assert [p.player_id for p in drop_candidates(roster, add, 4, worth)] == ["qb3", "qb2", "wr2", "rb2"]
    assert [p.player_id for p in drop_candidates(roster, add, 3, worth)] == ["qb3", "qb2", "rb2"]
    assert "qb3" not in [p.player_id for p in drop_candidates(roster, add, 4)]           # by value alone


def test_the_cheapest_to_let_go_skip_streamers_reserve_slots_and_a_last_body():
    from streamer.roster.roster_view import RosterValue, cheapest

    roster = [_p("qb", "QB", 18), _p("te", "TE", 9), _p("k", "K", 8), _p("hurt", "RB", 12, ir=True),
              _p("rb", "RB", 6), _p("wr", "WR", 7)]
    vals = [RosterValue(player=p, ros=p.ros_value, per_week=0, low=0, high=0, title=t, noise=0)
            for p, t in zip(roster, [0.0, 0.0, 0.0, 0.0, 0.01, 0.02])]
    assert [r.player.player_id for r in cheapest(vals, roster, 3)] == ["rb", "wr"]


def test_a_cheap_player_the_market_still_values_is_a_trade_not_a_drop():
    from streamer.roster.page import _cut_names
    from streamer.roster.roster_view import RosterValue

    jacobs = RosterValue(player=_p("Josh Jacobs", "RB", 12.6), ros=12.6, per_week=6.7, low=1, high=13,
                         title=0.006, noise=0.001, market=14.1)
    fodder = RosterValue(player=_p("KC Concepcion", "WR", 9.4), ros=9.4, per_week=7.3, low=3, high=12,
                         title=0.004, noise=0.001, market=9.3)
    text = _cut_names([fodder, jacobs])
    assert "KC Concepcion (+0.4)" in text
    assert "Josh Jacobs (+0.6; trade, don't drop: the market sees 14.1 a game)" in text
