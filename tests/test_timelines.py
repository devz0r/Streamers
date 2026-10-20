"""How long the news says a player is out: parsing, attaching, and the simulation."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from streamer.data import news
from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow
from streamer.roster.futures import GONE, simulate

NOW = datetime(2026, 10, 20, 14, 0, tzinfo=UTC)


@pytest.mark.parametrize("text,expected", [
    ("Davante Adams (knee) is expected to miss 6-10 weeks.", (6, 10)),
    ("He is out four to six weeks with a high-ankle sprain.", (4, 6)),
    ("Out 2-4 weeks with a hamstring strain", (2, 4)),
    ("Expected to be sidelined for about a month.", (4, 4)),
    ("will be out a couple of weeks", (2, 2)),
    ("Timetable of 8 weeks for recovery", (8, 8)),
    ("Will miss the rest of the season with a torn ACL.", "season"),
    ("Placed on IR; the team says he's out for the season.", "season"),
    ("He has missed the last three weeks and is week-to-week.", None),
    ("Lamb is questionable for Week 7 with a shoulder issue.", None),
    ("He practiced fully and should return after the bye in Week 9.", None),
    ("As a result, Hurts will be required to miss at least four games.", None),             # an IR floor
    ("McLaurin is slated for an MRI after what the Commanders fear was a season-ending knee injury.", None),
    ("He will be eligible to play as soon as the Packers' Week 7 game at Detroit.", None),
    ("Ekeler returns after his 2025 campaign ended with a torn Achilles.", None),            # a past injury
    ("The Colts are without a timeline for a return.", None),
    ("Hurts is expected to miss the rest of the season after new imaging.", "season"),
    ("had 12 games with 5 receptions", None),
])
def test_the_timeline_is_read_from_the_report(text, expected):
    assert news.timeline(text) == expected


def _snap(status="OUT", slot="IR"):
    adams = PlayerRow(player_id="1", name="Davante Adams", position="WR", team="LA", status=status, slot=slot,
                      projection=0.0, ros_value=12.0, ros_sd=2.0)
    healthy = PlayerRow(player_id="2", name="Healthy Guy", position="WR", team="LA", status="", slot="WR",
                        projection=10.0, ros_value=10.0)
    return LeagueSnapshot(platform="espn", profile="espn", league_id="1", league_name="L", season=2026, week=7,
                          slots={"WR": 2}, bench_size=2, free_agents=[], matchup=None, synced_at="",
                          teams=[TeamRow(team_id="1", name="T", roster=[adams, healthy], is_mine=True)])


def _items(*rows):
    return [{"type": "Rotowire", "published": when, "headline": head, "story": story} for when, head, story in rows]


def test_a_reported_timeline_sets_when_he_is_back(cfg, monkeypatch):
    snap = _snap()
    monkeypatch.setattr(news, "player_items", lambda *a, **k: _items(
        ("2026-10-19T15:00:00Z", "Davante Adams (knee) expected to miss 6-10 weeks", "Adams will be out 6-10 weeks.")))
    lines = news.timelines(snap, cfg, {"davante adams": "9", "healthy guy": "8"}, allow_network=False, now=NOW)
    p = snap.teams[0].roster[0]
    assert p.reported_back == [13, 17] and p.reported_share == pytest.approx(3 / 11, abs=0.01)
    assert "6-10 weeks" in p.reported_note and lines and "back around week 13-17" in lines[0]
    assert snap.teams[0].roster[1].reported_back is None            # nothing about him


def test_an_older_report_counts_down_and_a_newer_healthy_item_cancels_it(cfg, monkeypatch):
    snap = _snap()
    monkeypatch.setattr(news, "player_items", lambda *a, **k: _items(
        ("2026-10-06T15:00:00Z", "Davante Adams out 6-10 weeks", "He will miss 6-10 weeks.")))        # two weeks ago
    news.timelines(snap, cfg, {"davante adams": "9"}, allow_network=False, now=NOW)
    assert snap.teams[0].roster[0].reported_back == [11, 15]
    monkeypatch.setattr(news, "player_items", lambda *a, **k: _items(
        ("2026-10-20T09:00:00Z", "Davante Adams practiced fully", "He is a full participant."),
        ("2026-10-06T15:00:00Z", "Davante Adams out 6-10 weeks", "He will miss 6-10 weeks.")))
    news.timelines(snap, cfg, {"davante adams": "9"}, allow_network=False, now=NOW)
    assert snap.teams[0].roster[0].reported_back is None


def test_a_season_ending_report_removes_him_for_the_year(cfg, monkeypatch):
    snap = _snap()
    monkeypatch.setattr(news, "player_items", lambda *a, **k: _items(
        ("2026-10-19T15:00:00Z", "Adams tears ACL; out for the season", "Adams is done for the season.")))
    news.timelines(snap, cfg, {"davante adams": "9"}, allow_network=False, now=NOW)
    p = snap.teams[0].roster[0]
    assert p.reported_back == [99, 99] and p.reported_share == 0.0


def _wr(pid, back=None):
    p = PlayerRow(player_id=pid, name=pid, position="WR", team="LA", status="OUT", slot="BN", projection=0.0,
                  projection_sd=5.0, outcome_sd=5.0, ros_value=12.0)
    p.reported_back = back
    return p


def test_the_simulation_sits_him_for_the_reported_weeks():
    weeks = list(range(7, 18))
    out = simulate([_wr("a", [13, 17])], weeks=weeks, byes={}, n_sims=3000, seed=2)
    played = out.played[:, 0, :]
    assert not played[:, :6].any()                       # weeks 7-12: out in every simulated season
    assert played[:, weeks.index(17)].mean() > 0.6       # back by 17 in most
    assert 0.1 < played[:, weeks.index(13)].mean() < 0.45  # back the first possible week in about a fifth


def test_the_generic_draw_without_a_report_has_him_back_much_sooner():
    weeks = list(range(7, 18))
    generic = simulate([_wr("a")], weeks=weeks, byes={}, n_sims=3000, seed=2).played[:, 0, :]
    assert generic[:, weeks.index(10)].mean() > 0.6      # a plain OUT tag: back within a few games


def test_a_season_ending_report_is_gone_and_byes_are_not_games():
    weeks = list(range(7, 18))
    gone = simulate([_wr("a", [99, 99])], weeks=weeks, byes={}, n_sims=500, seed=2)
    assert not gone.played[:, 0, :].any()
    byed = simulate([_wr("a", [12, 12])], weeks=weeks, byes={"LA": {9}}, n_sims=500, seed=2)
    assert byed.played[:, 0, weeks.index(12)].mean() > 0.9       # out_left counted five games, not six
    assert GONE > 17


def test_the_screening_value_counts_only_the_games_after_he_is_back():
    from streamer.roster.waivers import _ros

    p = _wr("a", [13, 17])
    p.reported_share = 0.2
    assert _ros(p) == pytest.approx(2.4)
    p.reported_share = None
    assert _ros(p) == pytest.approx(12.0)
    assert np.isclose(_ros(_wr("b", None)), 12.0)


def test_each_report_is_logged_once(tmp_path):
    path = tmp_path / "timeline_log.parquet"
    row = {"season": 2026, "name": "Davante Adams", "position": "WR", "reported": "2026-10-19T15:00:00+00:00",
           "weeks_low": 6, "weeks_high": 10, "season_ending": False, "back_lo": 13, "back_hi": 17, "logged_week": 7}
    assert news.record_timelines(path, [row]) == 1
    assert news.record_timelines(path, [row, dict(row, name="Other Guy")]) == 2


def test_a_timeline_in_an_article_counts_only_for_the_player_it_names():
    article = {"type": "Story", "headline": "Fantasy buzz",
               "description": "<p>The news on <a href='x'>Davante Adams</a>' injury is devastating. "
                              "Adams is expected to miss 6-10 weeks.</p><p>Evans, meanwhile, will miss 4 weeks.</p>"}
    assert news.player_timeline(article, "Adams") == ((6, 10), "Adams is expected to miss 6-10 weeks.")
    assert news.player_timeline(article, "Evans")[0] == (4, 4)
    assert news.player_timeline(article, "Kelce") is None                  # not named anywhere
    blurb = {"type": "Rotowire", "headline": "Cooper Kupp (hamstring) is out.", "story": "He should miss four weeks."}
    assert news.player_timeline(blurb, "Kupp")[0] == (4, 4)                 # "He" in a blurb about him


def test_an_article_is_used_when_the_blurbs_give_no_time(cfg, monkeypatch):
    snap = _snap()
    items = _items(("2026-10-19T15:00:00Z", "Adams sustained a ruptured hamstring", "He will have a procedure."))
    items.append({"type": "Story", "published": "2026-10-20T08:00:00Z", "headline": "Fantasy buzz",
                  "description": "<p>Adams is expected to miss 6-10 weeks.</p>"})
    monkeypatch.setattr(news, "player_items", lambda *a, **k: items)
    news.timelines(snap, cfg, {"davante adams": "9"}, allow_network=False, now=NOW)
    assert snap.teams[0].roster[0].reported_back == [13, 17]
    old = dict(items[-1], published="2026-10-05T08:00:00Z")                 # an old article is not trusted
    monkeypatch.setattr(news, "player_items", lambda *a, **k: items[:-1] + [old])
    news.timelines(snap, cfg, {"davante adams": "9"}, allow_network=False, now=NOW)
    assert snap.teams[0].roster[0].reported_back is None
