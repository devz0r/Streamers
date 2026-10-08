"""Player news: a free agent the news has close to a team is priced, not zeroed."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from streamer.data import news

NOW = datetime(2026, 10, 8, 12, tzinfo=UTC)


def _item(headline, when="2026-10-05T14:53:50Z", story="", kind="Rotowire"):
    return {"type": kind, "published": when, "headline": headline, "story": story}


HILL = [
    _item("Is Tyreek Hill relevant in fantasy football?", kind="Media"),
    _item("Hill's (knee) agent said he plans to begin negotiating with multiple teams Monday.",
          story="Schefter notes that the Chiefs are among the teams interested."),
    _item("The Chiefs reached out to Hill's (knee) agent Sunday.", when="2026-10-05T01:47:59Z"),
    _item("Hill (knee) has been advised to delay his comeback.", when="2026-09-09T18:33:30Z"),
    _item("Tyreek Hill's wife cross-examined during Day 2 of trial", when="2026-10-08T00:38:11Z",
          kind="HeadlineNews"),
]


def test_talks_with_one_team_named_and_a_legal_case_flagged():
    s = news.read(HILL, "Tyreek Hill", NOW)
    assert s.stage == "talks" and s.team == "KC" and s.legal
    assert "negotiating" in s.headline or "reached out" in s.headline


def test_the_most_advanced_stage_wins_and_only_a_headline_with_a_team_is_a_signing():
    close = HILL + [_item("Hill is expected to sign with the Chiefs this week.", when="2026-10-07T10:00:00Z")]
    assert news.read(close, "Tyreek Hill", NOW).stage == "close"
    signed = close + [_item("Hill signs one-year deal with Chiefs", when="2026-10-08T09:00:00Z")]
    s = news.read(signed, "Tyreek Hill", NOW)
    assert s.stage == "signed" and s.team == "KC"
    # A past signing mentioned in a story is not news of one.
    old = [_item("Hill (knee) is not yet 100 percent.", story="Hill signed with Miami in 2022.")]
    assert news.read(old, "Tyreek Hill", NOW) is None


def test_old_news_and_columns_do_not_count():
    assert news.read(HILL[3:4], "Tyreek Hill", NOW) is None                  # September: stale
    assert news.read([_item("Is Hill worth a visit to your waiver wire?", kind="Story")],
                     "Tyreek Hill", NOW) is None


def test_the_expected_share_follows_the_odds_and_the_calendar():
    full = news.expected_share(1.0, 0, 12)
    assert 0.6 < full < 1.0                                  # the measured wait for a first game
    assert news.expected_share(0.5, 0, 12) == pytest.approx(full / 2)
    assert news.expected_share(1.0, 2, 12) < full
    assert news.expected_share(1.0, 0, 1) < full


def _league(cfg):
    from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow

    hill = PlayerRow(player_id="3116406", name="Tyreek Hill", position="WR", team=None, status="OUT")
    rival = TeamRow(team_id="2", name="Rival", roster=[hill])
    me = TeamRow(team_id="1", name="Me", roster=[], is_mine=True)
    stray = PlayerRow(player_id="9", name="Nobody Owns", position="WR", team=None, slot="FA")
    return LeagueSnapshot(platform="espn", profile="espn", league_id="1", league_name="L", season=2026, week=6,
                          slots={"WR": 2}, bench_size=5, teams=[me, rival], free_agents=[stray],
                          matchup=None, synced_at="")


def test_a_rostered_free_agent_is_marked_with_odds_timing_and_the_news(cfg, monkeypatch):
    monkeypatch.setattr(news, "player_items", lambda c, espn_id, allow_network=True: HILL[:4])
    monkeypatch.setattr(news, "league_items", lambda c, allow_network=True: HILL[4:])
    monkeypatch.setattr(news, "trending_adds", lambda c, allow_network=True: [])
    snap = _league(cfg)
    lines = news.attach(snap, cfg.for_profile("espn"), {"tyreek hill": "3116406"}, allow_network=False, now=NOW)
    hill, stray = snap.teams[1].roster[0], snap.free_agents[0]
    assert len(lines) == 1 and "talks with KC" in lines[0]
    assert hill.signing_team == "KC" and hill.signing_wait == 2
    assert hill.signing_odds == pytest.approx(0.5 * 13 / 17, abs=1e-3)
    assert 0 < hill.signing_share < hill.signing_odds
    assert hill.news.startswith("news Oct 05") and "legal case" in hill.news
    assert stray.signing_odds is None                      # not rostered, owned or trending: not asked about


def test_the_simulation_plays_him_only_if_he_signs_and_after_the_wait():
    from streamer.league.model import PlayerRow
    from streamer.roster.futures import GONE, _initial_absence

    p = PlayerRow(player_id="h", name="H", position="WR", status="OUT", signing_odds=0.4, signing_wait=2)
    out = _initial_absence(p, 20000, np.random.default_rng(1), {})
    assert (out == GONE).mean() == pytest.approx(0.6, abs=0.02)
    assert out[out != GONE].min() >= 2


def test_the_screens_count_him_for_the_share_he_plays():
    from streamer.league.model import PlayerRow
    from streamer.roster.waivers import _ros

    p = PlayerRow(player_id="h", name="H", position="WR", status="OUT", ros_value=11.0, signing_share=0.25)
    assert _ros(p) == pytest.approx(2.75)


def test_the_projection_prices_a_signing_free_agent_and_zeroes_this_week(cfg, monkeypatch):
    import streamer.roster.projections as pj
    from test_roster_projections import _stale_history, _stale_snapshot

    def project(odds):
        snap = _stale_snapshot("OUT", None, None)
        ghost = snap.free_agents[0]
        if odds:
            ghost.signing_odds, ghost.signing_team, ghost.signing_share = odds, "KC", 0.3
            ghost.news = "news Oct 05: the Chiefs reached out to his agent"
        monkeypatch.setattr(pj, "_implied_scale", lambda *a, **k: ({}, "test"))
        monkeypatch.setattr(pj, "match_players",
                            lambda rows, index: type("M", (), {"mapping": {"g1": "nfl-g1"}, "unmatched": []})())
        pj.project_snapshot(snap, cfg.for_profile("espn"), rankings=None, allow_network=False,
                            history=_stale_history())
        return ghost

    quiet, talks = project(None), project(0.38)
    assert quiet.ros_value == 0.0                                  # no news: nothing until he signs
    assert talks.ros_value > 5.0 and talks.projection == 0.0
    assert talks.signals[0].startswith("news Oct 05") and "for KC when he plays" in talks.signals[0]
    assert not any("no value until" in s for s in talks.signals)


# -- breaking news, for everyone ---------------------------------------------
def _matchup_league():
    from streamer.league.model import LeagueSnapshot, Matchup, PlayerRow, TeamRow

    mine = TeamRow(team_id="1", name="Me", is_mine=True, roster=[
        PlayerRow(player_id="e1", name="Justin Jefferson", position="WR", team="MIN", status="QUESTIONABLE")])
    opp = TeamRow(team_id="2", name="Them", roster=[
        PlayerRow(player_id="e2", name="Bijan Robinson", position="RB", team="ATL")])
    other = TeamRow(team_id="3", name="Else", roster=[
        PlayerRow(player_id="e3", name="Ladd McConkey", position="WR", team="LAC")])
    fa = [PlayerRow(player_id="e4", name="Keon Coleman", position="WR", team="BUF", slot="FA"),
          PlayerRow(player_id="e5", name="Quiet Guy", position="TE", team="NYG", slot="FA")]
    return LeagueSnapshot(platform="espn", profile="espn", league_id="1", league_name="L", season=2026, week=6,
                          slots={"WR": 2}, bench_size=5, teams=[mine, opp, other], free_agents=fa,
                          matchup=Matchup(week=6, my_team_id="1", opponent_team_id="2"), synced_at="")


ESPN = {
    "e1": [_item("Jefferson (ankle) ruled out for Sunday", when="2026-10-09T20:00:00Z")],
    "e2": [_item("Robinson (knee) practiced fully Thursday", when="2026-10-09T18:00:00Z")],
    "e3": [_item("McConkey (hamstring) won't return Sunday", when="2026-10-09T17:00:00Z"),
           _item("McConkey (hamstring) did not practice", when="2026-10-01T17:00:00Z")],
    "e4": [],
}
FP = [{"player_id": "fp4", "created": "2026-10-09 21:00:00", "link": "https://example.test/coleman"},
      {"player_id": "fp3", "created": "2026-10-09 17:30:00", "link": "https://example.test/mcconkey"}]


def _breaking(cfg, monkeypatch):
    asked = []
    monkeypatch.setattr(news, "fp_feed", lambda c, allow_network=True: FP)

    def items(c, espn_id, allow_network=True, newer_than=None):
        asked.append(espn_id)
        return ESPN.get(espn_id, [])

    monkeypatch.setattr(news, "player_items", items)
    snap = _matchup_league()
    ids = {"justin jefferson": "e1", "bijan robinson": "e2", "ladd mcconkey": "e3", "keon coleman": "e4",
           "quiet guy": "e5"}
    got = news.breaking(snap, cfg, ids, {"fp4": "Keon Coleman", "fp3": "Ladd McConkey"}, allow_network=False,
                        now=datetime(2026, 10, 10, 12, tzinfo=UTC))
    return snap, got, asked


def test_news_for_everyone_yours_first_and_only_who_has_news_is_asked(cfg, monkeypatch):
    snap, got, asked = _breaking(cfg, monkeypatch)
    assert [n.whose for n in got] == ["yours", "opponent", "rostered", "free agent"]
    assert sorted(asked) == ["e1", "e2", "e3", "e4"]          # the quiet free agent is never asked about
    tags = {n.name: n.tag for n in got}
    assert tags["Justin Jefferson"] == "out" and tags["Bijan Robinson"] == "healthy"
    coleman = next(n for n in got if n.name == "Keon Coleman")
    assert coleman.source == "FantasyPros" and coleman.headline == "news at FantasyPros"   # its text is never shown
    assert [n.headline for n in got if n.name == "Ladd McConkey"] == ["McConkey (hamstring) won't return Sunday"]


def test_ruled_out_this_week_sets_him_out_but_an_in_game_injury_does_not(cfg, monkeypatch):
    snap, got, _asked = _breaking(cfg, monkeypatch)
    jj = snap.my_team.roster[0]
    assert jj.status == "OUT"
    assert next(n for n in got if n.name == "Justin Jefferson").acted.startswith("set out for this week")
    assert snap.teams[2].roster[0].status == ""                              # "won't return": last game


def test_the_hub_lists_the_news(cfg, monkeypatch):
    from streamer.roster.page import _news

    snap, got, _asked = _breaking(cfg, monkeypatch)
    snap._news = got
    html = _news(snap, now=datetime(2026, 10, 10, 12, tzinfo=UTC))
    assert "Breaking news" in html and html.count("<li>") == 4
    assert "t-out" in html and "set out for this week" in html and "16h ago" in html
    assert _news(_matchup_league()) == ""


def test_each_player_shows_once_before_anyone_repeats():
    from streamer.roster.page import NEWS_SHOWN, _news

    snap = _matchup_league()
    t = datetime(2026, 10, 10, 11, tzinfo=UTC)
    busy = [news.NewsItem("Busy Guy", "WR", "MIN", t, f"Guy item {i}", "", "ESPN", "injury", "yours") for i in range(15)]
    other = news.NewsItem("Other Guy", "RB", "ATL", t, "Guy other", "", "ESPN", "", "rostered")
    snap._news = busy + [other]
    html = _news(snap, now=datetime(2026, 10, 10, 12, tzinfo=UTC))
    top = html.split("<details>")[0]
    assert top.count("Busy Guy") == NEWS_SHOWN - 1 and "Other Guy" in top
