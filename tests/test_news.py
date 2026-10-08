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
