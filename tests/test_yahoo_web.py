"""Parsing Yahoo's fantasy pages, against fixtures rebuilt from the probed markup."""

from __future__ import annotations

import pytest

from fixtures_yahoo_web import (
    LEAGUE,
    free_agent_page,
    league_page,
    matchup_page,
    team_page,
)
from streamer.league.yahoo_web import (
    my_team_id,
    opponent_id,
    parse_free_agents,
    parse_roster,
    parse_standings,
)


def _by_name(players):
    return {p.name: p for p in players}


def test_roster_players_slots_and_bench():
    players, slots, bench = parse_roster(team_page())
    names = _by_name(players)
    assert len(players) == 11                        # the empty RB slot is skipped
    assert slots == {"QB": 1, "WR": 2, "RB": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1}
    assert bench == 2

    qb = names["Quinn Arrow"]
    assert (qb.position, qb.team, qb.slot) == ("QB", "LA", "QB")      # LAR -> LA
    assert qb.platform_projection == pytest.approx(17.84)
    assert qb.percent_owned == pytest.approx(97.0)
    assert qb.player_id == "101"


def test_multi_position_flex_and_defense():
    names = _by_name(parse_roster(team_page())[0])
    flex = names["Flex Man"]
    assert flex.slot == "FLEX" and flex.position == "RB" and flex.team == "WAS"
    assert {"RB", "WR", "FLEX"} <= set(flex.eligible_slots)

    dst = names["Tampa Bay"]
    assert (dst.position, dst.team, dst.slot) == ("DST", "TB", "DST")
    assert names["Kip Kick"].position == "K"


def test_injury_tags_and_ir_slot():
    names = _by_name(parse_roster(team_page())[0])
    assert names["Wade Runner"].status == "Q" and names["Wade Runner"].is_questionable
    ir = names["Ira Hurt"]
    assert ir.slot == "IR" and ir.in_ir_slot and ir.is_out
    assert ir.platform_projection is None             # "–" is no projection, not zero
    assert names["Quinn Arrow"].status == ""


def test_standings_and_league_name():
    name, teams = parse_standings(league_page(), LEAGUE)
    assert name == "Test League"
    by_id = {t["team_id"]: t for t in teams}
    assert set(by_id) == {"2", "5", "6"}
    assert by_id["2"]["name"] == "Alpha Squad"
    assert (by_id["2"]["wins"], by_id["2"]["losses"], by_id["2"]["ties"]) == (2, 0, 0)
    assert by_id["5"]["points_for"] == pytest.approx(230.1)


def test_my_team_and_opponent_are_found_without_extra_secrets():
    assert my_team_id(league_page(), LEAGUE) == "5"
    assert opponent_id(matchup_page("5", "6"), LEAGUE, "5") == "6"
    # Ambiguous or absent: say so rather than guess.
    assert opponent_id("<html></html>", LEAGUE, "5") is None


def test_free_agent_projection_only_trusted_when_the_page_says_so():
    """A season total mistaken for a weekly projection would poison the blend."""
    projected = _by_name(parse_free_agents(free_agent_page(projected=True, week=3), 3))
    assert projected["Fay Agent"].platform_projection == pytest.approx(16.4)
    assert projected["Fay Agent"].percent_owned == pytest.approx(41.0)
    assert projected["Fay Agent"].slot == "FA" and projected["Fay Agent"].team == "MIN"

    season = _by_name(parse_free_agents(free_agent_page(projected=False, week=3), 3))
    assert season["Fay Agent"].platform_projection is None
    assert season["Walt Wire"].percent_owned == pytest.approx(12.0)


def test_logged_out_page_parses_to_nothing_rather_than_garbage():
    assert parse_roster("<html><body>Sign in</body></html>") == ([], {}, 0)
    assert parse_standings("<html></html>", LEAGUE) == ("", [])
    assert parse_free_agents("<html></html>", 3) == []


class _FakeResp:
    def __init__(self, url, text, status=200):
        self.url, self.text, self.status_code = url, text, status
        self.content = text.encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, pages, logged_out=False):
        self.pages, self.logged_out, self.asked = pages, logged_out, []

    def get(self, url, timeout=None):
        self.asked.append(url)
        if self.logged_out:
            return _FakeResp("https://login.yahoo.com/?done=x", "Sign in to Yahoo")
        for needle, html in self.pages:
            if needle in url:
                return _FakeResp(url, html)
        return _FakeResp(url, "<html></html>")


def _pages():
    return [
        (f"/f1/{LEAGUE}/players", free_agent_page(True, 3)),
        (f"/f1/{LEAGUE}/matchup", matchup_page("5", "6")),
        (f"/f1/{LEAGUE}/5?", team_page("1")),
        (f"/f1/{LEAGUE}/6?", team_page("2")),
        (f"/f1/{LEAGUE}", league_page()),
    ]


def test_snapshot_assembles_from_the_website(monkeypatch):
    import streamer.league.yahoo_web as yw

    fake = _FakeSession(_pages())
    monkeypatch.setattr(yw, "_session", lambda cookie: fake)
    monkeypatch.setenv("YAHOO_COOKIE", "A1=x; T=y; Y=z")
    monkeypatch.setenv("YAHOO_LEAGUE_ID", LEAGUE)
    monkeypatch.delenv("YAHOO_TEAM_ID", raising=False)

    snap = yw.fetch_snapshot(2026, 3, "yahoo", pause=0)
    assert snap.platform == "yahoo" and snap.league_name == "Test League"
    assert snap.my_team.team_id == "5" and snap.my_team.name == "My Team Name"
    assert len(snap.my_team.roster) == 11
    assert snap.opponent is not None and snap.opponent.team_id == "6"
    assert snap.opponent.name == "The Rival" and len(snap.opponent.roster) == 11
    assert snap.slots["FLEX"] == 1 and snap.bench_size == 2
    # Offense (two pages), K and DEF lists were each requested.
    assert sum("players?" in u for u in fake.asked) == 4
    assert all("stat1=S_PW_3" in u for u in fake.asked if "players?" in u)
    assert len(snap.free_agents) == 8
    # Round-trips through the same JSON the ESPN snapshots use.
    from streamer.league.model import LeagueSnapshot

    again = LeagueSnapshot.from_dict(__import__("json").loads(snap.to_json()))
    assert again.my_team.team_id == "5" and len(again.free_agents) == 8


def test_expired_cookie_fails_loudly_with_the_fix(monkeypatch):
    import streamer.league.yahoo_web as yw

    monkeypatch.setattr(yw, "_session", lambda cookie: _FakeSession([], logged_out=True))
    monkeypatch.setenv("YAHOO_COOKIE", "A1=x")
    monkeypatch.setenv("YAHOO_LEAGUE_ID", LEAGUE)
    with pytest.raises(RuntimeError, match="YAHOO_COOKIE"):
        yw.fetch_snapshot(2026, 3, "yahoo", pause=0)
