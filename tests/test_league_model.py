"""Snapshot model, label normalisation, and both adapters' pure functions.

The adapters are tested on stand-in objects shaped exactly like what
``espn-api`` and ``yahoo_fantasy_api`` return, because the platforms cannot be
reached from the test environment -- and because that is the layer where a
silent field mix-up would corrupt every downstream recommendation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fixtures_league import snapshot
from streamer.league import espn, yahoo
from streamer.league.model import (
    LeagueSnapshot,
    PlayerRow,
    canonical_position,
    canonical_slot,
    canonical_status,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def test_snapshot_round_trips_through_json(tmp_path):
    snap = snapshot()
    path = snap.save(tmp_path / "week_5.json")
    back = LeagueSnapshot.load(path)
    assert back.my_team.name == "My Team"
    assert back.opponent is not None and back.opponent.name == "Rival"
    assert len(back.free_agents) == len(snap.free_agents)
    assert back.starting_slots == snap.starting_slots
    assert back.my_team.roster[0].projection == snap.my_team.roster[0].projection


def test_starting_slots_exclude_bench_and_ir():
    snap = snapshot()
    assert "BN" not in snap.starting_slots
    assert "IR" not in snap.starting_slots
    assert snap.starting_slots["FLEX"] == 1


def test_player_availability_flags():
    out = PlayerRow(player_id="1", name="x", position="WR", status="OUT")
    q = PlayerRow(player_id="2", name="y", position="WR", status="Q")
    bye = PlayerRow(player_id="3", name="z", position="WR", on_bye=True)
    ok = PlayerRow(player_id="4", name="w", position="WR")
    assert out.is_out and not out.is_questionable
    assert q.is_questionable and not q.is_out
    assert bye.is_out
    assert not ok.is_out and not ok.is_questionable


@pytest.mark.parametrize("raw,expected", [
    ("D/ST", "DST"), ("DEF", "DST"), ("PK", "K"), ("qb", "QB"), ("", ""), (None, ""),
])
def test_canonical_position(raw, expected):
    assert canonical_position(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("RB/WR/TE", "FLEX"), ("W/R/T", "FLEX"), ("OP", "SUPERFLEX"), ("BE", "BN"),
    ("BN", "BN"), ("IR", "IR"), ("D/ST", "DST"), ("DEF", "DST"), (None, "BN"),
])
def test_canonical_slot(raw, expected):
    assert canonical_slot(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("ACTIVE", ""), ("", ""), (None, ""), ("QUESTIONABLE", "QUESTIONABLE"),
    ("out", "OUT"), ("IR", "IR"),
])
def test_canonical_status(raw, expected):
    assert canonical_status(raw) == expected


# ---------------------------------------------------------------------------
# ESPN normalisation on espn-api-shaped stand-ins
# ---------------------------------------------------------------------------
def _espn_player(**kw):
    base = dict(
        playerId=4242, name="Kenneth Walker III", position="RB", proTeam="SEA",
        injuryStatus="ACTIVE", lineupSlot="RB", eligibleSlots=["RB", "RB/WR/TE", "BE"],
        stats={5: {"projected_points": 13.4, "points": 0.0}}, avg_points=12.1,
        percent_owned=97.5, on_bye_week=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_espn_player_normalises_fields():
    row = espn.normalize_player(_espn_player(), week=5)
    assert row.player_id == "4242"
    assert row.position == "RB" and row.team == "SEA"
    assert row.slot == "RB"
    assert row.eligible_slots == ["FLEX", "RB"]
    assert row.platform_projection == pytest.approx(13.4)
    assert row.season_avg == pytest.approx(12.1)
    assert row.status == ""


def test_espn_box_player_prefers_its_direct_projection():
    p = _espn_player(projected_points=15.0, slot_position="RB/WR/TE", injuryStatus="QUESTIONABLE")
    row = espn.normalize_player(p, week=5, box=True)
    assert row.platform_projection == pytest.approx(15.0)
    assert row.slot == "FLEX"
    assert row.status == "QUESTIONABLE"


def test_espn_defense_and_washington_map_correctly():
    p = _espn_player(playerId=-16028, name="Commanders D/ST", position="D/ST", proTeam="WSH",
                     eligibleSlots=["D/ST", "BE"], lineupSlot="D/ST")
    row = espn.normalize_player(p, week=5)
    assert row.position == "DST"
    assert row.team == "WAS"
    assert row.slot == "DST"


def test_espn_slot_counts_split_bench_from_starters():
    slots, bench = espn.normalize_slots({
        "QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "D/ST": 1, "K": 1, "BE": 7, "IR": 1,
    })
    assert slots == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1}
    assert bench == 7


def test_espn_identifies_my_team_by_swid_or_override():
    teams = [
        SimpleNamespace(team_id=1, owners=[{"id": "{ABC-123}", "displayName": "me"}]),
        SimpleNamespace(team_id=2, owners=[{"id": "{DEF-456}", "displayName": "them"}]),
    ]
    assert espn.identify_my_team(teams, "{abc-123}", None) == "1"
    assert espn.identify_my_team(teams, "{zzz}", None) is None
    assert espn.identify_my_team(teams, None, "2") == "2"


def test_espn_build_snapshot_uses_box_scores_and_finds_matchup():
    me = SimpleNamespace(team_id=1, team_name="Mine", owners=[{"displayName": "me"}],
                         wins=3, losses=1, ties=0, points_for=400.0, roster=[])
    them = SimpleNamespace(team_id=2, team_name="Theirs", owners=[], wins=1, losses=3,
                           ties=0, points_for=300.0, roster=[])
    box = SimpleNamespace(
        home_team=me, away_team=them,
        home_lineup=[_espn_player(playerId=1, projected_points=20.0, slot_position="QB", position="QB")],
        away_lineup=[_espn_player(playerId=2, projected_points=10.0, slot_position="RB")],
        home_projected=120.0, away_projected=95.0, home_score=0.0, away_score=0.0,
    )
    league = SimpleNamespace(
        league_id=777, year=2026, nfl_week=5, current_week=5, teams=[me, them],
        settings=SimpleNamespace(name="Test", position_slot_counts={"QB": 1, "RB": 1, "BE": 3}),
    )
    snap = espn.build_snapshot(league, 5, "espn", "1", [box], [_espn_player(playerId=9, lineupSlot="")])
    assert snap.my_team.team_id == "1"
    assert snap.opponent.team_id == "2"
    assert snap.matchup.my_platform_projection == 120.0
    assert snap.my_team.roster[0].platform_projection == 20.0   # from the box lineup
    assert snap.free_agents[0].slot == "FA"
    assert snap.slots == {"QB": 1, "RB": 1} and snap.bench_size == 3


# ---------------------------------------------------------------------------
# Yahoo normalisation on the dict shapes the library documents
# ---------------------------------------------------------------------------
def test_yahoo_roster_entry_normalises():
    row = yahoo.normalize_player({
        "player_id": 30123, "name": "Puka Nacua", "position_type": "O",
        "eligible_positions": ["WR", "W/R/T"], "selected_position": "WR",
        "status": "Q", "editorial_team_abbr": "LAR", "percent_owned": 99,
    }, projection=14.2)
    assert row.player_id == "30123" and row.position == "WR"
    assert row.team == "LA"
    assert row.slot == "WR" and row.eligible_slots == ["FLEX", "WR"]
    assert row.status == "Q" and row.is_questionable
    assert row.platform_projection == pytest.approx(14.2)


def test_yahoo_defense_is_a_dst():
    row = yahoo.normalize_player({
        "player_id": 100029, "name": "Denver", "position_type": "DT",
        "eligible_positions": ["DEF"], "editorial_team_abbr": "Den",
    })
    assert row.position == "DST" and row.team == "DEN"


def test_yahoo_positions_split_bench():
    slots, bench = yahoo.normalize_slots({
        "QB": {"count": 1}, "WR": {"count": 3}, "RB": {"count": 2}, "TE": {"count": 1},
        "W/R/T": {"count": 1}, "K": {"count": 1}, "DEF": {"count": 1},
        "BN": {"count": 6}, "IR": {"count": "2"},
    })
    assert slots["FLEX"] == 1 and slots["DST"] == 1 and slots["WR"] == 3
    assert bench == 6


def test_yahoo_matchup_parser_finds_my_pairing():
    """Yahoo's scoreboard is a nest of numbered dicts; walk it, don't index it."""
    raw = {"fantasy_content": {"league": [{}, {"scoreboard": {"0": {"matchups": {
        "0": {"matchup": {"0": {"teams": {
            "0": {"team": [[{"team_key": "449.l.1.t.3"}], {"team_points": {"total": "0.0"},
                                                          "team_projected_points": {"total": "101.5"}}]},
            "1": {"team": [[{"team_key": "449.l.1.t.7"}], {"team_points": {"total": "0.0"},
                                                          "team_projected_points": {"total": "97.2"}}]},
            "count": 2}}}},
        "1": {"matchup": {"0": {"teams": {
            "0": {"team": [[{"team_key": "449.l.1.t.2"}], {}]},
            "1": {"team": [[{"team_key": "449.l.1.t.5"}], {}]},
            "count": 2}}}},
        "count": 2}}}}]}}
    m = yahoo.parse_matchups(raw, 5, "449.l.1.t.7")
    assert m is not None
    assert m.my_team_id == "449.l.1.t.7" and m.opponent_team_id == "449.l.1.t.3"
    assert m.my_platform_projection == pytest.approx(97.2)
    assert m.opp_platform_projection == pytest.approx(101.5)
    assert yahoo.parse_matchups(raw, 5, "449.l.1.t.99") is None


def test_yahoo_oauth_file_is_built_from_env(tmp_path):
    path = yahoo.write_oauth_file(tmp_path / "o.json", {
        "client_id": "cid", "client_secret": "sec", "refresh_token": "rt",
    })
    import json

    data = json.loads(path.read_text())
    assert data["consumer_key"] == "cid" and data["refresh_token"] == "rt"
    assert data["token_time"] == 0                      # forces a refresh


def test_yahoo_oauth_file_names_what_is_missing(tmp_path):
    with pytest.raises(RuntimeError, match="YAHOO_REFRESH_TOKEN"):
        yahoo.write_oauth_file(tmp_path / "o.json", {"client_id": "a", "client_secret": "b"})


def test_yahoo_errors_are_explained():
    from streamer.league.yahoo import explain_error

    scope = explain_error(RuntimeError(
        'b\'{"error":{"description":"Please provide valid credentials. '
        'OAuth oauth_problem=\\"additional_authorization_required\\"'))
    assert "Fantasy Sports" in scope and "yahoo-auth" in scope
    # Must not trip the --skip-missing heuristic, which looks for these words.
    assert "not set" not in scope and "missing" not in scope.lower()
    assert "yahoo-auth" in explain_error(ValueError("invalid_grant"))
    assert explain_error(ValueError("something else")) == "something else"
