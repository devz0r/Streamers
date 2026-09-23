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
    # It must point at the access programme, not at a checkbox that Yahoo
    # removed from the app form in 2026.
    assert "sports.yahoo.com/developer/access" in scope
    assert "Client ID" in scope
    assert "no box to tick" in scope.lower()      # says so, rather than telling you to
    # Must not trip the --skip-missing heuristic, which looks for these words.
    assert "not set" not in scope and "missing" not in scope.lower()
    assert "yahoo-auth" in explain_error(ValueError("invalid_grant"))
    assert explain_error(ValueError("something else")) == "something else"


def test_ir_slot_eligibility_gives_the_roster_the_benefit_of_the_doubt():
    """An OUT player may sit on IR; a day-to-day one may not."""
    from streamer.league.model import PlayerRow

    def row(status, slot="IR"):
        return PlayerRow(player_id="x", name="x", position="RB", status=status, slot=slot)

    assert row("OUT").ir_slot_is_valid
    assert row("INJURY_RESERVE").ir_slot_is_valid
    assert row("SUSPENSION").ir_slot_is_valid
    assert not row("DAY_TO_DAY").ir_slot_is_valid
    assert not row("QUESTIONABLE").ir_slot_is_valid
    assert not row("").ir_slot_is_valid
    # Not in an IR slot at all: nothing to complain about.
    assert row("DAY_TO_DAY", slot="BN").ir_slot_is_valid
    # OUT is week-to-week, so it must not zero rest-of-season value.
    assert not row("OUT").is_long_term_out
    assert row("INJURY_RESERVE").is_long_term_out


# ---------------------------------------------------------------------------
# Yahoo browser-session reader
# ---------------------------------------------------------------------------
def test_probe_describes_shape_without_leaking_content():
    """The probe reports structure; it must not echo cookies or page text."""
    from streamer.league.yahoo_web import ProbeResult, _describe_tables, embedded_json

    html = """
    <html><head><title>My Team - Yahoo Fantasy</title></head><body>
    <script>root.App.main = {"context":{"a":1},"league":{"b":2}};</script>
    <table id="statTable0" class="Table"><tr><th>Player</th><th>Proj</th></tr>
    <tr><td>Secret Player</td><td>12.3</td></tr></table>
    </body></html>"""
    blob = embedded_json(html)
    assert blob is not None and sorted(blob) == ["context", "league"]

    tables = _describe_tables(html)
    assert len(tables) == 1
    assert "statTable0" in tables[0] and "rows=2" in tables[0]
    assert "Player" in tables[0]                     # headers are structure
    assert "Secret Player" not in tables[0]          # cell values are not

    res = ProbeResult(url="u", status=200, title="t", tables=tables)
    rendered = "\n".join(res.lines())
    assert "Secret Player" not in rendered


def test_logged_out_detection():
    from types import SimpleNamespace

    from streamer.league.yahoo_web import looks_logged_out

    assert looks_logged_out(SimpleNamespace(
        url="https://login.yahoo.com/?done=x", text=""))
    assert not looks_logged_out(SimpleNamespace(
        url="https://football.fantasysports.yahoo.com/f1/123/4",
        text="<html>fantasy roster</html>"))


def test_unknown_embedded_shape_is_none_not_an_error():
    from streamer.league.yahoo_web import embedded_json

    assert embedded_json("<html>nothing here</html>") is None
    assert embedded_json("root.App.main = {not json};") is None


def test_cookie_paste_formats_all_normalise():
    """Browsers offer cookies in several shapes; none is the one HTTP wants."""
    from streamer.league.yahoo_web import cookie_names, normalize_cookie

    header = "A1=d=AQABxyz; T=z=abc&sk=DA ; SSL=1"
    table = (
        "_ygpc\tVALUE1\t.yahoo.com\t/\t9/15/2027, 10:35:45 AM\t85 B\t✓\n"
        "A1\td=AQABxyz\t.yahoo.com\t/\t9/15/2027\t120 B\t✓\n"
        "T\tz=abc&sk=DA\t.yahoo.com\t/\tSession\t60 B\t✓"
    )
    per_line = "A1=d=AQABxyz\nT=z=abc&sk=DA\nSSL=1"

    assert normalize_cookie(header) == "A1=d=AQABxyz; T=z=abc&sk=DA; SSL=1"
    assert normalize_cookie(per_line) == "A1=d=AQABxyz; T=z=abc&sk=DA; SSL=1"
    # The storage-table paste keeps names and values, drops the metadata columns.
    out = normalize_cookie(table)
    assert cookie_names(out) == ["_ygpc", "A1", "T"]
    assert "yahoo.com" not in out and "85 B" not in out and "2027" not in out
    # Values containing '=' survive intact.
    assert "T=z=abc&sk=DA" in out


def test_cookie_normalisation_edge_cases():
    from streamer.league.yahoo_web import normalize_cookie

    assert normalize_cookie("") == ""
    assert normalize_cookie("   ") == ""
    assert normalize_cookie("garbage with no pairs") == ""
    # A repeated name keeps the last value, as a browser would.
    assert normalize_cookie("A=1; A=2") == "A=2"
    # Trailing semicolons and blank lines are tolerated.
    assert normalize_cookie("A=1;\n\nB=2;") == "A=1; B=2"


def test_analytics_cookies_can_be_dropped():
    from streamer.league.yahoo_web import cookie_names, normalize_cookie

    raw = "A1=d=x; _ga=GA1.2.3; _gid=GA1.2.4; T=z=y; __gads=abc"
    assert cookie_names(normalize_cookie(raw, drop_junk=True)) == ["A1", "T"]


def test_detail_skeleton_publishes_nothing_identifying():
    """Workflow logs on a public repo are public: names, emails and tokens
    must not survive; structural codes the parser needs must."""
    from streamer.league.yahoo_web import detail_page

    html = """
    <html><body>
    <a href="/f1/555/4">My Team</a>
    <script>var crumbData = {"crumb": "aB3xQ9zLmN0pRsT"};</script>
    <table id="statTable0"><thead>
      <tr><th rowspan="2">Pos</th><th colspan="2">Fantasy</th><th>Devon's Team</th></tr>
      <tr><th>Proj</th><th>% Start</th></tr></thead><tbody>
      <tr data-tst="row">
        <td class="pos-label" data-pos="QB"><span>QB</span></td>
        <td class="player">
          <a class="name" href="https://sports.yahoo.com/nfl/players/30123/news?x=1"
             id="playernote-9265"
             data-ys-playerid="30123" title="Joshua Allenby - Player Notes">Joshua Allenby</a>
          <span class="Fz-xxs">BUF - QB</span>
          <abbr class="F-injury" title="Questionable">Q</abbr>
        </td>
        <td>owner.person@example.com</td>
        <td>23.45</td>
      </tr>
    </tbody></table></body></html>"""
    out = "\n".join(detail_page(html, "555"))

    assert "Devon" not in out           # a team name in a header is scrubbed
    for leaked in ("Joshua", "Allenby", "example.com", "owner.person", "aB3xQ9zLmN0pRsT",
                   "30123", "Player Notes", "9265"):
        assert leaked not in out, leaked
    for kept in ('"QB"', "BUF - QB", '"Q"', 'title="Questionable"', "F-injury",
                 'data-tst="row"', 'data-pos="QB"', "/nfl/players/«n»/news", "«n»",
                 "team id 4", "crumbData", "Pos{r2}", "% Start", "Proj"):
        assert kept in out, kept
