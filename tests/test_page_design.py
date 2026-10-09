"""The page's look: photos, chips, range bars, the phone's More sheet and the
home-screen files -- additions that must never cost the page any content."""

from __future__ import annotations


def test_photos_come_from_espn_ids_and_a_defence_shows_its_logo():
    from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow
    from streamer.roster import faces

    def league(platform, players):
        return LeagueSnapshot(platform=platform, profile=platform, league_id="1", league_name="L", season=2026, week=5,
                              slots={}, bench_size=5, teams=[TeamRow(team_id="1", name="Me", roster=players, is_mine=True)],
                              free_agents=[], matchup=None, synced_at="")

    espn = league("espn", [PlayerRow(player_id="3054211", name="Derrick Henry", position="RB", team="BAL"),
                           PlayerRow(player_id="-16020", name="Jets D/ST", position="DST", team="NYJ")])
    assert faces.attach(espn, {}) == 2
    henry, jets = espn.teams[0].roster
    assert henry.photo.endswith("/players/full/3054211.png") and jets.photo.endswith("/500/nyj.png")
    yahoo = league("yahoo", [PlayerRow(player_id="nfl.p.30121", name="Derrick Henry", position="RB", team="BAL"),
                             PlayerRow(player_id="nfl.p.1", name="Nobody Known", position="WR", team="LA")])
    faces.attach(yahoo, {"derrick henry": "3054211"})
    assert yahoo.teams[0].roster[0].photo.endswith("3054211.png") and yahoo.teams[0].roster[1].photo == ""
    assert faces.logo("LA").endswith("/lar.png") and faces.logo("WAS").endswith("/wsh.png")


def test_chips_bars_and_photos_render_and_escape():
    from streamer.league.model import PlayerRow
    from streamer.roster.page import face, pos_chip, range_bar

    assert pos_chip("WR") == '<span class="pc pc-WR">WR</span>' and 'pc-DST">D/ST' in pos_chip("D/ST")
    assert pos_chip(None) == "" and range_bar(None, 10) == ""
    bar = range_bar(7, 24)
    assert "left:20%" in bar and "width:49%" in bar
    assert "width:2%" in range_bar(50, 60)                       # off the scale: a sliver at the end, not nothing
    p = PlayerRow(player_id="1", name="X", position="WR", photo='https://x/"><script>.png')
    assert "<script>" not in face(p) and 'onerror="this.remove()"' in face(p)
    assert face(PlayerRow(player_id="2", name="Y", position="WR")) == ""


def test_phone_tabs_put_the_rest_behind_more_and_light_it_when_one_is_open():
    from streamer.publish import PHONE_TABS, _section_rules, _section_tabs

    html = _section_tabs("espn", {"hub": "h", "lineup": "l", "roster": "r", "model": "m"})
    assert 'id="more-espn"' in html and 'class="more-btn" for="more-espn"' in html
    sheet = html.split('class="more-sheet"')[1]
    assert 'for="sec-espn-roster"' in sheet and 'for="sec-espn-model"' in sheet and 'for="sec-espn-hub"' not in sheet
    assert set(PHONE_TABS) >= {"hub", "lineup"}
    assert ".more-btn{color:var(--accent);}" in _section_rules(["espn"])
    assert "more-espn" not in _section_tabs("espn", {"hub": "h", "lineup": "l"})   # nothing to hide: no More


def test_publishing_writes_the_home_screen_files(tmp_cfg):
    import json

    import numpy as np
    import pandas as pd

    from streamer.publish import ASSETS, publish_profiles
    from streamer.rankings import Rankings

    frame = pd.DataFrame({
        "season": 2026, "week": 5, "position": "DST", "team": ["KC", "BUF"], "opponent": ["DEN", "MIA"],
        "display_name": ["Chiefs D/ST", "Bills D/ST"], "is_home": 1, "rank": [1, 2],
        "expected_points": [9.0, 8.0], "floor": [3.0, 2.0], "ceiling": [15.0, 14.0], "p_top12": [0.7, 0.6],
        "expected_points_allowed": 20.0, "estimator": "ridge", "rationale": "because", "two_week_hold": False,
        "next_rank": np.nan, "next_opponent": None})
    publish_profiles({"espn": Rankings(season=2026, week=5, dst=frame, kicker=frame.copy(), context=None)}, tmp_cfg)
    manifest = json.loads((tmp_cfg.docs_dir / "manifest.webmanifest").read_text())
    assert manifest["display"] == "standalone" and manifest["icons"]
    for name in ASSETS:
        assert (tmp_cfg.docs_dir / name).stat().st_size > 0
    html = (tmp_cfg.docs_dir / "index.html").read_text()
    assert 'rel="manifest"' in html and 'rel="apple-touch-icon"' in html and '<time class="ago"' in html
