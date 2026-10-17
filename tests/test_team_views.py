"""Viewing the page as any team: the snapshot seen from that team, the view
files, and the picker."""

from __future__ import annotations

import json

from streamer import publish
from test_season import _league


def test_view_as_makes_that_team_yours_with_its_scheduled_opponent():
    snap = _league([1.0, 1.1, 1.0, 1.05, 0.95, 1.0])
    other = next(t for t in snap.teams if not t.is_mine)
    v = publish.view_as(snap, other.team_id)
    assert v.my_team.team_id == other.team_id
    assert sum(t.is_mine for t in v.teams) == 1
    from streamer.roster.all_lineups import pairings

    assert v.matchup is None or v.matchup.opponent_team_id == pairings(snap)[str(other.team_id)]
    # The original is untouched.
    assert snap.my_team.team_id != other.team_id


def test_view_files_are_written_and_old_ones_cleared(tmp_path):
    (tmp_path / "views" / "espn").mkdir(parents=True)
    (tmp_path / "views" / "espn" / "old.json").write_text("{}")
    panels = {"espn": {"hub": "x", "_views": {"7": {"name": "Let Kevin Cook", "full": True, "tabs": {"hub": "<p>k</p>"}},
                                              "8": {"name": "Skipped", "full": False, "tabs": None}}}}
    assert publish.write_views(tmp_path, panels) == 1
    files = sorted(p.name for p in (tmp_path / "views" / "espn").iterdir())
    assert files == ["7.json"]
    got = json.loads((tmp_path / "views" / "espn" / "7.json").read_text())
    assert got == {"name": "Let Kevin Cook", "full": True, "tabs": {"hub": "<p>k</p>"}}


def test_the_picker_lists_you_first_then_full_views_and_marks_missing_ones():
    team = {"_me": "Drakes on a Plane", "_views": {
        "3": {"name": "Zebras", "full": False, "tabs": {"hub": "z"}},
        "7": {"name": "Let Kevin Cook", "full": True, "tabs": {"hub": "k"}},
        "9": {"name": "Alphas", "full": False, "tabs": None}}}
    html = publish._team_picker("espn", team, "2026-10-17T00:00:00+00:00")
    order = [html.index(s) for s in ("Drakes on a Plane (you)", "Let Kevin Cook", "Alphas", "Zebras")]
    assert order == sorted(order)
    assert 'value="9" data-full="0" disabled>Alphas (not built this refresh)' in html
