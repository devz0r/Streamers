"""A player's season value against last week's: the change, its band, and the history."""

from __future__ import annotations

import pandas as pd

from fixtures_league import snapshot
from streamer.league.model import PlayerRow
from streamer.roster import value_trend as vt
from streamer.roster.page import trend


def test_bands_are_absolute_points_a_game():
    assert vt.level(0.2) == 0 and vt.level(-0.29) == 0
    assert vt.level(0.5) == 1 and vt.level(1.5) == 2 and vt.level(2.4) == 3
    assert vt.level(-0.5) == -1 and vt.level(-3.0) == -3


def test_last_weeks_value_comes_from_the_history_then_the_projection_log(tmp_path):
    from types import SimpleNamespace

    cfg = SimpleNamespace(results_dir=tmp_path)
    snap = snapshot(week=5)
    players = [p for p in snap.all_players() if p.position in ("RB", "WR")][:3]
    for i, p in enumerate(players):
        p.ros_value = 6.0 + i
    # Week 4 only in the projection log: it stands in.
    pd.DataFrame({"season": snap.season, "week": 4, "player_id": [players[0].player_id],
                  "ros_value": [4.0]}).to_parquet(tmp_path / "skill_log.parquet")
    assert vt.attach(snap, cfg) == 1 and players[0].ros_prev == 4.0 and vt.change(players[0]) == 2.0
    # This week recorded, overwritten in place on a second refresh.
    assert vt.record(snap, cfg) >= 3
    players[1].ros_value = 9.5
    vt.record(snap, cfg)
    h = pd.read_parquet(tmp_path / vt.HISTORY)
    mine = h[h.player_id == players[1].player_id]
    assert len(mine) == 1 and float(mine.ros_value.iloc[0]) == 9.5
    # Next week compares with this week's last value.
    snap.week = 6
    for p in players:
        p.ros_prev = None
    vt.attach(snap, cfg)
    assert players[1].ros_prev == 9.5 and not players[1].ros_prev_model_changed


def test_the_badge_shows_direction_and_size():
    p = PlayerRow(player_id="j", name="Jonah Coleman", position="RB", ros_value=8.4)
    assert trend(p) == ""                                  # no history: nothing shown
    p.ros_prev = 6.0
    html = trend(p)
    assert 'class="tr tr+3"' in html and "2.4" in html and "6.0 &rarr; 8.4" in html
    p.ros_prev = 8.3
    assert 'class="tr tr0"' in trend(p)
    p.ros_prev, p.ros_prev_model_changed = 9.0, True
    assert 'tr-1' in trend(p) and "the model changed" in trend(p)
    assert vt.payload(p) == [-0.6, -1, 9.0]
