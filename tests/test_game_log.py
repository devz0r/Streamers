"""The tap-a-name game log: stat lines by position, byes, missed games, matching."""

from __future__ import annotations

import pandas as pd

from fixtures_league import snapshot
from streamer.roster import game_log
from types import SimpleNamespace as NS


def test_stat_lines_read_like_a_box_score():
    rb = {"carries": 9, "rushing_yards": 54, "rushing_tds": 1, "receptions": 2, "targets": 3, "receiving_yards": 15}
    assert game_log.line("RB", rb) == "9 car 54 yds, 2 rec (3 tgt) 15 yds, 1 TD"
    qb = {"completions": 18, "attempts": 27, "passing_yards": 231, "passing_tds": 2, "passing_interceptions": 1,
          "carries": 4, "rushing_yards": 21}
    assert game_log.line("QB", qb) == "18/27 231 yds, 2 TD, 1 INT, 4 car 21 yds"
    assert game_log.line("K", {"fg_made": 2, "fg_att": 3, "fg_long": 51, "pat_made": 1, "pat_att": 1}) == \
        "FG 2/3, long 51, XP 1/1"
    assert game_log.line("WR", {}) == ""


def _stats(rows):
    cols = ["player_id", "player_display_name", "position", "season_type", "week", "team", "opponent_team",
            "fantasy_points_ppr", "carries", "rushing_yards", "rushing_tds", "receptions", "targets",
            "receiving_yards", "receiving_tds"]
    return pd.DataFrame(rows, columns=cols)


def test_a_season_log_marks_byes_and_missed_games(monkeypatch, tmp_path):
    snap = snapshot(week=5)
    me = next(p for p in snap.my_team.roster if p.position == "RB")
    me.nfl_id = "00-1"
    me.team = "KC"
    other = "BUF"
    rows = [["00-1", me.name, "RB", "REG", 1, "KC", "BUF", 12.5, 10, 60, 1, 2, 3, 20, 0],
            ["00-1", me.name, "RB", "REG", 3, "KC", "MIA", 4.1, 6, 20, 0, 1, 1, 9, 0],
            ["00-9", "Someone Else", "WR", "REG", 1, other, "KC", 5.0, 0, 0, 0, 1, 2, 9, 0],
            ["00-9", "Someone Else", "WR", "REG", 2, other, "NYJ", 7.0, 0, 0, 0, 1, 2, 9, 0],
            ["00-9", "Someone Else", "WR", "REG", 3, other, "KC", 5.0, 0, 0, 0, 1, 2, 9, 0],
            ["00-8", "Chiefs Teammate", "WR", "REG", 1, "KC", "BUF", 5.0, 0, 0, 0, 1, 2, 9, 0],
            ["00-8", "Chiefs Teammate", "WR", "REG", 3, "KC", "MIA", 5.0, 0, 0, 0, 1, 2, 9, 0]]
    monkeypatch.setattr(game_log, "_frames", lambda cfg, season: (
        _stats(rows).assign(_team=lambda d: d["team"], _name=lambda d: d["player_display_name"].str.lower(),
                            offense_pct=0.62), None))
    out = game_log.build(snap, NS(raw_dir=tmp_path))
    rec = out["00-1"]
    weeks = {g[0]: g for g in rec["g"]}
    assert weeks[1][2] == 12.5 and weeks[1][3] == 62 and "10 car 60 yds" in weeks[1][4]
    assert weeks[2] == [2, "bye"]                    # KC did not play week 2 (BUF did)
    assert weeks[3][2] == 4.1 and len(rec["g"]) == 3
    assert game_log.merge({"a": {"n": 1}}, {"a": {"n": 2}, "b": {"n": 3}}) == {"a": {"n": 1}, "b": {"n": 3}}


def test_the_page_embeds_the_log_once_and_the_sheet_script():
    from streamer.publish import _game_log_block

    block = _game_log_block({"espn": {"_gamelog": {"1": {"n": "A </b>", "g": []}}}, "yahoo": {"_gamelog": {"1": {"n": "B"}}}})
    assert block.count('id="gamelog-data"') == 1 and "<\\/b>" in block and "caretRangeFromPoint" in block
    assert _game_log_block({"espn": {"hub": "x"}}) == ""


def test_a_week_not_yet_played_is_not_a_bye(monkeypatch, tmp_path):
    """Week 8 began on Thursday: Miami's game is on Sunday, so it has no stat
    line yet -- that is not a bye (their bye was week 6)."""
    snap = snapshot(week=8)
    me = next(p for p in snap.my_team.roster if p.position == "WR")
    me.nfl_id, me.team = "00-7", "MIA"
    rows = [["00-7", me.name, "WR", "REG", w, "MIA", "NYJ", 8.0, 0, 0, 0, 3, 5, 40, 0] for w in (1, 2, 3, 4, 5, 7)]
    rows.append(["00-5", "Thursday Guy", "WR", "REG", 8, "LA", "SF", 9.0, 0, 0, 0, 3, 5, 40, 0])
    monkeypatch.setattr(game_log, "_frames", lambda cfg, season: (
        _stats(rows).assign(_team=lambda d: d["team"], _name=lambda d: d["player_display_name"].str.lower(),
                            offense_pct=0.8), None))
    sched = [{"season": snap.season, "game_type": "REG", "week": w, "home_team": "MIA", "away_team": "NYJ"}
             for w in (1, 2, 3, 4, 5, 7, 8)]
    pd.DataFrame(sched).to_parquet(tmp_path / "schedules.parquet")
    weeks = {g[0]: g for g in game_log.build(snap, NS(raw_dir=tmp_path))["00-7"]["g"]}
    assert weeks[6] == [6, "bye"] and 8 not in weeks
