"""Games on the field with no stat line are counted as zero-point games."""

from __future__ import annotations

import pandas as pd

from streamer.roster import idle


def _history():
    rows = []
    for w in (1, 2, 3):
        rows.append({"player_id": "00-A", "player_display_name": "Al Back", "position": "RB", "season": 2026,
                     "week": w, "team": "KC", "fantasy_points_ppr": 10.0, "total_fantasy_points_exp": 9.0,
                     "targets": 3.0, "carries": 12.0, "target_share": 0.1, "xtd": 0.3})
    return pd.DataFrame(rows)


def _snaps(cfg):
    sn = pd.DataFrame([
        {"game_type": "REG", "season": 2026, "week": 1, "player": "Al Back", "position": "RB", "team": "KC",
         "offense_snaps": 40, "offense_pct": 0.6},
        {"game_type": "REG", "season": 2026, "week": 4, "player": "Al Back", "position": "RB", "team": "KC",
         "offense_snaps": 9, "offense_pct": 0.12},          # on the field, no stat line
        {"game_type": "REG", "season": 2026, "week": 4, "player": "Al Back", "position": "RB", "team": "KC",
         "offense_snaps": 0, "offense_pct": 0.0},           # no snaps: not counted
        {"game_type": "REG", "season": 2026, "week": 4, "player": "Big Tackle", "position": "T", "team": "KC",
         "offense_snaps": 70, "offense_pct": 1.0},          # not a skill position
    ])
    (cfg.raw_dir).mkdir(parents=True, exist_ok=True)
    sn.to_parquet(cfg.raw_dir / "snap_counts_2026.parquet", index=False)


def test_a_snap_with_no_stat_line_becomes_a_zero_game(tmp_cfg):
    _snaps(tmp_cfg)
    tmp_cfg.raw["roster"] = {**tmp_cfg.raw["roster"], "idle_games": True}
    h = idle.with_idle_games(_history(), tmp_cfg, allow_network=False)
    added = h[h["week"] == 4]
    assert len(h) == 4 and len(added) == 1
    r = added.iloc[0]
    assert r["player_id"] == "00-A" and r["fantasy_points_ppr"] == 0.0 and r["total_fantasy_points_exp"] == 0.0
    # Games he has a stat line for are not doubled.
    assert (h["week"] == 1).sum() == 1


def test_a_replay_does_not_see_the_week_it_projects(tmp_cfg):
    _snaps(tmp_cfg)
    tmp_cfg.raw["roster"] = {**tmp_cfg.raw["roster"], "idle_games": True}
    assert len(idle.with_idle_games(_history(), tmp_cfg, 2026, 4, allow_network=False)) == 3
    assert len(idle.with_idle_games(_history(), tmp_cfg, 2026, 5, allow_network=False)) == 4


def test_switched_off_the_history_is_untouched(tmp_cfg):
    _snaps(tmp_cfg)
    tmp_cfg.raw["roster"] = {**tmp_cfg.raw["roster"], "idle_games": False}
    h = _history()
    assert idle.with_idle_games(h, tmp_cfg, allow_network=False) is h
