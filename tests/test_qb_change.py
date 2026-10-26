"""A clearly worse quarterback this week, for his receivers."""

from __future__ import annotations

import pandas as pd

from streamer.league.model import PlayerRow
from streamer.roster import qb_change


def _stats(tmp_cfg):
    rows = []
    for w in range(1, 5):          # the starter, efficient, threw weeks 1-4
        rows.append({"season": 2026, "week": w, "team": "BAL", "player_id": "star-qb", "player_display_name": "Star",
                     "position": "QB", "season_type": "REG", "attempts": 30, "passing_epa": 9.0})
        rows.append({"season": 2026, "week": w, "team": "KC", "player_id": "kc-qb", "player_display_name": "KC QB",
                     "position": "QB", "season_type": "REG", "attempts": 30, "passing_epa": 3.0})
    rows.append({"season": 2025, "week": 10, "team": "BAL", "player_id": "backup-qb", "player_display_name": "Backup",
                 "position": "QB", "season_type": "REG", "attempts": 300, "passing_epa": -45.0})
    rows.append({"season": 2025, "week": 10, "team": "DAL", "player_id": "avg-qb", "player_display_name": "Avg",
                 "position": "QB", "season_type": "REG", "attempts": 300, "passing_epa": 10.0})
    pd.DataFrame(rows).to_parquet(tmp_cfg.raw_dir / "player_stats_2026.parquet")


def _hist():
    return pd.DataFrame([{"season": 2026, "week": w, "team": t, "player_id": pid, "position": "WR",
                          "fantasy_points_ppr": 12.0} for w in range(1, 5)
                         for t, pid in (("BAL", "bal-wr"), ("KC", "kc-wr"))])


def test_receivers_lose_part_of_their_week_to_a_clearly_worse_backup(tmp_cfg, monkeypatch):
    """The mechanism, switched on (the shipped config has it off: inside the
    full pipeline the cut did not hold up)."""
    monkeypatch.setattr(qb_change, "conf", lambda _c: {"positions": ["WR"], "worse_sd": 0.5, "cut": 0.15,
                                                     "shrink": 200, "recent": 6})
    _stats(tmp_cfg)
    star = PlayerRow(player_id="s", name="Star", position="QB", team="BAL", status="OUT", depth_order=1)
    backup = PlayerRow(player_id="b", name="Backup", position="QB", team="BAL", depth_order=2)
    kc = PlayerRow(player_id="k", name="KC QB", position="QB", team="KC", depth_order=1)
    bal_wr = PlayerRow(player_id="w", name="Bal WR", position="WR", team="BAL")
    kc_wr = PlayerRow(player_id="x", name="KC WR", position="WR", team="KC")
    bal_te = PlayerRow(player_id="t", name="Bal TE", position="TE", team="BAL")
    ids = {"s": "star-qb", "b": "backup-qb", "k": "kc-qb", "w": "bal-wr", "x": "kc-wr", "t": "bal-wr"}
    got = qb_change.factors([star, backup, kc, bal_wr, kc_wr, bal_te], ids, _hist(), tmp_cfg, 2026, 5)
    assert set(got) == {"w"}                                  # his starter is out; KC's plays; TEs untouched
    assert got["w"][0] == 0.85 and "Backup is in line to start" in got["w"][1]
    # The starter healthy: nothing changes.
    star.status = ""
    assert qb_change.factors([star, backup, kc, bal_wr, kc_wr], ids, _hist(), tmp_cfg, 2026, 5) == {}


def test_the_shipped_config_leaves_receivers_alone(tmp_cfg):
    _stats(tmp_cfg)
    star = PlayerRow(player_id="s", name="Star", position="QB", team="BAL", status="OUT", depth_order=1)
    backup = PlayerRow(player_id="b", name="Backup", position="QB", team="BAL", depth_order=2)
    bal_wr = PlayerRow(player_id="w", name="Bal WR", position="WR", team="BAL")
    ids = {"s": "star-qb", "b": "backup-qb", "w": "bal-wr"}
    assert qb_change.factors([star, backup, bal_wr], ids, _hist(), tmp_cfg, 2026, 5) == {}
