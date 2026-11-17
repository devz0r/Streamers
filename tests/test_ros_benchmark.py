"""The rest-of-season benchmark grades our values before and after the consensus blend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from streamer.data import fantasypros as fp
from streamer.roster.scorecard import ros_benchmark


def test_the_benchmark_grades_the_blended_value_beside_ours_and_the_consensus(tmp_cfg):
    rng = np.random.default_rng(4)
    n = 40
    truth = rng.uniform(4, 20, n)
    ids = [f"n{i}" for i in range(n)]
    pos = ["WR"] * n
    ours = truth + rng.normal(0, 5, n)                    # a noisy view
    cons_rank = (-(truth + rng.normal(0, 1, n))).argsort().argsort() + 1      # a sharp one
    log_ = pd.DataFrame({"season": 2026, "week": 3, "nfl_id": ids, "position": pos, "our_ros": ours,
                         "ecr_ros_pos_rank": cons_rank})
    log_.to_parquet(fp.cache_dir(tmp_cfg) / "log.parquet")
    blended = 0.5 * ours + 0.5 * truth                    # halfway: better than ours, not as good as theirs
    sk = pd.DataFrame({"season": 2026, "week": 3, "nfl_id": ids, "ros_value": blended, "logged_at": "2026-10-01"})
    tmp_cfg.results_dir.mkdir(parents=True, exist_ok=True)
    sk.to_parquet(tmp_cfg.results_dir / "skill_log.parquet")
    hist = pd.DataFrame([{"player_id": i, "season": 2026, "week": w, "fantasy_points_ppr": t}
                         for i, t in zip(ids, truth) for w in (3, 4, 5, 6)])
    row, = ros_benchmark(tmp_cfg, hist)
    assert row["ours"] < row["used"] < row["consensus"] and row["weeks_since"] == 4
    # No final values logged for the week: the row is as before, without "used".
    (tmp_cfg.results_dir / "skill_log.parquet").unlink()
    row, = ros_benchmark(tmp_cfg, hist)
    assert "used" not in row and row["ours"] < row["consensus"]
