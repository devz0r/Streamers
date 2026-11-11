"""This season's D/ST and K picks graded against a consensus: aggregates only."""

import pandas as pd

from streamer import dstk_consensus as dc


def test_season_grade_prints_aggregates_for_a_consensus_equal_to_ours(cfg, monkeypatch, tmp_path):
    rows = []
    for w in (1, 2, 3):
        for pos in ("DST", "K"):
            for i in range(12):
                rows.append({"season": 2026, "week": w, "position": pos, "team": f"T{i}", "model_rank": i + 1,
                             "expected_points": 12.0 - i, "baseline_points": 11.0 - i * 0.5,
                             "actual_points": float((i * 7 + w) % 13), "actual_rank": float(((i * 7 + w) % 12) + 1)})
    hist = pd.DataFrame(rows)
    pcfg = cfg.for_profile("espn")
    monkeypatch.setattr(type(pcfg), "results_dir", property(lambda self: tmp_path))
    hist.to_parquet(tmp_path / "history.parquet")
    cons = hist[["season", "week", "position", "team"]].assign(rank=hist["model_rank"])
    monkeypatch.setattr(dc, "load", lambda c: cons)
    monkeypatch.setattr(dc, "normalize_team", lambda t: t)
    out = dc.season_grade(cfg, "espn", 2026)
    text = "\n".join(out)
    assert "-- DST: 3 weeks" in text and "-- K: 3 weeks" in text
    assert "ours - consensus: top5 +0.00" in text                      # the same ranking, no difference
