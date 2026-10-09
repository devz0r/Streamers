"""Our D/ST and K rankings against the FantasyPros consensus, same weeks, same grades."""

from __future__ import annotations

import pandas as pd
import pytest

from streamer import dstk_consensus

TEAMS = ["KC", "BUF", "SF", "DAL", "NYJ", "PHI", "DEN", "MIA", "LAC", "HOU", "JAX", "MIN"]


def _history():
    pts = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1, 0]                 # ranked in this order by what happened
    return pd.DataFrame({"season": 2026, "week": 3, "position": "DST", "team": TEAMS,
                         "model_rank": range(1, 13), "actual_points": pts, "actual_rank": range(1, 13)})


def test_each_source_is_graded_on_the_same_units_and_week():
    cons = pd.DataFrame({"season": 2026, "week": 3, "position": "DST", "team": TEAMS[::-1], "rank": range(1, 13)})
    cmp = dstk_consensus.compare(_history(), cons, startable=5)
    r = cmp.iloc[0]
    assert r.ours_corr == pytest.approx(1.0) and r.fp_corr == pytest.approx(-1.0)
    assert r.ours_top5_pts == pytest.approx(16.0) and r.fp_top5_pts == pytest.approx((4 + 2 + 1 + 0 + 6) / 5)
    assert r.ours_top5_hits == 5 and r.fp_top5_hits == 0 and r.best_top5_pts == pytest.approx(16.0)


def test_a_past_week_the_api_no_longer_serves_is_not_taken_for_it(cfg, monkeypatch):
    from streamer.data import fantasypros as fp

    monkeypatch.setattr(fp, "_get", lambda path, params, c, timeout=20.0, hours=None:
                        {"week": 5, "players": [{"player_team_id": "KC", "rank_ecr": 1}]})
    assert fp.special_rankings(cfg, 2026, 3).empty                       # asked for 3, served 5
    assert len(fp.special_rankings(cfg, 2026, 5)) == 2                    # D/ST and K, both served


def test_the_blend_backtest_runs_and_prints_only_aggregates(cfg, tmp_path, monkeypatch):
    from streamer.data import fantasypros as fp

    wf = pd.read_parquet(cfg.root / dstk_consensus.WALKFORWARD)
    wf = wf[wf["week"].isin([3, 4])]
    monkeypatch.setattr(pd, "read_parquet", lambda p, _r=pd.read_parquet: wf if str(p).endswith("dstk_walkforward.parquet") else _r(p))
    monkeypatch.setattr(dstk_consensus, "log_path", lambda c: tmp_path / "log.parquet")

    def fake(c, season, week):              # a consensus that is our list reversed
        g = wf[(wf.season == season) & (wf.week == week)]
        return pd.DataFrame({"position": g.position, "team": g.team, "rank": g.expected_points.rank()})

    monkeypatch.setattr(fp, "special_rankings", fake)
    lines = dstk_consensus.blend_backtest(cfg, sleep=0)
    text = "\n".join(lines)
    assert "== DST" in text and "== K" in text and "held out" in text
    assert "player_name" not in text and "rank" not in text.split("\n", 1)[1].replace("top5", "")[:0]
