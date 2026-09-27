"""Blending sportsbook-implied points into projections."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fixtures_league import snapshot
from streamer.league.model import PlayerRow
from streamer.roster.vegas import blend_market, fit_market_weight, vegas_lineup


def test_blend_moves_toward_the_market_by_book_depth(cfg):
    snap = snapshot(week=3)
    a, b = snap.my_team.roster[4], snap.my_team.roster[5]       # WR 14.0, WR 11.0
    a.vegas_points, a.vegas_books = 20.0, 3
    b.vegas_points, b.vegas_books = 20.0, 1
    blend_market(snap, cfg, weight=0.4)
    assert a.projection == 14.0 + 0.4 * 6.0 and a.pre_market_projection == 14.0
    assert b.projection == round(11.0 + 0.4 * 0.6 * 9.0, 2)    # one book counts for less


def test_blend_respects_injury_tags_and_locks(cfg):
    q = PlayerRow(player_id="q", name="q", position="WR", projection=7.5, play_probability=0.75,
                  vegas_points=14.0, vegas_books=3)
    done = PlayerRow(player_id="d", name="d", position="WR", projection=10.0, locked=True,
                     actual_points=28.4, vegas_points=12.0, vegas_books=3)
    snap = snapshot(week=3)
    snap.free_agents = [q, done]
    blend_market(snap, cfg, weight=0.5)
    assert q.projection == round(0.75 * (0.5 * 10.0 + 0.5 * 14.0), 2)
    assert done.projection == 10.0 and done.market_weight == 0.0


def test_weight_is_the_prior_until_enough_games(tmp_cfg):
    w, n = fit_market_weight(tmp_cfg, history=pd.DataFrame(columns=["player_id", "season", "week",
                                                                     "fantasy_points_ppr"]))
    assert w == 0.4 and n == 0


def test_weight_is_fitted_and_shrunk_once_the_log_fills(tmp_cfg):
    rng = np.random.default_rng(0)
    n = 900
    truth = rng.uniform(4, 20, n)
    log_ = pd.DataFrame({
        "season": 2026, "week": np.repeat(np.arange(1, 10), 100), "player_id": [f"p{i}" for i in range(n)],
        "nfl_id": [f"n{i}" for i in range(n)], "projection": truth + rng.normal(0, 4, n),
        "vegas_points": truth + rng.normal(0, 1, n),              # a much sharper market
    })
    path = tmp_cfg.results_dir / "skill_log.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    log_.to_parquet(path)
    hist = pd.DataFrame({"player_id": log_["nfl_id"], "season": 2026, "week": log_["week"],
                         "fantasy_points_ppr": truth + rng.normal(0, 0.5, n)})
    w, used = fit_market_weight(tmp_cfg, history=hist)
    assert used == n and 0.7 < w < 0.95                          # well above the 0.4 prior


def test_market_lineup_keeps_locked_starters_and_final_scores(cfg):
    snap = snapshot(week=3)
    roster = {p.player_id: p for p in snap.my_team.roster}
    roster["9"].locked, roster["9"].actual_points = True, 1.5    # TE starter, played badly
    roster["10"].vegas_points = 12.0                            # bench TE the market loves
    roster["7"].locked = True                                   # bench WR, game started
    roster["7"].vegas_points = 30.0
    lu = vegas_lineup(snap, cfg)
    assert [p.player_id for p in lu.starters["TE"]] == ["9"]   # locked in his slot
    assert "7" not in lu.player_ids                             # cannot come off the bench
