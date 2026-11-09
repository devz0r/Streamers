"""The score charts are drawn from the same draws as P(win), unsmoothed, and
a side with nothing left to play is one score, not a curve."""

from types import SimpleNamespace as NS

import numpy as np

from streamer.league.model import PlayerRow
from streamer.roster import sim_dist
from streamer.roster.lineup import calibrated


def _opt(rng, n=20000):
    me = rng.normal(34, 9, n)
    opp = rng.normal(30, 9, n)
    raw = float((me > opp).mean())
    samples = np.column_stack([rng.normal(22, 7, 5000), rng.normal(12, 6, 5000)])
    return NS(best_win=NS(win_probability=float(calibrated(raw))), win_totals=me, opp_totals=opp,
              sample_ids=["a", "b"], samples=samples, opponent=object(), opp_ids=[], opp_player_samples=None)


def test_the_chart_is_the_tiles_draws_and_shows_the_tiles_number():
    opt = _opt(np.random.default_rng(1))
    m = sim_dist.matchup(opt, "Me", "Them")
    raw = float((opt.win_totals > opt.opp_totals).mean())
    assert abs(m["raw"] - raw) < 1e-9 and m["p"] == round(opt.best_win.win_probability, 4) and m["n"] == 20000
    # A plain histogram: the bars' area is one, and the part right of zero is the win share.
    d = m["margin"]
    xs = d["x0"] + np.arange(len(d["y"])) * d["dx"]          # left edges
    y = np.array(d["y"])
    assert abs(y.sum() * d["dx"] - 1.0) < 1e-3
    assert abs(y[xs >= 0].sum() * d["dx"] - raw) < 0.01
    assert m["me"]["p10"] < m["me"]["p50"] < m["me"]["p90"] and m["names"] == ["Me", "Them"]


def test_bars_are_counts_not_smoothed():
    x = np.array([10.0] * 50 + [20.0] * 50)
    d = sim_dist.density(x, 2.0, 0.0, 30.0)
    nz = [i for i, v in enumerate(d["y"]) if v > 0]
    assert nz == [5, 10]                                      # two bars, nothing smeared between them


def test_a_finished_side_is_one_score_while_the_other_is_still_a_distribution():
    rng = np.random.default_rng(4)
    opt = _opt(rng)
    opt.opp_totals = np.full(20000, 112.0)                    # their players are all done
    m = sim_dist.matchup(opt, "Me", "Tegan")
    assert m["opp"]["fixed"] == 112.0 and "y" not in m["opp"]
    assert "y" in m["me"] and "fixed" not in m["me"]


def test_each_player_gets_his_own_histogram_and_a_locked_player_a_fixed_score():
    rng = np.random.default_rng(2)
    opt = _opt(rng)
    opt.samples[:, 1] = 14.2                                  # his game is final
    got = sim_dist.players(opt)
    assert got["b"]["fixed"] == 14.2
    assert got["a"]["p10"] < got["a"]["p50"] < got["a"]["p90"] and abs(sum(got["a"]["y"]) * got["a"]["dx"] - 1) < 1e-3


def test_with_nobody_left_on_either_side_it_is_a_result_not_a_curve():
    rng = np.random.default_rng(3)
    opt = _opt(rng)
    opt.win_totals = np.full(20000, 150.0)
    opt.opp_totals = np.full(20000, 138.0)
    m = sim_dist.matchup(opt, "Me", "Them")
    assert m["final"] == [150.0, 138.0] and m["p"] == 1.0 and "me" not in m
