"""The score charts are drawn from the same draws as P(win)."""

from types import SimpleNamespace as NS

import numpy as np

from streamer.league.model import PlayerRow
from streamer.roster import sim_dist
from streamer.roster.lineup import calibrated


def _opt(rng):
    a = PlayerRow(player_id="a", name="A", position="QB", team="X")
    b = PlayerRow(player_id="b", name="B", position="RB", team="X")
    samples = np.column_stack([rng.normal(22, 7, 5000), rng.normal(12, 6, 5000)])
    opp = rng.normal(30, 9, 5000)
    return NS(best_win=NS(flat=lambda: [("QB", a), ("RB", b)]), sample_ids=["a", "b"], samples=samples,
              opp_samples=opp, opponent=object(), opp_ids=[], opp_player_samples=None)


def test_the_margin_chart_area_is_the_win_chance_the_page_shows():
    opt = _opt(np.random.default_rng(1))
    m = sim_dist.matchup(opt, "Me", "Them")
    me = opt.samples.sum(axis=1)
    raw = float((me - opt.opp_samples > 0).mean())
    assert abs(m["raw"] - raw) < 1e-3 and abs(m["p"] - float(calibrated(raw))) < 1e-3
    # The density integrates to one, and the part right of zero is the raw win share.
    d = m["margin"]
    xs = d["x0"] + np.arange(len(d["y"])) * d["dx"]
    y = np.array(d["y"])
    assert abs(y.sum() * d["dx"] - 1.0) < 0.02
    assert abs(y[xs > 0].sum() * d["dx"] - raw) < 0.03
    assert m["me"]["p10"] < m["me"]["p50"] < m["me"]["p90"] and m["names"] == ["Me", "Them"]


def test_each_player_gets_his_own_density_and_a_locked_player_a_fixed_score():
    rng = np.random.default_rng(2)
    opt = _opt(rng)
    opt.samples[:, 1] = 14.2                                  # his game is final
    got = sim_dist.players(opt)
    assert got["b"] == {"fixed": 14.2}
    assert got["a"]["p10"] < got["a"]["p50"] < got["a"]["p90"] and abs(sum(got["a"]["y"]) * got["a"]["dx"] - 1) < 0.03
