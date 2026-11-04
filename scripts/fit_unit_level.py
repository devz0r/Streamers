"""A season level for defences and kickers, fitted on two seasons, scored on the third.

    python scripts/fit_unit_level.py

From the walk-forward D/ST and K rankings 2023-2025 (``baseline_points``: this
week's Vegas-only projection; ``fantasy_points``: actual), at checkpoints
weeks 3-10: predict each unit's points a game over the rest of the season
(weeks c-17) as ``a * baseline + (1 - a) * shrunk``, ``shrunk`` its points so
far this season pulled toward the league mean by ``k`` games. ``(a, k)`` on a
grid, fitted on two seasons and scored on the third (RMSE weighted by games).
DECISIONS.md, "A season level for defences and kickers".
"""
import numpy as np
import pandas as pd

w = pd.read_parquet('data/backtest/dstk_walkforward.parquet')
KS = (8, 16, 24, 32, 48, 10**9)
AS = np.round(np.arange(0, 1.01, 0.1), 1)
for pos in ("DST", "K"):
    d = w[w.position == pos].sort_values("expected_points", ascending=False).drop_duplicates(["season", "week", "team"])
    rows = []
    for s in sorted(d.season.unique()):
        for c in range(3, 11):
            past = d[(d.season == s) & (d.week < c)]
            mu = past.fantasy_points.mean()
            g = past.groupby("team").fantasy_points.agg(["sum", "count"])
            now = d[(d.season == s) & (d.week == c)].set_index("team")
            fut = d[(d.season == s) & (d.week >= c) & (d.week <= 17)].groupby("team").fantasy_points.agg(["mean", "count"])
            for t, a in fut.iterrows():
                if t not in now.index:
                    continue
                su, n = (g.loc[t, "sum"], g.loc[t, "count"]) if t in g.index else (0.0, 0)
                rows.append({"s": s, "base": now.baseline_points[t], "mu": mu, "su": su, "n": n, "a": a["mean"], "w": a["count"]})
    r = pd.DataFrame(rows)
    def err(sub, a, k):
        shr = (sub.su + k * sub.mu) / (sub.n + k)
        p = a * sub.base + (1 - a) * shr
        return float(np.sqrt(np.average((p - sub.a) ** 2, weights=sub.w)))
    print(pos)
    for held in sorted(r.s.unique()):
        tr, te = r[r.s != held], r[r.s == held]
        a, k = min(((a, k) for a in AS for k in KS), key=lambda x: err(tr, *x))
        print(f"  held {held}: fit a={a} k={k if k < 1e9 else 'inf'}  test {err(te, a, k):.3f}  vs const {err(te, 0, 10**9):.3f}  current {err(te, 1, 8):.3f}")
    a, k = min(((a, k) for a in AS for k in KS), key=lambda x: err(r, *x))
    print(f"  all seasons: a={a} k={k}")
