"""Streaming a D/ST or kicker against the best free agent's season level, replayed 2023-2025.

    python scripts/fit_stream_premium.py

Levels as in roster.unit_level; rosters hold the top 60% by level, other streamers take the week's best 20%
(or not: "you first"). DECISIONS.md, "Streaming in the season simulation".
"""
# ruff: noqa: E702
import pandas as pd

w = pd.read_parquet("data/backtest/dstk_walkforward.parquet")
CONF = {"DST": (0.2, 48.0), "K": (0.7, 8.0)}
for pos, (line, k) in CONF.items():
    d = w[w.position == pos].sort_values("expected_points", ascending=False).drop_duplicates(["season", "week", "team"])
    for n, label in ((10, "ESPN 10"), (14, "Yahoo 14")):
        held_n, ahead = round(0.6 * n), round(0.2 * n)
        rows = []
        for (s, wk), g in d.groupby(["season", "week"]):
            if wk < 3 or wk > 17:
                continue
            past = d[(d.season == s) & (d.week < wk)]
            mu = past.fantasy_points.mean()
            agg = past.groupby("team").fantasy_points.agg(["sum", "count"])
            g = g.set_index("team").copy()
            su = agg["sum"].reindex(g.index).fillna(0); c = agg["count"].reindex(g.index).fillna(0)
            g["level"] = line * g.baseline_points + (1 - line) * (su + k * mu) / (c + k)
            held = g.level.sort_values(ascending=False).index[:held_n]
            pool = g.drop(index=held)
            best_level = pool.level.max()
            for a in (0, ahead):
                pick = pool.expected_points.sort_values(ascending=False).index[a]
                rows.append({"s": s, "ahead": a, "level": best_level, "stream": pool.fantasy_points[pick],
                             "level_unit_actual": pool.fantasy_points[pool.level.idxmax()]})
        r = pd.DataFrame(rows)
        for a, g in r.groupby("ahead"):
            per = g.groupby("s").apply(lambda x: (x.stream.mean() - x.level.mean(), x.stream.mean() / x.level.mean()), include_groups=False)
            print(pos, label, "others ahead" if a else "you first", " ".join(f"{s}: +{v[0]:.2f} (x{v[1]:.3f})" for s, v in per.items()),
                  f"| best-FA level {g.level.mean():.2f}, that unit scored {g.level_unit_actual.mean():.2f}, stream {g.stream.mean():.2f}")
