"""When our projection disagrees with the betting market, which side wins?

    python scripts/grade_market_edge.py

From the projection log (``results/<league>/skill_log.parquet``): each
player-week with both our model's number and the market's (prop lines
converted to fantasy points, conversion version 2+), against what he scored.
Fantasy points are skewed (most games land below the mean), so "below the
market's mean" wins about 60% for anyone; the fair test is against the
market's median, estimated as its mean plus the typical mean-to-median gap
at that level, measured on the other weeks (held out by week). A bet at
-110 needs 52.4%. DECISIONS.md, "Is there a betting edge?".
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from streamer.config import get_config  # noqa: E402
from streamer.roster.projections import load_history  # noqa: E402

for lg in ("espn", "yahoo"):
    s = pd.read_parquet(f"results/{lg}/skill_log.parquet")
    s = s.sort_values("logged_at").groupby(["season", "week", "player_id"]).tail(1)
    s = s[s.vegas_points.notna() & s.model_projection.notna() & (s.vegas_version.fillna(0) >= 2)]
    h = load_history(get_config().for_profile(lg))[["season", "week", "player_id", "fantasy_points_ppr"]]
    d = s.merge(h.rename(columns={"player_id": "nfl_id"}), on=["season", "week", "nfl_id"])
    d["gap"] = d.model_projection - d.vegas_points
    d["won"] = np.sign(d.fantasy_points_ppr - d.vegas_points) == np.sign(d.gap)
    d = d[(d.fantasy_points_ppr != d.vegas_points) & (d.gap != 0)]
    print(f"{lg}: {len(d)} player-weeks, weeks {sorted(d.week.unique())}")
    for lo in (0, 1, 2, 3, 4):
        g = d[d.gap.abs() >= lo]
        n, k = len(g), g.won.sum()
        se = np.sqrt(0.25 / n) if n else np.nan
        by = g.groupby("week").won.mean().round(2).to_dict()
        print(f"  |ours - market| >= {lo}: n {n:4d}  our side {k / n:.1%} (+/-{1.64 * se:.1%} 90%)  by week {by}")
    # direction: do we beat it more when we're higher or lower?
    for side, g in (("we're higher", d[d.gap >= 2]), ("we're lower", d[d.gap <= -2])):
        print(f"  {side} by 2+: n {len(g)}, our side {g.won.mean():.1%}")
    print("  baseline: actual below the market's mean", f"{(d.fantasy_points_ppr < d.vegas_points).mean():.1%}")
    # Against the market's median instead: a fair test. Median of the player's points ~ market mean shifted by
    # the typical mean-median gap at that level, estimated on the OTHER weeks.
    out = []
    for wk in sorted(d.week.unique()):
        tr, te = d[d.week != wk], d[d.week == wk].copy()
        bins = pd.qcut(tr.vegas_points, 6, duplicates="drop")
        gapm = (tr.groupby(bins, observed=True).apply(lambda g: (g.fantasy_points_ppr - g.vegas_points).median(), include_groups=False))
        edges = [iv.right for iv in gapm.index]
        te["med"] = te.vegas_points + [gapm.iloc[min(np.searchsorted(edges, v), len(edges) - 1)] for v in te.vegas_points]
        te["ours_side"] = np.sign(te.model_projection - te.med)
        te["res"] = np.sign(te.fantasy_points_ppr - te.med)
        te = te[(te.res != 0) & (te.ours_side != 0)]
        out.append(te)
    t = pd.concat(out)
    t["won"] = t.ours_side == t.res
    t["gapmed"] = (t.model_projection - t.med).abs()
    print("  against the market's median (held-out by week): baseline under", f"{(t.res < 0).mean():.1%}")
    for lo in (0, 1, 2, 3):
        g = t[t.gapmed >= lo]
        print(f"    disagree by {lo}+: n {len(g)}, our side {g.won.mean():.1%} (+/-{1.64*np.sqrt(.25/len(g)):.1%}), by week {g.groupby('week').won.mean().round(2).to_dict()}")
