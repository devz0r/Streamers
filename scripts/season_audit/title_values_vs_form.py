"""Season audit, part 3: do our season values forecast the rest of the season better than this season's points a game?
Run: PYTHONPATH=src python scripts/season_audit/title_values_vs_form.py
"""
import numpy as np, pandas as pd
from streamer.config import load_config
from streamer.roster.projections import load_history
SK = ("QB", "RB", "WR", "TE")
for lg in ("espn", "yahoo"):
    cfg = load_config().for_profile(lg)
    hist = load_history(cfg); hist = hist[hist.season == 2026]
    last = int(hist.week.max())
    log = pd.read_parquet(f"results/{lg}/skill_log.parquet"); log = log[log.season == 2026]
    lastlog = log.sort_values("logged_at").drop_duplicates(["week", "player_id"], keep="last")
    rows = []
    for w in range(4, last - 1):
        before = hist[hist.week <= w].groupby("player_id").fantasy_points_ppr.agg(["mean", "count"])
        after = hist[(hist.week > w) & (hist.week <= last)].groupby("player_id").fantasy_points_ppr.agg(["mean", "count"])
        s = lastlog[(lastlog.week == w) & lastlog.position.isin(SK) & lastlog.ros_value.notna() & lastlog.nfl_id.notna()]
        j = s.merge(before, left_on="nfl_id", right_index=True).merge(after, left_on="nfl_id", right_index=True, suffixes=("_before", "_after"))
        j = j[(j.count_before >= 3) & (j.count_after >= 2) & (j.ros_value >= 4)]
        rows.append(j.assign(w=w)[["w", "nfl_id", "position", "ros_value", "mean_before", "count_before", "mean_after", "count_after"]])
    d = pd.concat(rows)
    print(f"\n================ {lg}: {len(d)} player-checkpoints (weeks 4-{last - 2}), {d.nfl_id.nunique()} players")
    for pos in SK + ("all",):
        x = d if pos == "all" else d[d.position == pos]
        rv = x.groupby("w").apply(lambda g: g.ros_value.corr(g.mean_after, method="spearman")).mean()
        rp = x.groupby("w").apply(lambda g: g.mean_before.corr(g.mean_after, method="spearman")).mean()
        mv, mp = (x.ros_value - x.mean_after).abs().mean(), (x.mean_before - x.mean_after).abs().mean()
        best = min(((a, (a * x.ros_value + (1 - a) * x.mean_before - x.mean_after).abs().mean()) for a in np.linspace(0, 1, 11)), key=lambda t: t[1])
        print(f"  {pos:3s} n={len(x):4d}  rank corr: value {rv:.3f}, points so far {rp:.3f} | miss: value {mv:.2f}, points so far {mp:.2f}, "
              f"best blend {best[0]:.1f} value -> {best[1]:.2f}")
    # by checkpoint, does either side win consistently?
    by = d.groupby("w").apply(lambda g: pd.Series({"value": (g.ros_value - g.mean_after).abs().mean(),
                                                   "so_far": (g.mean_before - g.mean_after).abs().mean(), "n": len(g)}))
    print("  miss by checkpoint:"); print(by.round(2).to_string())
