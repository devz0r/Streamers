"""Is the quarterback projection biased low, and does an in-season correction fix it?

    python scripts/check_qb_bias.py

Replays the production quarterback projection (with the player comparisons on
and off) at weeks 5-13 of 2022-2025 and 3-6 of 2026, reports the bias by
season, level and week, and tests shifting each week by the shrunk mean error
of the season's earlier checkpoints. DECISIONS.md, "Quarterback bias".
"""

# ruff: noqa: E402, E702
import logging
import sys
import warnings

sys.path.insert(0, "src"); warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import pandas as pd

from streamer.config import get_config
from streamer.roster import ros_backtest as rb
from streamer.roster.projections import load_history

cfg = get_config().for_profile("espn")
prod = load_history(cfg)
actual = prod.groupby(["season","week","player_id"])["fantasy_points_ppr"].sum().rename("actual").reset_index()
rows = []
for off in (False, True):
    cfg.raw["roster"]["comps"] = {"QB": {"weight": 0.8 if off else 0.0, "k": 100, "since": 2015}} if off else {}
    for s, weeks in ((2022,(5,7,9,11,13)),(2023,(5,7,9,11,13)),(2024,(5,7,9,11,13)),(2025,(5,7,9,11,13)),(2026,(3,4,5,6))):
        rost, inj = rb.rosters(cfg, s), rb.injuries(cfg, s)
        for w in weeks:
            ps = rb.project(prod, cfg, s, w, rb.statuses(rost, inj, w))
            rows += [{"comps": off, "season": s, "week": w, "player_id": p.player_id, "proj": p.projection,
                      "status": p.status} for p in ps if p.position == "QB" and p.projection is not None]
d = pd.DataFrame(rows).merge(actual, on=["season","week","player_id"]).query("proj >= 3")
d["err"] = d.proj - d.actual
print(d.groupby(["comps","season"]).err.agg(["mean","size"]).round(2).unstack(0))
print("\nby projection level (comps on):")
x = d[d.comps]; x = x.assign(lvl=pd.cut(x.proj,[3,12,16,19,22,40]))
print(x.groupby("lvl",observed=True).agg(bias=("err","mean"),n=("err","size"),mae=("err",lambda e:e.abs().mean())).round(2))
print("\nweek:", x.groupby("week").err.mean().round(2).to_dict())

# In-season correction: shift each week's QB projections by the shrunk mean error of earlier checkpoint weeks of the same season.
x = d[d.comps].sort_values(["season","week"]).copy()
for K in (40, 80, 160):
    for lam in (0.5, 1.0):
        adj = []
        for (s, w), g in x.groupby(["season","week"]):
            prior = x[(x.season == s) & (x.week < w)]
            shift = prior.err.sum() / (len(prior) + K) if len(prior) else 0.0
            adj.append(g.assign(new=g.proj - lam * shift))
        a = pd.concat(adj)
        by = a.groupby("season").apply(lambda g: pd.Series({"base": g.err.abs().mean(), "new": (g.new - g.actual).abs().mean()}))
        print(f"K={K} lam={lam}: MAE base {by.base.mean():.3f} -> {by.new.mean():.3f}  per season", {int(s): f"{r.base:.3f}->{r.new:.3f}" for s, r in by.iterrows()})
# is a QB scoring-environment drift visible? average actual vs projection by season
print(x.groupby("season").agg(proj=("proj","mean"), actual=("actual","mean")).round(2))
