"""Do October schedule factors predict D/ST and kicker points in weeks 15-17?

    python scripts/fit_playoff_units.py

The playoff-unit cards hold a defence or kicker for weeks 15-17 on the
strength of the schedule factor (:mod:`streamer.roster.schedule`) as it
stands at the checkpoint. ``fit_schedule_dstk.py`` fitted that factor on
every later week; this asks the narrower question the cards depend on: from
checkpoints 4-9, does the factor (at the production exponents) improve the
weeks 15-17 forecast in held-out seasons, and how far should it be shrunk
(``factor ** s``, ``s`` fitted on the other seasons' weeks 15-17)? Also the
unit-level question: does a unit's mean playoff factor rank its playoff
points-over-baseline? DECISIONS.md, "Playoff-week factors for held units".
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

LATE = (15, 16, 17)
S_GRID = np.round(np.arange(0.0, 1.51, 0.05), 2)


def main() -> int:
    logging.disable(logging.WARNING)
    import fit_schedule_dstk as f

    from streamer.config import get_config
    from streamer.data.nflverse import games_frame
    from streamer.teams import normalize_team

    cfg = get_config().for_profile("espn")
    conf = (cfg.raw.get("roster") or {}).get("schedule") or {}
    lam, prev_w, k = float(conf.get("ridge", 4.0)), float(conf.get("previous_season_weight", 0.5)), \
        float(conf.get("allowed_k", 6.0))
    games = games_frame(cfg)
    games = games[games["game_type"] == "REG"].assign(
        team=lambda g: g["team"].map(normalize_team), opponent=lambda g: g["opponent"].map(normalize_team))
    walk = pd.read_parquet(f.WALK).assign(team=lambda w: w["team"].map(normalize_team))
    f.CHECKPOINTS = range(4, 10)
    d = f.signals(games, walk, lam, prev_w, k)
    d = d[d["week"].isin(LATE)]
    seasons = sorted(d["season"].unique())
    for pos in ("DST", "K"):
        c = (conf.get("positions") or {}).get(pos) or {}
        a, b = float(c.get("implied", 0.0)), float(c.get("allowed", 0.0))
        p = d[d["position"] == pos].copy()
        if p.empty:
            continue
        p["factor"] = p["env"] ** a * p["allowed"] ** b

        def sse(x: pd.DataFrame, s: float) -> float:
            return float(((x["base"] * x["factor"] ** s - x["actual"]) ** 2).sum())

        print(f"\n{pos} (implied {a}, allowed {b}): weeks 15-17 from checkpoints 4-9, {len(p)} unit-weeks")
        print(f"  factor spread: sd {p['factor'].std():.3f}, 10-90% {p['factor'].quantile(.1):.3f}-{p['factor'].quantile(.9):.3f}")
        better_prod = better_fit = 0
        tf = tp = ts = 0.0
        for s in seasons:
            tr, te = p[p["season"] != s], p[p["season"] == s]
            sh = min(S_GRID, key=lambda v: sse(tr, v))
            flat, prod, fit = sse(te, 0.0), sse(te, 1.0), sse(te, sh)
            tf, tp, ts = tf + flat, tp + prod, ts + fit
            better_prod += prod < flat
            better_fit += fit < flat
            n = len(te)
            print(f"   {s}: RMSE flat {np.sqrt(flat / n):.3f}  production {np.sqrt(prod / n):.3f}  "
                  f"shrunk s={sh:.2f} {np.sqrt(fit / n):.3f}")
        n = len(p)
        print(f"  overall: flat {np.sqrt(tf / n):.4f}  production {np.sqrt(tp / n):.4f} ({better_prod}/{len(seasons)})  "
              f"shrunk {np.sqrt(ts / n):.4f} ({better_fit}/{len(seasons)}); s on all {min(S_GRID, key=lambda v: sse(p, v)):.2f}")
        # Unit level: mean factor over the playoff weeks vs points over baseline.
        u = p.assign(ratio=p["actual"] / p["base"]).groupby(["season", "checkpoint", "team"]).agg(
            factor=("factor", "mean"), ratio=("ratio", "mean"), games=("week", "size")).reset_index()
        u = u[u["games"] >= 2]
        corr = u.groupby(["season", "checkpoint"]).apply(
            lambda g: g["factor"].rank().corr(g["ratio"].rank())).groupby("season").mean()
        print("  rank corr, mean playoff factor vs playoff points/baseline: "
              + "  ".join(f"{s} {v:+.3f}" for s, v in corr.items()))
        top = u.groupby(["season", "checkpoint"]).apply(
            lambda g: g.nlargest(5, "factor")["ratio"].mean() - g["ratio"].mean()).groupby("season").mean()
        print("  top-5 schedule units, points/baseline over average: "
              + "  ".join(f"{s} {v:+.3f}" for s, v in top.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
