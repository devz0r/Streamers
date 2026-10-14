"""Rest-of-season player comparisons: do similar past players improve the season value?

    python scripts/fit_ros_comps.py

Season-long moves (waivers, trades, title odds) run on each player's season
value: our forecast of his points per game for the rest of the season. This
asks whether the players most like him at the same point of earlier
seasons, and how their seasons finished, improve that forecast.

At each checkpoint (a season and week), a player is described by his games
before that week only:

* season so far: games played, PPR points and nflverse expected points per
  game, targets and carries per game, target share;
* the last four games: expected and actual points per game;
* last season: points and expected points per game (0 with a flag if he
  had fewer than four games);
* career games since 2015, and the checkpoint week.

His comparison forecast is the mean rest-of-season points per game of the
``k`` nearest player-checkpoints at his position, from *earlier seasons only*
(the same season's endings would be the future), pool back to 2015. It is
graded against our production season value, replayed through
``ros_backtest`` at weeks 3, 4, 5, 6, 8 and 10 of 2022-2025, on players
active (no injury tag beyond questionable) with a season value of 3+ and 3+
games played after the checkpoint. ``k`` and the weight are picked on three
seasons and scored on the fourth. Kept on the usual rule: better overall and
worse in at most one season. DECISIONS.md, "Rest-of-season comparisons".
"""

from __future__ import annotations

import itertools
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

POOL_FROM = 2015
SEASONS = (2022, 2023, 2024, 2025)
CHECKPOINTS = (3, 4, 5, 6, 8, 10)
POOL_WEEKS = tuple(range(3, 11))
LAST_WEEK = 17
K_GRID = (25, 50, 100, 200, 400)
W_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)
# The production definitions (streamer.roster.comps), so test and page agree.
from streamer.roster.comps import ROS_FEATURES as FEATURES  # noqa: E402
from streamer.roster.comps import ros_table as build  # noqa: E402


def nearest(pool: pd.DataFrame, rows: pd.DataFrame, ks) -> pd.DataFrame:
    mu, sd = pool[FEATURES].mean(), pool[FEATURES].std().replace(0, 1.0)
    a = ((pool[FEATURES] - mu) / sd).to_numpy(np.float64)
    b = ((rows[FEATURES] - mu) / sd).to_numpy(np.float64)
    d = (b * b).sum(1)[:, None] - 2 * b @ a.T + (a * a).sum(1)[None, :]
    order = np.argsort(d, axis=1)[:, :max(ks)]
    y = pool["ros_pg"].to_numpy(float)
    out = rows[["season", "week", "player_id"]].copy()
    for k in ks:
        out[f"c{k}"] = y[order[:, :k]].mean(1)
    return out


def score(d: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for s, g in d.groupby("season"):
        rk = g.groupby(["week", "position"]).apply(lambda x: x[col].rank().corr(x["actual"].rank()))
        rows.append({"season": s, "mae": float((g[col] - g["actual"]).abs().mean()),
                     "bias": float((g[col] - g["actual"]).mean()), "rank": float(rk.mean())})
    return pd.DataFrame(rows).set_index("season")


def main() -> int:
    logging.disable(logging.WARNING)
    from streamer.config import get_config
    from streamer.roster import comps
    from streamer.roster import ros_backtest as rb
    from streamer.roster.projections import load_history

    cfg = get_config().for_profile("espn")
    deep = comps.deep_history(cfg, POOL_FROM)
    feats = build(deep, range(POOL_FROM, max(SEASONS) + 1), POOL_WEEKS)
    print(f"player-checkpoints {len(feats)} ({feats.season.min()}-{feats.season.max()}), "
          f"with a rest-of-season outcome {feats.ros_pg.notna().sum()}")

    prod = load_history(cfg)
    # Our season value as it stands without the comparison (the page applies it).
    cfg.raw.setdefault("roster", {})["ros_comps"] = {"weight": 0.0}
    ours = []
    for s in SEASONS:
        rost, inj = rb.rosters(cfg, s), rb.injuries(cfg, s)
        for w in CHECKPOINTS:
            ps = rb.project(prod, cfg, s, w, rb.statuses(rost, inj, w))
            ours += [{"season": s, "week": w, "player_id": p.player_id, "position": p.position, "status": p.status,
                      "ours": float(p.ros_value)} for p in ps if p.ros_value is not None]
    ours = pd.DataFrame(ours)
    print(f"our season values: {len(ours)}")

    test = feats[feats.season.isin(SEASONS) & feats.week.isin(CHECKPOINTS)]
    d = ours.merge(test.rename(columns={"position": "pos_h"}), on=["season", "week", "player_id"])
    d = d[(d.ours >= 3) & d.status.isin(["", "Q"]) & d.ros_pg.notna()].rename(columns={"ros_pg": "actual"})
    cs = []
    for (s, pos), t in d.groupby(["season", "position"]):
        pool = feats[(feats.season < s) & (feats.position == pos) & feats.ros_pg.notna()]
        cs.append(nearest(pool, t, K_GRID))
    d = d.merge(pd.concat(cs), on=["season", "week", "player_id"])
    print(f"graded: {len(d)} player-checkpoints")

    base = score(d, "ours")
    print(f"  ours        MAE {base.mae.mean():.4f} rank {base['rank'].mean():.4f} bias {base.bias.mean():+.3f}  "
          + "  ".join(f"{s}: {r.mae:.3f}/{r['rank']:.3f}" for s, r in base.iterrows()))
    for k in K_GRID:
        sc = score(d, f"c{k}")
        print(f"  comps k={k:<4}MAE {sc.mae.mean():.4f} rank {sc['rank'].mean():.4f} bias {sc.bias.mean():+.3f}")

    def loso(frame):
        out, picks = [], []
        for s in SEASONS:
            tr, te = frame[frame.season != s], frame[frame.season == s]
            k, w = min(itertools.product(K_GRID, W_GRID), key=lambda kw: float(
                (((1 - kw[1]) * tr.ours + kw[1] * tr[f"c{kw[0]}"]) - tr.actual).abs().mean()))
            picks.append(f"{s}: k {k} w {w:.2f}")
            out.append(te.assign(blend=(1 - w) * te.ours + w * te[f"c{k}"]))
        return pd.concat(out), picks

    b, picks = loso(d)
    bl = score(b, "blend")
    better = int((bl.mae < base.mae).sum())
    keep = bl.mae.mean() < base.mae.mean() and better >= len(bl) - 1 and bl["rank"].mean() >= base["rank"].mean()
    print(f"\nblend, all positions: MAE {base.mae.mean():.4f} -> {bl.mae.mean():.4f} ({better}/{len(bl)} better)  "
          f"rank {base['rank'].mean():.4f} -> {bl['rank'].mean():.4f}  [{'; '.join(picks)}]")
    for s in bl.index:
        print(f"    {s}: MAE {base.mae[s]:.3f} -> {bl.mae[s]:.3f}   rank {base['rank'][s]:.3f} -> {bl['rank'][s]:.3f}")
    print(f"VERDICT all positions: {'KEEP' if keep else 'DROP'}")
    for pos, g in d.groupby("position"):
        bp, pk = loso(g)
        b0, b1 = score(bp, "ours"), score(bp, "blend")
        nb = int((b1.mae < b0.mae).sum())
        ok = b1.mae.mean() < b0.mae.mean() and nb >= len(b1) - 1 and b1["rank"].mean() >= b0["rank"].mean()
        print(f"  {pos}: n {len(g)}  MAE {b0.mae.mean():.3f} -> {b1.mae.mean():.3f} ({nb}/{len(b1)})  "
              f"rank {b0['rank'].mean():.3f} -> {b1['rank'].mean():.3f}  [{'; '.join(pk)}]  {'KEEP' if ok else 'DROP'}")
    if len(sys.argv) > 1:
        d.to_parquet(sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
