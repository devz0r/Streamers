"""Similar past player-weeks for QB/RB/WR/TE: do they improve the projection?

    python scripts/fit_player_comps.py

The kicker version (scripts/fit_analogs.py) works: the mean score of a
kicker's 100 most similar earlier games, blended with the model, beat the
model in every season. This asks the same of skill players. Each
player-week is described only by what was known before it:

* opportunity: nflverse expected points per game over his last 4 and last
  8 games (any season), targets, carries and target share over the last 4;
* production: PPR points per game over the last 4;
* the game: his team's implied total and the spread.

Its comparison forecast is the mean PPR score of the ``k`` nearest
player-weeks at the same position, strictly before the week being
projected, from a pool back to 2015. It is judged against our production
model replayed on the games before each week (as ``ros_backtest`` does;
weeks 4-14, every other week, 2023-2025), alone and blended, with ``k``
and the weight picked on two seasons and scored on the third. Graded on
players who played and were projected 3+ points. Kept on the usual rule:
better overall and worse in at most one season. DECISIONS.md, "Player
comparisons".
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
SEASONS = (2023, 2024, 2025)
MODEL_WEEKS = (4, 6, 8, 10, 12, 14)
K_GRID = (25, 50, 100, 200, 400)
W_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)
FEATURES = ["xfp4", "xfp8", "fp4", "tgt4", "car4", "tsh4", "implied", "spread"]
MIN_GAMES = 3


def history(cfg) -> pd.DataFrame:
    """Weekly skill-player rows back to ``POOL_FROM`` (load_history over more seasons)."""
    from streamer.roster import projections

    seasons = list(range(POOL_FROM, int(cfg.current_season) + 1))
    saved = cfg.season["train_seasons"]
    cfg.season["train_seasons"] = seasons
    try:
        return projections.load_history(cfg)
    finally:
        cfg.season["train_seasons"] = saved


def features(h: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per player-week with trailing features from earlier games only."""
    from streamer.teams import normalize_team

    h = h.sort_values(["player_id", "season", "week"]).copy()
    g = h.groupby("player_id", sort=False)

    def trail(col, n):
        return g[col].transform(lambda s: s.shift(1).rolling(n, min_periods=MIN_GAMES).mean())

    h["xfp4"] = trail("total_fantasy_points_exp", 4)
    h["xfp8"] = trail("total_fantasy_points_exp", 8)
    h["fp4"] = trail("fantasy_points_ppr", 4)
    h["tgt4"] = trail("targets", 4)
    h["car4"] = trail("carries", 4)
    h["tsh4"] = trail("target_share", 4)
    gm = games[games["game_type"] == "REG"].assign(team=lambda x: x["team"].map(normalize_team))
    gm = gm[["season", "week", "team", "spread_line", "total_line", "is_home"]].copy()
    # nflverse spread_line is the home team's margin: positive = home favoured.
    gm["spread"] = np.where(gm["is_home"] == 1, gm["spread_line"], -gm["spread_line"])
    gm["implied"] = gm["total_line"] / 2.0 + gm["spread"] / 2.0
    h["team"] = h["team"].map(normalize_team)
    out = h.merge(gm[["season", "week", "team", "implied", "spread"]], on=["season", "week", "team"], how="left")
    return out.dropna(subset=FEATURES)


def comps(f: pd.DataFrame, rows: pd.DataFrame, ks=K_GRID) -> pd.DataFrame:
    """Mean score of each row's k nearest earlier player-weeks at his position."""
    out = []
    for (season, week, pos), t in rows.groupby(["season", "week", "position"]):
        pool = f[(f["position"] == pos) & ((f["season"] < season) | ((f["season"] == season) & (f["week"] < week)))]
        if len(pool) < max(ks):
            continue
        mu, sd = pool[FEATURES].mean(), pool[FEATURES].std().replace(0, 1.0)
        a = ((pool[FEATURES] - mu) / sd).to_numpy(np.float32)
        b = ((t[FEATURES] - mu) / sd).to_numpy(np.float32)
        d = (b * b).sum(1)[:, None] - 2 * b @ a.T + (a * a).sum(1)[None, :]
        order = np.argpartition(d, max(ks) - 1, axis=1)[:, :max(ks)]
        dd = np.take_along_axis(d, order, 1)
        order = np.take_along_axis(order, np.argsort(dd, 1), 1)
        pts = pool["fantasy_points_ppr"].to_numpy(float)
        r = t[["season", "week", "player_id"]].copy()
        for k in ks:
            r[f"c{k}"] = pts[order[:, :k]].mean(1)
        out.append(r)
    return pd.concat(out, ignore_index=True)


def score(d: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for s, g in d.groupby("season"):
        wk = g.groupby(["week", "position"]).apply(lambda x: x[col].rank().corr(x["actual"].rank()))
        rows.append({"season": s, "mae": float((g[col] - g["actual"]).abs().mean()),
                     "bias": float((g[col] - g["actual"]).mean()), "rank": float(wk.mean())})
    return pd.DataFrame(rows).set_index("season")


def main() -> int:
    logging.disable(logging.WARNING)
    from streamer.config import get_config
    from streamer.data.nflverse import games_frame
    from streamer.roster import ros_backtest as rb
    from streamer.roster.projections import load_history

    cfg = get_config().for_profile("espn")
    deep = history(cfg)
    print(f"pool history {deep.season.min()}-{deep.season.max()}: {len(deep)} player-weeks")
    f = features(deep, games_frame(cfg))
    print(f"with features: {len(f)}")

    # Our production model, replayed on the games before each week.
    prod = load_history(cfg)
    ours = []
    for s in SEASONS:
        rost, inj = rb.rosters(cfg, s), rb.injuries(cfg, s)
        for w in MODEL_WEEKS:
            ps = rb.project(prod, cfg, s, w, rb.statuses(rost, inj, w))
            ours.append(pd.DataFrame([{"season": s, "week": w, "player_id": p.player_id, "position": p.position,
                                       "ours": p.projection} for p in ps if p.projection is not None]))
    ours = pd.concat(ours, ignore_index=True)
    actual = prod.groupby(["season", "week", "player_id"])["fantasy_points_ppr"].sum().rename("actual").reset_index()
    d = ours.merge(actual, on=["season", "week", "player_id"]).query("ours >= 3")
    feats = f.merge(d[["season", "week", "player_id"]], on=["season", "week", "player_id"])
    c = comps(f, feats)
    d = d.merge(c, on=["season", "week", "player_id"], how="inner")
    print(f"graded rows (played, ours >= 3, with comparisons): {len(d)}  "
          f"(of {len(ours.merge(actual, on=['season', 'week', 'player_id']).query('ours >= 3'))})")

    base = score(d, "ours")
    print(f"  ours        MAE {base.mae.mean():.4f} rank {base['rank'].mean():.4f} bias {base.bias.mean():+.3f}  "
          + "  ".join(f"{s}: {r.mae:.3f}/{r['rank']:.3f}" for s, r in base.iterrows()))
    for k in (50, 100, 200, 400):
        sc = score(d, f"c{k}")
        print(f"  comps k={k:<4}MAE {sc.mae.mean():.4f} rank {sc['rank'].mean():.4f} bias {sc.bias.mean():+.3f}")

    blended, picks = [], []
    for s in SEASONS:
        tr, te = d[d.season != s], d[d.season == s]
        k, w = min(itertools.product(K_GRID, W_GRID),
                   key=lambda kw: float((((1 - kw[1]) * tr.ours + kw[1] * tr[f"c{kw[0]}"]) - tr.actual).abs().mean()))
        picks.append(f"{s}: k {k} w {w:.2f}")
        blended.append(te.assign(blend=(1 - w) * te.ours + w * te[f"c{k}"]))
    b = pd.concat(blended)
    bl = score(b, "blend")
    better = int((bl.mae < base.mae).sum())
    keep = bl.mae.mean() < base.mae.mean() and better >= len(bl) - 1 and bl["rank"].mean() >= base["rank"].mean()
    print(f"\nblend: MAE {base.mae.mean():.4f} -> {bl.mae.mean():.4f} ({better}/{len(bl)} better)  "
          f"rank {base['rank'].mean():.4f} -> {bl['rank'].mean():.4f}  [{'; '.join(picks)}]")
    for s in bl.index:
        print(f"    {s}: MAE {base.mae[s]:.3f} -> {bl.mae[s]:.3f}   rank {base['rank'][s]:.3f} -> {bl['rank'][s]:.3f}")
    for pos, g in b.groupby("position"):
        print(f"    {pos}: n {len(g)}  ours {(g.ours - g.actual).abs().mean():.3f}  "
              f"blend {(g.blend - g.actual).abs().mean():.3f}")
    print(f"VERDICT: {'KEEP' if keep else 'DROP'}")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        b.to_parquet(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
