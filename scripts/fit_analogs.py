"""Similar past matchups for D/ST and kickers: do they improve the model?

    python scripts/fit_analogs.py [--since 2016]

Subvertadown shows each defence the most similar historical matchups and
what they scored. This tests that as a forecast. For every week of
2023-2025, each unit is matched to its nearest earlier games (strictly
before the week being predicted) on standardised pre-game features:

* ``vegas`` -- spread, total and the unit's implied total;
* ``full``  -- those plus its own and the opponent's team ratings.

The analog forecast is the mean score of the ``k`` nearest. It is judged
alone and blended with the shipped model's walk-forward projection,
``(1 - w) * model + w * analog``, with ``k`` and ``w`` chosen on the other
two test seasons and scored on the third (leave one season out). Ships only
on the usual rule: better overall and worse in at most one season.
DECISIONS.md, "Similar past matchups".
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

SEASONS = [2023, 2024, 2025]
K_GRID = (25, 50, 100, 200, 400)
W_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)
MATCH = {
    "DST": {
        "vegas": ["opp_implied_total", "team_spread", "total_line"],
        "full": ["opp_implied_total", "team_spread", "total_line", "opp_sack_rate_allowed", "opp_int_rate",
                 "opp_fumble_rate", "opp_pass_rate", "def_sack_rate", "def_int_rate",
                 "def_points_allowed_per_drive"],
    },
    "K": {
        "vegas": ["team_implied_total", "team_spread", "total_line"],
        "full": ["team_implied_total", "team_spread", "total_line", "is_dome", "off_fga_per_drive",
                 "off_rz_td_rate", "def_rz_td_rate_allowed", "k_long_att_share"],
    },
}


def analogs(frame: pd.DataFrame, cols: list[str], ks: tuple[int, ...]) -> pd.DataFrame:
    """For every test-season week, each row's mean score over its k nearest earlier games."""
    played = frame[frame["fantasy_points"].notna()].dropna(subset=cols).reset_index(drop=True)
    out = []
    for season in SEASONS:
        for week in sorted(played.loc[played["season"] == season, "week"].unique()):
            pool = played[(played["season"] < season) | ((played["season"] == season) & (played["week"] < week))]
            test = played[(played["season"] == season) & (played["week"] == week)]
            if len(pool) < max(ks) or test.empty:
                continue
            mu, sd = pool[cols].mean(), pool[cols].std().replace(0, 1.0)
            a = ((pool[cols] - mu) / sd).to_numpy(float)
            b = ((test[cols] - mu) / sd).to_numpy(float)
            d = ((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)
            order = np.argsort(d, axis=1)
            pts = pool["fantasy_points"].to_numpy(float)
            row = test[["season", "week", "game_id", "team"]].copy()
            for k in ks:
                row[f"a{k}"] = pts[order[:, :k]].mean(1)
            out.append(row)
    return pd.concat(out, ignore_index=True)


def score(d: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for s, g in d.groupby("season"):
        wk = g.groupby("week").apply(lambda x: x[col].rank().corr(x["fantasy_points"].rank()))
        rows.append({"season": s, "mae": float((g[col] - g["fantasy_points"]).abs().mean()),
                     "rank": float(wk.mean())})
    return pd.DataFrame(rows).set_index("season")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=None, help="first season in the analog pool")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    from streamer.backtest import walk_forward
    from streamer.config import get_config
    from streamer.data.nflverse import games_frame, load_pbp
    from streamer.features.build import build_dst_features, build_kicker_features
    from streamer.pipeline import training_seasons

    cfg = get_config().for_profile("espn")
    seasons = training_seasons(cfg)
    pool_seasons = sorted(set(seasons) | set(range(args.since, min(seasons)))) if args.since else seasons
    pbp, games = load_pbp(pool_seasons, cfg), games_frame(cfg)
    print(f"model seasons {seasons[0]}-{seasons[-1]}; analog pool {pool_seasons[0]}-{pool_seasons[-1]}")
    builders = {"DST": build_dst_features, "K": build_kicker_features}
    for pos, build in builders.items():
        full = build(pbp, games, cfg)
        model_frame = full[full["season"] >= min(seasons)]
        preds = walk_forward(model_frame, pos, SEASONS, cfg)
        preds = preds[["season", "week", "game_id", "team", "expected_points", "fantasy_points"]]
        print(f"\n{pos}: {len(preds)} graded unit-games, pool {len(full)}")
        base = score(preds, "expected_points")
        print(f"  model        MAE {base.mae.mean():.4f} rank {base['rank'].mean():.4f}  "
              + "  ".join(f"{s}: {r.mae:.4f}/{r['rank']:.3f}" for s, r in base.iterrows()))
        for name, cols in MATCH[pos].items():
            a = analogs(full, cols, K_GRID)
            d = preds.merge(a, on=["season", "week", "game_id", "team"], how="inner")
            for k in (50, 200):
                sc = score(d, f"a{k}")
                print(f"  {name:5} k={k:<4} alone MAE {sc.mae.mean():.4f} rank {sc['rank'].mean():.4f}")
            # Leave one season out: choose k and w on the other two seasons.
            blended, picks = [], []
            for s in SEASONS:
                tr, te = d[d["season"] != s], d[d["season"] == s]

                def mae(x, k, w):
                    return float(((1 - w) * x["expected_points"] + w * x[f"a{k}"] - x["fantasy_points"]).abs().mean())

                k, w = min(itertools.product(K_GRID, W_GRID), key=lambda kw: mae(tr, *kw))
                picks.append(f"{s}: k {k} w {w:.2f}")
                blended.append(te.assign(blend=(1 - w) * te["expected_points"] + w * te[f"a{k}"]))
            bl = score(pd.concat(blended), "blend")
            base_d = score(d, "expected_points")
            mae_better = int((bl.mae < base_d.mae).sum())
            worse_rank = int((bl["rank"] < base_d["rank"]).sum())
            keep = bl.mae.mean() < base_d.mae.mean() and mae_better >= len(bl) - 1 and \
                bl["rank"].mean() >= base_d["rank"].mean()
            print(f"  {name:5} blend MAE {base_d.mae.mean():.4f} -> {bl.mae.mean():.4f} ({mae_better}/{len(bl)} better)  "
                  f"rank {base_d['rank'].mean():.4f} -> {bl['rank'].mean():.4f} ({worse_rank} worse)  "
                  f"[{'; '.join(picks)}]  -> {'KEEP' if keep else 'DROP'}")
            for s in bl.index:
                print(f"        {s}: {base_d.mae[s]:.4f} -> {bl.mae[s]:.4f}   rank {base_d['rank'][s]:.3f} -> {bl['rank'][s]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
