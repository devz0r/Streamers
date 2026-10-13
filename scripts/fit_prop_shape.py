"""The shape of each prop stat around its line, from nflverse history.

    python scripts/fit_prop_shape.py [--write]

A prop line sits near the median; fantasy points need the mean. Yardage is
right-skewed (a long touchdown is far above the line, a quiet game can only
fall to zero), so the mean sits above the median, and a bell curve -- whose
mean is its median -- understates it at every line. Here, for each stat a
book prices as a line (passing, rushing and receiving yards, receptions):
every regular-season player-game 2021-2025 by a player with 3+ earlier
games, bucketed by his trailing-4 average (the level a book prices), above a
floor where books post props. Within each bucket: the median, and the
quantiles of (result - median) on a grid -- so a line ``L`` with ``P(over) =
p`` gives median ``L - q(1 - p)`` and mean ``median + E[result - median]``.

Fitted on 2021-2023 and scored on 2024-2025 against the normal conversion
(each bucket's real mean, line at its median, P(over) as observed); then
refitted on every season and written to ``streamer/data/prop_shape.json``
with ``--write``. DECISIONS.md, "Turning a prop line into a mean".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
STATS = {"passing_yards": 150.0, "rushing_yards": 15.0, "receiving_yards": 15.0, "receptions": 1.5}
GRID = [round(x, 3) for x in np.linspace(0.02, 0.98, 49)]
BUCKETS = 10
OUT = Path(__file__).resolve().parents[1] / "src" / "streamer" / "data" / "prop_shape.json"


def games(seasons) -> pd.DataFrame:
    from streamer.config import get_config

    cfg = get_config()
    frames = [pd.read_parquet(cfg.raw_dir / f"player_stats_{s}.parquet") for s in seasons]
    g = pd.concat(frames)
    g = g[g["season_type"] == "REG"].sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    for stat in STATS:
        g[f"lvl_{stat}"] = g.groupby("player_id")[stat].transform(
            lambda x: x.shift(1).rolling(4, min_periods=3).mean())
    return g


def tables(g: pd.DataFrame) -> dict:
    out = {}
    for stat, floor in STATS.items():
        d = g[g[f"lvl_{stat}"] >= floor].dropna(subset=[f"lvl_{stat}", stat])
        d = d.assign(_b=pd.qcut(d[f"lvl_{stat}"], BUCKETS, labels=False, duplicates="drop"))
        rows = []
        for _b, x in d.groupby("_b"):
            y = x[stat].to_numpy(float)
            med = float(np.median(y))
            rows.append({"median": med, "mean_minus_median": float(y.mean() - med), "n": int(len(y)),
                         "q": [float(v) for v in np.quantile(y - med, GRID)]})
        out[stat] = sorted(rows, key=lambda r: r["median"])
    return {"grid": GRID, "stats": out}


def score(g: pd.DataFrame, shape: dict) -> None:
    from scipy import stats

    from streamer.config import get_config
    from streamer.data import props

    cfg = get_config()
    print("held-out 2024-2025, mean per bucket: actual | normal conversion | shape conversion")
    for stat, floor in STATS.items():
        d = g[g[f"lvl_{stat}"] >= floor].dropna(subset=[f"lvl_{stat}", stat])
        d = d.assign(_b=pd.qcut(d[f"lvl_{stat}"], BUCKETS, labels=False, duplicates="drop"))
        e_norm, e_shape, n = 0.0, 0.0, 0
        for _b, x in d.groupby("_b"):
            y = x[stat].to_numpy(float)
            line = float(np.floor(np.median(y))) + 0.5               # books post half lines near the median
            p = float((y > line).mean())
            sd = props.spread_for(stat, line, cfg)
            normal = line + sd * stats.norm.ppf(np.clip(p, 1e-3, 1 - 1e-3))
            shaped = props.mean_from_shape(stat, line, p, shape)
            e_norm += (normal - y.mean()) * len(y)
            e_shape += (shaped - y.mean()) * len(y)
            n += len(y)
        print(f"  {stat:16} bias normal {e_norm / n:+.2f}, shape {e_shape / n:+.2f}  (n={n})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    train = tables(games(range(2021, 2024)))
    score(games(range(2023, 2026)).query("season >= 2024"), train)
    for stat, rows in train["stats"].items():
        mid = rows[len(rows) // 2]
        print(f"  {stat:16} middle bucket: median {mid['median']:.1f}, mean - median {mid['mean_minus_median']:+.2f}")
    if args.write:
        full = tables(games(range(2021, 2026)))
        full["evidence"] = "scripts/fit_prop_shape.py: nflverse 2021-2025 player-games, by trailing-4 level"
        OUT.write_text(json.dumps(full, indent=1) + "\n", encoding="utf-8")
        print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
