"""Should this season's graded weeks correct the skill projection in-season?

    python scripts/fit_in_season.py [--cache rows.parquet]

Replays 2022-2025 week by week through the production volume model
(``projections.player_table``) and its range (``sd_table`` spread, shaped by
``outcome.shape``: the 10th and 90th percentiles). Two corrections, each
using only that season's earlier weeks at the player's position:

* **bias** -- add the season's mean miss so far, shrunk: ``sum(miss) / (n + k)``;
* **spread** -- scale the spread by the season's root-mean-square of
  miss / spread so far, shrunk toward 1 with ``k`` games of prior.

``k`` is picked on three seasons and scored on the fourth (RMSE for the
bias; quantile loss at the 10th and 90th percentiles for the spread, with
coverage of the 80% range); kept only if better overall and worse in at
most one season. DECISIONS.md, "Correcting the projection in-season".
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
POSITIONS = ("QB", "RB", "WR", "TE")
KS = (25, 50, 100, 200, 400, 800, 1600, 3200, np.inf)


def build(cfg) -> pd.DataFrame:
    from streamer.roster import outcome
    from streamer.roster import projections as pj

    h = pj.load_history(cfg)
    rows = []
    for season in (2022, 2023, 2024, 2025):
        sds = pj.sd_table(h[h["season"] < season], cfg)
        for week in range(2, 19):
            cur = h[(h["season"] == season) & (h["week"] == week)]
            if cur.empty:
                continue
            t = pj.player_table(h, season, week, cfg)
            got = cur.groupby("player_id")["fantasy_points_ppr"].sum()
            t = t[t["player_id"].isin(got.index) & t["position"].isin(POSITIONS) & t["blend"].notna()]
            for r in t.itertuples():
                m = float(r.blend)
                sd = pj.lookup_sd(sds, r.position, m)
                p, z = outcome.shape(r.position, m)
                rows.append({"season": season, "week": week, "position": r.position, "pred": m, "sd": sd,
                             "z10": float(np.interp(0.1, p, z)), "z90": float(np.interp(0.9, p, z)),
                             "pts": float(got[r.player_id])})
        print(season, len(rows), flush=True)
    return pd.DataFrame(rows)


def corrected(d: pd.DataFrame, k: float, kind: str) -> pd.DataFrame:
    """Each row's bias shift or spread scale from its season's earlier weeks."""
    d = d.sort_values(["season", "position", "week"]).copy()
    out = []
    for (_s, _p), g in d.groupby(["season", "position"]):
        wk = g.groupby("week").agg(miss=("pts", lambda x: 0), n=("pts", "size"))
        miss = (g["pts"] - g["pred"]).groupby(g["week"]).sum()
        sq = (((g["pts"] - g["pred"]) / g["sd"]) ** 2).groupby(g["week"]).sum()
        wk["miss"], wk["sq"] = miss, sq
        cum = wk[["miss", "sq", "n"]].cumsum().shift(1).fillna(0.0)
        if kind == "bias":
            adj = cum["miss"] / (cum["n"] + k) if np.isfinite(k) else cum["miss"] * 0.0
        else:
            adj = np.sqrt((cum["sq"] + k) / (cum["n"] + k)) if np.isfinite(k) else cum["sq"] * 0.0 + 1.0
        out.append(g.assign(adj=g["week"].map(adj)))
    return pd.concat(out)


def pinball(d: pd.DataFrame, scale: pd.Series) -> float:
    lo = d["pred"] + d["z10"] * d["sd"] * scale
    hi = d["pred"] + d["z90"] * d["sd"] * scale
    y = d["pts"]
    l10 = np.maximum(0.1 * (y - lo), 0.9 * (lo - y))
    l90 = np.maximum(0.9 * (y - hi), 0.1 * (hi - y))
    return float((l10 + l90).mean())


def coverage(d: pd.DataFrame, scale: pd.Series) -> float:
    lo = d["pred"] + d["z10"] * d["sd"] * scale
    hi = d["pred"] + d["z90"] * d["sd"] * scale
    return float(((d["pts"] >= lo) & (d["pts"] <= hi)).mean())


def score(d: pd.DataFrame, kind: str) -> float:
    if kind == "bias":
        return float(np.sqrt(((d["pts"] - d["pred"] - d["adj"]) ** 2).mean()))
    return pinball(d, d["adj"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    from streamer.config import get_config

    cfg = get_config().for_profile("espn")
    if args.cache and Path(args.cache).exists():
        d = pd.read_parquet(args.cache)
    else:
        d = build(cfg)
        if args.cache:
            d.to_parquet(args.cache)
    for kind in ("bias", "spread"):
        print(f"\n=== {kind} ({'RMSE' if kind == 'bias' else 'quantile loss, 10th+90th'}) ===")
        by_k = {k: corrected(d, k, kind) for k in KS}
        for pos in POSITIONS:
            wins, rows = 0, []
            tot0 = tot1 = 0.0
            n_all = 0
            for s in sorted(d["season"].unique()):
                tr_k = min(KS, key=lambda k: score(by_k[k][(by_k[k]["season"] != s) & (by_k[k]["position"] == pos)],
                                                   kind))
                te0 = by_k[np.inf][(by_k[np.inf]["season"] == s) & (by_k[np.inf]["position"] == pos)]
                te1 = by_k[tr_k][(by_k[tr_k]["season"] == s) & (by_k[tr_k]["position"] == pos)]
                a, b = score(te0, kind), score(te1, kind)
                wins += b < a
                w = len(te0)
                tot0 += (a ** 2 if kind == "bias" else a) * w
                tot1 += (b ** 2 if kind == "bias" else b) * w
                n_all += w
                extra = ""
                if kind == "spread":
                    extra = f" coverage {coverage(te0, te0['adj']):.3f} -> {coverage(te1, te1['adj']):.3f}"
                else:
                    extra = f" bias {(te0['pts'] - te0['pred']).mean():+.2f} -> {(te1['pts'] - te1['pred'] - te1['adj']).mean():+.2f}"
                rows.append(f"      {s}: k={tr_k} {a:.4f} -> {b:.4f}{extra}")
            o0 = np.sqrt(tot0 / n_all) if kind == "bias" else tot0 / n_all
            o1 = np.sqrt(tot1 / n_all) if kind == "bias" else tot1 / n_all
            keep = o1 < o0 and wins >= 3
            print(f"  {pos}: overall {o0:.4f} -> {o1:.4f}, better in {wins} of 4: {'KEEP' if keep else 'drop'}")
            for r in rows:
                print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
