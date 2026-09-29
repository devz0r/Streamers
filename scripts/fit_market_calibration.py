"""Refit where our rest-of-season projection should defer to the market.

    python scripts/fit_market_calibration.py [--dry-run]

For every player-week of 2022-2025 (weeks 3-13) with a name to have market
value (4+ games last season): our projection as of that week, the
name-value view (:func:`streamer.roster.market_calibration.name_value`),
and what he actually scored per game the rest of the season. Cells by
position, direction (we are above or below the market) and part of the
season get a shrink factor ``k`` in ``actual - market = k * (ours - market)``.

A cell is kept only if, fitted on three seasons and tested on the fourth
(each in turn), it beats our projection by 3%+ in average miss on the
disagreements that matter (1.5+ points apart), with 60+ such cases, and it
shrinks toward the market (k < 1): amplifying our own disagreements is not
something a few cells should be trusted to do.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

KEYS = ["position", "sign", "bucket"]
BUCKETS = {"3-6": (3, 6), "7-10": (7, 10), "11-13": (11, 13)}


def build(cfg) -> pd.DataFrame:
    from streamer.roster.market_calibration import name_value
    from streamer.roster.projections import load_history, player_table

    h = load_history(cfg)
    g = h.groupby(["player_id", "season"])["fantasy_points_ppr"].agg(["mean", "count"])
    lines = {(str(pid), int(s)): (float(r["mean"]), int(r["count"])) for (pid, s), r in g.iterrows()}
    rows = []
    for season in (2022, 2023, 2024, 2025):
        cur = h[h.season == season]
        for week in range(3, 14):
            tab = player_table(h, season, week, cfg)
            now = cur[cur.week < week].groupby("player_id")["fantasy_points_ppr"].agg(["mean", "count"])
            rest = cur[(cur.week >= week) & (cur.week <= 17)].groupby("player_id")["fantasy_points_ppr"].agg(
                ["mean", "count"])
            for r in tab.itertuples():
                if r.position not in ("QB", "RB", "WR", "TE") or r.player_id not in rest.index:
                    continue
                last = lines.get((str(r.player_id), season - 1))
                if last is None or last[1] < 4:
                    continue
                n = now.loc[r.player_id] if r.player_id in now.index else None
                mkt = name_value(float(r.blend), last, lines.get((str(r.player_id), season - 2)),
                                 (float(n["mean"]), int(n["count"])) if n is not None else None)
                rr = rest.loc[r.player_id]
                if rr["count"] < 4 or max(r.blend, mkt) < 6:
                    continue
                rows.append((season, week, r.position, float(r.blend), mkt, float(rr["mean"]), int(rr["count"])))
    d = pd.DataFrame(rows, columns=["season", "week", "position", "ours", "market", "actual", "games"])
    d["gap"] = d.ours - d.market
    d["sign"] = np.where(d.gap >= 0, "up", "down")
    d["bucket"] = pd.cut(d.week, [2, 6, 10, 13], labels=list(BUCKETS))
    return d


def fit_k(s: pd.DataFrame) -> float:
    w, x, y = s.games, s.gap, s.actual - s.market
    return float((w * x * y).sum() / (w * x * x).sum()) if (x * x).sum() > 0 else 1.0


def mae(pred: pd.Series, s: pd.DataFrame) -> float:
    return float((np.abs(pred - s.actual) * s.games).sum() / s.games.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    from streamer.config import get_config
    from streamer.roster import market_calibration as mc

    d = build(get_config())
    tested = []
    for season in sorted(d.season.unique()):
        train, test = d[d.season != season], d[d.season == season].copy()
        test = test.join(train.groupby(KEYS, observed=True).apply(fit_k).rename("k"), on=KEYS)
        tested.append(test)
    t = pd.concat(tested)
    t["cal"] = t.market + t.k.fillna(1.0) * t.gap
    full = d.groupby(KEYS, observed=True).apply(fit_k)
    print(f"{len(d)} player-weeks; average miss: ours {mae(t.ours, t):.3f}, market {mae(t.market, t):.3f}")
    cells = []
    for key, s in t[np.abs(t.gap) >= 1.5].groupby(KEYS, observed=True):
        pos, sign, bucket = key
        ours, cal = mae(s.ours, s), mae(s.cal, s)
        k = float(full.loc[key])
        keep = len(s) >= 60 and cal <= 0.97 * ours and k < 1.0
        print(f"  {pos} {sign:4} {bucket:5} n={len(s):4} ours {ours:.2f} calibrated {cal:.2f} "
              f"market {mae(s.market, s):.2f} k {k:.2f}{'  KEEP' if keep else ''}")
        if keep:
            lo, hi = BUCKETS[bucket]
            what = "a receiver's early target surge" if (pos, sign) == ("WR", "up") else \
                f"{'a rise' if sign == 'up' else 'a drop'} in a {pos}'s projection this early"
            cells.append({"position": pos, "sign": sign, "weeks": [lo, hi], "k": round(k, 2),
                          "note": f"{what}: players like this kept about {max(k, 0):.0%} of it "
                                  f"(2022-2025, weeks {lo}-{hi})",
                          "evidence": {"cases": int(len(s)), "miss_ours": round(ours, 2),
                                       "miss_calibrated": round(cal, 2)}})
    out = {"cells": cells, "fitted": {"seasons": [2022, 2025], "player_weeks": int(len(d))}}
    if args.dry_run:
        print(json.dumps(out, indent=2))
        return 0
    mc.MODEL_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {mc.MODEL_PATH} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
