"""Fit how much work a teammate back from an absence takes back.

    python scripts/fit_teammate_back.py

Walk-forward over 2022-2025, weeks 3-18: every player projected by the
production volume model (``projections.player_table``) on the games before
the week, and the same projection with :func:`projections._returning`
reading the games he had while a regular at his position -- one who plays
that week -- was out, less what he was expected to absorb: ``bigger`` of
the gap when the man coming back had the bigger role, ``smaller`` of the
returning man's volume when he had the smaller one. Scored on the points
each player scored that week (players who recorded a stat that week, as
every projection grade here).

Each position's pair of shares is picked on three seasons and scored on the
fourth. A position is kept only if the held-out error is lower overall and
worse in at most one season -- the same rule as the next man up. Then the
pairs better than none in every season are listed, lowest error first; the
shares under ``roster.teammate_back`` in config.yaml are chosen from those,
with the least bias among near-ties (TE was lowered from 0.5 to 0.35 after
the season replay showed it overshooting; DECISIONS.md, "A teammate back
from an absence").
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

BIGGER = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
SMALLER = (0.0, 0.1, 0.2, 0.3, 0.45)
POSITIONS = ("RB", "WR", "TE")


def build(cfg) -> pd.DataFrame:
    from streamer.roster import projections as pj

    h = pj.load_history(cfg)
    conf = dict(cfg.raw["roster"])
    conf.pop("teammate_back", None)
    cfg.raw["roster"] = conf
    rows = []
    for season in (2022, 2023, 2024, 2025):
        for week in range(3, 19):
            cur = h[(h["season"] == season) & (h["week"] == week)]
            if cur.empty:
                continue
            prior = h[(h["season"] < season) | ((h["season"] == season) & (h["week"] < week))]
            pos_mean = pj.position_rates(prior)[0]
            base = pj.player_table(h, season, week, cfg)
            back = {pid: (1.0, 1.0) for pid in cur["player_id"]}
            got = cur.set_index("player_id")["fantasy_points_ppr"]
            keep = base[base["player_id"].isin(got.index) & (base["vol"] >= 3.0)
                        & base["position"].isin(POSITIONS)].set_index("player_id")
            out = keep[["position", "vol", "eff"]].assign(season=season, week=week,
                                                         pts=got.reindex(keep.index).to_numpy())
            for b, s in itertools.product(BIGGER, SMALLER):
                rule = {"positions": list(POSITIONS), "regular": 5.0,
                        "bigger": {q: b for q in POSITIONS}, "smaller": {q: s for q in POSITIONS}}
                t = pj._returning(prior, base, back, {**conf, "teammate_back": rule}, pos_mean, season)
                out[f"v{b}_{s}"] = t.set_index("player_id")["vol"].reindex(keep.index).to_numpy()
            rows.append(out.reset_index())
        print(season, flush=True)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    logging.disable(logging.WARNING)
    from streamer.config import get_config

    d = build(get_config().for_profile("espn"))
    cols = [f"v{b}_{s}" for b, s in itertools.product(BIGGER, SMALLER)]
    none = f"v{BIGGER[0]}_{SMALLER[0]}"
    edge = f"v{BIGGER[-1]}_{SMALLER[-1]}"

    def err(g: pd.DataFrame, c: str) -> np.ndarray:
        return (g["pts"] - g[c] * g["eff"]).to_numpy()

    def rmse(e: np.ndarray) -> float:
        return float(np.sqrt((e ** 2).mean()))

    for pos in POSITIONS:
        allp = d[d["position"] == pos]
        g = allp[(allp[edge] - allp[none]).abs() > 1e-9]          # those it can move
        print(f"== {pos}: {len(g)} of {len(allp)} player-weeks had a teammate back from an absence")
        rows, wins = [], 0
        for season in sorted(g["season"].unique()):
            tr, te = g[g["season"] != season], g[g["season"] == season]
            pick = min(cols, key=lambda c: rmse(err(tr, c)))
            e0, e1 = err(te, none), err(te, pick)
            rows.append((season, len(te), rmse(e0), rmse(e1), e0.mean(), e1.mean(), pick))
            wins += rmse(e1) < rmse(e0)
        n = sum(r[1] for r in rows)
        r0 = np.sqrt(sum(r[1] * r[2] ** 2 for r in rows) / n)
        r1 = np.sqrt(sum(r[1] * r[3] ** 2 for r in rows) / n)
        for season, k, a, b, ba, bb, pick in rows:
            print(f"   {season}: n={k:4d} picked {pick:9s} RMSE {a:.3f} -> {b:.3f}  bias {ba:+.2f} -> {bb:+.2f}")
        best = min(cols, key=lambda c: rmse(err(g, c)))
        print(f"   held-out RMSE {r0:.3f} -> {r1:.3f}, better in {wins} of {len(rows)} seasons; "
              f"all seasons best {best}: bias {err(g, none).mean():+.2f} -> {err(g, best).mean():+.2f}")
        keep = r1 < r0 and wins >= len(rows) - 1
        print(f"   {'keep' if keep else 'drop'} the correction at {pos}")
        steady = []
        for c in cols:
            by_season = [rmse(err(te, c)) < rmse(err(te, none)) for _y, te in g.groupby("season")]
            if c != none and all(by_season):
                steady.append((rmse(err(g, c)), float(err(g, c).mean()), c))
        for r, bias, c in sorted(steady)[:5]:
            b, sm = c[1:].split("_")
            print(f"   better in every season: bigger {b}, smaller {sm}: RMSE {r:.3f}, bias {bias:+.2f}")
        e_all0, e_all1 = err(allp, none), err(allp, best)
        print(f"   every {pos} player-week: MAE {np.abs(e_all0).mean():.4f} -> {np.abs(e_all1).mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
