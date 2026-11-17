"""Fit and grade the tight-end season line (roster/level_calibration.py, ``season.TE``).

    python scripts/fit_te_season_line.py [--weekly cache.pkl] [--profile espn|yahoo] [--sims 600]

Every 2022-2025 week 2-14, the top 40 untagged tight ends by season value
(walk-forward, ``grade_league_odds.weekly_projections``) are played forward
by the season simulator's own futures (absences, lost roles and all) with
their season value read through ``a + b x`` -- a pull toward ``P`` by a
factor ``k`` (``a = P (1 - k)``, ``b = k``) -- and the simulated points a
game over the rest of the season are scored against what each one really
scored a game (0 when he did not play; byes left out). Prints the RMSE for
each (P, k) by season, the pair picked on the other three seasons for each
season held out, and the pair on all four. DECISIONS.md, "Tight-end season
values pulled together".
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

SEASONS = (2022, 2023, 2024, 2025)
GRID = [(p, k) for p in (8.0, 9.0, 10.0, 11.0) for k in (0.75, 0.8, 0.85, 0.9)] + [(0.0, 1.0)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", help="cache of the walk-forward weekly projections (built if missing)")
    ap.add_argument("--profile", default="espn", choices=("espn", "yahoo"))
    ap.add_argument("--sims", type=int, default=600)
    args = ap.parse_args()
    os.environ.setdefault("STREAMER_CREDIT_FREE", "1")
    logging.disable(logging.WARNING)
    import grade_league_odds as base

    from streamer.config import get_config
    from streamer.roster import futures as fut
    from streamer.roster import league_replay as lr
    from streamer.roster import level_calibration as lc
    from streamer.roster import ros_backtest as rb
    from streamer.roster.lineup import use_profile
    from streamer.roster.projections import load_history

    cfg = get_config().for_profile(args.profile)
    use_profile(cfg)
    lc._SEASON_ON, lc.SEASON_OFF = True, False
    weekly = base.weekly_projections(cfg, args.weekly)
    history = load_history(cfg)
    schedules = pd.read_parquet(cfg.raw_dir / "schedules.parquet")
    pre = {}
    for s in SEASONS:
        pts = lr.points_by_week(history, s)
        byes = rb.byes(schedules, s, list(range(1, 18)))
        for w in range(2, 15):
            ps = sorted([p for p in weekly[(s, w)] if p.position == "TE" and not (p.status or "") and p.ros_value],
                        key=lambda p: -float(p.ros_value))[:40]
            weeks = list(range(w, 18))
            on = [[k for k, x in enumerate(weeks) if x not in byes.get(p.team, set())] for p in ps]
            real = [np.mean([pts.get((weeks[k], p.player_id), 0.0) for k in o]) if len(o) >= 3 else np.nan
                    for p, o in zip(ps, on)]
            pre[(s, w)] = (ps, weeks, byes, on, real)
    rows = []
    for pivot, k in GRID:
        conf = {"knot": 12.0, "positions": {}, "season": {"TE": {"a": pivot * (1 - k), "b": k, "c": 0.0}}}
        lc.load = lambda conf=conf: conf
        for (s, w), (ps, weeks, byes, on, real) in pre.items():
            mean = fut.simulate(ps, weeks, byes, n_sims=args.sims, seed=s * 100 + w).scores.mean(axis=0)
            rows += [(pivot, k, s, float(mean[i, on[i]].mean()), real[i]) for i in range(len(ps)) if not np.isnan(real[i])]
    g = pd.DataFrame(rows, columns=["P", "k", "season", "sim", "real"])
    table = g.groupby(["P", "k", "season"]).apply(lambda x: float(np.sqrt(np.mean((x.sim - x.real) ** 2)))).unstack()
    print(table.round(4).to_string())
    as_is = table.loc[(0.0, 1.0)]
    held = {s: table.drop(columns=s).mean(axis=1).idxmin() for s in SEASONS}
    got = [table.loc[held[s], s] for s in SEASONS]
    print("held out: " + "; ".join(f"{s} P={held[s][0]:g} k={held[s][1]:g} {x:.4f} (as is {as_is[s]:.4f})"
                                   for s, x in zip(SEASONS, got)))
    print(f"mean {np.mean(got):.4f} against {as_is.mean():.4f}, better in {sum(x < as_is[s] for s, x in zip(SEASONS, got))} of 4")
    best = table.mean(axis=1).idxmin()
    print(f"all four seasons: P={best[0]:g} k={best[1]:g} -> a={best[0] * (1 - best[1]):.3f} b={best[1]:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
