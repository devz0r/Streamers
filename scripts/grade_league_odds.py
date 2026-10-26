"""Grade weekly P(win) and the season odds on replayed leagues.

    python scripts/grade_league_odds.py [--leagues 8] [--weekly cache.pkl] [--out dir]

Drafts and plays ``--leagues`` ten-team leagues on each of 2022-2025
(:mod:`streamer.roster.league_replay`): projections from the games before
each week only, real scores, the production lineup draws and season model.
Prints, per season and pooled:

* this week's P(win) against the results: Brier score and log loss beside a
  coin flip, a calibration table with 95% Wilson intervals, and the spread
  of the margin (the share of results below each predicted percentile);
* the season model's P(playoffs) and P(title) at weeks 1, 4, 7, 10 and 13
  against who made it and who won, beside a baseline that knows only the
  standings (every game left a coin flip); expected wins against final wins.

With ``--out``, the games and odds are written there for further cuts. DECISIONS.md,
"Grading the odds on replayed leagues".
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")


def weekly_projections(cfg, path: str | None) -> dict:
    """(season, week) -> projected players, from the games before that week."""
    if path and Path(path).exists():
        return pickle.loads(Path(path).read_bytes())
    from streamer.roster import ros_backtest as rb
    from streamer.roster.projections import load_history

    history = load_history(cfg)
    out = {}
    for season in rb.SEASONS:
        rost, inj = rb.rosters(cfg, season), rb.injuries(cfg, season)
        for week in range(1, 18):
            out[(season, week)] = rb.project(history, cfg, season, week, rb.statuses(rost, inj, week))
        print(f"projected {season}", flush=True)
    if path:
        Path(path).write_bytes(pickle.dumps(out))
    return out


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration(p, y, edges) -> list[str]:
    p, y = np.asarray(p), np.asarray(y)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        n = int(m.sum())
        if not n:
            continue
        lo_ci, hi_ci = wilson(float(y[m].sum()), n)
        out.append(f"    {lo:4.0%}-{hi:4.0%}: n={n:5d}  predicted {p[m].mean():6.1%}  happened {y[m].mean():6.1%}"
                   f"  (95% {lo_ci:5.1%}-{hi_ci:5.1%})")
    return out


def report(games: pd.DataFrame, odds: pd.DataFrame) -> None:
    # Each game once, from the side the schedule lists first; folded so the
    # favourite is the subject (P >= 0.5).
    g = games.copy()
    fav = g["p_win"] >= 0.5
    g["pf"] = np.where(fav, g["p_win"], 1 - g["p_win"])
    g["yf"] = np.where(fav, g["won"], 1 - g["won"])
    print(f"\n=== This week's P(win): {len(g)} games ===")
    for s, gs in list(g.groupby("season")) + [("all", g)]:
        print(f"  {s}: Brier {brier(gs.p_win, gs.won):.4f} (coin flip 0.2500)  log loss {logloss(gs.p_win, gs.won):.4f}"
              f" (0.6931)  favourites won {gs.yf.mean():.1%} vs predicted {gs.pf.mean():.1%}")
    print("  calibration (favourite's side):")
    for line in calibration(g.pf, g.yf, [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]):
        print(line)
    pit = g["pit"].to_numpy()
    print("  margin spread: share of results below the predicted 10th/25th/50th/75th/90th percentile: "
          + " / ".join(f"{(pit < q).mean():.1%}" for q in (0.1, 0.25, 0.5, 0.75, 0.9)))
    print(f"  mean projected margin {(g.proj_a - g.proj_b).abs().mean():.1f}; mean real |margin| "
          f"{(g.score_a - g.score_b).abs().mean():.1f}; mean predicted margin sd {g.margin_sd.mean():.1f}, "
          f"real sd of (margin - projected) {((g.score_a - g.score_b) - (g.proj_a - g.proj_b)).std():.1f}")
    sm = g.dropna(subset=["p_win_season"])
    if len(sm):
        print(f"  the season model's own P(win) for the week (checkpoints, n={len(sm)}): Brier "
              f"{brier(sm.p_win_season, sm.won):.4f} vs the lineup draws' {brier(sm.p_win, sm.won):.4f}")

    for col, base, y, edges, name in (
            ("p_playoffs", "base_playoffs", "made", [0, .1, .25, .5, .75, .9, 1.01], "P(playoffs)"),
            ("p_title", "base_title", "won_title", [0, .03, .07, .12, .2, .35, 1.01], "P(title)")):
        print(f"\n=== {name}: {len(odds)} team-checkpoints ===")
        for w, gw in odds.groupby("week"):
            print(f"  week {w:2d}: Brier {brier(gw[col], gw[y]):.4f} vs standings-only {brier(gw[base], gw[y]):.4f}"
                  f"   log loss {logloss(gw[col], gw[y]):.4f} vs {logloss(gw[base], gw[y]):.4f}")
        for s, gs in list(odds.groupby("season")) + [("all", odds)]:
            print(f"  {s}: Brier {brier(gs[col], gs[y]):.4f} vs standings-only {brier(gs[base], gs[y]):.4f}"
                  f"   log loss {logloss(gs[col], gs[y]):.4f} vs {logloss(gs[base], gs[y]):.4f}")
        print("  calibration:")
        for line in calibration(odds[col], odds[y], edges):
            print(line)
    print("\n=== Expected wins against final wins; team strength ===")
    print("  (slope: real points for the rest of the regular season on the model's expected points, both")
    print("   against the league average that week -- 1 when the model separates teams as much as results do)")
    for w, gw in odds.groupby("week"):
        e = gw.final_wins - gw.exp_wins
        key = ["season", "seed"]
        x = gw.exp_points - gw.groupby(key).exp_points.transform("mean")
        y = gw.rest_points - gw.groupby(key).rest_points.transform("mean")
        slope = float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else np.nan
        print(f"  week {w:2d}: wins bias {e.mean():+.2f}  RMSE {np.sqrt((e ** 2).mean()):.2f}   strength slope "
              f"{slope:.2f}  (sd of expected {x.std():.1f}, of real {y.std():.1f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", type=int, default=8, help="leagues per season")
    ap.add_argument("--weekly", help="cache of the weekly projections (built if missing)")
    ap.add_argument("--out", help="a directory for the graded games and odds (parquet)")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--no-upgrades", action="store_true", help="managers only fill holes (no upgrade claims)")
    args = ap.parse_args()
    os.environ.setdefault("STREAMER_CREDIT_FREE", "1")
    logging.disable(logging.WARNING)
    from streamer.config import get_config
    from streamer.roster import league_replay as lr
    from streamer.roster import ros_backtest as rb
    from streamer.roster.projections import load_history

    cfg = get_config().for_profile("espn")
    weekly = weekly_projections(cfg, args.weekly)
    history = load_history(cfg)
    schedules = pd.read_parquet(cfg.raw_dir / "schedules.parquet")
    games, odds = [], []
    for season in rb.SEASONS:
        points = lr.points_by_week(history, season)
        byes = rb.byes(schedules, season, list(range(1, 18)))
        per_week = {w: weekly[(season, w)] for w in range(1, 18)}
        for i in range(args.leagues):
            r = lr.play(season, 1000 * season + i, per_week, points, byes, cfg, n_sims=args.sims,
                        upgrade=None if args.no_upgrades else lr.UPGRADE)
            games += r.games
            odds += r.odds
            print(f"{season} league {i}: champion T{r.finals['champion']}", flush=True)
    games, odds = pd.DataFrame(games), pd.DataFrame(odds)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        games.to_parquet(out / "games.parquet")
        odds.to_parquet(out / "odds.parquet")
    report(games, odds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
