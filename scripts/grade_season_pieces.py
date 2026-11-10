"""Grade the season simulator piece by piece on replayed leagues.

    python scripts/grade_season_pieces.py [--profile espn] [--leagues 4] [--seasons 2022,2023] [--weekly ...] [--units ...] [--out dir]
    python scripts/grade_season_pieces.py --report dir [dir ...]

P(playoffs) and P(title) are graded as wholes (``grade_league_odds.py``).
Underneath them the simulator does one thing: it draws every team's score for
every week left. The standings, seeds and bracket are bookkeeping on those
draws. So the pieces graded here are the draws themselves, at each checkpoint
(weeks 1, 4, 7, 10, 13), against what each replayed team really scored:

* **level** -- the mean of a team's simulated week against its real score,
  by how many weeks ahead: bias, and the strength slope (real points over the
  league's average that week on simulated points over it; 1 when teams are
  told apart as much as results do);
* **spread** -- where the real score fell in its simulated range (the share
  under each simulated decile; 10% each when the range is right);
* **persistence** -- the rest of the regular season summed: whether the real
  total falls in the simulated total's range (week-to-week streaks in a team's
  luck widen it beyond the weeks' own spreads);
* **games** -- each scheduled head-to-head game's simulated win chance
  against its result, by weeks ahead;
* **playoff weeks** -- the same for weeks 15-17, which decide the title.

The bracket itself is pinned by tests (``tests/test_league_replay.py``):
fed the real results, the simulator's seeding and bracket give the real
playoff field and champion. DECISIONS.md, "The season simulator, piece by piece".
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
DECILES = np.linspace(0.1, 0.9, 9)


def horizon(h: int, playoff: bool) -> str:
    if playoff:
        return "playoff weeks"
    return "this week" if h == 0 else "1-3 ahead" if h <= 3 else "4-7 ahead" if h <= 7 else "8-13 ahead"


def grade_league(r, reg_weeks: int, season: int, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Per (checkpoint, team, week): the simulated score's mean, spread and where the real one fell; per
    (checkpoint, team): the rest of the regular season summed; per (checkpoint, game): its win chance."""
    weeks_rows, totals, games = [], [], []
    for c, (weeks, arr) in r.sims.items():
        real = np.stack([np.asarray(r.scores[w], float) for w in weeks], axis=1)          # teams x weeks
        n_t = arr.shape[0]
        for k, w in enumerate(weeks):
            sim = arr[:, :, k].astype(float)
            mean = sim.mean(axis=1)
            for t in range(n_t):
                x = sim[t]
                weeks_rows.append({"season": season, "seed": seed, "checkpoint": c, "week": w, "team": t,
                                   "ahead": w - c, "playoff": w > reg_weeks, "real": real[t, k],
                                   "mean": mean[t], "sd": float(x.std()), "league_mean": float(mean.mean()),
                                   "league_real": float(real[:, k].mean()),
                                   "pit": float((x < real[t, k]).mean() + 0.5 * (x == real[t, k]).mean())})
        reg = [k for k, w in enumerate(weeks) if w <= reg_weeks]
        if reg:
            tot = arr[:, :, reg].astype(float).sum(axis=2)
            rt = real[:, reg].sum(axis=1)
            var_weeks = arr[:, :, reg].astype(float).var(axis=1).sum(axis=1)
            for t in range(n_t):
                totals.append({"season": season, "seed": seed, "checkpoint": c, "team": t, "weeks": len(reg),
                               "real": rt[t], "mean": float(tot[t].mean()), "sd": float(tot[t].std()),
                               "sd_independent": float(np.sqrt(var_weeks[t])),
                               "league_mean": float(tot.mean()), "league_real": float(rt.mean()),
                               "pit": float((tot[t] < rt[t]).mean())})
        for w, a, b in r.schedule:
            if w < c or w not in weeks:
                continue
            k = weeks.index(w)
            sa, sb = arr[a, :, k].astype(float), arr[b, :, k].astype(float)
            p = float((sa > sb).mean() + 0.5 * (sa == sb).mean())
            ra, rb = float(r.scores[w][a]), float(r.scores[w][b])
            games.append({"season": season, "seed": seed, "checkpoint": c, "week": w, "ahead": w - c, "p": p,
                          "won": 1.0 if ra > rb else 0.5 if ra == rb else 0.0})
    return weeks_rows, totals, games


def run(args) -> None:
    import grade_league_odds as base

    from streamer.config import get_config
    from streamer.roster import league_replay as lr
    from streamer.roster import ros_backtest as rb
    from streamer.roster.projections import load_history

    cfg = get_config().for_profile(args.profile)
    from streamer.roster.lineup import use_profile

    use_profile(cfg)                     # the league's own P(win) calibration, as production prices it
    fmt = lr.FORMATS[args.profile]
    weekly = base.weekly_projections(cfg, args.weekly)
    unit_frame = base.unit_projections(cfg, args.units)
    history = load_history(cfg)
    schedules = pd.read_parquet(cfg.raw_dir / "schedules.parquet")
    W, T, G = [], [], []
    for season in [int(s) for s in args.seasons.split(",")]:
        points = lr.points_by_week(history, season)
        units, unit_points = lr.unit_rows(unit_frame, season)
        points.update(unit_points)
        byes = rb.byes(schedules, season, list(range(1, 18)))
        per_week = {w: weekly[(season, w)] for w in range(1, 18)}
        for i in range(args.leagues):
            seed = 1000 * season + i
            r = lr.play(season, seed, per_week, points, byes, cfg, n_sims=args.sims, units=units, fmt=fmt,
                        keep_sims=True)
            w, t, g = grade_league(r, fmt.reg_weeks, season, seed)
            W += w
            T += t
            G += g
            print(f"{season} league {i}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.seasons.replace(",", "_")
    pd.DataFrame(W).to_parquet(out / f"weeks_{tag}.parquet")
    pd.DataFrame(T).to_parquet(out / f"totals_{tag}.parquet")
    pd.DataFrame(G).to_parquet(out / f"games_{tag}.parquet")


def slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else float("nan")


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def report(dirs: list[str]) -> None:
    W = pd.concat([pd.read_parquet(f) for d in dirs for f in sorted(Path(d).glob("weeks_*.parquet"))], ignore_index=True)
    T = pd.concat([pd.read_parquet(f) for d in dirs for f in sorted(Path(d).glob("totals_*.parquet"))], ignore_index=True)
    G = pd.concat([pd.read_parquet(f) for d in dirs for f in sorted(Path(d).glob("games_*.parquet"))], ignore_index=True)
    S = sorted(W.season.unique())
    W["h"] = [horizon(a, p) for a, p in zip(W.ahead, W.playoff)]
    G["h"] = [horizon(a, False) for a in G.ahead]
    order = ["this week", "1-3 ahead", "4-7 ahead", "8-13 ahead", "playoff weeks"]
    print(f"{W.groupby(['season', 'seed']).ngroups} leagues, seasons {S}: {len(W)} team-weeks simulated and graded")
    print("\nLevel: real minus simulated mean, and the strength slope (real over the league's week on simulated over it)")
    for h in order:
        x = W[W.h == h]
        if not len(x):
            continue
        by = ", ".join(f"{s} {slope((g['mean'] - g.league_mean), (g.real - g.league_real)):.2f}" for s, g in x.groupby("season"))
        print(f"  {h:13s} n={len(x):6d}  bias {(x.real - x['mean']).mean():+.2f} a week (league-wide {(x.league_real - x.league_mean).mean():+.2f})"
              f"  strength slope {slope(x['mean'] - x.league_mean, x.real - x.league_real):.2f} (by season {by})"
              f"  spread of simulated means {(x['mean'] - x.league_mean).std():.1f}, of real {(x.real - x.league_real).std():.1f}")
    print("\nSpread: the share of real scores under each simulated decile (10% each when right), and inside the 10-90 range (80%)")
    for h in order:
        x = W[W.h == h]
        if not len(x):
            continue
        q = np.histogram(x.pit, bins=np.r_[0, DECILES, 1])[0] / len(x)
        print(f"  {h:13s} " + " ".join(f"{v:4.1%}" for v in q) + f"   inside 10-90: {((x.pit >= .1) & (x.pit <= .9)).mean():.1%}"
              f"   real sd about the mean {(x.real - x['mean']).std():.1f} vs simulated {np.sqrt((x.sd ** 2).mean()):.1f}")
    print("\nPersistence: the rest of the regular season summed -- the real total's place in the simulated total's range,"
          " and the total's simulated spread against the weeks' own spreads added up (as if independent)")
    for c, x in T.groupby("checkpoint"):
        q = np.histogram(x.pit, bins=[0, .1, .25, .5, .75, .9, 1])[0] / len(x)
        resid = (x.real - x.league_real) - (x["mean"] - x.league_mean)
        print(f"  from week {c:2d} ({x.weeks.iloc[0]:2d} weeks): under 10/25/50/75/90th {q[0]:.1%} {q[:2].sum():.1%} {q[:3].sum():.1%} "
              f"{q[:4].sum():.1%} {q[:5].sum():.1%}  strength slope {slope(x['mean'] - x.league_mean, x.real - x.league_real):.2f}"
              f"  sd: real about the mean {resid.std():.1f}, simulated {np.sqrt((x.sd ** 2).mean()):.1f} (independent weeks"
              f" {np.sqrt((x.sd_independent ** 2).mean()):.1f})")
    print("\nGames: the simulated win chance against the result, by weeks ahead (Brier; a coin flip is 0.25)")
    for h in order[:-1]:
        x = G[G.h == h]
        if not len(x):
            continue
        fav = x.p >= 0.5
        pf, yf = np.where(fav, x.p, 1 - x.p), np.where(fav, x.won, 1 - x.won)
        z = np.log(np.clip(x.p, 1e-4, 1 - 1e-4) / np.clip(1 - x.p, 1e-4, 1 - 1e-4))
        ks = np.arange(0.4, 1.61, 0.02)
        kbest = float(ks[int(np.argmin([brier(1 / (1 + np.exp(-k * z)), x.won) for k in ks]))])
        print(f"  {h:13s} n={len(x):5d} Brier {brier(x.p, x.won):.4f}  favourites {pf.mean():.1%} drawn, {yf.mean():.1%} won;"
              f"  the logit scale that fits: {kbest:.2f} (by season "
              + ", ".join(f"{float(ks[int(np.argmin([brier(1 / (1 + np.exp(-k * z[x.season == s])), x.won[x.season == s]) for k in ks]))]):.2f}" for s in S)
              + ")")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="espn", choices=("espn", "yahoo"))
    ap.add_argument("--leagues", type=int, default=4)
    ap.add_argument("--seasons", default="2022,2023,2024,2025")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--weekly")
    ap.add_argument("--units")
    ap.add_argument("--out", default="season_pieces")
    ap.add_argument("--report", nargs="+")
    args = ap.parse_args()
    os.environ.setdefault("STREAMER_CREDIT_FREE", "1")
    logging.disable(logging.WARNING)
    if args.report:
        report(args.report)
        return 0
    run(args)
    report([args.out])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
