"""Fit and grade the calibration of the top of the weekly number (roster/level_calibration.py).

    python scripts/fit_level_calibration.py [--weekly cache.pkl] [--units cache.parquet] [--leagues 20] [--write]

Every 2022-2025 player-week, projected by the production code from the games
before it (``ros_backtest.project``, closing lines in, this calibration off),
against what he scored. Prints

* how much of a projected difference holds up among starter-calibre players,
  by position (the slope of points scored on points projected);
* the calibration line per position fitted with each season held out in turn,
  and the projection's RMSE and that slope on the held-out season, as it is
  and calibrated;
* with ``--leagues N`` (per season), the replayed leagues of
  ``league_replay`` played on the projections as they are, with each
  head-to-head game's win chance priced both ways on the same lineups and the
  same draws' seed -- so the comparison is of the same games -- each map
  fitted on the other three seasons: log loss as drawn and through the
  mapping ``Phi(k Phi^-1(p))`` with ``k`` held out by season, against today's.

``--write`` saves the fit on all four seasons to ``level_calibration.json``.
DECISIONS.md, "The top of the weekly number".
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

SEASONS = (2022, 2023, 2024, 2025)
FIT = ("RB", "WR")                       # positions calibrated
SHOWN = ("QB", "RB", "WR", "TE")
POOL = {"QB": 12, "RB": 30, "WR": 36, "TE": 12}      # starter-calibre: this many a week by projection
KNOT = 12.0
K_TODAY = 0.78                          # lineup.WIN_SHRINK before this calibration


def table(weekly: dict, history: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per projected skill player-week: position, tag, the if-he-plays
    number, his chance to play as the draws have it, and what he scored."""
    from streamer.roster import league_replay as lr
    from streamer.roster import ros_backtest as rb
    from streamer.roster.lineup import _if_plays

    rows = []
    for season in SEASONS:
        pts = lr.points_by_week(history, season)
        byes = rb.byes(schedules, season, list(range(1, 18)))
        for week in range(1, 18):
            for p in weekly[(season, week)]:
                m = float(p.projection or 0.0)
                if m <= 0 or (p.team and week in byes.get(p.team, set())):
                    continue
                mean, _sd, q = _if_plays(p)
                rows.append((season, week, p.player_id, p.position, p.status or "", m, mean, q,
                             float(p.ros_value or 0.0), pts.get((week, p.player_id), 0.0)))
    return pd.DataFrame(rows, columns=["season", "week", "pid", "pos", "status", "m", "mif", "q", "ros", "y"])


def fit(d: pd.DataFrame, col: str = "m") -> dict[str, dict]:
    """The line with a knot at KNOT, per position, on untagged players projected 3+:
    points scored on this week's number (``m``) or on the season value (``ros``)."""
    out = {}
    g = d[(d.status == "") & (d.m >= 3)]
    for pos in FIT:
        x = g[g.pos == pos]
        X = np.column_stack([np.ones(len(x)), x[col], np.maximum(x[col] - KNOT, 0)])
        a, b, c = np.linalg.lstsq(X, x.y, rcond=None)[0]
        out[pos] = {"a": round(float(a), 3), "b": round(float(b), 3), "c": round(float(c), 3), "n": int(len(x))}
    return out


def calibrated(d: pd.DataFrame, coef: dict) -> np.ndarray:
    from streamer.roster import level_calibration as lc

    conf = {"knot": KNOT, "positions": coef}
    return np.array([m if qq <= 0 else max(qq, 0.05) * lc.apply(pos, mif, conf)
                     for pos, m, mif, qq in zip(d.pos, d.m, d.mif, d.q)])


def slope(x, y) -> float:
    x = np.asarray(x, float) - np.mean(x)
    return float((x * (np.asarray(y, float) - np.mean(y))).sum() / (x * x).sum())


def rmse(e) -> float:
    return float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))


def report_players(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d.m >= 3].copy()
    d["rk"] = d.groupby(["season", "week", "pos"]).m.rank(ascending=False, method="first")
    print("How much of a projected difference holds up (points scored on points projected, 1 = all of it):")
    for pos in SHOWN:
        g = d[d.pos == pos]
        st = g[g.rk <= POOL[pos]]
        print(f"  {pos}: everyone projected 3+ {slope(g.m, g.y):.2f} (n={len(g)}); the {POOL[pos]} a week a lineup chooses "
              f"between {slope(st.m, st.y):.2f} (n={len(st)}; by season "
              + ", ".join(f"{slope(st[st.season == s].m, st[st.season == s].y):.2f}" for s in SEASONS) + ")")
        if pos in FIT:
            st = st.assign(b=pd.qcut(st.m, 5, labels=False))
            print("      by fifth of the pool, projected -> scored: "
                  + ", ".join(f"{a:.1f} -> {b:.1f}" for a, b in zip(st.groupby("b").m.mean(), st.groupby("b").y.mean())))
    d["c"] = d.m
    folds = {}
    for s in SEASONS:
        folds[s] = fit(d[d.season != s])
        mk = d.season == s
        d.loc[mk, "c"] = calibrated(d[mk], folds[s])
    print("\nThe line fitted without each season in turn (a + b x + c max(x - 12, 0)):")
    for s in SEASONS:
        print(f"  without {s}: " + "; ".join(f"{p} a={c['a']:+.2f} b={c['b']:.3f} c={c['c']:+.3f}" for p, c in folds[s].items()))
    print("\nHeld out (each season scored on the line fitted on the other three): RMSE as it is -> calibrated")
    for label, g in (("backs and receivers projected 3+", d[d.pos.isin(FIT)]),
                     ("  the 30 and 36 a week", d[d.pos.isin(FIT) & (d.rk <= d.pos.map(POOL))]),
                     ("  backs", d[d.pos == "RB"]), ("  receivers", d[d.pos == "WR"])):
        per = [(rmse(g[g.season == s].y - g[g.season == s].m), rmse(g[g.season == s].y - g[g.season == s].c)) for s in SEASONS]
        print(f"  {label} (n={len(g)}): {rmse(g.y - g.m):.4f} -> {rmse(g.y - g.c):.4f}; "
              + ", ".join(f"{s} {a:.3f} -> {b:.3f}" for s, (a, b) in zip(SEASONS, per))
              + f"; better in {sum(b < a for a, b in per)} of 4; slope {slope(g.m, g.y):.3f} -> {slope(g.c, g.y):.3f}")
    return d


def replay(d: pd.DataFrame, weekly: dict, units, history, schedules, cfg, leagues: int, fmt=None) -> None:
    """The same leagues and lineups (played on the projections as they are), each game priced both ways."""
    from streamer.roster import league_replay as lr
    from streamer.roster import level_calibration as lc
    from streamer.roster import lineup
    from streamer.roster import ros_backtest as rb
    from streamer.roster.lineup import _if_plays

    rows = []
    for season in SEASONS:
        conf = {"knot": KNOT, "positions": fit(d[d.season != season])}
        points = lr.points_by_week(history, season)
        unit_week = None
        if units is not None:
            unit_week, unit_points = lr.unit_rows(units, season)
            points.update(unit_points)
        byes = rb.byes(schedules, season, list(range(1, 18)))
        per_week = {w: weekly[(season, w)] for w in range(1, 18)}
        for i in range(leagues):
            seed, got = 1000 * season + i, []

            def on_game(week, a, b, la, lb, draws, _sat, got=got, seed=seed, conf=conf):
                both = []
                for p in la + lb:
                    m = float(p.projection or 0.0)
                    if p.position in FIT and m >= 3:
                        mean, _sd, q = _if_plays(p)
                        if q > 0:
                            p = copy.copy(p)
                            p.projection = max(q, 0.05) * lc.apply(p.position, m / max(q, 0.05), conf)
                    both.append(p)
                new = lineup.sample_points(both, draws.shape[0], np.random.default_rng(seed * 1000 + week * 10 + a))
                n_a = len(la)
                rec = {"week": week, "a": a}
                for name, x in (("old", draws), ("new", new)):
                    ta, tb = x[:, :n_a].sum(axis=1), x[:, n_a:].sum(axis=1)
                    rec[name] = float((ta > tb).mean() + 0.5 * (ta == tb).mean())
                    rec["gap_" + name] = float(ta.mean() - tb.mean())
                got.append(rec)

            r = lr.play(season, seed, per_week, points, byes, cfg, units=unit_week, checkpoints=(), on_game=on_game,
                        fmt=fmt or lr.ESPN)
            res = {(g["week"], g["a"]): g for g in r.games}
            rows += [{"season": season, **x, "won": res[(x["week"], x["a"])]["won"],
                      "margin": res[(x["week"], x["a"])]["score_a"] - res[(x["week"], x["a"])]["score_b"]} for x in got]
        print(f"replayed {season}", flush=True)
    g = pd.DataFrame(rows)
    y = g.won.to_numpy()

    def ll(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def mapped(p, k):
        return norm.cdf(k * norm.ppf(np.clip(p, 1e-6, 1 - 1e-6)))

    ks = np.round(np.arange(0.6, 1.21, 0.01), 2)
    today = ll(mapped(g.old.to_numpy(), K_TODAY))
    print(f"\nThis week's win chance on {len(g)} games of {leagues * len(SEASONS)} leagues, the same games priced both ways:")
    for name in ("old", "new"):
        gap = g["gap_" + name]
        print(f"  projections {'as they are' if name == 'old' else 'calibrated'}: the real margin on the projected one "
              f"{float((gap * g.margin).sum() / (gap ** 2).sum()):.3f} (by season "
              + ", ".join(f"{float((gap[g.season == s] * g.margin[g.season == s]).sum() / (gap[g.season == s] ** 2).sum()):.2f}" for s in SEASONS)
              + f"); favourites drawn at {np.where(g[name] >= 0.5, g[name], 1 - g[name]).mean():.1%}, won "
              f"{np.where(g[name] >= 0.5, y, 1 - y).mean():.1%}")
    raw = g.new.to_numpy()
    held = np.zeros(len(g))
    picked = []
    for s in SEASONS:
        tr = (g.season != s).to_numpy()
        k = float(ks[int(np.argmin([ll(mapped(raw, k))[tr].mean() for k in ks]))])
        picked.append(k)
        held[~tr] = mapped(raw[~tr], k)
    best = float(ks[int(np.argmin([ll(mapped(raw, k)).mean() for k in ks]))])
    rng = np.random.default_rng(0)
    for name, p in ((f"today: as they are, k={K_TODAY}", mapped(g.old.to_numpy(), K_TODAY)), ("calibrated, as drawn", raw),
                    (f"calibrated, k held out by season ({', '.join(f'{k:g}' for k in picked)})", held)):
        d_ = ll(p) - today
        blocks = pd.DataFrame({"s": g.season, "w": g.week, "d": d_}).groupby(["s", "w"]).d.agg(["sum", "size"])
        idx = rng.integers(0, len(blocks), (2000, len(blocks)))
        boot = blocks["sum"].to_numpy()[idx].sum(axis=1) / blocks["size"].to_numpy()[idx].sum(axis=1)
        print(f"  {name}: log loss {ll(p).mean():.5f}, Brier {((p - y) ** 2).mean():.5f}; by season "
              + ", ".join(f"{ll(p)[(g.season == s).to_numpy()].mean():.5f}" for s in SEASONS)
              + f"; against today {d_.mean():+.5f} (NFL weeks resampled: 95% {np.quantile(boot, 0.025):+.5f} to "
              f"{np.quantile(boot, 0.975):+.5f}), better in "
              f"{sum(ll(p)[(g.season == s).to_numpy()].mean() < today[(g.season == s).to_numpy()].mean() for s in SEASONS)} of 4")
    print(f"  k on all four seasons: {best:g} (lineup.WIN_SHRINK)")


def replay_season(d: pd.DataFrame, weekly: dict, units, history, schedules, cfg, leagues: int, fmt=None) -> None:
    """The season simulator's odds on the same leagues, with the week's number and the season value as
    they are and both calibrated (each line fitted on the other three seasons)."""
    from streamer.roster import league_replay as lr
    from streamer.roster import level_calibration as lc
    from streamer.roster import ros_backtest as rb
    from streamer.roster.lineup import _if_plays

    odds = []
    for season in SEASONS:
        tr = d[d.season != season]
        conf = {"knot": KNOT, "positions": fit(tr), "season": fit(tr, "ros")}

        def both(p, conf=conf):
            m = float(p.projection or 0.0)
            if p.position not in FIT:
                return p
            p = copy.copy(p)
            _mean, _sd, q = _if_plays(p)
            if m >= 3 and q > 0:
                p.projection = max(q, 0.05) * lc.apply(p.position, m / max(q, 0.05), conf)
            if p.ros_value is not None:
                p.ros_value = lc.season(p.position, float(p.ros_value), conf)
            return p

        points = lr.points_by_week(history, season)
        unit_week = None
        if units is not None:
            unit_week, unit_points = lr.unit_rows(units, season)
            points.update(unit_points)
        byes = rb.byes(schedules, season, list(range(1, 18)))
        per_week = {w: weekly[(season, w)] for w in range(1, 18)}
        for i in range(leagues):
            r = lr.play(season, 1000 * season + i, per_week, points, byes, cfg, units=unit_week,
                        odds_variants={"cal": both}, fmt=fmt or lr.ESPN)
            odds += [{k: v for k, v in o.items() if k != "roster"} for o in r.odds]
        print(f"season odds replayed {season}", flush=True)
    o = pd.DataFrame(odds)
    brier = lambda p, y: float(((p - y) ** 2).mean())                                      # noqa: E731
    print(f"\nThe season simulator on {len(o)} team-checkpoints of {leagues * len(SEASONS)} leagues (the same leagues both ways):")
    for col, y, base in (("p_playoffs", "made", "base_playoffs"), ("p_title", "won_title", "base_title")):
        a = [brier(o[o.season == s][col], o[o.season == s][y]) for s in SEASONS]
        b = [brier(o[o.season == s][col + "_cal"], o[o.season == s][y]) for s in SEASONS]
        print(f"  {col}: Brier {brier(o[col], o[y]):.5f} -> {brier(o[col + '_cal'], o[y]):.5f} (standings only "
              f"{brier(o[base], o[y]):.5f}); by season " + ", ".join(f"{s} {x:.4f} -> {z:.4f}" for s, x, z in zip(SEASONS, a, b))
              + f"; better in {sum(z < x for x, z in zip(a, b))} of 4")
    key = ["season", "seed", "week"]
    for name, col in (("as they are", "exp_points"), ("calibrated", "exp_points_cal")):
        x = o[col] - o.groupby(key)[col].transform("mean")
        yy = o.rest_points - o.groupby(key).rest_points.transform("mean")
        print(f"  team strength, real points for the rest of the season on expected ({name}): {float((x * yy).sum() / (x * x).sum()):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", help="cache of the weekly projections with this calibration off (built if missing)")
    ap.add_argument("--units", help="cache of the defence and kicker projections (built if missing)")
    ap.add_argument("--leagues", type=int, default=0, help="leagues per season to replay (0: the player tables only)")
    ap.add_argument("--season-leagues", type=int, default=0, help="leagues per season for the season simulator's odds")
    ap.add_argument("--profile", default="espn", choices=("espn", "yahoo"),
                    help="the league whose sheet and format are graded (--weekly must have been built for it)")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("STREAMER_CREDIT_FREE", "1")
    logging.disable(logging.WARNING)
    import grade_league_odds as base

    from streamer.config import get_config
    from streamer.roster import league_replay as lr
    from streamer.roster import level_calibration as lc
    from streamer.roster.projections import load_history

    lc.OFF = lc.SEASON_OFF = True        # the projection and the simulator's levels as they are before this step
    cfg = get_config().for_profile(args.profile)
    fmt = lr.FORMATS[args.profile]
    weekly = base.weekly_projections(cfg, args.weekly)
    history = load_history(cfg)
    schedules = pd.read_parquet(cfg.raw_dir / "schedules.parquet")
    d = table(weekly, history, schedules)
    report_players(d)
    full, full_season = fit(d), fit(d, "ros")
    print("\nOn all four seasons: " + "; ".join(f"{p} a={c['a']:+.3f} b={c['b']:.3f} c={c['c']:+.3f} (n={c['n']})" for p, c in full.items()))
    for pos in full:
        print(f"  {pos}: " + ", ".join(f"{x} -> {lc.apply(pos, x, {'knot': KNOT, 'positions': full}):.1f}" for x in (2, 5, 8, 10, 12, 15, 18, 22, 26)))
    print("The season value, the same line: " + "; ".join(f"{p} a={c['a']:+.3f} b={c['b']:.3f} c={c['c']:+.3f}" for p, c in full_season.items()))
    if args.leagues or args.season_leagues:
        units = base.unit_projections(cfg, args.units)
        if args.leagues:
            replay(d, weekly, units, history, schedules, cfg, args.leagues, fmt)
        if args.season_leagues:
            replay_season(d, weekly, units, history, schedules, cfg, args.season_leagues, fmt)
    if args.write and args.profile != "espn":
        print("--write is for the shipped (ESPN-fitted) line; not written for", args.profile)
    elif args.write:
        out = {"knot": KNOT, "positions": {p: {k: c[k] for k in ("a", "b", "c")} for p, c in full.items()},
               "season": {p: {k: c[k] for k in ("a", "b", "c")} for p, c in full_season.items()},
               "evidence": "scripts/fit_level_calibration.py: 2022-2025 walk-forward, untagged players projected 3+, "
                           + ", ".join(f"{p} n={c['n']}" for p, c in full.items())}
        lc.MODEL_PATH.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print("wrote", lc.MODEL_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
