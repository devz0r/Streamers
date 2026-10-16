"""Sleeper's weekly projections: do they add to ours? Runs on the runner.

    python scripts/fit_sleeper_projections.py

The sandbox cannot reach api.sleeper.app, so this is a workflow job
(``sleeper-fit``). The public log carries only aggregates.

1. **Probe.** Fetch Sleeper's weekly QB/RB/WR/TE projections for 2023-2025
   (every week) and this season's played weeks, cached under data/raw
   (never committed). Reports whether past weeks are served at all, rows per
   season, and how many carry a PPR projection and map to an nflverse id
   (through ``gsis_id`` in Sleeper's player database, else by name, team
   and position, as production matches).
2. **Accuracy.** Against what players scored: Sleeper alone, and our
   production model's projection for the same week (replayed on the games
   before it, as ``ros_backtest`` does; weeks 4, 6, 8, 10, 12, 14), for
   players who played and were projected 3+ points.
3. **Blend.** ``(1 - w) * ours + w * sleeper``, ``w`` fitted on the other
   two seasons and scored on the third (leave one season out), judged on
   MAE and rank correlation: better overall and worse in at most one season.
4. **This season.** The published blend (``skill_log``: model, platforms,
   Vegas and FantasyPros) against Sleeper on the weeks played so far, and
   whether adding Sleeper to the blend helps.

DECISIONS.md, "Sleeper projections".
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

BASE = "https://api.sleeper.app/projections/nfl/{season}/{week}"
PARAMS = {"season_type": "regular", "position[]": ["QB", "RB", "WR", "TE"]}
SEASONS = (2023, 2024, 2025)
MODEL_WEEKS = (4, 6, 8, 10, 12, 14)
W_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)


def fetch(cfg, season: int, week: int) -> list | None:
    """One week's projections (cached), or None when the endpoint does not serve it."""
    import requests

    path = cfg.raw_dir / "sleeper_proj" / f"{season}_{week}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    try:
        resp = requests.get(BASE.format(season=season, week=week), params=PARAMS, timeout=60)
    except Exception as exc:  # noqa: BLE001
        print(f"  {season} wk{week}: {type(exc).__name__}")
        return None
    if resp.status_code != 200:
        print(f"  {season} wk{week}: HTTP {resp.status_code}")
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(0.3)
    return data


def frame(cfg, data: list, db: dict, who: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """One week's projections mapped to nflverse ids the way production maps
    them (``sleeper.match``): by gsis id, else by name, team and position
    against the players in that week's box scores."""
    from streamer.data import sleeper

    rows = sleeper.projection_rows(cfg, season, week, allow_network=False, data=data, db=db)
    if not rows:
        return pd.DataFrame(columns=["season", "week", "gsis", "sleeper", "by_id"])
    wk = who[(who.season == season) & (who.week == week)]
    people = [(r.player_id, r.player_id, r.player_display_name, r.team, r.position) for r in wk.itertuples()]
    got = sleeper.match(rows, people)
    ids = {r["gsis"] for r in rows if r["gsis"]}
    return pd.DataFrame([{"season": season, "week": week, "gsis": k, "sleeper": v, "by_id": k in ids}
                         for k, v in got.items()])


def score(d: pd.DataFrame, col: str) -> pd.DataFrame:
    out = []
    for s, g in d.groupby("season"):
        wk = g.groupby("week").apply(lambda x: x[col].rank().corr(x["actual"].rank()))
        out.append({"season": s, "mae": float((g[col] - g["actual"]).abs().mean()),
                    "bias": float((g[col] - g["actual"]).mean()), "rank": float(wk.mean())})
    return pd.DataFrame(out).set_index("season")


def main() -> int:
    logging.disable(logging.WARNING)
    from streamer.config import get_config
    from streamer.data import sleeper
    from streamer.roster import ros_backtest as rb
    from streamer.roster.projections import load_history

    cfg = get_config().for_profile("espn")
    db = sleeper.players(cfg) or {}
    gsis = {str(k): v.get("gsis_id") for k, v in db.items() if isinstance(v, dict) and v.get("gsis_id")}
    print(f"Sleeper player database: {len(db)} players, {len(gsis)} with a gsis_id")
    history = load_history(cfg)
    who = history[["season", "week", "player_id", "player_display_name", "team", "position"]] \
        .drop_duplicates(["season", "week", "player_id"])

    # 1. Probe.
    frames = []
    for season in (*SEASONS, cfg.current_season):
        got = served = 0
        for week in range(1, 18 if season < cfg.current_season else 7):
            data = fetch(cfg, season, week)
            if not data:
                continue
            f = frame(cfg, data, db, who, season, week)
            frames.append(f)
            served += 1
            got += len(f)
        print(f"{season}: {served} weeks served, {got} rows")
    if not frames:
        print("VERDICT: Sleeper serves no projections here -> DROP")
        return 0
    sl = pd.concat(frames, ignore_index=True)
    sl["sleeper"] = pd.to_numeric(sl["sleeper"], errors="coerce")
    for s, g in sl.groupby("season"):
        print(f"  {s}: {len(g)} rows mapped to nflverse players who played, "
              f"{g.by_id.mean():.0%} of them by gsis id, the rest by name/team/position; "
              f"mean over the 3+ point rows {g.loc[g.sleeper >= 3, 'sleeper'].mean():.2f}")
    sl = sl.dropna(subset=["gsis", "sleeper"]).drop_duplicates(["season", "week", "gsis"])

    # 2. Ours, replayed on the games before each week.
    actual = (history.groupby(["season", "week", "player_id"])["fantasy_points_ppr"].sum()
              .rename("actual").reset_index().rename(columns={"player_id": "gsis"}))
    rost = {s: rb.rosters(cfg, s) for s in SEASONS}
    inj = {s: rb.injuries(cfg, s) for s in SEASONS}
    ours = []
    for s in SEASONS:
        for w in MODEL_WEEKS:
            try:
                ps = rb.project(history, cfg, s, w, rb.statuses(rost[s], inj[s], w))
            except Exception as exc:  # noqa: BLE001
                print(f"  model {s} wk{w}: {type(exc).__name__}")
                continue
            ours.append(pd.DataFrame([{"season": s, "week": w, "gsis": p.player_id, "position": p.position,
                                       "ours": p.projection} for p in ps if p.projection is not None]))
        print(f"  model {s}: {sum(len(o) for o in ours if o.season.iloc[0] == s) if ours else 0} projections")
    if not ours:
        print("VERDICT: could not replay our model -> DROP")
        return 0
    d = pd.concat(ours).merge(sl, on=["season", "week", "gsis"]).merge(actual, on=["season", "week", "gsis"])
    d = d[(d.ours >= 3) & (d.sleeper >= 0)]
    print(f"\ngraded rows (played, ours >= 3): {len(d)}")
    for name in ("ours", "sleeper"):
        sc = score(d, name)
        print(f"  {name:8} MAE {sc.mae.mean():.4f} rank {sc['rank'].mean():.4f} bias {sc.bias.mean():+.3f}  "
              + "  ".join(f"{s}: {r.mae:.3f}/{r['rank']:.3f}" for s, r in sc.iterrows()))
    for pos, g in d.groupby("position"):
        print(f"    {pos}: n {len(g)}  ours MAE {(g.ours - g.actual).abs().mean():.3f}  "
              f"sleeper {(g.sleeper - g.actual).abs().mean():.3f}  "
              f"50/50 {((g.ours + g.sleeper) / 2 - g.actual).abs().mean():.3f}")

    # 3. Leave-one-season-out blend weight.
    blended, picks = [], []
    for s in SEASONS:
        tr, te = d[d.season != s], d[d.season == s]
        w = min(W_GRID, key=lambda v: float((((1 - v) * tr.ours + v * tr.sleeper) - tr.actual).abs().mean()))
        picks.append(f"{s}: w {w:.2f}")
        blended.append(te.assign(blend=(1 - w) * te.ours + w * te.sleeper))
    bl = score(pd.concat(blended), "blend")
    base = score(d, "ours")
    better = int((bl.mae < base.mae).sum())
    keep = bl.mae.mean() < base.mae.mean() and better >= len(bl) - 1 and bl["rank"].mean() >= base["rank"].mean()
    print(f"\nblend with ours: MAE {base.mae.mean():.4f} -> {bl.mae.mean():.4f} ({better}/{len(bl)} better)  "
          f"rank {base['rank'].mean():.4f} -> {bl['rank'].mean():.4f}  [{'; '.join(picks)}]")
    for s in bl.index:
        print(f"    {s}: MAE {base.mae[s]:.3f} -> {bl.mae[s]:.3f}   rank {base['rank'][s]:.3f} -> {bl['rank'][s]:.3f}")
    allb = pd.concat(blended)
    for label, g in (("matched by gsis id", allb[allb.by_id]), ("matched by name", allb[~allb.by_id])):
        print(f"    {label:18} n {len(g):5}  MAE {(g.ours - g.actual).abs().mean():.3f} -> "
              f"{(g.blend - g.actual).abs().mean():.3f}")
    print(f"VERDICT (against our model alone): {'KEEP' if keep else 'DROP'}")

    # 4. This season, against the published blend.
    log_path = cfg.results_dir / "skill_log.parquet"
    if log_path.exists():
        log = pd.read_parquet(log_path)
        log = log[log.season == cfg.current_season].dropna(subset=["blended_projection"]) \
            .rename(columns={"nfl_id": "gsis"})
        cur = log.merge(sl[sl.season == cfg.current_season], on=["season", "week", "gsis"]).merge(
            actual, on=["season", "week", "gsis"])
        cur = cur[cur.blended_projection >= 3]
        print(f"\n{cfg.current_season} so far: {len(cur)} graded rows over weeks {sorted(cur.week.unique())}")
        for name in ("blended_projection", "sleeper", "model_projection"):
            x = cur.dropna(subset=[name])
            print(f"  {name:20} MAE {(x[name] - x.actual).abs().mean():.4f} (n {len(x)})")
        for w in (0.0, 0.1, 0.2, 0.3, 0.5):
            m = (1 - w) * cur.blended_projection + w * cur.sleeper
            print(f"  published blend + {w:.0%} Sleeper: MAE {(m - cur.actual).abs().mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
