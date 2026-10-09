"""Our D/ST and K rankings against the FantasyPros expert consensus, graded
on the same weeks: rank correlation, and each week's top five -- how many
finished as a startable unit, and what they scored.

The consensus is pulled each week (and past weeks the API still serves,
when it says so), kept with FantasyPros' other data on the runner (never
committed); the page shows only the grades, never their rankings.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config
from .models.base import spearman
from .teams import normalize_team

log = logging.getLogger(__name__)
LOG = "dstk_consensus.parquet"


def log_path(cfg: Config):
    from .data import fantasypros as fp

    return fp.cache_dir(cfg) / LOG


def load(cfg: Config) -> pd.DataFrame:
    path = log_path(cfg)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["season", "week", "position", "team", "rank"])


def update(cfg: Config, season: int, week: int, graded_weeks=()) -> int:
    """Pull this week's consensus (refreshed each run until the week turns
    over) and any graded week not yet logged. Returns weeks added or
    refreshed."""
    from .data import fantasypros as fp

    if not fp.api_key():
        return 0
    have = load(cfg)
    logged = set(zip(have["season"], have["week"]))
    todo = [w for w in sorted(set(graded_weeks)) if (season, w) not in logged] + [week]
    frames, done = [have], 0
    for w in todo:
        try:
            got = fp.special_rankings(cfg, season, int(w))
        except Exception as exc:  # noqa: BLE001 - a benchmark; never block the page
            log.warning("FantasyPros D/ST and K consensus for week %s unavailable: %s", w, type(exc).__name__)
            continue
        if got.empty:
            continue
        got = got.assign(season=int(season), week=int(w), team=got["team"].map(normalize_team))
        frames = [f[~((f["season"] == season) & (f["week"] == int(w)))] if not f.empty else f for f in frames]
        frames.append(got)
        done += 1
    if done:
        out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
        log_path(cfg).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(log_path(cfg), index=False)
    return done


def compare(history: pd.DataFrame, consensus: pd.DataFrame, startable: int) -> pd.DataFrame:
    """One row per position and week both ranked: each source's rank
    correlation, its top five's points and how many of them finished in the
    top ``startable``, and the best five possible."""
    if history.empty or consensus.empty:
        return pd.DataFrame()
    h = history.assign(team=history["team"].map(normalize_team))
    j = h.merge(consensus.rename(columns={"rank": "fp_rank"}), on=["season", "week", "position", "team"], how="inner")
    rows = []
    for (season, week, pos), g in j.groupby(["season", "week", "position"]):
        if len(g) < 10:
            continue
        ours, theirs = g.nsmallest(5, "model_rank"), g.nsmallest(5, "fp_rank")
        rows.append({
            "season": season, "week": week, "position": pos, "units": len(g),
            "ours_corr": spearman(-g["model_rank"], g["actual_points"]),
            "fp_corr": spearman(-g["fp_rank"], g["actual_points"]),
            "ours_top5_pts": float(ours["actual_points"].mean()), "fp_top5_pts": float(theirs["actual_points"].mean()),
            "ours_top5_hits": int((ours["actual_rank"] <= startable).sum()),
            "fp_top5_hits": int((theirs["actual_rank"] <= startable).sum()),
            "best_top5_pts": float(g.nlargest(5, "actual_points")["actual_points"].mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Would blending the consensus into ours help? (run on the runner; prints
# aggregates only)
# ---------------------------------------------------------------------------
WALKFORWARD = "data/backtest/dstk_walkforward.parquet"


def _blend_scores(g: pd.DataFrame, w: float) -> pd.Series:
    """Higher is better: our projection's percentile and the consensus
    rank's, mixed ``w`` toward the consensus."""
    ours = g["expected_points"].rank(pct=True)
    theirs = (-g["fp_rank"]).rank(pct=True)
    return (1.0 - w) * ours + w * theirs


def _grade(j: pd.DataFrame, score_of, startable: int) -> dict:
    corr, pts, hits = [], [], []
    for _k, g in j.groupby(["season", "week", "position"]):
        s = score_of(g)
        corr.append(spearman(s, g["fantasy_points"]))
        top = g.loc[s.nlargest(5).index]
        pts.append(float(top["fantasy_points"].mean()))
        hits.append(float((top["actual_rank"] <= startable).mean()))
    return {"corr": float(np.nanmean(corr)), "top5_pts": float(np.mean(pts)), "top5_hits": float(np.mean(hits)),
            "weeks": len(corr)}


def blend_backtest(cfg: Config, sleep: float = 1.5) -> list[str]:
    """Fetch the consensus for every week of the walk-forward file (cached;
    resumable), then grade ours, theirs and blends, the weight picked on two
    seasons and scored on the third. Lines safe for a public log."""
    import time

    from .data import fantasypros as fp

    wf = pd.read_parquet(cfg.root / WALKFORWARD)
    wf = wf.assign(team=wf["team"].map(normalize_team))
    have = load(cfg)
    fetched = missing = 0
    frames = [have]
    for (season, week) in sorted(set(zip(wf["season"], wf["week"]))):
        if ((have["season"] == season) & (have["week"] == week)).any():
            continue
        got = fp.special_rankings(cfg, int(season), int(week))
        fetched += 1
        if got.empty:
            missing += 1
        else:
            frames.append(got.assign(season=int(season), week=int(week), team=got["team"].map(normalize_team)))
        time.sleep(sleep)
    cons = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if fetched:
        cons.to_parquet(log_path(cfg), index=False)
    j = wf.merge(cons.rename(columns={"rank": "fp_rank"}), on=["season", "week", "position", "team"], how="inner")
    j = j.dropna(subset=["fantasy_points", "expected_points", "fp_rank"])
    j["actual_rank"] = j.groupby(["season", "week", "position"])["fantasy_points"].rank(ascending=False, method="min")
    j = j[j.groupby(["season", "week", "position"])["team"].transform("size") >= 10]
    k = cfg.startable_rank
    out = [f"consensus weeks fetched this run: {fetched} ({missing} not served); matched rows: {len(j)}"]
    grid = [round(x, 1) for x in np.linspace(0, 1, 11)]
    for pos, g in j.groupby("position"):
        out.append(f"== {pos}: {g[['season', 'week']].drop_duplicates().shape[0]} weeks")
        for label, f in (("ours", lambda x: x["expected_points"]), ("consensus", lambda x: -x["fp_rank"]),
                         ("vegas baseline", lambda x: x["baseline_points"])):
            r = _grade(g, f, k)
            out.append(f"  {label:15s} corr {r['corr']:.3f}  top5 pts {r['top5_pts']:.2f}  top5 top-{k} {r['top5_hits']:.0%}")
        for w in grid[1:-1]:
            r = _grade(g, lambda x, w=w: _blend_scores(x, w), k)
            out.append(f"  blend w={w:.1f}     corr {r['corr']:.3f}  top5 pts {r['top5_pts']:.2f}  top5 top-{k} {r['top5_hits']:.0%}")
        seasons = sorted(g["season"].unique())
        for objective in ("corr", "top5_pts"):
            folds = []
            for s in seasons:
                tr, te = g[g["season"] != s], g[g["season"] == s]
                best = max(grid, key=lambda w: _grade(tr, lambda x, w=w: _blend_scores(x, w), k)[objective])
                ours = _grade(te, lambda x: x["expected_points"], k)
                blend = _grade(te, lambda x, w=best: _blend_scores(x, w), k)
                folds.append(f"{s}: w={best:.1f} corr {ours['corr']:.3f}->{blend['corr']:.3f} "
                             f"top5 {ours['top5_pts']:.2f}->{blend['top5_pts']:.2f}")
            out.append(f"  held out, weight picked on {objective}: " + " | ".join(folds))
    return out
