"""Our D/ST and K rankings against the FantasyPros expert consensus, graded
on the same weeks: rank correlation, and each week's top five -- how many
finished as a startable unit, and what they scored.

The consensus is pulled each week (and past weeks the API still serves,
when it says so), kept with FantasyPros' other data on the runner (never
committed); the page shows only the grades, never their rankings.
"""

from __future__ import annotations

import logging

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
