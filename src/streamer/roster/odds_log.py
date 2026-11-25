"""A record of our season odds, to grade them when the season is over.

Nobody has checked whether "41% to make the playoffs" means 41%. Every run
appends each team's odds (playoffs, title, expected wins) with the week it was
made; once the regular season is over the real field is known, so each logged
playoff chance is scored against who got in -- Brier score per checkpoint
week, and calibration by bucket -- expected wins against actual, and the
expected points for the rest of the regular season against the points each
team really scored (``strength_slope``). It is
our own numbers about league mates' teams, kept in ``results/<profile>`` like
the projection log.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..league.model import LeagueSnapshot
from .season import SeasonOdds

log = logging.getLogger(__name__)

LOG = "odds_log.parquet"
#: Each team's simulated score for every week ahead, and each game's win chance
#: on the same simulation, as logged each run: the season odds above can only
#: be graded when the season ends, these are gradable a week later.
WEEKS_LOG = "team_weeks_log.parquet"
GAMES_LOG = "game_odds_log.parquet"


def record(cfg: Config, snapshot: LeagueSnapshot, odds: SeasonOdds, now: datetime | None = None) -> int:
    """Append this run's odds; the last run of a (season, week) replaces earlier ones."""
    now = (now or datetime.now(UTC)).isoformat()
    points = {str(t.team_id): float(t.points_for) for t in snapshot.teams}
    rows = [{"season": int(snapshot.season), "week": int(snapshot.week), "team_id": str(tid), "logged_at": now,
             "p_playoffs": float(odds.p_playoffs[i]), "p_title": float(odds.p_title[i]),
             "exp_wins": float(odds.exp_wins[i]), "mine": odds.mine == i,
             "exp_points": float(odds.exp_points[i]) if odds.exp_points is not None else np.nan,
             "points_for": float(points.get(str(tid), np.nan))}
            for i, tid in enumerate(odds.team_ids)]
    path = cfg.results_dir / LOG
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    out = out.sort_values("logged_at").drop_duplicates(["season", "week", "team_id"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return len(rows)


def record_weeks(cfg: Config, snapshot: LeagueSnapshot, model, now: datetime | None = None) -> tuple[int, int]:
    """Append what the season simulation says about each week ahead: every
    team's mean score and spread (``team_weeks_log``) and every regular-season
    game's win chance for the team listed first (``game_odds_log``), read off
    the same simulated scores the title odds come from. Every run is kept
    (a week is graded on the last forecast made before its first kickoff,
    :func:`grade_weeks`, and only the runs themselves know which that was);
    weeks already past are not logged. Returns (team-week rows, game rows)."""
    stamp = (now or datetime.now(UTC)).isoformat()
    scores = model._scores(None)                           # teams x sims x weeks, as the odds read them
    teams = [str(t.team_id) for t in model.teams]
    season, week = int(snapshot.season), int(snapshot.week)
    t_rows, g_rows = [], []
    for k, w in enumerate(model.weeks):
        if w < week:
            continue                  # a week already begun has nothing left to forecast
        for i, tid in enumerate(teams):
            col = scores[i, :, k]
            t_rows.append({"season": season, "week": week, "for_week": int(w), "team_id": tid, "logged_at": stamp,
                           "mean": float(col.mean()), "sd": float(col.std()), "mine": bool(model.teams[i].is_mine)})
    for w, a, b in model.schedule:
        if a not in model.team_index or b not in model.team_index or w not in model.weeks or w < week:
            continue
        k = model.weeks.index(w)
        sa, sb = scores[model.team_index[a], :, k], scores[model.team_index[b], :, k]
        g_rows.append({"season": season, "week": week, "for_week": int(w), "a": a, "b": b, "logged_at": stamp,
                       "p_a_wins": float((sa > sb).mean() + 0.5 * (sa == sb).mean()),
                       "mean_margin": float((sa - sb).mean()), "sd_margin": float((sa - sb).std())})
    for name, rows, key in ((WEEKS_LOG, t_rows, ["season", "week", "for_week", "team_id"]),
                            (GAMES_LOG, g_rows, ["season", "week", "for_week", "a", "b"])):
        path = cfg.results_dir / name
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
        out = out.sort_values("logged_at").drop_duplicates([*key, "logged_at"], keep="last")
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
    return len(t_rows), len(g_rows)


def final_field(snapshot: LeagueSnapshot) -> tuple[set[str], dict[str, float]] | None:
    """Who made the playoffs and each team's wins, once the regular season is
    over (the standings count every regular-season game); else None."""
    rules = snapshot.rules or {}
    reg, n = int(rules.get("regular_season_weeks") or 0), int(rules.get("playoff_teams") or 0)
    if not reg or not n or int(snapshot.week) <= reg:
        return None
    per_week = 2 if rules.get("median_game") else 1
    if any((t.wins + t.losses + t.ties) < reg * per_week for t in snapshot.teams):
        return None                      # the platform has not counted the last week yet
    ranked = sorted(snapshot.teams, key=lambda t: (t.wins + 0.5 * t.ties, t.points_for), reverse=True)
    return {t.team_id for t in ranked[:n]}, {t.team_id: t.wins + 0.5 * t.ties for t in snapshot.teams}


def strength_slope(g: pd.DataFrame, final_points: dict[str, float]) -> float | None:
    """How far the teams' real points for the rest of the regular season
    followed the expected points logged that week, both against the league
    average: 1 when the model separated teams as much as the results did
    (the replayed leagues' measure, DECISIONS.md "Grading the odds on
    replayed leagues"). None without the columns or the spread."""
    if "exp_points" not in g or "points_for" not in g:
        return None
    g = g.dropna(subset=["exp_points", "points_for"])
    if len(g) < 4:
        return None
    real = g["team_id"].map(final_points) - g["points_for"]
    x, y = g["exp_points"] - g["exp_points"].mean(), real - real.mean()
    return float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else None


def grade(cfg: Config, snapshot: LeagueSnapshot) -> dict | None:
    """Score the logged playoff chances and expected wins of this season
    against the final standings; None until the regular season is over."""
    path = cfg.results_dir / LOG
    final = final_field(snapshot)
    if final is None or not path.exists():
        return None
    made, wins = final
    log_ = pd.read_parquet(path)
    log_ = log_[log_["season"] == int(snapshot.season)]
    if log_.empty:
        return None
    final_points = {str(t.team_id): float(t.points_for) for t in snapshot.teams}
    log_ = log_.assign(made=log_["team_id"].isin(made).astype(float), wins=log_["team_id"].map(wins))
    by_week = []
    for wk, g in log_.groupby("week"):
        row = {"week": int(wk), "brier": float(np.mean((g["p_playoffs"] - g["made"]) ** 2)),
               "wins_mae": float((g["exp_wins"] - g["wins"]).abs().mean())}
        row["strength_slope"] = strength_slope(g, final_points)
        by_week.append(row)
    base = float(np.mean(log_["made"]) * (1 - np.mean(log_["made"])))      # always say the league rate
    buckets = []
    edges = [0, .1, .25, .5, .75, .9, 1.0001]
    for lo, hi in zip(edges, edges[1:], strict=False):
        g = log_[(log_["p_playoffs"] >= lo) & (log_["p_playoffs"] < hi)]
        if len(g):
            buckets.append({"range": f"{lo:.0%}-{min(hi, 1):.0%}", "n": int(len(g)),
                            "said": float(g["p_playoffs"].mean()), "happened": float(g["made"].mean())})
    return {"season": int(snapshot.season), "by_week": by_week, "buckets": buckets, "brier_league_rate": base}


def week_scores(points_by_snapshot: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    """Each team's score for each week, from the season points-for the
    snapshots carry: the snapshot taken in week w+1 counts week w, so week w's
    score is its points-for less the previous snapshot's. Only weeks with both
    snapshots are returned."""
    return {w: {t: pts - points_by_snapshot[w].get(t, np.nan) for t, pts in points_by_snapshot[w + 1].items()
                if t in points_by_snapshot[w]}
            for w in sorted(points_by_snapshot) if w + 1 in points_by_snapshot}


def first_kickoffs(cfg: Config, season: int) -> dict[int, datetime]:
    """Each NFL week's first kickoff (UTC), from the schedule (kickoff times are Eastern)."""
    from zoneinfo import ZoneInfo

    path = cfg.raw_dir / "schedules.parquet"
    if not path.exists():
        return {}
    sch = pd.read_parquet(path)
    sch = sch[(sch["season"] == season) & sch["gameday"].notna()]
    et = ZoneInfo("America/New_York")
    out: dict[int, datetime] = {}
    for w, g in sch.groupby("week"):
        times = []
        for day, clock in zip(g["gameday"], g.get("gametime", pd.Series([None] * len(g), index=g.index))):
            try:
                hh, mm = (str(clock) if clock else "13:00").split(":")[:2]
                d = datetime.fromisoformat(str(day)[:10]).replace(hour=int(hh), minute=int(mm), tzinfo=et)
                times.append(d.astimezone(UTC))
            except (ValueError, TypeError):
                continue
        if times:
            out[int(w)] = min(times)
    return out


def before_kickoff(df: pd.DataFrame, kickoffs: dict[int, datetime], key: list[str]) -> pd.DataFrame:
    """The last forecast of each ``key`` made before its week's first kickoff;
    a week with no known kickoff keeps its last forecast."""
    if df.empty:
        return df
    stamp = pd.to_datetime(df["logged_at"], utc=True)
    cut = df["for_week"].map(lambda w: kickoffs.get(int(w)))
    ok = pd.Series([c is None or t < pd.Timestamp(c) for t, c in zip(stamp, cut)], index=df.index)
    return df[ok].assign(_t=stamp[ok]).sort_values("_t").drop_duplicates(key, keep="last").drop(columns="_t")


def grade_weeks(cfg: Config, scores: dict[int, dict[str, float]], season: int,
                kickoffs: dict[int, datetime] | None = None) -> dict | None:
    """Score the weekly forecasts against real scores: for every forecast of a
    week already played, how far the team's mean was off (and whether the
    spread was right: z = miss / sd should have a standard deviation of 1),
    and for every game, the log loss and Brier score of the win chance given,
    with calibration buckets. Grouped by how many weeks ahead the forecast was
    (0 = made in that week). With ``kickoffs`` (:func:`first_kickoffs`) only the
    last forecast made before each week's first kickoff counts, so a forecast
    made after games were played is never graded. None when nothing logged has
    been played."""
    wpath, gpath = cfg.results_dir / WEEKS_LOG, cfg.results_dir / GAMES_LOG
    out: dict = {"season": season, "teams": [], "games": [], "buckets": []}
    if wpath.exists():
        w = pd.read_parquet(wpath)
        w = w[w["season"] == season].copy()
        w = before_kickoff(w, kickoffs or {}, ["for_week", "team_id"])
        w["real"] = [scores.get(int(f), {}).get(str(t), np.nan) for f, t in zip(w["for_week"], w["team_id"])]
        w = w.dropna(subset=["real"])
        w["ahead"] = w["for_week"] - w["week"]
        for ahead, g in w.groupby("ahead"):
            miss = g["mean"] - g["real"]
            z = miss / g["sd"].where(g["sd"] > 0)
            out["teams"].append({"ahead": int(ahead), "n": int(len(g)), "bias": float(miss.mean()),
                                 "mae": float(miss.abs().mean()), "z_sd": float(z.std()) if len(g) > 2 else None})
    if gpath.exists():
        g = pd.read_parquet(gpath)
        g = g[g["season"] == season].copy()
        g = before_kickoff(g, kickoffs or {}, ["for_week", "a", "b"])
        sa = [scores.get(int(f), {}).get(str(a), np.nan) for f, a in zip(g["for_week"], g["a"])]
        sb = [scores.get(int(f), {}).get(str(b), np.nan) for f, b in zip(g["for_week"], g["b"])]
        g["won"] = np.where(np.isnan(sa) | np.isnan(sb), np.nan, np.where(np.array(sa) > np.array(sb), 1.0, 0.0))
        g = g.dropna(subset=["won"])
        g["ahead"] = g["for_week"] - g["week"]
        p = g["p_a_wins"].clip(1e-4, 1 - 1e-4)
        g["ll"] = -(g["won"] * np.log(p) + (1 - g["won"]) * np.log(1 - p))
        g["brier"] = (g["p_a_wins"] - g["won"]) ** 2
        # The chance for the team that was the favourite, so 50% is the floor.
        fav = np.where(g["p_a_wins"] >= 0.5, g["p_a_wins"], 1 - g["p_a_wins"])
        hit = np.where(g["p_a_wins"] >= 0.5, g["won"], 1 - g["won"])
        g = g.assign(fav=fav, hit=hit)
        for ahead, h in g.groupby("ahead"):
            out["games"].append({"ahead": int(ahead), "n": int(len(h)), "log_loss": float(h["ll"].mean()),
                                 "brier": float(h["brier"].mean()), "favourite_said": float(h["fav"].mean()),
                                 "favourite_won": float(h["hit"].mean())})
        for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
            h = g[(g["fav"] >= lo) & (g["fav"] < hi)]
            if len(h):
                out["buckets"].append({"range": f"{lo:.0%}-{min(hi, 1):.0%}", "n": int(len(h)),
                                       "said": float(h["fav"].mean()), "happened": float(h["hit"].mean())})
    return out if out["teams"] or out["games"] else None
