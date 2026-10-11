"""A record of our season odds, to grade them when the season is over.

Nobody has checked whether "41% to make the playoffs" means 41%. Every run
appends each team's odds (playoffs, title, expected wins) with the week it was
made; once the regular season is over the real field is known, so each logged
playoff chance is scored against who got in -- Brier score per checkpoint
week, and calibration by bucket -- and expected wins against actual. It is
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


def record(cfg: Config, snapshot: LeagueSnapshot, odds: SeasonOdds, now: datetime | None = None) -> int:
    """Append this run's odds; the last run of a (season, week) replaces earlier ones."""
    now = (now or datetime.now(UTC)).isoformat()
    rows = [{"season": int(snapshot.season), "week": int(snapshot.week), "team_id": str(tid), "logged_at": now,
             "p_playoffs": float(odds.p_playoffs[i]), "p_title": float(odds.p_title[i]),
             "exp_wins": float(odds.exp_wins[i]), "mine": odds.mine == i}
            for i, tid in enumerate(odds.team_ids)]
    path = cfg.results_dir / LOG
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    out = out.sort_values("logged_at").drop_duplicates(["season", "week", "team_id"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return len(rows)


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
    log_ = log_.assign(made=log_["team_id"].isin(made).astype(float), wins=log_["team_id"].map(wins))
    by_week = []
    for wk, g in log_.groupby("week"):
        by_week.append({"week": int(wk), "brier": float(np.mean((g["p_playoffs"] - g["made"]) ** 2)),
                        "wins_mae": float((g["exp_wins"] - g["wins"]).abs().mean())})
    base = float(np.mean(log_["made"]) * (1 - np.mean(log_["made"])))      # always say the league rate
    buckets = []
    edges = [0, .1, .25, .5, .75, .9, 1.0001]
    for lo, hi in zip(edges, edges[1:], strict=False):
        g = log_[(log_["p_playoffs"] >= lo) & (log_["p_playoffs"] < hi)]
        if len(g):
            buckets.append({"range": f"{lo:.0%}-{min(hi, 1):.0%}", "n": int(len(g)),
                            "said": float(g["p_playoffs"].mean()), "happened": float(g["made"].mean())})
    return {"season": int(snapshot.season), "by_week": by_week, "buckets": buckets, "brier_league_rate": base}
