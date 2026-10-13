"""A clearly worse quarterback this week, for his receivers.

This week's implied team total already falls when a backup starts, and the
projection scales with it. Replayed on 2022-2025 (``scripts/fit_qb_change.py``)
that was not enough for receivers: with a starter clearly worse than the
quarterback of their recent games -- passing EPA per attempt, shrunk toward
the league by ``shrink`` attempts, ``worse_sd`` league spreads lower -- they
scored about 0.75 a game under projection. A cut of ``cut`` times the share
of their recent games thrown by someone else fixed it in all four held-out
seasons (5.37 -> 5.24 RMSE on those weeks). Tight ends and backs showed no
such effect. DECISIONS.md, "A new quarterback for the receivers".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..league.model import OUT_STATUSES, PlayerRow
from ..teams import normalize_team

log = logging.getLogger(__name__)


def conf(cfg: Config) -> dict:
    return (cfg.raw.get("roster") or {}).get("qb_change") or {}


def passing(cfg: Config, season: int, years: int = 4) -> pd.DataFrame:
    """Every regular-season quarterback game with a pass, from the cached
    nflverse player stats: season, week, team, player_id, name, attempts, EPA."""
    frames = []
    for s in range(season - years, season + 1):
        path = cfg.raw_dir / f"player_stats_{s}.parquet"
        if not path.exists():
            continue
        st = pd.read_parquet(path)
        if not {"attempts", "passing_epa"} <= set(st.columns):
            continue
        st = st[(st["position"] == "QB") & (st.get("season_type", "REG") == "REG") & (st["attempts"] > 0)]
        frames.append(st[["season", "week", "team", "player_id", "player_display_name", "attempts", "passing_epa"]])
    if not frames:
        return pd.DataFrame()
    q = pd.concat(frames, ignore_index=True)
    return q.assign(team=q["team"].map(normalize_team))


def ratings(q: pd.DataFrame, shrink: float) -> tuple[dict[str, float], float, float]:
    """Quarterback -> shrunk EPA per attempt; the league mean and spread."""
    g = q.groupby("player_id")[["attempts", "passing_epa"]].sum()
    mu = float(g["passing_epa"].sum() / g["attempts"].sum())
    rate = (g["passing_epa"] + shrink * mu) / (g["attempts"] + shrink)
    spread = float(rate[g["attempts"] >= 150].std())
    return rate.to_dict(), mu, spread


def this_weeks_starters(players: list[PlayerRow], ids: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Team -> (nflverse id, name) of the quarterback in line to start: the
    first on the depth chart who is not out or on a bye."""
    by_team: dict[str, list[PlayerRow]] = {}
    for p in players:
        if p.position == "QB" and p.team and ids.get(p.player_id):
            by_team.setdefault(normalize_team(p.team), []).append(p)
    out = {}
    for team, qbs in by_team.items():
        ok = [p for p in qbs if p.status not in OUT_STATUSES and not p.on_bye]
        if not ok:
            continue
        first = min(ok, key=lambda p: (p.depth_order is None, p.depth_order or 0, -float(p.ros_value or 0.0)))
        out[team] = (ids[first.player_id], first.name)
    return out


def factors(players: list[PlayerRow], ids: dict[str, str], history: pd.DataFrame, cfg: Config,
            season: int, week: int) -> dict[str, tuple[float, str]]:
    """Snapshot player id -> (factor on this week's projection, card note), for
    the positions ``roster.qb_change`` covers, where this week's starter is
    clearly worse than the quarterback of the player's recent games."""
    c = conf(cfg)
    positions = c.get("positions") or []
    if not positions or history is None or history.empty:
        return {}
    q = passing(cfg, season)
    if q.empty:
        return {}
    q = q[(q["season"] < season) | ((q["season"] == season) & (q["week"] < week))]
    rate, mu, spread = ratings(q, float(c.get("shrink", 200)))
    if not np.isfinite(spread) or spread <= 0:
        return {}
    start = (q.sort_values("attempts").groupby(["season", "week", "team"]).tail(1)
             .set_index(["season", "week", "team"])["player_id"])
    starters = this_weeks_starters(players, ids)
    worse, cut, recent = float(c.get("worse_sd", 0.5)), float(c.get("cut", 0.15)), int(c.get("recent", 6))
    h = history[((history["season"] == season) & (history["week"] < week)) | (history["season"] == season - 1)]
    h = h.assign(_team=h["team"].map(normalize_team))
    out = {}
    for p in players:
        nid = ids.get(p.player_id)
        if p.position not in positions or not nid or not p.team:
            continue
        team = normalize_team(p.team)
        if team not in starters:
            continue
        qb, qb_name = starters[team]
        mine = h[(h["player_id"] == nid) & (h["_team"] == team)].sort_values(["season", "week"]).tail(recent)
        qbs = [start.get((s, w, team)) for s, w in zip(mine["season"], mine["week"])]
        qbs = [x for x in qbs if x is not None]
        if not qbs:
            continue
        other = 1.0 - sum(x == qb for x in qbs) / len(qbs)
        gap = (rate.get(qb, mu) - float(np.mean([rate.get(x, mu) for x in qbs]))) / spread
        if other <= 0 or gap >= -worse:
            continue
        f = 1.0 - cut * other
        out[p.player_id] = (f, f"{qb_name} is in line to start at quarterback, clearly less efficient than "
                               f"who threw to him lately ({f - 1:+.0%}: receivers with a worse new starter "
                               "fell short of their projection, 2022-2025)")
    return out
