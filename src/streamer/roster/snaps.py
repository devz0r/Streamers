"""Snap share as an early sign of a role change.

Expected points move once the targets and carries do; who is on the field
moves first. Replayed on 2022-2025 (``scripts/fit_snaps.py``), through the
production projection, two signals from a player's games for his team this
season (4+ of them), offensive snap share from Pro Football Reference via
nflverse:

* ``delta`` -- snap share over his last 2 games less over his earlier ones;
* ``gap``   -- his snap share's percentile among players at his position
  less his share of his group's expected points' percentile: on the field
  more than he is used.

This week's projection is multiplied by ``exp(trend x delta + gap x gap)``
(``roster.snap_trend``, per position). Backs: better in all four held-out
seasons (6.355 -> 6.320 RMSE); tight ends on the trend, 3 of 4; receivers
showed nothing. DECISIONS.md, "Snap share as an early sign".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)
POSITIONS = ("RB", "WR", "TE")


def load(cfg: Config, season: int, allow_network: bool) -> pd.DataFrame:
    """This season's offensive snap shares (season, week, team, name, pct)."""
    path = cfg.raw_dir / f"snap_counts_{season}.parquet"
    if allow_network:
        from ..data.nflverse import load_snap_counts

        frame = load_snap_counts([season], cfg)
    elif path.exists():
        frame = pd.read_parquet(path)
    else:
        return pd.DataFrame()
    return tidy(frame)


def tidy(frame: pd.DataFrame) -> pd.DataFrame:
    from ..teams import normalize_team
    from .players import normalize_name

    if frame is None or frame.empty:
        return pd.DataFrame()
    if "game_type" in frame.columns:
        frame = frame[frame["game_type"] == "REG"]
    frame = frame[frame["position"].isin(POSITIONS)]
    return pd.DataFrame({"season": frame["season"].astype(int), "week": frame["week"].astype(int),
                         "_team": frame["team"].map(normalize_team), "_name": frame["player"].map(normalize_name),
                         "offense_pct": frame["offense_pct"].astype(float)})


def join(history: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """History with each game's snap share and share of the position group's
    expected points (``offense_pct``, ``exp_share``)."""
    from ..teams import normalize_team
    from .players import normalize_name

    h = history.assign(_team=history["team"].map(normalize_team),
                       _name=history["player_display_name"].map(normalize_name))
    if snaps is not None and not snaps.empty:
        h = h.merge(snaps, on=["season", "week", "_team", "_name"], how="left")
    else:
        h["offense_pct"] = np.nan
    exp = h["total_fantasy_points_exp"].fillna(h["fantasy_points_ppr"])
    grp = exp.groupby([h["season"], h["week"], h["_team"], h["position"]]).transform("sum")
    return h.assign(exp_share=exp / grp.where(grp > 0))


def signals(joined: pd.DataFrame, season: int, week: int, teams: dict[str, str] | None = None) -> pd.DataFrame:
    """Per nflverse id: ``delta``, ``gap``, ``s2`` and ``s_prior`` from his
    games for his team (``teams``: id -> current team; else his last) this
    season before ``week`` -- 4 or more of them."""
    from ..teams import normalize_team

    this = joined[(joined["season"] == season) & (joined["week"] < week) & joined["offense_pct"].notna()
                  & joined["position"].isin(POSITIONS)].sort_values("week")
    if this.empty:
        return pd.DataFrame(columns=["delta", "gap", "s2", "s_prior", "position"])
    last_team = this.groupby("player_id")["_team"].last()
    now = {pid: normalize_team(t) for pid, t in (teams or {}).items()}
    this = this[this["_team"].to_numpy() == this["player_id"].map(lambda p: now.get(p, last_team.get(p))).to_numpy()]
    rows = {}
    for pid, g in this.groupby("player_id"):
        if len(g) < 4:
            continue
        s2, sp = float(g["offense_pct"].tail(2).mean()), float(g["offense_pct"].iloc[:-2].mean())
        rows[pid] = {"position": g["position"].iloc[-1], "s2": s2, "s_prior": sp, "delta": s2 - sp,
                     "e2": float(g["exp_share"].tail(2).mean())}
    out = pd.DataFrame.from_dict(rows, orient="index")
    if out.empty:
        return out
    out["gap"] = out.groupby("position")["s2"].rank(pct=True) - out.groupby("position")["e2"].rank(pct=True)
    return out


def factors(sig: pd.DataFrame, cfg: Config) -> dict[str, tuple[float, str]]:
    """Per nflverse id: (factor on this week's projection, card note)."""
    rule = (cfg.raw.get("roster") or {}).get("snap_trend") or {}
    out = {}
    for pid, r in sig.iterrows():
        w = rule.get(r["position"]) or {}
        a, b = float(w.get("trend", 0.0)), float(w.get("gap", 0.0))
        if not a and not b:
            continue
        trend, gap = a * r["delta"], b * r["gap"]
        f = float(np.exp(trend + gap))
        note = ""
        if abs(f - 1.0) >= 0.04:
            if abs(trend) >= abs(gap):
                why = (f"on the field {'more' if trend > 0 else 'less'} lately: {r['s_prior']:.0%} of snaps "
                       f"before, {r['s2']:.0%} his last 2 games")
            else:
                why = (f"{r['s2']:.0%} of snaps his last 2 games and {r['e2']:.0%} of his group's work: "
                       f"{'on the field more than he is used' if gap > 0 else 'used more than his snaps suggest'}")
            note = (f"{why} ({f - 1:+.0%}; {r['position']}s like this "
                    f"{'beat' if f > 1 else 'fell short of'} their projection, 2022-2025)")
        out[str(pid)] = (f, note)
    return out
