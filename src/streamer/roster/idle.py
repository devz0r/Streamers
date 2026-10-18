"""Games a skill player was on the field and recorded nothing.

nflverse's weekly player stats have no row for a back, receiver or tight end
who played offensive snaps and carried, caught and scored nothing, so every
trailing average the projection uses skipped those games and ran high for
backups and rotational players: in weeks 3-5 of 2026 players projected under
3 points by the model alone were projected 2.65 and scored 1.79.

:func:`with_idle_games` adds a zero row for each such game (a player with an
offensive snap in a game the stats lack), from the snap counts. Replayed on
2023-2025 through the production projection (``scripts/check_active_zero.py``):
mean error 4.091 -> 4.013 on every snap-active player, better in all three
seasons, rank correlation 0.717 -> 0.722; the low end (projected under 3)
went from 1.93 to 1.55 against 1.61 scored. DECISIONS.md, "Games on the
field with no stat line". Switched by ``roster.idle_games``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)
SKILL = ("QB", "RB", "WR", "TE")
ZERO_COLUMNS = ("fantasy_points_ppr", "total_fantasy_points_exp", "xtd", "targets", "carries", "target_share")


def enabled(cfg: Config) -> bool:
    return bool((cfg.raw.get("roster") or {}).get("idle_games", False))


def snap_rows(cfg: Config, seasons, allow_network: bool = True) -> pd.DataFrame:
    """Offensive snap counts for QB/RB/WR/TE (a fullback is a back) in regular
    season games, one row per player-game with at least one snap."""
    from ..data.nflverse import load_snap_counts

    if allow_network:
        sn = load_snap_counts(list(seasons), cfg)
    else:
        files = [cfg.raw_dir / f"snap_counts_{int(s)}.parquet" for s in seasons]
        frames = [pd.read_parquet(f) for f in files if f.exists()]
        sn = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if sn is None or sn.empty:
        return pd.DataFrame()
    if "game_type" in sn.columns:
        sn = sn[sn["game_type"] == "REG"]
    sn = sn[sn["position"].isin([*SKILL, "FB"]) & (sn["offense_snaps"] > 0)].copy()
    sn["position"] = sn["position"].replace({"FB": "RB"})
    return sn


def active_rows(history: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """The snap rows with an nflverse id (matched on name and position
    through the stat history): player_id, season, week, team, position."""
    from ..teams import normalize_team_series
    from .players import normalize_name

    if snaps.empty:
        return pd.DataFrame(columns=["player_id", "season", "week", "team", "position"])
    ids = (history.assign(key=history["player_display_name"].map(normalize_name))
           .sort_values(["season", "week"]).groupby(["key", "position"])["player_id"].last())
    sn = snaps.assign(key=snaps["player"].map(normalize_name))
    sn["player_id"] = [ids.get((k, p)) for k, p in zip(sn["key"], sn["position"])]
    sn = sn.dropna(subset=["player_id"])
    sn["team"] = normalize_team_series(sn["team"])
    return sn[["player_id", "season", "week", "team", "position"]].astype({"season": int, "week": int})


def with_idle_games(history: pd.DataFrame, cfg: Config, season: int | None = None, week: int | None = None,
                    allow_network: bool = True) -> pd.DataFrame:
    """``history`` plus a zero row for each game a skill player had an
    offensive snap but no stat line. With ``season``/``week``, only games
    before them (a replay must not see the week it projects). The history is
    returned unchanged when the switch is off or there are no snap counts."""
    if not enabled(cfg) or history.empty:
        return history
    try:
        sn = snap_rows(cfg, sorted(history["season"].unique()), allow_network)
        act = active_rows(history, sn)
    except Exception as exc:  # noqa: BLE001 - a refinement; never block a projection
        log.warning("snap counts unavailable, idle games not counted: %s", exc)
        return history
    if act.empty:
        return history
    if season is not None and week is not None:
        act = act[(act["season"] < season) | ((act["season"] == season) & (act["week"] < week))]
    have = history[["player_id", "season", "week"]].assign(_h=1)
    new = act.drop_duplicates(["player_id", "season", "week"]).merge(
        have, on=["player_id", "season", "week"], how="left")
    new = new[new["_h"].isna()].drop(columns="_h")
    if new.empty:
        return history
    name = history.groupby("player_id")["player_display_name"].last()
    new["player_display_name"] = new["player_id"].map(name)
    for c in history.columns:
        if c not in new.columns:
            new[c] = 0.0 if c in ZERO_COLUMNS else np.nan
    out = pd.concat([history, new[history.columns]], ignore_index=True)
    log.info("counted %d games on the field with no stat line", len(new))
    return out.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
