"""Per team-game raw counts from play-by-play.

Everything downstream (rolling priors, kicker features, DST features) is built
from this one table, so it holds *counts and denominators*, never rates -- rates
are formed later, after shrinkage, to avoid averaging small samples.

Two rows exist per game: one for each team's **offense** (``posteam``) and the
mirror image describing what that team's **defense** faced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCRIMMAGE = ("pass", "run", "qb_kneel", "qb_spike")


def drive_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per offensive drive with red-zone and outcome flags."""
    plays = pbp[pbp["posteam"].notna() & pbp["fixed_drive"].notna()].copy()
    if plays.empty:
        return pd.DataFrame(
            columns=["season", "week", "game_id", "posteam", "fixed_drive",
                     "result", "reached_rz", "is_td", "is_fg", "is_missed_fg", "plays"]
        )
    plays["yardline_100"] = pd.to_numeric(plays["yardline_100"], errors="coerce")
    keys = ["season", "week", "game_id", "posteam", "fixed_drive"]
    grouped = plays.groupby(keys, dropna=False)
    out = pd.DataFrame(
        {
            "result": grouped["fixed_drive_result"].first(),
            "min_yardline": grouped["yardline_100"].min(),
            "plays": grouped["play_type"].apply(lambda s: s.isin(SCRIMMAGE).sum()),
        }
    ).reset_index()
    out["reached_rz"] = (out["min_yardline"] <= 20).astype(int)
    result = out["result"].fillna("")
    out["is_td"] = (result == "Touchdown").astype(int)
    out["is_fg"] = (result == "Field goal").astype(int)
    out["is_missed_fg"] = (result == "Missed field goal").astype(int)
    return out


def offense_game_counts(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, game_id, team) offensive counts and denominators."""
    if pbp.empty:
        return pd.DataFrame()
    df = pbp[pbp["posteam"].notna()].copy()
    numeric = ["field_goal_attempt", "extra_point_attempt", "sack", "qb_hit",
               "interception", "fumble_lost", "qb_dropback", "pass_attempt",
               "rush_attempt", "epa", "wp"]
    for col in numeric:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_scrimmage"] = df["play_type"].isin(SCRIMMAGE).astype(int)
    df["fg_made_play"] = (df["field_goal_result"] == "made").astype(int)
    # "Neutral" pace: plays with the game still in the balance, which strips out
    # garbage-time volume that inflates raw plays-per-game.
    wp = df["wp"]
    df["neutral_play"] = (df["is_scrimmage"].eq(1) & wp.between(0.2, 0.8)).astype(int)

    keys = ["season", "week", "game_id", "posteam"]
    agg = df.groupby(keys, dropna=False).agg(
        plays=("is_scrimmage", "sum"),
        neutral_plays=("neutral_play", "sum"),
        dropbacks=("qb_dropback", "sum"),
        pass_att=("pass_attempt", "sum"),
        rush_att=("rush_attempt", "sum"),
        fga=("field_goal_attempt", "sum"),
        fg_made=("fg_made_play", "sum"),
        pat_att=("extra_point_attempt", "sum"),
        sacks_allowed=("sack", "sum"),
        qb_hits_allowed=("qb_hit", "sum"),
        ints_thrown=("interception", "sum"),
        fumbles_lost=("fumble_lost", "sum"),
        epa_per_play=("epa", "mean"),
    ).reset_index()

    drives = drive_table(pbp)
    if not drives.empty:
        dagg = drives.groupby(keys, dropna=False).agg(
            drives=("fixed_drive", "nunique"),
            rz_trips=("reached_rz", "sum"),
            drive_tds=("is_td", "sum"),
            drive_fgs=("is_fg", "sum"),
            drive_missed_fgs=("is_missed_fg", "sum"),
        ).reset_index()
        rz = drives[drives["reached_rz"] == 1]
        if not rz.empty:
            rzagg = rz.groupby(keys, dropna=False).agg(
                rz_tds=("is_td", "sum"),
                rz_fgs=("is_fg", "sum"),
            ).reset_index()
            dagg = dagg.merge(rzagg, on=keys, how="left")
        agg = agg.merge(dagg, on=keys, how="left")

    for col in ("drives", "rz_trips", "drive_tds", "drive_fgs", "drive_missed_fgs",
                "rz_tds", "rz_fgs"):
        if col not in agg.columns:
            agg[col] = 0.0
        agg[col] = agg[col].fillna(0.0)

    # Long field-goal attempts: coach aggressiveness plus kicker leg.
    fg = pbp[(pbp["field_goal_attempt"] == 1) & pbp["posteam"].notna()].copy()
    if not fg.empty:
        fg["kick_distance"] = pd.to_numeric(fg["kick_distance"], errors="coerce")
        fg["fga_50"] = (fg["kick_distance"] >= 50).astype(int)
        fg["fga_40"] = fg["kick_distance"].between(40, 49).astype(int)
        fgagg = fg.groupby(keys, dropna=False).agg(
            fga_50_plus=("fga_50", "sum"),
            fga_40_49=("fga_40", "sum"),
        ).reset_index()
        agg = agg.merge(fgagg, on=keys, how="left")
    for col in ("fga_50_plus", "fga_40_49"):
        if col not in agg.columns:
            agg[col] = 0.0
        agg[col] = agg[col].fillna(0.0)

    return agg.rename(columns={"posteam": "team"}).fillna({"epa_per_play": 0.0})


def defense_game_counts(offense: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Mirror the offensive counts onto the defense that faced them.

    ``def_*`` columns describe what the defense *did* (sacks, INTs) and what it
    *allowed* (drives, red-zone trips), which is exactly the split the DST
    feature builder needs.
    """
    if offense.empty:
        return pd.DataFrame()
    link = games[["season", "week", "game_id", "team", "opponent"]].copy()
    faced = offense.rename(columns={"team": "opponent"})
    rename = {c: f"faced_{c}" for c in faced.columns
              if c not in ("season", "week", "game_id", "opponent")}
    faced = faced.rename(columns=rename)
    return link.merge(faced, on=["season", "week", "game_id", "opponent"], how="inner")


def future_game_rows(future_games: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """Zero-count rows for games that have not been played.

    They carry ``is_future = 1`` so the rolling-prior recursion hands them the
    accumulated state without folding their (nonexistent) counts back in.
    """
    if future_games is None or future_games.empty:
        return pd.DataFrame()
    rows = future_games[["season", "week", "game_id", "team", "opponent"]].copy()
    for col in template.columns:
        if col in rows.columns:
            continue
        rows[col] = 0.0 if pd.api.types.is_numeric_dtype(template[col]) else None
    rows["is_future"] = 1
    return rows[[*template.columns, "is_future"]] if "is_future" not in template.columns else rows


def build_team_week(
    pbp: pd.DataFrame, games: pd.DataFrame, future_games: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Offensive counts joined to the counts each team's defense faced.

    ``future_games`` (optional) appends placeholder rows for an upcoming slate
    so the same prior machinery produces features for unplayed games.
    """
    off = offense_game_counts(pbp)
    if off.empty:
        return pd.DataFrame()
    dfn = defense_game_counts(off, games)
    merged = off.merge(
        dfn, on=["season", "week", "game_id", "team"], how="inner"
    )
    # `opponent` already arrives with the defensive counts, so it is dropped
    # here to avoid a duplicate-column merge.
    merged["is_future"] = 0
    if future_games is not None and not future_games.empty:
        merged = pd.concat(
            [merged, future_game_rows(future_games, merged)], ignore_index=True
        )
    ctx = games[[
        "season", "week", "game_id", "team", "is_home",
        "team_score", "opp_score", "team_spread", "total_line",
        "team_implied_total", "opp_implied_total", "is_dome", "roof",
        "temp", "wind", "gameday",
    ]]
    out = merged.merge(ctx, on=["season", "week", "game_id", "team"], how="left")
    return out.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
