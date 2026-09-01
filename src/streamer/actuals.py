"""Derive realised weekly Kicker and D/ST fantasy points from play-by-play.

These are the labels every model is trained and scored against, so the
attribution rules are spelled out rather than inferred:

* On a **kickoff**, nflverse sets ``posteam`` to the *receiving* team, so a
  kick-return touchdown has ``td_team == posteam``. On a **punt**, ``posteam``
  is the *punting* team. Both are special-teams scores for the returning unit.
* A blocked field goal or PAT returned for a score is credited to the blocking
  (defensive) team.
* A half sack still belongs to one sack *play*; team sacks are counted as sack
  plays, not player credits, which is what every host does.
* A fourth-down stop is a *failed conversion attempt*. A punt or field goal on
  fourth down is neither, and does not count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, get_config
from .scoring import FG_BUCKETS, DstScoring, KickerScoring

KICK_PLAY_TYPES = ("punt", "kickoff", "field_goal", "extra_point")

#: Every non-ladder D/ST scoring component we count out of play-by-play.
DST_EVENT_COLUMNS: tuple[str, ...] = (
    "sacks", "interceptions", "fumble_recoveries", "safeties", "one_point_safeties",
    "defensive_tds", "return_tds", "blocked_kicks", "blocked_kick_tds",
    "extra_points_returned", "fourth_down_stops", "fumbles_lost",
)

#: Scrimmage plays that count toward total yards allowed. Penalty yardage is
#: excluded, which is what "total yards" means on every fantasy host.
YARDAGE_PLAY_TYPES = ("pass", "run")


# ---------------------------------------------------------------------------
# Kickers
# ---------------------------------------------------------------------------
def kicker_game_lines(pbp: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """One row per kicker-game with bucketed FG/PAT counts and fantasy points."""
    cfg = cfg or get_config()
    scoring = KickerScoring.from_config(cfg)

    kicks = pbp[
        (pbp["kicker_player_id"].notna())
        & ((pbp["field_goal_attempt"] == 1) | (pbp["extra_point_attempt"] == 1))
    ].copy()
    if kicks.empty:
        return _empty_kicker_frame()

    # `aborted_play` covers botched snaps/holds which are not charged to the
    # kicker's fantasy line on any major host.
    if "aborted_play" in kicks.columns:
        kicks = kicks[kicks["aborted_play"].fillna(0) != 1]

    kicks["team"] = kicks["posteam"]
    kicks["distance"] = pd.to_numeric(kicks["kick_distance"], errors="coerce")

    is_fg = kicks["field_goal_attempt"] == 1
    # A blocked field goal counts as a miss for the kicker.
    made = kicks["field_goal_result"].eq("made")

    # Distance is occasionally missing on blocked kicks; treat those as 0-39
    # (the modal blocked-kick bucket) rather than dropping the attempt.
    dist = kicks["distance"].fillna(35.0)
    conditions = [
        (dist >= low) & (dist <= high) if high is not None else (dist >= low)
        for _name, low, high in FG_BUCKETS
    ]
    bucket = pd.Series(
        np.select(conditions, [name for name, _l, _h in FG_BUCKETS], default=FG_BUCKETS[-1][0]),
        index=kicks.index,
    )

    for name, _low, _high in FG_BUCKETS:
        kicks[f"fg_made_{name}"] = (is_fg & made & bucket.eq(name)).astype(int)
        kicks[f"fg_missed_{name}"] = (is_fg & ~made & bucket.eq(name)).astype(int)

    is_xp = kicks["extra_point_attempt"] == 1
    kicks["pat_made"] = (is_xp & kicks["extra_point_result"].eq("good")).astype(int)
    kicks["pat_missed"] = (is_xp & ~kicks["extra_point_result"].eq("good")).astype(int)

    count_cols = (
        [f"fg_made_{n}" for n, _l, _h in FG_BUCKETS]
        + [f"fg_missed_{n}" for n, _l, _h in FG_BUCKETS]
        + ["pat_made", "pat_missed"]
    )
    grouped = (
        kicks.groupby(
            ["season", "week", "game_id", "team", "kicker_player_id", "kicker_player_name"],
            dropna=False,
        )[count_cols]
        .sum()
        .reset_index()
    )

    grouped["fg_made"] = grouped[[f"fg_made_{n}" for n, _l, _h in FG_BUCKETS]].sum(axis=1)
    grouped["fg_missed"] = grouped[[f"fg_missed_{n}" for n, _l, _h in FG_BUCKETS]].sum(axis=1)
    grouped["fg_att"] = grouped["fg_made"] + grouped["fg_missed"]

    points = np.zeros(len(grouped), dtype=float)
    for name, _low, _high in FG_BUCKETS:
        points += grouped[f"fg_made_{name}"] * scoring.made_value(name)
        points += grouped[f"fg_missed_{name}"] * scoring.missed_value(name)
    points += grouped["pat_made"] * scoring.pat_made
    points += grouped["pat_missed"] * scoring.pat_missed
    grouped["fantasy_points"] = points
    return grouped.rename(columns={"kicker_player_name": "player_name",
                                   "kicker_player_id": "player_id"})


def _empty_kicker_frame() -> pd.DataFrame:
    cols = (
        ["season", "week", "game_id", "team", "player_id", "player_name"]
        + [f"fg_made_{n}" for n, _l, _h in FG_BUCKETS]
        + [f"fg_missed_{n}" for n, _l, _h in FG_BUCKETS]
        + ["pat_made", "pat_missed", "fg_made", "fg_missed", "fg_att", "fantasy_points"]
    )
    return pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# D/ST
# ---------------------------------------------------------------------------
def dst_game_lines(
    pbp: pd.DataFrame, games: pd.DataFrame, cfg: Config | None = None
) -> pd.DataFrame:
    """One row per defense-game with every D/ST component and the profile total."""
    cfg = cfg or get_config()
    scoring = DstScoring.from_config(cfg)

    events = _dst_events(pbp)
    played = games[games["opp_score"].notna()].copy()
    if played.empty:
        return pd.DataFrame()

    out = played[
        ["season", "week", "game_id", "team", "opponent", "is_home",
         "team_score", "opp_score"]
    ].copy()
    out = out.merge(events, on=["season", "week", "game_id", "team"], how="left")

    # Yards allowed = yards the opponent's offence gained.
    yards = offense_yards(pbp).rename(columns={"posteam": "opponent", "yards": "yards_allowed"})
    out = out.merge(yards, on=["season", "week", "game_id", "opponent"], how="left")
    out["yards_allowed"] = out["yards_allowed"].fillna(0.0).round().clip(lower=0).astype(int)
    event_cols = [c for c in events.columns if c not in ("season", "week", "game_id", "team")]
    out[event_cols] = out[event_cols].fillna(0.0)

    out["points_allowed"] = out["opp_score"].astype(float)
    if scoring.points_allowed_excludes_opponent_dst_st:
        # Opt-in: strip points the opponent's own defense/ST put on the board
        # against our offense, which some hosts do not charge to our D/ST.
        opp_non_off = events.rename(columns={"team": "opponent"})[
            ["season", "week", "game_id", "opponent",
             "defensive_tds", "return_tds", "blocked_kick_tds",
             "safeties", "extra_points_returned"]
        ]
        out = out.merge(opp_non_off, on=["season", "week", "game_id", "opponent"],
                        how="left", suffixes=("", "_opp"))
        for c in ("defensive_tds_opp", "return_tds_opp", "blocked_kick_tds_opp",
                  "safeties_opp", "extra_points_returned_opp"):
            if c not in out.columns:
                out[c] = 0.0
        out[[c for c in out.columns if c.endswith("_opp")]] = out[
            [c for c in out.columns if c.endswith("_opp")]
        ].fillna(0.0)
        credited = (
            6.0 * (out["defensive_tds_opp"] + out["return_tds_opp"] + out["blocked_kick_tds_opp"])
            + 2.0 * out["safeties_opp"]
            + 2.0 * out["extra_points_returned_opp"]
        )
        out["points_allowed"] = (out["points_allowed"] - credited).clip(lower=0.0)

    out["points_allowed"] = out["points_allowed"].round().astype(int)
    out["tier_points"] = out["points_allowed"].map(scoring.points_allowed_points)
    out["yards_tier_points"] = (
        out["yards_allowed"].map(scoring.yards_allowed_points)
        if scoring.scores_yards_allowed
        else 0.0
    )
    out["big_play_points"] = (
        out["sacks"] * scoring.sack
        + out["interceptions"] * scoring.interception
        + out["fumble_recoveries"] * scoring.fumble_recovery
        + out["safeties"] * scoring.safety
        + out["one_point_safeties"] * scoring.one_point_safety
        + out["defensive_tds"] * scoring.defensive_td
        + out["return_tds"] * scoring.return_td
        + out["blocked_kicks"] * scoring.blocked_kick
        + out["blocked_kick_tds"] * scoring.blocked_kick_td
        + out["extra_points_returned"] * scoring.extra_point_returned
        + out["fourth_down_stops"] * scoring.fourth_down_stop
        + out["fumbles_lost"] * scoring.fumble_lost
    )
    out["fantasy_points"] = (
        out["tier_points"] + out["yards_tier_points"] + out["big_play_points"]
    )
    return out.reset_index(drop=True)


def offense_yards(pbp: pd.DataFrame) -> pd.DataFrame:
    """Total scrimmage yards gained per (game, offence).

    Sacks carry negative ``yards_gained`` and so reduce the total, which is the
    net-yards convention every fantasy host uses.
    """
    if pbp.empty:
        return pd.DataFrame(columns=["season", "week", "game_id", "posteam", "yards"])
    plays = pbp[
        pbp["posteam"].notna() & pbp["play_type"].isin(YARDAGE_PLAY_TYPES)
    ].copy()
    if plays.empty:
        return pd.DataFrame(columns=["season", "week", "game_id", "posteam", "yards"])
    plays["yards_gained"] = pd.to_numeric(plays["yards_gained"], errors="coerce").fillna(0.0)
    return (
        plays.groupby(["season", "week", "game_id", "posteam"], dropna=False)["yards_gained"]
        .sum()
        .reset_index()
        .rename(columns={"yards_gained": "yards"})
    )


def _dst_events(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per defense-game counts of every non-tier D/ST scoring event."""
    if pbp.empty:
        return pd.DataFrame(
            columns=["season", "week", "game_id", "team", *DST_EVENT_COLUMNS]
        )
    df = pbp.copy()
    for col in ("sack", "interception", "safety", "touchdown", "punt_blocked",
                "defensive_extra_point_conv"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    key = ["season", "week", "game_id"]
    #: stat name -> list of per-event frames. Several event types feed the same
    #: stat (blocked punts / FGs / PATs all count as blocked kicks), so they are
    #: accumulated under one key and summed once at the end.
    parts: dict[str, list[pd.DataFrame]] = {}

    def credit(mask: pd.Series, team_col: str, name: str) -> None:
        sub = df.loc[mask, key + [team_col]].copy()
        if sub.empty:
            return
        sub = sub.rename(columns={team_col: "team"})
        sub = sub[sub["team"].notna()]
        if sub.empty:
            return
        sub[name] = 1.0
        parts.setdefault(name, []).append(sub[key + ["team", name]])

    # Sacks, INTs and safeties are all credited to the defense on the play.
    credit(df["sack"] == 1, "defteam", "sacks")
    credit(df["interception"] == 1, "defteam", "interceptions")
    # Ordinary two-point safeties only; the one-point variety is credited below.
    credit(
        (df["safety"] == 1) & ~df["play_type"].fillna("").isin(("extra_point", "field_goal")),
        "defteam",
        "safeties",
    )

    # Fumble recoveries: the recovering team must differ from the fumbling one.
    for i in (1, 2):
        rec_col, fum_col = f"fumble_recovery_{i}_team", f"fumbled_{i}_team"
        if rec_col in df.columns and fum_col in df.columns:
            mask = (
                df[rec_col].notna()
                & df[fum_col].notna()
                & (df[rec_col] != df[fum_col])
            )
            credit(mask, rec_col, "fumble_recoveries")

    # Fumbles lost by the D/ST unit itself. Two cases, and neither is an
    # offensive fumble: a muffed punt or kickoff return (special teams is part
    # of the D/ST unit, so a botched snap on a punt counts too), and a defender
    # losing the ball back after taking it away.
    for i in (1, 2):
        rec_col, fum_col = f"fumble_recovery_{i}_team", f"fumbled_{i}_team"
        if rec_col not in df.columns or fum_col not in df.columns:
            continue
        lost = df[fum_col].notna() & df[rec_col].notna() & (df[fum_col] != df[rec_col])
        on_kick = df["play_type"].fillna("").isin(("punt", "kickoff"))
        on_defense = (
            df["play_type"].fillna("").isin(("pass", "run"))
            & (df[fum_col] == df["defteam"])
        )
        credit(lost & (on_kick | on_defense), fum_col, "fumbles_lost")

    # Fourth-down stops: the offence went for it and failed. Punts and field
    # goals are not conversion attempts and do not count. Yahoo scores these.
    if "fourth_down_failed" in df.columns:
        df["fourth_down_failed"] = pd.to_numeric(
            df["fourth_down_failed"], errors="coerce"
        ).fillna(0.0)
        credit(df["fourth_down_failed"] == 1, "defteam", "fourth_down_stops")

    # A one-point safety is a safety conceded on a try. Vanishingly rare, but
    # ESPN scores it, so it is counted rather than assumed away.
    credit(
        (df["safety"] == 1) & df["play_type"].fillna("").isin(("extra_point", "field_goal")),
        "defteam",
        "one_point_safeties",
    )

    # Blocked kicks: punts, field goals and PATs all count.
    credit(df["punt_blocked"] == 1, "defteam", "blocked_kicks")
    if "field_goal_result" in df.columns:
        credit(df["field_goal_result"].eq("blocked"), "defteam", "blocked_kicks")
    if "extra_point_result" in df.columns:
        credit(df["extra_point_result"].eq("blocked"), "defteam", "blocked_kicks")
    credit(df["defensive_extra_point_conv"] == 1, "defteam", "extra_points_returned")

    # Touchdowns. See the module docstring for the posteam conventions.
    td = df["touchdown"] == 1
    has_td_team = df["td_team"].notna()
    play_type = df["play_type"].fillna("")
    is_kick_play = play_type.isin(KICK_PLAY_TYPES)

    credit(td & has_td_team & play_type.isin(("punt", "kickoff")), "td_team", "return_tds")
    credit(td & has_td_team & play_type.isin(("field_goal", "extra_point")),
           "td_team", "blocked_kick_tds")
    credit(td & has_td_team & ~is_kick_play & (df["td_team"] != df["posteam"]),
           "td_team", "defensive_tds")

    out: pd.DataFrame | None = None
    for name, frames in parts.items():
        agg = (
            pd.concat(frames, ignore_index=True)
            .groupby(key + ["team"], dropna=False)[name]
            .sum()
            .reset_index()
        )
        out = agg if out is None else out.merge(agg, on=key + ["team"], how="outer")

    if out is None:
        out = pd.DataFrame(columns=key + ["team"])
    for col in DST_EVENT_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    return out.fillna(0.0)
