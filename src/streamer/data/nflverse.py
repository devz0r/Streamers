"""nflverse data access.

``nfl_data_py`` has not shipped a release since September 2024; the nflverse
maintainers now publish ``nflreadpy`` (Python 3.10+, polars-backed), which is
what this project uses. See DECISIONS.md.

Everything here returns pandas frames with normalised team abbreviations, and
everything is cached to ``data/raw/``.
"""

from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd

from ..config import Config, get_config
from ..teams import normalize_team_series
from .cache import cached_frame

log = logging.getLogger(__name__)

#: pbp columns the feature builders actually need. Loading a subset keeps the
#: 50k-row-per-season frames manageable.
PBP_COLUMNS: tuple[str, ...] = (
    "game_id", "season", "season_type", "week", "home_team", "away_team",
    "posteam", "defteam", "play_type", "play_id", "drive", "fixed_drive",
    "fixed_drive_result", "yardline_100", "down", "ydstogo", "qtr",
    "field_goal_attempt", "field_goal_result", "kick_distance",
    "extra_point_attempt", "extra_point_result",
    "kicker_player_id", "kicker_player_name",
    "sack", "qb_hit", "interception", "complete_pass", "incomplete_pass",
    "fourth_down_converted", "fourth_down_failed", "yards_gained",
    "pass_attempt", "rush_attempt", "qb_dropback", "qb_scramble",
    "fumble", "fumble_lost", "fumble_forced",
    "fumble_recovery_1_team", "fumble_recovery_2_team",
    "fumbled_1_team", "fumbled_2_team",
    "safety", "touchdown", "td_team", "return_touchdown",
    "punt_blocked", "punt_attempt", "kickoff_attempt",
    "defensive_extra_point_conv",
    "epa", "wp", "score_differential", "half_seconds_remaining",
    "game_seconds_remaining", "penalty", "aborted_play",
)


def _nflreadpy():
    import nflreadpy  # imported lazily so tests can run without a network stack

    return nflreadpy


def _to_pandas(frame) -> pd.DataFrame:
    """Accept a polars or pandas frame and return pandas."""
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return frame


def latest_available_season(cfg: Config | None = None) -> int:
    """Newest season for which nflverse publishes play-by-play.

    ``nflreadpy`` rejects seasons past the current data season, so callers that
    want "everything up to now" must clamp against this.
    """
    try:
        return int(_nflreadpy().get_current_season())
    except Exception:  # noqa: BLE001 - fall back to the schedule feed
        try:
            sched = load_schedules(cfg)
            played = sched.loc[sched["home_score"].notna(), "season"]
            if len(played):
                return int(played.max())
        except Exception:  # noqa: BLE001
            pass
        from datetime import date

        today = date.today()
        return today.year if today.month >= 9 else today.year - 1


def load_schedules(cfg: Config | None = None, refresh: bool = False) -> pd.DataFrame:
    """Game schedule with closing spread/total, roof, and final scores."""
    cfg = cfg or get_config()
    path = cfg.raw_dir / "schedules.parquet"

    def build() -> pd.DataFrame:
        df = _to_pandas(_nflreadpy().load_schedules())
        for col in ("home_team", "away_team"):
            df[col] = normalize_team_series(df[col])
        return df

    # Schedules carry live results, so they are refreshed whenever the caller
    # asks; the cache exists so backtests do not re-download on every run.
    df = cached_frame(path, build, refresh=refresh)
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    return df


def load_pbp(seasons: list[int], cfg: Config | None = None, refresh: bool = False) -> pd.DataFrame:
    """Play-by-play for ``seasons``, one cached parquet per season.

    Seasons with no published data yet (e.g. the upcoming season before Week 1)
    are skipped with a warning rather than raising.
    """
    cfg = cfg or get_config()
    frames: list[pd.DataFrame] = []
    newest = latest_available_season(cfg)
    for season in sorted(set(int(s) for s in seasons)):
        path = cfg.raw_dir / f"pbp_{season}.parquet"
        if season > newest and not path.exists():
            log.info("season %s has no published play-by-play yet; skipping", season)
            continue

        def build(season: int = season) -> pd.DataFrame:
            return _to_pandas(_nflreadpy().load_pbp(seasons=[season]))

        try:
            df = cached_frame(path, build, refresh=refresh)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load play-by-play for %s: %s", season, exc)
            continue
        keep = [c for c in PBP_COLUMNS if c in df.columns]
        df = df[keep].copy()
        df["season"] = season
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(PBP_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    for col in ("posteam", "defteam", "home_team", "away_team", "td_team",
                "fumble_recovery_1_team", "fumble_recovery_2_team",
                "fumbled_1_team", "fumbled_2_team"):
        if col in out.columns:
            out[col] = normalize_team_series(out[col])
    out["week"] = out["week"].astype(int)
    return out


def load_rosters(seasons: list[int], cfg: Config | None = None, refresh: bool = False) -> pd.DataFrame:
    """Weekly rosters, used to resolve each team's current kicker."""
    cfg = cfg or get_config()
    frames = []
    newest = latest_available_season(cfg)
    for season in sorted(set(int(s) for s in seasons)):
        path = cfg.raw_dir / f"rosters_{season}.parquet"
        if season > newest and not path.exists():
            continue

        def build(season: int = season) -> pd.DataFrame:
            return _to_pandas(_nflreadpy().load_rosters(seasons=[season]))

        try:
            df = cached_frame(path, build, refresh=refresh)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load rosters for %s: %s", season, exc)
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "team" in out.columns:
        out["team"] = normalize_team_series(out["team"])
    return out


def games_frame(cfg: Config | None = None, refresh: bool = False) -> pd.DataFrame:
    """One row per team-game: team, opponent, home flag, venue, result.

    This is the spine every feature table joins onto.
    """
    sched = load_schedules(cfg, refresh=refresh)
    cols = [
        "game_id", "season", "week", "game_type", "gameday", "gametime",
        "home_team", "away_team", "home_score", "away_score",
        "spread_line", "total_line", "roof", "surface", "temp", "wind",
        "stadium", "div_game", "home_rest", "away_rest",
    ]
    cols = [c for c in cols if c in sched.columns]
    sched = sched[cols].copy()

    home = sched.copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["is_home"] = 1
    home["team_score"] = home["home_score"]
    home["opp_score"] = home["away_score"]
    # nflverse spread_line is the HOME team's line: positive = home favoured.
    home["team_spread"] = home["spread_line"]
    home["rest"] = home.get("home_rest", np.nan)

    away = sched.copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["is_home"] = 0
    away["team_score"] = away["away_score"]
    away["opp_score"] = away["home_score"]
    away["team_spread"] = -away["spread_line"]
    away["rest"] = away.get("away_rest", np.nan)

    out = pd.concat([home, away], ignore_index=True)
    out["total_line"] = pd.to_numeric(out["total_line"], errors="coerce")
    out["team_spread"] = pd.to_numeric(out["team_spread"], errors="coerce")
    # Implied totals: half the game total, adjusted half the spread.
    out["team_implied_total"] = out["total_line"] / 2.0 + out["team_spread"] / 2.0
    out["opp_implied_total"] = out["total_line"] / 2.0 - out["team_spread"] / 2.0
    out["is_dome"] = out["roof"].isin(["dome", "closed"]).astype(int)
    out = out.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    return out


@functools.lru_cache(maxsize=4)
def cached_games_frame() -> pd.DataFrame:
    return games_frame()
