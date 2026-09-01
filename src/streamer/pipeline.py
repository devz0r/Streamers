"""Assembling everything into a projectable slate.

The CLI commands all need the same thing: historical features for training plus
feature rows for the week being projected -- including weeks that have not been
played, where every prior must come from the accumulator state as of the last
completed game.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, get_config
from .data.nflverse import games_frame, latest_available_season, load_pbp
from .data.odds import LinesResult, get_lines
from .data.weather import resolve_weather
from .features.build import build_dst_features, build_kicker_features
from .teams import CURRENT_TEAMS

log = logging.getLogger(__name__)


@dataclass
class SlateContext:
    """Training features plus the projectable rows for one (or more) weeks."""

    kicker_train: pd.DataFrame
    dst_train: pd.DataFrame
    kicker_slate: pd.DataFrame
    dst_slate: pd.DataFrame
    lines: LinesResult | None
    season: int
    week: int
    games: pd.DataFrame


def training_seasons(cfg: Config | None = None) -> list[int]:
    """Historical seasons plus the current one, clamped to what is published."""
    cfg = cfg or get_config()
    seasons = set(cfg.train_seasons) | {cfg.current_season}
    newest = latest_available_season(cfg)
    return sorted(s for s in seasons if s <= max(newest, min(seasons)))


def resolve_starting_kickers(
    pbp: pd.DataFrame, teams: list[str], cfg: Config | None = None
) -> pd.DataFrame:
    """Each team's most recent kicker, for projecting an unplayed slate.

    nflverse does not publish depth charts for a season before it starts, so the
    reliable signal is simply "who kicked most recently for this team".
    """
    cfg = cfg or get_config()
    kicks = pbp[
        pbp["kicker_player_id"].notna()
        & ((pbp["field_goal_attempt"] == 1) | (pbp["extra_point_attempt"] == 1))
    ]
    if kicks.empty:
        return pd.DataFrame(columns=["team", "player_id", "player_name"])
    kicks = kicks.sort_values(["season", "week"])
    latest = (
        kicks.groupby(["posteam", "kicker_player_id", "kicker_player_name"], dropna=False)
        .agg(last_season=("season", "max"), last_week=("week", "max"), kicks=("play_id", "count"))
        .reset_index()
        .sort_values(["posteam", "last_season", "last_week", "kicks"])
        .groupby("posteam")
        .tail(1)
    )
    out = latest.rename(
        columns={"posteam": "team", "kicker_player_id": "player_id",
                 "kicker_player_name": "player_name"}
    )[["team", "player_id", "player_name"]]
    missing = sorted(set(teams) - set(out["team"]))
    if missing:
        log.warning("no recent kicker found for: %s", ", ".join(missing))
    return out


def build_slate(
    season: int,
    week: int,
    cfg: Config | None = None,
    weeks_ahead: int = 1,
    allow_network: bool = True,
    lines: LinesResult | None = None,
) -> SlateContext:
    """Build training features and projectable rows for ``week`` (and beyond).

    ``weeks_ahead`` of 2 also builds week ``N+1``, which is what the two-week
    stream-candidate flag needs.
    """
    cfg = cfg or get_config()
    pbp = load_pbp(training_seasons(cfg), cfg)
    games = games_frame(cfg)

    target_weeks = list(range(week, week + max(1, weeks_ahead)))
    played_keys = set(
        zip(pbp["season"].astype(int), pbp["week"].astype(int))
    ) if not pbp.empty else set()

    slate_games = games[
        (games["season"] == season) & (games["week"].isin(target_weeks))
    ].copy()
    if slate_games.empty:
        raise ValueError(f"no scheduled games for {season} week {week}")

    future_games = slate_games[
        [(season, int(w)) not in played_keys for w in slate_games["week"]]
    ][["season", "week", "game_id", "team", "opponent"]].copy()

    kickers = resolve_starting_kickers(pbp, list(CURRENT_TEAMS), cfg)
    future_kickers = pd.DataFrame()
    if not future_games.empty and not kickers.empty:
        future_kickers = future_games.merge(kickers, on="team", how="left")
        future_kickers = future_kickers[future_kickers["player_id"].notna()]

    kicker_all = build_kicker_features(pbp, games, cfg, future_games, future_kickers)
    dst_all = build_dst_features(pbp, games, cfg, future_games)

    kicker_train = kicker_all[kicker_all["fantasy_points"].notna()].copy()
    dst_train = dst_all[dst_all["fantasy_points"].notna()].copy()

    def slice_slate(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[
            (frame["season"] == season) & (frame["week"].isin(target_weeks))
        ].copy()

    kicker_slate = slice_slate(kicker_all)
    dst_slate = slice_slate(dst_all)

    if lines is None:
        try:
            lines = get_lines(season, week, cfg, allow_network=allow_network)
        except Exception as exc:  # noqa: BLE001 - schedule lines still work
            log.warning("could not resolve lines: %s", exc)
            lines = None

    if lines is not None:
        kicker_slate = apply_lines(kicker_slate, lines, week, cfg)
        dst_slate = apply_lines(dst_slate, lines, week, cfg)

    return SlateContext(
        kicker_train=kicker_train,
        dst_train=dst_train,
        kicker_slate=kicker_slate,
        dst_slate=dst_slate,
        lines=lines,
        season=season,
        week=week,
        games=slate_games,
    )


def apply_lines(
    slate: pd.DataFrame, lines: LinesResult, week: int, cfg: Config | None = None
) -> pd.DataFrame:
    """Overwrite the Vegas columns with freshly resolved lines.

    The feature builder seeds them from the nflverse schedule; this replaces
    them with whatever the fallback chain actually produced, then rebuilds the
    interaction terms that depend on them.
    """
    cfg = cfg or get_config()
    if slate.empty or lines is None or lines.frame.empty:
        return slate
    out = slate.copy()
    lookup: dict[tuple[str, str], tuple[float, float, str]] = {}
    for row in lines.frame.itertuples():
        if pd.isna(row.spread_line) or pd.isna(row.total_line):
            continue
        lookup[(row.home_team, row.away_team)] = (
            float(row.spread_line), float(row.total_line), str(row.line_source)
        )

    if "line_source" not in out.columns:
        out["line_source"] = None
    for idx, row in out.iterrows():
        home = row["team"] if row.get("is_home") == 1 else row["opponent"]
        away = row["opponent"] if row.get("is_home") == 1 else row["team"]
        entry = lookup.get((home, away))
        if entry is None:
            continue
        spread_home, total, source = entry
        team_spread = spread_home if row.get("is_home") == 1 else -spread_home
        out.at[idx, "team_spread"] = team_spread
        out.at[idx, "total_line"] = total
        out.at[idx, "team_implied_total"] = total / 2.0 + team_spread / 2.0
        out.at[idx, "opp_implied_total"] = total / 2.0 - team_spread / 2.0
        out.at[idx, "line_source"] = source

    if "off_fga_per_drive" in out.columns:
        out["implied_x_fga_rate"] = out["team_implied_total"] * out["off_fga_per_drive"]
        out["implied_x_rz_stall"] = out["team_implied_total"] * (1.0 - out["off_rz_td_rate"])
    if "def_points_allowed_per_drive" in out.columns:
        out["implied_x_def_quality"] = (
            out["opp_implied_total"] * out["def_points_allowed_per_drive"]
        )
    if "sack_opportunity" in out.columns:
        out["sack_x_spread"] = out["sack_opportunity"] * out["team_spread"]
    if "wind" in out.columns and "k_long_att_share" in out.columns:
        out = resolve_weather(
            out.assign(home_team=np.where(out["is_home"] == 1, out["team"], out["opponent"])),
            week=week,
            cfg=cfg,
        )
        out["wind_x_long_share"] = out["wind"] * out["k_long_att_share"]
    return out


# ---------------------------------------------------------------------------
# Persisted model selection
# ---------------------------------------------------------------------------
def selection_path(cfg: Config | None = None) -> Path:
    cfg = cfg or get_config()
    return cfg.results_dir / "model_selection.json"


def load_selection(cfg: Config | None = None) -> dict[str, dict]:
    """What the last backtest chose, per position.

    Each entry is a mapping like ``{"estimator": "ridge", "ridge_alpha": 300.0}``.
    A bare string is also accepted, so a hand-written file stays valid.
    """
    path = selection_path(cfg)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        position: ({"estimator": value} if isinstance(value, str) else dict(value))
        for position, value in raw.items()
        if isinstance(value, (str, dict))
    }


def save_selection(selection: dict[str, dict], cfg: Config | None = None) -> None:
    selection_path(cfg).write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")


def estimator_for(position: str, cfg: Config | None = None) -> str:
    """Estimator the backtest selected for ``position`` (default: ridge)."""
    cfg = cfg or get_config()
    return str(load_selection(cfg).get(position, {}).get("estimator", "ridge"))


def tuned_alpha_for(position: str, cfg: Config | None = None) -> float | None:
    """Ridge penalty the backtest selected, if it swept one."""
    cfg = cfg or get_config()
    value = load_selection(cfg).get(position, {}).get("ridge_alpha")
    return None if value is None else float(value)
