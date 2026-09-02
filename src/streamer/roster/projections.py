"""Weekly projections for every player on a league snapshot.

Skill positions (QB/RB/WR/TE) are projected from nflverse history with a
formula chosen by walk-forward validation (see DECISIONS.md):

    blend  = shrink( 0.5 * trailing(actual PPR) + 0.5 * trailing(opportunity-expected PPR) )
    mean   = blend * (implied_total / trailing_implied_total) ** vegas_damping

The opportunity-expected term comes from nflverse ``ff_opportunity`` -- points
a player *should* have scored given their targets, carries and field position
-- and is less noisy than what they actually scored. The Vegas term scales by
how this week's game total compares with the team's recent ones, damped
because the full effect helps quarterbacks and tight ends but hurts receivers.

D/ST and K come from the streaming model, which is already validated against
the Vegas baseline and knows about two-week holds.

Every projection carries a standard deviation calibrated from historical
residuals by position and projection level, which is what the lineup
simulator draws from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config, get_config
from ..data.cache import cached_frame
from ..data.nflverse import _to_pandas, games_frame, latest_available_season
from ..data.odds import get_lines, lines_to_team_rows
from ..league.model import LeagueSnapshot, PlayerRow
from .players import build_index, match_players

log = logging.getLogger(__name__)

SKILL = ("QB", "RB", "WR", "TE")

#: Projection-level buckets for the sd calibration, per position.
SD_BUCKETS = (0.0, 5.0, 8.0, 12.0, 16.0, 20.0, 99.0)


@dataclass
class ProjectionReport:
    """What was projected, and what could not be."""

    projected: int = 0
    unmatched: list[str] = field(default_factory=list)
    line_source: str = ""
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def _nflreadpy():
    import nflreadpy

    return nflreadpy


def load_history(cfg: Config | None = None) -> pd.DataFrame:
    """Weekly PPR points and opportunity-expected points, all training seasons."""
    cfg = cfg or get_config()
    seasons = sorted(set(cfg.train_seasons) | {cfg.current_season})
    newest = latest_available_season(cfg)
    frames = []
    for season in seasons:
        stats_path = cfg.raw_dir / f"player_stats_{season}.parquet"
        opp_path = cfg.raw_dir / f"ff_opportunity_{season}.parquet"
        if season > newest and not stats_path.exists():
            continue
        try:
            stats = cached_frame(
                stats_path,
                lambda s=season: _to_pandas(_nflreadpy().load_player_stats(seasons=[s])),
            )
            opp = cached_frame(
                opp_path,
                lambda s=season: _to_pandas(_nflreadpy().load_ff_opportunity(seasons=[s])),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("player history for %s unavailable: %s", season, exc)
            continue
        stats = stats[stats["season_type"].eq("REG") & stats["position"].isin(SKILL)]
        stats = stats[["player_id", "player_display_name", "position", "season", "week",
                       "team", "fantasy_points_ppr"]].copy()
        stats["season"] = stats["season"].astype(int)
        stats["week"] = stats["week"].astype(int)
        opp = opp[["player_id", "season", "week", "total_fantasy_points_exp"]].copy()
        opp["season"] = opp["season"].astype(int)
        opp["week"] = opp["week"].astype(int)
        frames.append(stats.merge(opp, on=["player_id", "season", "week"], how="left"))
    if not frames:
        return pd.DataFrame(columns=["player_id", "player_display_name", "position", "season",
                                     "week", "team", "fantasy_points_ppr",
                                     "total_fantasy_points_exp"])
    out = pd.concat(frames, ignore_index=True)
    from ..teams import normalize_team_series

    out["team"] = normalize_team_series(out["team"])
    return out.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------
def _trailing_features(history: pd.DataFrame, n_games: int) -> pd.DataFrame:
    """Leak-free trailing means: each row sees only the rows before it."""
    h = history.copy()
    grp = h.groupby("player_id", group_keys=False)
    h["t_act"] = grp["fantasy_points_ppr"].transform(
        lambda s: s.shift(1).rolling(n_games, min_periods=2).mean()
    )
    h["t_exp"] = grp["total_fantasy_points_exp"].transform(
        lambda s: s.shift(1).rolling(n_games, min_periods=2).mean()
    )
    h["n"] = grp["fantasy_points_ppr"].transform(lambda s: s.shift(1).expanding().count())
    return h


def _blend(frame: pd.DataFrame, pos_mean: pd.Series, k: float) -> pd.Series:
    raw = 0.5 * frame["t_act"] + 0.5 * frame["t_exp"].fillna(frame["t_act"])
    return (frame["n"] * raw + k * pos_mean) / (frame["n"] + k)


def player_table(history: pd.DataFrame, season: int, week: int, cfg: Config) -> pd.DataFrame:
    """Per nflverse player: the blend as of (season, week), before any matchup scaling."""
    conf = cfg.raw["roster"]
    n_games = int(conf["trailing_games"])
    k = float(conf["shrink_games"])
    if history.empty:
        return pd.DataFrame(columns=["player_id", "player_display_name", "position", "team", "blend", "n"])

    prior = history[
        (history["season"] < season) | ((history["season"] == season) & (history["week"] < week))
    ]
    if prior.empty:
        return pd.DataFrame(columns=["player_id", "player_display_name", "position", "team", "blend", "n"])
    # Shrinkage target from completed games only: the target week must not
    # inform its own projection.
    pos_mean_all = prior.groupby("position")["fantasy_points_ppr"].mean()

    # Append one placeholder row per player for the target week so the trailing
    # window lands on exactly the games before it.
    last = prior.sort_values(["season", "week"]).groupby("player_id").tail(1)
    placeholder = last[["player_id", "player_display_name", "position", "team"]].copy()
    placeholder["season"], placeholder["week"] = season, week
    placeholder["fantasy_points_ppr"] = np.nan
    placeholder["total_fantasy_points_exp"] = np.nan
    stacked = pd.concat([prior, placeholder], ignore_index=True)
    stacked = stacked.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    feats = _trailing_features(stacked, n_games)
    target = feats[(feats["season"] == season) & (feats["week"] == week)].copy()
    target["pos_mean"] = target["position"].map(pos_mean_all)
    target["blend"] = _blend(target, target["pos_mean"], k)
    target = target[target["blend"].notna()]
    return target[["player_id", "player_display_name", "position", "team", "blend", "n"]]


def sd_table(history: pd.DataFrame, cfg: Config) -> dict[tuple[str, int], float]:
    """Residual sd of the blend by (position, projection bucket), from history."""
    conf = cfg.raw["roster"]
    if history.empty:
        return {}
    feats = _trailing_features(history, int(conf["trailing_games"]))
    pos_mean = feats.groupby("position")["fantasy_points_ppr"].transform("mean")
    feats["blend"] = _blend(feats, pos_mean, float(conf["shrink_games"]))
    feats = feats[feats["blend"].notna()]
    feats["resid"] = feats["fantasy_points_ppr"] - feats["blend"]
    feats["bucket"] = np.digitize(feats["blend"], SD_BUCKETS[1:-1])
    table: dict[tuple[str, int], float] = {}
    for (pos, bucket), g in feats.groupby(["position", "bucket"]):
        if len(g) >= 30:
            table[(str(pos), int(bucket))] = float(g["resid"].std())
    # Fallback per position for sparse buckets.
    for pos, g in feats.groupby("position"):
        table[(str(pos), -1)] = float(g["resid"].std())
    return table


def lookup_sd(table: dict[tuple[str, int], float], position: str, mean: float) -> float:
    bucket = int(np.digitize([mean], SD_BUCKETS[1:-1])[0])
    return table.get((position, bucket), table.get((position, -1), 7.5))


# ---------------------------------------------------------------------------
# Attaching projections to a snapshot
# ---------------------------------------------------------------------------
def _implied_scale(snapshot: LeagueSnapshot, cfg: Config, allow_network: bool) -> tuple[dict[str, float], str]:
    """Team -> (this week's implied total / trailing implied total)."""
    games = games_frame(cfg)
    hist = games[
        (games["season"] < snapshot.season)
        | ((games["season"] == snapshot.season) & (games["week"] < snapshot.week))
    ].dropna(subset=["team_implied_total"])
    trailing = (
        hist.sort_values(["season", "week"]).groupby("team")["team_implied_total"]
        .apply(lambda s: float(s.tail(6).mean()))
    )
    try:
        lines = get_lines(snapshot.season, snapshot.week, cfg, allow_network=allow_network)
        rows = lines_to_team_rows(lines)
        this_week = rows.set_index("team")["team_implied_total"].dropna()
        source = lines.describe()
    except Exception as exc:  # noqa: BLE001
        log.warning("no lines for implied-total scaling: %s", exc)
        return {}, f"unavailable ({exc})"
    scale = {}
    for team, implied in this_week.items():
        base = trailing.get(team)
        if base and base > 0:
            scale[team] = float(np.clip(implied / base, 0.7, 1.3))
    return scale, source


def _status_adjust(mean: float, sd: float, player: PlayerRow, cfg: Config) -> tuple[float, float]:
    """Scale a projection by the chance the player actually plays."""
    conf = cfg.raw["roster"]
    if player.is_out:
        return 0.0, 0.0
    if player.status in ("DOUBTFUL", "D"):
        p = float(conf["doubtful_play_probability"])
    elif player.is_questionable:
        p = float(conf["questionable_play_probability"])
    else:
        return mean, sd
    # Mixture of "plays" (mean, sd) and "does not" (0, 0).
    mix_mean = p * mean
    mix_var = p * (sd ** 2 + mean ** 2) - mix_mean ** 2
    return mix_mean, float(np.sqrt(max(mix_var, 0.0)))


def project_snapshot(
    snapshot: LeagueSnapshot,
    cfg: Config | None = None,
    rankings=None,
    allow_network: bool = True,
    history: pd.DataFrame | None = None,
) -> ProjectionReport:
    """Fill ``projection``/``projection_sd``/``ros_value`` on every player in place."""
    cfg = cfg or get_config()
    conf = cfg.raw["roster"]
    report = ProjectionReport()

    history = load_history(cfg) if history is None else history
    table = player_table(history, snapshot.season, snapshot.week, cfg)
    sds = sd_table(history, cfg)
    scale, report.line_source = _implied_scale(snapshot, cfg, allow_network)
    damping = float(conf["vegas_damping"])
    w_platform = float(conf["platform_projection_weight"])

    # nflverse index for name matching: newest team per player.
    latest = history.sort_values(["season", "week"]).groupby("player_id").tail(1)
    index = build_index(latest[["player_id", "player_display_name", "position", "team"]])
    players = snapshot.all_players()
    matched = match_players([p for p in players if p.position in SKILL], index)
    report.unmatched = matched.unmatched
    by_id = table.set_index("player_id") if not table.empty else pd.DataFrame()
    pos_mean = history.groupby("position")["fantasy_points_ppr"].mean() if not history.empty else pd.Series(dtype=float)

    # D/ST and K from the streaming model.
    dst_rows: dict[str, pd.Series] = {}
    k_rows: dict[str, pd.Series] = {}
    if rankings is not None:
        if rankings.dst is not None and not rankings.dst.empty:
            dst_rows = {r.team: r for r in rankings.dst.itertuples()}
        if rankings.kicker is not None and not rankings.kicker.empty:
            k_rows = {r.team: r for r in rankings.kicker.itertuples()}

    for p in players:
        mean: float | None = None
        sd: float | None = None
        ros: float | None = None
        source = ""

        if p.position in SKILL:
            nfl_id = matched.mapping.get(p.player_id)
            if nfl_id is not None and not by_id.empty and nfl_id in by_id.index:
                row = by_id.loc[nfl_id]
                blend = float(row["blend"])
                ros = blend
                s = scale.get(p.team, 1.0) if p.team else 1.0
                mean = blend * (s ** damping)
                source = "model"
            elif p.platform_projection is not None:
                mean = float(p.platform_projection)
                ros = mean
                source = "platform"
            else:
                # Unknown player: a heavily discounted position average.
                mean = 0.5 * float(pos_mean.get(p.position, 8.0)) if len(pos_mean) else 4.0
                ros = mean
                source = "prior"
            if p.platform_projection is not None and source == "model" and w_platform > 0:
                mean = (1 - w_platform) * mean + w_platform * float(p.platform_projection)
                source = "model+platform"
            sd = lookup_sd(sds, p.position, mean)

        elif p.position == "DST" and p.team in dst_rows:
            r = dst_rows[p.team]
            mean = float(r.expected_points)
            sd = max((float(r.ceiling) - float(r.floor)) / 2.07, 1.0)
            ros = float(getattr(r, "baseline_points", mean) or mean)
            source = "stream/hold" if bool(getattr(r, "two_week_hold", False)) else "stream"

        elif p.position == "K" and p.team in k_rows:
            r = k_rows[p.team]
            mean = float(r.expected_points)
            sd = max((float(r.ceiling) - float(r.floor)) / 2.07, 1.0)
            ros = float(getattr(r, "baseline_points", mean) or mean)
            source = "stream/hold" if bool(getattr(r, "two_week_hold", False)) else "stream"

        elif p.position in ("DST", "K"):
            if p.platform_projection is not None:
                mean, sd, ros, source = float(p.platform_projection), 4.5, float(p.platform_projection), "platform"
            else:
                mean, sd, ros, source = 6.0, 4.5, 6.0, "prior"

        if mean is None:
            continue
        mean, sd = _status_adjust(mean, sd or 0.0, p, cfg)
        p.projection = round(mean, 2)
        p.projection_sd = round(sd, 2)
        p.ros_value = round(ros, 2) if ros is not None else None
        p.projection_source = source
        report.projected += 1

    if report.unmatched:
        report.notes.append(f"{len(report.unmatched)} player(s) could not be matched to nflverse history")
    return report
