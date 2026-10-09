"""Weekly projections for every player on a league snapshot.

Skill positions (QB/RB/WR/TE) are projected from nflverse history as
**opportunity times efficiency**, a formula chosen by walk-forward validation
(see DECISIONS.md):

    volume     = recent opportunity (exponentially weighted, half-life 2.5
                 games), anchored to his ~17-game opportunity with 2 games
                 of weight
    efficiency = points per expected point over ~17 games, shrunk toward the
                 position's rate, with a quarter of the weight on the last 4
    mean       = volume * efficiency * (implied_total / trailing_implied) ** 0.5

Opportunity comes from nflverse ``ff_opportunity`` -- points a player *should*
have scored given his targets, carries and field position. It is sticky, so
it gets a fast window; efficiency (touchdowns, big plays) mostly regresses, so
it gets a slow one. When a starter is ruled out, the next man up at his
position inherits part of his opportunity before he has played a snap in the
role -- measured from every absence 2021-2025.

D/ST and K come from the streaming model, which is already validated against
the Vegas baseline and knows about two-week holds.

Every projection carries a standard deviation calibrated from historical
residuals by position and projection level, which is what the lineup
simulator draws from, and a role (QB, RB1, WR2, ...) and opponent, through
which it correlates players in the same game.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config, get_config
from ..data.cache import cached_frame
from ..data.nflverse import (
    _to_pandas,
    games_frame,
    latest_available_season,
    live_max_age,
)
from ..data.odds import get_lines, lines_to_team_rows
from ..league.model import LONG_TERM_OUT_STATUSES, OUT_STATUSES, LeagueSnapshot, PlayerRow
from . import depth, market_calibration, outcome
from .players import build_index, match_players

log = logging.getLogger(__name__)

SKILL = ("QB", "RB", "WR", "TE")
#: The team of a player the news has signing somewhere it does not name.
FREE_AGENT = "FA"

#: Projection-level buckets for the sd calibration, per position.
SD_BUCKETS = (0.0, 5.0, 8.0, 12.0, 16.0, 20.0, 99.0)


@dataclass
class ProjectionReport:
    """What was projected, and what could not be."""

    projected: int = 0
    unmatched: list[str] = field(default_factory=list)
    line_source: str = ""
    notes: list[str] = field(default_factory=list)
    #: Players the platform projects for zero this week (not playing).
    sitting: list[str] = field(default_factory=list)
    #: What was locked in from games already played.
    lock_notes: list[str] = field(default_factory=list)
    #: QBs/RBs/TEs placed on Sleeper's depth chart.
    depth_matched: int = 0
    #: The weight this week's projection gave the platform's own number, and
    #: the graded player-weeks it was fitted on (0: the configured prior).
    platform_weight: float = 0.5
    platform_games: int = 0


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
            # The season being played grows every week, so its cache expires;
            # completed seasons are immutable and keep theirs.
            ttl = live_max_age(season, cfg)
            stats = cached_frame(
                stats_path,
                lambda s=season: _to_pandas(_nflreadpy().load_player_stats(seasons=[s])),
                max_age_hours=ttl,
            )
            opp = cached_frame(
                opp_path,
                lambda s=season: _to_pandas(_nflreadpy().load_ff_opportunity(seasons=[s])),
                max_age_hours=ttl,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("player history for %s unavailable: %s", season, exc)
            continue
        stats = stats[stats["season_type"].eq("REG") & stats["position"].isin(SKILL)]
        usage = [c for c in ("targets", "carries", "target_share") if c in stats.columns]
        stats = stats[["player_id", "player_display_name", "position", "season", "week",
                       "team", "fantasy_points_ppr", *usage]].copy()
        stats["season"] = stats["season"].astype(int)
        stats["week"] = stats["week"].astype(int)
        td_cols = [c for c in ("rec_touchdown_exp", "rush_touchdown_exp") if c in opp.columns]
        opp = opp[["player_id", "season", "week", "total_fantasy_points_exp", *td_cols]].copy()
        if td_cols:
            opp["xtd"] = opp[td_cols].fillna(0).sum(axis=1)
            opp = opp.drop(columns=td_cols)
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
def _trailing_features(history: pd.DataFrame, n_games: int, long_games: int = 17,
                       halflife: float = 2.5) -> pd.DataFrame:
    """Leak-free trailing sums: each row sees only the rows before it.

    Two windows -- the last ``n_games`` (current form) and the last
    ``long_games`` (roughly a season) -- plus an exponentially weighted mean
    of opportunity, which is what moves fastest when a role changes. Expected
    points fall back to actual points where nflverse has no opportunity row.
    """
    h = history.reset_index(drop=True).copy()
    h["_exp"] = h["total_fantasy_points_exp"].fillna(h["fantasy_points_ppr"])
    pid = h["player_id"]
    # Every window looks strictly backwards: shift by one game within player.
    prev_pts = h.groupby(pid, sort=False)["fantasy_points_ppr"].shift(1)
    prev_exp = h.groupby(pid, sort=False)["_exp"].shift(1)

    def roll(series: pd.Series, n: int, how: str) -> pd.Series:
        r = getattr(series.groupby(pid, sort=False).rolling(n, min_periods=1), how)()
        return r.reset_index(level=0, drop=True).sort_index()

    h["a_short"] = roll(prev_pts, n_games, "sum")
    h["e_short"] = roll(prev_exp, n_games, "sum")
    h["c_short"] = roll(prev_pts, n_games, "count")
    h["a_long"] = roll(prev_pts, long_games, "sum")
    h["e_long"] = roll(prev_exp, long_games, "sum")
    h["c_long"] = roll(prev_pts, long_games, "count")
    h["e_2"] = roll(prev_exp, 2, "sum")
    h["c_2"] = roll(prev_pts, 2, "count")
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    ew = prev_exp.groupby(pid, sort=False).ewm(alpha=alpha, adjust=True).mean()
    h["v_recent"] = ew.reset_index(level=0, drop=True).sort_index()
    h["n"] = prev_pts.notna().astype(float).groupby(pid, sort=False).cumsum()
    return h.drop(columns=["_exp"])


def position_rates(history: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per position: mean points per game, and points per expected point."""
    exp = history["total_fantasy_points_exp"].fillna(history["fantasy_points_ppr"])
    grp = history.assign(_exp=exp).groupby("position")
    return grp["fantasy_points_ppr"].mean(), grp["fantasy_points_ppr"].sum() / grp["_exp"].sum()


def _volume_efficiency(frame: pd.DataFrame, pos_mean: pd.Series, pos_eff: pd.Series,
                       conf: dict) -> tuple[pd.Series, pd.Series]:
    """Opportunity per game, and points per unit of opportunity.

    Volume reacts quickly: an exponentially weighted mean of recent expected
    points, anchored to his ~17-game volume with ``volume_prior_games`` of
    weight. Efficiency reacts slowly: long-run points per expected point,
    shrunk toward the position rate with ``efficiency_prior_points``, and only
    ``recent_efficiency_weight`` on the last few games. Walk-forward 2022-2025
    this beat the production blend in every season on start/sit accuracy and
    MAE, and at every position on rank correlation -- opportunity changes
    persist, scoring streaks mostly do not.
    """
    k = float(conf["shrink_games"])
    k_vol = float(conf.get("volume_prior_games", 2.0))
    m_eff = float(conf.get("efficiency_prior_points", 150.0))
    m_rec = float(conf.get("recent_efficiency_prior_points", 60.0))
    w_rec = float(conf.get("recent_efficiency_weight", 0.25))

    c_long = frame["c_long"].fillna(0.0)
    e_long = frame["e_long"].fillna(0.0)
    vol_long = (e_long + k * pos_mean) / (c_long + k)
    c_short = frame["c_short"].fillna(0.0)
    vol = (c_short * frame["v_recent"] + k_vol * vol_long) / (c_short + k_vol)
    vol = vol.where(frame["v_recent"].notna(), vol_long)
    eff_long = (frame["a_long"].fillna(0.0) + m_eff * pos_eff) / (e_long + m_eff)
    eff_rec = (frame["a_short"].fillna(0.0) + m_rec * eff_long) / (frame["e_short"].fillna(0.0) + m_rec)
    eff = (1.0 - w_rec) * eff_long + w_rec * eff_rec
    # No games at all: nothing to project from.
    none = c_long <= 0
    return vol.mask(none), eff.mask(none)


def _blend(frame: pd.DataFrame, pos_mean: pd.Series, pos_eff: pd.Series, conf: dict) -> pd.Series:
    """The per-game projection before any matchup scaling: volume x efficiency."""
    vol, eff = _volume_efficiency(frame, pos_mean, pos_eff, conf)
    return vol * eff


TABLE_COLUMNS = ["player_id", "player_display_name", "position", "team", "blend", "vol", "eff",
                 "n", "last_season", "last_week", "recent_exp", "prior_exp", "first_season",
                 "team_games", "old_team", "vol_all_teams", "vol_ros", "vol_before_back", "back_from"]


def _window_volume(last: pd.DataFrame, weight, exp, conf: dict, pos_mean: pd.Series) -> pd.Series:
    """Per player: the volume :func:`_volume_efficiency` gives from his window
    of games (``last``: his last ``long_games``, with ``age`` = how many of
    his games ago), each counting ``weight`` at expected points ``exp`` --
    so a correction can say how far re-weighting or re-reading some of his
    games moves it."""
    k, k_vol = float(conf["shrink_games"]), float(conf.get("volume_prior_games", 2.0))
    alpha = 1.0 - 0.5 ** (1.0 / float(conf.get("volume_halflife_games", 2.5)))
    n_short = int(conf["trailing_games"])
    wt = np.asarray(weight, float)
    x = np.asarray(exp, float)
    age = last["age"].to_numpy()
    ew = (1.0 - alpha) ** age * wt
    f = pd.DataFrame({"player_id": last["player_id"].to_numpy(), "ew": ew, "ewx": ew * x, "wt": wt, "wtx": wt * x,
                      "short": np.where(age < n_short, wt, 0.0),
                      "pos_mean": last["position"].map(pos_mean).to_numpy()})
    s = f.groupby("player_id").agg(ew=("ew", "sum"), ewx=("ewx", "sum"), c_long=("wt", "sum"),
                                   e_long=("wtx", "sum"), c_short=("short", "sum"), pos_mean=("pos_mean", "first"))
    recent = s["ewx"] / s["ew"].where(s["ew"] > 1e-9)
    vol_long = (s["e_long"] + k * s["pos_mean"]) / (s["c_long"] + k)
    vol = (s["c_short"] * recent + k_vol * vol_long) / (s["c_short"] + k_vol)
    return vol.where(recent.notna(), vol_long)


def _team_discount(prior: pd.DataFrame, target: pd.DataFrame, teams: dict[str, str] | None,
                   conf: dict, pos_mean: pd.Series) -> pd.DataFrame:
    """A player's games for another team say less about his role on this one.

    For every player with games for another team among his last
    ``long_games``, volume is recomputed with those games counting
    ``old_team_weight`` as much, and his volume scaled by the ratio of that
    to the same computation with every game counting fully -- so a player
    who has not moved is untouched. Efficiency (points per opportunity) is
    his own and keeps every game.

    Measured on 2022-2025, every player-week of a player with old-team games
    in his window (2,580 projected 5+): the model had them 0.9 a game too
    high, most of all in their first games for the new team (1.6 with none
    yet, 2.2 for a mid-season signing). Old-team games at a fifth of the
    weight was the best in every held-out season, average miss 3.13 -> 2.99.
    Zach Ertz, a practice-squad stopgap in Philadelphia, was being valued on
    his two seasons as Washington's starter.

    ``teams`` (nflverse id -> current team) overrides the team of his last
    game, so a player who has just signed somewhere is caught before he has
    played for them.
    """
    from ..teams import normalize_team

    w_old = float(conf.get("old_team_weight", 1.0))
    n_long = int(conf.get("long_games", 17))
    out = target.copy()
    out["vol_all_teams"] = out["vol"]
    current = out.set_index("player_id")["team"].map(normalize_team)
    for pid, team in (teams or {}).items():
        if pid in current.index and normalize_team(team):
            current[pid] = normalize_team(team)
        elif pid in current.index and team == FREE_AGENT:
            current[pid] = FREE_AGENT       # signing somewhere not yet named: every game is old-team
    out["team"] = out["player_id"].map(current).fillna(out["team"])
    last = prior[prior["player_id"].isin(out["player_id"])].sort_values(["season", "week"])
    last = last.groupby("player_id").tail(n_long).copy()
    if last.empty:
        return out
    last["_team"] = last["team"].map(normalize_team)
    last["same"] = last["_team"].to_numpy() == last["player_id"].map(current).to_numpy()
    last["age"] = last.groupby("player_id").cumcount(ascending=False)
    team_games = last.groupby("player_id")["same"].sum()
    out["team_games"] = out["player_id"].map(team_games)
    moved = last[~last["same"]]
    if moved.empty or w_old >= 1.0:
        return out
    out["old_team"] = out["player_id"].map(moved.groupby("player_id")["_team"].last())
    last = last[last["player_id"].isin(moved["player_id"].unique())]
    last["_exp"] = last["total_fantasy_points_exp"].fillna(last["fantasy_points_ppr"])

    def volume(w: float) -> pd.Series:
        return _window_volume(last, np.where(last["same"], 1.0, w), last["_exp"], conf, pos_mean)

    ratio = (volume(w_old) / volume(1.0)).replace([np.inf, -np.inf], np.nan)
    factor = out["player_id"].map(ratio).fillna(1.0).clip(0.0, 1.5)
    out["vol"] = out["vol"] * factor
    return out


def _returning(prior: pd.DataFrame, target: pd.DataFrame, back: dict[str, tuple[float, float]],
               conf: dict, pos_mean: pd.Series, season: int) -> pd.DataFrame:
    """A teammate back from an absence takes his work back.

    The games a player had this season while a regular at his position --
    one playing again now -- was out show more work than he will get with
    him back. Each is read less what he was expected to absorb then:
    ``bigger`` of the gap between them when the man coming back had the
    bigger role (the next man up handing the job back; the gap from the
    player's volume in the games they shared), ``smaller`` of the returning
    man's volume when he had the smaller one (``teammate_back`` in the
    config, per position). ``back`` -- nflverse id -> (chance he plays this
    week, share of the rest of the season he plays), from the injury tag --
    scales it: ``vol`` becomes this week's volume, ``vol_ros`` the rest of
    the season's. Measured on 2022-2025; see scripts/fit_teammate_back.py
    and DECISIONS.md, "A teammate back from an absence".

    Adds ``vol_ros``, ``vol_before_back`` and ``back_from`` (the teammate
    whose return moves him most).
    """
    from ..teams import normalize_team

    out = target.copy()
    out["vol_before_back"] = out["vol"]
    out["vol_ros"] = out["vol"]
    out["back_from"] = None
    rule = conf.get("teammate_back") or {}
    bigger, smaller = rule.get("bigger") or {}, rule.get("smaller") or {}
    positions = [q for q in rule.get("positions", []) if bigger.get(q) or smaller.get(q)]
    if not positions or not back or out.empty:
        return out
    regular = float(rule.get("regular", 5.0))
    pool = out[out["position"].isin(positions) & out["vol"].notna()]
    returning = pool[pool["player_id"].map(lambda q: max(back.get(q, (0.0, 0.0))) > 0)
                     & (pool["vol"] >= regular)]
    if returning.empty:
        return out
    cur = prior[prior["season"] == season]
    played = set(zip(cur["week"].astype(int), cur["team"].map(normalize_team), cur["player_id"]))
    this_season = set(zip(cur["team"].map(normalize_team), cur["player_id"]))
    groups: dict[tuple[str, str], list] = {}
    for r in returning.itertuples():
        # Only a teammate who has played for this team this season: a player
        # carried on a roster without a snap (Salvon Ahmed, last seen in 2023,
        # listed by Miami without an injury tag) is not one coming back.
        if (r.team, r.player_id) in this_season:
            groups.setdefault((r.team, r.position), []).append(r)
    if not groups:
        return out
    ids = pool[[(t, q) in groups for t, q in zip(pool["team"], pool["position"])]]["player_id"]
    n_long = int(conf.get("long_games", 17))
    last = prior[prior["player_id"].isin(ids)].sort_values(["season", "week"])
    last = last.groupby("player_id").tail(n_long).copy()
    if last.empty:
        return out
    last["_team"] = last["team"].map(normalize_team)
    last["age"] = last.groupby("player_id").cumcount(ascending=False)
    last["_exp"] = last["total_fantasy_points_exp"].fillna(last["fantasy_points_ppr"]).astype(float)
    team_now = out.set_index("player_id")["team"]
    w_old = float(conf.get("old_team_weight", 1.0))
    base = np.where(last["_team"].to_numpy() == last["player_id"].map(team_now).to_numpy(), 1.0, w_old)
    vol_now = out.set_index("player_id")["vol"]
    pid_arr, season_arr = last["player_id"].to_numpy(), last["season"].to_numpy()
    week_arr, team_arr = last["week"].to_numpy().astype(int), last["_team"].to_numpy()
    pos_arr = last["player_id"].map(out.set_index("player_id")["position"]).to_numpy()
    # Per game of each player's window: which returning teammates missed it.
    missing: list[list] = [[] for _ in range(len(last))]
    for x in range(len(last)):
        if season_arr[x] != season or team_arr[x] != team_now.get(pid_arr[x]):
            continue
        for r in groups.get((team_arr[x], pos_arr[x]), []):
            if r.player_id != pid_arr[x] and (week_arr[x], team_arr[x], r.player_id) not in played:
                missing[x].append(r)
    flagged = np.array([bool(m) for m in missing])
    if not flagged.any():
        return out
    sel = last["player_id"].isin(last["player_id"][flagged].unique()).to_numpy()
    sub, base_s, miss_s = last[sel], base[sel], [m for m, k in zip(missing, sel) if k]
    flag_s = flagged[sel]
    exp = sub["_exp"].to_numpy()
    full = _window_volume(sub, base_s, exp, conf, pos_mean)
    shared = _window_volume(sub, np.where(flag_s, 0.0, base_s), exp, conf, pos_mean)
    has_shared = pd.Series(~flag_s, index=sub.index).groupby(sub["player_id"]).any()
    # His volume in the games they shared: what the gap is measured from.
    with_them = (vol_now.reindex(full.index) * shared / full).where(has_shared.reindex(full.index), vol_now)
    cut_week, cut_ros = np.zeros(len(sub)), np.zeros(len(sub))
    lead: dict[str, tuple[float, str]] = {}
    for x, (pid, ms) in enumerate(zip(sub["player_id"].to_numpy(), miss_s)):
        mine = float(with_them.get(pid, np.nan))
        for r in ms:
            vj = float(r.vol)
            if not np.isfinite(mine):
                continue
            took = (float(bigger.get(r.position, 0.0)) * (vj - mine) if vj > mine
                    else float(smaller.get(r.position, 0.0)) * vj)
            pw, pr = back.get(r.player_id, (0.0, 0.0))
            cut_week[x] += pw * took
            cut_ros[x] += pr * took
            if took * pr > lead.get(pid, (0.0, ""))[0]:
                lead[pid] = (took * pr, str(r.player_display_name))
    f_week = (_window_volume(sub, base_s, np.maximum(exp - cut_week, 0.0), conf, pos_mean) / full)
    f_ros = (_window_volume(sub, base_s, np.maximum(exp - cut_ros, 0.0), conf, pos_mean) / full)
    f_week = f_week.replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)
    f_ros = f_ros.replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)
    out["vol"] = out["vol"] * out["player_id"].map(f_week).fillna(1.0)
    out["vol_ros"] = out["vol_before_back"] * out["player_id"].map(f_ros).fillna(1.0)
    out["back_from"] = out["player_id"].map({pid: name for pid, (_c, name) in lead.items()})
    return out


def player_table(history: pd.DataFrame, season: int, week: int, cfg: Config,
                 teams: dict[str, str] | None = None,
                 back: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """Per nflverse player: the projection as of (season, week), before any
    matchup scaling, with the volume and efficiency behind it. ``teams``
    (nflverse id -> current team) says who plays where now, when a platform
    knows it before the box scores do. ``back`` (nflverse id -> chance he
    plays this week, share of the rest of the season he plays) lets a
    teammate back from an absence take his work back (:func:`_returning`):
    ``vol`` is then this week's volume and ``vol_ros`` the season's."""
    conf = cfg.raw["roster"]
    n_games = int(conf["trailing_games"])
    long_games = int(conf.get("long_games", 17))
    halflife = float(conf.get("volume_halflife_games", 2.5))
    if history.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    prior = history[
        (history["season"] < season) | ((history["season"] == season) & (history["week"] < week))
    ]
    if prior.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    # Shrinkage targets from completed games only: the target week must not
    # inform its own projection.
    pos_mean_all, pos_eff_all = position_rates(prior)

    # Append one placeholder row per player for the target week so the trailing
    # window lands on exactly the games before it.
    last = prior.sort_values(["season", "week"]).groupby("player_id").tail(1)
    placeholder = last[["player_id", "player_display_name", "position", "team"]].copy()
    placeholder["season"], placeholder["week"] = season, week
    placeholder["fantasy_points_ppr"] = np.nan
    placeholder["total_fantasy_points_exp"] = np.nan
    stacked = pd.concat([prior, placeholder], ignore_index=True)
    stacked = stacked.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    feats = _trailing_features(stacked, n_games, long_games, halflife)
    target = feats[(feats["season"] == season) & (feats["week"] == week)].copy()
    pos_mean = target["position"].map(pos_mean_all)
    pos_eff = target["position"].map(pos_eff_all)
    target["vol"], target["eff"] = _volume_efficiency(target, pos_mean, pos_eff, conf)
    target["team_games"], target["old_team"], target["vol_all_teams"] = np.nan, None, target["vol"]
    target = _team_discount(prior, target, teams, conf, pos_mean_all)
    target = _returning(prior, target, back or {}, conf, pos_mean_all, season)
    target["blend"] = target["vol"] * target["eff"]
    target = target[target["blend"].notna()]
    # Opportunity over the last two games against the games before them: the
    # "his role changed" signal shown next to a waiver pick.
    target["recent_exp"] = target["e_2"] / target["c_2"].where(target["c_2"] >= 2)
    before_n = (target["c_long"] - target["c_2"]).where(lambda x: x >= 3)
    target["prior_exp"] = (target["e_long"] - target["e_2"]) / before_n
    target = target.merge(
        last[["player_id", "season", "week"]].rename(columns={"season": "last_season", "week": "last_week"}),
        on="player_id", how="left",
    )
    first = prior.groupby("player_id")["season"].min().rename("first_season")
    target = target.merge(first, left_on="player_id", right_index=True, how="left")
    return target[TABLE_COLUMNS]


def recent_usage(history: pd.DataFrame, season: int, week: int, n: int = 3) -> pd.DataFrame:
    """Targets, carries and target share per game over each player's last
    ``n`` games before (season, week). Shown beside waiver and lottery picks
    as the evidence for their opportunity; the projection itself already
    prices it through expected points (raw counts, target share and catch
    rate over expected added nothing on top, walk-forward 2022-2025)."""
    cols = [c for c in ("targets", "carries", "target_share", "xtd") if c in history.columns]
    if not cols or history.empty:
        return pd.DataFrame(columns=cols)
    prior = history[(history["season"] < season) | ((history["season"] == season) & (history["week"] < week))]
    last = prior.sort_values(["season", "week"]).groupby("player_id").tail(n)
    out = last.groupby("player_id")[cols].mean()
    out["games"] = last.groupby("player_id").size()
    return out


def usage_line(row) -> str:
    """'7.3 targets (22% share), 1.0 carries a game over his last 3'."""
    bits = []
    tgt, car = float(row.get("targets", 0) or 0), float(row.get("carries", 0) or 0)
    share = row.get("target_share")
    if tgt >= 0.5:
        bits.append(f"{tgt:.1f} targets" + (f" ({float(share):.0%} share)" if share == share and share else ""))
    if car >= 0.5:
        bits.append(f"{car:.1f} carries")
    # Expected touchdowns are the red-zone role: sticky (split-half r ~0.7),
    # unlike converting above expectation (r ~0.05), which is not shown.
    xtd = row.get("xtd")
    if xtd == xtd and xtd is not None and float(xtd) >= 0.25:
        bits.append(f"{float(xtd):.2f} expected TDs")
    if not bits:
        return ""
    games = int(row.get("games", 0))
    if games <= 1:
        return f"{', '.join(bits)} in his only game so far"
    return f"{', '.join(bits)} a game over his last {games}"


def weeks_since(row, season: int, week: int, weeks_per_season: int = 18) -> int:
    """Calendar weeks between a player's last game and the target week."""
    try:
        ls, lw = int(row["last_season"]), int(row["last_week"])
    except (KeyError, TypeError, ValueError):
        return 0
    return (int(season) - ls) * weeks_per_season + (int(week) - lw)


def sd_table(history: pd.DataFrame, cfg: Config) -> dict[tuple[str, int], float]:
    """Residual sd of the projection by (position, projection bucket), from history."""
    conf = cfg.raw["roster"]
    if history.empty:
        return {}
    feats = _trailing_features(history, int(conf["trailing_games"]), int(conf.get("long_games", 17)),
                               float(conf.get("volume_halflife_games", 2.5)))
    pos_mean, pos_eff = position_rates(history)
    feats["blend"] = _blend(feats, feats["position"].map(pos_mean), feats["position"].map(pos_eff), conf)
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


def fit_platform_weight(cfg: Config, history: pd.DataFrame) -> tuple[float, int]:
    """How far our model's number moves toward the platform's: least squares on
    this league's logged healthy players (both numbers are if-he-plays) against
    the points they scored, shrunk toward ``platform_projection_weight`` with
    ``platform_weight_prior_games`` of weight, and the prior alone below
    ``platform_weight_min_games``. Returns (weight, graded player-weeks)."""
    conf = cfg.raw["roster"]
    prior = float(conf.get("platform_projection_weight", 0.5))
    k = float(conf.get("platform_weight_prior_games", 300))
    need = int(conf.get("platform_weight_min_games", 150))
    path = cfg.results_dir / "skill_log.parquet"
    if not path.exists() or history is None or history.empty:
        return prior, 0
    log_ = pd.read_parquet(path)
    if not {"model_projection", "platform_projection", "nfl_id"} <= set(log_.columns):
        return prior, 0
    log_ = log_[(log_["status"].fillna("") == "") & log_["platform_projection"].notna()
                & (log_["platform_projection"] > 0) & log_["model_projection"].notna() & log_["nfl_id"].notna()]
    actual = history[["player_id", "season", "week", "fantasy_points_ppr"]].rename(columns={"player_id": "nfl_id"})
    j = log_.merge(actual, on=["nfl_id", "season", "week"], how="inner")
    n = len(j)
    if n < need:
        return prior, n
    m, p, y = (j["model_projection"].to_numpy(float), j["platform_projection"].to_numpy(float),
               j["fantasy_points_ppr"].to_numpy(float))
    den = float(((p - m) ** 2).sum())
    if den <= 0:
        return prior, n
    fit = min(max(float(((y - m) * (p - m)).sum()) / den, 0.0), 1.0)
    return (n * fit + k * prior) / (n + k), n


def practice_reports(season: int, week: int, cfg: Config, allow_network: bool) -> dict[str, str]:
    """nflverse id -> his last practice this week ("DNP", "Limited", "Full"),
    for players whose final report -- the one that carries a game status --
    is out. Practice alone before then is a partial week."""
    path = cfg.raw_dir / f"injuries_{int(season)}.parquet"
    if allow_network:
        from ..data.nflverse import load_injuries

        inj = load_injuries(season, cfg)
    elif path.exists():
        inj = pd.read_parquet(path)
    else:
        return {}
    if inj is None or inj.empty or "practice_status" not in inj.columns:
        return {}
    wk = inj[(inj["week"] == int(week)) & inj["report_status"].isin(["Questionable", "Doubtful", "Out"])]
    if "date_modified" in wk.columns:
        wk = wk.sort_values("date_modified")
    out = {}
    for gid, prac in zip(wk["gsis_id"], wk["practice_status"]):
        text = str(prac or "")
        tag = "DNP" if "Did Not" in text else "Limited" if "Limited" in text else "Full" if "Full" in text else ""
        if gid and tag:
            out[str(gid)] = tag
    return out


def _play_probability(player: PlayerRow, cfg: Config) -> float:
    """Chance the player suits up this week, from his injury tag."""
    conf = cfg.raw["roster"]
    if player.is_out:
        return 0.0
    if player.status in ("DOUBTFUL", "D"):
        return float(conf["doubtful_play_probability"])
    if player.is_questionable:
        by = (conf.get("questionable_by_practice") or {})
        row = by.get(player.position, by.get("default")) if player.practice else None
        if row and player.practice in row:
            return float(row[player.practice])
        q = conf["questionable_play_probability"]
        if isinstance(q, dict):
            return float(q.get(player.position, q.get("default", 0.72)))
        return float(q)
    return 1.0


def _status_adjust(mean: float, sd: float, player: PlayerRow, cfg: Config) -> tuple[float, float]:
    """Scale a projection by the chance the player actually plays."""
    p = _play_probability(player, cfg)
    if p <= 0.0:
        return 0.0, 0.0
    if p >= 1.0:
        return mean, sd
    # Mixture of "plays" (mean, sd) and "does not" (0, 0).
    mix_mean = p * mean
    mix_var = p * (sd ** 2 + mean ** 2) - mix_mean ** 2
    return mix_mean, float(np.sqrt(max(mix_var, 0.0)))


# ---------------------------------------------------------------------------
# The next man up
# ---------------------------------------------------------------------------
def _absence(player: PlayerRow, sits: bool, cfg: Config) -> tuple[float, float, str]:
    """(this week, rest of season) weight that the player is missing, and how
    to say it. A bye is not an absence: his role is waiting for him."""
    if player.on_bye:
        return 0.0, 0.0, ""
    status = player.status
    if status in LONG_TERM_OUT_STATUSES:
        if status.startswith("SUSP"):
            label = "is suspended"
        elif status.startswith("PUP"):
            label = "is on the PUP list"
        elif status == "NFI":
            label = "is on the NFI list"
        else:
            label = "is on IR"
        return 1.0, 1.0, label
    if status in OUT_STATUSES:
        return 1.0, 0.3, "is out"
    if sits:
        return 1.0, 0.3, "is not expected to play"
    p = _play_probability(player, cfg)
    if p < 1.0:
        label = "is doubtful" if status in ("DOUBTFUL", "D") else "is questionable"
        return 1.0 - p, 0.5 * (1.0 - p) * 0.3, label
    return 0.0, 0.0, ""


def next_man_up(
    players: list[PlayerRow],
    mapping: dict[str, str],
    table: pd.DataFrame,
    sits: set[str],
    season: int,
    week: int,
    cfg: Config,
    snaps: pd.DataFrame | None = None,
) -> dict[str, tuple[float, float, str]]:
    """Opportunity inherited from absent teammates: platform id ->
    (extra volume this week, extra volume rest of season, reason).
    ``snaps`` (``depth.prepare``d snap counts) set each backup's share by how
    much of his position's other snaps he already had.

    When a regular misses a game, the teammate at his position with the most
    opportunity takes over part of the gap between them. Measured on every
    first game of an absence 2021-2025: the next-man-up running back
    averaged 13.9 points against 10.7 for backs projected the same with no
    absence, and scored 20+ a quarter of the time. The share of the gap that
    predicts his first game best (squared error, each season held out in
    turn) is ~0.55 for RBs, ~0.6 for QBs, ~0.35 for TEs (``next_man_up`` in
    the config); receivers spread a missing WR's targets too thinly to help
    any one of them, so they get none.
    """
    conf = cfg.raw["roster"]
    takeover = {k: float(v) for k, v in (conf.get("next_man_up") or {}).items()}
    inactive_weeks = int(conf.get("inactive_weeks", 4))
    if table.empty or not takeover:
        return {}
    by_id = table.set_index("player_id")
    team_of: dict[str, str] = {}
    platform_of: dict[str, PlayerRow] = {}
    blocked: set[str] = set()      # cannot inherit: missing, or likely missing
    absent: dict[tuple[str, str], list[tuple[float, float, float, str, str]]] = {}
    for p in players:
        if p.position not in SKILL:
            continue
        nid = mapping.get(p.player_id)
        if nid is None:
            continue
        if p.team:
            team_of[nid] = p.team
        platform_of[nid] = p
        w_week, w_ros, label = _absence(p, p.player_id in sits, cfg)
        if w_week > 0 or w_ros > 0:
            blocked.add(nid)
        if (w_week <= 0 and w_ros <= 0) or not p.team or nid not in by_id.index:
            continue
        row = by_id.loc[nid]
        # Long gone: his work was redistributed weeks ago and is already in
        # everyone else's recent numbers.
        if weeks_since(row, season, week) > inactive_weeks:
            continue
        absent.setdefault((p.team, p.position), []).append(
            (w_week, w_ros, float(row["vol"]), p.name, label))

    out: dict[str, tuple[float, float, str]] = {}
    fits = depth.load()
    for (team, pos), missing in absent.items():
        share = takeover.get(pos, 0.0)
        if share <= 0:
            continue
        pool = by_id[by_id["position"] == pos]
        best_nid, best_vol, best_key = None, -1.0, None
        for nid, row in pool.iterrows():
            if team_of.get(nid, row["team"]) != team or nid in blocked:
                continue
            if weeks_since(row, season, week) > inactive_weeks:
                continue
            # The teammate with the most work: the share he inherits was
            # measured on who actually took the snaps, and a depth chart can
            # lag them (Miami listed Jaylen Wright ahead of Ollie Gordon II
            # while Gordon had the backfield).
            if best_key is None or float(row["vol"]) > best_vol:
                best_nid, best_vol, best_key = nid, float(row["vol"]), True
        heir = platform_of.get(best_nid) if best_nid is not None else None
        if heir is None:
            continue
        gap_week = sum(w * max(0.0, v - best_vol) for w, _r, v, _n, _l in missing)
        gap_ros = sum(r * max(0.0, v - best_vol) for _w, r, v, _n, _l in missing)
        if gap_week <= 0 and gap_ros <= 0:
            continue
        lead = max(missing, key=lambda m: m[0] * m[2])
        # How much of the job he gets depends on who else is there: his share
        # of the position's other snaps (a lone backup takes most of it, one
        # of three takes little). See depth.py and next_man_up.json.
        why = f"next man up: {lead[3]} {lead[4]}"
        conc = None
        if snaps is not None and not snaps.empty and pos in fits:
            conc = depth.concentration(snaps, season, week, team, pos, str(by_id.loc[best_nid, "player_display_name"]),
                                       lead[3])
        if conc is not None:
            share = depth.share(pos, conc, share, fits)
            heir.backfield_share, heir.takeover_share = round(conc, 3), round(share, 3)
            why += f", and {conc:.0%} of his team's other {pos} snaps have been his"
        out[heir.player_id] = (share * gap_week, share * gap_ros, why)
    return out


def _season_snaps(season: int, cfg: Config, allow_network: bool) -> pd.DataFrame | None:
    """This season's snap counts, ready for :func:`depth.concentration`: from
    the cache when offline, else through the loader (re-pulled when stale)."""
    path = cfg.raw_dir / f"snap_counts_{season}.parquet"
    try:
        if allow_network:
            from ..data.nflverse import load_snap_counts

            frame = load_snap_counts([season], cfg)
        elif path.exists():
            frame = pd.read_parquet(path)
        else:
            return None
    except Exception as exc:  # noqa: BLE001 - snaps refine the next man up; never block a projection
        log.warning("snap counts unavailable: %s", exc)
        return None
    if frame is None or frame.empty:
        return None
    if "game_type" in frame.columns:
        frame = frame[frame["game_type"] == "REG"]
    from ..teams import normalize_team_series

    frame = frame.assign(team=normalize_team_series(frame["team"]))
    return depth.prepare(frame)


def _opponents(snapshot: LeagueSnapshot, cfg: Config) -> dict[str, str]:
    """NFL team -> this week's opponent."""
    try:
        games = games_frame(cfg)
    except Exception as exc:  # noqa: BLE001 - correlations degrade to none
        log.warning("no schedule for opponents: %s", exc)
        return {}
    wk = games[(games["season"] == snapshot.season) & (games["week"] == snapshot.week)]
    return dict(zip(wk["team"], wk["opponent"]))


def next_in_line(members: list[PlayerRow]) -> list[PlayerRow]:
    """A position group in order: the lead by season value; then a backup
    already standing in for an absent starter (chosen on the work he is
    getting); then the rest in depth-chart order where it is known (a No. 3
    is not the handcuff because he once had the job), else by value."""
    value = lambda q: float(q.ros_value if q.ros_value is not None else (q.projection or 0.0))  # noqa: E731
    ranked = sorted(members, key=lambda q: -value(q))
    if len(ranked) < 2:
        return ranked
    standing_in = lambda q: float(q.inherited_ros or 0.0) > 0 or q.takeover_share is not None  # noqa: E731
    rest = sorted(ranked[1:], key=lambda q: (not standing_in(q), q.depth_order is None, q.depth_order or 0,
                                              -value(q)))
    return [ranked[0], *rest]


def attach_depth(players: list[PlayerRow], orders: dict[tuple[str, str, str], int]) -> int:
    """Set each QB/RB/TE's depth-chart order where Sleeper has him on the same
    team at the same position. Returns how many were matched."""
    from ..teams import normalize_team
    from .players import normalize_name

    n = 0
    for p in players:
        p.depth_order = None
        if p.team and p.position in ("QB", "RB", "TE"):
            p.depth_order = orders.get((normalize_team(p.team), p.position, normalize_name(p.name)))
            n += p.depth_order is not None
    return n


def assign_roles(players: list[PlayerRow]) -> None:
    """Depth-chart role per player -- QB, RB1, RB2, WR1..., TE -- by healthy
    per-game projection among his NFL teammates in the player pool; behind the
    lead, QBs, RBs and TEs follow the real depth chart where it is known."""
    groups: dict[tuple[str, str], list[PlayerRow]] = {}
    for p in players:
        if p.position in ("DST", "K"):
            p.role = p.position
        elif p.position in SKILL and p.team:
            groups.setdefault((p.team, p.position), []).append(p)
    for (_team, pos), members in groups.items():
        if pos in ("QB", "RB", "TE"):
            members = next_in_line(members)
        else:
            members.sort(key=lambda q: -float(q.ros_value if q.ros_value is not None else (q.projection or 0.0)))
        for rank, p in enumerate(members, start=1):
            if pos in ("QB", "TE"):
                p.role = pos if rank == 1 else f"{pos}{rank}"
            else:
                p.role = f"{pos}{rank}"


def games_missed(history: pd.DataFrame, season: int, week: int) -> tuple[dict[str, list[int]], dict[str, set[int]]]:
    """This season before ``week``: the weeks each team played, and the
    weeks each player played -- enough to count how many of his team's
    games in a row a player has missed."""
    from ..teams import normalize_team

    cur = history[(history["season"] == season) & (history["week"] < week)]
    if cur.empty:
        return {}, {}
    team_weeks = {normalize_team(t): sorted(set(int(w) for w in g))
                  for t, g in cur.groupby("team")["week"]}
    played = {str(pid): set(int(w) for w in g) for pid, g in cur.groupby("player_id")["week"]}
    return team_weeks, played


def _missed_in_a_row(team_weeks: list[int], played: set[int]) -> int:
    n = 0
    for w in reversed(team_weeks):
        if w in played:
            break
        n += 1
    return n


def _back_signal(row, ros: float) -> str:
    """A teammate back from an absence: his season value with the games that
    teammate missed read down, against without."""
    who = row.get("back_from")
    if not isinstance(who, str) or not who:
        return ""
    try:
        cut = (float(row["vol_before_back"]) - float(row["vol_ros"])) * float(row["eff"])
    except (KeyError, TypeError, ValueError):
        return ""
    if not np.isfinite(cut) or cut < 0.3:
        return ""
    words = [w for w in who.split() if w.rstrip(".").upper() not in ("JR", "SR", "II", "III", "IV", "V")]
    return (f"{who} is back: the games {(words or [who])[-1]} missed count for less "
            f"({ros + cut:.1f} -> {ros:.1f} a game)")


def _moved_signal(row) -> str:
    """A new team: how far his old role was discounted."""
    old = row.get("old_team")
    if not isinstance(old, str) or not old:
        return ""
    try:
        now = float(row["vol_before_back"]) if pd.notna(row.get("vol_before_back")) else float(row["vol"])
        before, games = float(row["vol_all_teams"]), int(row["team_games"])
        eff = float(row["eff"])
    except (KeyError, TypeError, ValueError):
        return ""
    if not (np.isfinite(now) and np.isfinite(before)) or abs(before - now) * eff < 0.3:
        return ""
    return (f"new team: {games} game{'s' if games != 1 else ''} for {row['team']} so far, so his {old} "
            f"role counts for less ({before * eff:.1f} -> {now * eff:.1f} a game)")


def _trend_signal(row) -> str:
    """Opportunity over his last two games against the games before them."""
    try:
        recent, before = float(row["recent_exp"]), float(row["prior_exp"])
    except (KeyError, TypeError, ValueError):
        return ""
    if not (np.isfinite(recent) and np.isfinite(before)):
        return ""
    if recent >= 1.4 * before and recent - before >= 3.0:
        return f"opportunity up: {recent:.1f} expected pts/game over his last 2 vs {before:.1f} before"
    if recent <= 0.65 * before and before - recent >= 3.0:
        return f"opportunity down: {recent:.1f} expected pts/game over his last 2 vs {before:.1f} before"
    return ""


def _collapse_factor(row, p: PlayerRow, ros: float, conf: dict) -> float:
    """The share of his season value to keep when his opportunity has
    collapsed: last two games under ``below`` of the games before, by 3+
    expected points a game, at the configured positions and value."""
    rule = conf.get("usage_collapse") or {}
    if p.position not in rule.get("positions", []) or ros < float(rule.get("min_value", 0.0)):
        return 1.0
    if p.is_out or p.is_long_term_out:
        return 1.0                    # an injury, not a lost job: the absence is simulated
    try:
        recent, before = float(row["recent_exp"]), float(row["prior_exp"])
    except (KeyError, TypeError, ValueError):
        return 1.0
    if not (np.isfinite(recent) and np.isfinite(before)) or before - recent < 3.0:
        return 1.0
    return float(rule.get("factor", 1.0)) if recent < float(rule.get("below", 0.0)) * before else 1.0


def project_snapshot(
    snapshot: LeagueSnapshot,
    cfg: Config | None = None,
    rankings=None,
    allow_network: bool = True,
    history: pd.DataFrame | None = None,
    now=None,
) -> ProjectionReport:
    """Fill ``projection``/``projection_sd``/``ros_value`` on every player in
    place, and lock in the scores of games already played."""
    cfg = cfg or get_config()
    conf = cfg.raw["roster"]
    report = ProjectionReport()

    history = load_history(cfg) if history is None else history
    sds = sd_table(history, cfg)
    scale, report.line_source = _implied_scale(snapshot, cfg, allow_network)
    damping = float(conf["vegas_damping"])
    try:
        w_platform, report.platform_games = fit_platform_weight(cfg, history)
    except Exception as exc:  # noqa: BLE001 - fall back to the prior
        log.warning("platform weight fit failed: %s", exc)
        w_platform, report.platform_games = float(conf["platform_projection_weight"]), 0
    report.platform_weight = w_platform
    inactive_weeks = int(conf.get("inactive_weeks", 4))
    long_gone_weeks = int(conf.get("long_gone_weeks", 18))
    first_known = int(history["season"].min()) if not history.empty else snapshot.season

    # nflverse index for name matching: newest team per player.
    latest = history.sort_values(["season", "week"]).groupby("player_id").tail(1)
    index = build_index(latest[["player_id", "player_display_name", "position", "team"]])
    players = snapshot.all_players()
    matched = match_players([p for p in players if p.position in SKILL], index)
    report.unmatched = matched.unmatched
    # The platform knows who plays where before the box scores do.
    current_teams = {matched.mapping[p.player_id]: p.team for p in players
                     if p.team and p.player_id in matched.mapping}
    # A free agent the news has close to signing plays for his new team (or
    # one not yet named), not his old one (``data.news``).
    for p in players:
        if p.signing_odds is not None and p.player_id in matched.mapping:
            current_teams[matched.mapping[p.player_id]] = p.signing_team or FREE_AGENT
    pos_mean = history.groupby("position")["fantasy_points_ppr"].mean() if not history.empty else pd.Series(dtype=float)

    # D/ST and K from the streaming model.
    dst_rows: dict[str, pd.Series] = {}
    k_rows: dict[str, pd.Series] = {}
    if rankings is not None:
        if rankings.dst is not None and not rankings.dst.empty:
            dst_rows = {r.team: r for r in rankings.dst.itertuples()}
        if rankings.kicker is not None and not rankings.kicker.empty:
            k_rows = {r.team: r for r in rankings.kicker.itertuples()}

    active = [p for t in snapshot.teams for p in t.roster
              if p.team and not p.on_bye and not p.in_ir_slot and p.position in SKILL]
    projected = [p for p in active if (p.platform_projection or 0) > 0]
    platform_covers = len(active) >= 20 and len(projected) / len(active) >= 0.8

    # A platform projection of exactly zero, for a player on an NFL team and
    # not on bye, is the platform saying he will not play THIS week --
    # suspended, on the commissioner's exempt list, a doubtful tag it has
    # already resolved. ESPN has no status for some of these (it listed Josh
    # Jacobs, exempt, as DAY_TO_DAY), so the zero is the signal. It is trusted
    # only when the platform projects nearly everyone, so an unpopulated feed
    # cannot bench a whole roster. It says nothing about the rest of the
    # season, so rest-of-season value is kept.
    # His last practice, once the final injury report is out: a questionable
    # player who practised fully plays far more often than one who did not.
    try:
        practiced = practice_reports(int(snapshot.season), int(snapshot.week), cfg, allow_network)
    except Exception as exc:  # noqa: BLE001 - a refinement; never block a projection
        log.warning("practice reports unavailable: %s", exc)
        practiced = {}
    for p in players:
        nid = matched.mapping.get(p.player_id)
        p.practice = practiced.get(str(nid), "") if nid is not None else ""
    sits = {
        p.player_id for p in players
        if p.platform_projection is not None and p.platform_projection <= 0 and platform_covers
        and bool(p.team) and not p.on_bye and not p.in_ir_slot
    }
    # Who is playing: a teammate back from an absence takes his work back
    # (``_returning``), at his chance of playing this week and his share of
    # the rest of the season -- the same reading of tags as the next man up.
    back: dict[str, tuple[float, float]] = {}
    for p in players:
        nid = matched.mapping.get(p.player_id)
        if nid is not None and p.position in SKILL:
            w_week, w_ros, _label = _absence(p, p.player_id in sits, cfg)
            back[nid] = (1.0 - w_week, 1.0 - w_ros)
    table = player_table(history, snapshot.season, snapshot.week, cfg, teams=current_teams, back=back)
    by_id = table.set_index("player_id") if not table.empty else pd.DataFrame()
    for p in players:
        p.takeover_share = p.backfield_share = None
    # The real depth chart decides who is next in line behind a lead.
    from ..data import sleeper

    try:
        report.depth_matched = attach_depth(players, sleeper.depth_orders(cfg, allow_network))
    except Exception as exc:  # noqa: BLE001 - a bonus source; without it the order is by value
        log.warning("Sleeper depth charts unavailable: %s", exc)
    heirs = next_man_up(players, matched.mapping, table, sits, snapshot.season, snapshot.week, cfg,
                        snaps=_season_snaps(int(snapshot.season), cfg, allow_network))
    usage = recent_usage(history, snapshot.season, snapshot.week)
    team_weeks, played_weeks = games_missed(history, int(snapshot.season), int(snapshot.week))
    # Where our season value has been measured to overreact, defer toward
    # the market (see market_calibration: two early-season cells).
    market_conf = market_calibration.load()
    market_lines = market_calibration.season_lines(history, int(snapshot.season), int(snapshot.week)) \
        if market_conf.get("cells") else {}

    for p in players:
        mean: float | None = None
        model_mean: float | None = None
        sd: float | None = None
        ros: float | None = None
        source = ""
        p.signals = []
        p.usage = ""
        p.inherited_ros = 0.0
        p.games_missed = 0
        p.experience = None
        platform_says_sits = p.player_id in sits
        plat = p.platform_projection
        if plat is not None and plat <= 0:
            plat = None

        if p.position in SKILL:
            nfl_id = matched.mapping.get(p.player_id)
            # An absence in progress: how many of his team's games in a row
            # he has already missed (the simulation lets a long one run long).
            from ..teams import normalize_team

            tw = team_weeks.get(normalize_team(p.team)) if p.team else None
            if tw and nfl_id is not None:
                p.games_missed = _missed_in_a_row(tw, played_weeks.get(str(nfl_id), set()))
                if p.games_missed >= 2 and (platform_says_sits or p.is_out or p.is_long_term_out) \
                        and not p.on_bye:
                    p.signals.append(f"has missed his team's last {p.games_missed} games: an absence that has "
                                     "run this long tends to run longer")
            row = by_id.loc[nfl_id] if (nfl_id is not None and not by_id.empty
                                        and nfl_id in by_id.index) else None
            stale = row is not None and weeks_since(row, snapshot.season, snapshot.week) > inactive_weeks
            # No NFL team means not on an NFL roster: an unsigned or retired
            # player cannot play, whatever injury tag he still carries. ESPN
            # lists 240-odd of them in a free-agent pool and projects one;
            # Yahoo tags them NA and keeps their old team.
            unsigned = p.unsigned
            # Gone for more than a season -- a holdout, the reserve/left-squad
            # list, a long suspension -- is not an injury that heals in a few
            # weeks, whatever tag he carries: his old games say nothing about
            # this season. Brandon Aiyuk (last game October 2024, listed OUT)
            # kept an 11.9-a-game value and was recommended as a pickup.
            gone = row is not None and weeks_since(row, snapshot.season, snapshot.week) > long_gone_weeks
            signing = p.signing_odds is not None and row is not None
            if gone and plat is None and not signing:
                mean, ros, source = 0.0, 0.0, "inactive"
                p.signals.append(f"has not played since {int(row['last_season'])} week "
                                 f"{int(row['last_week'])}: no value until he is back on the field")
            # An injured player on a roster carries a status: he is real and
            # his history still speaks to rest-of-season value (this week is
            # zeroed by the status adjustment). A stale player with no status,
            # or any unsigned one, needs a platform projection to count.
            elif row is not None and (not unsigned or signing) and (not stale or p.status or signing):
                vol, eff = float(row["vol"]), float(row["eff"])
                vol_ros = float(row.get("vol_ros", vol)) if pd.notna(row.get("vol_ros", vol)) else vol
                extra_week, extra_ros, why = heirs.get(p.player_id, (0.0, 0.0, ""))
                first = row.get("first_season")
                if first is not None and np.isfinite(first) and int(first) > first_known:
                    p.experience = int(snapshot.season) - int(first)
                ros = (vol_ros + extra_ros) * eff + outcome.youth_drift(p.experience)
                returned = _back_signal(row, ros)
                if returned:
                    p.signals.append(returned)
                p.inherited_ros = round(extra_ros * eff, 2)
                adj = market_calibration.calibrate(p.position, int(snapshot.week), ros, nfl_id,
                                                   int(snapshot.season), market_lines, market_conf) \
                    if market_lines else None
                if adj is not None:
                    if abs(adj.value - ros) >= 0.3:
                        p.signals.append(f"season value tempered from {ros:.1f} to {adj.value:.1f}: {adj.note}")
                    ros = adj.value
                s = scale.get(p.team, 1.0) if p.team else 1.0
                mean = (vol + extra_week) * eff * (s ** damping)
                source = "model"
                # Only worth saying when it moves him: a questionable starter
                # barely shifts his backup, and "+0.0" is noise.
                if why and max(extra_week, extra_ros) * eff >= 0.3:
                    p.signals.append(f"{why} (+{extra_week * eff:.1f} projected)" if extra_week * eff >= 0.3
                                     else f"{why} (+{extra_ros * eff:.1f} a game rest of season)")
                moved = _moved_signal(row)
                if moved:
                    p.signals.append(moved)
                trend = _trend_signal(row)
                if trend:
                    p.signals.append(trend)
                cut = _collapse_factor(row, p, ros, conf)
                if cut < 1.0:
                    p.signals.append(f"season value lowered from {ros:.1f} to {ros * cut:.1f}: his work has "
                                     f"collapsed, and backs like this returned about {cut:.0%} of their value "
                                     "(2022-2025)")
                    ros *= cut
                if nfl_id in usage.index:
                    p.usage = usage_line(usage.loc[nfl_id])
            elif (stale or unsigned) and plat is None:
                # History windows count games played, not time elapsed, so a
                # player who has not taken a snap in two seasons would keep his
                # old average. With nothing current -- no recent game and no
                # platform projection -- he is not playing.
                mean, ros, source = 0.0, 0.0, "inactive"
                if unsigned:
                    p.signals = ["not on an NFL roster: no value until he signs with a team"]
            elif plat is not None:
                mean = float(plat)
                ros = mean
                source = "platform"
            else:
                # Unknown player: a heavily discounted position average.
                mean = 0.5 * float(pos_mean.get(p.position, 8.0)) if len(pos_mean) else 4.0
                ros = mean
                source = "prior"
            model_mean = mean
            if plat is not None and source == "model" and w_platform > 0:
                mean = (1 - w_platform) * mean + w_platform * float(plat)
                source = "model+platform"
            sd = lookup_sd(sds, p.position, mean)
            p.ros_sd = round(outcome.drift_sd(p.position, ros or 0.0, p.experience), 2)

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
            if plat is not None:
                mean, sd, ros, source = float(plat), 4.5, float(plat), "platform"
            else:
                mean, sd, ros, source = 6.0, 4.5, 6.0, "prior"

        if mean is None:
            continue
        p.outcome_sd = round(sd or 0.0, 2)
        p.play_probability = _play_probability(p, cfg)
        if p.is_questionable and p.practice and not p.is_out:
            words = {"DNP": "no practice", "Limited": "a limited practice", "Full": "a full practice"}
            p.signals.append(f"questionable after {words[p.practice]}: {p.play_probability:.0%} to play "
                             "(questionable players like him, 2022-2025)")
        own = model_mean if model_mean is not None else mean
        own, _own_sd = _status_adjust(own, sd or 0.0, p, cfg)
        mean, sd = _status_adjust(mean, sd or 0.0, p, cfg)
        if p.signing_odds is not None and p.position in SKILL:
            # Not on a roster yet: nothing this week; his season is his value
            # when he plays, at the chance and timing the simulation draws.
            own = mean = sd = 0.0
            p.play_probability = 0.0
            if ros is not None and p.news:
                where = f"for {p.signing_team}" if p.signing_team else "on a new team"
                p.signals = [f"{p.news}; priced at {ros:.1f} a game {where} when he plays, "
                             f"{p.signing_odds:.0%} to play this season, about {p.signing_share or 0:.0%} "
                             "of the rest of it"] + [q for q in p.signals if not any(
                                   w in q for w in ("no value until", "not on an NFL roster", "has missed his team"))]
        if platform_says_sits and p.position not in ("DST", "K"):
            own = mean = sd = 0.0
            p.play_probability = 0.0
            source = f"{source}/sits"
            report.sitting.append(p.name)
        p.model_projection = round(own, 2)
        p.projection = round(mean, 2)
        p.projection_sd = round(sd, 2)
        p.ros_value = round(ros, 2) if ros is not None else None
        p.projection_source = source
        report.projected += 1

    for p in players:
        p.nfl_id = matched.mapping.get(p.player_id)
    assign_roles(players)
    from .locked import lock_played

    report.lock_notes = lock_played(snapshot, cfg, history, now=now)
    opponents = _opponents(snapshot, cfg)
    for p in players:
        p.nfl_opponent = opponents.get(p.team) if p.team else None

    if report.unmatched:
        report.notes.append(f"{len(report.unmatched)} player(s) could not be matched to nflverse history")
    mine = {p.name for p in snapshot.my_team.roster}
    sitting = [n for n in report.sitting if n in mine]
    if sitting:
        report.notes.append(
            f"{snapshot.platform.upper()} projects {', '.join(sitting)} for 0 this week, "
            "so they are treated as not playing")
    return report
