"""Similar past player-weeks: the mean score of a player's nearest earlier
weeks at his position, from what was known before each.

Tested for every skill position (scripts/fit_player_comps.py). Only
quarterbacks gain: blended 80% with the model's own number they beat the
model alone in 2023, 2024 and 2026 on weeks the weight was not chosen on
(MAE 6.351 -> 6.232); backs, receivers and tight ends got slightly worse.
Config ``roster.comps``; DECISIONS.md, "Player comparisons".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)

FEATURES = ["xfp4", "xfp8", "fp4", "tgt4", "car4", "tsh4", "implied", "spread"]
MIN_GAMES = 3
_POOL: dict[tuple, pd.DataFrame] = {}


def settings(cfg: Config) -> dict:
    """Position -> {weight, k, since} for the positions that use comparisons."""
    conf = ((cfg.raw.get("roster") or {}).get("comps")) or {}
    return {pos: c for pos, c in conf.items() if isinstance(c, dict) and float(c.get("weight", 0.0)) > 0}


def game_lines(games: pd.DataFrame) -> pd.DataFrame:
    """(season, week, team) -> implied total and spread (positive = favoured)."""
    from ..teams import normalize_team

    gm = games[games["game_type"] == "REG"].assign(team=lambda x: x["team"].map(normalize_team))
    gm = gm[["season", "week", "team", "spread_line", "total_line", "is_home"]].copy()
    # nflverse's spread_line is the home team's margin.
    gm["spread"] = np.where(gm["is_home"] == 1, gm["spread_line"], -gm["spread_line"])
    gm["implied"] = gm["total_line"] / 2.0 + gm["spread"] / 2.0
    return gm[["season", "week", "team", "implied", "spread"]]


def features(h: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per played player-week, its features from earlier games only."""
    from ..teams import normalize_team

    h = h.sort_values(["player_id", "season", "week"]).copy()
    g = h.groupby("player_id", sort=False)

    def trail(col, n):
        return g[col].transform(lambda s: s.shift(1).rolling(n, min_periods=MIN_GAMES).mean())

    h["xfp4"] = trail("total_fantasy_points_exp", 4)
    h["xfp8"] = trail("total_fantasy_points_exp", 8)
    h["fp4"] = trail("fantasy_points_ppr", 4)
    h["tgt4"] = trail("targets", 4)
    h["car4"] = trail("carries", 4)
    h["tsh4"] = trail("target_share", 4)
    h["team"] = h["team"].map(normalize_team)
    out = h.merge(game_lines(games), on=["season", "week", "team"], how="left")
    return out.dropna(subset=FEATURES)


def deep_history(cfg: Config, since: int) -> pd.DataFrame:
    """Weekly skill-player rows from ``since`` on (the production loader, more seasons)."""
    from .projections import load_history

    saved = list(cfg.season["train_seasons"])
    cfg.season["train_seasons"] = list(range(int(since), int(cfg.current_season) + 1))
    try:
        return load_history(cfg)
    finally:
        cfg.season["train_seasons"] = saved


def pool(cfg: Config, since: int) -> pd.DataFrame:
    """Every played player-week with features since ``since`` (built once per process)."""
    key = (str(cfg.raw_dir), int(since))
    if key not in _POOL:
        from ..data.nflverse import games_frame

        _POOL[key] = features(deep_history(cfg, since), games_frame(cfg))
    return _POOL[key]


def current(history: pd.DataFrame, games: pd.DataFrame, ids: dict[str, str],
            season: int, week: int) -> pd.DataFrame:
    """Features for this week's game of each nflverse id (-> its team): his
    last 4 and 8 games before ``(season, week)``."""
    need = {"total_fantasy_points_exp", "fantasy_points_ppr", "targets", "carries", "target_share"}
    if not need <= set(history.columns):
        return pd.DataFrame(columns=["player_id", *FEATURES])
    h = history[(history["season"] < season) | ((history["season"] == season) & (history["week"] < week))]
    h = h[h["player_id"].isin(list(ids))].sort_values(["season", "week"])
    lines = game_lines(games)
    lines = lines[(lines["season"] == season) & (lines["week"] == week)].set_index("team")
    rows = []
    for pid, g in h.groupby("player_id"):
        if len(g) < MIN_GAMES or ids[pid] not in lines.index:
            continue
        l4, l8 = g.tail(4), g.tail(8)
        line = lines.loc[ids[pid]]
        rows.append({"player_id": pid, "xfp4": l4["total_fantasy_points_exp"].mean(),
                     "xfp8": l8["total_fantasy_points_exp"].mean(), "fp4": l4["fantasy_points_ppr"].mean(),
                     "tgt4": l4["targets"].mean(), "car4": l4["carries"].mean(),
                     "tsh4": l4["target_share"].mean(), "implied": line["implied"], "spread": line["spread"]})
    return pd.DataFrame(rows, columns=["player_id", *FEATURES]).dropna(subset=FEATURES)


def nearest_mean(pool_rows: pd.DataFrame, rows: pd.DataFrame, k: int) -> np.ndarray:
    """Mean PPR score of each row's ``k`` nearest pool rows (standardised features)."""
    mu, sd = pool_rows[FEATURES].mean(), pool_rows[FEATURES].std().replace(0.0, 1.0).fillna(1.0)
    a = ((pool_rows[FEATURES] - mu) / sd).to_numpy(np.float64)
    b = ((rows[FEATURES] - mu) / sd).to_numpy(np.float64)
    d = (b * b).sum(1)[:, None] - 2 * b @ a.T + (a * a).sum(1)[None, :]
    nearest = np.argpartition(d, k - 1, axis=1)[:, :k]
    return pool_rows["fantasy_points_ppr"].to_numpy(float)[nearest].mean(axis=1)


def for_week(cfg: Config, history: pd.DataFrame, ids: dict[str, tuple[str, str]],
             season: int, week: int) -> dict[str, float]:
    """nflverse id -> the comparison forecast, for the ids (-> (team, position))
    whose position uses comparisons. The pool holds only weeks before this one."""
    conf = settings(cfg)
    want = {pid: team for pid, (team, pos) in ids.items() if pos in conf and team}
    if not want:
        return {}
    from ..data.nflverse import games_frame

    since = min(int(c.get("since", 2015)) for c in conf.values())
    try:
        games = games_frame(cfg)
        cur = current(history, games, want, season, week)
        if cur.empty:
            return {}
        full = pool(cfg, since)
    except Exception as exc:  # noqa: BLE001 - a refinement; the model stands without it
        log.warning("player comparisons unavailable: %s", exc)
        return {}
    earlier = full[(full["season"] < season) | ((full["season"] == season) & (full["week"] < week))]
    out: dict[str, float] = {}
    for pos, c in conf.items():
        k = int(c.get("k", 100))
        rows = cur[cur["player_id"].map(lambda x, pos=pos: ids[x][1] == pos)]
        pp = earlier[earlier["position"] == pos]
        if rows.empty or len(pp) < k:
            continue
        out.update(zip(rows["player_id"], nearest_mean(pp, rows, k)))
    return out


# ---------------------------------------------------------------------------
# Rest of season: players most like him at the same point of earlier seasons,
# and how their seasons finished (scripts/fit_ros_comps.py). 20% toward their
# mean beat our season value alone in 2022, 2024 and 2025 (MAE 2.713 ->
# 2.704 per game); weeks 3-10 and players without an out/IR tag, as tested.
# ---------------------------------------------------------------------------
ROS_FEATURES = ["gp", "ppg", "xpg", "tpg", "cpg", "tsh", "fp4", "xfp4", "prev_ppg", "prev_xpg", "has_prev",
                "career", "week"]
ROS_LAST_WEEK = 17
ROS_MIN_GAMES = 3
ROS_POOL_WEEKS = tuple(range(3, 11))


def _ros_context(h: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Each player-season's points and expected points per game (4+ games),
    and his games in seasons before each season."""
    h = h[h["week"] <= ROS_LAST_WEEK]
    per = h.groupby(["season", "player_id"]).agg(ppg=("fantasy_points_ppr", "mean"),
                                                 xpg=("total_fantasy_points_exp", "mean"), n=("week", "size"))
    prev = per[per["n"] >= 4].reset_index()
    games = h.groupby(["season", "player_id"]).size().unstack(0, fill_value=0)
    career = games.cumsum(axis=1).shift(1, axis=1).fillna(0).stack().swaplevel().sort_index()
    return prev, career


def ros_rows(h: pd.DataFrame, season: int, week: int, prev: pd.DataFrame | None = None,
             career: pd.Series | None = None) -> pd.DataFrame:
    """Features at (season, week) for everyone with 2+ games before it, from
    those games only, and his rest-of-season points per game (NaN under
    ``ROS_MIN_GAMES`` games after it)."""
    if prev is None or career is None:
        prev, career = _ros_context(h[h["season"] <= season])
    s = h[(h["season"] == season) & (h["week"] <= ROS_LAST_WEEK)]
    before, after = s[s["week"] < week], s[s["week"] >= week]
    if before.empty:
        return pd.DataFrame(columns=["player_id", "position", *ROS_FEATURES, "ros_pg"])
    g = before.sort_values("week").groupby("player_id")
    f = pd.DataFrame({
        "position": g["position"].last(), "gp": g.size(), "ppg": g["fantasy_points_ppr"].mean(),
        "xpg": g["total_fantasy_points_exp"].mean(), "tpg": g["targets"].mean(), "cpg": g["carries"].mean(),
        "tsh": g["target_share"].mean(),
        "fp4": g["fantasy_points_ppr"].apply(lambda x: x.tail(4).mean()),
        "xfp4": g["total_fantasy_points_exp"].apply(lambda x: x.tail(4).mean()),
    })
    f = f[f["gp"] >= 2]
    a = after.groupby("player_id")["fantasy_points_ppr"].agg(["mean", "size"])
    f["ros_pg"] = a["mean"].reindex(f.index).where(a["size"].reindex(f.index) >= ROS_MIN_GAMES)
    pv = prev[prev["season"] == season - 1].set_index("player_id")
    f["prev_ppg"] = pv["ppg"].reindex(f.index)
    f["prev_xpg"] = pv["xpg"].reindex(f.index)
    f["has_prev"] = f["prev_ppg"].notna().astype(float)
    f[["prev_ppg", "prev_xpg"]] = f[["prev_ppg", "prev_xpg"]].fillna(0.0)
    idx = pd.MultiIndex.from_product([[season], f.index])
    f["career"] = career.reindex(idx).fillna(0).to_numpy().clip(max=80)
    f["season"], f["week"] = season, week
    f["tsh"] = f["tsh"].fillna(0.0)
    return f.reset_index()


def ros_table(h: pd.DataFrame, seasons, weeks) -> pd.DataFrame:
    """:func:`ros_rows` for every season and week given."""
    prev, career = _ros_context(h)
    return pd.concat([ros_rows(h, s, w, prev, career) for s in seasons for w in weeks], ignore_index=True)


def ros_pool(cfg: Config, since: int, before_season: int) -> pd.DataFrame:
    """Player-checkpoints of the seasons before ``before_season`` with an
    outcome, cached on disk (finished seasons do not change)."""
    path = cfg.raw_dir / f"ros_comp_pool_{since}_{before_season}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001 - rebuilt below
            pass
    h = deep_history(cfg, since)
    h = h[h["season"] < before_season]
    out = ros_table(h, range(int(since), int(before_season)), ROS_POOL_WEEKS)
    out = out[out["ros_pg"].notna()].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


def ros_nearest(pool_rows: pd.DataFrame, rows: pd.DataFrame, k: int) -> np.ndarray:
    """Mean rest-of-season points per game of each row's ``k`` nearest pool rows."""
    mu, sd = pool_rows[ROS_FEATURES].mean(), pool_rows[ROS_FEATURES].std().replace(0.0, 1.0).fillna(1.0)
    a = ((pool_rows[ROS_FEATURES] - mu) / sd).to_numpy(np.float64)
    b = ((rows[ROS_FEATURES] - mu) / sd).to_numpy(np.float64)
    d = (b * b).sum(1)[:, None] - 2 * b @ a.T + (a * a).sum(1)[None, :]
    nearest = np.argpartition(d, k - 1, axis=1)[:, :k]
    return pool_rows["ros_pg"].to_numpy(float)[nearest].mean(axis=1)


def ros_settings(cfg: Config) -> dict | None:
    conf = ((cfg.raw.get("roster") or {}).get("ros_comps")) or None
    return conf if conf and float(conf.get("weight", 0.0)) > 0 else None


def ros_for_week(cfg: Config, ids: set[str], season: int, week: int) -> dict[str, float]:
    """nflverse id -> the rest-of-season comparison mean, for weeks inside the
    tested range. Features come only from games before (season, week)."""
    conf = ros_settings(cfg)
    if conf is None or not ids or not (min(ROS_POOL_WEEKS) <= week <= max(ROS_POOL_WEEKS)):
        return {}
    since, k = int(conf.get("since", 2015)), int(conf.get("k", 100))
    try:
        h = deep_history(cfg, since)
        h = h[(h["season"] < season) | ((h["season"] == season) & (h["week"] < week))]
        rows = ros_rows(h, season, week)
        rows = rows[rows["player_id"].isin(ids)]
        if rows.empty:
            return {}
        pool_rows = ros_pool(cfg, since, season)
    except Exception as exc:  # noqa: BLE001 - a refinement; the season value stands without it
        log.warning("rest-of-season comparisons unavailable: %s", exc)
        return {}
    out: dict[str, float] = {}
    for pos, r in rows.groupby("position"):
        pp = pool_rows[pool_rows["position"] == pos]
        if len(pp) >= k:
            out.update(zip(r["player_id"], ros_nearest(pp, r, k)))
    return out
