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
