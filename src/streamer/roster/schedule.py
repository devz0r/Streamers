"""Who a player faces in the weeks to come, as a multiplier on his level.

A season value is a per-game level measured on the games he has played,
against whoever he played; the weeks to come bring their own opponents.
Two signals, both from games already played (no future lines exist):

* **the game's environment** -- every past game's closing implied team total
  fitted as league mean + offence + defence + home (ridge, this season and
  last at half weight), which predicts each future game's implied total;
  over his team's trailing implied total, it says how much more or less
  scoring that game should hold;
* **points allowed to his position** by the opponent, per game, shrunk
  toward the league mean by ``allowed_k`` games.

Each week's factor is ``environment ** implied * allowed ** allowed`` with
weights by position (``roster.schedule``), fitted on 2022-2025 and scored on
the season left out: quarterbacks and backs gain, receivers and tight ends
did not (DECISIONS.md, "Who he plays the rest of the way"). The current
week is left out -- its own lines already price it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..teams import normalize_team


def _ratings(games: pd.DataFrame, season: int, week: int, lam: float, prev_w: float):
    past = games[((games["season"] == season) & (games["week"] < week)) | (games["season"] == season - 1)]
    past = past.dropna(subset=["team_implied_total"])
    if past.empty:
        return None
    w = np.where(past["season"] == season, 1.0, prev_w)
    teams = sorted(set(past["team"]) | set(past["opponent"]))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    X = np.zeros((len(past), 2 * n + 2))
    rows = np.arange(len(past))
    X[:, 0] = 1.0
    X[rows, 1 + past["team"].map(ix).to_numpy()] = 1.0
    X[rows, 1 + n + past["opponent"].map(ix).to_numpy()] = 1.0
    X[:, -1] = past["is_home"].to_numpy()
    pen = np.eye(2 * n + 2) * lam
    pen[0, 0] = pen[-1, -1] = 0.0
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + pen, X.T @ (w * past["team_implied_total"].to_numpy()))

    def predict(team: str, opp: str, home: int) -> float:
        if team not in ix or opp not in ix:
            return float("nan")
        return float(beta[0] + beta[1 + ix[team]] + beta[1 + n + ix[opp]] + beta[-1] * home)

    return predict


def _allowed(history: pd.DataFrame, games: pd.DataFrame, season: int, week: int, pos: str, k: float,
             prev_w: float) -> dict[str, float]:
    """Opponent -> points allowed to ``pos`` per game over the league mean."""
    opp = {(s, w, t): o for s, w, t, o in zip(games["season"], games["week"], games["team"], games["opponent"])}
    h = history[(history["position"] == pos)
                & (((history["season"] == season) & (history["week"] < week)) | (history["season"] == season - 1))]
    if h.empty:
        return {}
    h = h.assign(_opp=[opp.get((s, w, normalize_team(t))) for s, w, t in zip(h["season"], h["week"], h["team"])])
    a = h.dropna(subset=["_opp"]).groupby(["season", "week", "_opp"])["fantasy_points_ppr"].sum().reset_index()
    if a.empty:
        return {}
    wt = np.where(a["season"] == season, 1.0, prev_w)
    mu = float(np.average(a["fantasy_points_ppr"], weights=wt))
    out = {}
    for t, g in a.assign(_w=wt).groupby("_opp"):
        out[t] = (float((g["fantasy_points_ppr"] * g["_w"]).sum()) + k * mu) / (float(g["_w"].sum()) + k) / mu
    return out


def factors(cfg: Config, season: int, week: int, history: pd.DataFrame,
            games: pd.DataFrame | None = None) -> dict[tuple[str, str], dict[int, float]]:
    """(team, position) -> {future week: factor on his level}, for the
    positions ``roster.schedule`` weights, weeks after ``week``."""
    conf = (cfg.raw.get("roster") or {}).get("schedule") or {}
    weights = {p: v for p, v in (conf.get("positions") or {}).items() if v}
    if not weights or history is None or history.empty:
        return {}
    if games is None:
        from ..data.nflverse import games_frame

        games = games_frame(cfg)
    games = games[games["game_type"] == "REG"].assign(
        team=lambda g: g["team"].map(normalize_team), opponent=lambda g: g["opponent"].map(normalize_team))
    lam, prev_w = float(conf.get("ridge", 4.0)), float(conf.get("previous_season_weight", 0.5))
    predict = _ratings(games, season, week, lam, prev_w)
    if predict is None:
        return {}
    trail = (games[((games["season"] < season) | ((games["season"] == season) & (games["week"] < week)))]
             .dropna(subset=["team_implied_total"]).sort_values(["season", "week"])
             .groupby("team")["team_implied_total"].apply(lambda s: float(s.tail(6).mean())))
    future = games[(games["season"] == season) & (games["week"] > week)]
    out: dict[tuple[str, str], dict[int, float]] = {}
    for pos, w in weights.items():
        a_imp, a_all = float(w.get("implied", 0.0)), float(w.get("allowed", 0.0))
        allowed = _allowed(history, games, season, week, pos, float(conf.get("allowed_k", 6.0)), prev_w) if a_all else {}
        for r in future.itertuples():
            env = predict(r.team, r.opponent, int(r.is_home)) / trail.get(r.team, np.nan)
            f = 1.0
            if a_imp and np.isfinite(env) and env > 0:
                f *= env ** a_imp
            if a_all:
                f *= allowed.get(r.opponent, 1.0) ** a_all
            out.setdefault((r.team, pos), {})[int(r.week)] = round(float(f), 4)
    return out


def describe(sched: dict[int, float], playoffs: tuple[int, ...] = (15, 16, 17), threshold: float = 0.04) -> str:
    """A card note when the weeks to come are clearly easier or harder."""
    if not sched:
        return ""
    rest = float(np.mean(list(sched.values())))
    late = [sched[w] for w in playoffs if w in sched]
    bits = []
    if abs(rest - 1.0) >= threshold:
        bits.append(f"{'easier' if rest > 1 else 'harder'} schedule the rest of the way ({rest - 1:+.0%})")
    if late and abs(np.mean(late) - 1.0) >= threshold:
        bits.append(f"playoff weeks {playoffs[0]}-{playoffs[-1]} {np.mean(late) - 1:+.0%}")
    return "; ".join(bits)
