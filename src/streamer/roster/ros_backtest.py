"""Rest-of-season projections, graded on seasons already played.

Season-long decisions -- who to drop, who to stash, who to trade for -- run
on two things: a player's season value (``ros_value``) and the simulated
futures around it (:mod:`streamer.roster.futures`), whose spread is where
breakouts and busts live. This replays past weeks through the production
code exactly as it runs on a live league and grades both against what
happened next.

For each checkpoint (a season and week):

* **Who**: every QB/RB/WR/TE on an NFL roster that week (nflverse weekly
  rosters: active, or on a reserve list -- IR, PUP), with a status the
  platforms would have shown: IR for the reserve list, and the week's
  injury report (out, doubtful, questionable).
* **Projected**: :func:`project_snapshot` on a league holding all of them as
  free agents, fed only the games before that week -- the same volume and
  efficiency model, team-change discount, next man up (snap concentration
  included), rookie drift and early-season calibration the page uses.
* **Simulated**: :func:`futures.simulate` from that week to week 17, with
  the real byes and every lead/backup pair, as the season engine runs it.
* **Realized**: what each player scored over his team's remaining games
  (an absence or a release scores 0 -- that is what holding him returned),
  and per game played.

Only what a manager could have seen that week goes in; no later game,
roster move or injury.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..league.model import LeagueSnapshot, PlayerRow, TeamRow
from ..teams import normalize_team

SKILL = ("QB", "RB", "WR", "TE")
SEASONS = (2022, 2023, 2024, 2025)
CHECKPOINTS = (3, 4, 5, 6, 8, 10)
LAST_WEEK = 17


@dataclass
class Checkpoint:
    season: int
    week: int
    players: list[PlayerRow]
    #: Per player id: the simulated rest-of-season points per team game,
    #: one per simulated season.
    simulated: dict[str, np.ndarray]
    #: Per player id: realized rest-of-season figures.
    realized: pd.DataFrame


def _cached(cfg: Config, name: str, season: int, loader) -> pd.DataFrame:
    path = cfg.raw_dir / f"{name}_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frame = loader(seasons=[season]).to_pandas()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


def rosters(cfg: Config, season: int) -> pd.DataFrame:
    import nflreadpy

    r = _cached(cfg, "rosters_weekly", season, nflreadpy.load_rosters_weekly)
    r = r[r["position"].isin(SKILL) & r["gsis_id"].notna()].copy()
    r["team"] = r["team"].map(normalize_team)
    return r


def injuries(cfg: Config, season: int) -> pd.DataFrame:
    import nflreadpy

    return _cached(cfg, "injuries", season, nflreadpy.load_injuries)


def statuses(rost: pd.DataFrame, inj: pd.DataFrame, week: int) -> dict[str, tuple[str, str]]:
    """nflverse id -> (team, platform-style status) for everyone on a roster
    that week: active ("" or the injury report's tag) or on a reserve list
    ("IR"). A game-day inactive (INA) is on the roster too -- the platforms
    show him with the injury report's tag, or none for a healthy scratch."""
    wk = rost[(rost["week"] == week) & rost["status"].isin(["ACT", "RES", "INA"])]
    report = inj[(inj["week"] == week) & inj["gsis_id"].notna()].drop_duplicates("gsis_id", keep="last")
    tag = {"Out": "OUT", "Doubtful": "D", "Questionable": "Q"}
    rep = {str(g): tag.get(s, "") for g, s in zip(report["gsis_id"], report["report_status"])}
    out = {}
    for row in wk.drop_duplicates("gsis_id", keep="last").itertuples():
        status = "IR" if row.status == "RES" else rep.get(str(row.gsis_id), "")
        out[str(row.gsis_id)] = (row.team, status)
    return out


def byes(schedules: pd.DataFrame, season: int, weeks: list[int]) -> dict[str, set[int]]:
    """Team -> the weeks among ``weeks`` it does not play."""
    s = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")]
    played: dict[str, set[int]] = {}
    for g in s.itertuples():
        for t in (g.home_team, g.away_team):
            played.setdefault(normalize_team(t), set()).add(int(g.week))
    return {t: {w for w in weeks if w not in ws} for t, ws in played.items()}


def project(history: pd.DataFrame, cfg: Config, season: int, week: int,
            status: dict[str, tuple[str, str]]) -> list[PlayerRow]:
    """Every rostered skill player, projected by the production code on the
    games before ``week`` only."""
    from . import locked, projections

    prior = history[(history["season"] < season) | ((history["season"] == season) & (history["week"] < week))]
    last = prior.sort_values(["season", "week"]).groupby("player_id").tail(1).set_index("player_id")
    rows = []
    for nid, (team, st) in status.items():
        if nid not in last.index:
            continue                     # no NFL games yet: the platform's projection would stand in
        r = last.loc[nid]
        elig = {"QB": ["QB"], "RB": ["RB", "FLEX"], "WR": ["WR", "FLEX"], "TE": ["TE", "FLEX"]}[r["position"]]
        rows.append(PlayerRow(player_id=nid, name=str(r["player_display_name"]), position=str(r["position"]),
                              team=team, status=st, slot="FA", eligible_slots=elig))
    snap = LeagueSnapshot(
        platform="espn", profile="espn", league_id="backtest", league_name="backtest", season=season,
        week=week, slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}, bench_size=7,
        teams=[TeamRow(team_id="me", name="me", roster=[], is_mine=True)], free_agents=rows,
        matchup=None, synced_at=datetime(season, 9, 1, tzinfo=UTC).isoformat())

    class _Identity:
        def __init__(self, players):
            self.mapping = {p.player_id: p.player_id for p in players}
            self.unmatched = []

    from ..data import sleeper

    saved = (projections._implied_scale, projections.match_players, locked.lock_played, sleeper.depth_orders)
    projections._implied_scale = lambda *a, **k: ({}, "backtest")
    projections.match_players = lambda players, index: _Identity(players)
    locked.lock_played = lambda *a, **k: []
    sleeper.depth_orders = lambda *a, **k: {}      # today's depth charts say nothing about past seasons
    try:
        projections.project_snapshot(snap, cfg, rankings=None, allow_network=True, history=prior)
    finally:
        (projections._implied_scale, projections.match_players, locked.lock_played,
         sleeper.depth_orders) = saved
    return [p for p in rows if p.ros_value is not None]


def simulate(players: list[PlayerRow], cfg: Config, weeks: list[int], bye: dict[str, set[int]],
             n_sims: int = 2000, seed: int = 17, move_week: int | None = None) -> dict[str, np.ndarray]:
    """Per player: simulated points per team game over ``weeks`` (byes left
    out), one per simulated season; and, under ``(id, "avail")``, the share
    of those games he played."""
    from . import futures
    from .season import heirs_for

    conf = cfg.raw.get("roster", {})
    fut = futures.simulate(players, weeks, bye, n_sims=n_sims, seed=seed, heirs=heirs_for(players),
                           current=0, takeover=conf.get("next_man_up"),
                           takeover_sd=conf.get("next_man_up_spread"), kept=conf.get("next_man_up_kept"))
    out = {}
    for j, p in enumerate(players):
        live = np.array([w not in bye.get(p.team or "", set()) for w in weeks])
        if live.any():
            out[p.player_id] = fut.scores[:, j, live].mean(axis=1)
            out[(p.player_id, "avail")] = fut.played[:, j, live].mean(axis=1)
            sc, pl = fut.scores[:, j, live], fut.played[:, j, live]
            games = pl.sum(axis=1)
            out[(p.player_id, "pg")] = np.where(games >= PG_GAMES, sc.sum(axis=1) / np.maximum(games, 1), np.nan)
            ge, gl = pl[:, :EARLY_GAMES].sum(axis=1), pl[:, EARLY_GAMES:].sum(axis=1)
            out[(p.player_id, "early")] = np.where(ge >= EARLY_MIN, sc[:, :EARLY_GAMES].sum(axis=1)
                                                   / np.maximum(ge, 1), np.nan)
            out[(p.player_id, "late")] = np.where(gl >= LATE_MIN, sc[:, EARLY_GAMES:].sum(axis=1)
                                                  / np.maximum(gl, 1), np.nan)
            if move_week in weeks:
                kb = weeks.index(move_week)
                vis = fut.levels[:, j, kb]
                after = live & (np.arange(len(weeks)) >= kb)
                ga = fut.played[:, j, after].sum(axis=1)
                out[(p.player_id, "vis_move")] = np.where(vis > 0, vis, np.nan)
                out[(p.player_id, "late_move")] = np.where(ga >= LATE_MIN, fut.scores[:, j, after].sum(axis=1)
                                                           / np.maximum(ga, 1), np.nan)
    return out


#: Games a player must play for his points per game played to be graded --
#: in the replay and in each simulated season alike.
PG_GAMES = 4


def realized(history: pd.DataFrame, players: list[PlayerRow], season: int, weeks: list[int],
             bye: dict[str, set[int]], move_week: int | None = None) -> pd.DataFrame:
    """Per player: points per team game over ``weeks`` (0 for a game he
    missed), per game played, and the counts behind them; and, from
    ``move_week`` on, points per game played (to grade how much of a move in
    the projection by then held up)."""
    h = history[(history["season"] == season) & history["week"].isin(weeks)]
    pts = h.groupby(["player_id", "week"])["fantasy_points_ppr"].sum()
    rows = []
    for p in players:
        team_weeks = [w for w in weeks if w not in bye.get(p.team or "", set())]
        got = [float(pts.get((p.player_id, w), np.nan)) for w in team_weeks]
        played = [x for x in got if not np.isnan(x)]
        early = [x for x in got[:EARLY_GAMES] if not np.isnan(x)]
        late = [x for x in got[EARLY_GAMES:] if not np.isnan(x)]
        rows.append({"player_id": p.player_id, "team_games": len(team_weeks), "games": len(played),
                     "per_team_game": (sum(played) / len(team_weeks)) if team_weeks else np.nan,
                     "per_game": (sum(played) / len(played)) if played else np.nan,
                     "early_pg": float(np.mean(early)) if len(early) >= EARLY_MIN else np.nan,
                     "late_pg": float(np.mean(late)) if len(late) >= LATE_MIN else np.nan,
                     "late_move_pg": _after(got, team_weeks, move_week)})
    return pd.DataFrame(rows).set_index("player_id")


def _after(got: list[float], team_weeks: list[int], move_week: int | None) -> float:
    if move_week is None:
        return np.nan
    after = [x for x, w in zip(got, team_weeks) if w >= move_week and not np.isnan(x)]
    return float(np.mean(after)) if len(after) >= LATE_MIN else np.nan


def move_week_for(week: int, checkpoints) -> int | None:
    """The checkpoint three to five weeks on, where the projection is read
    again to see how far it moved."""
    return min((w for w in checkpoints if 3 <= w - week <= 5), default=None)


#: How much early form says about the rest of the season: points per game
#: over a player's next EARLY_GAMES team games (at least EARLY_MIN played)
#: against the games after them (at least LATE_MIN played).
EARLY_GAMES, EARLY_MIN, LATE_MIN = 4, 3, 4


def prepared(history: pd.DataFrame, cfg: Config, season: int, week: int, schedules: pd.DataFrame):
    """The projected players, the weeks to come, their byes and what happened:
    everything but the simulation, so the simulation can be rerun cheaply."""
    rost, inj = rosters(cfg, season), injuries(cfg, season)
    players = project(history, cfg, season, week, statuses(rost, inj, week))
    weeks = list(range(week, LAST_WEEK + 1))
    bye = byes(schedules, season, weeks)
    return players, weeks, bye, realized(history, players, season, weeks, bye)


def checkpoint(history: pd.DataFrame, cfg: Config, season: int, week: int, schedules: pd.DataFrame,
               n_sims: int = 2000, ready=None, move_week: int | None = None) -> Checkpoint:
    players, weeks, bye, real = ready or prepared(history, cfg, season, week, schedules)
    sim = simulate(players, cfg, weeks, bye, n_sims=n_sims, seed=17 + season * 100 + week, move_week=move_week)
    return Checkpoint(season, week, players, sim, real)


PROBS = tuple(round(0.05 * i, 2) for i in range(1, 20))
#: Points per team game at which the chance of reaching it is recorded.
THRESHOLDS = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22)


def frame(cp: Checkpoint, probs=PROBS, ros_next: dict | None = None) -> pd.DataFrame:
    """One row per player: what we said (season value, the simulated
    distribution of points per team game) and what happened."""
    rows = []
    for p in cp.players:
        sim = cp.simulated.get(p.player_id)
        if sim is None or p.player_id not in cp.realized.index:
            continue
        r = cp.realized.loc[p.player_id]
        q = np.quantile(sim, probs)
        row = {"season": cp.season, "week": cp.week, "player_id": p.player_id, "name": p.name,
               "position": p.position, "team": p.team, "status": p.status, "role": p.role,
               "experience": p.experience, "ros_value": float(p.ros_value or 0.0),
               "projection": float(p.projection or 0.0), "inherited": float(p.inherited_ros or 0.0),
               "takeover_share": p.takeover_share, "games_missed": p.games_missed,
               "signals": "; ".join(p.signals[:2]), "sim_mean": float(sim.mean()), "sim_sd": float(sim.std()),
               "sim_avail": float(cp.simulated[(p.player_id, "avail")].mean()),
               "actual": float(r["per_team_game"]), "actual_per_game": float(r["per_game"]),
               "games": int(r["games"]), "team_games": int(r["team_games"]),
               "pit": float((sim < r["per_team_game"]).mean() + 0.5 * (sim == r["per_team_game"]).mean())}
        for pr, v in zip(probs, q):
            row[f"q{int(round(pr * 100))}"] = float(v)
        # Points per game played, in the seasons where he played enough: the
        # spread of a player's level once absences are set aside.
        pg = cp.simulated.get((p.player_id, "pg"))
        pg = pg[np.isfinite(pg)] if pg is not None else np.array([])
        for name, pr in (("pg_q25", 0.25), ("pg_q50", 0.5), ("pg_q75", 0.75)):
            row[name] = float(np.quantile(pg, pr)) if len(pg) >= 30 else np.nan
        # Early form against the rest, as moments over the simulated seasons
        # where both were played, for pooling across players.
        row["early_real"] = float(r.get("early_pg", np.nan))
        row["late_real"] = float(r.get("late_pg", np.nan))
        e, la = cp.simulated.get((p.player_id, "early")), cp.simulated.get((p.player_id, "late"))
        both = np.isfinite(e) & np.isfinite(la) if e is not None and la is not None else np.array([False])
        if both.sum() >= 30:
            e, la = e[both], la[both]
            row.update(el_n=int(both.sum()), e_mean=float(e.mean()), l_mean=float(la.mean()),
                       e_var=float(e.var()), el_cov=float(((e - e.mean()) * (la - la.mean())).mean()))
        else:
            row.update(el_n=0, e_mean=np.nan, l_mean=np.nan, e_var=np.nan, el_cov=np.nan)
        # The projection a few weeks on, against what came after it.
        row["ros_next"] = float((ros_next or {}).get(p.player_id, np.nan))
        row["late_move_real"] = float(r.get("late_move_pg", np.nan))
        v, la = cp.simulated.get((p.player_id, "vis_move")), cp.simulated.get((p.player_id, "late_move"))
        both = np.isfinite(v) & np.isfinite(la) if v is not None and la is not None else np.array([False])
        if both.sum() >= 30:
            v, la = v[both], la[both]
            row.update(mv_mean=float(v.mean()), ml_mean=float(la.mean()), mv_var=float(v.var()),
                       ml_cov=float(((v - v.mean()) * (la - la.mean())).mean()))
        else:
            row.update(mv_mean=np.nan, ml_mean=np.nan, mv_var=np.nan, ml_cov=np.nan)
        for t in THRESHOLDS:
            row[f"p{t}"] = float((sim >= t).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_all(cfg: Config, seasons=SEASONS, checkpoints=CHECKPOINTS, log=print) -> dict:
    """(season, week) -> :func:`prepared`, for every checkpoint."""
    from .projections import load_history

    history = load_history(cfg)
    import nflreadpy

    schedules = pd.read_parquet(cfg.raw_dir / "schedules.parquet") if (cfg.raw_dir / "schedules.parquet").exists() \
        else nflreadpy.load_schedules().to_pandas()
    out = {}
    for season in seasons:
        for week in checkpoints:
            out[(season, week)] = prepared(history, cfg, season, week, schedules)
            log(f"{season} week {week}: {len(out[(season, week)][0])} players")
    return out


def persistence(d: pd.DataFrame) -> tuple[float, float, int]:
    """How much early form carries into the rest of the season: the slope of
    later points per game on early points per game, both against the season
    value, real and simulated (pooled over the players' simulated seasons).
    Returns (real, simulated, players)."""
    g = d.dropna(subset=["early_real", "late_real", "e_mean"])
    if len(g) < 20:
        return np.nan, np.nan, len(g)
    er, lr = g.early_real - g.ros_value, g.late_real - g.ros_value
    real = float(np.cov(er, lr)[0, 1] / er.var())
    em, lm = g.e_mean - g.ros_value, g.l_mean - g.ros_value
    cov = g.el_cov.mean() + float(np.cov(em, lm)[0, 1])
    var = g.e_var.mean() + float(em.var())
    return real, float(cov / var), len(g)


def moves(d: pd.DataFrame) -> dict:
    """How far the projection moved a few weeks on, and how much of the move
    held up in the points per game that followed -- real (the production
    projection at the later checkpoint) and simulated (the level a manager
    sees then), pooled over players and simulated seasons."""
    g = d.dropna(subset=["ros_next", "late_move_real", "mv_mean"])
    if len(g) < 20:
        return {"n": len(g)}
    x, y = g.ros_next - g.ros_value, g.late_move_real - g.ros_value
    mx, my = g.mv_mean - g.ros_value, g.ml_mean - g.ros_value
    var = g.mv_var.mean() + float(mx.var())
    cov = g.ml_cov.mean() + float(np.cov(mx, my)[0, 1])
    return {"n": len(g), "real_sd": float(x.std()), "real_slope": float(np.cov(x, y)[0, 1] / x.var()),
            "sim_sd": float(np.sqrt(var)), "sim_slope": float(cov / var)}


def run(cfg: Config, seasons=SEASONS, checkpoints=CHECKPOINTS, n_sims: int = 2000, log=print,
        ready: dict | None = None, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every checkpoint, simulated and graded. With ``history``, each is also
    read against the next checkpoint a few weeks on (:func:`moves`)."""
    ready = ready or prepare_all(cfg, seasons, checkpoints, log=log)
    frames = []
    for (season, week), prep in ready.items():
        b = move_week_for(week, [w for s, w in ready if s == season]) if history is not None else None
        ros_next = None
        if b is not None:
            players, weeks, bye, _real = prep
            ros_next = {p.player_id: float(p.ros_value or 0.0) for p in ready[(season, b)][0]
                        if p.status in ("", "Q")}
            prep = (players, weeks, bye, realized(history, players, season, weeks, bye, move_week=b))
        cp = checkpoint(None, cfg, season, week, None, n_sims=n_sims, ready=prep, move_week=b)
        frames.append(frame(cp, ros_next=ros_next))
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)
