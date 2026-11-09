"""Grade the live in-game P(win) on replayed leagues, from play-by-play.

    python scripts/grade_live_odds.py [--leagues 6] [--weekly cache.pkl] [--units cache.parquet] [--out dir]

The live card (``static/live.js``) rescores the lineup draws as games are
played: a finished player counts what he scored, one not started his draw, one
in progress ``now + f m + sqrt(f) (draw - m)`` (``f`` the share of his game
left, ``m`` his projection; a defence ``(1 - f) now + f draw``). P(win) is the
share of draws ahead, mapped by an exponent that starts at the pre-game
mapping's. None of that had been set against results. Here it is:

* the leagues of :mod:`streamer.roster.league_replay` are played on 2022-2025
  (the same projections, lineups and draws as the weekly P(win) grading);
* each week's NFL games are put in kickoff order (``schedules``), and every
  head-to-head game is read at each kickoff window's start, end of the first
  quarter, half and end of the third, with the earlier windows final and the
  later ones not started -- each starter's points so far from the
  play-by-play (skill players play by play; kickers and defences from the
  production scorers on the plays so far and the score so far);
* the live number at each of those states is scored against who won.

Part one grades the player model (what he adds from here against ``f m``,
and its spread); part two the matchup's P(win) under each mapping, held out by
season. DECISIONS.md, "Grading the live win chance".
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

STAGES = (0.25, 0.5, 0.75, 1.0)          # share of a game played at each reading
SEASONS = (2022, 2023, 2024, 2025)
K0 = 0.91                                 # lineup.WIN_SHRINK, the pre-game mapping
LIVE_SIMS = 2000                          # draws the page carries (roster.live.LIVE_SIMS)
SKILL_COLUMNS = ["game_id", "season_type", "qtr", "game_seconds_remaining", "passer_player_id", "receiver_player_id",
                 "rusher_player_id", "fumbled_1_player_id", "td_player_id", "passing_yards", "receiving_yards",
                 "rushing_yards", "complete_pass", "pass_touchdown", "rush_touchdown", "interception", "fumble_lost",
                 "total_home_score", "total_away_score"]


# ---------------------------------------------------------------------------
# Points so far, from the play-by-play
# ---------------------------------------------------------------------------
def skill_so_far(pbp: pd.DataFrame) -> pd.DataFrame:
    """(game_id, player_id) -> PPR points through each stage (nflverse's
    standard sheet: the replay's ``fantasy_points_ppr``)."""
    p = pbp[pbp["game_seconds_remaining"].notna()]
    played = 1.0 - p["game_seconds_remaining"].clip(0, 3600) / 3600.0
    played = played.where(p["qtr"] <= 4, 1.0)                    # overtime counts at the end
    parts = []
    z = lambda c: p[c].fillna(0.0)                               # noqa: E731
    td_by = p["td_player_id"]
    parts.append(pd.DataFrame({"game_id": p["game_id"], "pid": p["passer_player_id"], "t": played,
                               "pts": 0.04 * z("passing_yards") + 4.0 * z("pass_touchdown") - 2.0 * z("interception")}))
    parts.append(pd.DataFrame({"game_id": p["game_id"], "pid": p["receiver_player_id"], "t": played,
                               "pts": z("complete_pass") + 0.1 * z("receiving_yards")
                               + 6.0 * (z("pass_touchdown") * (td_by == p["receiver_player_id"]))}))
    parts.append(pd.DataFrame({"game_id": p["game_id"], "pid": p["rusher_player_id"], "t": played,
                               "pts": 0.1 * z("rushing_yards")
                               + 6.0 * (z("rush_touchdown") * (td_by == p["rusher_player_id"]))}))
    parts.append(pd.DataFrame({"game_id": p["game_id"], "pid": p["fumbled_1_player_id"], "t": played,
                               "pts": -2.0 * z("fumble_lost")}))
    d = pd.concat(parts, ignore_index=True)
    d = d[d["pid"].notna() & (d["pts"] != 0)]
    out = None
    for q in STAGES:
        s = d[d["t"] <= q + 1e-9].groupby(["game_id", "pid"])["pts"].sum().rename(f"q{int(q * 100)}")
        out = s.to_frame() if out is None else out.join(s, how="outer")
    return out.fillna(0.0).reset_index()


def unit_so_far(pbp: pd.DataFrame, games: pd.DataFrame, cfg, raw: pd.DataFrame) -> pd.DataFrame:
    """(game_id, "K:TEAM" / "DST:TEAM") -> points through each stage, from the
    production scorers on the plays so far and the score so far."""
    from streamer.actuals import dst_game_lines, kicker_game_lines
    from streamer.teams import normalize_team

    def clock(f):
        f = f[f["game_seconds_remaining"].notna()].copy()
        f["_t"] = (1.0 - f["game_seconds_remaining"].clip(0, 3600) / 3600.0).where(f["qtr"] <= 4, 1.0)
        return f

    p, r = clock(pbp), clock(raw)
    out = None
    for q in STAGES:
        sub = p[p["_t"] <= q + 1e-9].drop(columns="_t")
        score = r[r["_t"] <= q + 1e-9].groupby("game_id").agg(h=("total_home_score", "max"), a=("total_away_score", "max"))
        g = games[games["game_id"].isin(score.index)].copy()
        hs, as_ = g["game_id"].map(score["h"]), g["game_id"].map(score["a"])
        home = g["is_home"].astype(bool)
        g["team_score"], g["opp_score"] = np.where(home, hs, as_), np.where(home, as_, hs)
        dst = dst_game_lines(sub, g, cfg)
        dst = pd.DataFrame({"game_id": dst["game_id"], "pid": "DST:" + dst["team"].map(normalize_team),
                            f"q{int(q * 100)}": dst["fantasy_points"].astype(float)})
        kick = kicker_game_lines(sub, cfg)
        team_col = "team" if "team" in kick.columns else "posteam"
        kick = (kick.assign(pid="K:" + kick[team_col].map(normalize_team))
                .groupby(["game_id", "pid"])["fantasy_points"].sum().rename(f"q{int(q * 100)}").reset_index())
        both = pd.concat([dst, kick], ignore_index=True).set_index(["game_id", "pid"])
        out = both if out is None else out.join(both, how="outer")
    return out.fillna(0.0).reset_index()


def kickoffs(schedules: pd.DataFrame, season: int) -> dict[tuple[int, str], tuple[int, str]]:
    """(week, team) -> (kickoff window in the week, in order; game id)."""
    from streamer.teams import normalize_team

    s = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")].copy()
    s["at"] = pd.to_datetime(s["gameday"].astype(str) + " " + s["gametime"].astype(str), errors="coerce")
    out = {}
    for week, g in s.groupby("week"):
        order = {t: i for i, t in enumerate(sorted(g["at"].dropna().unique()))}
        for r in g.itertuples():
            for team in (r.home_team, r.away_team):
                out[(int(week), normalize_team(team))] = (order.get(r.at, 0), r.game_id)
    return out


def first_touch(pbp: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(game id, player id) -> the share of the game played at his first pass, carry or target."""
    p = pbp[pbp["game_seconds_remaining"].notna()]
    t = (1.0 - p["game_seconds_remaining"].clip(0, 3600) / 3600.0).where(p["qtr"] <= 4, 1.0)
    d = pd.concat([pd.DataFrame({"g": p["game_id"], "pid": p[c], "t": t})
                   for c in ("passer_player_id", "rusher_player_id", "receiver_player_id")]).dropna()
    return d.groupby(["g", "pid"])["t"].min().to_dict()


class SoFar:
    """A season's points so far: ``now(week, player row, final points, stage)``."""

    def __init__(self, season: int, cfg, schedules: pd.DataFrame, units: bool = True):
        from streamer.data.nflverse import games_frame, load_pbp

        pbp = load_pbp([season], cfg)
        pbp = pbp[pbp["season_type"].eq("REG")] if "season_type" in pbp.columns else pbp
        self.when = kickoffs(schedules, season)
        raw = pd.read_parquet(cfg.raw_dir / f"pbp_{season}.parquet", columns=SKILL_COLUMNS)
        raw = raw[raw["season_type"].eq("REG")]
        frames = [skill_so_far(raw)]
        if units:
            games = games_frame(cfg)
            frames.append(unit_so_far(pbp, games[games["season"] == season], cfg, raw))
        d = pd.concat(frames, ignore_index=True)
        self.table = {(g, p): (a, b, c, e) for g, p, a, b, c, e in
                      d[["game_id", "pid", "q25", "q50", "q75", "q100"]].itertuples(index=False)}
        self.first = first_touch(raw)

    def game(self, week: int, team: str | None) -> tuple[int, str] | None:
        return self.when.get((int(week), team or ""))

    def touched(self, week: int, p, stage: float) -> bool:
        """Has he had a pass, carry or target by ``stage`` of his game?"""
        g = self.game(week, p.team)
        t = self.first.get((g[1], p.player_id)) if g is not None else None
        return t is not None and t <= stage + 1e-9

    def now(self, week: int, p, final: float, stage: float) -> float:
        """His points through ``stage`` of his game: the official final less
        what the play-by-play has him adding after it."""
        g = self.game(week, p.team)
        if g is None or stage >= 1.0:
            return float(final)
        got = self.table.get((g[1], p.player_id))
        if got is None:
            return float(final) if p.position in ("QB", "RB", "WR", "TE") and final == 0 else 0.0
        return float(final) - (got[3] - got[STAGES.index(stage)])


# ---------------------------------------------------------------------------
# The live rescoring, as static/live.js does it
# ---------------------------------------------------------------------------
def totals(draws: np.ndarray, m: np.ndarray, now: np.ndarray, left: np.ndarray, dst: np.ndarray,
           pace: float = 0.0, dst_params: tuple[float, float] | None = None) -> np.ndarray:
    """Each draw's total for one side (``live.js`` ``totals``). ``pace``: the
    share of a player's remaining mean taken from his pace so far instead of
    his projection (0 as shipped). ``dst_params``: ``(decay, giveback)`` for a
    defence in progress, ``now (1 - decay f^2) + f (m - giveback) + sqrt(f)
    (draw - m)`` (static/live.js ``dstParams``); without it the blend first
    shipped, ``(1 - f) now + f draw``."""
    out = np.zeros(draws.shape[0])
    for j in range(draws.shape[1]):
        f, d = float(left[j]), draws[:, j]
        if f >= 1:
            v = d
        elif f <= 0:
            v = np.full_like(d, now[j])
        elif dst[j] and dst_params is None:
            v = (1 - f) * now[j] + f * d
        elif dst[j]:
            v = now[j] * (1 - dst_params[0] * f * f) + f * (m[j] - dst_params[1]) + np.sqrt(f) * (d - m[j])
        else:
            mean = m[j] if pace <= 0 else (1 - pace) * m[j] + pace * max(now[j], 0.0) / (1 - f)
            v = np.maximum(now[j] + f * mean + np.sqrt(f) * (d - m[j]), now[j] - 2)
        out += v
    return out


SKILL = ("QB", "RB", "WR", "TE")
SHARE_PLAYED = (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.375, 0.5, 0.625, 0.75, 0.875, 1)
PLAYS_HEALTHY = 0.935                    # static/live.js: a starter with no injury tag who played, 2022-2025


def none_yet_table(rows: pd.DataFrame) -> dict[str, list[float]]:
    """Position -> the share of starters who played with no pass, carry or
    target yet at each point of SHARE_PLAYED (``rows``: pos, first)."""
    out = {}
    for pos, g in rows[rows["first"].notna()].groupby("pos"):
        out[pos] = [1.0] + [round(float((g["first"] > t).mean()), 3) for t in SHARE_PLAYED[1:-1]] + [0.0]
    return out


def playing(pos: str, q: float, touched: bool, played: float, table: dict, healthy: float = PLAYS_HEALTHY) -> float:
    """live.js ``playing``: 1 once he has touched the ball; with nothing yet,
    his prior chance against how rarely a player who is in has nothing by now."""
    if touched:
        return 1.0
    prior = min(q, healthy)
    s = float(np.interp(played, SHARE_PLAYED, table[pos]))
    return prior * s / (prior * s + 1 - prior)


def totals_live(draws: np.ndarray, sat: np.ndarray, m: np.ndarray, mp: np.ndarray, now: np.ndarray, left: np.ndarray,
                dst: np.ndarray, pp: list, dp: tuple[float, float], col0: int = 0) -> np.ndarray:
    """live.js ``totals`` with a starter's chance of playing: a player known to
    be in (``pp`` 1) has his sat draws replaced by played ones and his mean if
    he plays; one not seen yet plays in a ``pp`` share of the draws."""
    n = draws.shape[0]
    out = np.zeros(n)
    idx = np.arange(n)
    for j in range(draws.shape[1]):
        f, d, mean = float(left[j]), draws[:, j], m[j]
        if pp[j] is not None:
            ok = np.flatnonzero(~sat[:, j])
            if sat[:, j].any() and len(ok):
                d = np.where(sat[:, j], d[ok[(idx * 7919) % len(ok)]], d)
            mean = mp[j]
        if f >= 1:
            v = d
        elif f <= 0:
            v = np.full(n, now[j])
        elif dst[j]:
            v = now[j] * (1 - dp[0] * f * f) + f * (m[j] - dp[1]) + np.sqrt(f) * (d - m[j])
        else:
            v = np.maximum(now[j] + f * mean + np.sqrt(f) * (d - mean), now[j] - 2)
        if pp[j] is not None and pp[j] < 1 and 0 < f < 1:
            u = ((idx + 0.5) * 0.6180339887498949 + (j + col0) * 0.3819660112501051) % 1
            v = np.where(u < pp[j], v, now[j])
        out += v
    return out


def dst_base(cfg) -> float:
    """A defence's points at kickoff: the sheet's tiers for no points and no yards allowed."""
    from streamer.scoring import DstScoring

    sc = DstScoring.from_config(cfg)
    return float(sc.points_allowed_points(0)) + (float(sc.yards_allowed_points(0)) if sc.scores_yards_allowed else 0.0)


def dst_params(cfg) -> tuple[float, float]:
    """(decay, giveback) as static/live.js reads them from the sheet: the profile's
    ``live_decay`` / ``live_giveback``, else no decay and the shutout points."""
    d = cfg.dst_scoring
    return (float(d.get("live_decay", 0.0)), float(d.get("live_giveback", dst_base(cfg))))


def mapped(p, k):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return norm.cdf(np.asarray(k, float) * norm.ppf(p))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------------------
# Part one: the player model
# ---------------------------------------------------------------------------
def player_study(weekly: dict, units: pd.DataFrame | None, cfg, schedules: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Every projected starter-grade player-game at the end of each quarter:
    what he had, what he added, and what ``f m`` said he would."""
    from streamer.roster import league_replay as lr

    rows = []
    for season in SEASONS:
        so_far = SoFar(season, cfg, schedules, units=units is not None)
        points = lr.points_by_week(history, season)
        unit_week = {}
        if units is not None:
            unit_week, unit_points = lr.unit_rows(units, season)
            points.update(unit_points)
        for week in range(1, 18):
            for p in list(weekly[(season, week)]) + list(unit_week.get(week, [])):
                m = float(p.projection or 0.0)
                floor = 6.0 if p.position in ("QB", "RB", "WR", "TE") else 0.0
                if m < floor or so_far.game(week, p.team) is None or (week, p.player_id) not in points:
                    continue
                final = points[(week, p.player_id)]
                for q in STAGES[:-1]:
                    now = so_far.now(week, p, final, q)
                    rows.append((season, week, p.position, m, float(p.outcome_sd or p.projection_sd or 0.0),
                                 1 - q, now, final - now))
    return pd.DataFrame(rows, columns=["season", "week", "pos", "m", "sd", "f", "now", "rest"])


def touch_rows(weekly: dict, so_far: SoFar, season: int) -> pd.DataFrame:
    """Every starter-grade skill player-week (projected 6+): position, tag and
    when he first had a pass, carry or target (NaN: never)."""
    rows = []
    for week in range(1, 18):
        for p in weekly[(season, week)]:
            if p.position not in SKILL or float(p.projection or 0.0) < 6:
                continue
            g = so_far.game(week, p.team)
            if g is None:
                continue
            rows.append((season, week, p.position, p.status or "", so_far.first.get((g[1], p.player_id), np.nan)))
    d = pd.DataFrame(rows, columns=["season", "week", "pos", "status", "first"])
    d["played"] = d["first"].notna()
    return d


def report_defences(d: pd.DataFrame, base: float, dp: tuple[float, float] | None = None) -> None:
    """A defence in progress, as forecasts of its final score (RMSE by season): the blend first shipped,
    keeping what it has and giving back the shutout points (``base``), and -- the sheet's own two numbers
    ``dp`` = (decay, giveback), as shipped -- ``now (1 - decay f^2) + f (m - giveback)``; then those two
    numbers refitted on three seasons and scored on the fourth, with the fit on all four."""
    from scipy.optimize import least_squares

    x = d[d.pos == "DST"].copy()
    if x.empty:
        return
    x["final"] = x.now + x.rest
    dp = dp or (0.0, base)
    rm = lambda e: float(np.sqrt((np.asarray(e) ** 2).mean()))                              # noqa: E731
    form = lambda p, g: g.now * (1 - p[0] * g.f ** 2) + g.f * (g.m - p[1])                   # noqa: E731
    fit = lambda g: least_squares(lambda p: (g.final - form(p, g)).to_numpy(), [0.0, base]).x  # noqa: E731
    print(f"  a defence's final score from where it stands (shutout points {base:g}; shipped decay {dp[0]:g}, giveback {dp[1]:g}):")
    print("  RMSE by season: the blend (1 - f) now + f m  /  keeping what it has, giving back the shutout  /  the sheet's two numbers")
    for f, g in x.groupby("f"):
        blend = (1 - f) * g.now + f * g.m
        per = "  ".join(f"{s}: {rm((g.final - blend)[g.season == s]):.2f} / {rm((g.final - g.now - f * (g.m - base))[g.season == s]):.2f}"
                        f" / {rm((g.final - form(dp, g))[g.season == s]):.2f}" for s in SEASONS)
        print(f"  DST {1 - f:4.0%} played: n={len(g):5d}  {per}")
    print("  all quarter ends, RMSE: blend / keep / the sheet's two numbers, and the two numbers refitted on three seasons, held out:")
    held = np.concatenate([(x[x.season == s].final - form(fit(x[x.season != s]), x[x.season == s])).to_numpy() for s in SEASONS])
    blend = (1 - x.f) * x.now + x.f * x.m
    print(f"    {rm(x.final - blend):.3f} / {rm(x.final - x.now - x.f * (x.m - base)):.3f} / {rm(x.final - form(dp, x)):.3f}"
          f"   held out {rm(held):.3f}; fitted on all four: decay {fit(x)[0]:.2f}, giveback {fit(x)[1]:.2f}"
          f" (by season held out: {', '.join(f'{rm(x[x.season == s].final - form(fit(x[x.season != s]), x[x.season == s])):.3f}' for s in SEASONS)})")


def report_players(d: pd.DataFrame) -> None:
    print(f"\n=== The player model: {len(d) // 3} player-games at each quarter's end ===")
    print("  what he adds from here against f x projection, and its spread against sqrt(f) x his spread;")
    print("  'pace' is the weight on his pace so far (now / share played) that predicts the rest best")
    for grp, g in (("skill", d[d.pos.isin(["QB", "RB", "WR", "TE"])]), ("K", d[d.pos == "K"])):
        for f, x in g.groupby("f"):
            base = x.f * x.m
            pace = x.f * x.now.clip(lower=0) / (1 - x.f)
            w = float(((x.rest - base) * (pace - base)).sum() / ((pace - base) ** 2).sum())
            e0 = float(np.sqrt(((x.rest - base) ** 2).mean()))
            e1 = float(np.sqrt(((x.rest - (base + w * (pace - base))) ** 2).mean()))
            by = []
            for s in SEASONS:
                tr, te = x[x.season != s], x[x.season == s]
                bt, pt = tr.f * tr.m, tr.f * tr.now.clip(lower=0) / (1 - tr.f)
                ws = float(((tr.rest - bt) * (pt - bt)).sum() / ((pt - bt) ** 2).sum())
                be, pe = te.f * te.m, te.f * te.now.clip(lower=0) / (1 - te.f)
                by.append((ws, float(np.sqrt(((te.rest - be) ** 2).mean())),
                           float(np.sqrt(((te.rest - (be + ws * (pe - be))) ** 2).mean()))))
            better = sum(b[2] < b[1] for b in by)
            print(f"  {grp:5s} {1 - f:4.0%} played: n={len(x):6d}  rest {x.rest.mean():5.2f} vs f*m {base.mean():5.2f}"
                  f"  (ratio {x.rest.mean() / max(base.mean(), 1e-9):.3f})   sd of (rest - f*m) {(x.rest - base).std():5.2f}"
                  f" vs sqrt(f)*sd {(np.sqrt(x.f) * x.sd).mean():5.2f}   pace weight {w:+.3f}: RMSE {e0:.3f} -> {e1:.3f}"
                  f"  held out better in {better} of 4 (weights {', '.join(f'{b[0]:.2f}' for b in by)})")


# ---------------------------------------------------------------------------
# Part two: the matchup's P(win), state by state
# ---------------------------------------------------------------------------
PACES = (0.0, 0.05, 0.1)


def states(week: int, a_rows: list, b_rows: list, draws: np.ndarray, points: dict, so_far: SoFar,
           dp: tuple[float, float] | None = None, sat: np.ndarray | None = None, table: dict | None = None,
           shutout: float | None = None) -> list[dict]:
    """One head-to-head game read at every kickoff window's start, quarter
    ends and half: the raw live P(win) (for each pace weight) and the share
    of the pre-game margin variance still to play."""
    rows = list(a_rows) + list(b_rows)
    n_a = len(a_rows)
    from streamer.roster.lineup import _if_plays

    d = np.asarray(draws[:LIVE_SIMS], float)
    sat = np.zeros(d.shape, bool) if sat is None else np.asarray(sat[:LIVE_SIMS], bool)
    m = np.array([float(p.projection or 0.0) for p in rows])
    plays = [_if_plays(p) for p in rows]
    mp = np.array([x[0] if x[2] > 0 else 0.0 for x in plays])
    qs = [x[2] for x in plays]
    final = np.array([float(points.get((week, p.player_id), 0.0)) for p in rows])
    dst = np.array([p.position == "DST" for p in rows])
    win = [(so_far.game(week, p.team) or (None, None))[0] for p in rows]
    v0 = float((d[:, :n_a].sum(axis=1) - d[:, n_a:].sum(axis=1)).var())
    proj0 = float(m.sum())
    out, seen = [], set()
    for k, wi in enumerate(sorted({w for w in win if w is not None})):
        for q in (0.0, 0.25, 0.5, 0.75):
            if k == 0 and q == 0.0:
                continue                                    # nothing has kicked off: the pre-game number
            left = np.array([0.0 if w is None or w < wi else 1.0 if w > wi else 1.0 - q for w in win])
            key = tuple(left)
            if key in seen:
                continue
            seen.add(key)
            now = np.array([final[j] if left[j] <= 0 else 0.0 if left[j] >= 1 else
                            so_far.now(week, rows[j], final[j], round(1 - left[j], 2)) for j in range(len(rows))])
            rec = {"window": k, "played": q, "to_play": float((left * m).sum() / max(proj0, 1e-9)),
                   "in_progress": int(((left > 0) & (left < 1)).sum())}
            for pace in PACES:
                ta = totals(d[:, :n_a], m[:n_a], now[:n_a], left[:n_a], dst[:n_a], pace)
                tb = totals(d[:, n_a:], m[n_a:], now[n_a:], left[n_a:], dst[n_a:], pace)
                rec[f"raw{int(pace * 100)}"] = float((ta > tb).mean() + 0.5 * (ta == tb).mean())
                if pace == 0.0:
                    rec["left_blend"] = float(min((ta - tb).var() / v0, 1.0)) if v0 > 0 else 0.0
            # The defence in progress as shipped for this sheet ("raw" and "left"), and keeping what it has
            # and giving back only the shutout points (the ESPN-fitted first version, "raw_keep").
            ta = totals(d[:, :n_a], m[:n_a], now[:n_a], left[:n_a], dst[:n_a], 0.0, dp)
            tb = totals(d[:, n_a:], m[n_a:], now[n_a:], left[n_a:], dst[n_a:], 0.0, dp)
            rec["raw"] = float((ta > tb).mean() + 0.5 * (ta == tb).mean())
            rec["left"] = float(min((ta - tb).var() / v0, 1.0)) if v0 > 0 else 0.0
            ka = totals(d[:, :n_a], m[:n_a], now[:n_a], left[:n_a], dst[:n_a], 0.0, (0.0, shutout))
            kb = totals(d[:, n_a:], m[n_a:], now[n_a:], left[n_a:], dst[n_a:], 0.0, (0.0, shutout))
            rec["raw_keep"] = float((ka > kb).mean() + 0.5 * (ka == kb).mean())
            rec["dst_in_progress"] = int((dst & (left > 0) & (left < 1)).sum())
            if table is not None:
                # Whether each starter is playing, as the page now reads it from the box score.
                pp = []
                for j, p in enumerate(rows):
                    started = left[j] < 1
                    if p.position not in SKILL or not started:
                        pp.append(None)
                    else:
                        pp.append(playing(p.position, qs[j], so_far.touched(week, p, 1 - left[j]), 1 - left[j], table))
                ta = totals_live(d[:, :n_a], sat[:, :n_a], m[:n_a], mp[:n_a], now[:n_a], left[:n_a], dst[:n_a], pp[:n_a], dp)
                tb = totals_live(d[:, n_a:], sat[:, n_a:], m[n_a:], mp[n_a:], now[n_a:], left[n_a:], dst[n_a:], pp[n_a:], dp,
                                 col0=n_a)
                rec["raw_new"] = float((ta > tb).mean() + 0.5 * (ta == tb).mean())
                rec["left_new"] = float(min((ta - tb).var() / v0, 1.0)) if v0 > 0 else 0.0
                live = [j for j in range(len(rows)) if pp[j] is not None and 0 < left[j] < 1]
                rec["q_in_progress"] = int(sum(qs[j] < 1 for j in live))
                rec["unseen_in_progress"] = int(sum(pp[j] < 1 for j in live))
            out.append(rec)
    return out


K_END, K_POWER = 1.12, 2.0                # static/live.js: the exponent with nothing left, and the easing power


def k_live(left, k_end: float = K_END, power: float = K_POWER, k0: float = K0):
    """The exponent at a state: ``k0`` (the pre-game mapping) with everything
    to play, ``k_end`` with nothing: ``k_end - (k_end - k0) left^power``."""
    return k_end - (k_end - k0) * np.clip(np.asarray(left, float), 0, 1) ** power


def report_states(g: pd.DataFrame) -> None:
    y = g["won"].to_numpy()
    shipped = f"shipped: {K_END:g} - {K_END - K0:.2f} left^{K_POWER:g}"
    maps = {
        "as drawn (no mapping)": lambda d: d.raw.to_numpy(),
        f"pre-game mapping throughout (k={K0})": lambda d: mapped(d.raw, K0),
        "straight line from k to 1 (first shipped)": lambda d: mapped(d.raw, k_live(d.left, 1.0, 1.0)),
        shipped: lambda d: mapped(d.raw, k_live(d.left)),
    }
    print(f"\n=== The live P(win): {len(g)} in-game states of {g.groupby(['season', 'seed', 'week', 'a']).ngroups} games ===")
    print("  Brier (log loss) by season, each state against who won")
    for name, fn in maps.items():
        per = "  ".join(f"{s}: {brier(fn(gs), gs.won):.4f} ({logloss(fn(gs), gs.won):.4f})" for s, gs in g.groupby("season"))
        print(f"  {name:42s} {per}   all: {brier(fn(g), y):.5f} ({logloss(fn(g), y):.5f})")
    # Held out by season: the exponent's end and easing power picked on three seasons, scored on the fourth.
    # The start is the pre-game mapping's k, so the live number begins where the tile stood.
    grid = [(k1, a) for k1 in (1.0, 1.04, 1.08, 1.12, 1.16, 1.2) for a in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)]

    def score(d, k1, a):
        return logloss(mapped(d.raw, k_live(d.left, k1, a)), d.won)

    print("  held out by season (the exponent's end and easing power fitted on the other three):")
    tot = {"fit": 0.0, "line": 0.0, "raw": 0.0}
    over_raw = over_line = 0
    for s in sorted(g.season.unique()):
        tr, te = g[g.season != s], g[g.season == s]
        k1, a = min(grid, key=lambda ka: score(tr, *ka))
        fit, line, raw = score(te, k1, a), score(te, 1.0, 1.0), logloss(te.raw, te.won)
        over_raw += fit < raw
        over_line += fit < line
        for name, v in (("fit", fit), ("line", line), ("raw", raw)):
            tot[name] += v * len(te)
        print(f"    {s}: picked end={k1:g}, power={a:g}: log loss {fit:.5f} vs straight line {line:.5f} vs as drawn {raw:.5f}")
    print(f"    overall: fitted {tot['fit'] / len(g):.5f}, straight line {tot['line'] / len(g):.5f}, as drawn "
          f"{tot['raw'] / len(g):.5f}; fitted better than as drawn in {over_raw} of 4, than the straight line in {over_line} of 4")
    best = min(grid, key=lambda ka: score(g, *ka))
    print(f"    all four seasons together: end={best[0]:g}, power={best[1]:g} ({score(g, *best):.5f})")
    # How sure is the gain over the draws as they are? Leagues in a season share its NFL games, so resample NFL weeks.
    ll = lambda p: -(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1 - 1e-4)))  # noqa: E731
    blocks = pd.DataFrame({"season": g.season, "week": g.week, "d": ll(maps[shipped](g)) - ll(g.raw.to_numpy()),
                           "n": 1}).groupby(["season", "week"]).sum()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(blocks), (2000, len(blocks)))
    boot = blocks["d"].to_numpy()[idx].sum(axis=1) / blocks["n"].to_numpy()[idx].sum(axis=1)
    print(f"    shipped less as drawn, log loss, NFL weeks resampled: {boot.mean():+.5f} "
          f"(95% {np.quantile(boot, 0.025):+.5f} to {np.quantile(boot, 0.975):+.5f}; better in {np.mean(boot < 0):.0%} of resamples)")
    # Calibration, favourite's side, as drawn and as shipped; then by how much is left.
    for name in ("as drawn (no mapping)", shipped):
        p = maps[name](g)
        fav = p >= 0.5
        pf, yf = np.where(fav, p, 1 - p), np.where(fav, y, 1 - y)
        print(f"  calibration, {name} (favourite's side):")
        for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.97), (0.97, 1.01)):
            mk = (pf >= lo) & (pf < hi)
            if mk.sum():
                print(f"    {lo:4.0%}-{min(hi, 1):4.0%}: n={int(mk.sum()):6d}  predicted {pf[mk].mean():6.1%}  happened {yf[mk].mean():6.1%}")
    print("  by the share of the margin's uncertainty still to play: Brier as drawn / pre-game mapping / shipped,")
    print("  and the exponent that fits that slice best")
    for lo, hi in ((0.8, 1.01), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.05, 0.2), (0.0, 0.05)):
        d = g[(g.left >= lo) & (g.left < hi)]
        if len(d) < 50:
            continue
        ks = np.arange(0.5, 1.31, 0.02)
        kb = float(ks[int(np.argmin([logloss(mapped(d.raw, k), d.won) for k in ks]))])
        print(f"    {lo:4.0%}-{min(hi, 1):4.0%} left: n={len(d):6d}  {brier(d.raw, d.won):.4f} / {brier(mapped(d.raw, K0), d.won):.4f} / "
              f"{brier(mapped(d.raw, k_live(d.left)), d.won):.4f}   best exponent {kb:.2f} (shipped uses {k_live(d.left).mean():.2f})")
    d = g[g.dst_in_progress > 0]
    print(f"  a defence in progress (states with one, n={len(d)}; as drawn, Brier by season):")
    for name, c in (("blend (1 - f) now + f draw, first shipped", "raw0"), ("keeps what it has, gives back the shutout", "raw_keep"),
                    ("this sheet's two numbers (shipped)", "raw")):
        per = "  ".join(f"{s}: {brier(gs[c], gs.won):.5f}" for s, gs in d.groupby("season"))
        print(f"    {name:46s} {per}   all: {brier(d[c], d.won):.5f}")
    if "raw_new" in g:
        print("  whether a starter is playing, read from the box score (raw draws, then each through the shipped mapping):")
        for name, d in (("all states", g), ("a questionable starter in progress", g[g.q_in_progress > 0]),
                        ("a starter in progress with no touch yet", g[g.unseen_in_progress > 0])):
            line = []
            for c, lc in (("raw", "left"), ("raw_new", "left_new")):
                pm = mapped(d[c], k_live(d[lc]))
                per = " ".join(f"{s}: {logloss(pm[d.season == s], d.won[d.season == s]):.5f}" for s in sorted(d.season.unique()))
                line.append(f"{'as before' if c == 'raw' else 'reading it':10s} Brier {brier(pm, d.won):.5f} log loss {logloss(pm, d.won):.5f} ({per})")
            print(f"    {name} (n={len(d)}):")
            for x in line:
                print(f"      {x}")
    print("  the weight on a player's pace so far (as drawn, Brier by season):")
    for pace in PACES:
        c = f"raw{int(pace * 100)}"
        d = g[g.in_progress > 0]
        per = "  ".join(f"{s}: {brier(gs[c], gs.won):.5f}" for s, gs in d.groupby("season"))
        print(f"    pace {pace:.2f}: {per}   all: {brier(d[c], d.won):.5f}  (states with a game in progress, n={len(d)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", type=int, default=6, help="leagues per season")
    ap.add_argument("--weekly", help="cache of the weekly projections (built if missing)")
    ap.add_argument("--units", help="cache of the defence and kicker projections (built if missing)")
    ap.add_argument("--no-units", action="store_true")
    ap.add_argument("--out", help="a directory for the graded states (parquet)")
    ap.add_argument("--players-only", action="store_true")
    ap.add_argument("--profile", default="espn",
                    help="whose D/ST and K scoring and walk-forward unit projections (the weekly skill cache is "
                         "the ESPN-sheet one: with another profile only the defence and kicker results mean something)")
    args = ap.parse_args()
    os.environ.setdefault("STREAMER_CREDIT_FREE", "1")
    logging.disable(logging.WARNING)
    import grade_league_odds as base

    from streamer.config import get_config
    from streamer.roster import league_replay as lr
    from streamer.roster import ros_backtest as rb
    from streamer.roster.projections import load_history

    cfg = get_config().for_profile(args.profile)
    weekly = base.weekly_projections(get_config().for_profile("espn"), args.weekly)
    unit_frame = None if args.no_units else base.unit_projections(cfg, args.units)
    history = load_history(cfg)
    schedules = pd.read_parquet(cfg.raw_dir / "schedules.parquet")
    players = player_study(weekly, unit_frame, cfg, schedules, history)
    if args.profile == "espn":
        report_players(players)
    report_defences(players, dst_base(cfg), dst_params(cfg))
    if args.players_only:
        return 0
    rows = []
    dp, shutout = dst_params(cfg), dst_base(cfg)
    sofars = {s: SoFar(s, cfg, schedules, units=unit_frame is not None) for s in SEASONS}
    touches = pd.concat([touch_rows(weekly, sofars[s], s) for s in SEASONS], ignore_index=True)
    print("\nno pass, carry or target yet, by share of the game played (all four seasons; static/live.js NONE_YET):")
    for pos, ys in none_yet_table(touches).items():
        print(f"  {pos}: {ys}")
    print(f"  a starter with no injury tag played {touches[touches.status == ''].played.mean():.3f}")
    for season in SEASONS:
        so_far = sofars[season]
        table = none_yet_table(touches[touches.season != season])        # held out: the other three seasons
        points = lr.points_by_week(history, season)
        units = None
        if unit_frame is not None:
            units, unit_points = lr.unit_rows(unit_frame, season)
            points.update(unit_points)
        byes = rb.byes(schedules, season, list(range(1, 18)))
        per_week = {w: weekly[(season, w)] for w in range(1, 18)}
        for i in range(args.leagues):
            seed = 1000 * season + i
            got: list[dict] = []

            def on_game(week, a, b, la, lb, draws, sat, got=got, points=points, so_far=so_far, table=table):
                for r in states(week, la, lb, draws, points, so_far, dp, sat, table, shutout):
                    got.append({"week": week, "a": a, "b": b, **r})

            r = lr.play(season, seed, per_week, points, byes, cfg, units=units, checkpoints=(), on_game=on_game)
            won = {(g["week"], g["a"]): g["won"] for g in r.games}
            rows += [{"season": season, "seed": seed, **x, "won": won[(x["week"], x["a"])]} for x in got]
            print(f"{season} league {i}: {len(got)} states", flush=True)
    g = pd.DataFrame(rows)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        g.to_parquet(Path(args.out) / "live_states.parquet")
        players.to_parquet(Path(args.out) / "live_players.parquet")
    report_states(g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
