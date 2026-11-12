"""Whether a book still posts props on a questionable player, as a read on whether he plays.

A book that expects a player out takes his lines down; one that expects him to
play leaves them up. For a questionable or doubtful player whose game's props
were pulled, the signal is whether *he* is in them (``prop_lines``: how many
lines he has), against his teammates. A player whose game has not been pulled
says nothing: the pull is paced by credits, so an absent line is mostly an
unpulled game.

There is no archive of props to test this on (the feeds keep none), so it
earns its say as the FantasyPros chance to play did (:mod:`.play_odds`):
every run logs ours and whether he has lines, each week is graded against who
played, and it moves his chance by a log-odds step per group (has lines /
none) fitted on those weeks and shrunk toward zero -- no say for a group until
``min_cases`` are graded. Only the chance it helped set is shown. Players
tagged out have no chance to move and are not read.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..league.model import LeagueSnapshot

log = logging.getLogger(__name__)

LOG = "prop_play_log.parquet"
TAGGED = ("QUESTIONABLE", "Q", "DOUBTFUL", "D")
#: Players with lines in a game before it counts as pulled (a stray alt line is not a pull).
MIN_GAME_PLAYERS = 8
#: The log-odds step for each group this run's chances use (set by the publish step; 0: no say).
STEP = {"has": 0.0, "none": 0.0}


def conf(cfg: Config) -> dict:
    return (cfg.raw.get("roster") or {}).get("prop_presence") or {}


def log_path(cfg: Config):
    return cfg.path("results") / LOG


def _lines(frame: pd.DataFrame) -> dict[str, dict[str, set]]:
    """event id -> normalised player name -> the (book, stat) lines he has."""
    from .players import normalize_name

    out: dict[str, dict[str, set]] = {}
    if frame is None or frame.empty:
        return out
    for ev, who, book, stat in zip(frame["event_id"], frame["player"], frame["bookmaker"], frame["stat"]):
        out.setdefault(str(ev), {}).setdefault(normalize_name(who), set()).add((book, stat))
    return out


def attach(snapshot: LeagueSnapshot, frame: pd.DataFrame | None) -> int:
    """Set ``prop_lines`` on each questionable or doubtful skill player whose
    game's props were pulled: how many lines he has (0: a book has taken him
    down). None where the game was not pulled. Returns how many are set."""
    from .players import normalize_name

    for p in snapshot.all_players():
        p.prop_lines = None
    events = _lines(frame)
    if not events:
        return 0
    team_of = {normalize_name(p.name): p.team for p in snapshot.all_players() if p.team}
    game_teams: dict[str, set[str]] = {}
    for ev, who in events.items():
        votes: dict[str, int] = {}
        for name in who:
            t = team_of.get(name)
            if t:
                votes[t] = votes.get(t, 0) + 1
        top = sorted(votes.items(), key=lambda kv: -kv[1])[:2]
        if len(who) >= MIN_GAME_PLAYERS and top:
            game_teams[ev] = {t for t, n in top if n >= 2}
    n = 0
    for p in snapshot.all_players():
        if p.position not in ("QB", "RB", "WR", "TE") or str(p.status or "").upper() not in TAGGED or not p.team:
            continue
        for ev, teams in game_teams.items():
            if p.team in teams:
                p.prop_lines = len(events[ev].get(normalize_name(p.name), ()))
                n += 1
                break
    return n


def record(snapshot: LeagueSnapshot, cfg: Config, now: datetime | None = None) -> int:
    """Log ours and whether he has lines, for each tagged player whose game is
    still to come (the last read before kickoff is the one graded)."""
    from .projections import tag_probability

    now = now or datetime.now(UTC)
    rows = []
    for p in snapshot.all_players():
        if getattr(p, "prop_lines", None) is None or p.locked or p.actual_points is not None \
                or not getattr(p, "nfl_id", None):
            continue
        rows.append({"season": int(snapshot.season), "week": int(snapshot.week), "nfl_id": str(p.nfl_id),
                     "name": p.name, "position": p.position, "status": p.status, "practice": p.practice or "",
                     "ours": tag_probability(p, cfg), "lines": int(p.prop_lines), "at": now.isoformat()})
    if not rows:
        return 0
    path = log_path(cfg)
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    out = out.sort_values("at").drop_duplicates(["season", "week", "nfl_id"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return len(rows)


def graded(cfg: Config, history: pd.DataFrame) -> pd.DataFrame:
    """Every logged player-week whose week has been played: ``played`` is
    whether he has a box score that week."""
    path = log_path(cfg)
    if not path.exists() or history is None or history.empty:
        return pd.DataFrame()
    log_ = pd.read_parquet(path).sort_values("at").drop_duplicates(["season", "week", "nfl_id"], keep="last")
    weeks = history[["season", "week"]].drop_duplicates()
    done = log_.merge(weeks, on=["season", "week"], how="inner")
    box = set(zip(history["season"], history["week"], history["player_id"].astype(str)))
    return done.assign(played=[float((s, w, i) in box) for s, w, i in zip(done["season"], done["week"], done["nfl_id"])])


def _logit(p):
    p = np.clip(p, 0.02, 0.98)
    return np.log(p / (1.0 - p))


def fit_steps(cfg: Config, history: pd.DataFrame) -> tuple[dict[str, float], dict[str, int]]:
    """The log-odds step for each group, and the graded cases in each: the
    Brier-minimising step on a grid, shrunk toward 0 by ``prior_cases``; 0
    below ``min_cases``."""
    c = conf(cfg)
    need, k = int(c.get("min_cases", 30)), float(c.get("prior_cases", 40))
    g = graded(cfg, history)
    steps, counts = {"has": 0.0, "none": 0.0}, {"has": 0, "none": 0}
    if g.empty:
        return steps, counts
    g = g[(g["ours"] > 0) & (g["ours"] < 1)]
    grid = np.arange(-3.0, 3.01, 0.25)
    for name, part in (("has", g[g["lines"] > 0]), ("none", g[g["lines"] == 0])):
        n = len(part)
        counts[name] = n
        if n < need:
            continue
        z, y = _logit(part["ours"].to_numpy(float)), part["played"].to_numpy(float)
        brier = [float(np.mean((1.0 / (1.0 + np.exp(-(z + d))) - y) ** 2)) for d in grid]
        steps[name] = float(grid[int(np.argmin(brier))]) * n / (n + k)
    return steps, counts


def adjust(ours: float, lines: int | None, step: dict[str, float] | None = None) -> float:
    """His chance moved by the group's step; unchanged for a game not pulled."""
    s = STEP if step is None else step
    if lines is None or not 0.0 < ours < 1.0:
        return ours
    d = s["has"] if lines > 0 else s["none"]
    if d == 0.0:
        return ours
    return float(1.0 / (1.0 + np.exp(-(_logit(ours) + d))))
