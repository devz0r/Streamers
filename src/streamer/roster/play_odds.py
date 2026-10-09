"""FantasyPros' chance to play, as a second read on a questionable player.

Our chance that a questionable or doubtful player suits up comes from his tag
and his last practice (fitted on 2022-2025). FantasyPros publishes its own
("Are they playing?"), which can carry what a practice report cannot -- a
coach's word, a beat reporter. There is no archive of it to test on, so it
earns its say the way the consensus projection did: every run logs both
numbers for each tagged player whose game is still to come, each week is
graded against who actually played, and this week's chance is ours moved
``w`` of the way toward theirs, ``w`` the Brier-minimising weight shrunk
toward 0 -- no say until ``min_cases`` are graded. Their number is never
shown; only the chance it helped set.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..league.model import LeagueSnapshot

log = logging.getLogger(__name__)

#: Kept with FantasyPros' other data on the runner (data/raw/fantasypros,
#: cached between runs, never committed): their numbers are not published.
LOG = "play_log.parquet"


def log_path(cfg: Config):
    from ..data import fantasypros as fp

    return fp.cache_dir(cfg) / LOG
TAGGED = ("QUESTIONABLE", "Q", "DOUBTFUL", "D")
#: The weight this run's chances use (set by the publish step; 0: ours alone).
WEIGHT = 0.0


def conf(cfg: Config) -> dict:
    return (cfg.raw.get("fantasypros") or {}).get("play_odds") or {}


def attach(snapshot: LeagueSnapshot, cfg: Config) -> int:
    """Put FantasyPros' chance to play on every questionable or doubtful
    skill player it lists (``fp_play``). Returns how many."""
    from ..data import fantasypros as fp
    from .consensus import SKILL, _find, _index

    for p in snapshot.all_players():
        p.fp_play = None
    if not conf(cfg) or (not fp.api_key() and not any(fp.cache_dir(cfg).glob("injuries*.json"))):
        return 0
    try:
        inj, names = fp.injuries(cfg), fp.players(cfg)
    except Exception as exc:  # noqa: BLE001 - a bonus input
        log.warning("FantasyPros injuries unavailable: %s", type(exc).__name__)
        return 0
    if inj.empty or names.empty:
        return 0
    rows = names.assign(fp_id=names["fp_id"].astype(str)).merge(
        inj[["fp_id", "play"]].assign(fp_id=inj["fp_id"].astype(str)), on="fp_id", how="inner")
    idx = _index(rows)
    n = 0
    for p in snapshot.all_players():
        if p.position in SKILL and str(p.status or "").upper() in TAGGED:
            r = _find(p, idx)
            if r is not None and r.play is not None and np.isfinite(r.play):
                p.fp_play = float(min(max(r.play, 0.0), 1.0))
                n += 1
    return n


def record(snapshot: LeagueSnapshot, cfg: Config, now: datetime | None = None) -> int:
    """Log ours and theirs for each tagged player whose game is still to come
    (the last read before kickoff is the one graded)."""
    from .projections import tag_probability

    now = now or datetime.now(UTC)
    rows = []
    for p in snapshot.all_players():
        if p.fp_play is None or p.locked or p.actual_points is not None or not getattr(p, "nfl_id", None):
            continue
        rows.append({"season": int(snapshot.season), "week": int(snapshot.week), "nfl_id": str(p.nfl_id),
                     "name": p.name, "position": p.position, "status": p.status, "practice": p.practice or "",
                     "ours": tag_probability(p, cfg), "theirs": p.fp_play, "at": now.isoformat()})
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
    """Every logged player-week, both leagues, whose week has been played:
    ``played`` is whether he has a box score that week."""
    path = log_path(cfg)                 # both leagues share it: one row per player-week
    frames = [pd.read_parquet(path)] if path.exists() else []
    if not frames or history is None or history.empty:
        return pd.DataFrame()
    log_ = pd.concat(frames).sort_values("at").drop_duplicates(["season", "week", "nfl_id"], keep="last")
    last = history[["season", "week"]].drop_duplicates()
    done = log_.merge(last, on=["season", "week"], how="inner")          # weeks the history has
    box = set(zip(history["season"], history["week"], history["player_id"].astype(str)))
    done = done.assign(played=[float((s, w, i) in box) for s, w, i in zip(done["season"], done["week"], done["nfl_id"])])
    return done


def fit_weight(cfg: Config, history: pd.DataFrame) -> tuple[float, int]:
    """How far toward FantasyPros' chance ours moves, and on how many graded
    player-weeks: the Brier-minimising weight on a 0-1 grid, shrunk toward
    ``prior`` (0) by ``prior_cases``; the prior stands below ``min_cases``."""
    c = conf(cfg)
    prior, k, need = float(c.get("prior", 0.0)), float(c.get("prior_cases", 40)), int(c.get("min_cases", 40))
    g = graded(cfg, history)
    n = len(g)
    if n < need:
        return prior, n
    ours, theirs, y = (g[col].to_numpy(float) for col in ("ours", "theirs", "played"))
    grid = np.linspace(0.0, 1.0, 21)
    brier = [float(np.mean((ours + w * (theirs - ours) - y) ** 2)) for w in grid]
    fit = float(grid[int(np.argmin(brier))])
    return (n * fit + k * prior) / (n + k), n


def blend(ours: float, theirs: float | None, weight: float | None = None) -> float:
    w = WEIGHT if weight is None else weight
    if theirs is None or w <= 0.0:
        return ours
    return float(ours + w * (theirs - ours))
