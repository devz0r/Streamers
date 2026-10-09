"""FantasyPros consensus rankings and projections.

Needs ``FANTASYPROS_API_KEY`` (a repository secret, or ``.env`` locally --
never in code or chat). What comes back is FantasyPros' data: it is cached on
the runner under ``data/raw/fantasypros/`` (restored between runs by the
workflow's cache, ignored by git) and never committed or published raw. The
page shows only analysis derived from it -- accuracy scores, whether the
consensus agrees with us -- credited to FantasyPros as their terms require.
Their numbers themselves are never shown: that is redistribution, which the
standard API licence does not grant.

Three pulls, per position (QB, RB, WR, TE):

* rest-of-season consensus rankings (ECR) -- the market's season view;
* this week's consensus rankings -- graded against results;
* this week's projections -- graded against results, and one more number for
  what the other manager sees.

Each pull is cached for ``fantasypros.cache_hours`` so a busy day of refreshes
spends a handful of calls.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)

BASE = "https://api.fantasypros.com/public/v2/json/nfl"
POSITIONS = ("QB", "RB", "WR", "TE")


def api_key() -> str | None:
    return os.environ.get("FANTASYPROS_API_KEY", "").strip() or None


def conf(cfg: Config) -> dict:
    return cfg.raw.get("fantasypros") or {}


def cache_dir(cfg: Config) -> Path:
    d = cfg.data_dir / "raw" / "fantasypros"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(path: str, params: dict[str, Any], cfg: Config, timeout: float = 20.0,
         hours: float | None = None) -> dict | None:
    """One API call through the cache. None when there is no key or it fails."""
    name = path.strip("/").replace("/", "_") + "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    cached = cache_dir(cfg) / f"{name}.json"
    hours = float(conf(cfg).get("cache_hours", 6)) if hours is None else float(hours)
    if cached.exists() and time.time() - cached.stat().st_mtime < hours * 3600:
        try:
            return json.loads(cached.read_text(encoding="utf-8"))
        except ValueError:
            pass
    key = api_key()
    if not key:
        return None
    import requests

    try:
        resp = requests.get(f"{BASE}{path}", params=params, headers={"x-api-key": key}, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        log.warning("FantasyPros %s failed: %s", path, type(exc).__name__)
        return None
    if resp.status_code != 200:
        log.warning("FantasyPros %s: HTTP %s", path, resp.status_code)
        return None
    try:
        data = resp.json()
    except ValueError:
        log.warning("FantasyPros %s: not JSON", path)
        return None
    cached.write_text(json.dumps(data), encoding="utf-8")
    return data


def _players(data: dict | None) -> list[dict]:
    if not data:
        return []
    for key in ("players", "data", "rankings", "projections"):
        rows = data.get(key)
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    return []


def _first(row: dict, *keys: str) -> Any:
    for k in keys:
        if row.get(k) not in (None, ""):
            return row[k]
    return None


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pos_rank(row: dict) -> int | None:
    v = _first(row, "pos_rank", "position_rank")
    if v is None:
        return None
    digits = "".join(ch for ch in str(v) if ch.isdigit())
    return int(digits) if digits else None


def rankings(cfg: Config, season: int, week: int, kind: str) -> pd.DataFrame:
    """Consensus rankings, ``kind`` "ros" (rest of season) or "weekly":
    name, team, position, overall rank, position rank."""
    scoring = conf(cfg).get("scoring", "PPR")
    rows = []
    for pos in POSITIONS:
        params = {"position": pos, "type": kind.upper(), "scoring": scoring}
        if kind == "weekly":
            params["week"] = week
        for r in _players(_get(f"/{season}/consensus-rankings", params, cfg)):
            rows.append({
                "name": _first(r, "player_name", "name"),
                "team": _first(r, "player_team_id", "team_id", "team"),
                "position": _first(r, "player_position_id", "position_id", "position") or pos,
                "rank": _num(_first(r, "rank_ecr", "rank")),
                "pos_rank": _pos_rank(r),
                "yahoo_id": _first(r, "player_yahoo_id"),
                "fp_id": _first(r, "player_id", "id"),
            })
    return pd.DataFrame(rows, columns=["name", "team", "position", "rank", "pos_rank", "yahoo_id", "fp_id"])


def special_rankings(cfg: Config, season: int, week: int) -> pd.DataFrame:
    """The weekly expert consensus at D/ST and K: position, team, rank. Empty
    when the API answers with another week than the one asked for (a past
    week it no longer serves)."""
    rows = []
    for pos in ("DST", "K"):
        data = _get(f"/{season}/consensus-rankings", {"position": pos, "type": "WEEKLY", "week": week},
                    cfg, hours=24 * 365 if week else None)
        got = (data or {}).get("week")
        if got not in (None, "") and str(got) != str(week):
            continue
        for r in _players(data):
            rank = _num(_first(r, "rank_ecr", "rank"))
            team = _first(r, "player_team_id", "team_id", "team")
            if rank is not None and team:
                rows.append({"position": pos, "team": str(team), "rank": rank})
    return pd.DataFrame(rows, columns=["position", "team", "rank"])


def players(cfg: Config) -> pd.DataFrame:
    """FantasyPros' player list: name, team, position, fp_id -- to put names
    on records that carry only their id (the injury report)."""
    rows = []
    for r in _players(_get("/players", {}, cfg, hours=24)):
        rows.append({"name": _first(r, "player_name", "name"), "team": _first(r, "team_id", "player_team_id"),
                     "position": _first(r, "position_id", "player_position_id"), "fp_id": _first(r, "player_id")})
    return pd.DataFrame(rows, columns=["name", "team", "position", "fp_id"])


def injuries(cfg: Config) -> pd.DataFrame:
    """FantasyPros' injury report: fp_id, status, and their chance he plays
    this week (0-1) -- the site's "Are they playing?". Refreshed every two
    hours (``injury_cache_hours``): it moves with each practice report."""
    data = _get("/injuries", {}, cfg, hours=float(conf(cfg).get("injury_cache_hours", 2)))
    rows = []
    for r in (data or {}).get("injuries") or []:
        if not isinstance(r, dict):
            continue
        prob = _num(r.get("probability_of_playing"))
        if prob is None:
            continue
        rows.append({"fp_id": _first(r, "player_id"), "status": _first(r, "status_short", "status"),
                     "play": prob / 100.0 if prob > 1.0 else prob, "updated": _first(r, "injury_update_date")})
    return pd.DataFrame(rows, columns=["fp_id", "status", "play", "updated"])


def projections(cfg: Config, season: int, week: int) -> pd.DataFrame:
    """This week's projected fantasy points: name, team, position, points."""
    scoring = conf(cfg).get("scoring", "PPR")
    field = {"PPR": ("points_ppr", "ppr_points"), "HALF": ("points_half", "half_ppr_points")}.get(
        scoring, ("points", "fpts"))
    rows = []
    for pos in POSITIONS:
        for r in _players(_get(f"/{season}/projections", {"position": pos, "week": week, "scoring": scoring}, cfg)):
            stats = r.get("stats") if isinstance(r.get("stats"), dict) else r
            pts = _num(_first(stats, *field, "points", "fpts"))
            rows.append({
                "name": _first(r, "player_name", "name"),
                "team": _first(r, "player_team_id", "team_id", "team"),
                "position": _first(r, "player_position_id", "position_id", "position") or pos,
                "points": pts,
            })
    return pd.DataFrame(rows, columns=["name", "team", "position", "points"])


#: Field names that could carry a chance to play (names only are printed).
PLAY_FIELD = re.compile(r"chance|prob|pct|percent|likel|play|injur|status|practice|designation|game_?status", re.I)


def _key_names(data, depth: int = 4) -> set[str]:
    """Every field name in a JSON document, a few levels deep -- names only."""
    names: set[str] = set()

    def walk(x, d):
        if d > depth:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                names.add(str(k))
                walk(v, d + 1)
        elif isinstance(x, list):
            for v in x[:20]:
                walk(v, d + 1)

    walk(data, 0)
    return names


def probe(cfg: Config, season: int, week: int) -> list[str]:
    """What the API returns, in terms safe for a public log: status, field
    names and counts -- never values, never the key."""
    import requests

    key = api_key()
    if not key:
        return ["FANTASYPROS_API_KEY is not set"]
    out = []
    # Where a chance to play could live (the site's "Are they playing?"): an
    # injuries endpoint, or fields on the news and player records. The
    # rankings and projections calls this probe first made are in use now.
    calls = [("/injuries", {}),
             (f"/{season}/injuries", {}),
             (f"/{season}/injuries", {"week": week}),
             ("/news", {"limit": 5, "category": "injury"}),
             ("/players", {"limit": 5}),
             (f"/{season - 1}/consensus-rankings", {"position": "DST", "type": "WEEKLY", "week": 10}),
             (f"/{season - 2}/consensus-rankings", {"position": "K", "type": "WEEKLY", "week": 10})]
    for path, params in calls:
        try:
            resp = requests.get(f"{BASE}{path}", params=params, headers={"x-api-key": key}, timeout=20)
        except Exception as exc:  # noqa: BLE001
            out.append(f"{path} {params}: {type(exc).__name__}")
            continue
        line = f"{path} {params}: HTTP {resp.status_code}"
        try:
            data = resp.json()
        except ValueError:
            out.append(line + ", not JSON")
            continue
        if isinstance(data, dict):
            line += f", top-level keys {sorted(data)[:15]}"
            rows = _players(data)
            line += f", {len(rows)} player rows"
            if rows:
                line += f", row fields {sorted(rows[0])[:40]}"
                stats = rows[0].get("stats")
                if isinstance(stats, dict):
                    line += f", stats fields {sorted(stats)[:30]}"
        if isinstance(data, (dict, list)):
            names = _key_names(data)
            line += f", {len(names)} distinct field names"
            hits = sorted(n for n in names if PLAY_FIELD.search(n))
            line += f", play/injury fields {hits[:30] if hits else 'none'}"
        out.append(line)
    return out
