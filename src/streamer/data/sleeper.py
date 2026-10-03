"""Sleeper's free player database: depth charts and injury details.

``https://api.sleeper.app/v1/players/nfl`` needs no key and returns every NFL
player with his team's current depth-chart position and order, and injury
details the fantasy platforms' tags leave out -- body part, when it began,
practice participation, notes. Sleeper asks that it be pulled at most once a
day (about 5 MB), so it is cached for ``SLEEPER_MAX_AGE_HOURS`` under
``data/raw/`` (restored between runs, ignored by git).

The sandbox this was written in cannot reach Sleeper, so the first step is a
probe the workflow can run (``streamer sleeper-probe``): status, size, which
fields are filled, and the depth charts of a few backfields -- public data,
safe for a public log.
"""

from __future__ import annotations

import json
import logging
import time

from ..config import Config

log = logging.getLogger(__name__)

URL = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_MAX_AGE_HOURS = 20.0
FIELDS = ("depth_chart_position", "depth_chart_order", "injury_status", "injury_body_part",
          "injury_start_date", "injury_notes", "practice_participation", "practice_description",
          "news_updated", "status", "gsis_id", "espn_id", "yahoo_id")


def players(cfg: Config, refresh: bool = False) -> dict:
    """Player id -> Sleeper's record, cached for about a day. {} on failure
    with no cache."""
    path = cfg.raw_dir / "sleeper_players.json"
    fresh = path.exists() and time.time() - path.stat().st_mtime < SLEEPER_MAX_AGE_HOURS * 3600
    if fresh and not refresh:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    import requests

    try:
        resp = requests.get(URL, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - a bonus source: fall back to the last pull
        log.warning("Sleeper players failed: %s", type(exc).__name__)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


#: Positions whose depth order decides who is next in line.
DEPTH_POSITIONS = ("QB", "RB", "TE")


def depth_orders(cfg: Config, allow_network: bool = True) -> dict[tuple[str, str, str], int]:
    """(team, position, normalised name) -> depth-chart order (1 = starter)
    for QBs, RBs and TEs. Sleeper's ids for the other platforms are mostly
    blank, so players are matched by name on the same team and position.
    {} when Sleeper cannot be reached and nothing is cached."""
    from ..roster.players import normalize_name
    from ..teams import normalize_team

    if allow_network:
        data = players(cfg)
    else:
        path = cfg.raw_dir / "sleeper_players.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except ValueError:
            data = {}
    out = {}
    for p in (data or {}).values():
        if not isinstance(p, dict) or not p.get("team") or p.get("depth_chart_order") is None:
            continue
        pos = p.get("depth_chart_position") or p.get("position")
        if pos not in DEPTH_POSITIONS or p.get("position") not in DEPTH_POSITIONS:
            continue
        name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}"
        try:
            out[(normalize_team(p["team"]), p["position"], normalize_name(name))] = int(p["depth_chart_order"])
        except (TypeError, ValueError):
            continue
    return out


def probe(cfg: Config) -> list[str]:
    """What Sleeper returns, for a public log: counts, how often each field
    is filled for active skill players, and a few depth charts."""
    import requests

    try:
        resp = requests.get(URL, timeout=60)
    except Exception as exc:  # noqa: BLE001
        return [f"{URL}: {type(exc).__name__}"]
    lines = [f"{URL}: HTTP {resp.status_code}, {len(resp.content) / 1e6:.1f} MB"]
    if resp.status_code != 200:
        return lines
    data = resp.json()
    skill = [p for p in data.values() if isinstance(p, dict) and p.get("active") and p.get("team")
             and p.get("position") in ("QB", "RB", "WR", "TE")]
    lines.append(f"{len(data)} players, {len(skill)} active skill players on a team")
    for f in FIELDS:
        filled = sum(1 for p in skill if p.get(f) not in (None, "", []))
        lines.append(f"  {f}: filled for {filled} ({filled / max(len(skill), 1):.0%})")
    for team in ("NYJ", "MIA", "GB", "PHI"):
        pos = "TE" if team == "PHI" else "RB"
        room = sorted((p for p in skill if p.get("team") == team and p.get("position") == pos),
                      key=lambda p: (p.get("depth_chart_order") or 99))
        lines.append(f"  {team} {pos}: " + "; ".join(
            f"{p.get('full_name')} #{p.get('depth_chart_order')} {p.get('injury_status') or 'healthy'}"
            + (f" ({p.get('injury_body_part')}, since {p.get('injury_start_date')})" if p.get("injury_status") else "")
            for p in room[:4]))
    return lines
