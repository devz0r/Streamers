"""Trades made in the league, found from the rosters and priced for both teams.

No platform feed is read: every refresh keeps each team's roster
(``rosters.json`` beside the league's snapshots), and a trade is players
moving straight from one team's roster to another's *both ways* between two
refreshes. A one-way move is a drop and a claim, not a trade -- on waivers a
dropped player takes days to reach another roster, and a refresh is hours
apart, so a two-way swap between the same two teams is a trade.

Found trades are kept in ``trades.json`` and priced on the page's evaluator
data (:mod:`streamer.roster.trade_eval`): what the trade did is the opposite
of undoing it today -- each team giving back what it got -- so it is read on
the current rosters and the current simulation.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from ..config import Config
from ..league.model import LeagueSnapshot

log = logging.getLogger(__name__)

ROSTERS = "rosters.json"
TRADES = "trades.json"


def _dir(cfg: Config) -> Path:
    from ..league.store import snapshot_dir

    return snapshot_dir(cfg)


def rosters(snapshot: LeagueSnapshot) -> dict[str, dict[str, str]]:
    """team id -> {player id: name}."""
    return {str(t.team_id): {p.player_id: p.name for p in t.roster} for t in snapshot.teams}


def detect(before: dict[str, dict[str, str]], after: dict[str, dict[str, str]],
           names: dict[str, str]) -> list[dict]:
    """Two-way swaps between ``before`` and ``after`` rosters, one per pair of teams."""
    out = []
    ids = sorted(set(before) & set(after))
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            a_got = sorted(set(before[b]) & set(after[a]))
            b_got = sorted(set(before[a]) & set(after[b]))
            if a_got and b_got:
                out.append({"a": a, "a_name": names.get(a, a), "a_got": a_got,
                            "a_got_names": [before[b][p] for p in a_got],
                            "b": b, "b_name": names.get(b, b), "b_got": b_got,
                            "b_got_names": [before[a][p] for p in b_got]})
    return out


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def logged(cfg: Config, season: int) -> list[dict]:
    """The trades logged this season, without looking for new ones."""
    return [t for t in _read(_dir(cfg) / TRADES, []) if t.get("season") == season]


def record(cfg: Config, snapshot: LeagueSnapshot, now: datetime | None = None) -> list[dict]:
    """Compare this sync's rosters with the last ones kept, log any new trade,
    keep these rosters, and return every trade logged this season."""
    d = _dir(cfg)
    log_path, ros_path = d / TRADES, d / ROSTERS
    trades = [t for t in _read(log_path, []) if t.get("season") == snapshot.season]
    kept = _read(ros_path, {})
    current = rosters(snapshot)
    if kept.get("season") == snapshot.season and kept.get("teams"):
        names = {str(t.team_id): t.name for t in snapshot.teams}
        when = (now or datetime.now(UTC)).date().isoformat()
        for t in detect(kept["teams"], current, names):
            t.update({"season": snapshot.season, "week": snapshot.week, "seen": when})
            trades.append(t)
            log.info("%s: trade found: %s gets %s, %s gets %s", cfg.profile, t["a_name"],
                     ", ".join(t["a_got_names"]), t["b_name"], ", ".join(t["b_got_names"]))
    try:
        d.mkdir(parents=True, exist_ok=True)
        ros_path.write_text(json.dumps({"season": snapshot.season, "teams": current}, indent=1, sort_keys=True),
                            encoding="utf-8")
        log_path.write_text(json.dumps(trades, indent=1), encoding="utf-8")
    except OSError as exc:
        log.warning("could not keep rosters for trade detection: %s", exc)
    return trades


def price(data: dict, trade: dict) -> dict | None:
    """What ``trade`` did, on today's rosters: each team's title and playoff
    odds with it against without it (undone), and yours. None when it can no
    longer be undone (a player in it has since left the team that got him)."""
    from . import trade_eval

    teams = {t["id"]: t for t in data["teams"]}
    a, b = teams.get(trade["a"]), teams.get(trade["b"])
    if a is None or b is None:
        return None
    if not (set(trade["a_got"]) <= set(a["r"]) and set(trade["b_got"]) <= set(b["r"])):
        return None
    # Undo it: a gives back what it got, and gets back what b got.
    r = trade_eval.evaluate(data, b["id"], list(trade["a_got"]), list(trade["b_got"]), side=a["id"])
    me_i = next((i for i, t in enumerate(data["teams"]) if t["me"]), None)
    ti = {t["id"]: i for i, t in enumerate(data["teams"])}
    out = {"a_title": (r["title_me"][1], r["title_me"][0]), "b_title": (r["title_them"][1], r["title_them"][0]),
           "a_playoffs": (r["playoffs_me"][1], r["playoffs_me"][0]),
           "b_playoffs": (r["playoffs_them"][1], r["playoffs_them"][0]),
           "a_points": -r["shift_me"], "b_points": -r["shift_them"]}
    if me_i is not None and me_i not in (ti[a["id"]], ti[b["id"]]):
        base = data["base"]["title"][me_i]
        undone = base + (trade_eval._read(data, "title", ti[a["id"]], r["shift_me"], me_i) - base) \
            + (trade_eval._read(data, "title", ti[b["id"]], r["shift_them"], me_i) - base)
        out["you_title"] = (min(max(undone, 0.0), 1.0), base)
    return out
