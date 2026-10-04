"""SportsGameOdds: a second, free source of sportsbook player props.

The Odds API bills each prop market of each game (7 credits a game here, 500
a month free). SportsGameOdds bills each game once, whatever its markets and
books, with 2,500 a month on the free plan -- so it can price the whole slate
on any day, and its books join the same consensus (``props.consensus``).

Its events carry odds keyed ``{statID}-{statEntityID}-{periodID}-{betTypeID}-{sideID}``
(the entity is a player ID for player props) with prices per bookmaker under
``byBookmaker``. Only full-game over/under and yes/no markets for the stats in
:data:`STATS` are read; the rest are counted by :func:`probe` so new names
can be added. Which stat IDs carry touchdowns and interceptions is not in
the reference we were given, so several spellings are accepted and the probe
prints the ones the feed actually uses.

The key comes from the ``SPORTSGAMEODDS_API_KEY`` environment variable (a
GitHub Actions secret); it is sent as a header, never logged.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)

BASE = "https://api.sportsgameodds.com/v2"

#: statID -> (our stat, how it is modelled), as in ``props.MARKETS``.
STATS: dict[str, tuple[str, str]] = {
    "passing_yards": ("passing_yards", "normal"),
    "rushing_yards": ("rushing_yards", "normal"),
    "receiving_yards": ("receiving_yards", "normal"),
    "receptions": ("receptions", "normal"),
    "passing_touchdowns": ("passing_tds", "poisson"),
    "passing_tds": ("passing_tds", "poisson"),
    "passing_interceptions": ("passing_interceptions", "poisson"),
    "interceptions_thrown": ("passing_interceptions", "poisson"),
    # Rushing and receiving touchdowns: an over 0.5 or a yes is "anytime".
    "touchdowns": ("anytime_td", "poisson"),
    "rushing+receiving_touchdowns": ("anytime_td", "poisson"),
    "anytime_touchdown": ("anytime_td", "poisson"),
}

COLUMNS = ["event_id", "bookmaker", "market", "stat", "kind", "player", "line",
           "over_price", "under_price", "yes_price"]


def api_key() -> str | None:
    return os.environ.get("SPORTSGAMEODDS_API_KEY") or None


def conf(cfg: Config) -> dict:
    return (cfg.odds.get("props") or {}).get("sportsgameodds") or {}


def _price(v) -> float | None:
    try:
        return float(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return None


def _player_name(event: dict, pid: str) -> str:
    p = (event.get("players") or {}).get(pid) or {}
    name = p.get("name") or " ".join(x for x in (p.get("firstName"), p.get("lastName")) if x)
    if name:
        return str(name)
    # IDs read like PATRICK_MAHOMES_1_NFL: the name is all but the last two parts.
    parts = pid.split("_")
    return " ".join(w.capitalize() for w in parts[:-2]) if len(parts) > 2 else pid


def parse(events: list[dict]) -> pd.DataFrame:
    """SportsGameOdds events -> the props frame (one row per player, stat and
    book), in the same columns as :func:`props.parse_props_payload`."""
    rows = []
    for event in events or []:
        eid = str(event.get("eventID") or "")
        players = event.get("players") or {}
        sides: dict[tuple, dict] = {}
        for odd_id, odd in (event.get("odds") or {}).items():
            parts = str(odd_id).split("-")
            if len(parts) != 5:
                continue
            stat_id, entity, period, bet, side = parts
            if period != "game" or stat_id not in STATS or entity not in players:
                continue
            if bet not in ("ou", "yn"):
                continue
            for book, b in (odd.get("byBookmaker") or {}).items():
                if not isinstance(b, dict) or b.get("available") is False:
                    continue
                slot = sides.setdefault((stat_id, entity, bet, book), {})
                slot[side] = _price(b.get("odds"))
                if b.get("overUnder") is not None:
                    slot["line"] = _price(b.get("overUnder"))
        for (stat_id, entity, bet, book), slot in sides.items():
            stat, kind = STATS[stat_id]
            if bet == "yn":
                line, over, under, yes = 0.5, slot.get("yes"), slot.get("no"), None
                if under is None:
                    kind, yes, over = "anytime", over, None
            else:
                line, over, under, yes = slot.get("line"), slot.get("over"), slot.get("under"), None
                if line is None or over is None or under is None:
                    continue
            rows.append({"event_id": eid, "bookmaker": book, "market": f"sgo:{stat_id}", "stat": stat,
                         "kind": kind, "player": _player_name(event, entity), "line": line,
                         "over_price": over, "under_price": under, "yes_price": yes})
    return pd.DataFrame(rows, columns=COLUMNS)


def _cache(cfg: Config):
    path = cfg.raw_dir / "sportsgameodds"
    path.mkdir(parents=True, exist_ok=True)
    return path / "events.json"


def events(cfg: Config, timeout: float = 20.0) -> tuple[list[dict], str]:
    """This week's NFL events with odds, from the cache when it is fresh
    enough (``cache_minutes``, default 360: the free plan counts each game
    returned, 2,500 a month). Returns (events, where they came from)."""
    import requests

    path = _cache(cfg)
    max_age = float(conf(cfg).get("cache_minutes", 360))
    if path.exists():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            age = (datetime.now(UTC) - datetime.fromisoformat(blob["fetched_at"])).total_seconds() / 60
            if age <= max_age:
                return blob["events"], f"cached {age:.0f} min ago"
        except (OSError, ValueError, KeyError):
            pass
    key = api_key()
    if not key:
        return [], "no SPORTSGAMEODDS_API_KEY"
    out, cursor = [], None
    for _ in range(int(conf(cfg).get("max_pages", 4))):
        params = {"leagueID": "NFL", "oddsAvailable": "true", "limit": int(conf(cfg).get("page_size", 50))}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE}/events", params=params, headers={"x-api-key": key}, timeout=timeout)
        resp.raise_for_status()
        body = resp.json() or {}
        out.extend(body.get("data") or [])
        cursor = body.get("nextCursor")
        if not cursor:
            break
    out = [e for e in out if not (e.get("status") or {}).get("started")]
    try:
        path.write_text(json.dumps({"fetched_at": datetime.now(UTC).isoformat(), "events": out}), encoding="utf-8")
    except OSError:
        pass
    return out, f"{len(out)} games fetched"


#: Books whose names differ between the two feeds.
ALIASES = {"williamhill_us": "caesars"}


def merge(primary: pd.DataFrame, extra: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The Odds API's rows plus SportsGameOdds' where they add a book: a price
    the first feed already has for that player, stat and book counts once.
    Pick'em sites are not sportsbook prices and are left out."""
    from ..roster.players import normalize_name

    skip = set(conf(cfg).get("exclude_books", ["prizepicks", "underdog", "sleeper", "dabble"]))
    extra = extra[~extra["bookmaker"].isin(skip)]
    if extra.empty:
        return primary
    if primary is None or primary.empty:
        return extra.reset_index(drop=True)

    def key(f):
        return list(zip(f["player"].map(normalize_name), f["stat"], f["bookmaker"].map(lambda b: ALIASES.get(b, b))))

    have = set(key(primary))
    new = extra[[k not in have for k in key(extra)]]
    return pd.concat([primary, new], ignore_index=True)


def props_frame(cfg: Config) -> tuple[pd.DataFrame, str]:
    """The props frame from SportsGameOdds, and a note on where it came from.
    Empty (with the reason) when there is no key or the call fails."""
    if not conf(cfg).get("enabled", True):
        return pd.DataFrame(columns=COLUMNS), "disabled in config"
    try:
        evs, note = events(cfg)
    except Exception as exc:  # noqa: BLE001 - a second source never blocks the first
        return pd.DataFrame(columns=COLUMNS), f"unavailable: {type(exc).__name__}"
    frame = parse(evs)
    return frame, f"{note}; {frame['player'].nunique() if not frame.empty else 0} players priced"


def probe(cfg: Config) -> list[str]:
    """What the feed returns, in terms safe for a public log: status, counts,
    bookmaker IDs, stat IDs and bet types for full-game player markets, and
    field names -- never prices, lines, player names or the key."""
    import requests

    key = api_key()
    if not key:
        return ["SPORTSGAMEODDS_API_KEY is not set"]
    try:
        resp = requests.get(f"{BASE}/events", params={"leagueID": "NFL", "oddsAvailable": "true", "limit": 3},
                            headers={"x-api-key": key}, timeout=20)
    except Exception as exc:  # noqa: BLE001
        return [f"/events: {type(exc).__name__}"]
    out = [f"/events: HTTP {resp.status_code}"]
    try:
        body = resp.json()
    except ValueError:
        return out + ["not JSON"]
    if not isinstance(body, dict):
        return out + [f"top level is {type(body).__name__}"]
    out.append(f"top-level keys {sorted(body)[:12]}; nextCursor {'present' if body.get('nextCursor') else 'absent'}")
    data = body.get("data") or []
    out.append(f"{len(data)} events")
    if not data:
        return out
    e = data[0]
    out.append(f"event fields {sorted(e)[:25]}")
    players = e.get("players") or {}
    if players:
        out.append(f"player fields {sorted(next(iter(players.values())))[:20]}")
    stats, books, bets, periods = {}, {}, {}, {}
    for ev in data:
        pl = ev.get("players") or {}
        for odd_id, odd in (ev.get("odds") or {}).items():
            parts = str(odd_id).split("-")
            if len(parts) != 5 or parts[1] not in pl:
                continue
            stats[parts[0]] = stats.get(parts[0], 0) + 1
            bets[parts[3]] = bets.get(parts[3], 0) + 1
            periods[parts[2]] = periods.get(parts[2], 0) + 1
            for b in (odd.get("byBookmaker") or {}):
                books[b] = books.get(b, 0) + 1
    out.append(f"player-market stat IDs {sorted(stats.items(), key=lambda kv: -kv[1])[:30]}")
    out.append(f"bet types {bets}; periods {periods}")
    out.append(f"bookmakers on player markets {sorted(books.items(), key=lambda kv: -kv[1])}")
    sample = next(iter((e.get("odds") or {}).values()), {})
    if sample:
        out.append(f"odd fields {sorted(sample)[:20]}")
        bb = next(iter((sample.get("byBookmaker") or {}).values()), {})
        out.append(f"byBookmaker fields {sorted(bb)[:15]}")
    frame = parse(data)
    out.append(f"parsed: {len(frame)} rows, {frame['player'].nunique() if not frame.empty else 0} players, "
               f"stats {sorted(frame['stat'].unique()) if not frame.empty else []}")
    for h in ("x-ratelimit-remaining", "x-ratelimit-limit"):
        if h in resp.headers:
            out.append(f"{h}: {resp.headers[h]}")
    return out
