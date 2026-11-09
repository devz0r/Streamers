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
from datetime import UTC, datetime, timedelta

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
    "receiving_receptions": ("receptions", "normal"),     # the feed's name (probe, week 4)
    "passing_touchdowns": ("passing_tds", "poisson"),
    "passing_tds": ("passing_tds", "poisson"),
    "passing_interceptions": ("passing_interceptions", "poisson"),
    "interceptions_thrown": ("passing_interceptions", "poisson"),
    # Rushing and receiving touchdowns: an over 0.5 or a yes is "anytime".
    "touchdowns": ("anytime_td", "poisson"),
    "rushing+receiving_touchdowns": ("anytime_td", "poisson"),
    "anytime_touchdown": ("anytime_td", "poisson"),
    # A kicker's points (field goals x 3 + extra points), if the feed lists it so.
    "kicking_points": ("kicking_points", "normal"),
}

COLUMNS = ["event_id", "bookmaker", "market", "stat", "kind", "player", "line",
           "over_price", "under_price", "yes_price"]

#: When each event last returned by :func:`events` was read from the feed, by
#: event id (the cache's own time for a game it still holds): the moment its
#: prices were last known to be on offer. ``LAST_FETCHED``: the pull's own time.
EVENT_FETCHED: dict[str, datetime] = {}
LAST_FETCHED: datetime | None = None


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
        # A book that prices a stat both ways (touchdowns: yes/no and over
        # 0.5) is one opinion: keep the over/under.
        have_ou = {(STATS[k[0]][0], k[1], k[3]) for k in sides if k[2] == "ou"}
        for (stat_id, entity, bet, book), slot in sides.items():
            if bet == "yn" and (STATS[stat_id][0], entity, book) in have_ou:
                continue
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


def _usage_path(cfg: Config):
    return _cache(cfg).with_name("usage.json")


def used_this_month(cfg: Config, now: datetime | None = None) -> int:
    """Games returned to us this calendar month (the plan counts each one)."""
    now = now or datetime.now(UTC)
    try:
        return int(json.loads(_usage_path(cfg).read_text(encoding="utf-8")).get(now.strftime("%Y-%m"), 0))
    except (OSError, ValueError):
        return 0


def _count(cfg: Config, n: int, now: datetime) -> None:
    path = _usage_path(cfg)
    try:
        blob = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        blob = {}
    month = now.strftime("%Y-%m")
    blob = {month: int(blob.get(month, 0)) + n}
    try:
        path.write_text(json.dumps(blob), encoding="utf-8")
    except OSError:
        pass


def _load(path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fast_due(blob: dict | None, events_: list[dict], now: datetime, c: dict) -> bool:
    """Should the games about to start be re-read now? Games kicking off within
    ``fast_hours`` (default 6) are re-read once their newest read is
    ``fast_minutes`` (default 50) old, so a refresh an hour apart on a game day
    sees prices an hour old, not six; the rest of the slate waits for the
    ``cache_minutes`` pull. Off with ``fast_hours: 0``."""
    hours = float(c.get("fast_hours", 6))
    if hours <= 0 or not blob:
        return False
    soon = []
    for e in events_:
        try:
            when = datetime.fromisoformat(str((e.get("status") or {}).get("startsAt")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if now < when <= now + timedelta(hours=hours):
            soon.append(str(e.get("eventID") or ""))
    if not soon:
        return False
    times = blob.get("event_times") or {}
    oldest = min((datetime.fromisoformat(times[eid]) if eid in times else datetime.fromisoformat(blob["fetched_at"]))
                 for eid in soon)
    return (now - oldest).total_seconds() / 60.0 >= float(c.get("fast_minutes", 50))


def _fetch(cfg: Config, key: str, now: datetime, until: datetime, c: dict, timeout: float, pages: int) -> list[dict]:
    import requests

    out, cursor = [], None
    for _ in range(pages):
        params = {"leagueID": "NFL", "oddsAvailable": "true", "limit": int(c.get("page_size", 30)),
                  "startsAfter": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "startsBefore": until.strftime("%Y-%m-%dT%H:%M:%SZ")}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE}/events", params=params, headers={"x-api-key": key}, timeout=timeout)
        resp.raise_for_status()
        body = resp.json() or {}
        page = body.get("data") or []
        out.extend(page)
        _count(cfg, len(page), now)
        cursor = body.get("nextCursor")
        if not cursor:
            break
    return out


def events(cfg: Config, timeout: float = 20.0) -> tuple[list[dict], str]:
    """The NFL games starting in the next ``window_hours`` (default 96) with
    odds, from the cache when it is fresh enough (``cache_minutes``, default
    360). The free plan counts every game returned, 2,500 a month: without
    the window the feed returns every game already priced weeks ahead (173
    on the first call), so the window is asked for and checked, and a
    month's tally stops fetching at ``monthly_budget``. Games about to start
    are re-read sooner, on their own (:func:`fast_due`: the games within
    ``fast_hours`` once their read is ``fast_minutes`` old), which costs a
    few games a pull and stops at ``fast_share`` (default 0.85) of the
    month's budget. Returns (events, where they came from); :data:`EVENT_FETCHED`
    says when each game's prices were read."""
    global LAST_FETCHED

    c = conf(cfg)
    path = _cache(cfg)
    max_age = float(c.get("cache_minutes", 360))
    now = datetime.now(UTC)
    until = now + timedelta(hours=float(c.get("window_hours", 96)))
    blob = _load(path)
    held, note = None, ""
    if blob:
        try:
            age = (now - datetime.fromisoformat(blob["fetched_at"])).total_seconds() / 60
            if age <= max_age:
                held = _upcoming(blob["events"], now, until)
                note = f"cached {age:.0f} min ago"
        except (KeyError, ValueError):
            blob = None
    key = api_key()
    budget = int(c.get("monthly_budget", 2300))
    used = used_this_month(cfg, now)
    if held is not None:
        times = (blob or {}).get("event_times") or {}
        if key and fast_due(blob, held, now, c) and used < budget * float(c.get("fast_share", 0.85)):
            soon_until = now + timedelta(hours=float(c.get("fast_hours", 6)))
            try:
                fresh = _fetch(cfg, key, now, soon_until, c, timeout, 1)
            except Exception as exc:  # noqa: BLE001 - the cached pull still stands
                log.warning("SportsGameOdds fast re-read failed: %s", type(exc).__name__)
                fresh = []
            fresh = _upcoming(fresh, now, soon_until)
            if fresh:
                ids = {str(e.get("eventID")) for e in fresh}
                held = [e for e in held if str(e.get("eventID")) not in ids] + fresh
                times = {**times, **{i: now.isoformat() for i in ids}}
                try:
                    path.write_text(json.dumps({"fetched_at": blob["fetched_at"], "events": held, "event_times": times}),
                                    encoding="utf-8")
                except OSError:
                    pass
                note += f"; {len(fresh)} games about to start re-read"
        stamp = datetime.fromisoformat(blob["fetched_at"])
        EVENT_FETCHED.clear()
        EVENT_FETCHED.update({str(e.get("eventID")): datetime.fromisoformat(times[str(e.get("eventID"))])
                              if str(e.get("eventID")) in times else stamp for e in held})
        LAST_FETCHED = max(EVENT_FETCHED.values(), default=stamp)
        return held, note
    if not key:
        return [], "no SPORTSGAMEODDS_API_KEY"
    if used >= budget:
        return [], f"monthly budget reached ({used} of {budget} games)"
    out = _fetch(cfg, key, now, until, c, timeout, int(c.get("max_pages", 2)))
    returned = len(out)
    out = _upcoming(out, now, until)
    LAST_FETCHED = now
    EVENT_FETCHED.clear()
    EVENT_FETCHED.update({str(e.get("eventID")): now for e in out})
    try:
        path.write_text(json.dumps({"fetched_at": now.isoformat(), "events": out,
                                   "event_times": {i: now.isoformat() for i in EVENT_FETCHED}}), encoding="utf-8")
    except OSError:
        pass
    return out, f"{len(out)} games fetched ({returned} counted; {used + returned} this month)"


def _upcoming(evs: list[dict], now: datetime, until: datetime | None = None) -> list[dict]:
    """Games not yet started, and (when given) starting before ``until``."""
    out = []
    for e in evs:
        st = e.get("status") or {}
        if st.get("started") or st.get("ended"):
            continue
        if until is not None and st.get("startsAt"):
            try:
                when = datetime.fromisoformat(str(st["startsAt"]).replace("Z", "+00:00"))
                if when > until:
                    continue
            except ValueError:
                pass
        out.append(e)
    return out


#: Books whose names differ between the two feeds.
ALIASES = {"williamhill_us": "caesars"}


def merge(primary: pd.DataFrame, extra: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The Odds API's rows plus SportsGameOdds': a price both feeds have for
    a player, stat and book counts once -- the one read more recently
    (``quoted_at``; the first feed's when neither says). Pick'em sites are
    not sportsbook prices and are left out."""
    from ..roster.players import normalize_name

    skip = set(conf(cfg).get("exclude_books", ["prizepicks", "underdog", "sleeper", "dabble"]))
    extra = extra[~extra["bookmaker"].isin(skip)]
    if extra.empty:
        return primary
    if primary is None or primary.empty:
        return extra.reset_index(drop=True)

    def key(f):
        return list(zip(f["player"].map(normalize_name), f["stat"], f["bookmaker"].map(lambda b: ALIASES.get(b, b))))

    def when(f):
        col = f["quoted_at"] if "quoted_at" in f else pd.Series([None] * len(f), index=f.index)
        return pd.to_datetime(col, utc=True, errors="coerce")

    have = dict(zip(key(primary), when(primary)))
    fresher = [k not in have or (pd.notna(t) and (pd.isna(have[k]) or t > have[k]))
               for k, t in zip(key(extra), when(extra))]
    new = extra[fresher]
    taken = set(key(new))
    kept = primary[[k not in taken for k in key(primary)]]
    return pd.concat([kept, new], ignore_index=True)


def quote_stamps(event_ids: pd.Series) -> pd.Series:
    """When each row's game was read from the feed (ISO text), falling back to the pull's own time."""
    fallback = LAST_FETCHED.isoformat() if LAST_FETCHED is not None else None
    return event_ids.map(lambda e: EVENT_FETCHED[str(e)].isoformat() if str(e) in EVENT_FETCHED else fallback)


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
    frame["quoted_at"] = quote_stamps(frame["event_id"])
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
