"""Player news: who is about to sign, who just did.

A free agent like Tyreek Hill has no team, so every projection source
prices him at zero until the platforms move him -- after he has signed,
often after he has been claimed. The news says it first: a visit, then
"closing in on a deal", then the signing. Free sources that carry it:

* ESPN's fantasy news feed (``site.api.espn.com``), the blurbs the ESPN app
  shows -- per player, no key;
* Sleeper's player database (``team`` moves within hours of a signing) and
  its trending adds (what thousands of managers did in the last day) -- no
  key;
* Pro Football Rumors' RSS feed, which tracks visits, workouts and deals.

The sandbox this was written in cannot reach any of them, so the first step
is a probe the workflow can run (``streamer news-probe``): status, size,
field names and the items about a few players -- public news, safe for a
public log. FantasyPros is checked for status and field names only: its
data is used for analysis, never shown.
"""

from __future__ import annotations

import logging
import re

from ..config import Config

log = logging.getLogger(__name__)

ESPN_PLAYER_NEWS = "https://site.api.espn.com/apis/fantasy/v2/games/ffl/news/players"
ESPN_NFL_NEWS = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news"
SLEEPER_TRENDING = "https://api.sleeper.app/v1/players/nfl/trending/{kind}"
PFR_FEED = "https://www.profootballrumors.com/feed"
ROTOWIRE_FEED = "https://www.rotowire.com/rss/news.php?sport=NFL"
FANTASYPROS_NEWS = "https://api.fantasypros.com/public/v2/json/nfl/news"

#: Players the probe looks for, with their ESPN ids.
PROBE_PLAYERS = {"Tyreek Hill": "3116406"}
HEADERS = {"User-Agent": "streamer-fantasy-tool/1.0 (personal use)"}


def _get(url: str, **kw):
    import requests

    return requests.get(url, timeout=30, headers={**HEADERS, **kw.pop("headers", {})}, **kw)


def _keys(obj, depth: int = 0) -> str:
    if isinstance(obj, dict):
        return "{" + ", ".join(sorted(obj)[:25]) + "}"
    if isinstance(obj, list):
        return f"[{len(obj)} x {_keys(obj[0]) if obj else '-'}]"
    return type(obj).__name__


def _short(text, n: int = 160) -> str:
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text or ""))).strip()
    return text if len(text) <= n else text[: n - 1] + "..."


def _espn(lines: list[str]) -> None:
    for label, url, params in (("ESPN fantasy news, latest", ESPN_PLAYER_NEWS, {"limit": 50}),
                               *((f"ESPN fantasy news, {name}", ESPN_PLAYER_NEWS, {"playerId": pid, "limit": 10})
                                 for name, pid in PROBE_PLAYERS.items()),
                               ("ESPN NFL news", ESPN_NFL_NEWS, {"limit": 50})):
        try:
            r = _get(url, params=params)
        except Exception as exc:  # noqa: BLE001 - a probe reports, never fails
            lines.append(f"{label}: {type(exc).__name__}")
            continue
        lines.append(f"{label}: HTTP {r.status_code}, {len(r.content) / 1e3:.0f} kB")
        if r.status_code != 200:
            continue
        try:
            data = r.json()
        except ValueError:
            lines.append("  not JSON")
            continue
        lines.append(f"  top level: {_keys(data)}")
        items = next((data[k] for k in ("feed", "articles", "items", "news") if isinstance(data.get(k), list)), [])
        if items:
            lines.append(f"  {len(items)} items; first: {_keys(items[0])}")
            for it in items[:3] if "latest" in label or "NFL" in label else items[:6]:
                when = it.get("published") or it.get("lastModified") or it.get("date") or ""
                lines.append(f"    {when} | {_short(it.get('headline') or it.get('title'))} | "
                             f"type={it.get('type')} | {_short(it.get('story') or it.get('description'), 120)}")


def _sleeper(cfg: Config, lines: list[str]) -> None:
    from . import sleeper

    db = sleeper.players(cfg)
    lines.append(f"Sleeper players: {len(db)} (cached daily)")
    for name in PROBE_PLAYERS:
        for p in (q for q in db.values() if isinstance(q, dict) and q.get("full_name") == name):
            lines.append(f"  {name}: team={p.get('team')} status={p.get('status')} active={p.get('active')} "
                         f"injury={p.get('injury_status')} news_updated={p.get('news_updated')} "
                         f"years_exp={p.get('years_exp')}")
    for kind in ("add", "drop"):
        try:
            r = _get(SLEEPER_TRENDING.format(kind=kind), params={"lookback_hours": 24, "limit": 15})
        except Exception as exc:  # noqa: BLE001
            lines.append(f"Sleeper trending {kind}: {type(exc).__name__}")
            continue
        lines.append(f"Sleeper trending {kind}: HTTP {r.status_code}")
        if r.status_code == 200:
            rows = r.json()
            lines.append(f"  {len(rows)} rows, fields {_keys(rows[0]) if rows else '-'}")
            lines.append("  " + "; ".join(
                f"{(db.get(str(x.get('player_id'))) or {}).get('full_name', x.get('player_id'))} {x.get('count')}"
                for x in rows[:15]))


def _rss(label: str, url: str, lines: list[str]) -> None:
    import xml.etree.ElementTree as ET

    try:
        r = _get(url)
    except Exception as exc:  # noqa: BLE001
        lines.append(f"{label}: {type(exc).__name__}")
        return
    lines.append(f"{label}: HTTP {r.status_code}, {len(r.content) / 1e3:.0f} kB")
    if r.status_code != 200:
        return
    try:
        items = ET.fromstring(r.content).findall(".//item")
    except ET.ParseError as exc:
        lines.append(f"  not RSS: {exc}")
        return
    lines.append(f"  {len(items)} items; fields {sorted({c.tag for it in items[:5] for c in it})}")
    for it in items[:3]:
        lines.append(f"    {it.findtext('pubDate')} | {_short(it.findtext('title'))}")
    for name in PROBE_PLAYERS:
        last = name.split()[-1]
        hits = [it for it in items if last in (it.findtext("title") or "") + (it.findtext("description") or "")]
        for it in hits[:4]:
            lines.append(f"    [{name}] {it.findtext('pubDate')} | {_short(it.findtext('title'))}")


def _fantasypros(lines: list[str]) -> None:
    from .fantasypros import api_key

    key = api_key()
    if not key:
        lines.append("FantasyPros news: FANTASYPROS_API_KEY is not set")
        return
    try:
        r = _get(FANTASYPROS_NEWS, headers={"x-api-key": key}, params={"limit": 5})
    except Exception as exc:  # noqa: BLE001
        lines.append(f"FantasyPros news: {type(exc).__name__}")
        return
    lines.append(f"FantasyPros news: HTTP {r.status_code}, {len(r.content) / 1e3:.0f} kB")
    if r.status_code == 200:
        try:
            data = r.json()
            lines.append(f"  top level: {_keys(data)}")      # field names only: never shown
        except ValueError:
            lines.append("  not JSON")


def probe(cfg: Config) -> list[str]:
    """What each free news source returns, for a public log."""
    lines: list[str] = []
    _espn(lines)
    _sleeper(cfg, lines)
    _rss("Pro Football Rumors RSS", PFR_FEED, lines)
    _rss("RotoWire RSS", ROTOWIRE_FEED, lines)
    _fantasypros(lines)
    return lines
