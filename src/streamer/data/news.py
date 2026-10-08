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

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

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
            items = data.get("items") or []
            lines.append(f"  {data.get('count')} items; fields {_keys(items[0]) if items else '-'}")
        except ValueError:
            lines.append("  not JSON")
    try:
        r = _get(FANTASYPROS_NEWS, headers={"x-api-key": key}, params={"limit": 100})
        n = len((r.json() or {}).get("items") or []) if r.status_code == 200 else 0
        lines.append(f"FantasyPros news, limit 100: HTTP {r.status_code}, {n} items")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"FantasyPros news, limit 100: {type(exc).__name__}")


#: Ways to ask ESPN for many players' news at once (the plain latest-news
#: call answers 500).
BULK_TRIES = (
    (ESPN_PLAYER_NEWS, {"limit": 50, "days": 1}),
    (ESPN_PLAYER_NEWS, {"limit": 50, "offset": 0}),
    (ESPN_PLAYER_NEWS, {"playerId": "3116406,4426502,4362628", "limit": 30}),
    ("https://site.web.api.espn.com/apis/fantasy/v2/games/ffl/news/players", {"limit": 50}),
    ("https://site.api.espn.com/apis/fantasy/v2/games/ffl/news", {"limit": 50}),
    ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/news", {"limit": 100}),
)


def _bulk(lines: list[str]) -> None:
    for url, params in BULK_TRIES:
        label = f"{url.split('//')[1][:60]} {params}"
        try:
            r = _get(url, params=params)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"{label}: {type(exc).__name__}")
            continue
        lines.append(f"{label}: HTTP {r.status_code}, {len(r.content) / 1e3:.0f} kB")
        if r.status_code != 200:
            continue
        try:
            data = r.json()
        except ValueError:
            continue
        items = next((data[k] for k in ("feed", "articles", "items") if isinstance(data.get(k), list)), [])
        kinds: dict[str, int] = {}
        for it in items:
            kinds[str(it.get("type"))] = kinds.get(str(it.get("type")), 0) + 1
        when = sorted(str(it.get("published") or "") for it in items)
        lines.append(f"  {len(items)} items, types {kinds}, from {when[0] if when else '-'} to {when[-1] if when else '-'}")
        ids = {str(a.get("id")) for it in items for a in (it.get("athletes") or it.get("related") or [])
               if isinstance(a, dict)}
        lines.append(f"  player ids tagged: {len(ids)}; playerId field on {sum(1 for it in items if it.get('playerId'))}")
        for it in items[:4]:
            lines.append(f"    {it.get('published')} | {it.get('type')} | {_short(it.get('headline'), 110)}")


def probe(cfg: Config) -> list[str]:
    """What each free news source returns, for a public log."""
    lines: list[str] = []
    _bulk(lines)
    _espn(lines)
    _sleeper(cfg, lines)
    _rss("Pro Football Rumors RSS", PFR_FEED, lines)
    _rss("RotoWire RSS", ROTOWIRE_FEED, lines)
    _fantasypros(lines)
    return lines


# ---------------------------------------------------------------------------
# Reading it
# ---------------------------------------------------------------------------
#: Hours a player's news is reused before it is asked for again.
CACHE_HOURS = 6.0
#: What the news says, most advanced first. Only a headline that names a
#: team counts as a signing ("Hill signed with Miami in 2022" in a story
#: about something else does not).
SIGNED = re.compile(r"\b(signs|signed|signing with|agrees? to (?:terms|a deal|a contract)|"
                    r"agreed to (?:terms|a deal|a contract)|inks?)\b", re.I)
CLOSE = re.compile(r"\b(closing in|nearing (?:a )?deal|close to (?:a )?deal|expected to sign|set to sign|"
                   r"finaliz\w*|deal (?:is )?(?:close|imminent))\b", re.I)
TALKS = re.compile(r"\b(negotiat\w*|reached out|visit\w*|work(?:ed|s)? out|workout|interest(?:ed)?|"
                   r"meet\w* with|talks)\b", re.I)
LEGAL = re.compile(r"\b(trial|arrest\w*|suspen\w+|charged|indict\w*|lawsuit|investigat\w+)\b", re.I)
RANK = {"talks": 1, "close": 2, "signed": 3}
#: Item types that are news about the player himself: ESPN carries
#: RotoWire's blurbs and its own headlines; columns and videos that mention
#: him in passing are not read.
PLAYER_TYPES = ("Rotowire", "HeadlineNews")


@dataclass
class Signing:
    """What the latest news says about a free agent signing."""

    stage: str                    # "signed", "close" or "talks"
    team: str | None              # nflverse abbreviation, when one team is named
    published: datetime
    headline: str
    legal: bool = False           # a legal case is also in the news


def _when(item: dict) -> datetime | None:
    raw = item.get("published") or item.get("lastModified")
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def teams_named(text: str) -> list[str]:
    """Teams named in ``text`` by nickname ("Chiefs"), in order of first mention."""
    from ..teams import TEAM_NAMES

    found = []
    for abbr, name in TEAM_NAMES.items():
        m = re.search(rf"\b{re.escape(name.split()[-1])}\b", text)
        if m:
            found.append((m.start(), abbr))
    return [abbr for _pos, abbr in sorted(found)]


def read(items: list[dict], name: str, now: datetime, lookback_days: float = 14.0) -> Signing | None:
    """The most advanced signing news about ``name`` in the last
    ``lookback_days``: a signing with a team in a headline, then "closing in"
    or "expected to sign", then talks -- visits, workouts, a team reaching
    out, negotiations. None when nothing recent says he is near a team."""
    last = name.split()[-1]
    best: Signing | None = None
    legal = False
    for it in items:
        when = _when(it)
        if when is None or now - when > timedelta(days=lookback_days):
            continue
        head = str(it.get("headline") or it.get("title") or "")
        text = f"{head} {it.get('story') or it.get('description') or ''}"
        if last not in text:
            continue
        legal = legal or bool(LEGAL.search(text))
        if it.get("type") not in PLAYER_TYPES:
            continue
        if SIGNED.search(head) and teams_named(head):
            stage, teams = "signed", teams_named(head)
        elif CLOSE.search(text):
            stage, teams = "close", teams_named(text)
        elif TALKS.search(text):
            stage, teams = "talks", teams_named(text)
        else:
            continue
        if best is None or RANK[stage] > RANK[best.stage] or (RANK[stage] == RANK[best.stage]
                                                              and when > best.published):
            team = teams[0] if teams and (stage == "signed" or len(teams) == 1) else None
            best = Signing(stage, team, when, _short(head, 150))
    if best is not None:
        best.legal = legal
    return best


def _cached_get(cfg: Config, name: str, url: str, params: dict, allow_network: bool) -> dict:
    path = cfg.raw_dir / "news" / f"{name}.json"
    fresh = path.exists() and time.time() - path.stat().st_mtime < CACHE_HOURS * 3600
    if fresh or (path.exists() and not allow_network):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    if not allow_network:
        return {}
    try:
        r = _get(url, params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001 - news is a bonus: fall back to the last pull
        log.warning("news %s failed: %s", name, type(exc).__name__)
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except ValueError:
            return {}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def player_items(cfg: Config, espn_id: str, allow_network: bool = True) -> list[dict]:
    """ESPN's news about one player (its id), newest first."""
    data = _cached_get(cfg, f"espn_{espn_id}", ESPN_PLAYER_NEWS, {"playerId": espn_id, "limit": 15}, allow_network)
    return list(data.get("feed") or [])


def league_items(cfg: Config, allow_network: bool = True) -> list[dict]:
    """ESPN's latest NFL headlines -- where a legal case shows up."""
    data = _cached_get(cfg, "espn_nfl", ESPN_NFL_NEWS, {"limit": 50}, allow_network)
    return [dict(it, type="HeadlineNews") for it in data.get("articles") or []]


def trending_adds(cfg: Config, allow_network: bool = True) -> list[str]:
    """Names Sleeper's managers added most in the last day."""
    from . import sleeper

    data = _cached_get(cfg, "sleeper_trending", SLEEPER_TRENDING.format(kind="add"),
                       {"lookback_hours": 24, "limit": 50}, allow_network)
    db = sleeper.players(cfg) if allow_network else {}
    rows = data if isinstance(data, list) else []
    return [str((db.get(str(r.get("player_id"))) or {}).get("full_name") or "") for r in rows]


#: The last week the rest of the season is counted to.
LAST_WEEK = 17


def expected_share(odds: float, lag: int, weeks_left: int) -> float:
    """Share of the ``weeks_left`` he is expected to play: his chance of
    playing at all, times the weeks left after signing and the measured wait
    for a first game (:data:`roster.futures.DEBUT_WAIT`)."""
    from ..roster.futures import DEBUT_WAIT

    weights = [float(c) for c in DEBUT_WAIT]
    total = sum(weights) or 1.0
    left = sum(c * max(weeks_left - lag - w, 0) for w, c in enumerate(weights)) / total
    return odds * left / max(weeks_left, 1)


def attach(snapshot, cfg: Config, espn_ids: dict[str, str], allow_network: bool = True,
           now: datetime | None = None) -> list[str]:
    """Mark the players on no NFL roster whom the news has close to a team.

    Watched: those on a fantasy roster in this league, 2%+ owned, or among
    Sleeper's most-added of the last day. For each, ESPN's news about him
    (RotoWire's blurbs and ESPN's headlines) from the last ``lookback_days``
    is read for its most advanced stage -- signed, close, talks -- and
    Sleeper's database can say he has signed before the platforms do. The
    chance he signs by stage, and the weeks until he does, are judgment
    (``roster.free_agent_news``); what happens after signing is measured
    (:data:`roster.futures.PLAYS_AFTER_SIGNING`, ``DEBUT_WAIT``). Returns a
    line per player marked."""
    from ..roster.futures import PLAYS_AFTER_SIGNING
    from ..roster.players import normalize_name
    from ..teams import normalize_team

    conf = (cfg.raw.get("roster") or {}).get("free_agent_news") or {}
    for p in snapshot.all_players():
        p.signing_team = p.signing_odds = p.signing_share = None
        p.signing_wait, p.news = 0, ""
    if not conf:
        return []
    now = now or datetime.now(UTC)
    lookback = float(conf.get("lookback_days", 14))
    odds_by, lag_by = conf.get("stage_odds") or {}, conf.get("stage_lag") or {}
    rostered = {p.player_id for t in snapshot.teams for p in t.roster}
    watch = [p for p in snapshot.all_players() if p.position in ("QB", "RB", "WR", "TE") and p.unsigned]
    if not watch:
        return []
    trending = {normalize_name(n) for n in trending_adds(cfg, allow_network) if n}
    signed_on = {}
    if allow_network:
        from . import sleeper

        for q in (sleeper.players(cfg) or {}).values():
            if isinstance(q, dict) and q.get("team") and q.get("status") == "Active" and q.get("full_name"):
                signed_on[(normalize_name(q["full_name"]), q.get("position"))] = normalize_team(q["team"])
    headlines = league_items(cfg, allow_network)
    weeks_left = max(LAST_WEEK - int(snapshot.week) + 1, 1)
    notes = []
    for p in watch:
        key = normalize_name(p.name)
        if not (p.player_id in rostered or (p.percent_owned or 0) >= 2 or key in trending):
            continue
        espn_id = espn_ids.get(key)
        items = (player_items(cfg, espn_id, allow_network) if espn_id else []) + \
            [it for it in headlines if p.name in str(it.get("headline") or "")]
        sig = read(items, p.name, now, lookback)
        team = signed_on.get((key, p.position))
        if team and (sig is None or sig.stage != "signed"):
            sig = Signing("signed", team, now, f"Sleeper lists him with {team}", legal=bool(sig and sig.legal))
        if sig is None or sig.stage not in odds_by:
            continue
        odds = float(odds_by[sig.stage]) * PLAYS_AFTER_SIGNING
        lag = int(lag_by.get(sig.stage, 0))
        p.signing_team, p.signing_odds, p.signing_wait = sig.team, round(odds, 3), lag
        p.signing_share = round(expected_share(odds, lag, weeks_left), 3)
        p.news = (f"news {sig.published:%b %d}: {sig.headline}"
                  + (" -- a legal case is also in the news" if sig.legal else ""))
        notes.append(f"{p.name}: {sig.stage}{' with ' + sig.team if sig.team else ''}, "
                     f"{odds:.0%} to play this season, {p.signing_share:.0%} of the rest of it")
    return notes
