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


def _cached_get(cfg: Config, name: str, url: str, params: dict, allow_network: bool,
                stale_before: float | None = None) -> dict:
    path = cfg.raw_dir / "news" / f"{name}.json"
    fresh = (path.exists() and time.time() - path.stat().st_mtime < CACHE_HOURS * 3600
             and (stale_before is None or path.stat().st_mtime >= stale_before))
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


def player_items(cfg: Config, espn_id: str, allow_network: bool = True,
                 newer_than: datetime | None = None) -> list[dict]:
    """ESPN's news about one player (its id), newest first; asked again when
    something says there is news newer than the copy kept."""
    data = _cached_get(cfg, f"espn_{espn_id}", ESPN_PLAYER_NEWS, {"playerId": espn_id, "limit": 15}, allow_network,
                       stale_before=newer_than.timestamp() if newer_than else None)
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


# -- what the stage odds learn from -------------------------------------------
#: Every stage the news reached for a free agent (first sighting of each),
#: resolved later: signed (he turns up on a team) or not (still unsigned
#: ``resolve_days`` after). Shared by both leagues.
SIGNING_LOG = "signing_log.parquet"
LOG_COLUMNS = ["name", "position", "stage", "team", "seen", "signed", "signed_team", "resolved"]


def signing_log_path(cfg: Config):
    return cfg.results_dir.parent / SIGNING_LOG


def _read_log(path):
    import pandas as pd

    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - a log; start again rather than block
            log.warning("signing log unreadable: %s", exc)
    return pd.DataFrame(columns=LOG_COLUMNS)


def record_signings(path, sightings: list[tuple[str, str, str, str | None]], on_team: dict[tuple[str, str], str],
                    now: datetime, resolve_days: float = 28.0) -> None:
    """Add the first sighting of each (player, stage) and resolve open ones:
    ``on_team`` (normalized name, position) -> team for every player now on
    an NFL team. Unsigned ``resolve_days`` after the sighting counts as no."""
    import pandas as pd

    from ..roster.players import normalize_name

    d = _read_log(path)
    have = set(zip(d["name"], d["stage"]))
    new = [{"name": n, "position": pos, "stage": st, "team": tm or "", "seen": now.isoformat(),
            "signed": float("nan"), "signed_team": "", "resolved": ""}
           for n, pos, st, tm in sightings if (n, st) not in have]
    if new:
        d = pd.concat([d, pd.DataFrame(new)], ignore_index=True)
    if d.empty:
        return
    for i, r in d[d["resolved"].fillna("") == ""].iterrows():
        team = on_team.get((normalize_name(r["name"]), r["position"]))
        if team:
            d.loc[i, ["signed", "signed_team", "resolved"]] = [1.0, team, now.isoformat()]
        elif now - datetime.fromisoformat(r["seen"]) > timedelta(days=resolve_days):
            d.loc[i, ["signed", "resolved"]] = [0.0, now.isoformat()]
    path.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(path, index=False)


def fitted_stage_odds(path, prior: dict[str, float], k: float = 10.0, need: int = 5) -> dict[str, tuple[float, int]]:
    """Each stage's chance of signing: the resolved sightings' rate, shrunk
    toward the configured judgment by ``k`` cases once ``need`` are in."""
    d = _read_log(path)
    d = d[d["resolved"].fillna("") != ""] if not d.empty else d
    out = {}
    for stage, p0 in prior.items():
        got = d[d["stage"] == stage]["signed"].astype(float) if not d.empty else []
        n = len(got)
        out[stage] = ((float(got.sum()) + k * float(p0)) / (n + k), n) if n >= need else (float(p0), n)
    return out



def attach(snapshot, cfg: Config, espn_ids: dict[str, str], allow_network: bool = True,
           now: datetime | None = None) -> list[str]:
    """Mark the players on no NFL roster whom the news has close to a team.

    Watched: those on a fantasy roster in this league, 2%+ owned, or among
    Sleeper's most-added of the last day. For each, ESPN's news about him
    (RotoWire's blurbs and ESPN's headlines) from the last ``lookback_days``
    is read for its most advanced stage -- signed, close, talks -- and
    Sleeper's database can say he has signed before the platforms do. The
    chance he signs by stage starts as judgment (``roster.free_agent_news``)
    and is refitted from the logged sightings as they resolve
    (:func:`fitted_stage_odds`); the weeks until he does are judgment; what happens after signing is measured
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
    odds_by, lag_by = dict(conf.get("stage_odds") or {}), conf.get("stage_lag") or {}
    learn = conf.get("learn") or {}
    path = signing_log_path(cfg)
    if allow_network and learn:
        # The judgment by stage gives way to what the logged sightings did.
        for stage, (rate, n) in fitted_stage_odds(path, odds_by, float(learn.get("prior_cases", 10)),
                                                  int(learn.get("min_cases", 5))).items():
            if n >= int(learn.get("min_cases", 5)):
                odds_by[stage] = rate
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
    notes, sightings = [], []
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
        sightings.append((p.name, p.position, sig.stage, sig.team))
        odds = float(odds_by[sig.stage]) * PLAYS_AFTER_SIGNING
        lag = int(lag_by.get(sig.stage, 0))
        p.signing_team, p.signing_odds, p.signing_wait = sig.team, round(odds, 3), lag
        p.signing_share = round(expected_share(odds, lag, weeks_left), 3)
        p.news = (f"news {sig.published:%b %d}: {sig.headline}"
                  + (" -- a legal case is also in the news" if sig.legal else ""))
        notes.append(f"{p.name}: {sig.stage}{' with ' + sig.team if sig.team else ''}, "
                     f"{odds:.0%} to play this season, {p.signing_share:.0%} of the rest of it")
    if allow_network and learn:
        on_team = dict(signed_on)
        for q in snapshot.all_players():
            if q.team and not q.unsigned:
                on_team.setdefault((normalize_name(q.name), q.position), normalize_team(q.team))
        try:
            record_signings(path, sightings, on_team, now, float(learn.get("resolve_days", 28)))
        except Exception as exc:  # noqa: BLE001 - a log; never block the page
            log.warning("signing log not written: %s", exc)
    return notes


# ---------------------------------------------------------------------------
# How long the news says an injured player is out
# ---------------------------------------------------------------------------
_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "a couple": 2, "couple": 2}
_N = r"(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a couple of|a couple)"
_RANGE = re.compile(rf"\b{_N}\s*(?:-to-|-|\u2013|to|or)\s*{_N}\s*(?:more\s+)?(?:weeks?|games?)\b(?!\s*\d)", re.I)
_SINGLE = re.compile(rf"(?<![\d-])\b{_N}\s*(?:more\s+)?(?:weeks?|games?)\b(?!\s*\d)", re.I)
_MONTHS = re.compile(r"\b(?:about |around |roughly )?(a|one|two|three) months?\b", re.I)
_SEASON = re.compile(r"season-ending|out for (?:the )?(?:rest of the )?(?:season|year)|"
                     r"(?:miss|missing|sit out|done for) (?:the )?(?:rest|remainder) of (?:the )?(?:season|year)|"
                     r"done for the (?:season|year)", re.I)
_CONTEXT = re.compile(r"miss|out\b|sideline|timetable|recover|return|absence|expected|surgery|rehab|sit\b", re.I)
_PAST = re.compile(r"\b(last|past|previous|prior|first|opening|consecutive|straight|every|combined|"
                   r"total)\s*$|missed (?:the )?$|"
                   r"has (?:missed|been out)[^.]{0,20}$", re.I)
#: A floor, not a timeline: an injured-reserve stint "requires missing at
#: least four games", and the player is often out much longer.
_FLOOR = re.compile(r"\b(at least|minimum of|requisite|required to miss|requires? a player to miss|no fewer than|"
                    r"as few as)\b[^.]{0,30}$", re.I)
#: A guess or an absence of one: nothing is being reported.
_HEDGE = re.compile(r"\b(fear\w*|worr\w*|concern\w*|possib\w*|potential\w*|could|might|may|if|unclear|unknown|"
                    r"without a timeline|no timeline|hope\w*|optimis\w*|best[- ]case)\b", re.I)
_WEEK_LABEL = re.compile(r"\bweeks?\s*$|\bwk\.?\s*$", re.I)
#: A count of weeks is a timeline only right after words for time away
#: ("will miss", "out", "ruled out for the next", "sidelined for about") or
#: right before them ("a six-week recovery"): otherwise it is a stat line
#: ("8 of 12 targets over six games", "3.4 a carry through four games").
_CUE_BEFORE = re.compile(r"\b(miss|missing|out|sidelined|sideline|shelved|absent|timetable|timeline|"
                         r"recovery|return in|back in)\b(?:\W+[\w'-]+){0,4}?\W+$", re.I)
_CUE_AFTER = re.compile(r"\s*(?:absence|recovery|layoff|timetable|timeline|stint)\b", re.I)
#: "Name (injury)": who a clause is about. A sentence that names him can
#: still give someone else's timeline ("Moore is next up after Jayden Reed
#: (neck) will undergo season-ending surgery").
_TAGGED = re.compile(r"\b([A-Z][\w'.-]+)\s*\((?:[a-z]+[ /-]?){1,3}\)")


def _n(word: str) -> int:
    word = word.lower().strip()
    return int(word) if word.isdigit() else _NUMS.get(word.replace(" of", ""), 0)


def _someone_else(sentence: str, pos: int, last: str | None) -> bool:
    tagged = list(_TAGGED.finditer(sentence[:pos]))
    return bool(last and tagged and re.sub(r"'s?$", "", tagged[-1].group(1)) != last)


def timeline(text: str, last: str | None = None) -> tuple[int, int] | str | None:
    """How long a report says a player is out: ``(low, high)`` in weeks,
    ``"season"`` for a season-ending injury, None when no time is given.
    "Expected to miss 6-10 weeks" -> (6, 10); "out four to six weeks" -> (4, 6);
    "about a month" -> (4, 4). Not a timeline, so not read: past counts
    ("missed the last three weeks"), "Week 7" and "Week 7 game", a floor
    ("must miss at least four games" on injured reserve), a guess ("fear a
    season-ending injury", "hopeful to return"), a number with no words for
    time away beside it, and, given his ``last`` name, a time that follows
    another player's "Name (injury)"."""
    text = str(text or "")
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if _HEDGE.search(sentence):
            continue
        sm = _SEASON.search(sentence)
        if sm and not _FLOOR.search(sentence[:sm.start()]) and not _someone_else(sentence, sm.start(), last):
            return "season"
        if not _CONTEXT.search(sentence):
            continue
        for rx, pair in ((_RANGE, True), (_SINGLE, False)):
            for m in rx.finditer(sentence):
                before = sentence[max(0, m.start() - 40):m.start()]
                if _PAST.search(before[-24:]) or _FLOOR.search(before) or _WEEK_LABEL.search(before):
                    continue
                if not (_CUE_BEFORE.search(before) or _CUE_AFTER.match(sentence[m.end():])) \
                        or _someone_else(sentence, m.start(), last):
                    continue
                lo = _n(m.group(1))
                hi = _n(m.group(2)) if pair else lo
                if 0 < lo <= hi <= 17:
                    return (lo, hi)
        m = _MONTHS.search(sentence)
        if m and _CUE_BEFORE.search(sentence[max(0, m.start() - 40):m.start()]) \
                and not (_PAST.search(sentence[max(0, m.start() - 24):m.start()])
                         or _FLOOR.search(sentence[max(0, m.start() - 40):m.start()])
                         or _someone_else(sentence, m.start(), last)):
            k = {"a": 1, "one": 1, "two": 2, "three": 3}[m.group(1).lower()] * 4
            return (k, k)
    return None


def _plain(html: str) -> str:
    """An article's text with its markup out, one paragraph per line."""
    text = re.sub(r"</p>|<br\s*/?>", "\n", str(html or ""), flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"&amp;", "&", re.sub(r"&#?\w+;", " ", text))


def player_timeline(item: dict, last: str) -> tuple[tuple[int, int] | str, str] | None:
    """The timeline ``item`` gives for the player whose last name is ``last``,
    with the sentence it was read from. A RotoWire blurb or an ESPN headline is
    about him, so its sentences count when they name him or say "he/his"; an
    ESPN article is about many players, so only a sentence that names him
    counts."""
    head = str(item.get("headline") or item.get("title") or "")
    body = str(item.get("story") or item.get("description") or "")
    about_him = item.get("type") in PLAYER_TYPES
    text = f"{head}. {body}" if about_him else _plain(f"{head}. {body}")
    for para in text.split("\n"):
        for sentence in re.split(r"(?<=[.!?])\s+" if about_him else r"(?<=[.!?])\s+|;", para):
            names_him = re.search(rf"\b{re.escape(last)}\b", sentence) is not None
            if not (names_him or (about_him and re.match(r"\s*(He|His|The injury)\b", sentence))):
                continue
            t = timeline(sentence, last)
            if t is not None:
                return t, sentence.strip()[:240]
    return None


#: Reports are read when no older than this (days); ESPN's articles, which
#: cover many players, only when this fresh.
TIMELINE_DAYS = 28
ARTICLE_DAYS = 7


def timelines(snapshot, cfg: Config, espn_ids: dict[str, str], allow_network: bool = True,
              now: datetime | None = None) -> list[str]:
    """Who is out how long, from the news: for each player who is out (or
    just had news that says he may be) and has a recent report with a time
    in it, set ``reported_back`` (the earliest and latest NFL week he is
    back, or both 99 for a season-ending injury), ``reported_share`` (the
    share of the rest of the season he is expected to play) and a note. The
    season simulation then sits him until then instead of drawing how long
    absences usually last, which for a player not yet moved to IR said two
    weeks whatever the report. A newer item that says he is healthy (back at
    practice, cleared) cancels it. Returns a line per player."""
    from ..roster.players import normalize_name

    for p in snapshot.all_players():
        p.reported_back, p.reported_share, p.reported_note = None, None, ""
    now = now or datetime.now(UTC)
    week = int(snapshot.week)
    news_names = {normalize_name(n.name) for n in getattr(snapshot, "_news", []) if n.tag in ("out", "injury")}
    out, logged = [], []
    for p in snapshot.all_players():
        if p.position not in ("QB", "RB", "WR", "TE") or p.unsigned:
            continue
        key = normalize_name(p.name)
        if not (p.is_out or p.is_long_term_out or p.in_ir_slot or key in news_names):
            continue
        eid = espn_ids.get(key)
        if not eid:
            continue
        last = p.name.split()[-1]
        found = None
        for it in player_items(cfg, eid, allow_network):
            when = _when(it)
            head = str(it.get("headline") or "")
            kind = it.get("type")
            if when is None or now - when > timedelta(days=TIMELINE_DAYS) or IN_GAME.search(head):
                continue
            if kind not in PLAYER_TYPES and not (kind == "Story" and now - when <= timedelta(days=ARTICLE_DAYS)):
                continue
            if kind in PLAYER_TYPES and tag_of(head) == "healthy" and found is None:
                break                                  # the newest word is that he is back
            got = player_timeline(it, last)
            if got is not None:
                found = (got[0], when, got[1])
                break
        if found is None:
            continue
        t, when, evidence = found
        elapsed = max(int(round((now - when).total_seconds() / (7 * 86400))), 0)
        if t == "season":
            lo = hi = 99
            note = f"reported out for the season ({when:%b %d})"
            share = 0.0
        else:
            lo, hi = week + max(t[0] - elapsed, 0), week + max(t[1] - elapsed, 0)
            if hi <= week:
                continue                               # the reported time is up: nothing to say now
            lo = max(lo, week + 1) if (p.is_out or p.is_long_term_out or p.in_ir_slot) else lo
            span = f"{t[0]}-{t[1]} weeks" if t[1] != t[0] else f"{t[0]} weeks"
            note = (f"reported out {span} ({when:%b %d}): back around week "
                    f"{lo}{'-' + str(hi) if hi != lo else ''}")
            weeks_left = max(LAST_WEEK - week + 1, 1)
            share = min(max(LAST_WEEK - (lo + hi) / 2 + 1, 0.0) / weeks_left, 1.0)
        p.reported_back, p.reported_share, p.reported_note = [lo, hi], round(share, 3), note
        out.append(f"{p.name}: {note}")
        logged.append({"season": int(snapshot.season), "name": p.name, "position": p.position,
                       "reported": when.isoformat(), "weeks_low": None if t == "season" else t[0],
                       "weeks_high": None if t == "season" else t[1], "season_ending": t == "season",
                       "back_lo": lo, "back_hi": hi, "logged_week": week, "evidence": evidence})
    if allow_network and logged:
        try:
            record_timelines(cfg.results_dir / "timeline_log.parquet", logged)
        except Exception as exc:  # noqa: BLE001 - a log; never block the page
            log.warning("timeline log not written: %s", exc)
    return out


def record_timelines(path, rows: list[dict]) -> int:
    """Keep every reported timeline once (player and report time), so that
    over the season each can be graded against the week he actually came
    back: do reports run long or short, and by how much."""
    import pandas as pd

    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_parquet(path)
        seen = set(zip(old["season"], old["name"], old["reported"]))
        new = new[[k not in seen for k in zip(new["season"], new["name"], new["reported"])]]
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return len(new)


# ---------------------------------------------------------------------------
# Breaking news, for everyone
# ---------------------------------------------------------------------------
#: Hours FantasyPros' league-wide feed (its latest 100 items) is reused: it
#: says who has news; ESPN's player feed says what it is, in words the page
#: can show (FantasyPros' own text is never shown).
FEED_HOURS = 1.0
#: How far back the page lists news.
SHOW_HOURS = 48
#: What a headline is about, checked in this order (a RotoWire headline
#: carries the injury in parentheses -- "Hill (knee)" -- so "practiced
#: fully" must be read before the body part).
TAGS = (
    ("out", re.compile(r"\b(ruled out|won't play|will not play|won't suit up|inactive for|placed on (?:injured reserve|IR)\b|"
                       r"(?:moved|headed|going) to (?:injured reserve|IR)\b|season-ending|"
                       r"out for (?:the )?(?:season|year|week|sunday|monday|thursday|saturday))", re.I)),
    ("healthy", re.compile(r"\b(full participant|practiced fully|full practice|cleared|expected to play|will play|"
                           r"off the injury report|activated|returns? to practice|no injury designation)\b", re.I)),
    ("legal", LEGAL),
    ("move", re.compile(r"\b(traded|signs|signed|re-signs|released|waived|claimed|cut by|agree\w* to terms)\b", re.I)),
    ("role up", re.compile(r"\b(will start|named (?:the )?starter|starting (?:role|job|nod)|first-team|promoted|"
                           r"lead back|takes? over|top option)\b", re.I)),
    ("role down", re.compile(r"\b(benched|demoted|loses? (?:the |his )?(?:starting|job)|backup role|healthy scratch|"
                             r"reduced role)\b", re.I)),
    ("injury", re.compile(r"\b(doubtful|questionable|game-time decision|limited|did not practice|didn't practice|"
                          r"no practice|DNP|injur\w*|sprain\w*|strain\w*|concussion|MRI)\b|\(\w[\w ]*\)", re.I)),
)
#: An in-game injury ("won't return") is not news about the next game.
IN_GAME = re.compile(r"\b(won't return|will not return|remainder|rest of the game|for the rest of)\b", re.I)


@dataclass
class NewsItem:
    """One piece of news about one player, for the page."""

    name: str
    position: str
    team: str
    published: datetime
    headline: str
    link: str
    source: str
    tag: str = ""
    whose: str = ""           # "yours", "opponent", "rostered", "free agent"
    acted: str = ""           # what the model did with it


def tag_of(headline: str) -> str:
    for tag, pattern in TAGS:
        if pattern.search(headline or ""):
            return tag
    return ""


def fp_feed(cfg: Config, allow_network: bool = True) -> list[dict]:
    """FantasyPros' latest 100 news items, league-wide (who has news)."""
    from .fantasypros import api_key

    path = cfg.raw_dir / "news" / "fantasypros.json"
    fresh = path.exists() and time.time() - path.stat().st_mtime < FEED_HOURS * 3600
    if fresh or not allow_network or not api_key():
        try:
            return list(json.loads(path.read_text(encoding="utf-8")).get("items") or []) if path.exists() else []
        except ValueError:
            return []
    try:
        r = _get(FANTASYPROS_NEWS, headers={"x-api-key": api_key()}, params={"limit": 100})
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001 - a bonus source
        log.warning("FantasyPros news failed: %s", type(exc).__name__)
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return list(data.get("items") or [])


def _fp_when(item: dict) -> datetime | None:
    raw = str(item.get("created") or "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw[:19], fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return _when({"published": raw})


def week_start(now: datetime) -> datetime:
    """Tuesday 10:00 UTC before ``now``: the NFL week turns over once Monday
    night is done."""
    days = (now.weekday() - 1) % 7
    start = (now - timedelta(days=days)).replace(hour=10, minute=0, second=0, microsecond=0)
    return start if start <= now else start - timedelta(days=7)


def breaking(snapshot, cfg: Config, espn_ids: dict[str, str], fp_names: dict[str, str],
             allow_network: bool = True, now: datetime | None = None) -> list[NewsItem]:
    """The last :data:`SHOW_HOURS` of news about every player in this league
    -- yours, your opponent's, every rostered player and every free agent the
    platform lists -- newest first within each group.

    Who has news comes from FantasyPros' league-wide feed (``fp_names``:
    FantasyPros player id -> name); your roster and your opponent's are asked
    about every run as well. What the news says comes from ESPN's player
    feed (RotoWire's blurbs, ESPN's headlines), shown with a link. One thing
    it changes: a RotoWire headline this week that rules a player out of his
    next game sets him OUT before the platform's tag catches up."""
    from ..roster.players import normalize_name

    now = now or datetime.now(UTC)
    since = now - timedelta(hours=SHOW_HOURS)
    start = week_start(now)
    me = snapshot.my_team
    opp = snapshot.opponent
    mine = {p.player_id for p in me.roster}
    theirs = {p.player_id for p in opp.roster} if opp is not None else set()
    rostered = {p.player_id for t in snapshot.teams for p in t.roster}
    by_name: dict[str, object] = {}
    for p in snapshot.all_players():
        if p.position in ("QB", "RB", "WR", "TE"):
            by_name.setdefault(normalize_name(p.name), p)
    flagged: dict[str, dict] = {}
    for it in fp_feed(cfg, allow_network):
        when = _fp_when(it)
        name = fp_names.get(str(it.get("player_id") or ""))
        if when is None or when < since or not name:
            continue
        key = normalize_name(name)
        if key in by_name and (key not in flagged or when > _fp_when(flagged[key])):
            flagged[key] = it
    ask = set(flagged) | {normalize_name(p.name) for p in snapshot.all_players()
                          if p.player_id in mine | theirs and p.position in ("QB", "RB", "WR", "TE")}
    out: list[NewsItem] = []
    for key in ask:
        p = by_name.get(key)
        if p is None:
            continue
        whose = ("yours" if p.player_id in mine else "opponent" if p.player_id in theirs
                 else "rostered" if p.player_id in rostered else "free agent")
        espn_id = espn_ids.get(key)
        flag_time = _fp_when(flagged[key]) if key in flagged else None
        items = player_items(cfg, espn_id, allow_network, newer_than=flag_time) if espn_id else []
        last = p.name.split()[-1]
        shown = 0
        for it in items:
            when = _when(it)
            head = str(it.get("headline") or "")
            if when is None or when < since or it.get("type") not in PLAYER_TYPES or last not in head:
                continue
            link = ((it.get("links") or {}).get("web") or {}).get("href", "") if isinstance(it.get("links"), dict) else ""
            item = NewsItem(p.name, p.position, p.team or "FA", when, _short(head, 220), link,
                            "RotoWire via ESPN" if it.get("type") == "Rotowire" else "ESPN", tag_of(head), whose)
            if (item.tag == "out" and it.get("type") == "Rotowire" and head.startswith(last) and when >= start
                    and not IN_GAME.search(head) and not p.locked and p.actual_points is None
                    and p.status not in ("OUT", "O") and not p.is_long_term_out):
                long_term = re.search(r"injured reserve|\bIR\b|season-ending|out for (?:the )?(?:season|year)", head, re.I)
                p.status = "IR" if long_term else "OUT"
                item.acted = "set out for this week" + (" and on IR" if long_term else "") + ", ahead of the platform"
            out.append(item)
            shown += 1
        if not shown and key in flagged:
            it = flagged[key]
            out.append(NewsItem(p.name, p.position, p.team or "FA", _fp_when(it), "news at FantasyPros",
                                str(it.get("link") or ""), "FantasyPros", "", whose))
    order = {"yours": 0, "opponent": 1, "rostered": 2, "free agent": 3}
    out.sort(key=lambda n: (order.get(n.whose, 4), -n.published.timestamp()))
    return out
