"""Read a Yahoo league the way your browser does.

Yahoo stopped provisioning the Fantasy Sports API to new applications in 2026,
so the developer API is closed to most people. The fantasy website is not: it
serves your roster to your own logged-in session every time you open it. This
module uses that same session -- the cookie header copied out of a browser --
exactly as the ESPN adapter uses ``espn_s2``/``SWID``.

The trade is honest and worth stating: this reads pages meant for a browser,
so Yahoo changing their markup can break it, and a session cookie expires
every few months and has to be re-copied. Nothing here writes to a league or
touches an account that is not the cookie owner's own.

``probe`` exists because the parser cannot be written blind: it reports the
*shape* of what comes back -- status, whether the session is still logged in,
which embedded JSON blobs and tables are present -- without echoing cookies or
dumping page content.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .model import PlayerRow

log = logging.getLogger(__name__)

BASE = "https://football.fantasysports.yahoo.com"

#: A browser-shaped User-Agent. Yahoo serves a different (or no) page to
#: something that announces itself as a script.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)


#: Cookies that are noise for our purposes -- analytics, ad and consent
#: trackers. Dropping them shortens the header and narrows what is stored.
_JUNK_PREFIXES = ("_ga", "_gid", "_gcl", "_fbp", "_uetsid", "_uetvid", "fpc",
                  "gpp", "cmp", "axids", "tbla_id", "_cb", "__gads", "__gpi")


def normalize_cookie(raw: str, drop_junk: bool = False) -> str:
    """Turn whatever the person pasted into a Cookie header.

    Browsers offer cookies in several shapes and none of them is the one an
    HTTP client wants. Accepted here:

    * an actual ``Cookie:`` header -- ``a=1; b=2`` -- used as-is;
    * rows copied out of the DevTools storage table, where each line is
      ``name<TAB>value<TAB>domain<TAB>path<TAB>expiry...``;
    * one ``name=value`` per line.

    Asking someone to hand-assemble twenty pairs into one line is asking for a
    silent typo, and a malformed cookie fails as "logged out", which looks
    like the wrong problem entirely.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    pairs: list[tuple[str, str]] = []

    lines = [ln for ln in text.splitlines() if ln.strip()]
    tabular = any("\t" in ln for ln in lines)
    if tabular or len(lines) > 1:
        for line in lines:
            line = line.rstrip(";").strip()
            if "\t" in line:
                cells = [c.strip() for c in line.split("\t") if c.strip()]
                if len(cells) >= 2 and "=" not in cells[0]:
                    pairs.append((cells[0], cells[1]))
                    continue
                line = cells[0] if cells else ""
            # A line that is itself a header fragment, or one pair.
            for chunk in line.split(";"):
                chunk = chunk.strip()
                if "=" in chunk:
                    name, _, value = chunk.partition("=")
                    if name.strip():
                        pairs.append((name.strip(), value.strip()))
    else:
        for chunk in text.split(";"):
            chunk = chunk.strip()
            if "=" in chunk:
                name, _, value = chunk.partition("=")
                if name.strip():
                    pairs.append((name.strip(), value.strip()))

    seen: dict[str, str] = {}
    for name, value in pairs:
        if drop_junk and name.lower().startswith(_JUNK_PREFIXES):
            continue
        seen[name] = value            # last one wins, as a browser would
    return "; ".join(f"{k}={v}" for k, v in seen.items())


def cookie_names(cookie: str) -> list[str]:
    """Just the names, for diagnostics that must not echo values."""
    return [c.partition("=")[0].strip() for c in (cookie or "").split(";") if "=" in c]


def credentials() -> dict[str, str | None]:
    """Cookie header and league identifiers from the environment."""
    return {
        "cookie": normalize_cookie(os.environ.get("YAHOO_COOKIE", "")) or None,
        "league_id": os.environ.get("YAHOO_LEAGUE_ID", "").strip() or None,
        "team_id": os.environ.get("YAHOO_TEAM_ID", "").strip() or None,
    }


def _session(cookie: str):
    import requests

    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Cookie": cookie,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


def fetch(url: str, cookie: str, timeout: float = 25.0):
    """GET ``url`` with the browser session. Returns the response."""
    return _session(cookie).get(url, timeout=timeout, allow_redirects=True)


def looks_logged_out(resp) -> bool:
    """Whether Yahoo bounced us to a login wall instead of the page."""
    final = str(getattr(resp, "url", "")).lower()
    if any(k in final for k in ("login", "signin", "account/challenge")):
        return True
    body = (resp.text or "")[:4000].lower()
    return "sign in" in body and "fantasy" not in body


# ---------------------------------------------------------------------------
# Embedded JSON
# ---------------------------------------------------------------------------
#: Yahoo front ends embed their state in a script tag. The exact variable has
#: changed over the years, so several spellings are tried.
_JSON_PATTERNS = (
    re.compile(r"root\.App\.main\s*=\s*(\{.*?\});", re.S),
    re.compile(r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\});", re.S),
    re.compile(r"window\.__PRELOADED_STATE__\s*=\s*(\{.*?\});", re.S),
    re.compile(r'<script[^>]+type="application/json"[^>]*>(\{.*?\})</script>', re.S),
)


def embedded_json(html: str) -> dict[str, Any] | None:
    """The page's embedded state object, if one of the known shapes is there."""
    for pattern in _JSON_PATTERNS:
        m = pattern.search(html or "")
        if not m:
            continue
        try:
            return json.loads(m.group(1))
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
@dataclass
class ProbeResult:
    """What a page looks like, in terms safe to print."""

    url: str
    status: int = 0
    final_url: str = ""
    logged_out: bool = False
    bytes: int = 0
    title: str = ""
    json_blob: str = ""
    json_top_keys: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    error: str = ""

    def lines(self) -> list[str]:
        if self.error:
            return [f"  {self.url}", f"    ERROR {self.error}"]
        out = [
            f"  {self.url}",
            f"    status {self.status}  bytes {self.bytes}  "
            f"{'LOGGED OUT' if self.logged_out else 'session ok'}",
        ]
        if self.final_url and self.final_url != self.url:
            out.append(f"    redirected to {self.final_url}")
        if self.title:
            out.append(f"    title: {self.title[:90]}")
        if self.json_blob:
            out.append(f"    embedded JSON: {self.json_blob} "
                       f"top-level keys: {', '.join(self.json_top_keys[:14])}")
        else:
            out.append("    embedded JSON: none of the known shapes")
        if self.tables:
            out.append(f"    tables ({len(self.tables)}): {'; '.join(self.tables[:10])}")
        for h in self.hints:
            out.append(f"    hint: {h}")
        return out


def _describe_tables(html: str) -> list[str]:
    """id/class of each table, plus its header cells -- structure, not data."""
    out = []
    for m in re.finditer(r"<table\b([^>]*)>(.*?)</table>", html or "", re.S | re.I):
        attrs, body = m.group(1), m.group(2)
        ident = " ".join(
            f"{k}={v}" for k, v in re.findall(r'\b(id|class)="([^"]{0,70})"', attrs)
        ) or "(no id/class)"
        heads = re.findall(r"<th\b[^>]*>(.*?)</th>", body, re.S | re.I)[:9]
        heads = [re.sub(r"<[^>]+>", "", h).strip()[:16] for h in heads]
        rows = len(re.findall(r"<tr\b", body, re.I))
        out.append(f"[{ident}] rows={rows} th={heads}")
    return out


def probe_page(url: str, cookie: str, league_id: str = "", detail: bool = False) -> ProbeResult:
    """Fetch one page and describe its shape."""
    res = ProbeResult(url=url)
    try:
        resp = fetch(url, cookie)
    except Exception as exc:  # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"
        return res
    res.status = resp.status_code
    res.final_url = str(resp.url)
    res.bytes = len(resp.content or b"")
    html = resp.text or ""
    res.logged_out = looks_logged_out(resp)
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    if m:
        res.title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1))).strip()
    for pattern in _JSON_PATTERNS:
        hit = pattern.search(html)
        if hit:
            res.json_blob = pattern.pattern[:40]
            try:
                res.json_top_keys = sorted(json.loads(hit.group(1)).keys())
            except ValueError:
                res.json_top_keys = ["(did not parse)"]
            break
    res.tables = _describe_tables(html)
    for needle, note in (
        ("data-tst=", "carries data-tst test hooks, usable as parse anchors"),
        ("Ttl-t", "uses Ttl-* atomic classes"),
        ("player-status", "player-status markers present"),
        ("js-team-roster", "js-team-roster container present"),
    ):
        if needle in html:
            res.hints.append(note)
    if detail and not res.logged_out and res.status < 400:
        try:
            res.hints.extend(["detail:"] + detail_page(html, league_id))
        except Exception as exc:  # noqa: BLE001 - diagnostics must not crash
            res.hints.append(f"detail failed: {type(exc).__name__}: {exc}")
    return res


def my_team_id(html: str, league_id: str) -> str | None:
    """The logged-in manager's team id, from the 'My Team' navigation link."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a"):
        if a.get_text(strip=True).lower() in ("my team", "my roster"):
            m = re.search(rf"/f1/{re.escape(league_id)}/(\d+)", a.get("href", ""))
            if m:
                return m.group(1)
    return None


def probe(league_id: str, team_id: str | None, cookie: str, week: int | None = None,
          detail: bool = False) -> list[ProbeResult]:
    """Describe the handful of pages a snapshot would need."""
    wk = f"?week={week}" if week else ""
    results = [probe_page(f"{BASE}/f1/{league_id}", cookie, league_id, detail)]
    if not team_id:
        try:
            team_id = my_team_id(fetch(f"{BASE}/f1/{league_id}", cookie).text, league_id)
        except Exception:  # noqa: BLE001
            team_id = None
        if team_id:
            results[0].hints.append(f"auto-detected your team id: {team_id}")
    if team_id:
        results.append(probe_page(f"{BASE}/f1/{league_id}/{team_id}{wk}", cookie, league_id, detail))
    results.append(probe_page(f"{BASE}{free_agent_path(league_id, 'O', week or 1, 0)}",
                              cookie, league_id, detail))
    results.append(probe_page(f"{BASE}/f1/{league_id}/matchup{wk}", cookie, league_id, detail))
    return results


# ---------------------------------------------------------------------------
# Detailed structure (for writing the parser)
#
# Workflow logs on a public repository are publicly readable, so everything
# printed here must be safe to publish: tag names, class names and Yahoo's
# data-tst test hooks are the same for every user and are kept; any text
# that could be a name, an email or a token is replaced by a placeholder.
# Short structural codes -- positions, slots, injury tags, team codes -- are
# kept, because they are what the parser has to recognise and they identify
# nobody.
# ---------------------------------------------------------------------------
_KEEP_TEXT = re.compile(
    r"^(?:[A-Z]{1,4}|W/R/T|W/R|W/T|Q/W/R/T|IR-R|IR-NR|PUP-R|NFI-R|DEF|FLEX|BN|IR|"
    r"Proj|Fan Pts|Pos|Stats|Player|Bye|Opp|Status|Rank|Team|Owner|Waivers?|FA|"
    r"Questionable|Doubtful|Out|Probable|Injured Reserve|Suspended)$"
)
_KEEP_ATTRS = ("class", "id", "data-tst", "data-pos", "data-ys-playerid", "data-player-id",
               "data-target", "role", "scope", "colspan")


def _scrub_text(text: str) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    if not t:
        return ""
    if _KEEP_TEXT.match(t):
        return t
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", t):
        return "«n»"
    if re.fullmatch(r"\d+-\d+(?:-\d+)?", t):
        return "«rec»"
    # "BUF - QB" / "Buf - QB" style: team code and position, both structural.
    m = re.fullmatch(r"([A-Za-z]{2,4})\s*-\s*([A-Z/,\s]{1,20})", t)
    if m:
        return f"{m.group(1)} - {m.group(2).strip()}"
    # Game status: "Sun 1:00 pm vs NYJ", "Final W 24-17 @ Buf", "Bye".
    g = re.fullmatch(r"((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{1,2}:\d{2}\s*[ap]m)\s+(vs|@)\s+([A-Za-z]{2,4})", t)
    if g:
        return f"«day time» {g.group(2)} {g.group(3)}"
    g = re.fullmatch(r"(Final|Bye|Live|Halftime|Q\d)\b.*", t)
    if g:
        return f"{g.group(1)}…"
    return f"«t{len(t)}»"


def _scrub_attr(name: str, value) -> str | None:
    if isinstance(value, list):
        value = " ".join(value)
    value = str(value)
    if name == "href":
        path = value.split("?")[0].split("#")[0]
        path = re.sub(r"https?://[^/]+", "", path)
        return re.sub(r"\d+", "«n»", path)[:70]
    if name in ("data-ys-playerid", "data-player-id"):
        return "«n»" if value.strip() else ""
    if name == "id":
        return re.sub(r"\d+", "«n»", value)[:80]
    if name in _KEEP_ATTRS:
        return value[:80]
    if name.startswith("data-") and len(value) <= 24 and not re.search(r"@|\d{6,}", value):
        return value
    return None


def skeleton(node, depth: int = 0, max_depth: int = 9, budget: list | None = None) -> list[str]:
    """Indented tag/attribute outline of ``node`` with text scrubbed."""
    from bs4 import NavigableString, Tag

    budget = budget if budget is not None else [120]
    out: list[str] = []
    if budget[0] <= 0 or depth > max_depth:
        return out
    for child in node.children:
        if budget[0] <= 0:
            break
        if isinstance(child, NavigableString):
            t = _scrub_text(str(child))
            if t:
                out.append("  " * depth + f'"{t}"')
                budget[0] -= 1
            continue
        if not isinstance(child, Tag) or child.name in ("script", "style", "svg", "path"):
            continue
        out.append(_tag_line(child, depth))
        budget[0] -= 1
        out.extend(skeleton(child, depth + 1, max_depth, budget))
    return out


_HEADER_WORDS = {
    "pos", "player", "fan", "pts", "proj", "projected", "%", "start", "started", "ros",
    "rost", "rostered", "opp", "opponent", "bye", "status", "action", "stats", "rank",
    "yds", "td", "int", "att", "rec", "tgt", "ret", "2pt", "fum", "lost", "fg", "pat",
    "made", "miss", "sack", "safe", "pa", "ya", "allow", "allowed", "passing", "rushing",
    "receiving", "returns", "misc", "fumbles", "kicking", "defense", "team", "owner",
    "waivers", "waiver", "forecast", "game", "gp", "pre-season", "current", "o-rank",
    "pct", "avg", "total", "points", "fantasy", "rankings", "trends", "edit", "comp",
    "cmp", "ints", "tds", "tack", "solo", "blk", "kick", "punt", "xpm", "fgm", "time",
}


def _scrub_header(text: str) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    if not t:
        return ""
    ok = True
    for tok in re.split(r"[\s/]+", t):
        low = tok.lower().strip(".:*")
        if not low or low in _HEADER_WORDS or re.fullmatch(r"\d{1,2}(-\d{1,2}|\+)?|[%#+]+|w\d+|wk\d+", low):
            continue
        ok = False
        break
    return t if ok else f"«t{len(t)}»"


def _tag_line(tag, depth: int) -> str:
    """``<name attr="scrubbed"...>`` for one element."""
    attrs = []
    for k, v in tag.attrs.items():
        if k in ("href", *_KEEP_ATTRS) or k.startswith("data-") or k == "title":
            if k == "title":
                attrs.append(f'title="{_scrub_text(str(v))}"')
                continue
            sv = _scrub_attr(k, v)
            if sv is not None:
                attrs.append(f'{k}="{sv}"')
    return "  " * depth + f"<{tag.name}{(' ' + ' '.join(attrs)) if attrs else ''}>"


def detail_page(html: str, league_id: str) -> list[str]:
    """Structure worth seeing before writing a parser for this page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    lines: list[str] = []

    # Script-assigned state objects, by name and size only.
    names = []
    for sc in soup.find_all("script"):
        body = sc.string or sc.get_text() or ""
        for m in re.finditer(r"(?:window\.|var |let |const )([A-Za-z_$][\w$.]{2,40})\s*=\s*[\[{]", body):
            names.append(f"{m.group(1)}({len(body) // 1024}KB)")
        if sc.get("type") == "application/json" or sc.get("id", "").lower().endswith("data"):
            names.append(f"<script type={sc.get('type')} id={sc.get('id')}>({len(body) // 1024}KB)")
    if names:
        lines.append("    script state: " + ", ".join(sorted(set(names))[:12]))

    # Team ids linked from this page -- small integers, safe.
    ids = re.findall(rf"/f1/{re.escape(league_id)}/(\d+)(?=[\"'?#/]|$)", html)
    if ids:
        uniq = sorted({int(i) for i in ids})
        lines.append(f"    team ids linked: {uniq}")
    for a in soup.find_all("a"):
        if a.get_text(strip=True).lower() in ("my team", "my roster"):
            m = re.search(rf"/f1/{re.escape(league_id)}/(\d+)", a.get("href", ""))
            if m:
                lines.append(f"    'My Team' nav link -> team id {m.group(1)}")
                break

    # Every table's first data rows, outlined. Nested tables are handled by
    # the real parser, so direct rows only.
    for i, table in enumerate(soup.find_all("table")):
        rows = [tr for tr in table.find_all("tr", recursive=True)
                if tr.find_parent("table") is table and tr.find("td")]
        if not rows:
            continue
        label = table.get("id") or " ".join(table.get("class", [])[:3]) or f"table#{i}"
        lines.append(f"    --- table [{label}] data rows={len(rows)}")
        # Header rows, with spans, so columns can be mapped by name.
        for hr in [tr for tr in table.find_all("tr") if tr.find_parent("table") is table
                   and tr.find("th") and not tr.find("td")]:
            cells = []
            for th in hr.find_all("th"):
                if th.find_parent("tr") is not hr:
                    continue
                span = ""
                if th.get("colspan") not in (None, "1"):
                    span += f"c{th.get('colspan')}"
                if th.get("rowspan") not in (None, "1"):
                    span += f"r{th.get('rowspan')}"
                cls = " ".join(c for c in th.get("class", []) if not c.startswith(("Ta-", "Px-", "Fz-")))[:30]
                cells.append(f"{_scrub_header(th.get_text(' ', strip=True))}"
                             f"{'{' + span + '}' if span else ''}{'[' + cls + ']' if cls else ''}")
            lines.append("      th: " + " | ".join(cells))
        for row in rows[:1]:
            lines.append(_tag_line(row, 3))
            for ln in skeleton(row, depth=4, max_depth=18, budget=[260]):
                lines.append(ln)
        if i >= 6:
            lines.append("    (further tables omitted)")
            break
    return lines


# ---------------------------------------------------------------------------
# Parsing
#
# Anchors, all confirmed against live pages by `yahoo-probe --detail`:
#   * lineup slot        span.pos-label[data-pos]            ("QB", "W/R/T", "BN")
#   * player id and name a.name[data-ys-playerid]            (DEFs link to /nfl/teams/)
#   * NFL team, position span.D-b span.Fz-xxs                ("LAR - QB", "TB - DEF")
#   * columns            the table's last header row, by label ("Proj Pts", "% Ros")
# Columns are looked up by header label, never by position, so Yahoo adding or
# reordering a column moves nothing.
# ---------------------------------------------------------------------------
_STATUS_CODES = re.compile(
    r"^(Q|D|O|P|IR|IR-R|IR-NR|PUP|PUP-P|PUP-R|NFI|NFI-R|SUSP|NA|DTD|COVID-19|INJ)$", re.I
)


_REAL_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF", "D", "DST", "D/ST"}


def _num(text: str) -> float | None:
    t = (text or "").strip().replace("%", "").replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def leaf_headers(table) -> list[str]:
    """Labels of the table's last header row, one per data column."""
    rows = [tr for tr in table.find_all("tr")
            if tr.find_parent("table") is table and tr.find("th") and not tr.find("td")]
    if not rows:
        return []
    out = []
    for th in rows[-1].find_all("th"):
        if th.find_parent("tr") is not rows[-1]:
            continue
        span = int(th.get("colspan") or 1)
        out.extend([th.get_text(" ", strip=True)] * max(span, 1))
    return out


def column(headers: list[str], *labels: str) -> int | None:
    """Index of the first header matching any of ``labels`` (case-insensitive)."""
    wanted = [lab.lower() for lab in labels]
    for i, h in enumerate(headers):
        if h.strip().lower() in wanted:
            return i
    return None


def data_rows(table) -> list:
    return [tr for tr in table.find_all("tr")
            if tr.find_parent("table") is table and tr.find("td")]


def direct_cells(tr) -> list:
    return [td for td in tr.find_all("td") if td.find_parent("tr") is tr]


def parse_player_cell(td) -> dict | None:
    """Player id, name, NFL team, positions and injury tag from a player cell."""
    link = td.select_one("a.name[data-ys-playerid]") or td.select_one("a[data-ys-playerid]")
    if link is None:
        return None
    pid = str(link.get("data-ys-playerid") or "").strip()
    name = link.get_text(" ", strip=True)
    team, positions = None, []
    # "LAR - QB" normally sits in span.D-b > span.Fz-xxs. Injury tags can share
    # the Fz-xxs styling -- "PUP-R" once parsed as team PUP, position R -- so the
    # D-b span is tried first, status codes are skipped, and a match only counts
    # if every position is a real one.
    candidates = td.select("span.D-b span.Fz-xxs") + td.select("span.Fz-xxs")
    for span in candidates:
        text = span.get_text(" ", strip=True)
        if _STATUS_CODES.match(text):
            continue
        m = re.fullmatch(r"\s*([A-Za-z]{2,4})\s+-\s+([A-Za-z/,\s]+?)\s*", text)
        if not m:
            continue
        found = [p.strip().upper() for p in m.group(2).split(",") if p.strip()]
        if found and all(p in _REAL_POSITIONS for p in found):
            team, positions = m.group(1), found
            break
    status = ""
    for el in td.find_all(True):
        classes = " ".join(el.get("class", [])).lower()
        if "injury" in classes or "status-tag" in classes:
            text = el.get_text(strip=True)
            if text and _STATUS_CODES.match(text):
                status = text.upper()
                break
    if not status:
        # Fall back to a bare status code sitting in the name block.
        block = td.select_one(".ysf-player-name") or td
        for el in block.find_all(["span", "abbr"]):
            own = "".join(t for t in el.find_all(string=True, recursive=False)).strip()
            if own and _STATUS_CODES.match(own):
                status = own.upper()
                break
    return {"player_id": pid, "name": name, "team": team, "positions": positions, "status": status}


def _player_row(cell: dict, slot: str, projection: float | None = None,
                pct_owned: float | None = None) -> PlayerRow:
    from ..teams import normalize_team
    from .model import canonical_position, canonical_slot, canonical_status

    positions = [canonical_position(p) for p in cell["positions"]] or [""]
    position = positions[0]
    try:
        team = normalize_team(cell["team"]) if cell["team"] else None
    except Exception:  # noqa: BLE001 - an unknown code is not worth failing on
        team = None
    eligible = sorted({canonical_position(p) for p in cell["positions"]})
    if position in ("RB", "WR", "TE"):
        eligible.append("FLEX")
    return PlayerRow(
        player_id=cell["player_id"], name=cell["name"], position=position, team=team,
        status=canonical_status(cell["status"]), slot=canonical_slot(slot),
        eligible_slots=eligible, platform_projection=projection, percent_owned=pct_owned,
    )


def parse_roster(html: str) -> tuple[list, dict[str, int], int]:
    """A team page: players, starting-slot counts, bench size."""
    from bs4 import BeautifulSoup

    from .model import NON_STARTING_SLOTS, canonical_slot

    soup = BeautifulSoup(html, "html.parser")
    players, slots, bench = [], {}, 0
    for table in soup.select("table[id^=statTable]"):
        headers = leaf_headers(table)
        c_player = column(headers, "Offense", "Kickers", "Defense/Special Teams", "Player") or 2
        c_proj = column(headers, "Proj Pts", "Proj")
        c_ros = column(headers, "% Ros", "% Rost")
        for tr in data_rows(table):
            label = tr.select_one("span.pos-label[data-pos]")
            if label is None:
                continue
            slot_raw = label.get("data-pos")
            slot = canonical_slot(slot_raw)
            if slot == "BN":
                bench += 1
            elif slot not in NON_STARTING_SLOTS:
                slots[slot] = slots.get(slot, 0) + 1
            cells = direct_cells(tr)
            cell_td = tr.select_one("td.player") or (cells[c_player] if c_player < len(cells) else None)
            info = parse_player_cell(cell_td) if cell_td is not None else None
            if info is None:
                continue                              # an empty slot
            proj = _num(cells[c_proj].get_text(strip=True)) if c_proj is not None and c_proj < len(cells) else None
            ros = _num(cells[c_ros].get_text(strip=True)) if c_ros is not None and c_ros < len(cells) else None
            players.append(_player_row(info, slot_raw, proj, ros))
    return players, slots, bench


def parse_standings(html: str, league_id: str) -> tuple[str, list[dict]]:
    """League name and one dict per team: id, name, wins, losses, ties, points_for."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    league_name = title.split("|")[0].strip()
    table = soup.select_one("table#standingstable")
    teams = []
    if table is None:
        return league_name, teams
    headers = leaf_headers(table)
    c_pf = column(headers, "PF", "Pts For", "Points For")
    for tr in data_rows(table):
        target = tr.get("data-target") or ""
        m = re.search(rf"/f1/{re.escape(league_id)}/(\d+)", target)
        if not m:
            a = tr.select_one(f'a[href*="/f1/{league_id}/"]')
            m = re.search(rf"/f1/{re.escape(league_id)}/(\d+)", a.get("href", "")) if a else None
        if not m:
            continue
        names = [a.get_text(strip=True) for a in tr.select("td.Tst-manager a") if a.get_text(strip=True)]
        wlt = (tr.select_one("td.Tst-wlt").get_text(strip=True) if tr.select_one("td.Tst-wlt") else "")
        parts = [int(x) for x in re.findall(r"\d+", wlt)] + [0, 0, 0]
        cells = direct_cells(tr)
        pf = _num(cells[c_pf].get_text(strip=True)) if c_pf is not None and c_pf < len(cells) else None
        teams.append({"team_id": m.group(1), "name": names[-1] if names else f"Team {m.group(1)}",
                      "wins": parts[0], "losses": parts[1], "ties": parts[2],
                      "points_for": pf or 0.0})
    return league_name, teams


def opponent_id(html: str, league_id: str, my_id: str) -> str | None:
    """The other team linked from your matchup page."""
    ids = {i for i in re.findall(rf"/f1/{re.escape(league_id)}/(\d+)(?=[\"'?#/]|$)", html)}
    ids.discard(str(my_id))
    return sorted(ids, key=int)[0] if len(ids) == 1 else None


def free_agent_path(league_id: str, pos: str, week: int, offset: int) -> str:
    """The player-list URL: available players, this week's projection, best first.

    One definition shared by the sync and the probe, so the probe always
    describes the page the sync actually parses.
    """
    return (f"/f1/{league_id}/players?status=A&pos={pos}&cut_type=9"
            f"&stat1=S_PW_{week}&myteam=0&sort=PTS&sdir=1&count={offset}")


def projected_stat_label(soup) -> str:
    """The stat type the player list is showing, e.g. 'Projected Stats (Week 3)'."""
    for sel in ("select[name=stat1] option[selected]", "select#statselect option[selected]"):
        opt = soup.select_one(sel)
        if opt is not None:
            return opt.get_text(" ", strip=True)
    return ""


def parse_free_agents(html: str, week: int) -> list:
    """One page of the player list.

    The Fan Pts column holds whatever stat type the page was asked for. It is
    only recorded as Yahoo's weekly projection when the page confirms that is
    what it shows -- a season total mistaken for a weekly projection would
    quietly corrupt the blend.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    label = projected_stat_label(soup).lower()
    selector_says_projection = "proj" in label and "week" in label
    out = []
    # The player list is whichever table has player cells in its rows. Asking
    # for projections renames the points column ("Proj Pts" rather than "Fan
    # Pts"), and keying the table on one label dropped every free agent.
    for table in soup.find_all("table"):
        rows = [tr for tr in data_rows(table) if tr.select_one("td.player")]
        if not rows:
            continue
        headers = leaf_headers(table)
        c_proj = column(headers, "Proj Pts", "Proj", "Projected Pts")
        c_fan = column(headers, "Fan Pts")
        c_ros = column(headers, "% Ros", "% Rost")
        # A column labelled as a projection is one; "Fan Pts" only counts when
        # the stat selector confirms it is showing this week's projection.
        c_pts = c_proj if c_proj is not None else (c_fan if selector_says_projection else None)
        for tr in rows:
            info = parse_player_cell(tr.select_one("td.player"))
            if info is None:
                continue
            cells = direct_cells(tr)
            pts = _num(cells[c_pts].get_text(strip=True)) if c_pts is not None and c_pts < len(cells) else None
            ros = _num(cells[c_ros].get_text(strip=True)) if c_ros is not None and c_ros < len(cells) else None
            out.append(_player_row(info, "FA", pts, ros))
        break
    if not out:
        # Say what the page looked like -- structure only -- so a layout
        # change explains itself in the log instead of just emptying the wire.
        shapes = []
        for table in soup.find_all("table")[:6]:
            heads = [_scrub_header(h) for h in leaf_headers(table)][:12]
            shapes.append(f"{len(data_rows(table))} rows {heads}")
        log.warning("Yahoo free-agent page parsed to no players; tables: %s; stat label: %s",
                    "; ".join(shapes) or "none", _scrub_header(label) if label else "none")
    return out


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
def fetch_snapshot(season: int, week: int, profile: str, pause: float = 0.6):
    """Read the league from Yahoo's website with the browser session.

    About eight page loads: the league, your team, your matchup, your
    opponent's team, and a few pages of the free-agent list. A short pause
    between them keeps this at the pace of a person clicking around.
    """
    import time
    from datetime import UTC, datetime

    from .model import LeagueSnapshot, Matchup, TeamRow

    creds = credentials()
    if not creds["cookie"]:
        raise RuntimeError("YAHOO_COOKIE is not set")
    if not creds["league_id"]:
        raise RuntimeError("YAHOO_LEAGUE_ID is not set")
    league, cookie = creds["league_id"], creds["cookie"]
    session = _session(cookie)

    def get(path: str) -> str:
        resp = session.get(f"{BASE}{path}", timeout=25)
        if looks_logged_out(resp):
            raise RuntimeError(
                "Yahoo session has expired or was signed out: re-copy the Cookie header "
                "from a logged-in fantasy page into the YAHOO_COOKIE secret")
        resp.raise_for_status()
        time.sleep(pause)
        return resp.text

    league_html = get(f"/f1/{league}")
    league_name, standings = parse_standings(league_html, league)
    my_id = creds["team_id"] or my_team_id(league_html, league)
    if not my_id:
        raise RuntimeError("could not find your team on the league page; set YAHOO_TEAM_ID")

    my_players, slots, bench = parse_roster(get(f"/f1/{league}/{my_id}?week={week}"))
    if not my_players:
        raise RuntimeError("your Yahoo team page parsed to no players; the page layout may "
                           "have changed (run `streamer yahoo-probe --detail`)")

    opp_id = opponent_id(get(f"/f1/{league}/matchup?week={week}"), league, my_id)
    opp_players: list = []
    if opp_id:
        opp_players, _s, _b = parse_roster(get(f"/f1/{league}/{opp_id}?week={week}"))

    free_agents: list = []
    for pos, pages in (("O", (0, 25)), ("K", (0,)), ("DEF", (0,))):
        for offset in pages:
            path = free_agent_path(league, pos, week, offset)
            try:
                free_agents.extend(parse_free_agents(get(path), week))
            except RuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001 - a thin wire beats no snapshot
                log.warning("Yahoo free agents %s@%s failed: %s", pos, offset, exc)

    teams = []
    for row in standings or [{"team_id": my_id, "name": "My team", "wins": 0,
                              "losses": 0, "ties": 0, "points_for": 0.0}]:
        tid = row["team_id"]
        roster = my_players if tid == str(my_id) else opp_players if tid == str(opp_id) else []
        teams.append(TeamRow(team_id=tid, name=row["name"], wins=row["wins"],
                             losses=row["losses"], ties=row["ties"],
                             points_for=float(row["points_for"] or 0.0),
                             roster=roster, is_mine=(tid == str(my_id))))
    if not any(t.is_mine for t in teams):
        teams.append(TeamRow(team_id=str(my_id), name="My team", roster=my_players, is_mine=True))
    if opp_id and not any(t.team_id == str(opp_id) for t in teams):
        teams.append(TeamRow(team_id=str(opp_id), name="Opponent", roster=opp_players))

    return LeagueSnapshot(
        platform="yahoo", profile=profile, league_id=str(league), league_name=league_name,
        season=season, week=week, slots=slots, bench_size=bench, teams=teams,
        free_agents=free_agents,
        matchup=Matchup(week=week, my_team_id=str(my_id), opponent_team_id=str(opp_id))
        if opp_id else None,
        synced_at=datetime.now(UTC).isoformat(),
        extra={"source": "yahoo-web"},
    )
