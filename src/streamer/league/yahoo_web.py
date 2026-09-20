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

log = logging.getLogger(__name__)

BASE = "https://football.fantasysports.yahoo.com"

#: A browser-shaped User-Agent. Yahoo serves a different (or no) page to
#: something that announces itself as a script.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)


def credentials() -> dict[str, str | None]:
    """Cookie header and league identifiers from the environment."""
    return {
        "cookie": os.environ.get("YAHOO_COOKIE", "").strip() or None,
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


def probe_page(url: str, cookie: str) -> ProbeResult:
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
    return res


def probe(league_id: str, team_id: str | None, cookie: str, week: int | None = None) -> list[ProbeResult]:
    """Describe the handful of pages a snapshot would need."""
    wk = f"?week={week}" if week else ""
    urls = [f"{BASE}/f1/{league_id}"]
    if team_id:
        urls.append(f"{BASE}/f1/{league_id}/{team_id}{wk}")
    urls.append(f"{BASE}/f1/{league_id}/players?status=A&pos=O&sort=AR&count=0")
    urls.append(f"{BASE}/f1/{league_id}/matchup{wk}")
    return [probe_page(u, cookie) for u in urls]
