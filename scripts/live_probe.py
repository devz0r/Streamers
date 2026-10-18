"""Can the page read live NFL scores from ESPN straight from the browser?

    python scripts/live_probe.py

Asks ESPN's public NFL scoreboard and one game summary (box score) the way a
browser on another site would (an ``Origin`` header), and prints only what the
answer looks like: the status, the CORS headers, and the shape of the data --
key names and counts, never a name or a number. Run as the ``live-probe`` job.
"""

from __future__ import annotations

import json
import sys
import urllib.request

ORIGIN = "https://devz0r.github.io"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={}"


def ask(url: str, preflight: bool = False):
    req = urllib.request.Request(url, method="OPTIONS" if preflight else "GET", headers={
        "Origin": ORIGIN, "User-Agent": "Mozilla/5.0",
        **({"Access-Control-Request-Method": "GET"} if preflight else {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), b""


def cors(headers: dict) -> str:
    keys = {k.lower(): v for k, v in headers.items()}
    return (f"allow-origin={keys.get('access-control-allow-origin')!r} "
            f"allow-methods={keys.get('access-control-allow-methods')!r} "
            f"cache-control={keys.get('cache-control')!r}")


def shape(obj, depth=0, limit=3) -> str:
    if depth > limit:
        return "..."
    if isinstance(obj, dict):
        return "{" + ", ".join(f"{k}: {shape(v, depth + 1, limit)}" for k, v in list(obj.items())[:12]) + "}"
    if isinstance(obj, list):
        return f"[{len(obj)} x {shape(obj[0], depth + 1, limit) if obj else ''}]"
    return type(obj).__name__


def main() -> int:
    status, headers, body = ask(SCOREBOARD)
    print("scoreboard GET", status, cors(headers))
    status_o, headers_o, _ = ask(SCOREBOARD, preflight=True)
    print("scoreboard OPTIONS", status_o, cors(headers_o))
    if status != 200:
        return 0
    sb = json.loads(body)
    events = sb.get("events", [])
    states = [e.get("status", {}).get("type", {}).get("state") for e in events]
    print("events", len(events), "states", {s: states.count(s) for s in set(states)})
    if events:
        print("event shape:", shape(events[0], limit=2))
        print("status shape:", shape(events[0].get("status", {}), limit=3))
        pick = next((e for e in events if e.get("status", {}).get("type", {}).get("state") == "in"), None) \
            or next((e for e in events if e.get("status", {}).get("type", {}).get("state") == "post"), events[0])
        status, headers, body = ask(SUMMARY.format(pick["id"]))
        print("summary GET", status, cors(headers))
        if status == 200:
            s = json.loads(body)
            box = s.get("boxscore", {})
            print("summary keys:", sorted(s)[:20])
            teams = box.get("players", [])
            print("boxscore players teams:", len(teams))
            if teams:
                cats = teams[0].get("statistics", [])
                print("stat groups:", [(c.get("name"), len(c.get("athletes", [])), c.get("keys") or c.get("labels"))
                                       for c in cats])
                ath = (cats[0].get("athletes") or [{}])[0] if cats else {}
                print("athlete shape:", shape(ath, limit=2))
            print("team stats shape:", shape(box.get("teams", [{}])[0], limit=2) if box.get("teams") else None)
            print("scoring plays:", len(s.get("scoringPlays", [])), shape((s.get("scoringPlays") or [{}])[0], limit=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
