"""Why a defence's or kicker's number moved during the week.

Their projections are re-made every refresh from the game lines (the
opponent's implied total, the spread), and the defences near the top sit
within a point of each other, so ranks reshuffle all week with nothing
having changed that matters. This keeps each unit's first reading of the
week and its latest (``results/<profile>/unit_drift.json``, one file, weeks
older than the current one dropped) so the page can say what moved and
by how much, and the streaming tables can tell a real change from a wobble.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

from ..config import Config
from ..teams import normalize_team

FILE = "unit_drift.json"
#: A swap is worth making only for this many projected points: the three
#: defences at the top of Yahoo's list this week sat within 0.8, and moved
#: 0.3-0.7 a refresh with only game-line moves behind it.
MOVE_POINTS = 1.0
#: A projection change smaller than this is not reported.
SHOW_POINTS = 0.3
#: An implied-total change smaller than this is not reported.
SHOW_IMPLIED = 0.5
_IMPLIED = re.compile(r"implied ([0-9]+(?:\.[0-9]+)?)")


def _path(cfg: Config):
    return cfg.results_dir / FILE


def _read(cfg: Config) -> dict:
    try:
        return json.loads(_path(cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def record(cfg: Config, snapshot, rankings, now: datetime | None = None) -> int:
    """Note each ranked unit's reading now; the first of the week stands."""
    stamp = (now or datetime.now(UTC)).isoformat()
    season, week = int(snapshot.season), int(snapshot.week)
    data = _read(cfg)
    if data.get("week") != [season, week]:
        data = {"week": [season, week], "units": {}}
    n = 0
    for pos, frame in (("DST", getattr(rankings, "dst", None)), ("K", getattr(rankings, "kicker", None))):
        if frame is None or frame.empty:
            continue
        for r in frame.itertuples():
            team = normalize_team(r.team)
            m = _IMPLIED.search(str(getattr(r, "rationale", "") or ""))
            now_ = {"exp": round(float(r.expected_points), 2), "imp": float(m.group(1)) if m else None,
                    "opp": normalize_team(getattr(r, "opponent", "") or "")}
            u = data["units"].setdefault(f"{pos}:{team}", {"first": {**now_, "at": stamp}})
            u["last"] = now_
            n += 1
    try:
        _path(cfg).parent.mkdir(parents=True, exist_ok=True)
        _path(cfg).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    except OSError:
        return 0
    return n


def load(cfg: Config, season: int, week: int) -> dict:
    """``"DST:WAS"`` -> {first, last} for this week, empty for any other."""
    d = _read(cfg)
    return d.get("units", {}) if d.get("week") == [season, week] else {}


def note(unit: dict | None, position: str) -> str:
    """What moved since the week's first reading, or "" when nothing worth saying."""
    if not unit or "first" not in unit or "last" not in unit:
        return ""
    a, b = unit["first"], unit["last"]
    if a.get("opp") != b.get("opp"):
        return f"opponent changed ({a.get('opp') or '?'} to {b.get('opp') or '?'})"
    d = b["exp"] - a["exp"]
    di = None if a.get("imp") is None or b.get("imp") is None else b["imp"] - a["imp"]
    if abs(d) < SHOW_POINTS and (di is None or abs(di) < SHOW_IMPLIED):
        return ""
    when = datetime.fromisoformat(a["at"]).astimezone(__import__("zoneinfo").ZoneInfo("America/New_York"))
    since = f"since {when:%a %-I:%M %p} ET"
    who = "the books now expect their opponent to score" if position == "DST" else "the books now expect his team to score"
    why = "" if di is None or abs(di) < SHOW_IMPLIED else f": {who} {b['imp']:.1f} (was {a['imp']:.1f})"
    return f"projection {a['exp']:.1f} to {b['exp']:.1f} {since}{why}"
