"""What the injury-timeline reader saw, for a few players.

    python scripts/timeline_probe.py [Name ...]

For each named player (default: the ones in the timeline log that look odd,
plus Davante Adams) prints his status, and the last few items ESPN's player
news carries -- date, type, headline -- with what the reader made of each
(tag, timeline, and the sentence it read). That is public news text, the same
as the page's breaking-news list. Run as the ``timeline-probe`` job.
"""

from __future__ import annotations

import glob
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT = ["Davante Adams", "Jalen Hurts", "Austin Ekeler", "Terry McLaurin", "Baker Mayfield", "Josh Jacobs",
           "Romeo Doubs", "Jordan Mason", "Tank Bigsby", "Jadarian Price", "Alec Pierce"]


def main() -> int:
    from streamer.config import get_config
    from streamer.data import news
    from streamer.league.model import LeagueSnapshot
    from streamer.roster.players import normalize_name

    cfg = get_config().for_profile("espn")
    f = sorted(glob.glob(str(cfg.root / "data/leagues/espn/week_*.json")), key=lambda x: int(x.split("_")[-1][:-5]))[-1]
    snap = LeagueSnapshot.load(f)
    by = {normalize_name(p.name): p for p in snap.all_players()}
    now = datetime.now(UTC)
    for name in sys.argv[1:] or DEFAULT:
        p = by.get(normalize_name(name))
        print(f"== {name}: " + (f"status {p.status!r} slot {p.slot!r} team {p.team}" if p else "not in the ESPN snapshot"))
        if not p:
            continue
        items = news.player_items(cfg, p.player_id, True)
        print(f"   {len(items)} items")
        for it in items[:6]:
            when = news._when(it)
            head = str(it.get("headline") or "")
            text = f"{head} {it.get('story') or it.get('description') or ''}"
            t = news.timeline(text)
            hit = ""
            if t is not None:
                for sent in re.split(r"(?<=[.!?])\s+", text):
                    if news.timeline(sent) is not None:
                        hit = sent[:230]
                        break
            age = f"{(now - when).days}d" if when else "?"
            print(f"   [{age} {it.get('type')}] tag={news.tag_of(head)!r} timeline={t} :: {head[:140]}")
            if hit:
                print(f"        read: {hit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
