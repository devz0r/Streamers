"""Work out which NFL week the weekly job should act on.

Prints ``completed_week`` (the most recent week whose games are all in the
past) and ``upcoming_week`` (the next one to project) as ``key=value`` lines
for ``$GITHUB_OUTPUT``. ``STREAMER_WEEK_OVERRIDE`` forces the upcoming week.

The week is decided by the calendar, not by whether final scores have landed.
nflverse can take hours to publish Monday night's result, and a score-based
rule reads that lag as "week not finished yet" -- which is how the published
page once stayed on Week 1 after Week 1 had been played. Scores are still
consulted, but only to move the completed week *forward*, never to hold it
back.

``completed_week=0`` means no week has finished yet (preseason); callers
should skip scoring rather than pretend week 1 is in the books.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streamer.config import get_config  # noqa: E402
from streamer.data.nflverse import games_frame  # noqa: E402


def detect(games: pd.DataFrame, season: int, max_week: int, today=None) -> tuple[int, int]:
    """``(completed_week, upcoming_week)`` for ``season`` as of ``today``."""
    today = today or datetime.now(UTC).date()
    rows = games[games["season"] == season]
    if rows.empty:
        return 0, 1

    last_kickoff = (
        rows.assign(_d=pd.to_datetime(rows["gameday"], errors="coerce"))
        .groupby("week")["_d"]
        .max()
    )
    finished = [int(w) for w, d in last_kickoff.items()
                if pd.notna(d) and d.date() < today]
    completed = max(finished) if finished else 0

    # A week whose last game has not kicked off yet is the one to project.
    ahead = [int(w) for w, d in last_kickoff.items()
             if pd.isna(d) or d.date() >= today]
    upcoming = min(ahead) if ahead else max_week

    # If scores say a later week is already done (a feed ahead of the
    # calendar, or a rescheduled game), trust that.
    scored = rows[rows["team_score"].notna()]
    if not scored.empty:
        by_week = rows.groupby("week")["team_score"].apply(lambda s: s.notna().all())
        all_done = [int(w) for w, done in by_week.items() if bool(done)]
        if all_done:
            completed = max(completed, max(all_done))
            upcoming = max(upcoming, min(max(all_done) + 1, max_week))

    return max(completed, 0), min(max(upcoming, 1), max_week)


def main() -> int:
    cfg = get_config()
    season = cfg.current_season
    max_week = int(cfg.season["max_regular_season_week"])

    completed, upcoming = detect(games_frame(cfg), season, max_week)

    override = os.environ.get("STREAMER_WEEK_OVERRIDE", "").strip()
    if override:
        upcoming = int(override)
        completed = max(0, upcoming - 1)

    print(f"completed_week={completed}")
    print(f"upcoming_week={upcoming}")
    print(f"season={season}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
