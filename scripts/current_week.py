"""Work out which NFL week the weekly job should act on.

Prints ``completed_week`` (the most recent week with final scores) and
``upcoming_week`` (the next one to project) as ``key=value`` lines for
``$GITHUB_OUTPUT``. ``STREAMER_WEEK_OVERRIDE`` forces the upcoming week.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streamer.config import get_config  # noqa: E402
from streamer.data.nflverse import games_frame  # noqa: E402


def main() -> int:
    cfg = get_config()
    season = cfg.current_season
    max_week = int(cfg.season["max_regular_season_week"])

    games = games_frame(cfg)
    this_season = games[games["season"] == season]
    played = this_season[this_season["team_score"].notna()]
    completed = int(played["week"].max()) if not played.empty else 0
    upcoming = min(max(completed + 1, 1), max_week)

    override = os.environ.get("STREAMER_WEEK_OVERRIDE", "").strip()
    if override:
        upcoming = int(override)
        completed = max(1, upcoming - 1)

    print(f"completed_week={max(1, completed)}")
    print(f"upcoming_week={upcoming}")
    print(f"season={season}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
