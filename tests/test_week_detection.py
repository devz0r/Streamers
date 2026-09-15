"""Which NFL week the weekly job acts on.

The published page once stayed on Week 1 all the way through Week 1 being
played, because the detector asked "have final scores landed?" rather than
"has the week happened?". These pin the calendar behaviour.
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("current_week", ROOT / "scripts" / "current_week.py")
current_week = importlib.util.module_from_spec(spec)
spec.loader.exec_module(current_week)
detect = current_week.detect


#: Week 1 plays Sunday 13 and Monday 14 September, as the 2026 season does.
WEEK1_SUNDAY = date(2026, 9, 13)


def _season(weeks: int = 4, scored_through: int = 0) -> pd.DataFrame:
    """Two games a week, Sunday then Monday, a week apart."""
    rows = []
    for week in range(1, weeks + 1):
        sunday = WEEK1_SUNDAY + pd.Timedelta(days=7 * (week - 1))
        for offset, teams in ((0, ("KC", "BUF")), (1, ("SF", "DAL"))):
            day = sunday + pd.Timedelta(days=offset)
            score = 20.0 if week <= scored_through else None
            for team in teams:
                rows.append({"season": 2026, "week": week, "gameday": str(day),
                             "team": team, "team_score": score})
    return pd.DataFrame(rows)


def test_week_rolls_over_once_the_week_is_in_the_past():
    games = _season(scored_through=1)
    # The Tuesday after week 1's Monday night game.
    assert detect(games, 2026, 18, today=date(2026, 9, 15)) == (1, 2)


def test_monday_night_scores_not_posted_yet_still_rolls_over():
    """The regression: nflverse lagging on Monday's result must not pin the week."""
    games = _season(scored_through=0)          # no final scores anywhere
    assert detect(games, 2026, 18, today=date(2026, 9, 15)) == (1, 2)


def test_midweek_stays_on_the_upcoming_week():
    games = _season(scored_through=1)
    # Thursday of week 2: week 2 has not finished, so it is still what we project.
    assert detect(games, 2026, 18, today=date(2026, 9, 17)) == (1, 2)


def test_before_any_game_nothing_is_completed():
    games = _season()
    completed, upcoming = detect(games, 2026, 18, today=date(2026, 9, 1))
    assert completed == 0 and upcoming == 1


def test_scores_can_move_the_week_forward_but_never_back():
    """A feed ahead of the calendar is trusted; one behind it is not."""
    games = _season(scored_through=3)
    completed, upcoming = detect(games, 2026, 18, today=date(2026, 9, 15))
    assert completed == 3 and upcoming == 4


def test_upcoming_never_exceeds_the_regular_season():
    games = _season(weeks=2, scored_through=2)
    completed, upcoming = detect(games, 2026, 2, today=date(2026, 10, 5))
    assert completed == 2 and upcoming == 2


def test_empty_season_is_handled():
    assert detect(pd.DataFrame({"season": [], "week": [], "gameday": [],
                                "team_score": []}), 2026, 18) == (0, 1)


@pytest.mark.parametrize("today,expected", [
    (date(2026, 9, 8), (0, 1)),    # the Tuesday before week 1 kicks off
    (date(2026, 9, 14), (0, 1)),   # week 1's Monday: the night game is still to come
    (date(2026, 9, 15), (1, 2)),   # the morning after -- this is where it used to stick
])
def test_rollover_boundary(today, expected):
    assert detect(_season(scored_through=0), 2026, 18, today=today) == expected
