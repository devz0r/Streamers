"""A week that is partly played must still project the teams yet to play.

The Thursday-night regression: once any game in a week had play-by-play, the
whole week was treated as played, so no placeholder rows were built for the
teams still to kick off and the slate collapsed to the Thursday game. It held
from Thursday evening to the following Tuesday -- the exact window the
rankings are for.
"""

from __future__ import annotations

import pandas as pd

from streamer.pipeline import completed_game_ids, unplayed_games


def _week(week: int, games: list[tuple[str, str]], played: set[str]) -> pd.DataFrame:
    """Two rows per game, one per team, scored only if the game is played."""
    rows = []
    for home, away in games:
        gid = f"2026_{week:02d}_{away}_{home}"
        for team, opp in ((home, away), (away, home)):
            rows.append({
                "season": 2026, "week": week, "game_id": gid,
                "team": team, "opponent": opp,
                "team_score": 21.0 if gid in played else None,
            })
    return pd.DataFrame(rows)


SLATE = [("DET", "BUF"), ("KC", "LAC"), ("PHI", "DAL"), ("SF", "SEA")]


def test_thursday_game_does_not_hide_the_rest_of_the_week():
    thursday = "2026_02_BUF_DET"
    games = _week(2, SLATE, played={thursday})
    completed = completed_game_ids(games)
    assert completed == {thursday}

    future = unplayed_games(games, completed)
    # Six teams still to play; BUF and DET are done and must not be duplicated.
    assert len(future) == 6
    assert set(future["team"]) == {"KC", "LAC", "PHI", "DAL", "SF", "SEA"}
    assert "BUF" not in set(future["team"]) and "DET" not in set(future["team"])


def test_before_any_kickoff_the_whole_week_is_future():
    games = _week(2, SLATE, played=set())
    future = unplayed_games(games, completed_game_ids(games))
    assert len(future) == 8
    assert set(future["team"]) == {t for g in SLATE for t in g}


def test_after_the_week_finishes_nothing_is_future():
    all_ids = {f"2026_02_{a}_{h}" for h, a in SLATE}
    games = _week(2, SLATE, played=all_ids)
    assert unplayed_games(games, completed_game_ids(games)).empty


def test_a_game_in_progress_counts_as_unplayed():
    """No final score means the counts are partial and must not be trusted."""
    games = _week(2, SLATE, played={"2026_02_BUF_DET"})
    # SF/SEA is mid-game: still no score, so still future.
    future = unplayed_games(games, completed_game_ids(games))
    assert {"SF", "SEA"} <= set(future["team"])


def test_empty_and_missing_columns_are_handled():
    assert completed_game_ids(pd.DataFrame()) == set()
    assert completed_game_ids(pd.DataFrame({"game_id": ["a"]})) == set()
    empty = pd.DataFrame(columns=["season", "week", "game_id", "team", "opponent"])
    assert unplayed_games(empty, set()).empty
