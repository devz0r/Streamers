"""The season-odds log and its grade."""

from __future__ import annotations

import numpy as np

from streamer.league.model import LeagueSnapshot, TeamRow
from streamer.roster import odds_log
from streamer.roster.season import SeasonOdds


def _odds(p_po):
    n = len(p_po)
    return SeasonOdds(team_ids=[str(i) for i in range(n)], names=[f"T{i}" for i in range(n)], records=[""] * n,
                      p_playoffs=np.array(p_po), p_bye=np.zeros(n), p_title=np.array(p_po) / 2,
                      exp_wins=np.array([8.0] * n), n_sims=10, mine=0)


def _snap(week, wins):
    teams = [TeamRow(team_id=str(i), name=f"T{i}", wins=w, losses=4 - w, points_for=100.0 + i)
             for i, w in enumerate(wins)]
    return LeagueSnapshot(platform="espn", profile="espn", league_id="1", league_name="L", season=2026, week=week,
                          slots={}, bench_size=0, teams=teams, free_agents=[], matchup=None, synced_at="",
                          rules={"regular_season_weeks": 4, "playoff_teams": 2})


def test_odds_are_logged_and_graded_against_the_final_field(cfg):
    odds_log.record(cfg, _snap(2, [1, 1, 1, 1]), _odds([0.9, 0.6, 0.3, 0.2]))
    odds_log.record(cfg, _snap(2, [1, 1, 1, 1]), _odds([0.9, 0.6, 0.3, 0.2]))      # same week: replaced
    assert len(__import__("pandas").read_parquet(cfg.results_dir / odds_log.LOG)) == 4
    assert odds_log.grade(cfg, _snap(4, [3, 3, 1, 1])) is None                    # season not over
    g = odds_log.grade(cfg, _snap(5, [3, 3, 1, 1]))
    assert g["by_week"][0]["week"] == 2
    assert g["by_week"][0]["brier"] == np.mean([(0.9 - 1) ** 2, (0.6 - 1) ** 2, 0.3 ** 2, 0.2 ** 2])
    assert sum(b["n"] for b in g["buckets"]) == 4
