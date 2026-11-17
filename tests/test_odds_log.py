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


def test_odds_are_logged_and_graded_against_the_final_field(tmp_cfg):
    cfg = tmp_cfg
    odds_log.record(cfg, _snap(2, [1, 1, 1, 1]), _odds([0.9, 0.6, 0.3, 0.2]))
    odds_log.record(cfg, _snap(2, [1, 1, 1, 1]), _odds([0.9, 0.6, 0.3, 0.2]))      # same week: replaced
    assert len(__import__("pandas").read_parquet(cfg.results_dir / odds_log.LOG)) == 4
    assert odds_log.grade(cfg, _snap(4, [3, 3, 1, 1])) is None                    # season not over
    g = odds_log.grade(cfg, _snap(5, [3, 3, 1, 1]))
    assert g["by_week"][0]["week"] == 2
    assert g["by_week"][0]["brier"] == np.mean([(0.9 - 1) ** 2, (0.6 - 1) ** 2, 0.3 ** 2, 0.2 ** 2])
    assert sum(b["n"] for b in g["buckets"]) == 4


def test_expected_points_are_logged_and_graded_against_what_teams_scored(tmp_cfg):
    cfg = tmp_cfg
    o = _odds([0.9, 0.6, 0.3, 0.2])
    o.exp_points = np.array([130.0, 120.0, 110.0, 100.0])         # the rest of the regular season
    odds_log.record(cfg, _snap(2, [1, 1, 1, 1]), o)
    logged = __import__("pandas").read_parquet(cfg.results_dir / odds_log.LOG)
    assert list(logged["exp_points"]) == [130.0, 120.0, 110.0, 100.0]
    assert list(logged["points_for"]) == [100.0, 101.0, 102.0, 103.0]
    end = _snap(5, [3, 3, 1, 1])
    # Each team then scored half the model's spread above its average.
    for t, extra in zip(end.teams, [120.0, 115.0, 110.0, 105.0]):
        t.points_for += extra
    g = odds_log.grade(cfg, end)
    assert g["by_week"][0]["strength_slope"] == __import__("pytest").approx(0.5)


def test_each_weeks_scores_and_each_games_chance_are_logged_from_the_simulation(tmp_cfg):
    import pandas as pd

    class M:
        weeks = [3, 4]
        schedule = [(3, "0", "1"), (4, "0", "2"), (4, "9", "1")]            # the last names a team not in the league
        team_index = {"0": 0, "1": 1, "2": 2}
        teams = [TeamRow(team_id=str(i), name=f"T{i}", is_mine=(i == 0)) for i in range(3)]

        def _scores(self, override):
            rng = np.random.default_rng(0)
            s = rng.normal(100, 10, (3, 4000, 2))
            s[0] += 20                                                          # team 0 is the strong one
            return s

    snap = _snap(3, [1, 1, 1])
    assert odds_log.record_weeks(tmp_cfg, snap, M()) == (6, 2)
    assert odds_log.record_weeks(tmp_cfg, snap, M()) == (6, 2)                  # same week: replaced, not doubled
    t = pd.read_parquet(tmp_cfg.results_dir / odds_log.WEEKS_LOG)
    g = pd.read_parquet(tmp_cfg.results_dir / odds_log.GAMES_LOG)
    assert len(t) == 6 and len(g) == 2
    assert abs(t[(t.team_id == "0") & (t.for_week == 3)]["mean"].iloc[0] - 120) < 1 and t[t.team_id == "0"].mine.all()
    strong = g[(g.a == "0") & (g.b == "1")].iloc[0]
    assert 0.88 < strong.p_a_wins < 0.95 and abs(strong.mean_margin - 20) < 1.5 and strong.for_week == 3
