"""Rest-of-season simulation: playoffs and title odds."""

from __future__ import annotations

import pytest

from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow
from streamer.roster import season as season_mod
from streamer.roster.season import SeasonModel, bracket_order


def test_bracket_order_is_the_standard_fixed_bracket():
    order = bracket_order(8)
    pairs = [tuple(sorted(order[i:i + 2])) for i in range(0, 8, 2)]
    assert pairs == [(1, 8), (4, 5), (2, 7), (3, 6)]          # 1 meets the 4/5 winner
    assert set(order[:4]) == {1, 8, 4, 5}
    assert bracket_order(4) == [1, 4, 2, 3]


def _team(tid: str, strength: float, mine: bool = False, wins: int = 0) -> TeamRow:
    roster = []
    for pos, n in (("QB", 1), ("RB", 3), ("WR", 3), ("TE", 1)):
        for k in range(n):
            pid = f"{tid}-{pos}{k}"
            base = strength * {"QB": 18, "RB": 12, "WR": 12, "TE": 8}[pos] * (1 - 0.15 * k)
            roster.append(PlayerRow(player_id=pid, name=pid, position=pos, team="KC", projection=base,
                                    projection_sd=6.0, outcome_sd=6.0, ros_value=base,
                                    eligible_slots=[pos, "FLEX"] if pos != "QB" else ["QB"]))
    return TeamRow(team_id=tid, name=f"Team {tid}", roster=roster, is_mine=mine, wins=wins, losses=2 - wins)


def _league(strengths, median=False) -> LeagueSnapshot:
    teams = [_team(str(i), s, mine=(i == 0)) for i, s in enumerate(strengths)]
    ids = [t.team_id for t in teams]
    schedule = []
    for w in range(3, 11):                     # round robin, rotating
        rot = ids[:1] + ids[1:][w % (len(ids) - 1):] + ids[1:][:w % (len(ids) - 1)]
        for i in range(len(rot) // 2):
            schedule.append([w, rot[i], rot[-1 - i]])
    return LeagueSnapshot(
        platform="espn", profile="espn", league_id="1", league_name="L", season=2026, week=3,
        slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}, bench_size=2, teams=teams,
        free_agents=[], matchup=None, synced_at="",
        rules={"regular_season_weeks": 10, "playoff_teams": 4, "playoff_round_weeks": 1,
               "median_game": median, "schedule": schedule})


@pytest.fixture(autouse=True)
def _no_byes(monkeypatch):
    monkeypatch.setattr(season_mod, "nfl_byes", lambda *a, **k: {})


def test_odds_add_up_and_the_strongest_team_is_the_favourite(cfg):
    snap = _league([1.0, 1.3, 0.9, 0.8, 1.0, 0.95])
    odds = SeasonModel(snap, cfg, n_sims=1500).odds()
    assert odds.p_title.sum() == pytest.approx(1.0)
    assert odds.p_playoffs.sum() == pytest.approx(4.0)
    assert odds.p_title.argmax() == 1
    assert odds.p_title[1] > 2 * odds.p_title[3]


def test_a_better_roster_raises_your_title_odds_on_the_same_futures(cfg):
    snap = _league([1.0, 1.1, 1.0, 1.0])
    model = SeasonModel(snap, cfg, n_sims=1500)
    base = model.odds().p_title[0]
    me = snap.teams[0]
    better = [p.player_id for p in me.roster if p.position != "WR"] + \
             [p.player_id for p in snap.teams[1].roster if p.position == "WR"]
    boosted = model.odds(override={"0": model.team_scores(better)}).p_title[0]
    assert boosted > base


def test_espn_tiebreak_key_in_rules(cfg):
    from types import SimpleNamespace as NS

    from streamer.league import espn

    league = NS(teams=[], settings=NS(reg_season_count=14, playoff_team_count=6,
                                      playoff_matchup_period_length=1,
                                      playoff_seed_tie_rule="H2H_RECORD", median_scoring=False,
                                      faab=False, acquisition_budget=0))
    assert espn.league_rules(league)["tiebreak"] == "h2h"


def test_reseeding_pairs_the_best_seed_left_with_the_worst(cfg):
    import numpy as np

    snap = _league([1.0, 1.1, 1.0, 1.0, 0.9, 0.95])
    snap.rules.update(playoff_teams=6, reseed=True)
    model = SeasonModel(snap, cfg, n_sims=4)
    # After round one of a six-team bracket: seeds 1 and 2 had byes; in sim 0
    # seeds 4 and 3 won, in sim 1 seeds 5 and 6 did (team index = seed * 10).
    alive = [np.array([10, 10, 10, 10]), np.array([40, 50, 40, 50]),
             np.array([30, 60, 60, 30]), np.array([20, 20, 20, 20])]
    seeds = [1, np.array([4, 5, 4, 5]), np.array([3, 6, 6, 3]), 2]
    teams, s = model._reseeded(alive, seeds)
    pairs = [sorted((int(s[i][k]), int(s[i + 1][k]))) for k in range(4) for i in (0, 2)]
    assert pairs[:2] == [[1, 4], [2, 3]]          # 1 meets the worst seed left
    assert pairs[2:4] == [[1, 6], [2, 5]]
    assert all(int(t[k]) == int(sd[k]) * 10 for t, sd in zip(teams, s) for k in range(4))
    odds = model.odds()
    assert odds.p_title.sum() == pytest.approx(1.0) and odds.p_playoffs.sum() == pytest.approx(6.0)


def test_a_mid_week_pickup_cannot_use_points_already_scored(cfg):
    import numpy as np

    snap = _league([1.0, 1.1, 1.0, 1.0])
    star = snap.teams[1].roster[1]                 # someone else's back, game already played
    star.locked, star.actual_points = True, 40.0
    mine = snap.teams[0]
    mine.roster[1].locked, mine.roster[1].actual_points = True, 3.0   # my back, played, scored 3
    model = SeasonModel(snap, cfg, n_sims=500)
    swapped = [p.player_id for p in mine.roster if p is not mine.roster[1]] + [star.player_id]
    naive = model.team_scores(swapped)
    settled = model.team_scores(swapped, owner=mine.team_id)
    base = model.team_scores([p.player_id for p in mine.roster], owner=mine.team_id)
    # This week: my own finished score stands, the newcomer's 40 does not count.
    assert np.allclose(settled[:, 0], base[:, 0])
    assert naive[:, 0].mean() > settled[:, 0].mean() + 20
    # From next week he plays for me.
    assert np.allclose(settled[:, 1:], naive[:, 1:])
