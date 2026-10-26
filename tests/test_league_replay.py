"""Replayed leagues: the draft, waivers, lineups and the real bracket."""

from __future__ import annotations

import numpy as np
import pytest

import streamer.roster.season as season_mod
from streamer.league.model import PlayerRow
from streamer.roster import league_replay as lr


def _p(pid, pos, ros, team="KC", status="", proj=None):
    return PlayerRow(player_id=pid, name=pid, position=pos, team=team, status=status,
                     projection=ros if proj is None else proj, projection_sd=6.0, outcome_sd=6.0, ros_value=ros)


def _pool(n=200, seed=0):
    rng = np.random.default_rng(seed)
    teams = ["KC", "BUF", "MIA", "NYJ", "DAL", "PHI", "SF", "LA"]
    pos = ["QB"] * 30 + ["RB"] * 60 + ["WR"] * 80 + ["TE"] * 30
    return [_p(f"p{i}", pos[i % len(pos)], float(rng.uniform(2, 22)), teams[i % len(teams)]) for i in range(n)]


def test_a_round_robin_meets_everyone_before_a_rematch():
    games = lr.round_robin(10, 9)
    assert len(games) == 45
    pairs = {frozenset((a, b)) for _w, a, b in games}
    assert len(pairs) == 45
    for w in range(1, 10):
        teams = [t for ww, a, b in games if ww == w for t in (a, b)]
        assert sorted(teams) == list(range(10))


def test_the_draft_fills_every_starting_spot_within_the_caps():
    pool = {p.player_id: p for p in _pool()}
    rosters = lr.draft(list(pool.values()), 10, np.random.default_rng(1))
    assert all(len(r) == sum(lr.SLOTS.values()) + lr.BENCH for r in rosters)
    assert len({pid for r in rosters for pid in r}) == sum(len(r) for r in rosters)
    for r in rosters:
        counts = {pos: sum(pool[pid].position == pos for pid in r) for pos in lr.NEEDS}
        assert all(lr.NEEDS[pos] <= counts[pos] <= lr.CAPS[pos] for pos in lr.NEEDS)


def test_a_lineup_starts_the_best_who_play_and_leaves_the_out_on_the_bench():
    rows = [_p("qb", "QB", 20), _p("qb2", "QB", 15), _p("rb1", "RB", 15, status="OUT", proj=0.0),
            _p("rb2", "RB", 12), _p("rb3", "RB", 8), _p("wr1", "WR", 14), _p("wr2", "WR", 11),
            _p("wr3", "WR", 10), _p("te", "TE", 9)]
    rows[2].on_bye = False
    got = [p.player_id for p in lr.best_lineup(rows)]
    assert got == ["qb", "te", "rb2", "rb3", "wr1", "wr2", "wr3"]
    assert lr._holes(rows) == []
    rows[3].on_bye = True                                     # one back left: a hole at RB
    assert lr._holes(rows) == ["RB"]


def test_a_drop_keeps_the_starting_spots_and_the_caps():
    roster = [_p("qb", "QB", 20), _p("rb1", "RB", 15), _p("rb2", "RB", 12), _p("wr1", "WR", 14),
              _p("wr2", "WR", 11), _p("te", "TE", 1), _p("wr3", "WR", 2)]
    drop = lr._droppable(roster, _p("new", "RB", 9))
    assert drop.player_id == "wr3"                            # the TE is weaker, but the only one
    two_qbs = roster + [_p("qb2", "QB", 0.5)]
    assert lr._droppable(two_qbs, _p("qb3", "QB", 9)).player_id == "qb2"


def test_the_real_bracket_seeds_by_record_and_gives_the_top_two_a_bye():
    wins = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], float)
    points = np.zeros(10)
    scores = np.zeros((10, 3))
    # Round 1: 3 beats 6, 5 upsets 4. Round 2: 5 upsets 1, 2 beats 3. Final: 2.
    scores[2, 0], scores[5, 0], scores[3, 0], scores[4, 0] = 100, 90, 80, 95
    scores[0, 1], scores[4, 1], scores[1, 1], scores[2, 1] = 90, 99, 110, 100
    scores[4, 2], scores[1, 2] = 100, 120
    made, champ = lr.realized_bracket(wins, points, scores)
    assert made == {0, 1, 2, 3, 4, 5}
    assert champ == 1
    # A tie goes to the higher seed.
    tie = np.zeros((10, 3))
    assert lr.realized_bracket(wins, points, tie)[1] == 0


@pytest.fixture
def _quiet_schedule(monkeypatch):
    monkeypatch.setattr(season_mod, "nfl_byes", lambda *a, **k: {})


def test_a_replayed_league_plays_through_and_grades_every_game(cfg, _quiet_schedule):
    pool = _pool(220)
    weekly = {w: pool for w in range(1, 18)}
    rng = np.random.default_rng(3)
    points = {(w, p.player_id): float(max(rng.normal(p.ros_value, 6.0), 0.0)) for w in weekly for p in pool}
    r = lr.play(2024, 7, weekly, points, {}, cfg, n_sims=200, win_sims=500, checkpoints=(1, 7))
    assert len(r.games) == lr.REG_WEEKS * lr.N_TEAMS // 2
    assert {g["week"] for g in r.games} == set(range(1, lr.REG_WEEKS + 1))
    assert all(0.0 <= g["p_win"] <= 1.0 for g in r.games)
    assert len(r.odds) == 2 * lr.N_TEAMS
    first = [o for o in r.odds if o["week"] == 1]
    assert sum(o["p_playoffs"] for o in first) == pytest.approx(lr.PLAYOFF_TEAMS)
    assert sum(o["p_title"] for o in first) == pytest.approx(1.0)
    assert sum(o["base_title"] for o in first) == pytest.approx(1.0)
    assert sum(o["made"] for o in first) == lr.PLAYOFF_TEAMS and sum(o["won_title"] for o in first) == 1
    assert sum(r.finals["wins"]) == pytest.approx(len(r.games))
    # The model's expected points for the rest of the regular season sit
    # beside what each team really scored.
    assert all(o["exp_points"] > 0 and o["rest_points"] > 0 for o in r.odds)
    frozen = lr.play(2024, 7, weekly, points, {}, cfg, n_sims=200, win_sims=500, checkpoints=(), upgrade=None)
    assert len(frozen.games) == len(r.games)


def test_the_odds_check_fails_only_on_a_clearly_worse_score():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import grade_league_odds as g

    base = {"p_win": 0.2312, "p_playoffs": 0.1665, "p_title": 0.0854}
    assert g.check(dict(base), base) == []
    assert g.check({**base, "p_title": 0.0854 + 0.003}, base) == []
    assert len(g.check({**base, "p_playoffs": 0.1665 + 0.01}, base)) == 1
