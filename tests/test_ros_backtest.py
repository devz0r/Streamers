"""The rest-of-season replay: statuses as the platforms would show them,
outcomes as holding the player returned them, and the fitted IR model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from streamer.league.model import PlayerRow
from streamer.roster import ros_backtest as rb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_statuses_read_like_a_platform():
    rost = pd.DataFrame({"gsis_id": ["a", "b", "c", "d", "e"], "week": [5] * 5, "team": ["KC"] * 5,
                         "position": ["RB"] * 5, "status": ["ACT", "RES", "INA", "DEV", "ACT"]})
    inj = pd.DataFrame({"gsis_id": ["c", "e"], "week": [5, 5], "report_status": ["Out", "Questionable"]})
    st = rb.statuses(rost, inj, 5)
    assert st == {"a": ("KC", ""), "b": ("KC", "IR"), "c": ("KC", "OUT"), "e": ("KC", "Q")}   # practice squad left out


def test_an_absence_scores_zero_and_a_bye_does_not_count():
    hist = pd.DataFrame({"player_id": ["p"] * 3, "season": [2024] * 3, "week": [5, 7, 8],
                         "fantasy_points_ppr": [10.0, 20.0, 30.0]})
    p = PlayerRow(player_id="p", name="p", position="WR", team="KC")
    r = rb.realized(hist, [p], 2024, [5, 6, 7, 8, 9], {"KC": {6}}).loc["p"]
    assert r["team_games"] == 4 and r["games"] == 3                # week 6 a bye, week 9 missed
    assert r["per_team_game"] == 15.0 and r["per_game"] == 20.0


def test_the_ir_fit_tables_never_coming_back_and_the_wait():
    import fit_ros

    r = pd.DataFrame({"level": [3, 3, 3, 15, 15, 15], "missed": [0, 0, 7, 0, 1, 2],
                      "back": [np.nan, np.nan, np.nan, 2, 3, np.nan]})
    ir = fit_ros.fit_ir(r)
    assert ir["gone"][0][0] > ir["gone"][2][0]                     # low players stay gone more often
    assert abs(sum(ir["wait"]) - 1.0) < 1e-3 and ir["wait"][0] == 0.0
    assert ir["evidence"]["cases"] == 6


def test_early_form_and_a_later_read_are_graded_against_what_followed():
    hist = pd.DataFrame({"player_id": ["p"] * 10, "season": [2024] * 10, "week": list(range(5, 15)),
                         "fantasy_points_ppr": [20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 10.0, 12.0, 12.0]})
    p = PlayerRow(player_id="p", name="p", position="WR", team="KC")
    r = rb.realized(hist, [p], 2024, list(range(5, 15)), {}, move_week=9).loc["p"]
    assert r["early_pg"] == 20.0 and r["late_pg"] == 64.0 / 6
    assert r["late_move_pg"] == 64.0 / 6
    assert rb.move_week_for(4, [3, 4, 5, 6, 8, 10]) == 8 and rb.move_week_for(10, [3, 4, 10]) is None


def test_the_share_of_a_move_that_held_up_is_pooled_over_simulated_seasons():
    rng = np.random.default_rng(3)
    n = 400
    move = rng.normal(0, 2, n)
    d = pd.DataFrame({"ros_value": 10.0, "ros_next": 10.0 + move,
                      "late_move_real": 10.0 + 0.5 * move + rng.normal(0, 1, n),
                      # simulated: each player's seasons see a move that holds up fully
                      "mv_mean": 10.0 + move, "ml_mean": 10.0 + move, "mv_var": 1.0, "ml_cov": 1.0})
    m = rb.moves(d)
    assert abs(m["real_slope"] - 0.5) < 0.1 and abs(m["sim_slope"] - 1.0) < 0.02
    assert abs(m["sim_sd"] - np.sqrt(4.0 + 1.0)) < 0.2
