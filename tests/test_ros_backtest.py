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
