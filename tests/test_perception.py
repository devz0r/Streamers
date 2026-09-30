"""How the rest of the league sees a player, and whether he says yes."""

from __future__ import annotations

import pandas as pd

from streamer.league.model import PlayerRow
from streamer.roster import perception


def _p(pid, v, plat=None, status="", slot="BN"):
    return PlayerRow(player_id=pid, name=pid, position="WR", team="KC", projection=v, ros_value=v,
                     platform_projection=plat, status=status, slot=slot, nfl_id=pid)


def _history(rows):
    return pd.DataFrame([{"player_id": pid, "season": s, "week": w, "fantasy_points_ppr": pts,
                          "position": "WR"} for pid, s, n, pts in rows for w in range(1, n + 1)])


def test_name_value_lifts_a_player_the_same_projection_otherwise():
    lines = perception.season_lines(_history([("star", 2025, 16, 20.0)]), 2026)
    seen = perception.perceive([_p("star", 12.0), _p("nobody", 12.0)], lines, 2026)
    assert seen["star"].value > seen["nobody"].value + 2
    assert seen["star"].last_ppg == 20.0 and seen["star"].last_games == 16


def test_the_platform_projection_is_what_he_sees_unless_it_says_nothing():
    seen = perception.perceive([_p("a", 10.0, plat=15.0), _p("b", 10.0, plat=0.0)], {}, 2026)
    assert seen["a"].value == 15.0 and seen["b"].value == 10.0     # 0 = bye or out, not a verdict


def test_ir_and_a_short_last_season_read_lower():
    lines = perception.season_lines(_history([("hurt", 2025, 6, 12.0), ("fit", 2025, 16, 12.0)]), 2026)
    seen = perception.perceive([_p("hurt", 12.0), _p("fit", 12.0), _p("ir", 12.0, status="IR")], lines, 2026)
    assert seen["hurt"].value < seen["fit"].value
    assert seen["ir"].value < 12.0


def test_acceptance_rises_with_his_gain_and_the_best_player_and_his_activity():
    assert perception.p_accept(2.0, 0, 1.0) > perception.p_accept(0.0, 0, 1.0) > perception.p_accept(-2.0, 0, 1.0)
    assert perception.p_accept(0.5, 1, 1.0) > perception.p_accept(0.5, 0, 1.0) > perception.p_accept(0.5, -1, 1.0)
    assert perception.p_accept(1.0, 0, 1.0) == 2 * perception.p_accept(1.0, 0, 0.5)
    # No offer is a sure thing, however good it looks to him.
    assert perception.p_accept(20.0, 1, 1.0) <= perception.ACCEPT_CEILING < 1.0
    assert perception.engagement(0, 2) == 0.6 and perception.engagement(4, 2) == 1.0
    assert perception.engagement(None, 2) == 0.8
