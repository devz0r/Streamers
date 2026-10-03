"""Who is next in line (the real depth chart), and a back whose work has collapsed."""

from __future__ import annotations

import json

from streamer.league.model import PlayerRow
from streamer.roster.projections import _collapse_factor, assign_roles, attach_depth, next_in_line
from streamer.roster.season import heirs_for

RULE = {"usage_collapse": {"positions": ["RB"], "below": 0.5, "min_value": 4.0, "factor": 0.86}}


def _rb(name, ros, order=None, team="NYG", pos="RB", status=""):
    p = PlayerRow(player_id=name, name=name, position=pos, team=team, ros_value=ros, projection=ros, status=status)
    p.depth_order = order
    return p


def test_a_collapsed_back_keeps_the_measured_share():
    tracy = _rb("Tyrone Tracy Jr.", 7.6)
    row = {"recent_exp": 0.37, "prior_exp": 10.44}
    assert _collapse_factor(row, tracy, 7.6, RULE) == 0.86
    assert _collapse_factor(row, _rb("WR guy", 7.6, pos="WR"), 7.6, RULE) == 1.0      # receivers showed no lean
    assert _collapse_factor(row, _rb("Deep back", 3.0), 3.0, RULE) == 1.0             # below the value floor
    assert _collapse_factor(row, _rb("Hurt back", 7.6, status="OUT"), 7.6, RULE) == 1.0  # an injury is simulated
    assert _collapse_factor({"recent_exp": 6.0, "prior_exp": 10.0}, tracy, 7.6, RULE) == 1.0   # kept 60%


def test_the_depth_chart_decides_who_is_next_behind_the_lead():
    """The Giants, week 4: Tracy's 2025 role made him look like the handcuff;
    the depth chart has Najee Harris there."""
    room = [_rb("Cam Skattebo", 14.2, 1), _rb("Devin Singletary", 7.8), _rb("Tyrone Tracy Jr.", 7.6, 3),
            _rb("Najee Harris", 6.4, 2)]
    assert [p.name for p in next_in_line(room)] == ["Cam Skattebo", "Najee Harris", "Tyrone Tracy Jr.",
                                                    "Devin Singletary"]
    assert heirs_for(room) == [("Cam Skattebo", "Najee Harris")]
    assign_roles(room)
    assert {p.name: p.role for p in room}["Najee Harris"] == "RB2"
    # The lead stays the lead by value even when Sleeper lists him lower (an injured starter).
    jets = [_rb("Breece Hall", 13.4, 2, "NYJ"), _rb("Braelon Allen", 6.2, 1, "NYJ"), _rb("Isaiah Davis", 3.0, 3, "NYJ")]
    assert heirs_for(jets) == [("Breece Hall", "Braelon Allen")]
    # No depth chart: by value, as before.
    for p in room:
        p.depth_order = None
    assert heirs_for(room) == [("Cam Skattebo", "Devin Singletary")]


def test_sleeper_depth_charts_are_read_and_matched_by_name_team_and_position(tmp_cfg):
    from streamer.data import sleeper

    (tmp_cfg.raw_dir / "sleeper_players.json").write_text(json.dumps({
        "1": {"full_name": "Najee Harris", "team": "NYG", "position": "RB", "depth_chart_position": "RB",
              "depth_chart_order": 2},
        "2": {"full_name": "Tyrone Tracy", "team": "NYG", "position": "RB", "depth_chart_position": "RB",
              "depth_chart_order": 3},
        "3": {"full_name": "Free Agent", "team": None, "position": "RB", "depth_chart_order": 1},
        "4": {"full_name": "Some Receiver", "team": "NYG", "position": "WR", "depth_chart_order": 1},
    }))
    orders = sleeper.depth_orders(tmp_cfg, allow_network=False)
    assert len(orders) == 2
    room = [_rb("Najee Harris", 6.4), _rb("Tyrone Tracy Jr.", 7.6), _rb("Najee Harris", 6.4, team="LAC")]
    assert attach_depth(room, orders) == 2
    assert [p.depth_order for p in room] == [2, 3, None]     # "Jr." still matches; another team does not


def test_a_backup_already_standing_in_stays_next_in_line_over_the_depth_chart():
    """Miami, week 4: Achane on IR, Gordon carrying the load, and Sleeper
    listing Wright first. The one getting the work keeps the job."""
    achane, gordon, wright = _rb("De'Von Achane", 15.0, 4, "MIA"), _rb("Ollie Gordon II", 9.0, 2, "MIA"), \
        _rb("Jaylen Wright", 5.0, 1, "MIA")
    gordon.inherited_ros = 2.0
    assert heirs_for([achane, gordon, wright]) == [("De'Von Achane", "Ollie Gordon II")]
