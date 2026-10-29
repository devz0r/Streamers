"""A league's position limits: read from ESPN, and kept by every pickup and trade."""

from __future__ import annotations

from types import SimpleNamespace as NS

from streamer.league.espn import position_limits as espn_limits
from streamer.league.model import PlayerRow, over_limit, position_limits
from streamer.roster.title_moves import _droppable, drop_candidates


def _p(pid, pos, ros=5.0, ir=False):
    p = PlayerRow(player_id=pid, name=pid, position=pos, team="KC", ros_value=ros, projection=ros)
    if ir:
        p.slot = "IR"
    return p


def test_espn_limits_are_read_from_its_roster_settings():
    raw = {"settings": {"rosterSettings": {"positionLimits": {"0": 4, "2": 6, "4": 0, "6": 3, "16": 2, "17": 0}}}}
    league = NS(espn_request=NS(league_get=lambda params=None: raw))
    assert espn_limits(league) == {"QB": 4, "RB": 6, "TE": 3, "DST": 2}       # 0 is no limit
    broken = NS(espn_request=NS(league_get=lambda params=None: (_ for _ in ()).throw(RuntimeError("down"))))
    assert espn_limits(broken) == {}


def test_a_full_position_can_only_make_room_at_that_position():
    """Six backs with a six-back limit: adding a seventh has to drop a back,
    not a receiver -- ESPN would refuse the move otherwise."""
    roster = [_p(f"rb{i}", "RB", 6.0 + i) for i in range(6)] + [_p("wr0", "WR", 1.0), _p("wr1", "WR", 8.0),
                                                                 _p("wr2", "WR", 9.0), _p("wr3", "WR", 9.5)]
    vaki = _p("vaki", "RB", 3.3)
    free = _droppable(roster, vaki)
    assert free[0].player_id == "wr0"                                  # no limit: the cheapest anyone
    capped = _droppable(roster, vaki, limits={"RB": 6})
    assert capped and all(p.position == "RB" for p in capped)
    assert all(p.position == "RB" for p in drop_candidates(roster, vaki, 3, limits={"RB": 6}))
    # A back in an IR slot does not count toward the limit.
    roster[0].slot = "IR"
    assert _droppable(roster, vaki, limits={"RB": 6})[0].player_id == "wr0"
    assert not over_limit(roster + [vaki], {"RB": 6})
    roster[0].slot = "BN"
    assert over_limit(roster + [vaki], {"RB": 6}) and not over_limit(roster + [vaki], {})


def test_limits_come_from_the_snapshot_rules():
    assert position_limits(NS(rules={"position_limits": {"RB": 6}})) == {"RB": 6}
    assert position_limits(NS(rules=None)) == {} and position_limits(None) == {}
