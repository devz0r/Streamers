"""A league's position limits: read from ESPN, and kept by every pickup and trade."""

from __future__ import annotations

from types import SimpleNamespace as NS

from streamer.league.espn import position_limits as espn_limits
from streamer.league.model import PlayerRow, breaks_limit, over_limit, position_limits
from streamer.roster.title_moves import _droppable, drop_candidates


def _p(pid, pos, ros=5.0, ir=False):
    p = PlayerRow(player_id=pid, name=pid, position=pos, team="KC", ros_value=ros, projection=ros)
    if ir:
        p.slot = "IR"
    return p


def test_espn_limits_are_read_from_its_roster_settings():
    # Keyed by player position id (1 QB, 2 RB, 3 WR, 4 TE, 5 K, 16 D/ST), as
    # this league sends them: six backs, three tight ends, three defences.
    raw = {"settings": {"rosterSettings": {"positionLimits": {"0": 0, "1": 0, "2": 6, "3": 0, "4": 3, "5": 0,
                                                               "16": 3, "17": 0}}}}
    league = NS(espn_request=NS(league_get=lambda params=None: raw))
    assert espn_limits(league) == {"RB": 6, "TE": 3, "DST": 3}               # 0 is no limit
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


def test_a_deal_is_only_blocked_by_a_limit_it_breaks():
    """A roster already over a limit (one set mid-season, or read wrong) is
    not blocked from deals that leave that position alone."""
    roster = [_p(f"wr{i}", "WR") for i in range(5)] + [_p("rb", "RB")]
    swap = roster[:4] + [_p("wr9", "WR"), _p("rb", "RB")]                 # a receiver for a receiver
    assert not breaks_limit(roster, swap, {"WR": 3})
    assert breaks_limit(roster, roster + [_p("wr8", "WR")], {"WR": 3})   # one more where already over
    assert breaks_limit(roster, roster + [_p("rb2", "RB")], {"RB": 1})
