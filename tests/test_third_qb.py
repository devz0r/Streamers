"""A third quarterback adds no upside the second does not already cover."""

from streamer.league.model import PlayerRow
from streamer.roster.waivers import roster_value


def _qb(pid, v, sd=4.0):
    p = PlayerRow(player_id=pid, name=pid, position="QB", team="AAA", projection=v)
    p.ros_value, p.ros_sd = v, sd
    return p


def test_a_third_quarterback_adds_almost_nothing():
    key = lambda p: float(p.ros_value)                       # noqa: E731
    slots, repl = {"QB": 1}, {"QB": 15.0}
    two = [_qb("starter", 21.0), _qb("backup", 19.8)]
    three = two + [_qb("third", 18.6)]
    gain_second = roster_value(two, slots, key, repl, with_upside=True) - roster_value(two[:1], slots, key, repl, with_upside=True)
    gain_third = roster_value(three, slots, key, repl, with_upside=True) - roster_value(two, slots, key, repl, with_upside=True)
    assert gain_second > 0.0
    assert gain_third < 0.05 * gain_second
