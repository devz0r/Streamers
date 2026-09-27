"""D/ST and kicker streams ranked by this week's P(win), and availability labels."""

from __future__ import annotations

from fixtures_league import SLOTS, my_roster, opponent_roster, snapshot
from streamer.league.model import PlayerRow
from streamer.roster import lineup


def _dst(pid: str, team: str, opp: str) -> PlayerRow:
    return PlayerRow(player_id=pid, name=f"{team} D/ST", position="DST", team=team, slot="FA",
                     eligible_slots=["DST"], projection=8.0, projection_sd=5.0, role="DST",
                     nfl_opponent=opp)


def _stream(strength: float):
    roster = my_roster()
    for p in roster:
        p.role = {"QB": "QB", "K": "K", "DST": "DST"}.get(p.position, p.position + "2")
    opt = lineup.optimise(roster, SLOTS, opponent_roster(strength), n_sims=6000)
    faces_mine = _dst("x1", "LV", "KC")         # our QB plays for KC
    neutral = _dst("x2", "NYG", "TEN")
    opts = lineup.stream_options(opt, "DST", [faces_mine, neutral], n_sims=40000)
    return {o.player.player_id: o for o in opts}


def test_a_defence_facing_your_quarterback_is_a_hedge():
    """It moves against your QB (-0.44), so it narrows the spread of your
    total: what a favourite wants and an underdog does not. The P(win)
    effect is small (tenths of a point), so the mechanism is what is tested."""
    import numpy as np

    under = _stream(1.4)
    assert any("works against your QB Player 1" in link for link in under["x1"].links)
    qb = next(p for p in my_roster() if p.position == "QB")
    qb.role = "QB"
    rng = np.random.default_rng(5)
    s = lineup.sample_points([qb, _dst("x1", "LV", "KC"), _dst("x2", "NYG", "TEN")], 60000, rng)
    assert (s[:, 0] + s[:, 1]).std() < (s[:, 0] + s[:, 2]).std() - 0.5


def test_no_streams_once_your_defence_has_kicked_off():
    roster = my_roster()
    dst = next(p for p in roster if p.position == "DST")
    dst.locked = True
    opt = lineup.optimise(roster, SLOTS, opponent_roster(), n_sims=2000)
    assert lineup.stream_options(opt, "DST", [_dst("x2", "NYG", "TEN")], n_sims=2000) == []


def test_availability_labels(tmp_cfg):
    from streamer.league.store import snapshot_path
    from streamer.publish import unit_availability

    snap = snapshot(week=3)
    snap.free_agents.append(_dst("fa", "NYG", "TEN"))
    path = snapshot_path(tmp_cfg, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    snap.save(path)
    avail = unit_availability(tmp_cfg, 3, {"DST": ["SF", "NYG", "PIT", "SEA"]})
    assert avail[("DST", "SF")][0] == "yours"
    assert avail[("DST", "NYG")][0] == "available"
    assert avail[("DST", "PIT")][0] == "taken"          # on the opponent's roster
    assert ("DST", "SEA") not in avail                  # ESPN: unknown stays unlabelled
    snap.platform = "yahoo"
    snap.save(path)
    avail = unit_availability(tmp_cfg, 3, {"DST": ["SF", "NYG", "PIT", "SEA"]})
    assert avail[("DST", "SEA")][0] == "taken"          # Yahoo's short list is the whole pool
