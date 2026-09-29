"""Waiver moves valued by title odds."""

from __future__ import annotations

import pytest

from streamer.league.model import PlayerRow
from streamer.roster import season as season_mod
from streamer.roster.title_moves import TitleEngine, p_win_claim
from test_season import _league


@pytest.fixture(autouse=True)
def _no_byes(monkeypatch):
    monkeypatch.setattr(season_mod, "nfl_byes", lambda *a, **k: {})


def test_claim_odds_fall_as_priority_falls_and_with_more_rivals():
    top = p_win_claim(1, 10, 0.3)
    mid = p_win_claim(5, 10, 0.3)
    last = p_win_claim(10, 10, 0.3)
    assert top == pytest.approx(1.0) and top > mid > last
    assert last == pytest.approx(0.7 ** 9)                  # only if nobody else claims
    assert p_win_claim(5, 10, 0.6) < mid


def _fa(pid: str, pos: str, ros: float) -> PlayerRow:
    return PlayerRow(player_id=pid, name=pid, position=pos, team="SEA", slot="FA", projection=ros,
                     projection_sd=6.0, outcome_sd=6.0, ros_value=ros, ros_sd=2.0,
                     eligible_slots=[pos, "FLEX"])


def test_a_star_on_the_wire_is_a_claim_and_a_scrub_is_not(cfg):
    snap = _league([1.0, 1.1, 1.0, 1.05, 0.95, 1.0])
    snap.free_agents = [_fa("star", "RB", 24.0), _fa("scrub", "WR", 3.0),
                        _fa("rep1", "RB", 6.0), _fa("rep2", "WR", 6.0), _fa("rep3", "RB", 5.5), _fa("rep4", "WR", 5.5)]
    snap.teams[0].waiver_rank = 4
    for t in snap.teams:
        t.acquisitions = 3
    engine = TitleEngine(snap, cfg, n_sims=3000, candidates=8)
    moves = {m.add.player_id: m for m in engine.moves(8)}
    assert "star" in moves and moves["star"].verdict == "claim"
    assert moves["star"].p_now > engine.base + 0.02
    assert "scrub" not in moves


def test_upside_plays_are_priced_even_when_their_average_is_low(cfg):
    snap = _league([1.0, 1.1, 1.0, 1.05, 0.95, 1.0])
    lead = snap.teams[1].roster[1]
    lead.team, lead.role, lead.ros_value = "SEA", "RB1", 18.0
    cuff = _fa("cuff", "RB", 4.0)
    cuff.role = "RB2"
    rookie = _fa("rook", "WR", 4.5)
    rookie.experience = 0
    snap.free_agents = [_fa(f"rb{i}", "RB", 9.0 - i * 0.2) for i in range(10)] + \
        [_fa(f"wr{i}", "WR", 9.0 - i * 0.2) for i in range(10)] + [cuff, rookie]
    engine = TitleEngine(snap, cfg, n_sims=500, candidates=8)
    ids = {p.player_id for p in engine.candidates}
    assert {"cuff", "rook"} <= ids
    assert "behind" in engine.upside["cuff"] and "rookie" in engine.upside["rook"]
