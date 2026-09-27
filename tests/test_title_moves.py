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
