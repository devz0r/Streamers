"""Waiver moves valued by title odds."""

from __future__ import annotations

import pytest

from streamer.league.model import PlayerRow
from streamer.roster import season as season_mod
from streamer.roster.title_moves import TitleEngine, drop_choice, p_win_claim, toss_ups
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


def test_passing_on_a_breakout_lets_a_rival_have_him(cfg):
    snap = _league([1.0, 1.1, 1.0, 1.05, 0.95, 1.0])
    riser = _fa("riser", "RB", 11.0)
    riser.ros_sd = 3.0
    snap.free_agents = [riser, _fa("rep1", "RB", 5.0), _fa("rep2", "WR", 5.0),
                        _fa("rep3", "RB", 4.5), _fa("rep4", "WR", 4.5)]
    for i, t in enumerate(snap.teams):
        t.acquisitions, t.waiver_rank = 6, i + 1
    engine = TitleEngine(snap, cfg, n_sims=3000, candidates=8)
    drop = sorted(engine.me.roster, key=lambda p: p.ros_value)[0]
    pat = engine.stand_pat(riser, drop)
    claimant = engine._claimant(riser)
    assert (claimant >= 0).any() and not (claimant == engine.mine).any()
    # A rival landing him can only hurt you (same seasons otherwise).
    assert pat.mean() <= engine.base + 0.004
    moves = {m.add.player_id: m for m in engine.moves(5)}
    if "riser" in moves:
        m = moves["riser"]
        assert m.p_free == engine.base and m.rival_share > 0 and m.rival_name


def test_drops_the_simulation_cannot_separate_are_all_offered_with_their_odds(cfg):
    snap = _league([1.0, 1.1, 1.0, 1.05, 0.95, 1.0])
    me = snap.teams[0]
    # Two bench bodies below replacement: neither ever starts, so dropping
    # either is exactly as good for the title -- but one still has a name.
    for pid in ("no-name", "big-name"):
        me.roster.append(_fa(pid, "WR", 2.0))
    snap.free_agents = [_fa("star", "RB", 24.0), _fa("rep1", "RB", 6.0), _fa("rep2", "WR", 6.0),
                        _fa("rep3", "RB", 5.5), _fa("rep4", "WR", 5.5)]
    for t in snap.teams:
        t.acquisitions = 3
    engine = TitleEngine(snap, cfg, n_sims=1500, candidates=8)
    engine.market = {"no-name": 2.0, "big-name": 11.0}
    star = next(m for m in engine.moves(8) if m.add.player_id == "star")
    offered = [q.player_id for q, _p in star.drop_options]
    assert {"no-name", "big-name"} <= set(offered)
    assert star.drop.player_id == offered[0] and star.p_now == star.drop_options[0][1]   # best listed first
    assert any("toss-up" in r and "big-name (title" in r and "no-name (title" in r for r in star.reasons)
    assert drop_choice(star.drop, star.drop_options).startswith(f"{star.drop.name} ({star.p_now:.1%}) or ")


def test_a_clear_best_drop_is_not_offered_as_a_toss_up():
    import numpy as np

    rng = np.random.default_rng(0)
    base = (rng.random(4000) < 0.1).astype(float)
    better = np.maximum(base, (rng.random(4000) < 0.05).astype(float))   # wins everything base wins, and more
    a, b = _fa("a", "WR", 5.0), _fa("b", "WR", 5.0)
    assert [q.player_id for q, _p in toss_ups([(a, base), (b, better)])] == ["b"]
    assert drop_choice(b, toss_ups([(a, base), (b, better)])) == "b"


def test_drops_the_simulation_cannot_separate_go_cheapest_first():
    """Noise does not order a toss-up: a 2-point bench back is cut before a
    10-point receiver whose drop happened to price a hair higher."""
    import numpy as np

    rng = np.random.default_rng(3)
    won = (rng.random(4000) < 0.1).astype(float)
    noisy = won.copy()
    # Wins 30 seasons the other does not and loses 22 it does: a hair better, inside the noise.
    noisy[rng.choice(np.flatnonzero(won == 0), 30, replace=False)] = 1.0
    noisy[rng.choice(np.flatnonzero(won == 1), 22, replace=False)] = 0.0
    receiver, back = _fa("receiver", "WR", 9.9), _fa("back", "RB", 2.2)
    got = toss_ups([(receiver, noisy), (back, won)], market=lambda p: {"receiver": 11.5, "back": 2.0}[p.player_id])
    assert [q.player_id for q, _p in got] == ["back", "receiver"]


def test_upside_plays_that_fall_short_are_listed_with_their_value(cfg):
    snap = _league([1.0, 1.1, 1.0, 1.05, 0.95, 1.0])
    lead = snap.teams[1].roster[1]
    lead.team, lead.role, lead.ros_value = "SEA", "RB1", 18.0
    cuff = _fa("cuff", "RB", 3.0)
    cuff.role = "RB2"
    snap.free_agents = [_fa("star", "RB", 24.0), _fa("rep1", "RB", 6.0), _fa("rep2", "WR", 6.0),
                        _fa("rep3", "RB", 5.5), _fa("rep4", "WR", 5.5), cuff]
    engine = TitleEngine(snap, cfg, n_sims=1500, candidates=8)
    shown = {m.add.player_id for m in engine.moves(1)}
    passed = {p.player_id: (gain, why) for p, gain, why in engine.passed}
    assert "cuff" in engine.upside and "cuff" not in shown and "cuff" in passed
    assert "behind" in passed["cuff"][1]
