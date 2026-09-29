"""Several waiver moves priced together."""

from __future__ import annotations

import pytest

from streamer.roster import season as season_mod
from streamer.roster.title_moves import TitleEngine
from streamer.roster.waiver_plans import PlanFinder
from test_season import _league
from test_title_moves import _fa


@pytest.fixture(autouse=True)
def _no_byes(monkeypatch):
    monkeypatch.setattr(season_mod, "nfl_byes", lambda *a, **k: {})


@pytest.fixture
def engine(cfg):
    snap = _league([0.8, 1.1, 1.0, 1.05, 0.95, 1.0])
    # Two starters on the wire at different positions: each helps alone,
    # both help more.
    snap.free_agents = [_fa("rb-star", "RB", 22.0), _fa("wr-star", "WR", 22.0),
                        _fa("rep1", "RB", 5.0), _fa("rep2", "WR", 5.0), _fa("rep3", "RB", 4.5),
                        _fa("rep4", "WR", 4.5), _fa("qb", "QB", 9.0), _fa("te", "TE", 4.0)]
    snap.teams[0].waiver_rank = 2
    for t in snap.teams:
        t.acquisitions = 3
    return TitleEngine(snap, cfg, n_sims=3000, candidates=8)


def test_two_starters_together_beat_either_alone(engine):
    plans = PlanFinder(engine, engine.moves(5)).find(3)
    assert plans, "adding both stars should beat adding one"
    best = plans[0]
    assert {"rb-star", "wr-star"} <= {a.player_id for a in best.adds}
    assert best.p_plan > best.p_single > best.p_base
    assert best.over_single >= 2 * best.noise and best.verdict == "plan"
    assert any("over the best single move" in r for r in best.reasons)


def test_a_plan_swaps_one_for_one_and_keeps_a_quarterback(engine):
    for plan in PlanFinder(engine).screen():
        assert len(plan.adds) == len(plan.drops) == len({d.player_id for d in plan.drops})
        roster = [p for p in engine.me.roster if p not in plan.drops] + plan.adds
        assert any(p.position == "QB" for p in roster) and any(p.position == "TE" for p in roster)


def test_claims_go_in_order_of_what_each_is_worth_alone(engine):
    for plan in PlanFinder(engine, engine.moves(5)).find(3):
        gains = [plan.single_gains[a.player_id] for a in plan.adds]
        assert gains == sorted(gains, reverse=True)


def test_the_page_lists_the_steps(engine):
    from types import SimpleNamespace

    from streamer.roster.page import _plans

    plans = PlanFinder(engine, engine.moves(5)).find(3)
    html = _plans(SimpleNamespace(plans=plans))
    assert "Waiver plans" in html and html.count("<li>") == sum(len(p.moves) for p in plans)
    assert _plans(SimpleNamespace(plans=None)) == ""
    assert "No plan beats" in _plans(SimpleNamespace(plans=[]))


def test_the_first_claim_goes_where_priority_is_worth_most():
    """Brissett is worth more on his own but nobody else wants him; Sadiq is
    wanted by half the league. On a rolling list the first claim spends your
    priority, so it goes on Sadiq, and Brissett clears from the back."""
    from types import SimpleNamespace

    from streamer.league.model import PlayerRow
    from streamer.roster.waiver_plans import Plan, PlanFinder, claims_text, drops_text

    qb, te = PlayerRow(player_id="qb", name="QB2", position="QB"), PlayerRow(player_id="te", name="TE", position="TE")
    d1, d2 = PlayerRow(player_id="d1", name="Cut1", position="WR"), PlayerRow(player_id="d2", name="Cut2", position="WR")
    alt = PlayerRow(player_id="d3", name="Cut3", position="WR")
    plan = Plan(moves=[(qb, d1), (te, d2)], proxy=1.0, single_gains={"qb": 0.009, "te": 0.007},
                drop_options={"d2": [(d2, 0.048), (alt, 0.047)]})
    odds = {"qb": (0.05, 0.01), "te": (0.60, 0.30)}
    engine = SimpleNamespace(priority_waivers=True, claim_odds=lambda x: odds[x.player_id])
    PlanFinder._claim_order(SimpleNamespace(engine=engine), plan)
    assert claims_text(plan) == "TE, QB2"
    assert drops_text(plan) == "Cut1 and one of Cut2 (4.8%) or Cut3 (4.7%)"
    # FAAB or an order reset by standings: nothing to spend, most valuable first.
    engine.priority_waivers = False
    PlanFinder._claim_order(SimpleNamespace(engine=engine), plan)
    assert claims_text(plan) == "QB2, TE"


def test_a_backup_quarterback_would_not_start_for_a_team_with_a_starter():
    from streamer.league.model import PlayerRow
    from streamer.roster.title_moves import would_start

    def p(pid, pos, proj):
        return PlayerRow(player_id=pid, name=pid, position=pos, projection=proj, ros_value=proj)

    slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
    assert not would_start(p("qb2", "QB", 16.0), [p("qb1", "QB", 19.0)], slots)
    assert would_start(p("qb2", "QB", 16.0), [p("qb1", "QB", 12.0)], slots)
    backs = [p("r1", "RB", 18.0), p("r2", "RB", 12.0), p("r3", "RB", 6.0)]
    assert would_start(p("rb", "RB", 11.0), backs, slots)          # beats their flex back
    assert not would_start(p("rb", "RB", 5.0), backs, slots)
