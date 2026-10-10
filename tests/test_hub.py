"""The championship hub: every move on one list, in title odds."""

from __future__ import annotations

from types import SimpleNamespace as NS

from streamer.league.model import PlayerRow
from streamer.roster.hub import actions, week_title_value


def _p(name, pos="WR"):
    return PlayerRow(player_id=name, name=name, position=pos)


def _report(title_moves=(), plans=(), trades=None, changes=True, streams=None):
    stakes = NS(games=[NS(week=4, title_win=0.06, title_loss=0.02)])
    opt = NS(changes=[("FLEX", _p("Bench Guy"), _p("Starter Guy"))] if changes else [])
    return NS(snapshot=NS(week=4), stakes=stakes, optimisation=opt, current_win_probability=0.40,
              win_probability=0.45, streams=streams or {}, title_moves=list(title_moves),
              plans=list(plans), trades=trades)


def _move(name, gain, cost=0.0, block=0.0, verdict="claim"):
    return NS(add=_p(name), drop=_p("Drop Guy"), gain_now=gain, priority_cost=cost, block_value=block,
              rival_name="Rival" if block else "", verdict=verdict)


def test_a_win_this_week_is_worth_its_title_swing():
    assert abs(week_title_value(_report()) - 0.04) < 1e-12
    lineup = [a for a in actions(_report()) if a.kind == "Lineup"][0]
    assert abs(lineup.gain - 0.05 * 0.04) < 1e-12             # +5 points of P(win) x 4 points of title


def test_every_kind_lands_on_one_list_best_first():
    board = NS(top=[NS(key=1, give=[_p("A")], get=[_p("B")], partner=NS(name="T"), gain=0.012,
                       p_accept=0.6, win_win=False)], best=[], likely=[])
    board.__bool__ = lambda self=None: True
    plan = NS(moves=[(_p("X"), _p("Y")), (_p("Z"), _p("W"))], gain=0.02, verdict="plan")
    streams = {"DST": [NS(player=_p("Mine D", "DST"), current=True, win_probability=0.45),
                       NS(player=_p("Wire D", "DST"), current=False, win_probability=0.50)]}
    r = _report(title_moves=[_move("Claimer", 0.009, cost=0.001), _move("Blocker", 0.006, block=0.004),
                             _move("Nope", -0.002)],
                plans=[plan], trades=board, streams=streams)
    acts = actions(r)
    kinds = [a.kind for a in acts]
    assert kinds[0] == "Plan" and "Trade" in kinds and "Block" in kinds and "Stream" in kinds
    assert [a.gain for a in acts] == sorted((a.gain for a in acts), reverse=True)
    assert all(a.gain > 0 for a in acts)                      # a move that hurts is not listed
    claim = next(a for a in acts if a.headline == "Add Claimer")
    assert abs(claim.gain - 0.008) < 1e-12                    # priority cost comes off
    trade = next(a for a in acts if a.kind == "Trade")
    assert trade.firm is False and abs(trade.gain - 0.6 * 0.012) < 1e-12   # expected: yes-chance x gain


def test_no_season_model_means_no_weekly_title_value():
    r = _report()
    r.stakes = None
    assert week_title_value(r) is None
    assert not [a for a in actions(r) if a.kind in ("Lineup", "Stream")]


def test_a_move_priced_on_the_season_also_shows_its_playoff_change():
    from streamer.roster.hub import HubAction
    from streamer.roster.page import _hub  # noqa: F401  (the table reads the field)

    a = HubAction("Trade", "Trade A for B", "with C", 0.014, "yes", "trades", firm=False, playoffs=0.04)
    assert a.playoffs == 0.04 and HubAction("Lineup", "x", "y", 0.01, "z", "lineup").playoffs is None
