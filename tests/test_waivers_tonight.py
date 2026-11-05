"""The Waivers tonight card: only moves worth making, in order, with the priority cost."""

from types import SimpleNamespace as NS

from streamer.league.model import PlayerRow
from streamer.roster.page import _tonight


def _move(name, gain, on_waivers, verdict="claim", rival=0.4):
    add = PlayerRow(player_id=name, name=name, position="RB", team="TB", on_waivers=on_waivers)
    drop = PlayerRow(player_id="d" + name, name="Dropped", position="WR", team="PIT")
    return NS(add=add, drop=drop, p_base=0.05, p_now=0.05 + gain, gain_now=gain, verdict=verdict, watch=False,
              rival_share=rival, how="", later=None, drop_options=[])


def _report(moves, plans=None, rank=2, rules=None, platform="espn"):
    snap = NS(my_team=NS(waiver_rank=rank), teams=[0] * 10, rules=rules or {}, platform=platform)
    return NS(snapshot=snap, title_moves=moves, plans=plans or [])


def test_the_card_lists_only_real_moves_best_first_with_the_priority_cost():
    html = _tonight(_report([_move("Small", 0.004, True), _move("Big", 0.012, True), _move("Free", 0.008, False),
                            _move("Meh", 0.0, True, verdict="wait")]))
    assert "2 of 10" in html and "back of the list" in html
    assert html.index("Big") < html.index("Free") < html.index("Small") and "Meh" not in html
    assert "he is yours at once" in html                                  # the free agent needs no claim
    assert "neither platform shows them" in html


def test_with_nothing_worth_claiming_it_says_so():
    html = _tonight(_report([_move("Meh", 0.0, True, verdict="wait")]))
    assert "Nothing to claim tonight" in html
