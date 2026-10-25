"""Head to head, the opponent's obvious fixes, and the latest game in the flex."""

from __future__ import annotations

from streamer.league.model import PlayerRow
from streamer.roster.lineup import flex_last, opponent_fixes
from streamer.roster.page import pair_slots


def _p(pid, pos, proj, kick=None, **kw):
    return PlayerRow(player_id=pid, name=pid, position=pos, team="KC", projection=proj, kickoff=kick, **kw)


EARLY, LATE, SNF = "2026-10-25T17:00:00+00:00", "2026-10-25T20:25:00+00:00", "2026-10-26T00:20:00+00:00"


def test_the_latest_kickoff_at_the_flexed_position_sits_in_the_flex():
    rb1, rb2, rb3 = _p("rb1", "RB", 15, SNF), _p("rb2", "RB", 12, EARLY), _p("rb3", "RB", 10, LATE)
    wr = _p("wr", "WR", 14, SNF)
    lineup = {"RB": [rb1, rb2], "WR": [wr], "FLEX": [rb3]}
    out, moves = flex_last(lineup)
    assert out["FLEX"] == [rb1] and {q.player_id for q in out["RB"]} == {"rb2", "rb3"}
    assert moves == [("FLEX", rb1, "RB", rb3)]
    assert out["WR"] == [wr]                         # another position is not the one being flexed


def test_a_started_game_never_moves():
    rb1 = _p("rb1", "RB", 15, SNF, locked=True)
    rb3 = _p("rb3", "RB", 10, LATE)
    out, moves = flex_last({"RB": [rb1], "FLEX": [rb3]})
    assert out["FLEX"] == [rb3] and not moves


def test_an_opponent_starter_already_out_is_replaced_from_his_bench():
    out_wr = _p("out", "WR", 0, LATE, status="OUT")
    bench = [_p("b1", "WR", 9, SNF), _p("b2", "WR", 7, SNF), _p("rb", "RB", 11, SNF)]
    fixed, notes = opponent_fixes({"WR": [out_wr]}, [out_wr] + bench)
    assert fixed["WR"] == [bench[0]] and "out: assumed replaced by b1" in notes[0]
    q = _p("q", "WR", 12, LATE, status="QUESTIONABLE")
    kept, notes = opponent_fixes({"WR": [q]}, [q] + bench)
    assert kept["WR"] == [q] and not notes          # questionable: replaced only in the draws he sits


def test_slots_pair_best_against_best():
    a1, a2 = _p("a1", "RB", 10), _p("a2", "RB", 20)
    b1, b2 = _p("b1", "RB", 15), _p("b2", "RB", 5)
    rows = pair_slots({"RB": [a1, a2]}, {"RB": [b2, b1]})
    assert [(x.player_id, y.player_id) for _s, x, y in rows] == [("a2", "b1"), ("a1", "b2")]
