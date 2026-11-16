"""Trades made in the league: found from the rosters, logged, priced."""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import pytest

from streamer.roster import league_trades, trade_eval
from test_trade_eval import league  # noqa: F401 - the shared league and its evaluator data


def test_a_two_way_swap_is_a_trade_and_a_one_way_move_is_not():
    before = {"1": {"a": "A", "b": "B"}, "2": {"c": "C", "d": "D"}, "3": {"e": "E"}}
    after = {"1": {"b": "B", "c": "C"}, "2": {"a": "A", "d": "D"}, "3": {"e": "E", "x": "X"}}
    found = league_trades.detect(before, after, {"1": "One", "2": "Two", "3": "Three"})
    assert len(found) == 1
    t = found[0]
    assert (t["a_name"], t["a_got_names"], t["b_name"], t["b_got_names"]) == ("One", ["C"], "Two", ["A"])
    moved = {"1": {"b": "B"}, "2": {"c": "C", "d": "D", "a": "A"}, "3": {"e": "E"}}     # dropped and claimed
    assert league_trades.detect(before, moved, {}) == []


def test_trades_are_logged_once_and_the_rosters_kept(league, tmp_cfg):  # noqa: F811
    snap, _finder, _data = league
    snap = copy.deepcopy(snap)
    cfg = tmp_cfg.for_profile(snap.profile)
    when = datetime(2026, 11, 16, tzinfo=UTC)
    assert league_trades.record(cfg, snap, when) == []                 # first sight: nothing to compare with
    a, b = snap.teams[1], snap.teams[2]
    pa, pb = a.roster.pop(0), b.roster.pop(0)
    a.roster.append(pb)
    b.roster.append(pa)
    trades = league_trades.record(cfg, snap, when)
    assert len(trades) == 1 and trades[0]["a_got"] == [pb.player_id] and trades[0]["seen"] == "2026-11-16"
    assert league_trades.record(cfg, snap, when) == trades             # the same rosters: nothing new
    assert league_trades.logged(cfg, snap.season) == trades


def test_a_trade_is_priced_as_the_opposite_of_undoing_it(league):  # noqa: F811
    _snap, _finder, data = league
    skill = lambda ids: [i for i in ids if data["players"][i]["pos"] in ("QB", "RB", "WR", "TE")]  # noqa: E731
    a, b = [t for t in data["teams"] if not t["me"]][:2]
    t = {"a": a["id"], "a_name": a["n"], "a_got": skill(a["r"])[:1], "a_got_names": ["x"],
         "b": b["id"], "b_name": b["n"], "b_got": skill(b["r"])[:1], "b_got_names": ["y"]}
    p = league_trades.price(data, t)
    undo = trade_eval.evaluate(data, b["id"], t["a_got"], t["b_got"], side=a["id"])
    assert p["a_title"] == pytest.approx((undo["title_me"][1], undo["title_me"][0]))
    assert p["b_title"] == pytest.approx((undo["title_them"][1], undo["title_them"][0]))
    assert p["a_points"] == pytest.approx(-undo["shift_me"]) and "you_title" in p
    assert p["a_title"][1] == pytest.approx(data["base"]["title"][[x["id"] for x in data["teams"]].index(a["id"])])
    gone = dict(t, a_got=["nobody"])
    assert league_trades.price(data, gone) is None


def test_the_card_lists_each_trade_with_both_teams_odds(league):  # noqa: F811
    from streamer.roster.page import _league_trades

    _snap, _finder, data = league
    a, b = [t for t in data["teams"] if not t["me"]][:2]
    t = {"a": a["id"], "a_name": a["n"], "a_got": [a["r"][0]], "a_got_names": ["Star"],
         "b": b["id"], "b_name": b["n"], "b_got": [b["r"][0]], "b_got_names": ["Scrub"], "seen": "2026-11-16"}

    class R:
        league_trades = [(t, league_trades.price(data, t)), (dict(t, seen="2026-11-01"), None)]

    html = _league_trades(R())
    assert "Trades made in the league" in html and "gets Star" in html and "gets Scrub" in html
    assert f"{a['n']} title" in html and "your title" in html and "not priced" in html
    assert html.index("2026-11-16") < html.index("2026-11-01")          # newest first

    class Empty:
        league_trades = None

    assert _league_trades(Empty()) == ""
