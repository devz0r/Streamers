"""The best-price board: fair chance from the other books, flags logged once, graded later."""

import pandas as pd

from streamer.config import get_config
from streamer.data import betting


def _q(book, over, under, line=54.5, at="2026-11-08T12:00:00+00:00", player="Wide Out"):
    return {"season": 2026, "week": 9, "event_id": "e1", "bookmaker": book, "stat": "receiving_yards", "player": player,
            "line": line, "over_price": over, "under_price": under, "market": "m", "kind": "normal",
            "yes_price": None, "logged_at": at}


def test_a_price_better_than_the_others_read_is_flagged_once_and_graded(tmp_path, monkeypatch):
    cfg = get_config()
    monkeypatch.setattr(betting, "path", lambda _c: tmp_path / "bet_log.parquet")
    lines = pd.DataFrame([_q("fanduel", 120, -150), _q("betmgm", -110, -110), _q("caesars", -112, -108),
                          _q("pinnacle", -105, -115)])
    b = betting.board(lines)
    top = b.iloc[0]
    assert (top.side, top.book, top.price) == ("over", "fanduel", 120.0)
    assert 0.45 < top.fair < 0.55 and top.ev > 0.05                   # +120 on a near coin flip
    n = int((b.ev >= betting.MIN_EV).sum())
    assert n >= 1 and betting.flag(b, cfg) == n and betting.flag(b, cfg) == 0      # logged once
    assert betting.settle(cfg, lines)["clv"].isna().all()                 # no later pull yet: nothing to say
    assert betting.board(lines, exclude={"wide out"}).empty              # a started game is off the board
    # The market then moved toward it: closing prices lower the over's price and the actual lands over.
    close = pd.concat([lines, pd.DataFrame([_q("betmgm", -125, 105, at="2026-11-09T17:00:00+00:00"),
                                            _q("caesars", -128, 108, at="2026-11-09T17:00:00+00:00"),
                                            _q("pinnacle", -130, 110, at="2026-11-09T17:00:00+00:00"),
                                            _q("fanduel", -126, 106, at="2026-11-09T17:00:00+00:00")])])
    s = betting.settle(cfg, close)
    over = s[(s.side == "over") & (s.book == "fanduel")].iloc[0]
    assert over.clv > 0 and over.result is None                          # no stats file here: ungraded
    assert betting.summary(s)["flagged"] == n


def test_a_lone_line_far_off_the_others_is_priced_and_flagged_way_off():
    lines = pd.DataFrame([_q("draftkings", -110, -110, line=72.5), _q("betmgm", -112, -108, line=71.5),
                          _q("pinnacle", -108, -112, line=72.5), _q("fanduel", -110, -110, line=30.5)])
    b = betting.board(lines, ours={"wide out": {"rec_yards": 74.0}})
    top = b.iloc[0]
    assert (top.side, top.book, top.line) == ("over", "fanduel", 30.5)
    assert top.p_market > 0.85 and top.p_ours > 0.85 and top.way_off
    assert top.ev > 0.6                                                  # a -110 price on a ~90% chance


def test_a_sure_bet_and_a_middle_are_found():
    sure = pd.DataFrame([_q("a", 110, -140), _q("b", -150, 105), _q("c", -115, -105)])
    h = betting.hedges(sure)
    assert len(h["arbs"]) == 1
    a = h["arbs"][0]
    assert (a["over_book"], a["under_book"]) == ("a", "b") and 0.03 < a["profit"] < 0.06
    # Each side returns the same: stake share x over odds = (1 - share) x under odds.
    do, du = 2.10, 2.05
    assert abs(a["over_share"] * do - (1 - a["over_share"]) * du) < 1e-9
    none = pd.DataFrame([_q("a", -110, -110), _q("b", -108, -112), _q("c", -112, -108)])
    assert betting.hedges(none)["arbs"] == []
    mid = pd.DataFrame([_q("a", -110, -110, line=70.5), _q("b", -110, -110, line=78.5), _q("c", -110, -110, line=74.5)])
    m = betting.hedges(mid)["middles"]
    assert m and m[0]["over_line"] < m[0]["under_line"] and m[0]["gap"] > 0.05


def test_a_book_missing_from_the_latest_pull_is_not_judged_on_its_old_quote():
    old = "2026-11-05T14:34:00+00:00"
    new = "2026-11-05T20:16:00+00:00"
    lines = pd.DataFrame([_q("betmgm", -120, -111, line=14.5, at=old)]
                         + [_q(b, -110, -115, line=9.5, at=new) for b in ("bovada", "caesars", "draftkings", "fanduel")])
    b = betting.board(lines)
    assert b.empty or "betmgm" not in set(b.book)


def test_parlays_combine_flagged_legs_at_one_book_from_different_games():
    def leg(book, event, player, p, price, ev_market):
        return {"book": book, "event_id": event, "player": player, "stat": "receptions", "side": "over", "line": 3.5,
                "price": price, "p_market": p, "ev": ev_market, "ev_market": ev_market, "kind": "market"}

    rows = pd.DataFrame([leg("fanduel", "e1", "A", 0.42, 154.0, 0.067), leg("fanduel", "e2", "B", 0.54, -108.0, 0.041),
                         leg("fanduel", "e1", "C", 0.55, 100.0, 0.10),          # the same game as A
                         leg("betmgm", "e3", "D", 0.5, 110.0, 0.05)])           # a lone leg at another book
    out = betting.parlays(rows)
    two = [r for r in out if len(r["legs"]) == 2]
    assert {tuple(sorted(x["player"] for x in r["legs"])) for r in two} == {("A", "B"), ("B", "C")}
    ab = next(r for r in two if {x["player"] for x in r["legs"]} == {"A", "B"})
    assert abs(ab["win"] - 0.42 * 0.54) < 1e-9 and abs(ab["payout"] - 2.54 * (1 + 100 / 108)) < 1e-9
    assert abs(ab["ev"] - (ab["win"] * ab["payout"] - 1)) < 1e-9 and ab["ev"] > ab["ev_legs"]
    assert all(r["book"] == "fanduel" for r in out)
