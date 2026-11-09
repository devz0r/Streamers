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


def test_a_book_frozen_since_an_earlier_pull_is_not_on_the_board():
    """Three books read at noon, one last read at 08:00: its price is history, not an offer."""
    noon, early = "2026-11-08T12:00:00+00:00", "2026-11-08T08:00:00+00:00"
    lines = pd.DataFrame([_q("fanduel", 140, -170, at=early), _q("betmgm", -110, -110, at=noon),
                          _q("caesars", -112, -108, at=noon), _q("draftkings", 108, -128, at=noon)])
    b = betting.board(lines)
    assert len(b) and "fanduel" not in set(b.book)                       # judged among the three read at noon
    # And the row carries the time its own book was read, which is what the page shows as its age.
    fresh = pd.DataFrame([_q("fanduel", 120, -150, at="2026-11-08T11:30:00+00:00"), _q("betmgm", -110, -110, at=noon),
                          _q("caesars", -112, -108, at=noon), _q("pinnacle", -105, -115, at=noon)])
    top = betting.board(fresh).iloc[0]
    assert top.book == "fanduel" and top.logged_at.startswith("2026-11-08T11:30")


def test_a_sure_bet_needs_both_prices_read_close_together():
    stale = pd.DataFrame([_q("a", 110, -140, at="2026-11-08T11:00:00+00:00"), _q("b", -150, 105), _q("c", -115, -105)])
    assert betting.hedges(stale)["arbs"] == []                           # an hour apart: one leg may be gone
    fresh = pd.DataFrame([_q("a", 110, -140, at="2026-11-08T11:50:00+00:00"), _q("b", -150, 105), _q("c", -115, -105)])
    assert len(betting.hedges(fresh)["arbs"]) == 1


def test_the_record_counts_a_bet_once_and_reads_the_close_two_ways():
    row = {"season": 2026, "week": 9, "event_id": "e1", "stat": "receiving_yards", "player": "Wide Out", "side": "over",
           "price": 100.0, "ev": 0.05, "kind": "market", "p_market": 0.52}
    bets = pd.DataFrame([
        {**row, "line": 54.5, "book": "a", "logged_at": "2026-11-08T12:00:00+00:00", "result": "win", "clv": 0.03, "move": 0.01},
        {**row, "line": 55.5, "book": "b", "logged_at": "2026-11-08T14:00:00+00:00", "result": "win", "clv": 0.02, "move": 0.02},
        {**row, "player": "Tight End", "line": 30.5, "book": "a", "logged_at": "2026-11-08T12:00:00+00:00",
         "result": "loss", "clv": -0.02, "move": -0.03}])
    s = betting.summary(bets)
    assert (s["flagged"], s["prices"], s["wins"], s["losses"]) == (2, 3, 1, 1)      # the second price is the same bet
    assert s["profit"] == 0.0 and s["clv_hit"] == 0.5 and s["move_toward"] == 0.5 and s["move_away"] == 0.5


def test_a_line_that_moved_is_still_graded_at_the_close(tmp_path, monkeypatch):
    """Flagged over 54.5; by the close every book has moved to 58.5: the market came to the bet."""
    cfg = get_config()
    monkeypatch.setattr(betting, "path", lambda _c: tmp_path / "bet_log.parquet")
    lines = pd.DataFrame([_q("fanduel", 120, -150), _q("betmgm", -110, -110), _q("caesars", -112, -108),
                          _q("pinnacle", -105, -115)])
    betting.flag(betting.board(lines), cfg)
    later = "2026-11-09T17:00:00+00:00"
    close = pd.concat([lines, pd.DataFrame([_q(b, -110, -110, line=58.5, at=later)
                                            for b in ("fanduel", "betmgm", "caesars", "pinnacle")])])
    s = betting.settle(cfg, close)
    over = s[(s.side == "over") & (s.book == "fanduel")].iloc[0]
    assert over.close_fair > 0.53 and over.move > 0.03 and over.clv > 0


def test_a_whole_number_line_on_a_count_stat_pushes_on_the_number():
    from scipy import stats as st

    from streamer.data.props import mean_from_poisson_line

    assert betting.p_over("receptions", 4.0, 4.0, {}) == st.poisson.sf(4, 4.0)        # five or more
    assert betting.p_over("receptions", 4.0, 4.5, {}) == st.poisson.sf(4, 4.0)
    assert betting.p_push("receptions", 4.0, 4.0) == st.poisson.pmf(4, 4.0) and betting.p_push("receptions", 4.0, 4.5) == 0
    assert abs(st.poisson.sf(4, mean_from_poisson_line(4.0, 0.4)) - 0.4) < 1e-6
