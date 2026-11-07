"""Game lines: parsing the feed, pricing numbers on the margin distribution, sure bets, settling."""

import pandas as pd

from streamer.config import get_config
from streamer.data import game_lines as G

STARTS = "2026-11-09T18:00:00Z"
NOW = pd.Timestamp("2026-11-08T12:00:00Z")


def _event(books: dict) -> dict:
    """books: name -> (home ml, away ml, home spread, home sp price, away sp price, total, over, under)."""
    odds = {}
    for book, (hml, aml, sp, hsp, asp, tot, ov, un) in books.items():
        def put(key, side_price, **extra):
            odds.setdefault(key, {"byBookmaker": {}})["byBookmaker"][book] = {"odds": str(side_price), **extra}
        put("points-home-game-ml-home", hml)
        put("points-away-game-ml-away", aml)
        put("points-home-game-sp-home", hsp, spread=str(sp))
        put("points-away-game-sp-away", asp, spread=str(-sp))
        put("points-all-game-ou-over", ov, overUnder=str(tot))
        put("points-all-game-ou-under", un, overUnder=str(tot))
    return {"eventID": "g1", "status": {"startsAt": STARTS},
            "teams": {"home": {"names": {"long": "Buffalo Bills"}}, "away": {"names": {"long": "Miami Dolphins"}}},
            "odds": odds}


def test_the_feed_is_read_into_one_row_per_book_and_market():
    f = G.parse([_event({"draftkings": (-160, 135, -3.0, -110, -110, 46.5, -110, -110)})])
    assert set(f.market) == {"moneyline", "spread", "total"}
    sp = f[f.market == "spread"].iloc[0]
    assert (sp.home, sp.away, sp.line) == ("BUF", "MIA", -3.0)


def test_a_book_still_at_minus_two_and_a_half_is_flagged_against_minus_three_and_a_half(tmp_path, monkeypatch):
    cfg = get_config()
    books = {b: (-180, 150, -3.5, -110, -110, 46.5, -110, -110) for b in ("draftkings", "fanduel", "caesars")}
    books["betmgm"] = (-180, 150, -2.5, -110, -110, 46.5, -110, -110)       # a stale half point through 3
    f = G.parse([_event(books)]).assign(logged_at="2026-11-08T11:00:00+00:00")
    b = G.board(cfg, f, now=NOW)
    top = b.iloc[0]
    assert (top.market, top.book, top.side) == ("spread", "betmgm", "a") and top.ev > 0.05
    assert G.side_text(top) == "BUF -2.5"


def test_a_sure_bet_on_the_moneyline():
    cfg = get_config()
    books = {"draftkings": (-120, 110, -1.5, -110, -110, 46.5, -110, -110),
             "fanduel": (-105, -105, -1.5, -110, -110, 46.5, -110, -110),
             "caesars": (-130, 125, -1.5, -110, -110, 46.5, -110, -110)}
    f = G.parse([_event(books)]).assign(logged_at="2026-11-08T11:00:00+00:00")
    arbs = [a for a in G.hedges(cfg, f, now=NOW)["arbs"] if a["market"] == "moneyline"]
    assert arbs and arbs[0]["book_a"] == "fanduel" and arbs[0]["book_b"] == "caesars" and arbs[0]["profit"] > 0
