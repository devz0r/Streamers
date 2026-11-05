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
