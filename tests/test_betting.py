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
    assert 0.47 < top.fair < 0.52 and top.ev > 0.05                   # +120 on a near coin flip
    assert betting.flag(b, cfg) == 1 and betting.flag(b, cfg) == 0      # logged once
    assert betting.board(lines, exclude={"wide out"}).empty              # a started game is off the board
    # The market then moved toward it: closing prices lower the over's price and the actual lands over.
    close = pd.concat([lines, pd.DataFrame([_q("betmgm", -125, 105, at="2026-11-09T17:00:00+00:00"),
                                            _q("caesars", -128, 108, at="2026-11-09T17:00:00+00:00"),
                                            _q("pinnacle", -130, 110, at="2026-11-09T17:00:00+00:00"),
                                            _q("fanduel", -126, 106, at="2026-11-09T17:00:00+00:00")])])
    s = betting.settle(cfg, close)
    assert s.iloc[0].clv > 0 and s.iloc[0].result is None               # no stats file here: ungraded
    sm = betting.summary(s)
    assert sm["flagged"] == 1 and sm["clv_hit"] == 1.0
