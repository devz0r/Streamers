"""SportsGameOdds: a second props feed, parsed into the same consensus."""

from __future__ import annotations

import pandas as pd

from streamer.data import sgo
from streamer.data.props import consensus, implied_means, to_fantasy_points


def _book(odds, line=None, available=True):
    b = {"odds": odds, "available": available}
    if line is not None:
        b["overUnder"] = line
    return b


def _event():
    pid = "PATRICK_MAHOMES_1_NFL"
    wr = "RASHEE_RICE_1_NFL"
    return {
        "eventID": "ev1", "leagueID": "NFL",
        "status": {"started": False},
        "players": {pid: {"name": "Patrick Mahomes"}, wr: {"firstName": "Rashee", "lastName": "Rice"}},
        "odds": {
            f"passing_yards-{pid}-game-ou-over": {"byBookmaker": {"pinnacle": _book("-110", "265.5"),
                                                                   "fanduel": _book("-115", "264.5")}},
            f"passing_yards-{pid}-game-ou-under": {"byBookmaker": {"pinnacle": _book("-110", "265.5"),
                                                                    "fanduel": _book("-105", "264.5")}},
            f"receptions-{wr}-game-ou-over": {"byBookmaker": {"draftkings": _book("+105", "5.5")}},
            f"receptions-{wr}-game-ou-under": {"byBookmaker": {"draftkings": _book("-125", "5.5")}},
            f"touchdowns-{wr}-game-yn-yes": {"byBookmaker": {"draftkings": _book("+150")}},
            f"touchdowns-{wr}-game-yn-no": {"byBookmaker": {"draftkings": _book("-190")}},
            # Ignored: first half, a closed market, a team market.
            f"passing_yards-{pid}-1h-ou-over": {"byBookmaker": {"pinnacle": _book("-110", "130.5")}},
            f"receptions-{wr}-game-ou-over-x": {"byBookmaker": {}},
            "points-home-game-ml-home": {"byBookmaker": {"pinnacle": _book("-150")}},
        },
    }


def test_player_markets_become_rows_like_the_first_feeds(cfg):
    f = sgo.parse([_event()])
    assert set(f.columns) == set(sgo.COLUMNS)
    py = f[f.stat == "passing_yards"]
    assert sorted(py.bookmaker) == ["fanduel", "pinnacle"] and set(py.player) == {"Patrick Mahomes"}
    assert py.set_index("bookmaker").loc["pinnacle", "line"] == 265.5
    td = f[f.stat == "anytime_td"].iloc[0]
    assert td.player == "Rashee Rice" and td.kind == "poisson" and td.line == 0.5
    assert td.over_price == 150.0 and td.under_price == -190.0
    # Through the same pipeline as The Odds API's rows.
    pts = to_fantasy_points(consensus(implied_means(f, cfg), cfg), cfg.for_profile("espn"))
    assert set(pts.player) == {"Patrick Mahomes", "Rashee Rice"}
    assert (pts.vegas_points > 0).all()


def test_a_price_the_first_feed_has_counts_once_and_pickem_sites_are_left_out(cfg):
    extra = sgo.parse([_event()])
    extra = pd.concat([extra, extra.iloc[[0]].assign(bookmaker="prizepicks")], ignore_index=True)
    primary = extra[extra.bookmaker == "pinnacle"].assign(player="Patrick Mahomes")
    merged = sgo.merge(primary, extra, cfg)
    assert (merged.bookmaker == "pinnacle").sum() == 1            # not twice
    assert "prizepicks" not in set(merged.bookmaker)
    assert {"fanduel", "draftkings"} <= set(merged.bookmaker)
    assert len(sgo.merge(primary.iloc[:0], extra, cfg)) == len(extra) - 1


def test_without_a_key_it_says_so_and_returns_nothing(cfg, monkeypatch, tmp_path):
    monkeypatch.delenv("SPORTSGAMEODDS_API_KEY", raising=False)
    monkeypatch.setattr(sgo, "_cache", lambda c: tmp_path / "events.json")
    frame, note = sgo.props_frame(cfg)
    assert frame.empty and "SPORTSGAMEODDS_API_KEY" in note
    assert sgo.probe(cfg) == ["SPORTSGAMEODDS_API_KEY is not set"]
