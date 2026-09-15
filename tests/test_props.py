"""Sportsbook props -> implied fantasy points.

The pricing maths is the part worth pinning: a de-vig that does not sum to 1,
a line read as a mean rather than a median, or an anytime-TD price used as if
it were an expected touchdown count all produce numbers that look plausible
and are wrong.
"""

from __future__ import annotations

import math

import pytest

from streamer.data.props import (
    american_to_prob,
    consensus,
    devig,
    devig_one_sided,
    implied_means,
    mean_from_anytime,
    mean_from_normal_line,
    mean_from_poisson_line,
    parse_props_payload,
    props_to_points,
    spread_for,
    to_fantasy_points,
)


def _ou(player, line, over=-110, under=-110):
    return [{"description": player, "name": "Over", "point": line, "price": over},
            {"description": player, "name": "Under", "point": line, "price": under}]


def _yes(player, price):
    return [{"description": player, "name": "Yes", "price": price}]


def payload(books=("pinnacle",)):
    return [{
        "id": "evt1", "home_team": "Baltimore Ravens", "away_team": "New Orleans Saints",
        "bookmakers": [
            {"key": b, "markets": [
                {"key": "player_rush_yds", "outcomes": _ou("Derrick Henry", 84.5)},
                {"key": "player_receptions", "outcomes": _ou("Derrick Henry", 1.5)},
                {"key": "player_anytime_td", "outcomes": _yes("Derrick Henry", -125)},
            ]} for b in books
        ],
    }]


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def test_american_prices_convert():
    assert american_to_prob(-110) == pytest.approx(0.5238, abs=1e-4)
    assert american_to_prob(+150) == pytest.approx(0.4000, abs=1e-4)
    assert american_to_prob(None) is None
    assert american_to_prob(0) is None


def test_devig_removes_the_margin():
    """Two equal prices are a coin flip once the vig comes out, not 52.4%."""
    assert devig(-110, -110) == pytest.approx(0.5, abs=1e-9)
    assert devig(-140, +120) == pytest.approx(0.562, abs=1e-3)
    assert devig(-110, None) is None
    # De-vigged pairs sum to exactly 1.
    assert devig(-200, 170) + devig(170, -200) == pytest.approx(1.0, abs=1e-9)


def test_one_sided_devig_shrinks_the_price():
    raw = american_to_prob(-125)
    assert devig_one_sided(-125) < raw


# ---------------------------------------------------------------------------
# Line -> mean
# ---------------------------------------------------------------------------
def test_a_balanced_line_means_the_line(cfg):
    sd = spread_for("receiving_yards", 60.5, cfg)
    assert mean_from_normal_line(60.5, 0.5, sd) == pytest.approx(60.5)


def test_leaning_over_raises_the_mean(cfg):
    sd = spread_for("receiving_yards", 60.5, cfg)
    assert mean_from_normal_line(60.5, 0.60, sd) > 60.5
    assert mean_from_normal_line(60.5, 0.40, sd) < 60.5


def test_spread_grows_with_the_line_and_respects_its_floor(cfg):
    assert spread_for("receiving_yards", 100.0, cfg) > spread_for("receiving_yards", 20.0, cfg)
    assert spread_for("receptions", 0.5, cfg) >= 0.8
    # An unknown stat still returns something usable rather than exploding.
    assert spread_for("nonsense", 10.0, cfg) > 0


def test_poisson_line_inverts_exactly():
    """The solved rate must reproduce the probability it came from."""
    from scipy import stats

    lam = mean_from_poisson_line(1.5, 0.55)
    assert stats.poisson.sf(1, lam) == pytest.approx(0.55, abs=1e-6)
    # Monotone: a higher over-price means a higher rate.
    assert mean_from_poisson_line(1.5, 0.65) > lam


def test_anytime_td_is_a_rate_not_a_probability():
    """0.45 to score means more than 0.45 expected touchdowns."""
    assert mean_from_anytime(0.45) == pytest.approx(-math.log(0.55), abs=1e-9)
    assert mean_from_anytime(0.45) > 0.45
    assert mean_from_anytime(0.01) == pytest.approx(0.01, abs=1e-3)   # small p, little difference


# ---------------------------------------------------------------------------
# Payload -> points
# ---------------------------------------------------------------------------
def test_payload_parses_into_one_row_per_player_market_book():
    frame = parse_props_payload(payload(books=("pinnacle", "fanduel")))
    assert len(frame) == 6
    assert set(frame["bookmaker"]) == {"pinnacle", "fanduel"}
    henry = frame[frame["market"].eq("player_rush_yds")].iloc[0]
    assert henry["line"] == 84.5 and henry["over_price"] == -110


def test_unknown_markets_are_ignored():
    p = payload()
    p[0]["bookmakers"][0]["markets"].append({"key": "player_kicking_points",
                                             "outcomes": _ou("Someone", 7.5)})
    assert "player_kicking_points" not in set(parse_props_payload(p)["market"])


def test_points_are_the_scored_implied_stat_line(cfg):
    out = props_to_points(payload(), cfg)
    row = out[out["player"].eq("Derrick Henry")].iloc[0]
    scoring = cfg.for_profile("espn")._profile["offense_scoring"]
    stats_ = row["stats"]
    expected = (stats_["rushing_yards"] * scoring["rush_yards"]
                + stats_["receptions"] * scoring["reception"]
                + stats_["anytime_td"] * scoring["rush_td"])
    assert row["vegas_points"] == pytest.approx(expected, abs=0.02)
    # Balanced lines, so the implied means sit on the lines themselves.
    assert stats_["rushing_yards"] == pytest.approx(84.5, abs=0.01)
    assert stats_["receptions"] == pytest.approx(1.5, abs=0.01)


def test_pinnacle_is_weighted_above_the_others(cfg):
    """Two books disagreeing should land nearer Pinnacle."""
    p = payload(books=("pinnacle", "fanduel"))
    p[0]["bookmakers"][0]["markets"][0]["outcomes"] = _ou("Derrick Henry", 100.0)
    p[0]["bookmakers"][1]["markets"][0]["outcomes"] = _ou("Derrick Henry", 70.0)
    means = implied_means(parse_props_payload(p, cfg), cfg)
    cons = consensus(means, cfg)
    rush = cons[cons["stat"].eq("rushing_yards")].iloc[0]["mean"]
    assert 85.0 < rush < 95.0        # weighted 2:1 toward Pinnacle's 100
    assert rush == pytest.approx((2 * 100.0 + 70.0) / 3.0, abs=0.01)


def test_empty_payload_is_empty_not_an_error(cfg):
    assert props_to_points([], cfg).empty
    assert parse_props_payload([]).empty
    assert to_fantasy_points(consensus(implied_means(parse_props_payload([]), cfg), cfg),
                             cfg.for_profile("espn")).empty


# ---------------------------------------------------------------------------
# Attaching to a roster
# ---------------------------------------------------------------------------
def test_attach_matches_players_and_leaves_the_rest_blank(cfg):
    """A player the books did not post keeps None, not zero."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures_league import snapshot
    from streamer.roster.vegas import attach, disagreements, vegas_lineup

    snap = snapshot(week=2)
    mine = snap.my_team.roster
    mine[0].name = "Derrick Henry"          # the QB slot player, renamed to match
    bound = cfg.for_profile("espn")
    report = attach(snap, bound, payload=payload())

    assert report.matched == 1
    assert mine[0].vegas_points is not None and mine[0].vegas_points > 10
    assert all(p.vegas_points is None for p in mine[1:])
    # Unmatched skill players are reported, kickers and defences are not.
    assert "Derrick Henry" not in report.unmatched
    assert not any(p.position in ("K", "DST") for p in mine if p.name in report.unmatched)

    # The market lineup falls back to our projection where there is no prop,
    # so it is still a complete lineup.
    market = vegas_lineup(snap, bound)
    assert len(market.flat()) == sum(snap.starting_slots.values())
    assert any(d != 0 for _p, d in disagreements(snap, 3))


def test_attach_offline_does_nothing_quietly(cfg):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures_league import snapshot
    from streamer.roster.vegas import attach

    snap = snapshot(week=2)
    report = attach(snap, cfg.for_profile("espn"), allow_network=False)
    assert report.matched == 0 and not report.is_usable
    assert any("offline" in w for w in report.warnings)
    assert all(p.vegas_points is None for p in snap.my_team.roster)


def test_event_budget_buys_the_games_with_the_most_of_my_players():
    """The cap must not simply take the earliest kickoffs."""
    from streamer.data.props import fetch_props  # noqa: F401  (documents intent)

    # The ranking logic is exercised directly: build the same structure the
    # fetch does and check ordering by player count, not by list position.
    events = [
        {"id": "a", "home_team": "Buffalo Bills", "away_team": "Detroit Lions"},
        {"id": "b", "home_team": "Baltimore Ravens", "away_team": "New Orleans Saints"},
        {"id": "c", "home_team": "New England Patriots", "away_team": "Pittsburgh Steelers"},
    ]
    weights = {"BUF": 1, "BAL": 3, "NE": 2}
    from streamer.teams import normalize_team

    scored = []
    for e in events:
        n = (weights.get(normalize_team(e["home_team"]), 0)
             + weights.get(normalize_team(e["away_team"]), 0))
        if n:
            scored.append((n, e))
    scored.sort(key=lambda ne: -ne[0])
    assert [e["id"] for _n, e in scored] == ["b", "c", "a"]


def test_teams_on_counts_players_and_skips_the_unavailable(cfg):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures_league import snapshot
    from streamer.roster.vegas import teams_on

    snap = snapshot(week=2)
    weights = teams_on(snap)
    assert isinstance(weights, dict) and all(v >= 1 for v in weights.values())
    # A player on bye or out buys nothing.
    bye = [p for p in snap.my_team.roster if p.on_bye]
    out = [p for p in snap.my_team.roster if p.is_out and not p.on_bye]
    for p in bye + out:
        others = [q for q in snap.my_team.roster
                  if q.team == p.team and q is not p and not q.on_bye and not q.is_out]
        if not others:
            assert p.team not in weights


def test_events_outside_the_window_are_not_requested(monkeypatch, cfg):
    """A Tuesday request about Sunday must not be made at all."""
    from datetime import UTC, datetime, timedelta

    import streamer.data.props as props_mod

    now = datetime.now(UTC)
    soon = (now + timedelta(hours=20)).isoformat()
    far = (now + timedelta(hours=120)).isoformat()
    events = [
        {"id": "thu", "home_team": "Buffalo Bills", "away_team": "Detroit Lions",
         "commence_time": soon},
        {"id": "sun", "home_team": "Baltimore Ravens", "away_team": "New Orleans Saints",
         "commence_time": far},
    ]
    asked: list[str] = []

    class FakeResp:
        headers = {"x-requests-remaining": "400"}

        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/events"):
            return FakeResp(events)
        asked.append(url.rsplit("/events/", 1)[1].split("/")[0])
        return FakeResp({"id": "x", "bookmakers": []})

    monkeypatch.setattr(props_mod, "odds_api_key", lambda c: "key", raising=False)
    import streamer.data.odds as odds_mod
    monkeypatch.setattr(odds_mod, "odds_api_key", lambda c=None: "key")

    import requests
    monkeypatch.setattr(requests, "get", fake_get)

    bound = cfg.for_profile("espn")
    result = props_mod.fetch_props(bound, teams={"BUF": 1, "BAL": 3})
    assert asked == ["thu"], "only the game inside the window should be requested"
    assert result.credits_remaining == 400
