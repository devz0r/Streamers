"""Data-layer parsing: lines, weather, teams and the cache.

None of these touch the network -- the HTTP call is separated from the payload
parsing precisely so the parsing can be tested against fixtures.
"""

from __future__ import annotations

import pandas as pd
import pytest

from streamer.data.cache import cached_frame
from streamer.data.odds import parse_odds_api_payload, read_manual_lines
from streamer.data.weather import read_manual_weather, resolve_weather
from streamer.teams import CURRENT_TEAMS, normalize_team, require_team


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("KC", "KC"), ("kc", "KC"), ("  KC  ", "KC"),
        ("OAK", "LV"), ("SD", "LAC"), ("STL", "LA"), ("LAR", "LA"), ("WSH", "WAS"),
        ("Kansas City Chiefs", "KC"), ("Chiefs", "KC"), ("Green Bay Packers", "GB"),
        ("Washington Commanders", "WAS"),
    ],
)
def test_normalize_team(raw, expected):
    assert normalize_team(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "Nottingham Forest", float("nan")])
def test_normalize_team_rejects_junk(raw):
    assert normalize_team(raw) is None


def test_require_team_raises_on_junk():
    with pytest.raises(ValueError):
        require_team("Nottingham Forest")


def test_there_are_thirty_two_teams():
    assert len(CURRENT_TEAMS) == 32
    assert len(set(CURRENT_TEAMS)) == 32


# ---------------------------------------------------------------------------
# The Odds API
# ---------------------------------------------------------------------------
def _payload():
    return [
        {
            "home_team": "Kansas City Chiefs",
            "away_team": "Denver Broncos",
            "commence_time": "2026-09-14T00:20:00Z",
            "bookmakers": [
                {"markets": [
                    {"key": "spreads", "outcomes": [
                        {"name": "Kansas City Chiefs", "point": -3.0},
                        {"name": "Denver Broncos", "point": 3.0},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 42.5}, {"name": "Under", "point": 42.5},
                    ]},
                ]},
                {"markets": [
                    {"key": "spreads", "outcomes": [
                        {"name": "Kansas City Chiefs", "point": -3.5},
                        {"name": "Denver Broncos", "point": 3.5},
                    ]},
                    {"key": "totals", "outcomes": [{"name": "Over", "point": 43.5}]},
                ]},
            ],
        }
    ]


def test_odds_payload_is_converted_to_the_nflverse_convention():
    """A book prints the favourite as -3; nflverse stores home -3 as +3."""
    frame = parse_odds_api_payload(_payload())
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["home_team"] == "KC"
    assert row["away_team"] == "DEN"
    assert row["spread_line"] == pytest.approx(3.25)   # median of 3.0 and 3.5
    assert row["total_line"] == pytest.approx(43.0)    # median of 42.5 and 43.5


def test_odds_payload_skips_unrecognised_teams():
    payload = _payload()
    payload[0]["home_team"] = "Some Other Club"
    assert parse_odds_api_payload(payload).empty


def test_odds_payload_handles_empty_input():
    assert parse_odds_api_payload([]).empty
    assert parse_odds_api_payload(None).empty


# ---------------------------------------------------------------------------
# Manual CSVs
# ---------------------------------------------------------------------------
def test_manual_lines_flip_the_sportsbook_sign(tmp_cfg):
    path = tmp_cfg.data_dir / "lines_week_5.csv"
    path.write_text("home_team,away_team,spread,total\nKC,DEN,-3.5,42.5\nNYJ,BUF,6.5,38\n")
    frame = read_manual_lines(5, tmp_cfg)
    kc = frame[frame["home_team"] == "KC"].iloc[0]
    assert kc["spread_line"] == pytest.approx(3.5)   # KC favoured by 3.5
    nyj = frame[frame["home_team"] == "NYJ"].iloc[0]
    assert nyj["spread_line"] == pytest.approx(-6.5)  # NYJ a 6.5-point dog


def test_manual_lines_accept_short_column_names(tmp_cfg):
    path = tmp_cfg.data_dir / "lines_week_6.csv"
    path.write_text("home,away,spread,total\nChiefs,Broncos,-3,44\n")
    frame = read_manual_lines(6, tmp_cfg)
    assert frame.iloc[0]["home_team"] == "KC"
    assert frame.iloc[0]["away_team"] == "DEN"


def test_missing_manual_lines_file_is_not_an_error(tmp_cfg):
    assert read_manual_lines(99, tmp_cfg).empty


def test_manual_lines_reject_a_file_without_teams(tmp_cfg):
    path = tmp_cfg.data_dir / "lines_week_7.csv"
    path.write_text("spread,total\n-3,44\n")
    with pytest.raises(ValueError, match="missing column"):
        read_manual_lines(7, tmp_cfg)


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def test_dome_games_are_forced_to_neutral_weather(tmp_cfg):
    slate = pd.DataFrame({
        "home_team": ["MIN", "BUF"],
        "roof": ["dome", "outdoors"],
        "wind": [30.0, 18.0],
        "temp": [10.0, 25.0],
    })
    out = resolve_weather(slate, week=1, cfg=tmp_cfg)
    minnesota = out[out["home_team"] == "MIN"].iloc[0]
    assert minnesota["wind"] == 0.0
    assert minnesota["weather_source"] == "dome"
    assert minnesota["high_wind"] == 0
    buffalo = out[out["home_team"] == "BUF"].iloc[0]
    assert buffalo["wind"] == 18.0
    assert buffalo["high_wind"] == 1


def test_missing_weather_falls_back_to_neutral(tmp_cfg):
    slate = pd.DataFrame({"home_team": ["BUF"], "roof": ["outdoors"],
                          "wind": [float("nan")], "temp": [float("nan")]})
    out = resolve_weather(slate, week=1, cfg=tmp_cfg)
    assert out.iloc[0]["wind"] == tmp_cfg.weather["default_wind_mph"]
    assert out.iloc[0]["weather_source"] == "neutral"


def test_manual_weather_overrides_the_schedule(tmp_cfg):
    (tmp_cfg.data_dir / "weather_week_3.csv").write_text("home_team,wind,temp\nBUF,25,20\n")
    assert read_manual_weather(3, tmp_cfg).iloc[0]["wind"] == 25
    slate = pd.DataFrame({"home_team": ["BUF"], "roof": ["outdoors"],
                          "wind": [4.0], "temp": [50.0]})
    out = resolve_weather(slate, week=3, cfg=tmp_cfg)
    assert out.iloc[0]["wind"] == 25.0
    assert out.iloc[0]["weather_source"] == "manual"
    assert out.iloc[0]["high_wind"] == 1


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def test_cache_writes_then_reads_without_refetching(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return pd.DataFrame({"a": [1, 2, 3]})

    path = tmp_path / "thing.parquet"
    first = cached_frame(path, build)
    second = cached_frame(path, build)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)


def test_cache_falls_back_to_stale_data_when_the_fetch_fails(tmp_path):
    path = tmp_path / "thing.parquet"
    cached_frame(path, lambda: pd.DataFrame({"a": [1]}))

    def boom():
        raise RuntimeError("upstream is down")

    out = cached_frame(path, boom, refresh=True)
    assert out["a"].tolist() == [1]


def test_cache_reraises_when_there_is_nothing_to_fall_back_to(tmp_path):
    def boom():
        raise RuntimeError("upstream is down")

    with pytest.raises(RuntimeError):
        cached_frame(tmp_path / "missing.parquet", boom)
