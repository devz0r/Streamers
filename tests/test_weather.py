"""Weather for kicker projections."""

from __future__ import annotations

import pytest


def test_a_game_with_no_recorded_wind_gets_its_stadiums_typical_wind_not_calm(cfg, monkeypatch):
    import pandas as pd

    from streamer.data import weather

    games = pd.DataFrame({
        "game_id": ["a", "b", "c", "d"], "game_type": "REG", "season": [2025, 2025, 2026, 2026],
        "week": [3, 4, 2, 3], "home_team": ["BUF", "BUF", "MIA", "BUF"], "is_dome": [0, 0, 0, 0],
        "wind": [12.0, 18.0, 4.0, float("nan")]})
    monkeypatch.setattr("streamer.data.nflverse.games_frame", lambda _c: games)
    slate = pd.DataFrame({"season": 2026, "week": 5, "home_team": ["BUF", "MIA", "NYG", "DAL"],
                          "is_dome": [0, 0, 0, 1], "wind": float("nan"), "temp": float("nan")})
    out = weather.resolve_weather(slate, 5, cfg).set_index("home_team")
    assert out.loc["BUF", "wind"] == 15.0 and out.loc["BUF", "weather_source"] == "typical"
    assert out.loc["MIA", "wind"] == 4.0
    assert out.loc["NYG", "wind"] == 0.0 and out.loc["NYG", "weather_source"] == "neutral"   # no history: neutral default
    assert out.loc["DAL", "wind"] == 0.0 and out.loc["DAL", "weather_source"] == "dome"
    again = weather.resolve_weather(out.reset_index(), 5, cfg).set_index("home_team")        # a second pass keeps the label
    assert again.loc["BUF", "weather_source"] == "typical"


def test_forecast_wind_blends_with_typical_by_how_far_off_the_game_is():
    from datetime import UTC, datetime

    from streamer.data import forecast_wind as fw

    cal = {"a": 1.0, "b": 0.8, "full_days": 2.0, "zero_days": 7.0}
    assert fw.lead_weight(1.0, cal) == 1.0 and fw.lead_weight(7.0, cal) == 0.0
    assert fw.lead_weight(4.5, cal) == pytest.approx(0.5)
    assert fw.blend(20.0, 8.0, 1.0, cal) == pytest.approx(1.0 + 0.8 * 20.0)          # inside two days: the forecast
    assert fw.blend(20.0, 8.0, 9.0, cal) == pytest.approx(8.0)                        # a week out: typical
    ko = fw.kickoff_utc("2026-10-18", "13:00")                                       # 1 pm Eastern (EDT) = 17:00 UTC
    assert ko == datetime(2026, 10, 18, 17, 0, tzinfo=UTC)
    payload = {"hourly": {"time": [f"2026-10-18T{h:02d}:00" for h in range(15, 24)],
                          "wind_speed_10m": [5, 6, 10, 12, 14, 16, 9, 8, 7],
                          "wind_gusts_10m": [9, 10, 18, 20, 24, 26, 15, 12, 11]}}
    wind, gust = fw.kickoff_wind(payload, ko)                                        # 17:00-20:00 -> 10, 12, 14, 16
    assert wind == pytest.approx(13.0) and gust == pytest.approx(22.0)
    assert fw.kickoff_wind({"hourly": {"time": [], "wind_speed_10m": []}}, ko) is None
    assert fw.venue("BUF") == fw.STADIUMS["BUF"] and fw.venue("JAX", "Tottenham Hotspur Stadium")[0] > 50


def test_without_a_calibration_the_forecast_is_never_asked(cfg, monkeypatch):
    import pandas as pd

    from streamer.data import forecast_wind as fw
    from streamer.data import weather

    games = pd.DataFrame({"game_id": ["a"], "game_type": "REG", "season": [2025], "week": [3],
                          "home_team": ["BUF"], "is_dome": [0], "wind": [12.0]})
    monkeypatch.setattr("streamer.data.nflverse.games_frame", lambda _c: games)
    monkeypatch.setattr(fw, "load_calibration", lambda: {})
    monkeypatch.setattr(fw, "fetch", lambda *a, **k: pytest.fail("no calibration: no request"))
    slate = pd.DataFrame({"season": 2026, "week": 5, "home_team": ["BUF"], "is_dome": [0],
                          "wind": float("nan"), "temp": float("nan")})
    assert weather.resolve_weather(slate, 5, cfg)["weather_source"].iloc[0] == "typical"


def test_with_a_calibration_a_near_game_uses_the_forecast(cfg, monkeypatch):
    import pandas as pd

    from streamer.data import forecast_wind as fw
    from streamer.data import weather

    games = pd.DataFrame({"game_id": ["a", "b"], "game_type": "REG", "season": [2025, 2026], "week": [3, 5],
                          "home_team": ["BUF", "BUF"], "is_dome": [0, 0], "wind": [10.0, float("nan")],
                          "gameday": ["2025-09-21", "2026-10-18"], "gametime": ["13:00", "13:00"],
                          "stadium": ["Highmark Stadium"] * 2})
    monkeypatch.setattr("streamer.data.nflverse.games_frame", lambda _c: games)
    monkeypatch.setattr(fw, "load_calibration", lambda: {"a": 0.0, "b": 1.0, "full_days": 2.0, "zero_days": 7.0})
    monkeypatch.setattr(fw, "for_slate", lambda cfg, g, now=None: {"BUF": (22.0, 30.0, 0.5)})
    slate = pd.DataFrame({"season": 2026, "week": 5, "home_team": ["BUF", "MIA"], "is_dome": [0, 0],
                          "wind": float("nan"), "temp": float("nan")})
    out = weather.resolve_weather(slate, 5, cfg).set_index("home_team")
    assert out.loc["BUF", "wind"] == 22.0 and out.loc["BUF", "weather_source"] == "forecast"
    assert out.loc["MIA", "weather_source"] == "neutral"                              # no history, no forecast
