"""Weather for kicker projections."""

from __future__ import annotations


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
