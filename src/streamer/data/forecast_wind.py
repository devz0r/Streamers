"""Kickoff-hour wind from a weather forecast, on the scale the kicker model trained on.

nflverse records the wind at each played game, and has none for games still to
come; production fills those with the home stadium's typical wind
(:func:`weather.typical_wind`). A forecast does better in a windy week, but a
forecast model's 10 m wind is a different measurement from the recorded one, so
it is used only after it has been calibrated against recorded wind
(``scripts/fit_wind_forecast.py`` writes ``wind_forecast.json``): recorded =
``a + b x forecast``, blended with the stadium's typical wind by a weight that
falls as the game gets further away. With no calibration file the module does
nothing, and the typical wind stands.

Source: Open-Meteo's free forecast API (no key). Domes and closed roofs are
never asked. DECISIONS.md, "Kicker wind forecast".
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from ..config import Config

log = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
CALIBRATION = Path(__file__).with_name("wind_forecast.json")
#: Hours of the game averaged, from kickoff.
GAME_HOURS = 3

#: Home team -> (latitude, longitude) of its stadium.
STADIUMS: dict[str, tuple[float, float]] = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4009), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "LA": (33.9535, -118.3392), "LAC": (33.9535, -118.3392),
    "LV": (36.0909, -115.1833), "MIA": (25.9580, -80.2389), "MIN": (44.9735, -93.2575),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316), "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9078, -76.8645),
}
#: Neutral-site venues by stadium name (a substring of nflverse's ``stadium``).
NEUTRAL_SITES: dict[str, tuple[float, float]] = {
    "Tottenham": (51.6043, -0.0663), "Wembley": (51.5560, -0.2796), "Allianz": (48.2188, 11.6247),
    "Deutsche Bank": (50.0686, 8.6455), "Azteca": (19.3029, -99.1505), "Maracan": (-22.9122, -43.2302),
    "Santiago Bernab": (40.4531, -3.6883), "Croke": (53.3609, -6.2512), "Melbourne": (-37.8200, 144.9835),
    "MCG": (-37.8200, 144.9835),
}


def kickoff_utc(gameday: str, gametime: str | None) -> datetime:
    """nflverse gives the game day and the Eastern kickoff time."""
    hh, mm = (str(gametime or "13:00").split(":") + ["0"])[:2]
    local = datetime.fromisoformat(f"{gameday}T{int(hh):02d}:{int(mm):02d}:00").replace(tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(UTC)


def venue(home_team: str, stadium: str | None = None) -> tuple[float, float] | None:
    for key, where in NEUTRAL_SITES.items():
        if stadium and key.lower() in str(stadium).lower():
            return where
    return STADIUMS.get(home_team)


def kickoff_wind(payload: dict, kickoff: datetime, hours: int = GAME_HOURS) -> tuple[float, float] | None:
    """(mean wind, mean gust) in mph over ``hours`` from kickoff, from an
    Open-Meteo hourly payload (UTC)."""
    hourly = (payload or {}).get("hourly") or {}
    times, wind, gust = hourly.get("time") or [], hourly.get("wind_speed_10m") or [], hourly.get("wind_gusts_10m") or []
    want = {(kickoff + timedelta(hours=h)).strftime("%Y-%m-%dT%H:00") for h in range(hours + 1)}
    pick = [i for i, t in enumerate(times) if t in want and i < len(wind) and wind[i] is not None]
    if not pick:
        return None
    w = float(np.mean([wind[i] for i in pick]))
    g = [gust[i] for i in pick if i < len(gust) and gust[i] is not None]
    return w, float(np.mean(g)) if g else w


def fetch(lat: float, lon: float, start: str, end: str, archive: bool = False, session=None) -> dict:
    """Hourly wind for a place between two dates (UTC, ISO)."""
    import requests

    r = (session or requests).get(
        ARCHIVE_URL if archive else FORECAST_URL, timeout=30,
        params={"latitude": lat, "longitude": lon, "hourly": "wind_speed_10m,wind_gusts_10m",
                "wind_speed_unit": "mph", "timezone": "UTC", "start_date": start, "end_date": end})
    r.raise_for_status()
    return r.json()


def load_calibration() -> dict:
    try:
        return json.loads(CALIBRATION.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def lead_weight(days: float, cal: dict) -> float:
    """How much of the calibrated forecast (against the typical wind) a game
    ``days`` away gets: full inside ``full_days``, falling linearly to zero at
    ``zero_days`` (a judgment: the archive holds only near-term forecasts)."""
    full, zero = float(cal.get("full_days", 2.0)), float(cal.get("zero_days", 7.0))
    return float(np.clip((zero - days) / max(zero - full, 1e-9), 0.0, 1.0))


def blend(forecast: float, typical: float, days: float, cal: dict) -> float:
    """Recorded-scale wind for a game: calibrated forecast against typical."""
    w = lead_weight(days, cal)
    calibrated = float(cal["a"]) + float(cal["b"]) * float(forecast)
    return max(w * calibrated + (1.0 - w) * float(typical), 0.0)


def for_slate(cfg: Config, games, now: datetime | None = None) -> dict[str, tuple[float, float, float]]:
    """Home team -> (forecast wind, gust, days ahead) for each outdoor game in
    ``games`` (rows with ``home_team``, ``gameday``, ``gametime``, ``stadium``,
    ``is_dome``). Empty with no calibration, or on any failure."""
    cal = load_calibration()
    if not cal or games is None or len(games) == 0:
        return {}
    now = now or datetime.now(UTC)
    out = {}
    for g in games.drop_duplicates("home_team").itertuples():
        if int(getattr(g, "is_dome", 0) or 0):
            continue
        where = venue(g.home_team, getattr(g, "stadium", None))
        if where is None:
            continue
        ko = kickoff_utc(str(g.gameday), getattr(g, "gametime", None))
        days = (ko - now).total_seconds() / 86400.0
        if days > 15:
            continue                                 # beyond the forecast
        try:
            day = ko.strftime("%Y-%m-%d")
            payload = fetch(where[0], where[1], day, (ko + timedelta(hours=4)).strftime("%Y-%m-%d"))
            got = kickoff_wind(payload, ko)
        except Exception as exc:  # noqa: BLE001 - a refinement; the typical wind stands
            log.warning("wind forecast for %s unavailable: %s", g.home_team, type(exc).__name__)
            continue
        if got:
            out[g.home_team] = (got[0], got[1], max(days, 0.0))
    return out
