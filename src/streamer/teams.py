"""Team abbreviation normalisation.

nflverse data spans franchise relocations and rebrands, so the same club shows
up as ``OAK``/``LV``, ``SD``/``LAC``, ``STL``/``LA``/``LAR`` and ``WSH``/``WAS``
depending on the season and the feed. Every join key in this project is passed
through :func:`normalize_team` first.
"""

from __future__ import annotations

import pandas as pd

#: Historic / feed-specific aliases mapped to the modern nflverse abbreviation.
TEAM_ALIASES: dict[str, str] = {
    "OAK": "LV",
    "SD": "LAC",
    "SL": "LA",
    "STL": "LA",
    "LAR": "LA",
    "RAM": "LA",
    "WSH": "WAS",
    "WFT": "WAS",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "JAC": "JAX",
    "GNB": "GB",
    "KAN": "KC",
    "NOR": "NO",
    "NWE": "NE",
    "SFO": "SF",
    "TAM": "TB",
    "LVR": "LV",
}

#: The 32 current franchises, in nflverse abbreviation form.
CURRENT_TEAMS: tuple[str, ...] = (
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
)

#: Full names, used for display and for matching hand-entered CSVs.
TEAM_NAMES: dict[str, str] = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos",
    "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots",
    "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def normalize_team(value: object) -> str | None:
    """Return the canonical nflverse abbreviation for ``value``.

    Accepts abbreviations in any case, with surrounding whitespace, or a full
    team name / nickname ("Rams", "Los Angeles Rams"). Returns ``None`` for
    anything unrecognisable so callers can decide how to handle it.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    upper = TEAM_ALIASES.get(upper, upper)
    if upper in CURRENT_TEAMS:
        return upper
    return _match_by_name(text)


def _match_by_name(text: str) -> str | None:
    """Best-effort match of a full name / nickname to an abbreviation."""
    lowered = " ".join(text.lower().split())
    for abbr, name in TEAM_NAMES.items():
        if lowered == name.lower():
            return abbr
    # Nickname match ("commanders", "washington"), longest token first so that
    # "giants" beats "new york".
    for abbr, name in sorted(TEAM_NAMES.items(), key=lambda kv: -len(kv[1])):
        parts = name.lower().split()
        nickname = parts[-1]
        city = " ".join(parts[:-1])
        if lowered == nickname or lowered == city:
            return abbr
    return None


def normalize_team_series(series: pd.Series) -> pd.Series:
    """Vectorised :func:`normalize_team` over a pandas Series."""
    return series.map(normalize_team)


def require_team(value: object) -> str:
    """:func:`normalize_team` that raises instead of returning ``None``."""
    team = normalize_team(value)
    if team is None:
        raise ValueError(f"unrecognised NFL team: {value!r}")
    return team
