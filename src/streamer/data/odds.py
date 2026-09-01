"""Betting lines: spread, total and derived implied team totals.

Lines are the backbone of every projection, so this module never hard-fails.
It walks the configured fallback chain (``odds.fallback_order`` in
``config.yaml``) and reports which source actually supplied each game so the
published page can flag stale or manually-entered numbers.

Sources, in default order:

``api``
    The Odds API v4 (free tier). Needs ``ODDS_API_KEY`` in the environment or
    ``.env``. Consensus = median across the returned books.
``csv``
    ``data/lines_week_N.csv`` -- hand-entered from a Mac. Columns:
    ``home_team,away_team,spread,total`` where ``spread`` is the **home**
    team's line (negative = home favoured, the way a sportsbook prints it).
``schedule``
    nflverse schedule ``spread_line`` / ``total_line`` (closing lines; present
    for upcoming games once books post them).
``last``
    The most recent cached pull for this week, however it was obtained.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, get_config
from ..teams import normalize_team
from .nflverse import games_frame

log = logging.getLogger(__name__)

#: Columns of the canonical lines frame produced by :func:`get_lines`.
LINE_COLUMNS = ("game_id", "season", "week", "home_team", "away_team",
                "spread_line", "total_line", "line_source")


@dataclass
class LinesResult:
    """Lines for one week plus provenance for the published page."""

    frame: pd.DataFrame
    sources: dict[str, int]
    fetched_at: datetime
    warnings: list[str]
    #: Games on the slate with no spread or total from any source.
    missing: int = 0

    #: Sources that carry genuine market numbers rather than a stopgap. The
    #: nflverse schedule ships the closing spread and total, so a slate priced
    #: from it is properly priced -- just not freshly pulled.
    MARKET_SOURCES = ("api", "csv", "schedule")

    @property
    def primary_source(self) -> str:
        if not self.sources:
            return "none"
        return max(self.sources.items(), key=lambda kv: kv[1])[0]

    @property
    def status(self) -> str:
        """``live``, ``fallback`` or ``incomplete``.

        Only ``incomplete`` is a problem worth alarming about. ``fallback``
        means every game is priced, just not from a fresh API pull -- which is
        the normal state without an API key, and perfectly usable.
        """
        if self.missing or not self.sources:
            return "incomplete"
        if set(self.sources) <= {"api"}:
            return "live"
        return "fallback"

    @property
    def is_degraded(self) -> bool:
        """True when anything other than a live API pull supplied lines."""
        return self.status != "live"

    @property
    def is_usable(self) -> bool:
        """Every game on the slate has a real market spread and total."""
        return self.status in ("live", "fallback")

    #: Human-readable name per source, for the published page.
    SOURCE_LABELS = {
        "api": "live odds",
        "csv": "your manual CSV",
        "schedule": "nflverse closing lines",
        "last": "the last cached pull",
    }

    def source_phrase(self) -> str:
        """Just the sources, e.g. ``nflverse closing lines``."""
        if not self.sources:
            return "no source"
        names = [
            self.SOURCE_LABELS.get(src, src)
            for src, _n in sorted(self.sources.items(), key=lambda kv: -kv[1])
        ]
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:-1]) + f" and {names[-1]}"

    def describe(self) -> str:
        if not self.sources:
            return "no lines available"
        labels = {
            "api": "live odds",
            "csv": "your manual CSV",
            "schedule": "nflverse closing lines",
            "last": "last cached pull",
        }
        parts = [
            f"{n} from {labels.get(src, src)}"
            for src, n in sorted(self.sources.items(), key=lambda kv: -kv[1])
        ]
        text = ", ".join(parts)
        return text if not self.missing else f"{text}; {self.missing} game(s) unpriced"


def _load_dotenv(root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a pinned dependency
        return
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def odds_api_key(cfg: Config | None = None) -> str | None:
    cfg = cfg or get_config()
    _load_dotenv(cfg.root)
    key = os.environ.get("ODDS_API_KEY", "").strip()
    return key or None


# ---------------------------------------------------------------------------
# Source: The Odds API
# ---------------------------------------------------------------------------
def fetch_odds_api(cfg: Config | None = None, timeout: float = 20.0) -> pd.DataFrame:
    """Pull current spreads/totals from The Odds API.

    Returns a frame keyed by ``home_team``/``away_team`` with consensus
    (median) numbers. Raises on any transport or auth failure; callers handle
    the fallback.
    """
    import requests

    cfg = cfg or get_config()
    key = odds_api_key(cfg)
    if not key:
        raise RuntimeError("no ODDS_API_KEY configured")
    conf = cfg.odds
    url = f"{conf['api_base']}/sports/{conf['sport']}/odds"
    params = {
        "apiKey": key,
        "regions": conf["regions"],
        "markets": conf["markets"],
        "oddsFormat": conf["odds_format"],
    }
    if conf.get("bookmakers"):
        params["bookmakers"] = ",".join(conf["bookmakers"])
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return parse_odds_api_payload(resp.json())


def parse_odds_api_payload(payload: list[dict]) -> pd.DataFrame:
    """Convert The Odds API JSON into a consensus lines frame.

    Split out from the HTTP call so it can be unit-tested against a fixture.
    """
    rows = []
    for event in payload or []:
        home = normalize_team(event.get("home_team"))
        away = normalize_team(event.get("away_team"))
        if not home or not away:
            continue
        spreads: list[float] = []
        totals: list[float] = []
        for book in event.get("bookmakers", []) or []:
            for market in book.get("markets", []) or []:
                outcomes = market.get("outcomes", []) or []
                if market.get("key") == "spreads":
                    for outcome in outcomes:
                        if normalize_team(outcome.get("name")) == home:
                            point = outcome.get("point")
                            if point is not None:
                                # Book prints home -3.5 when home is favoured;
                                # nflverse convention is +3.5, so negate.
                                spreads.append(-float(point))
                elif market.get("key") == "totals":
                    for outcome in outcomes:
                        point = outcome.get("point")
                        if point is not None:
                            totals.append(float(point))
                            break
        if not spreads and not totals:
            continue
        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "spread_line": float(np.median(spreads)) if spreads else np.nan,
                "total_line": float(np.median(totals)) if totals else np.nan,
                "commence_time": event.get("commence_time"),
            }
        )
    return pd.DataFrame(rows, columns=["home_team", "away_team", "spread_line",
                                       "total_line", "commence_time"])


# ---------------------------------------------------------------------------
# Source: manual CSV
# ---------------------------------------------------------------------------
def manual_lines_path(week: int, cfg: Config | None = None) -> Path:
    cfg = cfg or get_config()
    return cfg.data_dir / f"lines_week_{week}.csv"


def read_manual_lines(week: int, cfg: Config | None = None) -> pd.DataFrame:
    """Read ``data/lines_week_N.csv``.

    ``spread`` is the home team's sportsbook line (negative = home favoured);
    it is flipped to the nflverse convention on the way in. A ``team``/``line``
    long format is also accepted for convenience.
    """
    cfg = cfg or get_config()
    path = manual_lines_path(week, cfg)
    if not path.exists():
        return pd.DataFrame(columns=["home_team", "away_team", "spread_line", "total_line"])
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {"home": "home_team", "away": "away_team",
              "spread": "spread_line", "total": "total_line"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    missing = {"home_team", "away_team"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    df = df[df["home_team"].notna() & df["away_team"].notna()].copy()
    if "spread_line" in df.columns:
        df["spread_line"] = -pd.to_numeric(df["spread_line"], errors="coerce")
    else:
        df["spread_line"] = np.nan
    if "total_line" in df.columns:
        df["total_line"] = pd.to_numeric(df["total_line"], errors="coerce")
    else:
        df["total_line"] = np.nan
    return df[["home_team", "away_team", "spread_line", "total_line"]]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def get_lines(
    season: int,
    week: int,
    cfg: Config | None = None,
    allow_network: bool = True,
) -> LinesResult:
    """Assemble lines for one week by walking the configured fallback chain."""
    cfg = cfg or get_config()
    games = games_frame(cfg)
    slate = games[
        (games["season"] == season) & (games["week"] == week) & (games["is_home"] == 1)
    ][["game_id", "season", "week", "home_team", "away_team", "spread_line", "total_line"]].copy()
    if slate.empty:
        raise ValueError(f"no scheduled games for {season} week {week}")

    slate = slate.rename(columns={"spread_line": "schedule_spread",
                                  "total_line": "schedule_total"})
    slate["spread_line"] = np.nan
    slate["total_line"] = np.nan
    slate["line_source"] = None

    warnings: list[str] = []
    cache_path = cfg.raw_dir / f"lines_{season}_wk{week}.parquet"

    for source in cfg.odds["fallback_order"]:
        if not slate["spread_line"].isna().any() and not slate["total_line"].isna().any():
            break
        try:
            supplied = _apply_source(source, slate, week, season, cfg, cache_path, allow_network)
        except Exception as exc:  # noqa: BLE001 - any source may fail; keep walking
            warnings.append(f"{source} lines unavailable: {exc}")
            log.warning("lines source %r failed: %s", source, exc)
            continue
        if supplied == 0 and source == "api":
            warnings.append("The Odds API returned no usable lines")

    still_missing = slate["spread_line"].isna() | slate["total_line"].isna()
    if still_missing.any():
        names = ", ".join(
            f"{r.away_team}@{r.home_team}" for r in slate[still_missing].itertuples()
        )
        warnings.append(f"no lines for: {names}")

    slate["team_implied_home"] = slate["total_line"] / 2.0 + slate["spread_line"] / 2.0
    slate["team_implied_away"] = slate["total_line"] / 2.0 - slate["spread_line"] / 2.0

    have = slate[slate["line_source"].notna()]
    if not have.empty:
        have[list(LINE_COLUMNS)].to_parquet(cache_path, index=False)

    sources = slate["line_source"].dropna().value_counts().to_dict()
    return LinesResult(
        frame=slate.reset_index(drop=True),
        sources={str(k): int(v) for k, v in sources.items()},
        fetched_at=datetime.now(UTC),
        warnings=warnings,
        missing=int(still_missing.sum()),
    )


def _apply_source(
    source: str,
    slate: pd.DataFrame,
    week: int,
    season: int,
    cfg: Config,
    cache_path: Path,
    allow_network: bool,
) -> int:
    """Fill missing spread/total rows in ``slate`` from ``source``; return count."""
    if source == "api":
        if not allow_network:
            raise RuntimeError("--offline was set, so the API was not called")
        incoming = fetch_odds_api(cfg)
    elif source == "csv":
        incoming = read_manual_lines(week, cfg)
    elif source == "schedule":
        incoming = slate[["home_team", "away_team"]].copy()
        incoming["spread_line"] = slate["schedule_spread"].to_numpy()
        incoming["total_line"] = slate["schedule_total"].to_numpy()
    elif source == "last":
        if not cache_path.exists():
            raise FileNotFoundError(cache_path.name)
        incoming = pd.read_parquet(cache_path)
    else:
        raise ValueError(f"unknown lines source: {source!r}")

    if incoming is None or incoming.empty:
        return 0

    lookup = {
        (r.home_team, r.away_team): (r.spread_line, r.total_line)
        for r in incoming.itertuples()
    }
    filled = 0
    for idx, row in slate.iterrows():
        pair = lookup.get((row["home_team"], row["away_team"]))
        if pair is None:
            continue
        spread, total = pair
        touched = False
        if pd.isna(row["spread_line"]) and spread is not None and not pd.isna(spread):
            slate.at[idx, "spread_line"] = float(spread)
            touched = True
        if pd.isna(row["total_line"]) and total is not None and not pd.isna(total):
            slate.at[idx, "total_line"] = float(total)
            touched = True
        if touched:
            slate.at[idx, "line_source"] = source
            filled += 1
    return filled


def lines_to_team_rows(result: LinesResult) -> pd.DataFrame:
    """Explode a :class:`LinesResult` into one row per team with implied totals."""
    frame = result.frame
    home = frame[["game_id", "season", "week", "home_team", "away_team",
                  "spread_line", "total_line", "line_source"]].copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["is_home"] = 1
    home["team_spread"] = home["spread_line"]

    away = home.copy()
    away["team"] = frame["away_team"].to_numpy()
    away["opponent"] = frame["home_team"].to_numpy()
    away["is_home"] = 0
    away["team_spread"] = -home["spread_line"].to_numpy()

    out = pd.concat([home, away], ignore_index=True)
    out["team_implied_total"] = out["total_line"] / 2.0 + out["team_spread"] / 2.0
    out["opp_implied_total"] = out["total_line"] / 2.0 - out["team_spread"] / 2.0
    return out.drop(columns=["home_team", "away_team"])
