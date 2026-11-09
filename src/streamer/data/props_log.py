"""Every book's raw prop line and price, kept week by week.

The projection log keeps one blended market number per player. To grade a bet
(does our side of a line win more often than its price needs?) or to see which
book sat off the rest, the lines and prices themselves are needed, and only
weeks logged from now on can ever be graded. ``results/prop_lines.parquet``
(shared by both leagues, the props being pulled once for both) holds, per
(season, week, game, book, stat, player), the first pull and the latest pull
before his game started: line, over/under or yes price, and when. Rows for a
player whose game has begun are not touched, so a live price never replaces
the closing one. DECISIONS.md, "Is there a betting edge?".
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)

KEY = ["season", "week", "event_id", "bookmaker", "stat", "player"]
COLUMNS = KEY + ["market", "kind", "line", "over_price", "under_price", "yes_price", "logged_at"]


def path(cfg: Config):
    return cfg.path("results") / "prop_lines.parquet"


def quote_times(frame: pd.DataFrame, now: str | None = None) -> pd.Series:
    """When each row's price was read from its feed (``quoted_at``), as ISO
    text; this run's time where the feed did not say, and never later than it."""
    stamp = pd.Timestamp(now) if now else pd.Timestamp.now(tz="UTC")
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if "quoted_at" not in frame:
        return pd.Series([stamp.isoformat()] * len(frame), index=frame.index)
    at = pd.to_datetime(frame["quoted_at"], utc=True, errors="coerce").fillna(stamp).clip(upper=stamp)
    return at.map(lambda t: t.isoformat())


def record(frame: pd.DataFrame, cfg: Config, season: int, week: int, started: set[str] | None = None,
           now: str | None = None) -> int:
    """Add this pull to the log; returns the rows now held for the week.
    ``started``: normalised names of players whose game has begun (skipped)."""
    from ..roster.players import normalize_name

    if frame is None or frame.empty:
        return 0
    new = frame.copy()
    if started:
        new = new[~new["player"].map(normalize_name).isin(started)]
    if new.empty:
        return 0
    new["season"], new["week"] = int(season), int(week)
    new["logged_at"] = quote_times(new, now)
    new = new.reindex(columns=COLUMNS)
    p = path(cfg)
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=COLUMNS)
    both = pd.concat([old, new], ignore_index=True).sort_values("logged_at")
    # The first and the latest pull of each line; a repeat of the same prices adds nothing.
    first = both.groupby(KEY, dropna=False, sort=False).head(1)
    last = both.groupby(KEY, dropna=False, sort=False).tail(1)
    out = pd.concat([first, last]).drop_duplicates(KEY + ["logged_at"]).sort_values(KEY + ["logged_at"])
    # The same price at the first and the latest pull is a single row.
    out = out.reset_index(drop=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False, compression="zstd")
    return int(((out["season"] == season) & (out["week"] == week)).sum())
