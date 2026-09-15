"""Local parquet cache for every raw pull.

Backtests must be reproducible, so nothing is fetched twice: every remote pull
lands in ``data/raw/`` as parquet and is read from there on subsequent runs.
Set ``STREAMER_REFRESH=1`` (or pass ``refresh=True``) to force a re-fetch.

Completed seasons are immutable and cached forever. Anything carrying live
results -- the schedule with its scores and closing lines, the current
season's play-by-play and player stats -- is cached with a ``max_age_hours``
so a long-lived cache (the weekly job restores ``data/raw`` between runs)
cannot freeze the season in place. A cached copy is still served when the
re-fetch fails, so an upstream outage degrades to stale rather than fatal.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def refresh_requested() -> bool:
    return os.environ.get("STREAMER_REFRESH", "").strip().lower() in {"1", "true", "yes"}


def is_stale(path: Path, max_age_hours: float | None) -> bool:
    """True if ``path`` is older than ``max_age_hours``. Never stale if None."""
    if max_age_hours is None or not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) > max_age_hours * 3600.0


def cached_frame(
    path: Path,
    builder: Callable[[], pd.DataFrame],
    refresh: bool = False,
    allow_stale_on_error: bool = True,
    max_age_hours: float | None = None,
) -> pd.DataFrame:
    """Return ``builder()``'s frame, memoised to ``path`` as parquet.

    ``max_age_hours`` re-fetches a cached copy older than that, which is what
    keeps live feeds live. If the network call fails but a cached copy exists,
    the cached copy is returned with a warning -- the weekly job must never
    hard-fail on a transient upstream outage.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stale = is_stale(path, max_age_hours)
    if path.exists() and not stale and not (refresh or refresh_requested()):
        return pd.read_parquet(path)
    if stale:
        log.info("%s is older than %sh; re-fetching", path.name, max_age_hours)
    try:
        frame = builder()
    except Exception as exc:  # noqa: BLE001 - deliberate: fall back to cache
        if path.exists() and allow_stale_on_error:
            log.warning("fetch failed (%s); using cached %s", exc, path.name)
            return pd.read_parquet(path)
        raise
    frame.to_parquet(path, index=False)
    return frame


def cache_path(root: Path, name: str) -> Path:
    return root / f"{name}.parquet"
