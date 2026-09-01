"""Local parquet cache for every raw pull.

Backtests must be reproducible, so nothing is fetched twice: every remote pull
lands in ``data/raw/`` as parquet and is read from there on subsequent runs.
Set ``STREAMER_REFRESH=1`` (or pass ``refresh=True``) to force a re-fetch.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import pandas as pd

log = logging.getLogger(__name__)


def refresh_requested() -> bool:
    return os.environ.get("STREAMER_REFRESH", "").strip().lower() in {"1", "true", "yes"}


def cached_frame(
    path: Path,
    builder: Callable[[], pd.DataFrame],
    refresh: bool = False,
    allow_stale_on_error: bool = True,
) -> pd.DataFrame:
    """Return ``builder()``'s frame, memoised to ``path`` as parquet.

    If the network call fails but a cached copy exists, the cached copy is
    returned with a warning -- the weekly job must never hard-fail on a
    transient upstream outage.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not (refresh or refresh_requested()):
        return pd.read_parquet(path)
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
