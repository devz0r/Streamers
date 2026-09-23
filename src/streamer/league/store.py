"""Where league snapshots live, and how the sync command produces them."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from ..config import Config, get_config
from ..data.nflverse import games_frame
from .model import LeagueSnapshot

log = logging.getLogger(__name__)


def snapshot_dir(cfg: Config) -> Path:
    p = cfg.data_dir / "leagues" / cfg.profile
    p.mkdir(parents=True, exist_ok=True)
    return p


def snapshot_path(cfg: Config, week: int) -> Path:
    return snapshot_dir(cfg) / f"week_{week}.json"


def latest_snapshot_path(cfg: Config) -> Path | None:
    files = sorted(snapshot_dir(cfg).glob("week_*.json"),
                   key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1)
    return files[-1] if files else None


def load_snapshot(cfg: Config, week: int | None = None) -> LeagueSnapshot:
    """The snapshot for ``week``, or the newest one if ``week`` is None."""
    path = snapshot_path(cfg, week) if week is not None else latest_snapshot_path(cfg)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"no league snapshot for the {cfg.profile_label} profile"
            + (f" for week {week}" if week is not None else "")
            + " -- run `streamer sync` first"
        )
    return LeagueSnapshot.load(path)


def status_path(cfg: Config) -> Path:
    return snapshot_dir(cfg) / "sync_status.json"


def record_sync(cfg: Config, week: int, ok: bool, error: str = "") -> Path:
    """Write what the last sync attempt did, so a failure is visible.

    A league that fails to sync used to vanish silently: no snapshot, no
    panel, and the reason buried in a workflow log that scrolls past. The
    status file is committed alongside the snapshots and read by the page.
    """
    path = status_path(cfg)
    path.write_text(json.dumps({
        "profile": cfg.profile,
        "platform": platform_for(cfg),
        "week": week,
        "ok": bool(ok),
        "error": error,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")
    return path


def read_status(cfg: Config) -> dict | None:
    path = status_path(cfg)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def platform_for(cfg: Config) -> str:
    """Which platform a profile syncs from, from ``profiles.<name>.league_sync``."""
    block = cfg._profile.get("league_sync") or {}
    platform = str(block.get("platform") or cfg.profile).lower()
    if platform not in ("espn", "yahoo"):
        raise ValueError(f"profile {cfg.profile!r} has no supported league_sync.platform")
    return platform


def apply_byes(snapshot: LeagueSnapshot, cfg: Config) -> LeagueSnapshot:
    """Mark players whose NFL team does not play this week.

    Derived from the nflverse schedule for both platforms, so bye handling is
    identical regardless of what each platform does or does not expose.
    """
    games = games_frame(cfg)
    playing = set(games[(games["season"] == snapshot.season) & (games["week"] == snapshot.week)]["team"])
    if not playing:
        return snapshot
    for player in snapshot.all_players():
        if player.team and player.team not in playing:
            player.on_bye = True
    return snapshot


def sync(cfg: Config | None = None, week: int | None = None, season: int | None = None) -> Path:
    """Pull the profile's league into a snapshot file and return its path."""
    cfg = cfg or get_config()
    season = season or cfg.current_season
    if week is None:
        raise ValueError("week is required")

    from ..data.odds import _load_dotenv  # reuse the .env loader

    _load_dotenv(cfg.root)

    platform = platform_for(cfg)
    if platform == "espn":
        from .espn import fetch_snapshot

        snap = fetch_snapshot(season, week, cfg.profile)
    else:
        import os

        if os.environ.get("YAHOO_COOKIE", "").strip():
            # Yahoo stopped provisioning the developer API, so the website --
            # read with the user's own browser session -- is the working route.
            from .yahoo_web import fetch_snapshot as fetch_web

            snap = fetch_web(season, week, cfg.profile)
        else:
            from .yahoo import fetch_snapshot

            snap = fetch_snapshot(season, week, cfg.profile,
                                  oauth_path=cfg.data_dir / ".yahoo_oauth.json")

    snap = apply_byes(snap, cfg)
    # Snapshots are committed by the workflow, and Pages needs a public repo:
    # the other managers' screen names have no use downstream, so they are not
    # written to disk. Team names are what the page and the CLI show.
    for team in snap.teams:
        team.owner = ""
    path = snapshot_path(cfg, week)
    snap.save(path)
    log.info("synced %s league %s -> %s", platform, snap.league_id, path)
    return path
