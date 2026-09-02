"""Where league snapshots live, and how the sync command produces them."""

from __future__ import annotations

import logging
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
        from .yahoo import fetch_snapshot

        snap = fetch_snapshot(season, week, cfg.profile, oauth_path=cfg.data_dir / ".yahoo_oauth.json")

    snap = apply_byes(snap, cfg)
    path = snapshot_path(cfg, week)
    snap.save(path)
    log.info("synced %s league %s -> %s", platform, snap.league_id, path)
    return path
