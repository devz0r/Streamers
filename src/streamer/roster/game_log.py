"""Each player's season so far, game by game, for the page's tap-a-name sheet.

nflverse's weekly stats already carry every game; this reshapes them into a
compact record per player the page embeds once (shared by both leagues):
``{"n": name, "t": team, "p": position, "g": [game, ...]}`` where a game is
``[week, opponent, PPR points, offensive snap %, "stat line"]``, ``[week,
"bye"]`` for his team's bye, or ``[week, "-"]`` when his team played and he
has no stat line (inactive, or never got on the field).

Players are matched to nflverse by the id projections already found
(``nfl_id``), falling back to name and team for those without one (kickers).
Defences have no weekly stat line in nflverse and are left out.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import Config
from ..league.model import LeagueSnapshot, PlayerRow

log = logging.getLogger(__name__)

POSITIONS = ("QB", "RB", "WR", "TE", "K")


def _n(row, col: str) -> int:
    v = row.get(col)
    return 0 if v is None or pd.isna(v) else int(v)


def line(pos: str, r) -> str:
    """The stat line a fantasy manager reads: yards and scores by position."""
    parts: list[str] = []
    td = _n(r, "rushing_tds") + _n(r, "receiving_tds")
    if pos == "QB":
        parts.append(f"{_n(r, 'completions')}/{_n(r, 'attempts')} {_n(r, 'passing_yards')} yds")
        if _n(r, "passing_tds"):
            parts.append(f"{_n(r, 'passing_tds')} TD")
        if _n(r, "passing_interceptions"):
            parts.append(f"{_n(r, 'passing_interceptions')} INT")
        if _n(r, "carries"):
            parts.append(f"{_n(r, 'carries')} car {_n(r, 'rushing_yards')} yds")
        if _n(r, "rushing_tds"):
            parts.append(f"{_n(r, 'rushing_tds')} rush TD")
    elif pos == "K":
        parts.append(f"FG {_n(r, 'fg_made')}/{_n(r, 'fg_att')}")
        if _n(r, "fg_long"):
            parts.append(f"long {_n(r, 'fg_long')}")
        parts.append(f"XP {_n(r, 'pat_made')}/{_n(r, 'pat_att')}")
    else:
        if _n(r, "carries"):
            parts.append(f"{_n(r, 'carries')} car {_n(r, 'rushing_yards')} yds")
        if _n(r, "targets") or _n(r, "receptions"):
            parts.append(f"{_n(r, 'receptions')} rec ({_n(r, 'targets')} tgt) {_n(r, 'receiving_yards')} yds")
        if td:
            parts.append(f"{td} TD")
    lost = _n(r, "rushing_fumbles_lost") + _n(r, "receiving_fumbles_lost") + _n(r, "sack_fumbles_lost")
    if lost:
        parts.append(f"{lost} fum lost")
    return ", ".join(parts)


def _frames(cfg: Config, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ..teams import normalize_team
    from .players import normalize_name
    from .snaps import load as load_snaps

    path = cfg.raw_dir / f"player_stats_{season}.parquet"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    stats = pd.read_parquet(path)
    stats = stats[stats["season_type"].eq("REG")].copy()
    stats["_team"] = stats["team"].map(normalize_team)
    stats["_name"] = stats["player_display_name"].map(normalize_name)
    try:
        snaps = load_snaps(cfg, season, allow_network=False)
    except Exception as exc:  # noqa: BLE001 - snap shares are a bonus column
        log.warning("snap shares for the game log skipped: %s", exc)
        snaps = pd.DataFrame()
    if not snaps.empty:
        stats = stats.merge(snaps, on=["week", "_team", "_name"], how="left", suffixes=("", "_s"))
    else:
        stats["offense_pct"] = float("nan")
    return stats, snaps


def build(snapshot: LeagueSnapshot, cfg: Config) -> dict[str, dict]:
    """``{key: record}`` for every skill player and kicker in the league
    (rostered or free); the key is his nflverse id, or ``name|team``."""
    from ..teams import normalize_team
    from .players import normalize_name

    season = int(snapshot.season)
    stats, _ = _frames(cfg, season)
    if stats.empty:
        return {}
    last = int(stats["week"].max())
    played = stats.groupby("_team")["week"].agg(set).to_dict()          # team -> weeks it played
    schedule = _schedule(cfg, season)                                     # team -> weeks it has a game
    by_id = {pid: g for pid, g in stats.groupby("player_id")}
    by_name = {k: g for k, g in stats[stats["position"].isin(POSITIONS)].groupby(["_name", "_team"])}
    out: dict[str, dict] = {}
    for p in snapshot.all_players():
        if p.position not in POSITIONS:
            continue
        team = normalize_team(p.team) if p.team else None
        g = by_id.get(str(p.nfl_id)) if p.nfl_id else None
        key = str(p.nfl_id) if g is not None else f"{normalize_name(p.name)}|{team}"
        if g is None:
            g = by_name.get((normalize_name(p.name), team))
        if key in out:
            continue
        rows = {int(r["week"]): r for _i, r in g.iterrows()} if g is not None else {}
        if not rows and p.position != "K":
            continue                                                    # a rookie who has not played: nothing to show
        games_by_team = schedule
        games = []
        for w in range(1, last + 1):
            r = rows.get(w)
            if r is not None:
                snap = None if pd.isna(r.get("offense_pct")) else round(float(r["offense_pct"]) * 100)
                games.append([w, r.get("opponent_team") or "", round(float(r["fantasy_points_ppr"]), 1), snap,
                              line(p.position, r)])
            elif team and w not in played.get(team, set()):
                # No stats for his team that week: a bye only if the schedule
                # has no game; otherwise it has not been played yet (the week
                # in progress -- Thursday's game is in, Sunday's are not).
                if games_by_team is None:
                    if w < last:
                        games.append([w, "bye"])
                elif w not in games_by_team.get(team, set()):
                    games.append([w, "bye"])
            else:
                games.append([w, "-"])
        out[key] = {"n": p.name, "t": p.team or "", "p": p.position, "g": games,
                    "v": _value(p)}
    return out


def _schedule(cfg: Config, season: int) -> dict[str, set[int]] | None:
    """Team -> the regular-season weeks it has a game, from nflverse's
    schedule; None when it is not on disk."""
    from ..teams import normalize_team

    path = cfg.raw_dir / "schedules.parquet"
    if not path.exists():
        return None
    s = pd.read_parquet(path, columns=["season", "game_type", "week", "home_team", "away_team"])
    s = s[(s["season"] == season) & (s["game_type"] == "REG")]
    out: dict[str, set[int]] = {}
    for r in s.itertuples():
        for t in (r.home_team, r.away_team):
            out.setdefault(normalize_team(t), set()).add(int(r.week))
    return out


def _value(p: PlayerRow) -> list | None:
    """[season value, last week's], for the sheet's header."""
    if p.ros_value is None:
        return None
    return [round(float(p.ros_value), 1), None if p.ros_prev is None else round(float(p.ros_prev), 1)]


def merge(*parts: dict | None) -> dict[str, dict]:
    """One record per player across leagues (the first league's wins)."""
    out: dict[str, dict] = {}
    for part in parts:
        for k, v in (part or {}).items():
            out.setdefault(k, v)
    return out
