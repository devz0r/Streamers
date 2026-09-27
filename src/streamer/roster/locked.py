"""Scores that are already in.

Once a player's game kicks off his lineup spot is frozen, and once it is
final his score is a fact, not a forecast. Both matter for P(win): a
receiver who put up 28 on Thursday is 28 in every simulation, and a player
whose game has started cannot be swapped in or out.

Final scores come from nflverse, scored the way the league scores them:
skill players from the weekly player stats (full PPR, as both leagues are),
kickers and defences from play-by-play through the same scoring code the
streaming models are trained on. nflverse publishes a game's numbers some
hours after it ends; until they appear, a finished game's players stay
simulated and the panel says so.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ..config import Config
from ..league.model import LeagueSnapshot, PlayerRow

log = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
SKILL = ("QB", "RB", "WR", "TE")


def _now() -> datetime:
    return datetime.now(UTC)


def kickoff_utc(gameday: object, gametime: object) -> datetime | None:
    """nflverse schedules give the date and Eastern kickoff time."""
    try:
        local = datetime.strptime(f"{gameday} {str(gametime)[:5]}", "%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return None
    return local.replace(tzinfo=EASTERN).astimezone(UTC)


def _week_games(snapshot: LeagueSnapshot, cfg: Config) -> pd.DataFrame:
    from ..data.nflverse import games_frame

    games = games_frame(cfg)
    return games[(games["season"] == snapshot.season) & (games["week"] == snapshot.week)]


def _team_points(frame: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {}
    for r in frame.itertuples():
        out.setdefault(str(r.team), []).append((str(getattr(r, "player_name", "") or ""), float(r.fantasy_points)))
    return out


def _kicker_score(p: PlayerRow, rows: list[tuple[str, float]]) -> float:
    """A team's kicker line; if two kickers kicked, the one whose surname matches."""
    if len(rows) == 1:
        return rows[0][1]
    surname = p.name.split()[-1].lower() if p.name else ""
    for name, pts in rows:
        if surname and name.lower().endswith(surname):
            return pts
    return sum(pts for _n, pts in rows)


def lock_played(
    snapshot: LeagueSnapshot,
    cfg: Config,
    history: pd.DataFrame,
    now: datetime | None = None,
    games: pd.DataFrame | None = None,
    pbp: pd.DataFrame | None = None,
) -> list[str]:
    """Mark started games' players as locked and fill in final scores.

    Returns notes for the panel. Never raises: a missing feed leaves players
    simulated, which is what the page did before.
    """
    now = now or _now()
    notes: list[str] = []
    try:
        games = _week_games(snapshot, cfg) if games is None else games
    except Exception as exc:  # noqa: BLE001
        log.warning("no schedule, so nothing is locked: %s", exc)
        return notes
    kickoff: dict[str, datetime] = {}
    final: dict[str, str] = {}              # team -> game id, for finished games
    for g in games.itertuples():
        k = kickoff_utc(g.gameday, g.gametime)
        if k is not None:
            kickoff[g.team] = k
        if pd.notna(getattr(g, "team_score", None)):
            final[g.team] = g.game_id

    players = snapshot.all_players()
    for p in players:
        p.locked = bool(p.team and p.team in kickoff and now >= kickoff[p.team])
        p.actual_points = None
    if not final:
        return notes

    # Skill players: this week's nflverse rows. A finished game counts as
    # published once any player on either side has a row.
    week = history[(history["season"] == snapshot.season) & (history["week"] == snapshot.week)]
    scored = dict(zip(week["player_id"], week["fantasy_points_ppr"]))
    teams_with_stats = set(week["team"])
    published = {t for t in final if t in teams_with_stats}

    kick_pts: dict[str, list[tuple[str, float]]] = {}
    dst_pts: dict[str, list[tuple[str, float]]] = {}
    if any(p.position in ("K", "DST") and p.team in final for p in players):
        try:
            from ..actuals import dst_game_lines, kicker_game_lines
            from ..data.nflverse import load_pbp

            if pbp is None:
                pbp = load_pbp([snapshot.season], cfg)
            pbp = pbp[pbp["week"] == snapshot.week]
            kick_pts = _team_points(kicker_game_lines(pbp, cfg))
            dst_pts = _team_points(dst_game_lines(pbp, games, cfg))
        except Exception as exc:  # noqa: BLE001
            log.warning("kicker/defence finals unavailable: %s", exc)

    waiting: set[str] = set()
    for p in players:
        if not p.team or p.team not in final:
            continue
        if p.position in SKILL:
            if p.team in published:
                p.actual_points = round(float(scored.get(p.nfl_id, 0.0) or 0.0), 2) if p.nfl_id else None
            else:
                waiting.add(p.team)
        elif p.position == "K" and p.team in kick_pts:
            p.actual_points = round(_kicker_score(p, kick_pts[p.team]), 2)
        elif p.position == "DST" and p.team in dst_pts:
            p.actual_points = round(dst_pts[p.team][0][1], 2)
        elif p.position in ("K", "DST"):
            waiting.add(p.team)

    mine = [p for t in snapshot.teams for p in t.roster
            if (t.is_mine or (snapshot.matchup and t.team_id == snapshot.matchup.opponent_team_id))]
    done = [p for p in mine if p.actual_points is not None and p.starting]
    if done:
        notes.append(f"{len(done)} starter(s) in this matchup have played; their final scores "
                     "are locked into P(win)")
    stuck = sorted({p.team for p in mine if p.team in waiting and p.starting})
    if stuck:
        notes.append(f"games involving {', '.join(stuck)} are final but nflverse has not published "
                     "their stats yet, so those players are still simulated")
    return notes
