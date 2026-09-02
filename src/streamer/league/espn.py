"""ESPN Fantasy Football adapter.

Uses ``espn-api`` (``espn_api.football``). Private leagues need two cookies
from a logged-in browser session, ``espn_s2`` and ``SWID``; public leagues
need neither. Both are read from the environment (``ESPN_S2``, ``ESPN_SWID``)
alongside ``ESPN_LEAGUE_ID`` and, optionally, ``ESPN_TEAM_ID``.

The network call is confined to :func:`fetch_snapshot`. Everything that turns
ESPN's objects into a :class:`~streamer.league.model.LeagueSnapshot` is a pure
function over plain attributes, so it is unit-tested against stand-ins without
touching ESPN at all -- which is also the only way it *can* be tested from an
environment that cannot reach ESPN.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

from ..teams import normalize_team
from .model import (
    LeagueSnapshot,
    Matchup,
    PlayerRow,
    TeamRow,
    canonical_position,
    canonical_slot,
    canonical_status,
)

log = logging.getLogger(__name__)

PLATFORM = "espn"


def credentials() -> dict[str, str | None]:
    """ESPN settings from the environment (``.env`` is loaded by the caller)."""
    return {
        "league_id": os.environ.get("ESPN_LEAGUE_ID", "").strip() or None,
        "espn_s2": os.environ.get("ESPN_S2", "").strip() or None,
        "swid": os.environ.get("ESPN_SWID", "").strip() or None,
        "team_id": os.environ.get("ESPN_TEAM_ID", "").strip() or None,
    }


# ---------------------------------------------------------------------------
# Pure normalisation over espn-api objects (duck-typed, so stand-ins work)
# ---------------------------------------------------------------------------
def _attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def normalize_player(player: Any, week: int, box: bool = False) -> PlayerRow:
    """Turn an ``espn_api`` ``Player``/``BoxPlayer`` into a :class:`PlayerRow`.

    ``BoxPlayer`` (from ``box_scores``) carries ``projected_points`` for the
    week directly; a roster/free-agent ``Player`` carries a ``stats`` dict
    keyed by scoring period, from which the week's projection is read.
    """
    stats = _attr(player, "stats", {}) or {}
    week_stats = stats.get(week, {}) if isinstance(stats, dict) else {}

    if box and _attr(player, "projected_points") is not None:
        platform_projection = float(_attr(player, "projected_points") or 0.0)
    else:
        raw = week_stats.get("projected_points") if isinstance(week_stats, dict) else None
        platform_projection = float(raw) if raw is not None else None

    season_avg = _attr(player, "avg_points")
    slot = canonical_slot(_attr(player, "slot_position") or _attr(player, "lineupSlot"))
    eligible = [canonical_slot(s) for s in (_attr(player, "eligibleSlots") or [])]
    eligible = [s for s in eligible if s in ("QB", "RB", "WR", "TE", "K", "DST",
                                             "FLEX", "WR/RB", "WR/TE", "SUPERFLEX")]

    return PlayerRow(
        player_id=str(_attr(player, "playerId")),
        name=str(_attr(player, "name", "")),
        position=canonical_position(_attr(player, "position")),
        team=normalize_team(_attr(player, "proTeam")),
        status=canonical_status(_attr(player, "injuryStatus")),
        slot=slot,
        eligible_slots=sorted(set(eligible)),
        on_bye=bool(_attr(player, "on_bye_week", False)),
        platform_projection=platform_projection,
        season_avg=float(season_avg) if season_avg is not None else None,
        percent_owned=(
            float(_attr(player, "percent_owned"))
            if _attr(player, "percent_owned") not in (None, -1) else None
        ),
    )


def normalize_slots(position_slot_counts: dict[str, int]) -> tuple[dict[str, int], int]:
    """ESPN's ``settings.position_slot_counts`` -> (starting slots, bench size)."""
    slots: dict[str, int] = {}
    bench = 0
    for label, count in (position_slot_counts or {}).items():
        n = int(count or 0)
        if n <= 0:
            continue
        canon = canonical_slot(label)
        if canon in ("BN",):
            bench += n
        elif canon in ("IR", "NA", ""):
            continue
        else:
            slots[canon] = slots.get(canon, 0) + n
    return slots, bench


def identify_my_team(teams: list[Any], swid: str | None, team_id: str | None) -> str | None:
    """Work out which team is the user's.

    Preference: an explicit ``ESPN_TEAM_ID``; otherwise match the SWID cookie
    against each team's owner ids, which ESPN exposes on ``team.owners``.
    """
    if team_id:
        return str(team_id)
    if swid:
        wanted = swid.strip().upper().strip("{}")
        for team in teams:
            for owner in _attr(team, "owners", []) or []:
                oid = owner.get("id") if isinstance(owner, dict) else str(owner)
                if oid and str(oid).upper().strip("{}") == wanted:
                    return str(_attr(team, "team_id"))
    return None


def build_snapshot(
    league: Any,
    week: int,
    profile: str,
    my_team_id: str | None,
    box_scores: list[Any] | None = None,
    free_agents: list[Any] | None = None,
) -> LeagueSnapshot:
    """Assemble a snapshot from already-fetched ``espn_api`` objects."""
    slots, bench = normalize_slots(_attr(_attr(league, "settings"), "position_slot_counts", {}))

    # Box scores carry each side's week lineup with projections; prefer them.
    lineup_by_team: dict[str, list[Any]] = {}
    matchup: Matchup | None = None
    for bs in box_scores or []:
        home, away = _attr(bs, "home_team"), _attr(bs, "away_team")
        for side, lineup in ((home, _attr(bs, "home_lineup", [])), (away, _attr(bs, "away_lineup", []))):
            if side is not None and _attr(side, "team_id") is not None:
                lineup_by_team[str(_attr(side, "team_id"))] = list(lineup or [])
        ids = {str(_attr(home, "team_id")), str(_attr(away, "team_id"))}
        if my_team_id and str(my_team_id) in ids and away is not None and home is not None:
            i_am_home = str(_attr(home, "team_id")) == str(my_team_id)
            me, opp = (home, away) if i_am_home else (away, home)
            matchup = Matchup(
                week=week,
                my_team_id=str(_attr(me, "team_id")),
                opponent_team_id=str(_attr(opp, "team_id")),
                my_platform_projection=_attr(bs, "home_projected" if i_am_home else "away_projected"),
                opp_platform_projection=_attr(bs, "away_projected" if i_am_home else "home_projected"),
                my_points=float(_attr(bs, "home_score" if i_am_home else "away_score", 0.0) or 0.0),
                opp_points=float(_attr(bs, "away_score" if i_am_home else "home_score", 0.0) or 0.0),
            )

    teams: list[TeamRow] = []
    for team in _attr(league, "teams", []) or []:
        tid = str(_attr(team, "team_id"))
        source = lineup_by_team.get(tid)
        roster_objs = source if source is not None else (_attr(team, "roster", []) or [])
        roster = [normalize_player(p, week, box=source is not None) for p in roster_objs]
        owners = _attr(team, "owners", []) or []
        owner = ""
        if owners:
            first = owners[0]
            owner = first.get("displayName", "") if isinstance(first, dict) else str(first)
        teams.append(TeamRow(
            team_id=tid,
            name=str(_attr(team, "team_name", tid)),
            owner=owner,
            wins=int(_attr(team, "wins", 0) or 0),
            losses=int(_attr(team, "losses", 0) or 0),
            ties=int(_attr(team, "ties", 0) or 0),
            points_for=float(_attr(team, "points_for", 0.0) or 0.0),
            roster=roster,
            is_mine=(my_team_id is not None and tid == str(my_team_id)),
        ))

    fas = [normalize_player(p, week) for p in (free_agents or [])]
    for fa in fas:
        fa.slot = "FA"

    settings = _attr(league, "settings")
    return LeagueSnapshot(
        platform=PLATFORM,
        profile=profile,
        league_id=str(_attr(league, "league_id", "")),
        league_name=str(_attr(settings, "name", "") or ""),
        season=int(_attr(league, "year", 0) or 0),
        week=int(week),
        slots=slots,
        bench_size=bench,
        teams=teams,
        free_agents=fas,
        matchup=matchup,
        synced_at=datetime.now(UTC).isoformat(),
        extra={"nfl_week": _attr(league, "nfl_week"), "current_week": _attr(league, "current_week")},
    )


# ---------------------------------------------------------------------------
# The one function that talks to ESPN
# ---------------------------------------------------------------------------
def fetch_snapshot(
    season: int,
    week: int,
    profile: str,
    free_agent_size: int = 200,
) -> LeagueSnapshot:
    """Pull the league from ESPN and normalise it.

    Raises with a clear message if the league id is missing. A private league
    without cookies raises whatever ``espn-api`` raises (usually a 401).
    """
    from espn_api.football import League  # imported lazily: optional dependency

    creds = credentials()
    if not creds["league_id"]:
        raise RuntimeError("ESPN_LEAGUE_ID is not set")

    league = League(
        league_id=int(creds["league_id"]),
        year=int(season),
        espn_s2=creds["espn_s2"],
        swid=creds["swid"],
    )
    my_team_id = identify_my_team(league.teams, creds["swid"], creds["team_id"])
    if my_team_id is None:
        names = ", ".join(f"{t.team_id}={t.team_name}" for t in league.teams)
        raise RuntimeError(
            "could not tell which ESPN team is yours; set ESPN_TEAM_ID to one of: " + names
        )

    box_scores = league.box_scores(week)
    free_agents: list[Any] = []
    for position in ("QB", "RB", "WR", "TE", "K", "D/ST"):
        try:
            free_agents.extend(league.free_agents(week=week, size=free_agent_size, position=position))
        except Exception as exc:  # noqa: BLE001 - one position failing must not sink the sync
            log.warning("ESPN free agents for %s failed: %s", position, exc)

    return build_snapshot(league, week, profile, my_team_id, box_scores, free_agents)
