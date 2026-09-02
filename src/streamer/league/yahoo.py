"""Yahoo Fantasy Football adapter.

Uses ``yahoo_fantasy_api`` over ``yahoo_oauth``. Yahoo requires OAuth2: a
one-time browser authorisation against an app you register at
developer.yahoo.com yields a refresh token, after which everything is
non-interactive. Credentials come from the environment:

    YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET   the app
    YAHOO_REFRESH_TOKEN                    from ``streamer yahoo-auth``
    YAHOO_LEAGUE_ID                        e.g. ``12345`` (the numeric id)

``yahoo_oauth`` insists on a JSON file, so :func:`oauth_session` writes one
from those variables into the data directory and points the library at it.

As with ESPN, the network is confined to :func:`fetch_snapshot`; the
normalisation is pure and unit-tested on the dict shapes the library documents.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
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

PLATFORM = "yahoo"

#: Yahoo position labels we care about; ``DEF`` is the team defence.
YAHOO_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")


def credentials() -> dict[str, str | None]:
    return {
        "client_id": os.environ.get("YAHOO_CLIENT_ID", "").strip() or None,
        "client_secret": os.environ.get("YAHOO_CLIENT_SECRET", "").strip() or None,
        "refresh_token": os.environ.get("YAHOO_REFRESH_TOKEN", "").strip() or None,
        "league_id": os.environ.get("YAHOO_LEAGUE_ID", "").strip() or None,
    }


def write_oauth_file(path: Path, creds: dict[str, str | None]) -> Path:
    """Materialise the JSON file ``yahoo_oauth`` wants from env variables."""
    missing = [k for k in ("client_id", "client_secret", "refresh_token") if not creds.get(k)]
    if missing:
        raise RuntimeError("Yahoo credentials missing: " + ", ".join(k.upper() for k in
                                                                    (f"YAHOO_{m}" for m in missing)))
    payload = {
        "consumer_key": creds["client_id"],
        "consumer_secret": creds["client_secret"],
        "access_token": "expired",          # forces an immediate refresh
        "refresh_token": creds["refresh_token"],
        "token_type": "bearer",
        "token_time": 0,
        "guid": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def explain_error(exc: BaseException) -> str:
    """Translate Yahoo's terse API errors into the thing to actually do."""
    text = str(exc)
    if "additional_authorization_required" in text:
        return (
            "Yahoo accepted the token but this app is not permitted to read Fantasy "
            "Sports data. On developer.yahoo.com open the app, tick 'Fantasy Sports' "
            "with 'Read' under API Permissions, then run `streamer yahoo-auth` again "
            "and update YAHOO_REFRESH_TOKEN -- the old token keeps the old permissions."
        )
    if "invalid_grant" in text or "INVALID_REFRESH_TOKEN" in text.upper():
        return (
            "Yahoo rejected the refresh token. Run `streamer yahoo-auth` again and "
            "update YAHOO_REFRESH_TOKEN."
        )
    if "invalid_client" in text:
        return "Yahoo rejected the client id/secret; check YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET."
    return text


def oauth_session(oauth_path: Path):
    """An authenticated ``yahoo_oauth.OAuth2`` session, refreshing as needed."""
    import logging as _logging

    from yahoo_oauth import OAuth2  # optional dependency, imported lazily

    # yahoo_oauth logs every token check at DEBUG through its own handler.
    _logging.getLogger("yahoo_oauth").setLevel(_logging.WARNING)

    creds = credentials()
    write_oauth_file(oauth_path, creds)
    session = OAuth2(None, None, from_file=str(oauth_path))
    if not session.token_is_valid():
        session.refresh_access_token()
    return session


# ---------------------------------------------------------------------------
# Pure normalisation over the dicts yahoo_fantasy_api returns
# ---------------------------------------------------------------------------
def normalize_player(entry: dict[str, Any], projection: float | None = None) -> PlayerRow:
    """Turn a roster / free-agent dict into a :class:`PlayerRow`."""
    eligible_raw = entry.get("eligible_positions") or []
    eligible = [canonical_slot(p) for p in eligible_raw]
    eligible = [s for s in eligible if s in ("QB", "RB", "WR", "TE", "K", "DST",
                                             "FLEX", "WR/RB", "WR/TE", "SUPERFLEX")]
    # Primary position: the first non-flex eligible position.
    primary = next((canonical_position(p) for p in eligible_raw
                    if canonical_position(p) in ("QB", "RB", "WR", "TE", "K", "DST")), "")
    if not primary:
        primary = canonical_position(entry.get("position_type") or entry.get("primary_position"))

    selected = entry.get("selected_position")
    slot = canonical_slot(selected) if selected else "BN"

    pct = entry.get("percent_owned")
    return PlayerRow(
        player_id=str(entry.get("player_id")),
        name=str(entry.get("name", "")),
        position=primary,
        team=normalize_team(entry.get("editorial_team_abbr")),
        status=canonical_status(entry.get("status")),
        slot=slot,
        eligible_slots=sorted(set(eligible)),
        on_bye=False,                       # filled from the NFL schedule later
        platform_projection=projection,
        percent_owned=float(pct) if pct not in (None, "") else None,
    )


def normalize_slots(positions: dict[str, dict[str, Any]]) -> tuple[dict[str, int], int]:
    """Yahoo's ``lg.positions()`` -> (starting slots, bench size)."""
    slots: dict[str, int] = {}
    bench = 0
    for label, info in (positions or {}).items():
        n = int((info or {}).get("count", 0) or 0)
        if n <= 0:
            continue
        canon = canonical_slot(label)
        if canon == "BN":
            bench += n
        elif canon in ("IR", "NA", ""):
            continue
        else:
            slots[canon] = slots.get(canon, 0) + n
    return slots, bench


def parse_matchups(raw: dict[str, Any], week: int, my_team_key: str) -> Matchup | None:
    """Pull the user's matchup out of Yahoo's scoreboard JSON.

    The document is Yahoo's usual nest of numbered dicts; we walk it for the
    matchup containing ``my_team_key`` rather than trusting any fixed path.
    """
    def walk(node: Any):
        if isinstance(node, dict):
            if "matchup" in node and isinstance(node["matchup"], dict):
                yield node["matchup"]
            for v in node.values():
                yield from walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v)

    for matchup in walk(raw):
        teams: list[tuple[str, float, float]] = []
        for team_blob in _iter_teams(matchup):
            key, points, projected = _team_fields(team_blob)
            if key:
                teams.append((key, points, projected))
        keys = [t[0] for t in teams]
        if my_team_key in keys and len(teams) == 2:
            me = next(t for t in teams if t[0] == my_team_key)
            opp = next(t for t in teams if t[0] != my_team_key)
            return Matchup(
                week=week,
                my_team_id=me[0],
                opponent_team_id=opp[0],
                my_platform_projection=me[2] or None,
                opp_platform_projection=opp[2] or None,
                my_points=me[1],
                opp_points=opp[1],
            )
    return None


def _iter_teams(matchup: dict[str, Any]):
    teams = None
    for k, v in matchup.items():
        if k == "0" and isinstance(v, dict) and "teams" in v:
            teams = v["teams"]
        elif k == "teams":
            teams = v
    if not isinstance(teams, dict):
        return
    for k, v in teams.items():
        if k == "count":
            continue
        if isinstance(v, dict) and "team" in v:
            yield v["team"]


def _team_fields(team_blob: Any) -> tuple[str | None, float, float]:
    key, points, projected = None, 0.0, 0.0
    parts = team_blob if isinstance(team_blob, list) else [team_blob]
    for part in parts:
        if isinstance(part, list):
            for item in part:
                if isinstance(item, dict) and "team_key" in item:
                    key = item["team_key"]
        elif isinstance(part, dict):
            if "team_key" in part:
                key = part["team_key"]
            if "team_points" in part:
                points = float((part["team_points"] or {}).get("total") or 0.0)
            if "team_projected_points" in part:
                projected = float((part["team_projected_points"] or {}).get("total") or 0.0)
    return key, points, projected


def build_snapshot(
    league: Any,
    week: int,
    profile: str,
    my_team_key: str,
    rosters: dict[str, list[dict[str, Any]]],
    free_agents: list[dict[str, Any]],
    teams_meta: dict[str, dict[str, Any]],
    matchups_raw: dict[str, Any] | None,
    projections: dict[str, float] | None = None,
) -> LeagueSnapshot:
    """Assemble a snapshot from already-fetched Yahoo data."""
    projections = projections or {}
    slots, bench = normalize_slots(league.positions() if hasattr(league, "positions") else {})
    settings = league.settings() if hasattr(league, "settings") else {}

    teams: list[TeamRow] = []
    for key, meta in teams_meta.items():
        roster = [normalize_player(e, projections.get(str(e.get("player_id"))))
                  for e in rosters.get(key, [])]
        managers = meta.get("managers") or []
        owner = ""
        if managers:
            m = managers[0]
            if isinstance(m, dict):
                m = m.get("manager", m)
                owner = str(m.get("nickname", ""))
        teams.append(TeamRow(
            team_id=key,
            name=str(meta.get("name", key)),
            owner=owner,
            roster=roster,
            is_mine=(key == my_team_key),
        ))

    fas = [normalize_player(e, projections.get(str(e.get("player_id")))) for e in free_agents]
    for fa in fas:
        fa.slot = "FA"

    matchup = parse_matchups(matchups_raw, week, my_team_key) if matchups_raw else None

    return LeagueSnapshot(
        platform=PLATFORM,
        profile=profile,
        league_id=str(settings.get("league_id", league.league_id if hasattr(league, "league_id") else "")),
        league_name=str(settings.get("name", "")),
        season=int(settings.get("season", 0) or 0),
        week=int(week),
        slots=slots,
        bench_size=bench,
        teams=teams,
        free_agents=fas,
        matchup=matchup,
        synced_at=datetime.now(UTC).isoformat(),
        extra={"current_week": settings.get("current_week")},
    )


# ---------------------------------------------------------------------------
# The one function that talks to Yahoo
# ---------------------------------------------------------------------------
def fetch_snapshot(season: int, week: int, profile: str, oauth_path: Path) -> LeagueSnapshot:
    """Pull the league from Yahoo and normalise it."""
    import yahoo_fantasy_api as yfa  # optional dependency, imported lazily

    creds = credentials()
    if not creds["league_id"]:
        raise RuntimeError("YAHOO_LEAGUE_ID is not set")

    try:
        sc = oauth_session(oauth_path)
        game = yfa.Game(sc, "nfl")
        league_key = creds["league_id"]
        if ".l." not in league_key:
            # Numeric id: qualify it with this season's game id.
            league_key = f"{game.game_id()}.l.{league_key}"
        league = game.to_league(league_key)

        my_team_key = league.team_key()
        teams_meta = league.teams()
        rosters = {key: league.to_team(key).roster(week) for key in teams_meta}
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(explain_error(exc)) from exc

    free_agents: list[dict[str, Any]] = []
    for position in YAHOO_POSITIONS:
        try:
            free_agents.extend(league.free_agents(position))
        except Exception as exc:  # noqa: BLE001
            log.warning("Yahoo free agents for %s failed: %s", position, exc)

    try:
        matchups_raw = league.matchups(week)
    except Exception as exc:  # noqa: BLE001
        log.warning("Yahoo matchups failed: %s", exc)
        matchups_raw = None

    return build_snapshot(
        league, week, profile, my_team_key, rosters, free_agents, teams_meta, matchups_raw
    )
