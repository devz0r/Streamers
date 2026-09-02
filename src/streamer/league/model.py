"""A platform-neutral snapshot of one fantasy league at one moment.

Everything the in-season tools need is captured here so that ESPN and Yahoo
look identical downstream: rosters, free agents, this week's matchup, the
league's starting slots, and each player's projection, status and bye. The
adapters in :mod:`streamer.league.espn` and :mod:`streamer.league.yahoo` fill
this in; nothing else in the package knows which platform it came from.

Snapshots serialise to JSON (``data/leagues/<profile>/week_N.json``) so a sync
can run wherever the platform APIs are reachable -- a Mac, or the weekly
GitHub Actions job -- and every analysis command works offline from the file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Canonical positions. Platform-specific labels (``D/ST``, ``DEF``, ``PK``)
#: are mapped onto these on the way in.
POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DST")

#: Canonical starting-slot names and the positions eligible for each.
SLOT_ELIGIBILITY: dict[str, tuple[str, ...]] = {
    "QB": ("QB",),
    "RB": ("RB",),
    "WR": ("WR",),
    "TE": ("TE",),
    "FLEX": ("RB", "WR", "TE"),
    "WR/RB": ("RB", "WR"),
    "WR/TE": ("WR", "TE"),
    "SUPERFLEX": ("QB", "RB", "WR", "TE"),
    "K": ("K",),
    "DST": ("DST",),
}

#: Slot labels that are not starting slots.
NON_STARTING_SLOTS: tuple[str, ...] = ("BN", "BE", "IR", "IR+", "NA")

#: Injured-reserve slots. A player here cannot be started or swapped for a
#: free agent without first being activated, which needs a free roster spot.
IR_SLOTS: tuple[str, ...] = ("IR", "IR+")

#: Statuses that keep a player out for weeks, not days (zero rest-of-season
#: value for waiver purposes). ESPN spells them INJURY_RESERVE / SUSPENSION.
LONG_TERM_OUT_STATUSES: tuple[str, ...] = (
    "IR", "IR-R", "INJURY_RESERVE", "PUP", "PUP-R", "NFI", "SUSP", "SUSPENSION", "SUSPENDED",
)

#: Injury/availability statuses that mean "will not play".
OUT_STATUSES: tuple[str, ...] = LONG_TERM_OUT_STATUSES + ("OUT", "O", "NA", "COVID", "INACTIVE")

#: Statuses that mean "probably plays, but discount". ESPN's DAY_TO_DAY is a
#: minor-injury tag that usually resolves to playing.
QUESTIONABLE_STATUSES: tuple[str, ...] = ("QUESTIONABLE", "Q", "DAY_TO_DAY", "DOUBTFUL", "D")


@dataclass
class PlayerRow:
    """One player as the platform sees them, plus what we attach."""

    player_id: str
    name: str
    position: str
    team: str | None = None            # NFL team, nflverse abbreviation
    status: str = ""                   # platform injury status, upper-cased
    slot: str = "BN"                   # current lineup slot, canonical
    eligible_slots: list[str] = field(default_factory=list)
    on_bye: bool = False
    #: Platform's own projection for the target week, if it publishes one.
    platform_projection: float | None = None
    #: Platform's season-to-date average, if available.
    season_avg: float | None = None
    percent_owned: float | None = None
    #: Filled by the projection engine: our mean and sd for the target week.
    projection: float | None = None
    projection_sd: float | None = None
    projection_source: str = ""
    #: A rest-of-season value proxy (per-game), for waiver decisions.
    ros_value: float | None = None

    @property
    def is_out(self) -> bool:
        return self.on_bye or self.status in OUT_STATUSES

    @property
    def is_questionable(self) -> bool:
        return self.status in QUESTIONABLE_STATUSES

    @property
    def starting(self) -> bool:
        return self.slot not in NON_STARTING_SLOTS

    @property
    def in_ir_slot(self) -> bool:
        return self.slot in IR_SLOTS

    @property
    def is_long_term_out(self) -> bool:
        return self.status in LONG_TERM_OUT_STATUSES


@dataclass
class TeamRow:
    """A fantasy team in the league."""

    team_id: str
    name: str
    owner: str = ""
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    roster: list[PlayerRow] = field(default_factory=list)
    is_mine: bool = False

    def starters(self) -> list[PlayerRow]:
        return [p for p in self.roster if p.starting]


@dataclass
class Matchup:
    """This week's head-to-head for the user's team."""

    week: int
    my_team_id: str
    opponent_team_id: str
    #: Platform's projected totals for each side, if it publishes them.
    my_platform_projection: float | None = None
    opp_platform_projection: float | None = None
    #: Points already scored this week (mid-week syncs).
    my_points: float = 0.0
    opp_points: float = 0.0


@dataclass
class LeagueSnapshot:
    """Everything about one league at one moment."""

    platform: str                      # "espn" | "yahoo"
    profile: str                       # the scoring profile it belongs to
    league_id: str
    league_name: str
    season: int
    week: int                          # the week being managed
    #: Starting-slot counts, canonical labels, e.g. {"QB": 1, "RB": 2, "FLEX": 1}.
    slots: dict[str, int]
    bench_size: int
    teams: list[TeamRow]
    free_agents: list[PlayerRow]
    matchup: Matchup | None
    synced_at: str
    #: Anything platform-specific worth keeping for the report.
    extra: dict[str, Any] = field(default_factory=dict)

    # -- accessors ---------------------------------------------------------
    @property
    def my_team(self) -> TeamRow:
        for team in self.teams:
            if team.is_mine:
                return team
        raise LookupError("no team in this snapshot is marked as yours")

    def team(self, team_id: str) -> TeamRow:
        for team in self.teams:
            if team.team_id == team_id:
                return team
        raise LookupError(f"no team with id {team_id!r}")

    @property
    def opponent(self) -> TeamRow | None:
        if self.matchup is None:
            return None
        return self.team(self.matchup.opponent_team_id)

    def all_players(self) -> list[PlayerRow]:
        out = [p for team in self.teams for p in team.roster]
        out.extend(self.free_agents)
        return out

    @property
    def starting_slots(self) -> dict[str, int]:
        return {s: n for s, n in self.slots.items() if s not in NON_STARTING_SLOTS and n > 0}

    # -- persistence -------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LeagueSnapshot:
        teams = [
            TeamRow(**{**t, "roster": [PlayerRow(**p) for p in t.get("roster", [])]})
            for t in raw.get("teams", [])
        ]
        fas = [PlayerRow(**p) for p in raw.get("free_agents", [])]
        matchup = Matchup(**raw["matchup"]) if raw.get("matchup") else None
        return cls(
            platform=raw["platform"], profile=raw["profile"],
            league_id=str(raw["league_id"]), league_name=raw.get("league_name", ""),
            season=int(raw["season"]), week=int(raw["week"]),
            slots=dict(raw.get("slots", {})), bench_size=int(raw.get("bench_size", 0)),
            teams=teams, free_agents=fas, matchup=matchup,
            synced_at=raw.get("synced_at", ""), extra=dict(raw.get("extra", {})),
        )

    @classmethod
    def load(cls, path: Path) -> LeagueSnapshot:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Label normalisation shared by both adapters
# ---------------------------------------------------------------------------
_POSITION_ALIASES = {
    "D/ST": "DST", "DEF": "DST", "DST": "DST", "D": "DST",
    "PK": "K", "K": "K",
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
}

_SLOT_ALIASES = {
    "RB/WR/TE": "FLEX", "W/R/T": "FLEX", "FLEX": "FLEX", "RB/WR": "WR/RB",
    "WR/RB": "WR/RB", "W/R": "WR/RB", "WR/TE": "WR/TE", "W/T": "WR/TE",
    "OP": "SUPERFLEX", "Q/W/R/T": "SUPERFLEX", "SUPERFLEX": "SUPERFLEX",
    "D/ST": "DST", "DEF": "DST", "DST": "DST", "PK": "K", "K": "K",
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
    "BE": "BN", "BN": "BN", "IR": "IR", "IR+": "IR", "NA": "NA",
}


def canonical_position(label: str | None) -> str:
    """Map a platform position label onto :data:`POSITIONS`."""
    if not label:
        return ""
    return _POSITION_ALIASES.get(str(label).strip().upper(), str(label).strip().upper())


def canonical_slot(label: str | None) -> str:
    """Map a platform lineup-slot label onto a canonical slot name."""
    if not label:
        return "BN"
    return _SLOT_ALIASES.get(str(label).strip().upper(), str(label).strip().upper())


#: Short tags for display, keyed by canonical status.
_SHORT_STATUS = {
    "QUESTIONABLE": "Q", "DOUBTFUL": "D", "DAY_TO_DAY": "DTD", "OUT": "OUT",
    "INJURY_RESERVE": "IR", "IR-R": "IR", "PUP-R": "PUP", "SUSPENSION": "SUSP",
    "SUSPENDED": "SUSP", "INACTIVE": "OUT",
}


def short_status(status: str | None) -> str:
    """A tag short enough for a table cell: ``QUESTIONABLE`` -> ``Q``."""
    if not status:
        return ""
    return _SHORT_STATUS.get(status, status[:4])


def canonical_status(label: str | None) -> str:
    """Upper-case a platform injury status; ``ACTIVE``/``""`` become empty."""
    if not label:
        return ""
    text = str(label).strip().upper()
    return "" if text in ("ACTIVE", "NORMAL", "HEALTHY", "NONE", "PROBABLE") else text
