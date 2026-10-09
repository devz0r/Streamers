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

#: Statuses a platform will accept in an IR slot. ESPN's own rule varies by
#: league setting -- some allow any OUT designation, some only a true IR tag --
#: so the benefit of the doubt goes to the roster: a player is only called out
#: as misfiled when no common rule would let them sit there, which means they
#: are healthy, questionable or day-to-day.
IR_ELIGIBLE_STATUSES: tuple[str, ...] = LONG_TERM_OUT_STATUSES + ("OUT", "O", "INACTIVE")

#: Injury/availability statuses that mean "will not play".
OUT_STATUSES: tuple[str, ...] = LONG_TERM_OUT_STATUSES + ("OUT", "O", "NA", "COVID", "INACTIVE")

#: Not on an NFL roster at all, whatever team the platform still shows.
#: Yahoo marks unsigned, released and practice-squad players "NA" (not
#: active) and keeps their last team beside it: Tyreek Hill, unsigned and
#: listed with no team by ESPN, showed as NA for Miami on Yahoo and kept a
#: 12-a-game season value there.
NOT_ON_ROSTER_STATUSES: tuple[str, ...] = ("NA",)

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
    #: For a next man up: his share of his position's snaps that did not go
    #: to the absent lead (team's last four games), and the share of the
    #: lead's work that predicts for him. None when not a next man up.
    backfield_share: float | None = None
    takeover_share: float | None = None
    #: His place on his NFL team's depth chart at his position (1 = starter),
    #: from Sleeper; None when it is not known.
    depth_order: int | None = None
    #: His team's games in a row, up to now this season, that he has not
    #: played: how long an absence in progress has already run.
    games_missed: int = 0
    #: A photo of him (or his team's logo, for a D/ST) for the page; empty: none.
    photo: str = ""
    #: FantasyPros' chance he plays this week, for a questionable or doubtful
    #: player (:mod:`streamer.roster.play_odds`); None: not listed.
    fp_play: float | None = None
    #: Chance an absence that starts from here ends his season -- a lost job,
    #: a release, a long injury -- read from his season so far
    #: (:func:`streamer.roster.futures.season_loss`); None: the simulator's
    #: base share.
    season_loss: float | None = None
    #: For a player on no NFL roster whom the news has close to signing
    #: (``data.news``): the team when one is named, his chance of signing and
    #: playing this season, the games before he could, the share of the rest
    #: of the season he is expected to play, and what the news said. None /
    #: empty without recent news: he is worth nothing until he signs.
    signing_team: str | None = None
    signing_odds: float | None = None
    signing_wait: int = 0
    signing_share: float | None = None
    news: str = ""
    #: The part of ``ros_value`` inherited from an absent teammate's work.
    #: The season simulation takes it back out when it plays that absence
    #: out itself, week by week.
    inherited_ros: float = 0.0
    #: This week's projection from our model alone, before any platform
    #: projection is blended in. Waivers fall back to it when the free agents
    #: carry no platform number, so both sides of a move are priced alike.
    model_projection: float | None = None
    #: Fantasy points implied by sportsbook player props for this week, and
    #: the implied stat line behind them. None when no book posted the player.
    vegas_points: float | None = None
    vegas_stats: dict = field(default_factory=dict)
    vegas_books: int = 0
    #: This week's NFL opponent, and the player's role on his NFL team (QB,
    #: RB1, WR2, ...). The lineup simulator correlates players through these.
    nfl_opponent: str | None = None
    role: str = ""
    #: Chance the player suits up this week (injury tags), and his spread if
    #: he does. ``projection``/``projection_sd`` are the blend of both cases.
    play_probability: float = 1.0
    #: His last practice on this week's official injury report ("DNP",
    #: "Limited", "Full"), once the final report with his game status is out;
    #: "" otherwise. Separates questionable players (``_play_probability``).
    practice: str = ""
    outcome_sd: float | None = None
    #: How far his per-game projection could move over the next month, and
    #: seasons in the league (0 = rookie) where known. Waivers price upside
    #: from these.
    ros_sd: float | None = None
    experience: int | None = None
    #: Short, human reasons the projection moved: a starter ahead of him is
    #: out, his opportunity has changed.
    signals: list[str] = field(default_factory=list)
    #: Recent usage -- targets, target share, carries a game -- for display.
    usage: str = ""
    #: His game has kicked off, so his lineup spot can no longer change; and,
    #: once the game is final, the points he actually scored.
    locked: bool = False
    actual_points: float | None = None
    #: nflverse player id, where matched.
    nfl_id: str | None = None
    #: The projection before the second forecasts were blended in, and the
    #: weights the betting market and the FantasyPros consensus projection
    #: got (0 where he had none).
    pre_market_projection: float | None = None
    market_weight: float = 0.0
    consensus_weight: float = 0.0
    #: FantasyPros consensus (attached at publish time, never saved): the
    #: rest-of-season and this week's position rank, and this week's
    #: projected points.
    ecr_ros_pos_rank: int | None = None
    ecr_week_pos_rank: int | None = None
    fp_projection: float | None = None

    @property
    def week_value(self) -> float:
        """This week's number for lineup maths: the final score once there is
        one, the projection until then."""
        if self.actual_points is not None:
            return float(self.actual_points)
        return float(self.projection or 0.0)

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

    @property
    def unsigned(self) -> bool:
        """Not on an NFL roster: no team, or a platform's not-active tag."""
        return not self.team or self.status in NOT_ON_ROSTER_STATUSES

    @property
    def ir_slot_is_valid(self) -> bool:
        """Whether this player may occupy an IR slot. True if not in one."""
        return (not self.in_ir_slot) or self.status in IR_ELIGIBLE_STATUSES


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
    #: Waiver standing: priority (1 = first claim) or FAAB spent, and how
    #: many pickups the team has made (a read on how active the league is).
    waiver_rank: int | None = None
    faab_spent: float | None = None
    acquisitions: int | None = None

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
    #: League structure for season simulation: ``regular_season_weeks``,
    #: ``playoff_teams``, ``playoff_round_weeks``, ``tiebreak`` for seeds
    #: level on record ("points" or "h2h"), ``median_game``, ``waiver`` ("faab" / "priority"),
    #: ``faab_budget``, and ``schedule`` -- every regular-season pairing as
    #: [week, team_id, team_id]. Empty when the platform adapter cannot read
    #: it yet.
    rules: dict[str, Any] = field(default_factory=dict)

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
            rules=dict(raw.get("rules", {})),
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
