"""The rest of the season, simulated: P(playoffs) and P(title).

Every rostered player's future is simulated with :mod:`streamer.roster.futures`
(validated: 82% of real six-game outcomes inside its 80% band). Then, in
each simulated season:

* every team starts, each week, the best lineup it could see *at the time*
  -- chosen on that week's projected level, not on hindsight -- and scores
  what those starters score;
* the regular season is played on the league's real schedule (plus a win
  against the median if the league plays one), on top of the current
  records and points;
* seeds go by record, ties broken by points scored;
* the playoffs are a bracket (byes for the top seeds when the field is not
  a power of two), each round as many weeks as the league plays; a league
  that reseeds pairs the best seed left with the worst after every round,
  otherwise the bracket is fixed.

P(title) is the share of seasons a team wins it all. Moves are valued by
re-running only the changed team's lineups on the same simulated futures
(common random numbers), so the difference is the move, not noise.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..league.model import SLOT_ELIGIBILITY, LeagueSnapshot, PlayerRow
from . import futures as fut
from . import level_calibration

log = logging.getLogger(__name__)

SKILL = ("QB", "RB", "WR", "TE")


@dataclass
class SeasonOdds:
    team_ids: list[str]
    names: list[str]
    records: list[str]
    p_playoffs: np.ndarray
    p_bye: np.ndarray
    p_title: np.ndarray
    exp_wins: np.ndarray
    n_sims: int
    mine: int | None = None
    #: Champion's team index in each simulated season (for paired comparisons).
    champion: np.ndarray | None = None
    #: Expected points for the rest of the regular season, this week on.
    exp_points: np.ndarray | None = None
    #: Power: the share of the other teams' scores a team beats in an average
    #: remaining regular-season week (all-play), whatever the schedule, the
    #: standings and the seeding say. Record-blind.
    power: np.ndarray | None = None
    #: Expected points a week for the rest of the regular season, and the
    #: points a week scored so far (None where no week has been counted).
    pts_ahead: np.ndarray | None = None
    pts_so_far: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    def of(self, team_id: str) -> dict:
        i = self.team_ids.index(team_id)
        return {"p_playoffs": float(self.p_playoffs[i]), "p_bye": float(self.p_bye[i]),
                "p_title": float(self.p_title[i]), "exp_wins": float(self.exp_wins[i])}


@dataclass
class GameStakes:
    week: int
    opponent: str
    opponent_id: str
    p_win: float
    playoffs_win: float
    playoffs_loss: float
    title_win: float
    title_loss: float

    @property
    def swing(self) -> float:
        return self.playoffs_win - self.playoffs_loss


@dataclass
class Stakes:
    """What each remaining game is worth to one team."""

    games: list[GameStakes]
    by_wins: dict[int, tuple[float, int]]     # final wins -> (P(playoffs), seasons)
    p_playoffs: float
    exp_wins: float
    n_sims: int

    def must_win(self, share: float = 1.25) -> list[GameStakes]:
        """Games whose swing stands out: at least ``share`` times the
        average swing, and at least 10 points."""
        if not self.games:
            return []
        avg = sum(g.swing for g in self.games) / len(self.games)
        return [g for g in self.games if g.swing >= max(share * avg, 0.10)]

    def wins_needed(self, target: float = 0.5) -> int | None:
        """The fewest final wins that get in at least ``target`` of the time."""
        for k in sorted(self.by_wins):
            p, n = self.by_wins[k]
            if n >= 30 and p >= target:
                return k
        return None


def bracket_order(slots: int) -> list[int]:
    """Seed order for a fixed bracket: 8 -> 1,8,4,5,3,6,2,7."""
    order = [1, 2]
    while len(order) < slots:
        n = len(order) * 2
        order = [x for s in order for x in (s, n + 1 - s)]
    return order


def nfl_byes(snapshot: LeagueSnapshot, weeks: list[int], cfg: Config) -> dict[str, set[int]]:
    """NFL team -> weeks it does not play."""
    from ..data.nflverse import games_frame

    g = games_frame(cfg)
    g = g[(g["season"] == snapshot.season) & (g["week"].isin(weeks))]
    teams = set(g["team"])
    out: dict[str, set[int]] = {}
    for w in weeks:
        playing = set(g[g["week"] == w]["team"])
        for t in teams - playing:
            out.setdefault(t, set()).add(w)
    return out


def nfl_opponents(snapshot: LeagueSnapshot, weeks: list[int], cfg: Config) -> dict[str, dict[int, str]]:
    """NFL team -> week -> the team it plays."""
    from ..data.nflverse import games_frame
    from ..teams import normalize_team

    g = games_frame(cfg)
    g = g[(g["season"] == snapshot.season) & (g["week"].isin(weeks))]
    out: dict[str, dict[int, str]] = {}
    for r in g.itertuples():
        out.setdefault(normalize_team(r.team), {})[int(r.week)] = normalize_team(r.opponent)
    return out


def heirs_for(players: list[PlayerRow]) -> list[tuple[str, str]]:
    """(lead, next man up) at RB/QB/TE on each NFL team, by healthy value."""
    groups: dict[tuple[str, str], list[PlayerRow]] = {}
    for p in players:
        if p.position in fut.TAKEOVER and p.team:
            groups.setdefault((p.team, p.position), []).append(p)
    from .projections import next_in_line

    out = []
    for members in groups.values():
        members = next_in_line(members)
        if len(members) >= 2:
            out.append((members[0].player_id, members[1].player_id))
    return out


#: Weekly spread of a replacement-level pickup's score.
REPLACEMENT_SD = 6.0


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval for a share ``p`` of ``n`` simulated seasons:
    how far the simulation's own noise could move it (not whether the model
    is right). Well behaved near 0% and 100%, unlike p +- 2 SE."""
    if n <= 0:
        return 0.0, 1.0
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


#: Streaming a D/ST or kicker -- the best-projected unit left on the wire each
#: week -- beats the best free agent's season level, because each week's pick
#: has a good matchup. Replayed 2023-2025 with the other managers who stream
#: picking first (scripts/fit_stream_premium.py): defences 1.29/1.22/1.33 a
#: season in a ten-team league and 0.87/1.20/0.94 in a fourteen, kickers
#: 1.05/1.04/1.07 and 1.11/1.28/1.22; the mean of all six each. The wire's
#: D/ST and K are filled at this multiple of the best free agent's level.
#: DECISIONS.md, "Streaming in the season simulation".
STREAM_PREMIUM = {"DST": 1.14, "K": 1.13}


def replacement_levels(snapshot: LeagueSnapshot, simulated: bool = False) -> dict[str, float]:
    """Per position: the second-best free agent's per-game value (D/ST and
    K: the best, since they are streamed weekly, times :data:`STREAM_PREMIUM`).
    ``simulated``: as the season simulator reads levels (``level_calibration``),
    for its own fill-ins; everything that sets a season value beside it reads
    the values as shown."""
    def value(p: PlayerRow) -> float:
        if simulated:
            return level_calibration.season_level(p)
        return float(p.ros_value if p.ros_value is not None else (p.projection or 0.0))

    out = {}
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        vals = sorted((value(p)
                       for p in snapshot.free_agents if p.position == pos and not p.is_long_term_out
                       and p.team), reverse=True)
        if vals:
            out[pos] = vals[0] * STREAM_PREMIUM[pos] if pos in ("K", "DST") else vals[min(1, len(vals) - 1)]
    return out


def _eligible(p: PlayerRow, slot: str) -> bool:
    if p.eligible_slots and slot in p.eligible_slots:
        return True
    return p.position in SLOT_ELIGIBILITY.get(slot, ())


def _first_best(sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Down each column of ``sub`` (players by simulated weeks, no NaN): the
    row of the largest value, the first on a tie, and that value -- what
    ``argmax(axis=0)`` gives, a row at a time, which for the handful of
    players a slot has is several times quicker."""
    best = sub[0].copy()
    local = np.zeros(sub.shape[1], dtype=np.intp)
    ahead = np.empty(sub.shape[1], dtype=bool)
    for r in range(1, sub.shape[0]):
        row = sub[r]
        np.greater(row, best, out=ahead)
        np.maximum(best, row, out=best)
        np.putmask(local, ahead, r)
    return local, best


def all_play(scores: np.ndarray) -> np.ndarray:
    """Each team's share of the other teams' scores it beats (ties half), over
    every simulated week: ``scores`` is ``(teams, sims, weeks)``. A team that
    outscores everyone every week is 1.0, an average team 0.5."""
    n_t = scores.shape[0]
    if n_t < 2 or scores.shape[2] == 0:
        return np.full(n_t, 0.5)
    out = np.zeros(n_t)
    for k in range(scores.shape[2]):
        wk = scores[:, :, k]
        beat = (wk[:, None, :] > wk[None, :, :]).sum(axis=1) + 0.5 * ((wk[:, None, :] == wk[None, :, :]).sum(axis=1) - 1)
        out += (beat / (n_t - 1)).mean(axis=1)
    return out / scores.shape[2]


def division_seeds(key: np.ndarray, divisions: list[str], p_teams: int) -> np.ndarray:
    """Seeds per simulated season when division winners come first: each
    division's best ``key`` (record, then points) is seeded ahead of every
    other team, in key order; the rest fill in by key, divisions aside.
    ``key``: (teams, sims). Returns (p_teams, sims) of team indices."""
    n = key.shape[1]
    cols = np.arange(n)
    div = np.asarray(divisions)
    winners = []
    for d in dict.fromkeys(divisions):
        members = np.flatnonzero(div == d)
        winners.append(members[np.argmax(key[members], axis=0)])
    w = np.stack(winners)                                                  # (k, sims)
    w = np.take_along_axis(w, np.argsort(-key[w, cols], axis=0), axis=0)
    rest_key = key.astype(float).copy()
    rest_key[w, cols] = -np.inf
    rest = np.argsort(-rest_key, axis=0)[: max(p_teams - len(winners), 0)]
    return np.vstack([w, rest])[:p_teams]


class SeasonModel:
    """Simulated futures for a league, and the machinery to score rosters."""

    def __init__(self, snapshot: LeagueSnapshot, cfg: Config, n_sims: int = 2000, seed: int = 23,
                 extra_players: list[PlayerRow] | None = None):
        self.snapshot, self.cfg, self.n = snapshot, cfg, n_sims
        rules = snapshot.rules or {}
        self.reg_weeks = int(rules.get("regular_season_weeks") or 14)
        self.playoff_teams = int(rules.get("playoff_teams") or 4)
        self.round_weeks = max(int(rules.get("playoff_round_weeks") or 1), 1)
        self.median = bool(rules.get("median_game"))
        self.reseed = bool(rules.get("reseed"))
        # A tied playoff game goes to the higher seed unless the league breaks
        # it on season points.
        self.playoff_points_tiebreak = rules.get("playoff_tiebreak") == "points"
        # Division winners seeded first (and so given the byes), where the
        # league says so and every team has a division.
        league_conf = ((cfg.profiles.get(cfg.profile) or {}).get("league") or {}) if cfg.profile else {}
        divs = [t.division for t in snapshot.teams]
        self.divisions = (divs if league_conf.get("division_byes") and all(divs) and len(set(divs)) > 1
                          else None)
        self.rounds = max(math.ceil(math.log2(max(self.playoff_teams, 2))), 1)
        start = int(snapshot.week)
        # The platform can be a week behind the calendar: Monday night's game
        # is over, the week has turned, but the standings have not counted it
        # yet. Skipping that week would lose a result for every team, so it
        # is played out here instead (with today's rosters).
        per_week = 2 if self.median else 1
        counted = min((t.wins + t.losses + t.ties for t in snapshot.teams), default=0) // per_week
        self.lag = max(0, min(start - 1 - counted, 2)) if snapshot.teams else 0
        first = start - self.lag
        self.last_week = self.reg_weeks + self.rounds * self.round_weeks
        self.weeks = list(range(first, self.last_week + 1))
        self.schedule = [(int(w), str(a), str(b)) for w, a, b in rules.get("schedule", [])
                         if first <= int(w) <= self.reg_weeks]
        self.notes = ([f"The standings have not counted week {first} yet, so its results are simulated "
                       "here; the odds firm up once the platform finalises it."] if self.lag else [])
        self.teams = snapshot.teams
        self.team_index = {t.team_id: i for i, t in enumerate(self.teams)}
        pool: dict[str, PlayerRow] = {}
        for t in self.teams:
            for p in t.roster:
                pool.setdefault(p.player_id, p)
        for p in extra_players or []:
            pool.setdefault(p.player_id, p)
        self.players = [p for p in pool.values() if p.projection is not None or p.ros_value is not None]
        self.pid = {p.player_id: i for i, p in enumerate(self.players)}
        byes = nfl_byes(snapshot, self.weeks, cfg)
        everyone = snapshot.all_players()
        roster_conf = cfg.raw.get("roster", {})
        # Teammates and opponents drawn together each week (season_correlation).
        opponents = None
        if roster_conf.get("season_correlation"):
            try:
                opponents = nfl_opponents(snapshot, self.weeks, cfg)
            except Exception as exc:  # noqa: BLE001 - a refinement; never block the odds
                log.warning("NFL schedule unavailable, players drawn on their own: %s", exc)
        self.futures = fut.simulate(self.players, self.weeks, byes, n_sims=n_sims, seed=seed,
                                    heirs=heirs_for(everyone), current=self.lag,
                                    takeover=roster_conf.get("next_man_up"),
                                    takeover_sd=roster_conf.get("next_man_up_spread"),
                                    kept=roster_conf.get("next_man_up_kept"), opponents=opponents)
        # Players by (sims * weeks): a player's seasons in one contiguous row,
        # which is how the lineups read them (one slot's players at a time).
        n_all = len(self.players)
        self._lv_p = np.ascontiguousarray(self.futures.levels.transpose(1, 0, 2), dtype=np.float32).reshape(n_all, -1)
        self._sc_p = np.ascontiguousarray(self.futures.scores.transpose(1, 0, 2), dtype=np.float32).reshape(n_all, -1)
        self._flat = np.arange(n_sims * len(self.weeks), dtype=np.intp)
        self._lv_plain = not bool(np.isnan(self._lv_p).any())     # no NaN: the quick pick is argmax
        self._fill_cache: dict[tuple, tuple] = {}
        self._slot_memo: dict[tuple, tuple] = {}
        self._slot_memo_size = 0.0
        self.slots = [(s, c) for s, c in sorted(snapshot.starting_slots.items(),
                      key=lambda kv: (len(SLOT_ELIGIBILITY.get(kv[0], (kv[0],))), kv[0]))]
        # Replacement level: what any team can pick up off the wire in a
        # given week -- the second-best free agent at each position. A slot
        # whose best rostered option is worse (a bye, an injury, no backup)
        # is filled at that level instead of scoring zero, which is what a
        # real manager does and what stops a frozen roster from overvaluing
        # bench depth.
        self.replacement = replacement_levels(snapshot, simulated=True)
        noise_rng = np.random.default_rng(seed + 7)
        self.rep_noise = {slot: noise_rng.standard_normal((n_sims, len(self.weeks))) for slot, _c in self.slots}
        # The wire, week by week (``roster.wire_thinning``): a free agent on
        # bye fills nothing that week, and when several teams need the same
        # fill-in, the ones ahead of you in waiver order take the best. Each
        # team's holes are counted first on today's rosters (``_holes``), then
        # every lineup is scored against what the others leave.
        self.thinning = bool(roster_conf.get("wire_thinning", True))
        # Claims compete at these slots only: D/ST and K streams follow the
        # week's matchups, so the k-th best streamer is far closer to the best
        # than a list by season value says (that list cost a streaming team
        # 4 points of playoff odds), and they are left to byes alone.
        self._compete_slots = set(roster_conf.get("wire_compete", ["QB", "RB", "WR", "TE", "FLEX"]))
        # The week under way: lineups are mostly set, so the wire fills only a
        # spot nobody on the roster can play (a bye, an injury) -- it does not
        # stand in for a weak starter as it does in the weeks to come
        # (``roster.stream_this_week``; DECISIONS.md, "The season simulator,
        # piece by piece").
        self.stream_now = bool(roster_conf.get("stream_this_week", True))
        self._wire_cache: dict[tuple, np.ndarray] = {}
        self._wire_ready = False                # true once the holes below are counted
        self._rep_table: dict[str, np.ndarray] = {}
        self._holes: list[dict[str, np.ndarray]] = []
        self._demand: dict[str, np.ndarray] = {}
        self._ahead_share = np.full(len(self.teams), 0.5)
        if self.thinning:
            self._rep_table = self._wire_by_week(snapshot, byes)
            n_t = len(self.teams)
            ranks = [t.waiver_rank for t in self.teams]
            if n_t > 1 and any(ranks):
                # Waiver order (1 = first claim): the share of rival claimants
                # ahead of each team; an unknown rank sits in the middle. The
                # order turns over as claims are made, so today's holds only
                # for the next ``PRIORITY_WEEKS``; after that, the middle.
                self._ahead_share = np.array([(int(r) - 1) / (n_t - 1) if r else 0.5 for r in ranks], float)
            for t in self.teams:
                holes: dict[str, np.ndarray] = {}
                self.team_scores([p.player_id for p in t.roster], _holes=holes, _compete=False)
                self._holes.append(holes)
            for slot, _c in self.slots:
                self._demand[slot] = sum((h.get(slot, 0) for h in self._holes),
                                         np.zeros((n_sims, len(self.weeks)), np.int16))
        self._wire_ready = True
        self.base_scores = np.stack([self.team_scores([p.player_id for p in t.roster], owner=None,
                                                      _team=i) for i, t in enumerate(self.teams)])
        # Rosters do not stay put: every manager claims, streams and trades,
        # and the weakest claim first, so a team's edge over the league fades
        # the further ahead the week (``roster.strength_fade``: the share kept
        # per week ahead). Taken from today's rosters, the same for any
        # override, so what a move adds passes through whole.
        self.retention = float(roster_conf.get("strength_fade", 1.0))
        self._fades: dict[float, np.ndarray] = {}
        self._base_season: dict | None = None

    def _fade(self, retention: float) -> np.ndarray:
        """``(teams, 1, weeks)`` points taken off each team's weekly score:
        ``1 - retention ** weeks_ahead`` of its expected edge over the league
        that week (nothing this week)."""
        if retention not in self._fades:
            m = self.base_scores.mean(axis=1)                                   # (teams, weeks)
            ahead = np.maximum(np.arange(len(self.weeks)) - self.lag, 0)
            self._fades[retention] = ((1.0 - retention ** ahead)[None, :] * (m - m.mean(axis=0)[None, :]))[:, None, :]
        return self._fades[retention]

    #: Free agents per slot kept for the week-by-week wire: deeper than any
    #: week's competition reaches.
    WIRE_DEPTH = 16
    #: Weeks (from now) today's waiver order still holds for.
    PRIORITY_WEEKS = 2

    def _wire_by_week(self, snapshot: LeagueSnapshot, byes: dict[str, set[int]]) -> dict[str, np.ndarray]:
        """Per starting slot, ``(weeks, WIRE_DEPTH)``: the free agents a team
        could fill it with each week, best first, those on bye left out --
        starting from the second best, as :func:`replacement_levels` does
        (the best for D/ST and K, streamed weekly)."""
        pool: dict[str, list[tuple[float, str]]] = {}
        for p in snapshot.free_agents:
            if p.is_long_term_out or not p.team:
                continue
            v = level_calibration.season_level(p)
            pool.setdefault(p.position, []).append((v, p.team))
        out = {}
        for slot, _c in self.slots:
            positions = SLOT_ELIGIBILITY.get(slot, (slot,))
            table = np.zeros((len(self.weeks), self.WIRE_DEPTH))
            for k, week in enumerate(self.weeks):
                # Each position's own list from its second best (its best for
                # D/ST and K), merged: with nobody else claiming, a flex slot
                # is filled at the best of those, as slot_replacement says.
                vals = []
                for pos in positions:
                    here = sorted((v for v, team in pool.get(pos, []) if week not in byes.get(team, set())),
                                  reverse=True)
                    first = 0 if pos in ("K", "DST") else min(1, max(len(here) - 1, 0))
                    if pos in ("K", "DST") and here:
                        # Streamed: each week's pick is a good matchup, worth more than the best level.
                        here = [here[0] * STREAM_PREMIUM[pos]] + here[1:]
                    vals += here[first:]
                vals = sorted(vals, reverse=True)[:self.WIRE_DEPTH]
                if vals:
                    vals += [vals[-1]] * (self.WIRE_DEPTH - len(vals))
                    table[k] = vals
            out[slot] = table
        return out

    # -- lineups ---------------------------------------------------------
    def team_scores(self, roster_ids: list[str], available_from: dict[str, int] | None = None,
                    gone_from: dict[str, int] | None = None, owner: str | None = None,
                    _team: int | None = None, _holes: dict | None = None, _compete: bool = True,
                    _sims: np.ndarray | None = None) -> np.ndarray:
        """``(n_sims, weeks)`` points from starting the best lineup each week.

        ``available_from`` / ``gone_from`` map a player to the week index he
        joins / leaves the roster, per simulation (arrays) or for all.
        ``owner``: whose roster this is (his settled week, his place in the
        waiver order and his own holes are then known). ``_sims``: score
        only these simulations (indices; the rows returned, in that order) --
        each simulation is scored on its own, so the rows are the same ones
        a full call returns.
        """
        team = _team if _team is not None else self.team_index.get(owner) if owner is not None else None
        if owner is not None:
            roster_ids, available_from, gone_from = self._this_week_is_settled(
                owner, list(roster_ids), dict(available_from or {}), dict(gone_from or {}))
        ids = [i for i in roster_ids if i in self.pid]
        n = self.n if _sims is None else len(_sims)
        n_w = len(self.weeks)
        if not ids:
            return np.zeros((n, n_w))
        cols = np.array([self.pid[i] for i in ids], dtype=np.intp)
        players = [self.players[c] for c in cols]
        # Worked flat: a player-week of a simulation is one index into the
        # player's row (``_lv_p`` / ``_sc_p``: players by sims * weeks).
        if _sims is None:
            m = self._flat
        else:
            m = (np.asarray(_sims, dtype=np.intp)[:, None] * n_w + np.arange(n_w)[None, :]).reshape(-1)
        here = np.arange(m.size, dtype=np.intp) if _sims is not None else m
        # When each player is on the roster (weeks he joins / leaves): the
        # masks to apply to his row, and the same as a key (None: per
        # simulation, so not something to remember a result under).
        k_idx = np.arange(n_w)[None, :]
        masks: dict[int, list[np.ndarray]] = {}
        marks: dict[int, tuple | None] = {}
        for table, joins in ((available_from, True), (gone_from, False)):
            for pid, when in (table or {}).items():
                if pid not in ids:
                    continue
                j = ids.index(pid)
                when = np.asarray(when)
                per_sim = bool(when.ndim)
                if per_sim and _sims is not None:
                    when = when[_sims]
                w = when[:, None] if per_sim else when
                on_roster = (k_idx >= w) if joins else (k_idx < w)
                masks.setdefault(j, []).append(np.broadcast_to(on_roster, (n, n_w)).reshape(-1))
                marks[j] = None if per_sim or marks.get(j, ()) is None \
                    else (*marks.get(j, ()), (joins, when.item()))
        eligible = [np.array([j for j, p in enumerate(players) if _eligible(p, slot)], dtype=np.intp)
                    for slot, _c in self.slots]
        # Whether a later slot can still take one of this slot's players: if
        # not, what the slot leaves behind is never read.
        later, after = [], set()
        for idx in reversed(eligible):
            mine = set(idx.tolist())
            later.append(bool(mine & after))
            after |= mine
        later.reverse()
        # A slot's result depends on its own players alone until another slot
        # has taken from them, so it is kept (``_slot_memo``) and a roster that
        # differs by a player re-does only the slots he can fill.
        keep = _sims is None and _holes is None
        state: dict[int, np.ndarray] = {}                  # a player's row once a slot has used it
        total = np.zeros(m.size)
        used = np.float32(-2.0)
        for i, (slot, count) in enumerate(self.slots):
            idx = eligible[i]
            if idx.size == 0:
                rep_score, _floor = self._fill(slot, team, _compete, _sims)
                for _ in range(count):
                    total += rep_score
                    if _holes is not None:
                        _holes[slot] = _holes.get(slot, 0) + np.ones((n, n_w), np.int16)
                continue
            key = None
            if keep and not any(int(j) in state for j in idx) and all(marks.get(int(j), ()) is not None for j in idx):
                key = (slot, team, _compete, tuple((ids[j], marks.get(int(j), ())) for j in idx))
                hit = self._slot_memo.pop(key, None)
                if hit is not None:
                    self._slot_memo[key] = hit                 # most recently used last
                    for part in hit[0]:
                        total += part
                    if later[i]:
                        for at, j in enumerate(idx):
                            state[int(j)] = hit[1][at]
                    continue
            rep_score, floor = self._fill(slot, team, _compete, _sims)
            # The slot's eligible players, each as he stands now.
            sub = self._lv_p[cols[idx]] if _sims is None else self._lv_p[np.ix_(cols[idx], m)]
            for at, j in enumerate(idx):
                j = int(j)
                if j in state:
                    sub[at] = state[j]
                else:
                    for mask in masks.get(j, ()):
                        sub[at] = np.where(mask, sub[at], -1.0)
            parts = []
            for _ in range(count):
                # Each pick takes the best eligible player not yet used, week
                # by week; a used player is marked -2. Ties fall to the first.
                if self._lv_plain:
                    local, best = _first_best(sub)
                else:
                    local = sub.argmax(axis=0)
                    best = sub[local, here]
                ok = (best >= floor) & (best > -0.5)
                if _holes is not None:
                    _holes[slot] = _holes.get(slot, 0) + (~ok).reshape(n, n_w).astype(np.int16)
                part = np.where(ok, self._sc_p[cols[idx[local]], m], rep_score)
                total += part
                parts.append(part)
                sub[local, here] = np.where(ok, used, best)
            if later[i]:
                for at, j in enumerate(idx):
                    state[int(j)] = sub[at]
            if key is not None:
                self._remember(key, parts, sub)
        return total.reshape(n, n_w)

    def _fill(self, slot: str, team: int | None, compete: bool, sims: np.ndarray | None):
        """What ``slot`` scores filled from the wire, and the level a rostered
        player must reach to be started over it (each flat over the
        simulations asked for, or one number). Worked out once per slot and
        team on every simulation, then read for the ones wanted."""
        key = (slot, team, compete)
        full = self._fill_cache.get(key) if self._wire_ready else None
        if full is None:
            rep = self._wire_level(slot, team, compete)
            noise = self.rep_noise[slot]
            if np.ndim(rep):
                rep_score = np.where(rep > 0, np.maximum(rep + REPLACEMENT_SD * noise, 0.0), 0.0)
                floor = np.maximum(rep, 0.0) + 1e-6
            else:
                rep_score = np.maximum(rep + REPLACEMENT_SD * noise, 0.0) if rep > 0 else 0.0
                floor = max(rep, 0.0) + 1e-6                                  # beats the wire
            if not self.stream_now and self.lag < len(self.weeks):
                # This week only a hole is filled: any player who plays beats the wire.
                floor = np.broadcast_to(np.asarray(floor, dtype=float), (self.n, len(self.weeks))).copy()
                floor[:, self.lag] = 1e-6
            full = (rep_score, floor)
            if self._wire_ready:
                self._fill_cache[key] = full
        if sims is None:
            return tuple(x.reshape(-1) if np.ndim(x) else x for x in full)
        return tuple(x[sims].reshape(-1) if np.ndim(x) else x for x in full)

    #: Memory the kept slot results may hold per simulation (bytes).
    SLOT_MEMO_BYTES = 256e6

    def _remember(self, key: tuple, parts: list[np.ndarray], sub: np.ndarray) -> None:
        """Keep a slot's result, dropping the least recently used past the budget."""
        size = sub.nbytes + sum(part.nbytes for part in parts)
        if size > self.SLOT_MEMO_BYTES / 4:
            return
        self._slot_memo[key] = (parts, sub, size)
        self._slot_memo_size += size
        while self._slot_memo_size > self.SLOT_MEMO_BYTES:
            oldest = next(iter(self._slot_memo))
            self._slot_memo_size -= self._slot_memo.pop(oldest)[2]

    def _this_week_is_settled(self, owner: str, roster_ids: list[str], available_from: dict,
                              gone_from: dict) -> tuple[list[str], dict, dict]:
        """A move made mid-week cannot reach back into games already played.

        Against ``owner``'s current roster: a newcomer whose game has kicked
        off joins from next week (his Sunday points were scored for someone
        else's wire), and a departing player whose game has kicked off still
        counts this week (those points are already yours)."""
        team = self.snapshot.teams[self.team_index[owner]]
        original = {p.player_id for p in team.roster}
        played = lambda pid: pid in self.pid and (self.players[self.pid[pid]].locked  # noqa: E731
                                                  or self.players[self.pid[pid]].actual_points is not None)
        # Weeks before the current one (a lagging platform's) are history too.
        for pid in roster_ids:
            join = self.lag + (1 if played(pid) else 0)
            if pid not in original and join > 0:
                cur = available_from.get(pid, 0)
                available_from[pid] = np.maximum(np.asarray(cur), join) if np.ndim(cur) else max(int(cur), join)
        for pid in original:
            leave = self.lag + (1 if played(pid) else 0)
            if pid not in roster_ids and leave > 0:
                roster_ids.append(pid)
                gone_from[pid] = leave
        return roster_ids, available_from, gone_from

    def _wire_level(self, slot: str, team: int | None, compete: bool = True):
        """What a team can fill ``slot`` with from the wire: one number for
        every week (thinning off), else ``(n_sims, weeks)`` -- that week's
        free agents off bye, as deep as the rival claimants ahead of him in
        waiver order reach (holes the other teams have that week, at the
        same slot, times the share of them ahead of him)."""
        table = self._rep_table.get(slot)
        if table is None:
            return self.slot_replacement(slot)
        if not compete or not self._demand or slot not in self._compete_slots:
            return np.broadcast_to(table[:, 0], (self.n, len(self.weeks)))
        # Fixed once the simulation is built (the holes and the order are),
        # and asked for on every lineup: kept, read-only.
        key = (slot, team)
        if self._wire_ready and key in self._wire_cache:
            return self._wire_cache[key]
        demand = self._demand[slot].astype(float)
        n_w = len(self.weeks)
        share = np.full(n_w, 0.5)
        if team is not None and team < len(self._holes):
            others = demand - self._holes[team].get(slot, 0)
            share[: self.PRIORITY_WEEKS] = float(self._ahead_share[team])
        else:
            n_t = max(len(self.teams), 1)
            others = demand * (n_t - 1) / n_t
        ahead = np.clip(np.rint(np.maximum(others, 0.0) * share[None, :]).astype(np.intp), 0, table.shape[1] - 1)
        level = table[np.arange(len(self.weeks))[None, :], ahead]
        if self._wire_ready:
            level.setflags(write=False)
            self._wire_cache[key] = level
        return level

    def slot_replacement(self, slot: str) -> float:
        positions = SLOT_ELIGIBILITY.get(slot, (slot,))
        return max((self.replacement.get(pos, 0.0) for pos in positions), default=0.0)

    # -- the season ------------------------------------------------------
    def odds(self, override: dict[str, np.ndarray] | None = None, retention: float | None = None) -> SeasonOdds:
        """Play out every simulated season. ``override`` swaps in a team's
        weekly scores (from :meth:`team_scores` on a changed roster);
        ``retention`` replaces ``roster.strength_fade`` (for grading)."""
        scores = self._scores(override, retention)
        n_t = len(self.teams)
        wins, points, _results = self._regular_season(scores)
        made, byes, champ = self._playoffs(scores, wins, points)
        title = np.zeros(n_t)
        if champ is not None:
            title = np.bincount(np.asarray(champ).ravel(), minlength=n_t) / self.n
        mine = next((i for i, t in enumerate(self.teams) if t.is_mine), None)
        return SeasonOdds(
            team_ids=[t.team_id for t in self.teams], names=[t.name for t in self.teams],
            records=[f"{t.wins}-{t.losses}" + (f"-{t.ties}" if t.ties else "") for t in self.teams],
            p_playoffs=made.mean(axis=1), p_bye=byes.mean(axis=1), p_title=title,
            exp_wins=wins.mean(axis=1), n_sims=self.n, mine=mine,
            champion=np.asarray(champ) if champ is not None else None, notes=list(self.notes),
            exp_points=scores[:, :, [k for k, w in enumerate(self.weeks) if w <= self.reg_weeks]]
            .sum(axis=2).mean(axis=1),
            power=all_play(scores[:, :, [k for k, w in enumerate(self.weeks) if w <= self.reg_weeks]]),
            pts_ahead=scores[:, :, [k for k, w in enumerate(self.weeks) if w <= self.reg_weeks]]
            .mean(axis=(1, 2)), pts_so_far=self._pts_so_far())

    def champion(self, override: dict[str, np.ndarray] | None = None) -> np.ndarray | None:
        """The champion (team index) of every simulated season -- what
        :meth:`odds` reports as ``champion``, without the standings, power
        and points summaries a caller pricing a title does not read. Only
        the overridden teams' games and points are played again; the rest of
        the regular season is the one already played on today's rosters."""
        if self.median:
            # A median game's result moves with every team's score.
            scores = self._scores(override)
            wins, points, _results = self._regular_season(scores)
            champ = self._playoffs(scores, wins, points)[2]
            return np.asarray(champ) if champ is not None else None
        base = self._played()
        scores, fade = base["scores"], base["fade"]
        rows = {self.team_index[tid]: arr for tid, arr in (override or {}).items()}
        kept = {t: scores[t].copy() for t in rows}
        try:
            for t, arr in rows.items():
                scores[t] = arr if fade is None else arr - fade[t]
            wins, points = base["wins"].copy(), base["points"].copy()
            for t in rows:
                points[t] = float(self.teams[t].points_for)
                for k in base["weeks"]:
                    points[t] += scores[t, :, k]
            for k, ia, ib, was in base["games"]:
                if ia in rows or ib in rows:
                    wk = scores[:, :, k]
                    # Wins are halves, so taking a result out and putting the
                    # new one in is exact.
                    change = ((wk[ia] > wk[ib]) + 0.5 * (wk[ia] == wk[ib])) - was
                    wins[ia] += change
                    wins[ib] -= change
            champ = self._playoffs(scores, wins, points)[2]
            return np.asarray(champ) if champ is not None else None
        finally:
            for t, row in kept.items():
                scores[t] = row

    def _played(self) -> dict:
        """The regular season on today's rosters, played once: the scores
        (faded as :meth:`_scores` does), the standings and each game's
        result. Played again if the teams are replaced (another view)."""
        if self._base_season is None or self._base_season["teams"] is not self.teams:
            scores = self._scores(None)
            wins, points, results = self._regular_season(scores)
            reg = []
            for k, week in enumerate(self.weeks):
                if week > self.reg_weeks:
                    break                          # as the regular season stops
                reg.append((k, week))
            games = [(k, self.team_index[a], self.team_index[b], results[(week, self.team_index[a], self.team_index[b])])
                     for k, week in reg for w, a, b in self.schedule
                     if w == week and a in self.team_index and b in self.team_index]
            self._base_season = {
                "teams": self.teams, "scores": scores, "wins": wins, "points": points, "games": games,
                "weeks": [k for k, _week in reg],
                "fade": self._fade(self.retention) if self.retention < 1.0 else None}
        return self._base_season

    def _pts_so_far(self) -> np.ndarray:
        """Points a week each team has scored in the weeks the standings
        count (NaN for a team with none)."""
        per_week = 2 if self.median else 1
        games = np.array([(t.wins + t.losses + t.ties) / per_week for t in self.teams], float)
        pts = np.array([t.points_for for t in self.teams], float)
        return np.where(games > 0, pts / np.maximum(games, 1), np.nan)

    def _scores(self, override: dict[str, np.ndarray] | None, retention: float | None = None) -> np.ndarray:
        scores = self.base_scores.copy()
        for tid, arr in (override or {}).items():
            scores[self.team_index[tid]] = arr
        r = self.retention if retention is None else float(retention)
        if r < 1.0:
            scores = scores - self._fade(r)
        return scores

    def _regular_season(self, scores: np.ndarray):
        """Wins and points after the regular season, and each head-to-head
        game's result: ``results[(week, a, b)]`` is a's win (1, 0.5, 0) per sim."""
        wins = np.array([t.wins + 0.5 * t.ties for t in self.teams], float)[:, None].repeat(self.n, 1)
        points = np.array([t.points_for for t in self.teams], float)[:, None].repeat(self.n, 1)
        results: dict[tuple[int, int, int], np.ndarray] = {}
        for k, week in enumerate(self.weeks):
            if week > self.reg_weeks:
                break
            wk = scores[:, :, k]                                               # (teams, n)
            points += wk
            for w, a, b in self.schedule:
                if w != week or a not in self.team_index or b not in self.team_index:
                    continue
                ia, ib = self.team_index[a], self.team_index[b]
                res = (wk[ia] > wk[ib]) + 0.5 * (wk[ia] == wk[ib])
                wins[ia] += res
                wins[ib] += 1.0 - res
                results[(week, ia, ib)] = res
            if self.median:
                med = np.median(wk, axis=0)
                wins += wk > med[None, :]
        return wins, points, results

    def _playoffs(self, scores: np.ndarray, wins: np.ndarray, points: np.ndarray):
        """Seeds by record then points, then the bracket: who made it, who
        had a bye, and the champion per simulation."""
        n_t = len(self.teams)
        key = wins * 1e6 + points
        order = np.argsort(-key, axis=0)                                       # (teams, n)
        p_teams = min(self.playoff_teams, n_t)
        seeds = order[:p_teams]                                                # seeds[s-1] = team idx
        if self.divisions is not None:
            seeds = division_seeds(key, self.divisions, p_teams)
        made = np.zeros((n_t, self.n), bool)
        np.put_along_axis(made, seeds, True, axis=0)
        slots = 2 ** self.rounds
        bracket = bracket_order(slots)
        byes = np.zeros((n_t, self.n), bool)
        if p_teams < slots:
            np.put_along_axis(byes, seeds[: slots - p_teams], True, axis=0)
        # Walk the bracket: entries are (seed number) -> team index per sim.
        alive = [seeds[s - 1] if s <= p_teams else None for s in bracket]
        seed_of = [s for s in bracket]
        first_playoff = self.weeks.index(self.reg_weeks + 1) if (self.reg_weeks + 1) in self.weeks else None
        for r in range(self.rounds):
            nxt, nxt_seed = [], []
            if first_playoff is None:
                break
            if self.reseed and r > 0 and len(alive) > 2:
                alive, seed_of = self._reseeded(alive, seed_of)
            ks = [first_playoff + r * self.round_weeks + i for i in range(self.round_weeks)]
            ks = [k for k in ks if k < len(self.weeks)]
            for i in range(0, len(alive), 2):
                a, b = alive[i], alive[i + 1]
                sa, sb = seed_of[i], seed_of[i + 1]
                if a is None or b is None:
                    nxt.append(a if b is None else b)
                    nxt_seed.append(sa if b is None else sb)
                    continue
                ra = sum(scores[a, np.arange(self.n), k] for k in ks)
                rb = sum(scores[b, np.arange(self.n), k] for k in ks)
                if self.playoff_points_tiebreak:
                    pa, pb = points[a, np.arange(self.n)], points[b, np.arange(self.n)]
                    a_wins = (ra > rb) | ((ra == rb) & ((pa > pb) | ((pa == pb) & (sa < sb))))
                else:
                    a_wins = (ra > rb) | ((ra == rb) & (sa < sb))
                nxt.append(np.where(a_wins, a, b))
                nxt_seed.append(np.where(a_wins, sa, sb))
            alive, seed_of = nxt, nxt_seed
        champ = alive[0] if alive and alive[0] is not None else None
        return made, byes, champ

    # -- what each game is worth ---------------------------------------
    def stakes(self, team_id: str, override: dict[str, np.ndarray] | None = None) -> Stakes:
        """For each of the team's remaining regular-season games: P(playoffs)
        and P(title) if he wins it and if he loses it, everything else in
        each simulated season held as it fell. Flipping one game moves both
        teams' records, so a game against a rival for the same playoff spot
        swings more than one against a team out of reach.

        Also the playoff odds by final win total -- what record gets in."""
        scores = self._scores(override)
        i = self.team_index[team_id]
        wins, points, results = self._regular_season(scores)
        made0, _b, champ0 = self._playoffs(scores, wins, points)
        games = []
        for (week, ia, ib), res in sorted(results.items()):
            if i not in (ia, ib) or week < int(self.snapshot.week):
                continue
            opp = ib if ia == i else ia
            mine = res if ia == i else 1.0 - res
            out = {}
            for forced in (1.0, 0.0):
                w = wins.copy()
                w[i] += forced - mine
                w[opp] -= forced - mine
                made, _byes, champ = self._playoffs(scores, w, points)
                out[forced] = (float(made[i].mean()),
                               float((champ == i).mean()) if champ is not None else 0.0)
            games.append(GameStakes(week=week, opponent=self.teams[opp].name, opponent_id=self.teams[opp].team_id,
                                    p_win=float(mine.mean()),
                                    playoffs_win=out[1.0][0], playoffs_loss=out[0.0][0],
                                    title_win=out[1.0][1], title_loss=out[0.0][1]))
        final = np.rint(wins[i]).astype(int)
        by_wins = {int(k): (float(made0[i][final == k].mean()), int((final == k).sum()))
                   for k in np.unique(final)}
        return Stakes(games=games, by_wins=by_wins, p_playoffs=float(made0[i].mean()),
                      exp_wins=float(wins[i].mean()), n_sims=self.n)

    def _reseeded(self, alive: list, seed_of: list) -> tuple[list, list]:
        """Survivors re-paired by seed in every simulation: best left against
        worst left, second-best against second-worst, and so on."""
        teams = np.stack([np.broadcast_to(np.asarray(a), (self.n,)) for a in alive])
        seeds = np.stack([np.broadcast_to(np.asarray(s), (self.n,)) for s in seed_of])
        order = np.argsort(seeds, axis=0, kind="stable")
        teams = np.take_along_axis(teams, order, axis=0)
        seeds = np.take_along_axis(seeds, order, axis=0)
        m = len(alive)
        pairing = [x for i in range(m // 2) for x in (i, m - 1 - i)]
        return [teams[i] for i in pairing], [seeds[i] for i in pairing]


def season_odds(snapshot: LeagueSnapshot, cfg: Config, n_sims: int = 2000) -> SeasonOdds | None:
    """P(playoffs) and P(title) for every team, or None if the league's
    structure (schedule, playoff format) has not been read."""
    rules = snapshot.rules or {}
    if not rules.get("schedule") or not rules.get("playoff_teams"):
        return None
    return SeasonModel(snapshot, cfg, n_sims=n_sims).odds()
