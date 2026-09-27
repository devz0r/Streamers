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
* the playoffs are a fixed bracket (byes for the top seeds when the field is
  not a power of two), each round as many weeks as the league plays.

P(title) is the share of seasons a team wins it all. Moves are valued by
re-running only the changed team's lineups on the same simulated futures
(common random numbers), so the difference is the move, not noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..league.model import SLOT_ELIGIBILITY, LeagueSnapshot, PlayerRow
from . import futures as fut

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
    notes: list[str] = field(default_factory=list)

    def of(self, team_id: str) -> dict:
        i = self.team_ids.index(team_id)
        return {"p_playoffs": float(self.p_playoffs[i]), "p_bye": float(self.p_bye[i]),
                "p_title": float(self.p_title[i]), "exp_wins": float(self.exp_wins[i])}


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


def heirs_for(players: list[PlayerRow]) -> list[tuple[str, str]]:
    """(lead, next man up) at RB/QB/TE on each NFL team, by healthy value."""
    groups: dict[tuple[str, str], list[PlayerRow]] = {}
    for p in players:
        if p.position in fut.TAKEOVER and p.team:
            groups.setdefault((p.team, p.position), []).append(p)
    out = []
    for members in groups.values():
        members = sorted(members, key=lambda q: -float(q.ros_value or 0.0))
        if len(members) >= 2:
            out.append((members[0].player_id, members[1].player_id))
    return out


def _eligible(p: PlayerRow, slot: str) -> bool:
    if p.eligible_slots and slot in p.eligible_slots:
        return True
    return p.position in SLOT_ELIGIBILITY.get(slot, ())


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
        self.rounds = max(math.ceil(math.log2(max(self.playoff_teams, 2))), 1)
        start = int(snapshot.week)
        self.last_week = self.reg_weeks + self.rounds * self.round_weeks
        self.weeks = list(range(start, self.last_week + 1))
        self.schedule = [(int(w), str(a), str(b)) for w, a, b in rules.get("schedule", [])
                         if start <= int(w) <= self.reg_weeks]
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
        self.futures = fut.simulate(self.players, self.weeks, byes, n_sims=n_sims, seed=seed,
                                    heirs=heirs_for(everyone))
        self.slots = [(s, c) for s, c in sorted(snapshot.starting_slots.items(),
                      key=lambda kv: (len(SLOT_ELIGIBILITY.get(kv[0], (kv[0],))), kv[0]))]
        self.base_scores = np.stack([self.team_scores([p.player_id for p in t.roster]) for t in self.teams])

    # -- lineups ---------------------------------------------------------
    def team_scores(self, roster_ids: list[str], available_from: dict[str, int] | None = None,
                    gone_from: dict[str, int] | None = None) -> np.ndarray:
        """``(n_sims, weeks)`` points from starting the best lineup each week.

        ``available_from`` / ``gone_from`` map a player to the week index he
        joins / leaves the roster, per simulation (arrays) or for all.
        """
        ids = [i for i in roster_ids if i in self.pid]
        if not ids:
            return np.zeros((self.n, len(self.weeks)))
        cols = [self.pid[i] for i in ids]
        lv = self.futures.levels[:, cols, :].copy()
        sc = self.futures.scores[:, cols, :]
        k_idx = np.arange(len(self.weeks))[None, :]
        for table, joins in ((available_from, True), (gone_from, False)):
            for pid, when in (table or {}).items():
                if pid not in ids:
                    continue
                j = ids.index(pid)
                when = np.asarray(when)
                w = when[:, None] if when.ndim else when
                on_roster = (k_idx >= w) if joins else (k_idx < w)
                lv[:, j, :] = np.where(on_roster, lv[:, j, :], -1.0)
        players = [self.players[c] for c in cols]
        used = np.zeros(lv.shape, bool)
        total = np.zeros((self.n, len(self.weeks)))
        for slot, count in self.slots:
            elig = np.array([_eligible(p, slot) for p in players])
            if not elig.any():
                continue
            for _ in range(count):
                masked = np.where(elig[None, :, None] & ~used, lv, -2.0)
                pick = masked.argmax(axis=1)                                  # (n, k)
                best = np.take_along_axis(masked, pick[:, None, :], axis=1)[:, 0, :]
                ok = best > -0.5                                              # someone rostered
                gained = np.take_along_axis(sc, pick[:, None, :], axis=1)[:, 0, :]
                total += np.where(ok, gained, 0.0)
                used |= (np.arange(len(players))[None, :, None] == pick[:, None, :]) & ok[:, None, :]
        return total

    # -- the season ------------------------------------------------------
    def odds(self, override: dict[str, np.ndarray] | None = None) -> SeasonOdds:
        """Play out every simulated season. ``override`` swaps in a team's
        weekly scores (from :meth:`team_scores` on a changed roster)."""
        scores = self.base_scores.copy()
        for tid, arr in (override or {}).items():
            scores[self.team_index[tid]] = arr
        n_t = len(self.teams)
        wins = np.array([t.wins + 0.5 * t.ties for t in self.teams], float)[:, None].repeat(self.n, 1)
        points = np.array([t.points_for for t in self.teams], float)[:, None].repeat(self.n, 1)
        for k, week in enumerate(self.weeks):
            if week > self.reg_weeks:
                break
            wk = scores[:, :, k]                                               # (teams, n)
            points += wk
            for w, a, b in self.schedule:
                if w != week or a not in self.team_index or b not in self.team_index:
                    continue
                ia, ib = self.team_index[a], self.team_index[b]
                wins[ia] += (wk[ia] > wk[ib]) + 0.5 * (wk[ia] == wk[ib])
                wins[ib] += (wk[ib] > wk[ia]) + 0.5 * (wk[ia] == wk[ib])
            if self.median:
                med = np.median(wk, axis=0)
                wins += wk > med[None, :]
        # Seeds: record, then points.
        key = wins * 1e6 + points
        order = np.argsort(-key, axis=0)                                       # (teams, n)
        p_teams = min(self.playoff_teams, n_t)
        seeds = order[:p_teams]                                                # seeds[s-1] = team idx
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
                a_wins = (ra > rb) | ((ra == rb) & (sa < sb))
                nxt.append(np.where(a_wins, a, b))
                nxt_seed.append(np.where(a_wins, sa, sb))
            alive, seed_of = nxt, nxt_seed
        champ = alive[0] if alive and alive[0] is not None else None
        title = np.zeros(n_t)
        if champ is not None:
            title = np.bincount(np.asarray(champ).ravel(), minlength=n_t) / self.n
        mine = next((i for i, t in enumerate(self.teams) if t.is_mine), None)
        return SeasonOdds(
            team_ids=[t.team_id for t in self.teams], names=[t.name for t in self.teams],
            records=[f"{t.wins}-{t.losses}" + (f"-{t.ties}" if t.ties else "") for t in self.teams],
            p_playoffs=made.mean(axis=1), p_bye=byes.mean(axis=1), p_title=title,
            exp_wins=wins.mean(axis=1), n_sims=self.n, mine=mine)


def season_odds(snapshot: LeagueSnapshot, cfg: Config, n_sims: int = 2000) -> SeasonOdds | None:
    """P(playoffs) and P(title) for every team, or None if the league's
    structure (schedule, playoff format) has not been read."""
    rules = snapshot.rules or {}
    if not rules.get("schedule") or not rules.get("playoff_teams"):
        return None
    return SeasonModel(snapshot, cfg, n_sims=n_sims).odds()
