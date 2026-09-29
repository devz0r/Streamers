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
    #: Champion's team index in each simulated season (for paired comparisons).
    champion: np.ndarray | None = None
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


#: Weekly spread of a replacement-level pickup's score.
REPLACEMENT_SD = 6.0


def replacement_levels(snapshot: LeagueSnapshot) -> dict[str, float]:
    """Per position: the second-best free agent's per-game value (D/ST and
    K: the best, since they are streamed weekly)."""
    out = {}
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        vals = sorted((float(p.ros_value if p.ros_value is not None else (p.projection or 0.0))
                       for p in snapshot.free_agents if p.position == pos and not p.is_long_term_out
                       and p.team), reverse=True)
        if vals:
            out[pos] = vals[0] if pos in ("K", "DST") else vals[min(1, len(vals) - 1)]
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
        self.reseed = bool(rules.get("reseed"))
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
        self.futures = fut.simulate(self.players, self.weeks, byes, n_sims=n_sims, seed=seed,
                                    heirs=heirs_for(everyone), current=self.lag,
                                    takeover=roster_conf.get("next_man_up"),
                                    takeover_sd=roster_conf.get("next_man_up_spread"),
                                    kept=roster_conf.get("next_man_up_kept"))
        self._lv_t = np.ascontiguousarray(self.futures.levels.transpose(0, 2, 1), dtype=np.float32)
        self._sc_t = np.ascontiguousarray(self.futures.scores.transpose(0, 2, 1), dtype=np.float32)
        self.slots = [(s, c) for s, c in sorted(snapshot.starting_slots.items(),
                      key=lambda kv: (len(SLOT_ELIGIBILITY.get(kv[0], (kv[0],))), kv[0]))]
        # Replacement level: what any team can pick up off the wire in a
        # given week -- the second-best free agent at each position. A slot
        # whose best rostered option is worse (a bye, an injury, no backup)
        # is filled at that level instead of scoring zero, which is what a
        # real manager does and what stops a frozen roster from overvaluing
        # bench depth.
        self.replacement = replacement_levels(snapshot)
        noise_rng = np.random.default_rng(seed + 7)
        self.rep_noise = {slot: noise_rng.standard_normal((n_sims, len(self.weeks))) for slot, _c in self.slots}
        self.base_scores = np.stack([self.team_scores([p.player_id for p in t.roster]) for t in self.teams])

    # -- lineups ---------------------------------------------------------
    def team_scores(self, roster_ids: list[str], available_from: dict[str, int] | None = None,
                    gone_from: dict[str, int] | None = None, owner: str | None = None) -> np.ndarray:
        """``(n_sims, weeks)`` points from starting the best lineup each week.

        ``available_from`` / ``gone_from`` map a player to the week index he
        joins / leaves the roster, per simulation (arrays) or for all.
        """
        if owner is not None:
            roster_ids, available_from, gone_from = self._this_week_is_settled(
                owner, list(roster_ids), dict(available_from or {}), dict(gone_from or {}))
        ids = [i for i in roster_ids if i in self.pid]
        if not ids:
            return np.zeros((self.n, len(self.weeks)))
        cols = [self.pid[i] for i in ids]
        # (sims, weeks, players): players on the last, contiguous axis, where
        # the per-slot argmax runs -- several times faster than the middle.
        lv = self._lv_t[:, :, cols]
        sc = self._sc_t[:, :, cols]
        k_idx = np.arange(len(self.weeks))[None, :]
        for table, joins in ((available_from, True), (gone_from, False)):
            for pid, when in (table or {}).items():
                if pid not in ids:
                    continue
                j = ids.index(pid)
                when = np.asarray(when)
                w = when[:, None] if when.ndim else when
                on_roster = (k_idx >= w) if joins else (k_idx < w)
                lv[:, :, j] = np.where(on_roster, lv[:, :, j], -1.0)
        players = [self.players[c] for c in cols]
        used = np.zeros(lv.shape, bool)
        total = np.zeros((self.n, len(self.weeks)))
        slot_ix = np.arange(len(players))[None, None, :]
        for slot, count in self.slots:
            elig = np.array([_eligible(p, slot) for p in players])
            rep = self.slot_replacement(slot)
            rep_score = np.maximum(rep + REPLACEMENT_SD * self.rep_noise[slot], 0.0) if rep > 0 else 0.0
            for _ in range(count):
                if not elig.any():
                    total += rep_score
                    continue
                masked = np.where(elig[None, None, :] & ~used, lv, np.float32(-2.0))
                pick = masked.argmax(axis=2)                                  # (n, k)
                best = np.take_along_axis(masked, pick[:, :, None], axis=2)[:, :, 0]
                ok = best >= max(rep, 0.0) + 1e-6                             # beats the wire
                ok &= best > -0.5
                gained = np.take_along_axis(sc, pick[:, :, None], axis=2)[:, :, 0]
                total += np.where(ok, gained, rep_score)
                used |= (slot_ix == pick[:, :, None]) & ok[:, :, None]
        return total

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

    def slot_replacement(self, slot: str) -> float:
        positions = SLOT_ELIGIBILITY.get(slot, (slot,))
        return max((self.replacement.get(pos, 0.0) for pos in positions), default=0.0)

    # -- the season ------------------------------------------------------
    def odds(self, override: dict[str, np.ndarray] | None = None) -> SeasonOdds:
        """Play out every simulated season. ``override`` swaps in a team's
        weekly scores (from :meth:`team_scores` on a changed roster)."""
        scores = self._scores(override)
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
            champion=np.asarray(champ) if champ is not None else None, notes=list(self.notes))

    def _scores(self, override: dict[str, np.ndarray] | None) -> np.ndarray:
        scores = self.base_scores.copy()
        for tid, arr in (override or {}).items():
            scores[self.team_index[tid]] = arr
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
