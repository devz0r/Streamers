"""Waiver moves valued by what they do to P(win the title).

Three questions, each answered on the same simulated futures (so the
difference between two answers is the decision, not noise):

1. **Is he worth more than the player he replaces?** P(title) with "add X,
   drop Y" from now on, against the roster as it stands. Y's upside counts
   too: a drop gives up every future in which he would have mattered.
2. **Could you wait?** In every simulated future where X breaks out -- his
   visible level climbs past Y's and his own starting point -- you can put
   in a claim then. Whether you get him depends on how many other managers
   claim him too (the league's own pickup rate says how active it is) and
   whether your priority beats theirs.
3. **Is it worth your priority?** Claiming now sends you to the back of a
   rolling waiver order. The cost is the future contested claims you would
   then lose: the most valuable wait-and-claim among the other free agents,
   priced at your current rank against last place.

A claim is recommended only when adding now beats waiting by more than the
priority it burns. "Worth adding, not worth priority" is said out loud: the
player is an upgrade, but the simulation expects you could still land him
when it matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np

from ..config import Config
from ..league.model import LeagueSnapshot, PlayerRow
from .season import SeasonModel
from .waivers import MIN_KEEP

SKILL = ("QB", "RB", "WR", "TE")

#: A breakout, for the wait-and-claim option: his visible level beats the
#: player he would replace by this much and has risen this much from today.
BREAKOUT_MARGIN = 2.0
#: Title-odds gains below this (in probability) are not worth a transaction.
MIN_GAIN = 0.002


@dataclass
class TitleMove:
    add: PlayerRow
    drop: PlayerRow
    p_now: float              # P(title) adding him now
    p_wait: float             # P(title) waiting, claiming on a breakout
    p_base: float             # P(title) standing pat
    priority_cost: float      # P(title) lost on future claims by spending priority now
    verdict: str              # "claim", "lean", "wait"
    reasons: list[str] = field(default_factory=list)
    #: Simulation noise on the claim-now edge (paired standard error).
    noise: float = 0.0

    @property
    def gain_now(self) -> float:
        return self.p_now - self.p_base

    @property
    def net(self) -> float:
        return self.p_now - max(self.p_wait, self.p_base) - self.priority_cost


#: Of the active managers, the share who would put in a claim for any one
#: breakout (they need the position and have a spot to open). An assumption,
#: stated on the page: about 2-3 rival claims for a real breakout in an
#: active ten-team league. Failed claims are not published, so it cannot be
#: fitted from transaction logs.
CLAIM_INTEREST = 0.35


def league_activity(snapshot: LeagueSnapshot) -> float:
    """Chance a given rival claims a breakout: the share of teams that have
    made at least one pickup a week so far (clipped to 0.3-0.9), times
    :data:`CLAIM_INTEREST`."""
    weeks = max(int(snapshot.week) - 1, 1)
    teams = snapshot.teams
    if not teams or all(t.acquisitions is None for t in teams):
        return 0.6 * CLAIM_INTEREST
    active = sum(1 for t in teams if (t.acquisitions or 0) >= weeks)
    return float(np.clip(active / len(teams), 0.3, 0.9)) * CLAIM_INTEREST


def p_win_claim(rank: int, n_teams: int, activity: float) -> float:
    """Chance your claim beats every other claimant for a breakout.

    Each other team claims with probability ``activity``; claimants hold
    random places in the order; you win if all of them rank behind you."""
    others = n_teams - 1
    worse = n_teams - rank                       # teams behind you
    total = 0.0
    for m in range(others + 1):
        p_m = comb(others, m) * activity ** m * (1 - activity) ** (others - m)
        win = comb(worse, m) / comb(others, m) if m <= worse else 0.0
        total += p_m * win
    return total


def _droppable(roster: list[PlayerRow], add: PlayerRow) -> list[PlayerRow]:
    counts: dict[str, int] = {}
    for p in roster:
        counts[p.position] = counts.get(p.position, 0) + 1
    out = []
    for p in roster:
        if p.in_ir_slot:
            continue
        if p.position != add.position and counts.get(p.position, 0) - 1 < MIN_KEEP.get(p.position, 0):
            continue
        out.append(p)
    return sorted(out, key=lambda p: float(p.ros_value or 0.0) + float(p.ros_sd or 0.0))


class TitleEngine:
    def __init__(self, snapshot: LeagueSnapshot, cfg: Config, n_sims: int = 6000,
                 candidates: int = 20, drops: int = 6, seed: int = 29):
        self.snapshot, self.cfg = snapshot, cfg
        me = snapshot.my_team
        self.me = me
        pool = [p for p in snapshot.free_agents
                if p.position in SKILL and not p.is_long_term_out and p.team and p.ros_value is not None]
        # Best free agents per position by their upside (per-game value plus
        # how far it could move), so backup quarterbacks cannot crowd out the
        # backs and receivers who usually matter.
        quota = {"RB": 7, "WR": 7, "QB": 3, "TE": 3}
        self.candidates = []
        for pos, n in quota.items():
            at = sorted((p for p in pool if p.position == pos),
                        key=lambda p: -(float(p.ros_value or 0.0) + float(p.ros_sd or 0.0)))
            self.candidates.extend(at[:max(1, round(n * candidates / 20))])
        self.n_drops = drops
        self.model = SeasonModel(snapshot, cfg, n_sims=n_sims, seed=seed, extra_players=self.candidates)
        self.mine = self.model.team_index[me.team_id]
        self.base_won = (self.model.odds().champion == self.mine).astype(float)
        self.base = float(self.base_won.mean())
        self.n_teams = len(snapshot.teams)
        self.rank = int(me.waiver_rank or self.n_teams)
        self.activity = league_activity(snapshot)
        rules = snapshot.rules or {}
        # Only a rolling list charges for a claim (you go to the back); an
        # order reset by standings each week, or FAAB, does not.
        self.priority_waivers = (rules.get("waiver", "priority") == "priority"
                                 and rules.get("waiver_order", "rolling") == "rolling")
        # One set of claim-success draws for every candidate: common random
        # numbers, so candidates differ by who they are, not by luck.
        self.u = np.random.default_rng(seed + 1).random(n_sims)

    # -- building blocks -------------------------------------------------
    def _title(self, roster_ids: list[str], **kw) -> float:
        return float(self._won(roster_ids, **kw).mean())

    def _won(self, roster_ids: list[str], **kw) -> np.ndarray:
        """Per simulated season: 1 if you win the title with this roster."""
        scores = self.model.team_scores(roster_ids, owner=self.me.team_id, **kw)
        champ = self.model.odds(override={self.me.team_id: scores}).champion
        return (champ == self.mine).astype(float)

    def _breakout_week(self, add: PlayerRow, drop: PlayerRow) -> np.ndarray:
        """First week index (>= 1) each simulation shows him breaking out, or
        a week past the season when he never does."""
        lv = self.model.futures.levels
        xa, xd = self.model.pid[add.player_id], self.model.pid.get(drop.player_id)
        lx = lv[:, xa, :]
        ld = lv[:, xd, :] if xd is not None else np.zeros_like(lx)
        start = float(add.ros_value or 0.0)
        hit = (lx >= ld + BREAKOUT_MARGIN) & (lx >= start + BREAKOUT_MARGIN)
        hit[:, 0] = False                            # this week is already being claimed on
        k = np.where(hit.any(axis=1), hit.argmax(axis=1), lv.shape[2] + 1)
        return k

    def value_wait(self, add: PlayerRow, drop: PlayerRow, rank: int, per_sim: bool = False):
        """P(title) if you hold, and claim him the week after he breaks out."""
        roster = [p.player_id for p in self.me.roster]
        k = self._breakout_week(add, drop)
        p_win = p_win_claim(rank, self.n_teams, self.activity)
        got = self.u < p_win
        joins = np.where(got, k + 1, 10_000)
        won = self._won(roster + [add.player_id], available_from={add.player_id: joins},
                        gone_from={drop.player_id: joins})
        return won if per_sim else float(won.mean())

    def value_now(self, add: PlayerRow, drop: PlayerRow, per_sim: bool = False):
        roster = [p.player_id for p in self.me.roster if p.player_id != drop.player_id] + [add.player_id]
        won = self._won(roster)
        return won if per_sim else float(won.mean())

    def priority_cost(self, exclude: str) -> float:
        """Title odds lost on the best future contested claim if you drop to
        the back of the order now."""
        if not self.priority_waivers or self.rank >= self.n_teams:
            return 0.0
        worst = sorted([p for p in self.me.roster if not p.in_ir_slot and p.position in SKILL],
                       key=lambda p: float(p.ros_value or 0.0))
        if not worst:
            return 0.0
        drop = worst[0]
        best = 0.0
        for z in self.candidates[:6]:
            if z.player_id == exclude:
                continue
            gap = self.value_wait(z, drop, self.rank) - self.value_wait(z, drop, self.n_teams)
            best = max(best, gap)
        return best

    # -- the recommendation ----------------------------------------------
    def moves(self, n: int = 5) -> list[TitleMove]:
        out: list[TitleMove] = []
        cost_cache: dict[str, float] = {}
        for x in self.candidates:
            best: TitleMove | None = None
            for y in _droppable(self.me.roster, x)[: self.n_drops]:
                p_now = self.value_now(x, y)
                if best is not None and p_now <= best.p_now:
                    continue
                best = TitleMove(add=x, drop=y, p_now=p_now, p_wait=0.0, p_base=self.base,
                                 priority_cost=0.0, verdict="skip")
            if best is None or best.gain_now < MIN_GAIN:
                continue
            now = self.value_now(x, best.drop, per_sim=True)
            wait = self.value_wait(x, best.drop, self.rank, per_sim=True)
            ref = wait if wait.mean() >= self.base else self.base_won
            best.p_wait = float(wait.mean())
            best.noise = float((now - ref).std() / np.sqrt(len(now)))
            if x.player_id not in cost_cache:
                cost_cache[x.player_id] = self.priority_cost(x.player_id)
            best.priority_cost = cost_cache[x.player_id]
            if best.net >= max(MIN_GAIN, 2.0 * best.noise):
                best.verdict = "claim"
            elif best.net > 0:
                best.verdict = "lean"
            else:
                best.verdict = "wait"
            best.reasons = self._reasons(best)
            out.append(best)
        rank = {"claim": 0, "lean": 1, "wait": 2}
        out.sort(key=lambda m: (rank[m.verdict], -m.net if m.verdict != "wait" else -m.gain_now))
        return out[:n]

    def _reasons(self, m: TitleMove) -> list[str]:
        pts = lambda v: f"{v * 100:+.1f}"      # noqa: E731
        bits = [f"{pts(m.gain_now)} points of title odds adding him now",
                f"{pts(m.p_wait - m.p_base)} if you wait and claim him when he breaks out"]
        if m.priority_cost > 0:
            bits.append(f"your #{self.rank} priority is worth {m.priority_cost * 100:.1f} on future claims")
        if m.verdict == "lean":
            bits.append(f"edge {m.net * 100:+.2f} is inside the simulation's noise (+-{2 * m.noise * 100:.2f})")
        bits.extend(m.add.signals)
        if m.add.usage:
            bits.append(m.add.usage)
        return bits
