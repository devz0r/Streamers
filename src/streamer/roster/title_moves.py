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
from .waivers import MIN_KEEP, handcuffs

SKILL = ("QB", "RB", "WR", "TE")

#: A breakout, for the wait-and-claim option: his visible level beats the
#: player he would replace by this much and has risen this much from today.
BREAKOUT_MARGIN = 2.0
#: Title-odds gains below this (in probability) are not worth a transaction.
MIN_GAIN = 0.002
#: The most a rival landing one breakout has been seen to cost (0.1 points
#: of title odds in the ESPN league at week 4), with room to spare: a move
#: further short than this on its own merits is not priced for blocking.
BLOCK_ALLOWANCE = 0.003
#: Blocking is named on a card only when it is worth at least this much:
#: smaller values sit inside the simulation's noise.
BLOCK_NOTE = 0.003
#: Two drops this close in title odds are a coin flip for the season; the
#: tie goes to keeping the player the market values more -- he can still be
#: traded, and a rival would claim him. About the paired noise of one move.
DROP_TIE = 0.002
#: Upside plays priced on purpose, of each kind (handcuffs, rookies, rising roles).
UPSIDE_EACH = 3


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
    #: P(title) if he stayed on the wire all season (nobody claims him) --
    #: ``p_base`` is standing pat with rivals free to claim his breakout.
    p_free: float = 0.0
    #: Share of seasons a rival lands him if you pass, and who most often.
    rival_share: float = 0.0
    rival_name: str = ""

    @property
    def block_value(self) -> float:
        """Title odds you lose when a rival, not the wire, keeps him."""
        return self.p_free - self.p_base

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
                 candidates: int = 20, drops: int = 4, seed: int = 29):
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
        self.upside = self._upside_candidates(snapshot, pool)
        have = {p.player_id for p in self.candidates}
        self.candidates.extend(p for p in pool if p.player_id in self.upside and p.player_id not in have)
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
        # Waiver order for every team (1 = first claim); unknown ranks go last.
        self.ranks = np.array([int(t.waiver_rank) if t.waiver_rank else self.n_teams
                               for t in snapshot.teams], float)
        self.ranks[self.mine] = self.rank
        self._seed = seed
        self._claimant_cache: dict[str, np.ndarray] = {}
        self._stand_pat_cache: dict[tuple[str, str], np.ndarray] = {}
        #: Player id -> what the market sees him worth (perception); set by
        #: the caller. Used to keep tradeable players when drops tie.
        self.market: dict[str, float] = {}

    def market_value(self, p: PlayerRow) -> float:
        return float(self.market.get(p.player_id, p.ros_value or 0.0))

    @staticmethod
    def _upside_candidates(snapshot: LeagueSnapshot, pool: list[PlayerRow]) -> dict[str, str]:
        """Upside a ranking by today's value never reaches: backs one injury
        from a lead role, rookies, players whose opportunity just rose. Their
        average is low, so they are added to the list on purpose; what the
        upside is worth is left to P(title) -- more when you are behind, when
        only the seasons where something breaks right end in a title."""
        out: dict[str, str] = {}
        ids = {p.player_id for p in pool}
        cuffs = sorted(((pid, lead, gain) for pid, (lead, gain) in handcuffs(snapshot).items() if pid in ids),
                       key=lambda t: -t[2])
        for pid, lead, _gain in cuffs[:UPSIDE_EACH]:
            out[pid] = f"next in line behind {lead.name}"
        by_value = sorted(pool, key=lambda p: -float(p.ros_value or 0.0))
        for p in [q for q in by_value if q.experience == 0 and q.player_id not in out][:UPSIDE_EACH]:
            out[p.player_id] = "rookie: roles grow, and his outlook moves more"
        rising = [q for q in by_value if q.player_id not in out
                  and any(s.startswith("opportunity up") for s in q.signals)]
        for p in rising[:UPSIDE_EACH]:
            out[p.player_id] = "his role is growing"
        return out

    # -- building blocks -------------------------------------------------
    def _title(self, roster_ids: list[str], **kw) -> float:
        return float(self._won(roster_ids, **kw).mean())

    def _won(self, roster_ids: list[str], rivals: dict[str, np.ndarray] | None = None, **kw) -> np.ndarray:
        """Per simulated season: 1 if you win the title with this roster
        (and ``rivals`` -- other teams' scores after they land a player)."""
        scores = self.model.team_scores(roster_ids, owner=self.me.team_id, **kw)
        champ = self.model.odds(override={**(rivals or {}), self.me.team_id: scores}).champion
        return (champ == self.mine).astype(float)

    # -- rivals on the wire ---------------------------------------------
    def _claimant(self, add: PlayerRow) -> np.ndarray:
        """Per simulation: the rival (team index) whose claim would win him on
        a breakout if you do not claim, or -1 when no rival claims. Each rival
        claims at the league's activity rate; among claimants the best waiver
        priority wins. Drawn per player (seeded by who he is), so the same
        rival does not land every breakout in a season."""
        if add.player_id in self._claimant_cache:
            return self._claimant_cache[add.player_id]
        idx = self.model.pid.get(add.player_id, 0)
        rng = np.random.default_rng(self._seed * 1000 + 7 + idx)
        claims = rng.random((self.model.n, self.n_teams)) < self.activity
        claims[:, self.mine] = False
        masked = np.where(claims, self.ranks[None, :], np.inf)
        who = np.where(claims.any(axis=1), masked.argmin(axis=1), -1)
        self._claimant_cache[add.player_id] = who
        return who

    def _rival_scores(self, add: PlayerRow, joins_at: np.ndarray, gets: np.ndarray,
                      claimant: np.ndarray) -> dict[str, np.ndarray]:
        """Weekly scores of every rival who lands him (in the seasons ``gets``),
        from the week after his breakout, in place of his weakest player."""
        out = {}
        for t_idx in np.unique(claimant[gets & (claimant >= 0)]):
            team = self.snapshot.teams[int(t_idx)]
            joins = np.where(gets & (claimant == t_idx), joins_at, 10_000)
            ids = [p.player_id for p in team.roster] + [add.player_id]
            worst = sorted((p for p in team.roster if not p.in_ir_slot and p.position in SKILL),
                           key=lambda p: float(p.ros_value or 0.0))
            gone = {worst[0].player_id: joins} if worst else {}
            out[team.team_id] = self.model.team_scores(ids, available_from={add.player_id: joins},
                                                       gone_from=gone, owner=team.team_id)
        return out

    def stand_pat(self, add: PlayerRow, drop: PlayerRow) -> np.ndarray:
        """Per season: you win the title if you pass on him, and a rival
        claims him when he breaks out."""
        key = (add.player_id, drop.player_id)
        if key not in self._stand_pat_cache:
            k = self._breakout_week(add, drop)
            breaks = k <= len(self.model.weeks)
            claimant = self._claimant(add)
            rivals = self._rival_scores(add, k + 1, breaks, claimant)
            champ = self.model.odds(override=rivals).champion if rivals else self.model.odds().champion
            self._stand_pat_cache[key] = (champ == self.mine).astype(float)
        return self._stand_pat_cache[key]

    def _breakout_week(self, add: PlayerRow, drop: PlayerRow) -> np.ndarray:
        """First week index (>= 1) each simulation shows him breaking out, or
        a week past the season when he never does."""
        lv = self.model.futures.levels
        xa, xd = self.model.pid[add.player_id], self.model.pid.get(drop.player_id)
        lx = lv[:, xa, :]
        ld = lv[:, xd, :] if xd is not None else np.zeros_like(lx)
        start = float(add.ros_value or 0.0)
        hit = (lx >= ld + BREAKOUT_MARGIN) & (lx >= start + BREAKOUT_MARGIN)
        hit[:, : self.model.lag + 1] = False         # this week is already being claimed on
        k = np.where(hit.any(axis=1), hit.argmax(axis=1), lv.shape[2] + 1)
        return k

    def value_wait(self, add: PlayerRow, drop: PlayerRow, rank: int, per_sim: bool = False):
        """P(title) if you hold, and claim him the week after he breaks out.
        You get him when no rival who also claims holds a better priority;
        otherwise that rival does."""
        roster = [p.player_id for p in self.me.roster]
        k = self._breakout_week(add, drop)
        breaks = k <= len(self.model.weeks)
        claimant = self._claimant(add)
        rival_rank = np.where(claimant >= 0, self.ranks[np.maximum(claimant, 0)], np.inf)
        got = breaks & (rank < rival_rank)
        joins = np.where(got, k + 1, 10_000)
        rivals = self._rival_scores(add, k + 1, breaks & ~got, claimant)
        won = self._won(roster + [add.player_id], rivals=rivals, available_from={add.player_id: joins},
                        gone_from={drop.player_id: joins})
        return won if per_sim else float(won.mean())

    def value_now(self, add: PlayerRow, drop: PlayerRow, per_sim: bool = False):
        roster = [p.player_id for p in self.me.roster if p.player_id != drop.player_id] + [add.player_id]
        won = self._won(roster)
        return won if per_sim else float(won.mean())

    def priority_cost(self, exclude: str) -> float:
        """Title odds lost on the best future contested claim if you drop to
        the back of the order now (the claim in question excluded)."""
        gaps = self._priority_gaps()
        return max((g for pid, g in gaps.items() if pid != exclude), default=0.0)

    def _priority_gaps(self) -> dict[str, float]:
        """For the top free agents: what your current rank is worth over last
        place on a wait-and-claim for him. Computed once and shared."""
        if getattr(self, "_gaps", None) is not None:
            return self._gaps
        self._gaps = {}
        if not self.priority_waivers or self.rank >= self.n_teams:
            return self._gaps
        worst = sorted([p for p in self.me.roster if not p.in_ir_slot and p.position in SKILL],
                       key=lambda p: float(p.ros_value or 0.0))
        if not worst:
            return self._gaps
        drop = worst[0]
        for z in self.candidates[:6]:
            self._gaps[z.player_id] = max(
                self.value_wait(z, drop, self.rank) - self.value_wait(z, drop, self.n_teams), 0.0)
        return self._gaps

    # -- the recommendation ----------------------------------------------
    def moves(self, n: int = 5) -> list[TitleMove]:
        out: list[TitleMove] = []
        for x in self.candidates:
            best: TitleMove | None = None
            priced = [(y, self.value_now(x, y)) for y in _droppable(self.me.roster, x)[: self.n_drops]]
            if priced:
                top = max(p for _y, p in priced)
                # As good for the title, within noise: drop whoever the market values least.
                ties = [(y, p) for y, p in priced if p >= top - DROP_TIE]
                y, p_now = min(ties, key=lambda t: (self.market_value(t[0]), -t[1]))
                best = TitleMove(add=x, drop=y, p_now=p_now, p_wait=0.0, p_base=self.base,
                                 priority_cost=0.0, verdict="skip")
            if best is None or best.p_now - self.base < MIN_GAIN - BLOCK_ALLOWANCE:
                continue                  # too far short for blocking to rescue
            # Standing pat is not "he stays on the wire": if he breaks out, a
            # rival claims him. What that costs you is part of adding him now.
            pat = self.stand_pat(x, best.drop)
            best.p_free, best.p_base = self.base, float(pat.mean())
            if best.gain_now < MIN_GAIN:
                continue
            claimant = self._claimant(x)
            breaks = self._breakout_week(x, best.drop) <= len(self.model.weeks)
            lands = breaks & (claimant >= 0)
            best.rival_share = float(lands.mean())
            if lands.any():
                top = np.bincount(claimant[lands], minlength=self.n_teams).argmax()
                best.rival_name = self.snapshot.teams[int(top)].name
            now = self.value_now(x, best.drop, per_sim=True)
            wait = self.value_wait(x, best.drop, self.rank, per_sim=True)
            ref = wait if wait.mean() >= pat.mean() else pat
            best.p_wait = float(wait.mean())
            best.noise = float((now - ref).std() / np.sqrt(len(now)))
            best.priority_cost = self.priority_cost(x.player_id)
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
        if m.block_value >= BLOCK_NOTE and m.rival_name:
            bits.append(f"if you pass, a rival lands him in {m.rival_share:.0%} of seasons (most often "
                        f"{m.rival_name}), which costs you {m.block_value * 100:.1f} of that")
        if m.add.player_id in self.upside:
            bits.append(f"upside play: {self.upside[m.add.player_id]}")
        if m.add.ecr_ros_pos_rank:
            from .consensus import note, our_ranks

            if getattr(self, "_our_ranks", None) is None:
                self._our_ranks = our_ranks(self.snapshot)
            second = note(m.add, self._our_ranks, self.cfg)
            if second:
                bits.append(second)
        bits.extend(m.add.signals)
        if m.add.usage:
            bits.append(m.add.usage)
        return bits
