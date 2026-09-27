"""Game-time lottery tickets.

Yahoo lets a bench player be dropped after his game has kicked off. That
makes a rotating roster spot worth something: pick up a free agent before
each kickoff window, keep him if his value jumps during the game, and
otherwise drop him for the next window's ticket.

A ticket earns its keep in one of two ways, and both are measured:

* **A big game.** The chance he has a top-N week at his position, N the
  number of teams (see :func:`starter_bars`), read off
  the same outcome distribution the lineup simulator uses (walk-forward, its
  ranges land where they claim: 71% inside the 70% band).
* **The job opens up.** A back next in line behind a lead back. Lead backs
  miss the following game 8.5% of the time (2021-2025), and when they do the
  next man up averaged 13.9 points.

Tickets are ranked by the chance either happens. That is a chance of
*holding* a player worth keeping, not a promise the value lasts: one-game
spikes without more opportunity mostly fade, so a boom is the start of a
look, not the end of one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from ..config import Config
from ..league.model import LeagueSnapshot, PlayerRow
from . import outcome
from .locked import EASTERN, kickoff_utc
from .waivers import LEAD_BACK_MISS_RATE, _eligible, _has_value, _ros, handcuffs, lineup_slots

SKILL = ("QB", "RB", "WR", "TE")


@dataclass
class Ticket:
    player: PlayerRow
    p_boom: float
    p_job: float
    reasons: list[str] = field(default_factory=list)

    @property
    def p_keep(self) -> float:
        return 1.0 - (1.0 - self.p_boom) * (1.0 - self.p_job)


@dataclass
class Window:
    label: str
    kickoff: datetime
    teams: set[str]
    tickets: list[Ticket] = field(default_factory=list)


def _now() -> datetime:
    return datetime.now(UTC)


def window_label(k: datetime) -> str:
    """'Thu 8:15 PM', 'Sun 1:00 PM', 'Sun 4 PM' (the 4:05 and 4:25 slots together)."""
    local = k.astimezone(EASTERN)
    day = local.strftime("%a")
    if local.weekday() == 6 and 16 <= local.hour < 17:
        return f"{day} 4 PM"
    return f"{day} {local.strftime('%-I:%M %p')}"


def kickoff_windows(games, now: datetime | None = None) -> list[Window]:
    """This week's kickoff slots still to come, earliest first."""
    now = now or _now()
    slots: dict[str, Window] = {}
    for g in games.itertuples():
        k = kickoff_utc(g.gameday, g.gametime)
        if k is None or k <= now:
            continue
        label = window_label(k)
        w = slots.setdefault(label, Window(label=label, kickoff=k, teams=set()))
        w.kickoff = min(w.kickoff, k)
        w.teams.add(g.team)
    return sorted(slots.values(), key=lambda w: w.kickoff)


def p_at_least(p: PlayerRow, points: float) -> float:
    """Chance he scores ``points`` or more this week, injury tag included."""
    if p.is_out or p.projection is None:
        return 0.0
    q = p.play_probability if p.play_probability is not None else 1.0
    if q <= 0:
        return 0.0
    mean = float(p.projection) / q if q < 1 else float(p.projection)
    sd = float(p.outcome_sd if p.outcome_sd is not None else (p.projection_sd or 0.0))
    if sd <= 0:
        return q * float(mean >= points)
    probs, zq = outcome.shape(p.position, mean)
    z = (points - mean) / sd
    u = float(np.interp(z, zq, probs, left=0.0, right=1.0))
    return q * (1.0 - u)


def starter_bars(snapshot: LeagueSnapshot, history) -> dict[str, float]:
    """The weekly score that says "his value just jumped", per position.

    A top-N week at his position, N being the number of teams: the kind of
    week that ranks with every team's best player there. Measured as the
    Nth-best score of each week, averaged over the seasons in ``history``.
    (A merely startable week is too low a bar in a deep league: with two
    flex spots in a 14-team league it is about 6 points for a back.)
    """
    rank = max(len(snapshot.teams), 1)
    bars = {}
    for pos in SKILL:
        h = history[history["position"] == pos]
        if h.empty:
            continue
        nth = (h.groupby(["season", "week"])["fantasy_points_ppr"]
               .apply(lambda x, r=rank: x.nlargest(r).min() if len(x) >= r else np.nan))
        bars[pos] = round(float(nth.mean()), 1)
    return bars


def lottery_tickets(snapshot: LeagueSnapshot, cfg: Config, games=None,
                    now: datetime | None = None, history=None) -> list[Window]:
    """The best free-agent tickets for each kickoff window still to come."""
    conf = cfg.raw["roster"].get("lottery_tickets") or {}
    per_window = int(conf.get("per_window", 3))
    min_keep = float(conf.get("min_keep", 0.03))
    if history is None:
        from .projections import load_history

        history = load_history(cfg)
    history = history[history["season"].isin(cfg.train_seasons)]
    bars = starter_bars(snapshot, history)
    # Only positions where a big week would matter to *you*: it has to beat
    # the weakest starter he could replace. A backup quarterback's big game
    # changes nothing behind an established starter.
    roster = [p for p in snapshot.my_team.roster if _has_value(p) and not p.in_ir_slot]
    assigned = lineup_slots(roster, snapshot.starting_slots, _ros)
    by_id = {p.player_id: p for p in roster}
    if games is None:
        from .locked import _week_games

        games = _week_games(snapshot, cfg)
    windows = kickoff_windows(games, now)
    cuffs = handcuffs(snapshot)
    for w in windows:
        tickets = []
        for fa in snapshot.free_agents:
            if fa.position not in SKILL or fa.team not in w.teams or fa.is_out or fa.locked:
                continue
            bar = bars.get(fa.position)
            if bar is None:
                continue
            rivals = [_ros(by_id[i]) for i, slot in assigned.items() if _eligible(fa, slot)]
            if rivals and bar <= min(rivals):
                continue
            pb = p_at_least(fa, bar)
            pj = 0.0
            reasons = [f"{pb:.0%} chance of a top-{len(snapshot.teams)} {fa.position} week ({bar:.0f}+)"]
            if fa.player_id in cuffs:
                lead, gain = cuffs[fa.player_id]
                pj = LEAD_BACK_MISS_RATE
                reasons.append(f"next in line behind {lead.name} (about +{gain:.1f} a game if he goes down)")
            reasons.extend(s for s in fa.signals if s.startswith(("opportunity up", "next man up")))
            if fa.experience == 0:
                reasons.append("rookie")
            tickets.append(Ticket(player=fa, p_boom=pb, p_job=pj, reasons=reasons))
        tickets = [t for t in tickets if t.p_keep >= min_keep]
        tickets.sort(key=lambda t: -t.p_keep)
        w.tickets = tickets[:per_window]
    return [w for w in windows if w.tickets]


def enabled(snapshot: LeagueSnapshot, cfg: Config) -> bool:
    conf = cfg.raw["roster"].get("lottery_tickets") or {}
    return snapshot.platform in (conf.get("platforms") or [])
