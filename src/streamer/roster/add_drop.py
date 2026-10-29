"""Pick a free agent, see what each drop does: the "Try a pickup" table.

For the top free agents (by season value: 120 at the skill positions, six
defences and six kickers), the sensible drops from your roster are priced
two ways on the simulations the rest of the page uses -- six drops each for
the top 40 skill players and every unit, the three cheapest for the rest:

* **Title odds**: the add and the drop on the full season simulation
  (:meth:`TitleEngine.value_now`), against your roster as it stands.
* **This week's win odds**: your best lineup rebuilt with him in and the
  drop gone, against your opponent's lineup, on draws shared by every pair
  and by your lineup as it stands (as :func:`lineup.pickup_options` does).

A page cannot run either simulation per click, so every pair is priced
ahead of time and the page only looks them up; the page searches them by
name. Every other free agent is listed too, unpriced, so a search for him
says why he is not there. Drops are the cheapest to
your title first (and always the cheapest at his position), never a player
whose game has started or one in an IR slot.
"""

from __future__ import annotations

import numpy as np

from ..league.model import LeagueSnapshot, PlayerRow, breaks_limit, position_limits
from . import lineup as lu
from .title_moves import drop_candidates
from .value_trend import payload

#: Free agents priced, skill positions; and defences and kickers on top.
N_SKILL = 120
N_UNITS = 6
#: Drops priced for each add: the top ``N_FULL`` skill players and every
#: unit get ``N_DROPS``, the long tail ``N_DROPS_TAIL`` (about 0.1 s each).
N_FULL = 40
N_DROPS = 6
N_DROPS_TAIL = 3
SIMS = 8000


def candidates(snapshot: LeagueSnapshot, n_skill: int | None = None, n_units: int | None = None) -> list[PlayerRow]:
    """The free agents worth pricing, best season value first."""
    def ok(p: PlayerRow) -> bool:
        return (bool(p.team) and not p.unsigned and not p.is_long_term_out
                and (p.ros_value is not None or p.projection is not None))

    def key(p: PlayerRow) -> float:
        return -float(p.ros_value if p.ros_value is not None else (p.projection or 0.0))

    pool = [p for p in snapshot.free_agents if ok(p)]
    skill = sorted((p for p in pool if p.position in ("QB", "RB", "WR", "TE")), key=key)[:N_SKILL if n_skill is None else n_skill]
    units = []
    for pos in ("DST", "K"):
        units += sorted((p for p in pool if p.position == pos), key=key)[:N_UNITS if n_units is None else n_units]
    return skill + units


def build(engine, report, n_drops: int = N_DROPS, n_sims: int = SIMS, seed: int = 31,
          n_full: int = N_FULL, n_tail: int = N_DROPS_TAIL, n_skill: int | None = None,
          n_units: int | None = None) -> dict | None:
    """``{"base": {...}, "adds": [{player, drops: [{player, title, win}]}],
    "others": [player, ...]}``; title and win are changes in points of
    probability (None where the week cannot be scored: no opponent);
    ``others`` are the free agents not priced, for the search."""
    snap = report.snapshot
    me = snap.my_team
    adds = [p for p in candidates(snap, n_skill, n_units) if p.player_id in engine.model.pid]
    # Everyone the waiver moves recommend is priced here too, whatever his
    # season value: the table must not call a recommended pickup "not priced".
    seen_ids = {p.player_id for p in adds}
    for m in getattr(report, "title_moves", None) or []:
        if m.add.player_id not in seen_ids and m.add.player_id in engine.model.pid:
            adds.append(m.add)
            seen_ids.add(m.add.player_id)
    limits = position_limits(snap)
    if not adds:
        return None
    worth = engine.worth_mean()
    opt = report.optimisation
    roster = [p for p in me.roster if not p.in_ir_slot]

    # This week's draws, shared by every pair.
    week = None
    if opt is not None and opt.opponent is not None:
        theirs = [p for _s, p in opt.opponent.flat()]
        everyone, seen = [], set()
        for p in roster + theirs + adds:
            if p.player_id not in seen:
                seen.add(p.player_id)
                everyone.append(p)
        index = {p.player_id: i for i, p in enumerate(everyone)}
        rng = np.random.default_rng(seed)
        select = lu.sample_points(everyone, n_sims, rng)
        confirm = lu.sample_points(everyone, n_sims, rng)
        opp_sel = lu._totals([opt.opponent.starters], select, index)[0]
        opp_conf = lu._totals([opt.opponent.starters], confirm, index)[0]
        base_tot = lu._totals([opt.best_win.starters], confirm, index)[0]
        base_ind = (base_tot > opp_conf) + 0.5 * (base_tot == opp_conf)
        week = (index, select, confirm, opp_sel, opp_conf, base_ind)

    def win_change(add: PlayerRow, drop: PlayerRow | None) -> float | None:
        if week is None:
            return None
        index, select, confirm, opp_sel, opp_conf, base_ind = week
        team = [p for p in roster if drop is None or p.player_id != drop.player_id] + [add]
        lineups = lu._with_locks(team, snap.starting_slots)
        if not lineups:
            return None
        best = lineups[int(np.argmax(lu._win_rates(lineups, select, index, opp_sel)))]
        tot = lu._totals([best], confirm, index)[0]
        ind = (tot > opp_conf) + 0.5 * (tot == opp_conf)
        return float(ind.mean() - base_ind.mean())

    def row(p: PlayerRow) -> dict:
        return {"id": p.player_id, "name": p.name, "pos": p.position, "team": p.team or "",
                "ros": round(float(p.ros_value or 0.0), 1), "proj": round(float(p.projection or 0.0), 1),
                "status": p.status or "", "tr": payload(p), "chg": payload(p)[0] if payload(p) else None}

    # An open roster spot: he can be added with nobody dropped.
    spots = sum(snap.starting_slots.values()) + int(snap.bench_size or 0)
    open_spot = len(roster) < spots
    out = []
    skill_rank = {p.player_id: i for i, p in enumerate(p for p in adds if p.position not in ("DST", "K"))}
    for a in adds:
        k = n_drops if skill_rank.get(a.player_id, 0) < n_full else min(n_tail, n_drops)
        drops = [d for d in drop_candidates(roster, a, k + 2, worth, limits) if not d.locked][:k]
        priced = []
        for d in drops:
            won = engine.value_now(a, d, per_sim=True)
            title = float(won.mean()) - engine.base
            noise = float(np.std(won - engine.base_won) / np.sqrt(len(won)))
            priced.append({**row(d), "title": round(title * 100, 2), "noise": round(2 * noise * 100, 2),
                           "win": None if (w := win_change(a, d)) is None else round(w * 100, 2)})
        if open_spot and not breaks_limit(me.roster, list(me.roster) + [a], limits):
            won = engine._won([p.player_id for p in me.roster] + [a.player_id])
            noise = float(np.std(won - engine.base_won) / np.sqrt(len(won)))
            w = win_change(a, None)
            priced.append({"id": "", "name": "No drop (open roster spot)", "pos": "", "team": "", "ros": 0.0,
                           "proj": 0.0, "status": "", "none": True,
                           "title": round((float(won.mean()) - engine.base) * 100, 2),
                           "noise": round(2 * noise * 100, 2), "win": None if w is None else round(w * 100, 2)})
        if priced:
            priced.sort(key=lambda r: (not r.get("none"), -r["title"]))
            out.append({**row(a), "drops": priced})
    out.sort(key=lambda r: -max(d["title"] for d in r["drops"]))
    base_win = None if week is None else float(week[5].mean())
    priced = {a["id"] for a in out}
    others = [{"name": p.name, "pos": p.position, "team": p.team or "", "status": p.status or "",
               "ros": round(float(p.ros_value or 0.0), 1), "proj": round(float(p.projection or 0.0), 1),
               "why": _why_not(p), "tr": payload(p), "chg": payload(p)[0] if payload(p) else None}
              for p in snap.free_agents if p.player_id not in priced and p.position in POSITIONS]
    others.sort(key=lambda r: -r["ros"])
    return {"base": {"title": round(engine.base * 100, 1),
                     "win": None if base_win is None else round(base_win * 100, 1)},
            "adds": out, "others": others, "open_spot": open_spot}


POSITIONS = ("QB", "RB", "WR", "TE", "DST", "K")


def _why_not(p: PlayerRow) -> str:
    if not p.team or p.unsigned:
        return "not on an NFL roster"
    if p.is_long_term_out:
        return "out long term"
    return "below the free agents priced at his position"
