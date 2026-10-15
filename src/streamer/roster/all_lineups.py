"""Any team's best lineup this week, on the Lineup tab.

For every team in the league, the lineup that maximises its P(win) against
this week's opponent (:func:`lineup.optimise`), two ways:

* **current**: against the lineup the opponent has set (their best projected
  one if none is set yet);
* **ideal**: against the opponent's best projected lineup.

Your own team uses the full simulation; the others use fewer draws to keep
the refresh short (their win odds are good to about a point). A page cannot
optimise lineups per click, so every team is priced during the refresh.
"""

from __future__ import annotations

import logging

from ..league.model import LeagueSnapshot, PlayerRow
from . import lineup as lu

log = logging.getLogger(__name__)
OTHER_SIMS = 4000


def pairings(snapshot: LeagueSnapshot) -> dict[str, str]:
    """team id -> this week's opponent id, from the schedule (and the matchup)."""
    out: dict[str, str] = {}
    for w, a, b in (snapshot.rules or {}).get("schedule", []):
        if int(w) == int(snapshot.week):
            out[str(a)], out[str(b)] = str(b), str(a)
    m = snapshot.matchup
    if m is not None and m.opponent_team_id:
        out[str(m.my_team_id)] = str(m.opponent_team_id)
        out[str(m.opponent_team_id)] = str(m.my_team_id)
    return out


def _player(p: PlayerRow) -> dict:
    return {"name": p.name, "pos": p.position, "team": p.team or "", "status": p.status or "",
            "proj": round(float(p.week_value), 1), "locked": bool(p.locked)}


def _view(opt: lu.Optimisation, slots: dict[str, int]) -> dict:
    best = opt.best_win
    order = sorted(slots, key=lambda s: (len(lu.SLOT_ELIGIBILITY.get(s, (s,))), s))
    rows = [{"slot": s, **_player(p)} for s in order for p in best.starters.get(s, [])]
    changes = [{"slot": s, "out": b.name if b is not None else None, "in": p.name}
               for s, b, p in opt.changes]
    return {"win": round(best.win_probability * 100, 1), "proj": round(best.expected, 1),
            "current_win": round(opt.current.win_probability * 100, 1) if opt.current else None,
            "opp_proj": round(opt.opponent.expected, 1) if opt.opponent else None,
            "lineup": rows, "changes": changes}


def build(snapshot: LeagueSnapshot, my_opt: lu.Optimisation | None, sims: int = 20000,
          other_sims: int = OTHER_SIMS) -> dict | None:
    """``{"mine": id, "teams": [{id, name, opp, views: {current, ideal}}]}``."""
    opp_of = pairings(snapshot)
    slots = snapshot.starting_slots
    me = snapshot.my_team.team_id if snapshot.my_team else None
    teams = []
    for t in snapshot.teams:
        opp = snapshot.team(opp_of.get(t.team_id, "")) if opp_of.get(t.team_id) else None
        if opp is None:
            continue
        views = {}
        for mode in ("current", "ideal"):
            try:
                if t.team_id == me and mode == "current" and my_opt is not None:
                    opt = my_opt
                else:
                    opt = lu.optimise(t.roster, slots, opp.roster, opponent_mode=mode,
                                      n_sims=sims if t.team_id == me else other_sims)
                views[mode] = _view(opt, slots)
            except Exception as exc:  # noqa: BLE001 - one team never blocks the rest
                log.warning("best lineup for %s (%s) skipped: %s", t.name, mode, exc)
        if views:
            teams.append({"id": t.team_id, "name": t.name, "opp": opp.name,
                          "record": f"{t.wins}-{t.losses}", "views": views})
    if not teams:
        return None
    teams.sort(key=lambda x: (x["id"] != me, x["name"].lower()))
    return {"mine": me, "teams": teams}
