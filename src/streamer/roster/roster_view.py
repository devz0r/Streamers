"""Your roster, valued for the rest of the season.

One row per player, from the same simulated seasons the championship hub
prices every move on:

* **season value** -- his projected points a game when he plays (the number
  waivers and trades start from);
* **points a week** -- what he is simulated to score per remaining week once
  injuries, a lost job, byes and the next man up have played out, with the
  10th-90th percentile of that average (his floor and upside);
* **title worth** -- how much lower your P(title) is without him, his spot
  filled from the wire, on the same seasons: the cost of dropping him or
  trading him away, the number to compare across positions, and the order
  waiver drops are tried in (:meth:`TitleEngine.worth`);
* **market** -- what other managers see him worth (the trade model's view),
  so a player the market rates well above us is a sell, and one below us a buy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..league.model import PlayerRow
from .waivers import MIN_KEEP


@dataclass
class RosterValue:
    player: PlayerRow
    ros: float | None              # points a game when he plays
    per_week: float                # simulated points per remaining week
    low: float                     # 10th percentile of that average
    high: float                    # 90th percentile
    title: float                   # P(title) lost without him
    noise: float                   # standard error of ``title``
    market: float | None = None    # what the rest of the league sees
    starting: bool = False         # in this week's recommended lineup


def roster_values(engine, report=None, seen: dict | None = None) -> list[RosterValue]:
    """Every player on your roster, most valuable to your title odds first."""
    m = engine.model
    starting = set()
    opt = getattr(report, "optimisation", None) if report is not None else None
    if opt is not None:
        starting = {p.player_id for _s, p in opt.best_win.flat()}
    worth = engine.worth()
    n = len(engine.base_won)
    out = []
    for p in engine.me.roster:
        j = m.pid.get(p.player_id)
        if j is None:
            continue
        avg = m.futures.scores[:, j, :].mean(axis=1)
        diff = worth[p.player_id]
        s = (seen or {}).get(p.player_id)
        out.append(RosterValue(
            player=p, ros=p.ros_value, per_week=float(avg.mean()),
            low=float(np.quantile(avg, 0.1)), high=float(np.quantile(avg, 0.9)),
            title=float(diff.mean()), noise=float(diff.std() / np.sqrt(n)),
            market=float(s.value) if s is not None else None, starting=p.player_id in starting))
    out.sort(key=lambda r: -r.title)
    return out


def cheapest(values: list[RosterValue], roster: list[PlayerRow], n: int = 3) -> list[RosterValue]:
    """The ``n`` players cheapest to let go, as waiver drops are chosen: by
    title worth, never one in an IR slot, a D/ST or kicker (streamed, and
    priced on the D/ST & K tab), or one that would leave a position short."""
    counts: dict[str, int] = {}
    for p in roster:
        counts[p.position] = counts.get(p.position, 0) + 1
    ok = [r for r in values if not r.player.in_ir_slot and r.player.position not in ("DST", "K")
          and counts.get(r.player.position, 0) - 1 >= MIN_KEEP.get(r.player.position, 0)]
    return sorted(ok, key=lambda r: r.title)[:n]
