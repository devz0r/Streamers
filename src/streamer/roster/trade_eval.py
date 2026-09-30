"""Evaluate any trade, on the page, for both teams.

The trade finder prices a shortlist on the full season simulator; a trade you
are offered, or want to offer, is rarely on it. A page cannot play 6,000
simulated seasons per click, so the evaluator splits the work in two:

1. **Each team's title odds against its weekly points** -- here, on the full
   simulation: every team's scores shifted by -8 to +8 points a week, and
   every team's title and playoff odds read at each shift (``curves``).
2. **What a trade does to a team's weekly points** -- in the browser, from
   each player's week-by-week chance of playing, his expected points when he
   does, and how far that expectation can drift over the season, all read
   off the same simulation. 1,000 seasons are drawn from those with the best
   lineup set each week, before and after the trade.

A trade moves each team's odds by its own change in weekly points, plus the
other team's through the games they play and the places they chase, both
read off the curves. Static season values will not do for the second step:
they ignore byes and injuries and overstated a trade's weekly effect by
about half, and tracked the exact title-odds change at a correlation of only
0.45 (Yahoo, week 4). The simulated weekly change tracks it at 0.8-0.9, with
a typical error of 0.4-0.6 points of title odds against the full simulation
-- whose own noise on a trade is about 0.4. Each run re-measures that on the
trades the finder priced exactly (``check``), and the page shows it.

:func:`evaluate` is the reference for what the page's script
(``trade_eval.js``) does, on the same data and the same random numbers; a
test runs both and requires them to agree.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..league.model import SLOT_ELIGIBILITY, LeagueSnapshot, PlayerRow
from . import perception
from .season import SeasonModel
from .waivers import BENCH_WEIGHTS, MIN_KEEP, _eligible, _ros

SKILL = ("QB", "RB", "WR", "TE")
SCRIPT = Path(__file__).with_name("trade_eval.js")

#: Seasons drawn in the browser per roster. 400 already came within 10% of
#: the full simulation's accuracy; 1,000 keeps a click under a tenth of a second.
NS = 1000
#: Weekly-points shifts at which every team's odds are read.
GRID = (-8.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)

M32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Random numbers the browser can reproduce exactly
# ---------------------------------------------------------------------------
def fnv1a(text: str) -> int:
    """32-bit FNV-1a of a player id: his stream of random numbers."""
    h = 0x811C9DC5
    for b in text.encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & M32
    return h


def _fmix(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> np.uint64(16))
    x = (x * np.uint64(0x85EBCA6B)) & np.uint64(M32)
    x = x ^ (x >> np.uint64(13))
    x = (x * np.uint64(0xC2B2AE35)) & np.uint64(M32)
    return x ^ (x >> np.uint64(16))


def uniforms(seed: int, n: int, weeks: int, offset: int = 0) -> np.ndarray:
    """(n, weeks) uniforms in [0, 1): a hash of (seed, sim, week), so any
    one of them can be computed on its own, in any language."""
    s = (np.arange(n, dtype=np.uint64)[:, None] + np.uint64(1)) * np.uint64(0x9E3779B1) & np.uint64(M32)
    k = (np.arange(weeks, dtype=np.uint64)[None, :] + np.uint64(1 + offset)) * np.uint64(0x85EBCA77) \
        & np.uint64(M32)
    return _fmix(np.uint64(seed) ^ s ^ k).astype(np.float64) / 4294967296.0


def normals(seed: int, n: int) -> np.ndarray:
    """(n,) standard normals by Box-Muller, one per simulated season."""
    alt = seed ^ 0x5BD1E995
    u = uniforms(alt, n, 2, offset=1000)
    return np.sqrt(-2.0 * np.log(np.maximum(u[:, 0], 1e-12))) * np.cos(2.0 * math.pi * u[:, 1])


# ---------------------------------------------------------------------------
# What the page is given
# ---------------------------------------------------------------------------
def _weekly(model: SeasonModel, p: PlayerRow) -> dict | None:
    """His chance of playing each week, his expected points when he does,
    and how far that expectation spreads across the simulated seasons."""
    i = model.pid.get(p.player_id)
    if i is None:
        return None
    lv = model.futures.levels[:, i, :]                                    # (n, weeks)
    on = lv > 0
    count = on.sum(axis=0)
    avail = count / lv.shape[0]
    mean = np.where(count > 0, (lv * on).sum(axis=0) / np.maximum(count, 1), 0.0)
    sq = np.where(count > 0, ((lv - mean[None, :]) ** 2 * on).sum(axis=0) / np.maximum(count, 1), 0.0)
    return {"a": [round(float(x), 3) for x in avail], "m": [round(float(x), 2) for x in mean],
            "s": [round(float(x), 2) for x in np.sqrt(sq)]}


def export(snapshot: LeagueSnapshot, model: SeasonModel, finder, priced=()) -> dict:
    """Everything the page's evaluator needs for one league.

    ``finder`` is the league's :class:`~streamer.roster.trades.TradeFinder`
    (for the market's and each manager's view of players); ``priced`` its
    trades priced on the full simulation, to show exactly where a trade
    matches one and to measure the estimate against."""
    from . import trades as tr

    slot_names = [s for s, _c in model.slots]
    players, seen = {}, finder.seen
    for t in snapshot.teams:
        for p in t.roster:
            s = seen.get(p.player_id)
            row = {
                "n": p.name, "pos": p.position, "tm": p.team or "", "slot": p.slot,
                "el": [slot for slot in slot_names if _eligible(p, slot)],
                "ir": p.in_ir_slot, "out": p.is_out, "q": p.is_questionable, "un": p.unsigned,
                "st": p.status or "", "o": round(_ros(p), 2), "sd": round(float(p.ros_sd or 0.0), 2),
                "v": round(finder.theirs(p), 2), "sig": p.signals[0] if p.signals else "",
                "pl": bool(p.locked or p.actual_points is not None),
            }
            if s is not None:
                row.update({"lp": s.last_ppg, "lg": s.last_games, "np": s.now_ppg, "ng": s.now_games,
                            "pf": round(s.platform, 2)})
            if p.position in SKILL:
                wk = _weekly(model, p)
                if wk:
                    row.update(wk)
            players[p.player_id] = row

    teams = []
    for t in snapshot.teams:
        own, over = finder.own.get(t.team_id, ({}, {}))
        teams.append({"id": t.team_id, "n": t.name, "me": bool(t.is_mine), "r": [p.player_id for p in t.roster],
                      "eng": round(finder.engaged.get(t.team_id, 0.8), 3), "acq": t.acquisitions,
                      "own": {k: round(v, 2) for k, v in own.items()}, "over": over})

    base = model.odds()
    title, playoffs = [], []
    for j, t in enumerate(model.teams):
        tj, pj = [], []
        for d in GRID:
            o = model.odds(override={t.team_id: model.base_scores[j] + d}) if d else base
            tj.append([round(float(x), 4) for x in o.p_title])
            pj.append([round(float(x), 4) for x in o.p_playoffs])
        title.append(tj)
        playoffs.append(pj)

    skill_slots = [(s, c) for s, c in model.slots if set(SLOT_ELIGIBILITY.get(s, (s,))) & set(SKILL)]
    data = {
        "ns": NS, "weeks": list(model.weeks), "reg": model.reg_weeks, "lag": model.lag,
        "slots": [[s, c] for s, c in model.slots],
        "sim_slots": [[s, c, round(model.slot_replacement(s), 2)] for s, c in skill_slots],
        "seen_repl": {k: round(v, 2) for k, v in finder.seen_replacement.items()},
        "bench_w": {k: list(v) for k, v in BENCH_WEIGHTS.items()},
        "min_keep": dict(MIN_KEEP),
        "params": {"center": perception.ACCEPT_CENTER, "scale": perception.ACCEPT_SCALE,
                   "cons": perception.CONSOLIDATION, "gap": tr.CONSOLIDATION_GAP,
                   "ceiling": perception.ACCEPT_CEILING, "gap_note": tr.GAP_NOTE},
        "grid": list(GRID),
        "title": title, "playoffs": playoffs,
        "base": {"title": [round(float(x), 4) for x in base.p_title],
                 "playoffs": [round(float(x), 4) for x in base.p_playoffs]},
        "teams": teams, "players": players,
        "exact": [[t.partner.team_id, sorted(p.player_id for p in t.give), sorted(p.player_id for p in t.get),
                   round(t.gain, 4), round(t.their_gain, 4)] for t in priced],
    }
    data["check"] = check(data, priced)
    return data


# ---------------------------------------------------------------------------
# The evaluation (mirrored line for line by trade_eval.js)
# ---------------------------------------------------------------------------
def _elig(data: dict, pid: str, slot: str) -> bool:
    return slot in data["players"][pid]["el"]


def lineup(data: dict, ids: list[str], key) -> dict[str, str]:
    """Greedy best lineup on ``key`` (as :func:`waivers.lineup_slots`)."""
    ranked = sorted(ids, key=key, reverse=True)
    assigned: dict[str, str] = {}
    for slot, count in data["slots"]:
        taken = 0
        for pid in ranked:
            if taken >= count:
                break
            if pid in assigned or not _elig(data, pid, slot):
                continue
            assigned[pid] = slot
            taken += 1
    return assigned


def roster_value(data: dict, ids: list[str], key) -> float:
    """Best lineup plus bench depth over the market's replacement level (as
    :func:`waivers.roster_value` without upside)."""
    assigned = lineup(data, ids, key)
    total = sum(key(pid) for pid in ids if pid in assigned)
    bench: dict[str, list[float]] = {}
    for pid in ids:
        if pid not in assigned:
            bench.setdefault(data["players"][pid]["pos"], []).append(key(pid))
    for pos, vals in bench.items():
        vals.sort(reverse=True)
        level = data["seen_repl"].get(pos, 0.0)
        for w, v in zip(data["bench_w"].get(pos, []), vals):
            total += w * max(v - level, 0.0)
    return total


def _counts(data: dict, ids: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pid in ids:
        p = data["players"][pid]
        if not p["ir"]:
            counts[p["pos"]] = counts.get(p["pos"], 0) + 1
    return counts


def keeps_minimums(data: dict, ids: list[str], before: list[str]) -> bool:
    """No position below its minimum, or below what the roster had."""
    now, was = _counts(data, ids), _counts(data, before)
    return all(now.get(pos, 0) >= min(n, was.get(pos, 0)) for pos, n in data["min_keep"].items())


def make_room(data: dict, ids: list[str], extra: int, keep: set[str], key) -> tuple[list[str], list[str]]:
    """Drop ``extra`` of the least valuable players on ``key``, never one just
    received, never one that leaves a position empty."""
    dropped, before = [], ids
    for _ in range(extra):
        options = [pid for pid in ids if pid not in keep and not data["players"][pid]["ir"]
                   and keeps_minimums(data, [q for q in ids if q != pid], before)]
        if not options:
            break
        worst = min(options, key=key)
        ids = [pid for pid in ids if pid != worst]
        dropped.append(worst)
    return ids, dropped


def consolidation(data: dict, give: list[str], get: list[str], his) -> float:
    if len(give) == len(get) or not give or not get:
        return 0.0
    he_gets_fewer = len(give) < len(get)
    fewer, more = (give, get) if he_gets_fewer else (get, give)
    lead = max(map(his, fewer)) - max(map(his, more))
    if lead <= 0:
        return 0.0
    clear = min(lead / data["params"]["gap"], 1.0)
    return clear if he_gets_fewer else -clear


def p_accept(data: dict, gain: float, cons: float, engaged: float) -> float:
    q = data["params"]
    x = gain + cons * q["cons"]
    return q["ceiling"] * engaged / (1.0 + math.exp(-(x - q["center"]) / q["scale"]))


class Sim:
    """Mean weekly points of a roster over the drawn seasons."""

    def __init__(self, data: dict):
        self.data = data
        self.n, self.k = int(data["ns"]), len(data["weeks"])
        self._draws: dict[str, np.ndarray] = {}

    def levels(self, pid: str) -> np.ndarray:
        """(n, k) his expected points each week of each season, 0 when he
        does not play."""
        if pid not in self._draws:
            p = self.data["players"][pid]
            if "a" not in p:
                self._draws[pid] = np.zeros((self.n, self.k))
            else:
                seed = fnv1a(pid)
                u = uniforms(seed, self.n, self.k)
                z = normals(seed, self.n)[:, None]
                lv = np.maximum(np.asarray(p["m"])[None, :] + np.asarray(p["s"])[None, :] * z, 0.01)
                self._draws[pid] = np.where(u < np.asarray(p["a"])[None, :], lv, 0.0)
        return self._draws[pid]

    def mean(self, ids: list[str], windows: dict[str, tuple[int, int]] | None = None) -> float:
        """``windows``: player -> (first week index, week index he leaves) on
        this roster; outside it he is not there."""
        ids = [pid for pid in ids if self.data["players"][pid]["pos"] in SKILL]
        if not ids:
            return 0.0
        lv = np.stack([self.levels(pid) for pid in ids], axis=2).copy()          # (n, k, p)
        kk = np.arange(self.k)
        for j, pid in enumerate(ids):
            if windows and pid in windows:
                a, b = windows[pid]
                lv[:, (kk < a) | (kk >= b), j] = -1.0
        used = np.zeros(lv.shape, bool)
        total = np.zeros((self.n, self.k))
        slot_ix = np.arange(len(ids))[None, None, :]
        for slot, count, rep in self.data["sim_slots"]:
            elig = np.array([_elig(self.data, pid, slot) for pid in ids])
            for _ in range(count):
                masked = np.where(elig[None, None, :] & ~used, lv, -2.0)
                pick = masked.argmax(axis=2)
                best = np.take_along_axis(masked, pick[:, :, None], axis=2)[:, :, 0]
                ok = (best >= max(rep, 0.0) + 1e-6) & (best > -0.5)
                total += np.where(ok, best, rep)
                used |= (slot_ix == pick[:, :, None]) & ok[:, :, None]
        return float(total.mean())


def _windows(data: dict, original: list[str], new: list[str]) -> tuple[list[str], dict]:
    """A move made mid-week cannot reach back into games already played: a
    newcomer whose game has kicked off joins next week, and a departing
    player whose game has kicked off still counts this week (as
    :meth:`SeasonModel._this_week_is_settled`)."""
    lag, k = int(data["lag"]), len(data["weeks"])
    ids, win = list(new), {}
    for pid in new:
        join = lag + (1 if data["players"][pid]["pl"] else 0)
        if pid not in original and join > 0:
            win[pid] = (join, k)
    for pid in original:
        leave = lag + (1 if data["players"][pid]["pl"] else 0)
        if pid not in new and leave > 0:
            ids.append(pid)
            win[pid] = (0, leave)
    return ids, win


def _read(data: dict, curve: str, team: int, shift: float, who: int) -> float:
    grid = data["grid"]
    col = [row[who] for row in data[curve][team]]
    return float(np.interp(shift, grid, col))


def evaluate(data: dict, partner: str, give: list[str], get: list[str],
             my_drops: list[str] | None = None, sim: Sim | None = None) -> dict:
    """Both teams' view of a trade: title and playoff odds, weekly points,
    lineups, forced drops, and his chance of saying yes."""
    sim = sim or Sim(data)
    P = data["players"]
    ti = {t["id"]: i for i, t in enumerate(data["teams"])}
    me_i = next(i for i, t in enumerate(data["teams"]) if t["me"])
    th_i = ti[partner]
    me, th = data["teams"][me_i], data["teams"][th_i]
    ours = lambda pid: P[pid]["o"]                                           # noqa: E731
    own = th["own"]
    his = lambda pid: own.get(pid, P[pid]["v"])                               # noqa: E731

    give_s, get_s = set(give), set(get)
    my_new = [pid for pid in me["r"] if pid not in give_s] + list(get)
    their_new = [pid for pid in th["r"] if pid not in get_s] + list(give)
    my_drops_out: list[str] = []
    their_drops: list[str] = []
    extra = len(get) - len(give)
    if extra > 0:
        chosen = [pid for pid in (my_drops or []) if pid in my_new and pid not in get_s][:extra]
        my_new = [pid for pid in my_new if pid not in chosen]
        my_new, auto = make_room(data, my_new, extra - len(chosen), get_s, lambda pid: P[pid]["o"] + P[pid]["sd"])
        my_drops_out = chosen + auto
    elif extra < 0:
        their_new, their_drops = make_room(data, their_new, -extra, give_s, his)

    # Weekly points and the odds they buy, for both teams.
    ids, win = _windows(data, me["r"], my_new)
    d_me = sim.mean(ids, win) - sim.mean(me["r"])
    ids, win = _windows(data, th["r"], their_new)
    d_th = sim.mean(ids, win) - sim.mean(th["r"])

    def odds(curve: str, who: int) -> tuple[float, float]:
        b = data["base"][curve][who]
        new = b + (_read(data, curve, me_i, d_me, who) - b) + (_read(data, curve, th_i, d_th, who) - b)
        return b, min(max(new, 0.0), 1.0)

    def lineup_total(ids_, key) -> float:
        a = lineup(data, ids_, key)
        return sum(key(pid) for pid in ids_ if pid in a)

    my_lineup = lineup_total(my_new, ours) - lineup_total(me["r"], ours)
    seen = roster_value(data, their_new, his) - roster_value(data, th["r"], his)
    cons = consolidation(data, give, get, his)
    yes = p_accept(data, seen, cons, th["eng"])
    out = {
        "shift_me": d_me, "shift_them": d_th, "lineup_me": my_lineup, "seen": seen,
        "consolidation": cons, "p_accept": yes, "my_drops": my_drops_out, "their_drops": their_drops,
        "title_me": odds("title", me_i), "title_them": odds("title", th_i),
        "playoffs_me": odds("playoffs", me_i), "playoffs_them": odds("playoffs", th_i),
        "legal": keeps_minimums(data, my_new, me["r"]) and keeps_minimums(data, their_new, th["r"]),
    }
    out["why_you"], out["why_them"] = _reasons(data, out, th, give, get, my_new, their_new, his)
    key = [partner, sorted(give), sorted(get)]
    out["exact"] = next((e[3:] for e in data["exact"] if e[:3] == key and not my_drops), None)
    return out


def _reasons(data, out, th, give, get, my_new, their_new, his) -> tuple[list[str], list[str]]:
    P, gap = data["players"], data["params"]["gap_note"]
    you = [f"your lineup {out['lineup_me']:+.1f} a game on our projections, "
           f"{out['shift_me']:+.1f} a week once byes and injuries play out"]
    for pid in give:
        p = P[pid]
        if p["v"] - p["o"] >= gap:
            you.append(f"sell high: the market sees {p['n']} at {p['v']:.1f} a game, we project {p['o']:.1f}")
    for pid in get:
        p = P[pid]
        if p["o"] - p["v"] >= gap:
            you.append(f"buy low: we project {p['n']} at {p['o']:.1f}, the market sees {p['v']:.1f}")
        if p["sig"]:
            you.append(f"{p['n']}: {p['sig']}")
    if out["my_drops"]:
        you.append("you would drop " + " and ".join(P[pid]["n"] for pid in out["my_drops"]) + " to make room")
    elif len(get) < len(give):
        you.append("opens a roster spot for a pickup")

    them = [f"as he would see it, his roster {out['seen']:+.1f} a game"]
    for pid in get:
        if pid in th["over"] and his(pid) >= P[pid]["v"] + 0.3:
            them.append(f"he starts {P[pid]['n']} over {th['over'][pid]}, so he rates him above the "
                        f"market's {P[pid]['v']:.1f} a game")
    start_old = set(lineup(data, th["r"], his))
    start_new = set(lineup(data, their_new, his))
    benched = [pid for pid in th["r"] if pid in start_old and pid in their_new and pid not in start_new]
    for pid in give:
        if pid in start_new and benched:
            worst = min(benched, key=his)
            them.append(f"{P[pid]['n']} would start for him over {P[worst]['n']}")
            break
    if out["consolidation"] >= 0.5:
        them.append("he gets the best player in the deal")
    elif out["consolidation"] <= -0.5:
        them.append("he gives up the best player in the deal, a harder sell")
    for pid in give:
        p = P[pid]
        if p.get("lp") is not None and p.get("lg", 0) >= 6 and p["lp"] >= p["o"] + 2:
            them.append(f"name value: {p['n']} scored {p['lp']:.1f} a game last season")
        if p.get("np") is not None and p.get("ng", 0) >= 2 and p["np"] >= p.get("pf", p["o"]) + 3:
            them.append(f"{p['n']} is averaging {p['np']:.1f} this season")
    for pid in get:
        p = P[pid]
        if p["out"] or p["st"]:
            them.append(f"{p['n']} is listed {p['st'] or 'out'}")
        elif p.get("np") is not None and p.get("ng", 0) >= 2 and p["np"] <= p.get("pf", p["o"]) - 3:
            them.append(f"{p['n']} has averaged only {p['np']:.1f} so far")
    if th["acq"] is not None and th["eng"] < 0.75:
        them.append(f"he rarely makes moves ({th['acq']} pickups), so any offer is a long shot")
    elif th["acq"] is not None and th["eng"] >= 0.95:
        them.append(f"an active manager ({th['acq']} pickups)")
    b, a = out["title_them"]
    if a >= b:
        them.append(f"his title odds rise too ({b:.1%} to {a:.1%})")
    if out["their_drops"]:
        them.append("he would drop " + " and ".join(P[pid]["n"] for pid in out["their_drops"]) + " to make room")
    return you, them


def check(data: dict, priced) -> dict:
    """How far the page's estimate is from the full simulation, on the trades
    priced exactly this run: typical error (root mean square) in points of
    title odds, for you and for him."""
    priced = list(priced)
    if len(priced) < 5:
        return {"n": len(priced)}
    sim = Sim(data)
    mine, theirs = [], []
    for t in priced:
        r = evaluate(data, t.partner.team_id, [p.player_id for p in t.give], [p.player_id for p in t.get], sim=sim)
        mine.append((r["title_me"][1] - r["title_me"][0], t.gain))
        theirs.append((r["title_them"][1] - r["title_them"][0], t.their_gain))
    m, h = np.array(mine), np.array(theirs)

    def rmse(a):
        return round(float(np.sqrt(((a[:, 0] - a[:, 1]) ** 2).mean())) * 100, 2)

    def corr(a):
        return round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 2) if a[:, 1].std() > 0 else None

    return {"n": len(priced), "me_rmse": rmse(m), "them_rmse": rmse(h), "me_corr": corr(m), "them_corr": corr(h)}
