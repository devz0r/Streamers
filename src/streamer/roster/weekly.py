"""Each player's expected points week by week, rest of the season, for the
player card.

Read off the season simulation (:mod:`streamer.roster.season`) that prices
title odds, trades and claims, so a week's number is what that model expects
of him *that week*: his chance of playing (injuries, a bye, the man ahead of
him), his opponent where the model reads one (quarterbacks, backs, defences
and kickers; receivers and tight ends held up no better against their
opponent in tests, so theirs is matchup-neutral), the fade of what is known
today, and the next man up when the lead is out. This week's row is the full
weekly projection; later weeks are more uncertain, and the range says so.

The rows add up to his season value: the future weeks are scaled together so
they total the season value times the games he is expected to play, the
number every other part of the page uses. The scale is within 7% for nine
players in ten, and is held to 0.7-1.4 where a stale value would make it
silly.
"""

from __future__ import annotations

import logging

import numpy as np

from ..teams import normalize_team

log = logging.getLogger(__name__)

#: The most the future weeks are scaled to meet the season value.
SCALE = (0.7, 1.4)


def _opponents(model, cfg) -> dict[str, dict[int, str]]:
    try:
        from .season import nfl_opponents

        return nfl_opponents(model.snapshot, list(model.weeks), cfg)
    except Exception as exc:  # noqa: BLE001 - a label; the rows stand without it
        log.warning("weekly opponents unavailable: %s", exc)
        return {}


def build(model, snapshot, cfg) -> dict[str, dict]:
    """Player id -> ``{"w": [[week, opponent | "bye", expected points, chance he
    plays, 10th, 90th percentile if he plays], ...], "tot": rest-of-season
    expected points, "gm": expected games played, "now": index of this
    week's row}``; a bye is ``[week, "bye"]``."""
    f = model.futures
    weeks = list(f.weeks)
    lag = int(model.lag)
    opps = _opponents(model, cfg)
    byid = {p.player_id: p for p in model.players}
    out: dict[str, dict] = {}
    for j, pid in enumerate(f.player_ids):
        p = byid.get(pid)
        if p is None:
            continue
        sc = f.scores[:, j, :]
        played = f.played[:, j, :] if f.played is not None else sc > 0
        mean, pl = sc.mean(axis=0), played.mean(axis=0)
        future = np.arange(len(weeks)) > lag
        scale = 1.0
        ros = p.ros_value
        if ros and mean[future].sum() > 0 and pl[future].sum() > 0:
            scale = float(np.clip(ros * pl[future].sum() / mean[future].sum(), *SCALE))
        team_opps = opps.get(normalize_team(p.team), {}) if p.team else {}
        rows = []
        for k, wk in enumerate(weeks):
            if team_opps and wk not in team_opps:
                rows.append([int(wk), "bye"])
                continue
            s = scale if k > lag else 1.0
            on = played[:, k]
            lo, hi = ((np.percentile(sc[on, k], [10, 90]) * s) if on.sum() >= 20 else (0.0, 0.0))
            rows.append([int(wk), team_opps.get(wk, ""), round(float(mean[k] * s), 1), round(float(pl[k]), 2),
                         round(float(lo), 1), round(float(hi), 1)])
        tot = sum(r[2] for r in rows if len(r) > 2)
        if tot <= 0 and not any(len(r) > 2 and r[3] > 0 for r in rows):
            continue
        out[pid] = {"w": rows, "tot": round(float(tot), 1), "gm": round(float(pl.sum()), 1), "now": lag}
    return out
