"""This week's simulated score distributions, for the charts the page draws.

P(win) is the share of simulated games in which your lineup outscores your
opponent's (:mod:`streamer.roster.lineup`). Here those same draws are counted
into plain histograms -- your total, your opponent's, and the margin between
-- and, for each player, his own points. Nothing is smoothed: a bar is the
share of draws that landed in it. A side with nothing left to play has one
score, not a distribution, and is shown as that score. Once games are on, the
page redraws the charts from the live draws (``static/live.js``).
"""

from __future__ import annotations

import numpy as np

#: Spread (points) below which a total is a fixed score: every draw the same.
FIXED = 0.05


def density(x: np.ndarray, step: float, lo: float | None = None, hi: float | None = None) -> dict:
    """A histogram of ``x`` on a regular grid (share of draws per point, so the
    bars' area is 1), plus its 10th/50th/90th percentiles; or ``{"fixed": v}``
    when every draw is the same score."""
    x = np.asarray(x, dtype=float)
    if float(x.max() - x.min()) < FIXED:
        v = round(float(x[0]), 2)
        return {"fixed": v, "p10": v, "p50": v, "p90": v, "mean": v}
    lo = float(np.floor(np.percentile(x, 0.2) / step) * step) if lo is None else lo
    hi = float(np.ceil(np.percentile(x, 99.8) / step) * step) if hi is None else hi
    if hi - lo < step * 3:
        hi = lo + step * 3
    edges = np.arange(lo, hi + step / 2, step)
    h, _ = np.histogram(np.clip(x, lo, hi - 1e-9), bins=edges)
    y = h / (len(x) * step)
    return {"x0": round(lo, 3), "dx": step, "y": [round(float(v), 5) for v in y],
            "p10": round(float(np.percentile(x, 10)), 1), "p50": round(float(np.percentile(x, 50)), 1),
            "p90": round(float(np.percentile(x, 90)), 1), "mean": round(float(np.mean(x)), 1)}


def matchup(opt, my_name: str, opp_name: str) -> dict | None:
    """Your total, your opponent's and the margin: every draw behind the
    recommended lineup's P(win) (late swaps counted), so the chart and the
    percentage are the same numbers."""
    if opt is None or opt.opponent is None:
        return None
    me, them = getattr(opt, "win_totals", None), getattr(opt, "opp_totals", None)
    if me is None or them is None:
        return None
    me, them = np.asarray(me, dtype=float), np.asarray(them, dtype=float)
    margin = me - them
    raw = float((margin > 0).mean() + 0.5 * (margin == 0).mean())
    names = [my_name or "You", opp_name or "Opponent"]
    if float(me.max() - me.min()) < FIXED and float(them.max() - them.min()) < FIXED:
        # Nobody left to play on either side: a result, not a distribution.
        return {"final": [round(float(me[0]), 2), round(float(them[0]), 2)], "p": round(raw, 4), "raw": round(raw, 4),
                "names": names}
    lo = float(np.floor(np.percentile(np.concatenate([me, them]), 0.2) / 2) * 2)
    hi = float(np.ceil(np.percentile(np.concatenate([me, them]), 99.8) / 2) * 2)
    return {"me": density(me, 2.0, lo, hi), "opp": density(them, 2.0, lo, hi), "margin": density(margin, 2.0),
            "raw": round(raw, 4), "p": round(float(opt.best_win.win_probability), 4), "names": names, "n": int(len(me)),
            "win_by": round(float(np.median(margin[margin > 0])), 1) if (margin > 0).any() else 0.0,
            "lose_by": round(float(np.median(-margin[margin < 0])), 1) if (margin < 0).any() else 0.0}


def players(opt) -> dict[str, dict]:
    """Player id -> the histogram of his points this week (yours and your opponent's starters)."""
    out: dict[str, dict] = {}
    if opt is None:
        return out
    sets = [(opt.sample_ids, opt.samples)]
    if getattr(opt, "opp_ids", None) and getattr(opt, "opp_player_samples", None) is not None:
        sets.append((opt.opp_ids, opt.opp_player_samples))
    for ids, arr in sets:
        if arr is None:
            continue
        for j, pid in enumerate(ids):
            if pid in out:
                continue
            x = np.asarray(arr[:, j], dtype=float)
            out[pid] = density(x, 1.0 if x.max() > 12 else 0.5)
    return out
