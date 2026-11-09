"""This week's simulated score distributions, for the charts the page draws.

P(win) is the share of simulated games in which your lineup outscores your
opponent's (:mod:`streamer.roster.lineup`); the editor already carries 5,000 of
those draws. Here they are binned into smooth densities -- your total, your
opponent's, and the margin between -- and, for each player, his own points, so
the page can show what the number is made of. Everything shown is drawn from
the same simulation as the percentage.
"""

from __future__ import annotations

import numpy as np

from .lineup import calibrated


def _smooth(y: np.ndarray, k: int = 3) -> np.ndarray:
    w = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2)
    return np.convolve(y, w / w.sum(), mode="same")


def density(x: np.ndarray, step: float, lo: float | None = None, hi: float | None = None, k: int = 3) -> dict:
    """A smoothed density of ``x`` on a regular grid, plus its 10th/50th/90th percentiles."""
    lo = float(np.floor(np.percentile(x, 0.3) / step) * step) if lo is None else lo
    hi = float(np.ceil(np.percentile(x, 99.7) / step) * step) if hi is None else hi
    if hi - lo < step * 3:
        hi = lo + step * 3
    edges = np.arange(lo, hi + step, step)
    h, _ = np.histogram(x, bins=edges)
    y = _smooth(h.astype(float), k)
    y = y / max(y.sum() * step, 1e-12)
    return {"x0": round(lo + step / 2, 3), "dx": step, "y": [round(float(v), 5) for v in y],
            "p10": round(float(np.percentile(x, 10)), 1), "p50": round(float(np.percentile(x, 50)), 1),
            "p90": round(float(np.percentile(x, 90)), 1), "mean": round(float(np.mean(x)), 1)}


def matchup(opt, my_name: str, opp_name: str) -> dict | None:
    """Your total, your opponent's and the margin, from the recommended lineup's draws."""
    if opt is None or opt.opp_samples is None or opt.samples is None or opt.opponent is None:
        return None
    col = {pid: i for i, pid in enumerate(opt.sample_ids)}
    mine = [p.player_id for _s, p in opt.best_win.flat() if p.player_id in col]
    if not mine:
        return None
    me = opt.samples[:, [col[i] for i in mine]].sum(axis=1)
    them = np.asarray(opt.opp_samples, dtype=float)
    n = min(len(me), len(them))
    me, them = me[:n], them[:n]
    margin = me - them
    raw = float((margin > 0).mean() + 0.5 * (margin == 0).mean())
    lo, hi = float(min(me.min(), them.min())), float(max(me.max(), them.max()))
    lo, hi = np.floor(lo / 2) * 2, np.ceil(hi / 2) * 2
    step = 2.0
    return {"me": density(me, step, lo, hi), "opp": density(them, step, lo, hi), "margin": density(margin, step, None, None),
            "raw": round(raw, 4), "p": round(float(calibrated(raw)), 4), "names": [my_name or "You", opp_name or "Opponent"],
            "win_by": round(float(np.median(margin[margin > 0])), 1) if (margin > 0).any() else 0.0,
            "lose_by": round(float(np.median(-margin[margin < 0])), 1) if (margin < 0).any() else 0.0}


def players(opt) -> dict[str, dict]:
    """Player id -> the density of his points this week (yours and your opponent's starters)."""
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
            x = np.asarray(arr[:, j], dtype=float)
            if pid in out:
                continue
            if float(x.max() - x.min()) < 0.05:
                out[pid] = {"fixed": round(float(x[0]), 2)}
                continue
            out[pid] = density(x, 1.0 if x.max() > 12 else 0.5, lo=max(0.0, float(np.floor(np.percentile(x, 0.5)))),
                               k=2)
    return out
