"""Refit how a player's record of missed games moves his absence hazard.

    python scripts/fit_hazard_history.py [--dry-run] [--sims 600] [--cache ready.pkl]

The season simulator starts new absences at a rate by position and
projection level (``futures.hazard``). This fits a multiplier on top, from
what a manager can see (:func:`futures.hazard_history_inputs`): the share of
his team's games a player has missed this season, the share of last season
he missed, and no games last season. Age was tried too and said nothing once
the level was known.

Fitted as a logistic over every game a contributor played, 2022-2025 (did he
miss his team's next game?), offset by the position-and-level rate, and
applied to the odds; centred so the average hazard is unchanged. Graded in
the replay (:mod:`streamer.roster.ros_backtest`): each season simulated with
the fit from the other three, against the simulator without it -- the share
of his team's remaining games each healthy or questionable player played,
and his points per team game. Writes ``futures.hazard_history`` into
``outcome_model.json``. DECISIONS.md, "A record of missed games".
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
os.environ.setdefault("STREAMER_CREDIT_FREE", "1")

SEASONS = (2022, 2023, 2024, 2025)
INPUTS = ("missed_share", "prev_missed", "no_prev")
#: Ridge penalty on the coefficients (the intercept is free).
LAM = 1.0
CLIP = [0.5, 3.0]
#: Quarterbacks left out: their 80% bands narrowed (a backup made to sit more
#: never gets the upside of taking over).
POSITIONS = ["RB", "WR", "TE"]


def events(cfg) -> pd.DataFrame:
    """Every game a contributor played (as ``fit_outcome_model.futures``
    counts them): his position, level, the inputs as of that game, and
    whether he missed his team's next one."""
    from fit_outcome_model import ABSENCE_LEVELS, skill_frame

    from streamer.data.nflverse import games_frame
    from streamer.roster.futures import hazard_history_inputs

    f = skill_frame(cfg).sort_values(["player_id", "season", "week"])
    games = games_frame(cfg)
    team_weeks = games[games["week"] <= 17].groupby(["team", "season"])["week"].apply(sorted).to_dict()
    played = f.groupby(["player_id", "season"])["week"].apply(set).to_dict()
    rows = []
    for (pid, season), x in f.groupby(["player_id", "season"]):
        if season not in SEASONS:
            continue
        have = set(x["week"])
        tw = team_weeks.get((x["team"].iloc[-1], season), [])
        prev = played.get((pid, season - 1))
        for w, proj, pos, c in zip(x["week"], x["proj"], x["position"], x["c_long"]):
            nxt = [t for t in tw if t > w]
            if w >= 17 or not nxt or not (proj >= 5) or c < 3 or pos not in POSITIONS:
                continue
            before = [t for t in tw if t <= w]
            row = hazard_history_inputs(sum(t in have for t in before), len(before),
                                        len(prev) if prev else None)
            rows.append({"season": season, "pos": pos, "lvl": int(np.digitize(proj, ABSENCE_LEVELS)),
                         "y": float(nxt[0] not in have), **row})
    return pd.DataFrame(rows)


def _offset(train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    g = train.groupby(["pos", "lvl"])["y"]
    rate = (g.sum() + 2 * train["y"].mean()) / (g.size() + 2)
    p = rows.set_index(["pos", "lvl"]).index.map(rate).to_numpy(float)
    return np.log(p / (1 - p))


def fit(ev: pd.DataFrame) -> dict:
    from scipy.optimize import minimize

    off = _offset(ev, ev)
    x = ev[list(INPUTS)].to_numpy(float)
    y = ev["y"].to_numpy()

    def nll(b):
        z = off + b[0] + x @ b[1:]
        return np.sum(np.logaddexp(0, z) - y * z) + LAM * np.sum(b[1:] ** 2)

    b = minimize(nll, np.zeros(len(INPUTS) + 1), method="BFGS").x
    # Centred so the average chance of an absence is what the table says.
    eta = x @ b[1:]
    base = 1 / (1 + np.exp(-off))
    lo, hi = float(np.min(eta)) - 5, float(np.max(eta)) + 5
    for _ in range(60):
        centre = 0.5 * (lo + hi)
        m = np.exp(eta - centre)
        if (base * m / (1 - base + base * m)).mean() > base.mean():
            lo = centre
        else:
            hi = centre
    return {"coef": {k: round(float(v), 4) for k, v in zip(INPUTS, b[1:])}, "centre": round(centre, 4),
            "clip": CLIP, "positions": POSITIONS,
            "evidence": {"games": int(len(ev)), "seasons": sorted(int(s) for s in ev.season.unique())}}


def held_out_loss(ev: pd.DataFrame) -> pd.DataFrame:
    """Log loss on each season's next-game absences, fit on the others."""
    rows = []
    for s in SEASONS:
        tr, te = ev[ev.season != s], ev[ev.season == s]
        hh = fit(tr)
        z0 = _offset(tr, te)
        eta = te[list(INPUTS)].to_numpy(float) @ np.array([hh["coef"][k] for k in INPUTS]) - hh["centre"]
        m = np.clip(np.exp(eta), *CLIP)
        p0 = 1 / (1 + np.exp(-z0))
        p1 = p0 * m / (1 - p0 + p0 * m)       # on the odds, as the simulator applies it
        y = te["y"].to_numpy()

        def ll(p, y=y):
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        rows.append({"season": s, "loss without": ll(p0), "loss with": ll(p1)})
    return pd.DataFrame(rows).set_index("season")


def replay(ready: dict, history: pd.DataFrame, cfg, sims: int, fits: dict | None) -> pd.DataFrame:
    """Each checkpoint simulated with the multiplier fitted without its
    season (``fits``: season -> hazard_history; None: none)."""
    from streamer.roster import ros_backtest as rb
    from streamer.roster.futures import hazard_mult
    from streamer.roster.projections import season_loss_counts

    out = []
    for (season, week), (players, weeks, bye, real) in sorted(ready.items()):
        for p in players:
            p.nfl_id = p.player_id               # the replay's ids are nflverse's
        counts = season_loss_counts(players, history, season, week)
        conf = {"hazard_history": fits[season]} if fits else {}
        for p in players:
            p.hazard_mult = hazard_mult(p, *counts[p.player_id], conf) if p.player_id in counts else None
        cp = rb.checkpoint(None, cfg, season, week, None, n_sims=sims, ready=(players, weeks, bye, real))
        d = rb.frame(cp)
        d["mult"] = d.player_id.map({p.player_id: p.hazard_mult or 1.0 for p in players})
        out.append(d)
    return pd.concat(out, ignore_index=True)


def grade(d: pd.DataFrame) -> pd.DataFrame:
    """Per season: how far the simulated share of games played was from the
    real one, the miss on points per team game, and how many outcomes the
    80% band held (all players, quarterbacks)."""
    d = d[(d.team_games >= 4) & d.status.isin(["", "Q"])].copy()
    d["share"] = d.games / d.team_games
    d["in80"] = (d.actual >= d.q10) & (d.actual <= d.q90)
    rows = {}
    for s, g in d.groupby("season"):
        rows[s] = {"avail sq err": float(((g.sim_avail - g.share) ** 2).mean()),
                   "points miss": float((g.sim_mean - g.actual).abs().mean()),
                   "in80": float(g.in80.mean()), "in80 QB": float(g[g.position == "QB"].in80.mean()),
                   "players": len(g)}
    return pd.DataFrame(rows).T


def by_level(d: pd.DataFrame) -> pd.Series:
    """Simulated minus real share of games played, by season value."""
    d = d[(d.team_games >= 4) & d.status.isin(["", "Q"])]
    lvl = pd.cut(d.ros_value, [-1, 4, 8, 12, 16, 99], labels=["<4", "4-8", "8-12", "12-16", "16+"])
    return (d.sim_avail - d.games / d.team_games).groupby(lvl, observed=True).mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sims", type=int, default=600)
    ap.add_argument("--cache", help="pickle of prepared checkpoints (made if missing)")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    from streamer.config import get_config
    from streamer.roster import outcome
    from streamer.roster import ros_backtest as rb
    from streamer.roster.projections import load_history

    cfg = get_config().for_profile("espn")
    ev = events(cfg)
    print(held_out_loss(ev).round(4).to_string())

    if args.cache and Path(args.cache).exists():
        ready = pickle.loads(Path(args.cache).read_bytes())
    else:
        ready = rb.prepare_all(cfg)
        if args.cache:
            Path(args.cache).write_bytes(pickle.dumps(ready))
    history = load_history(cfg)
    fits = {s: fit(ev[ev.season != s]) for s in SEASONS}
    without = replay(ready, history, cfg, args.sims, None)
    with_ = replay(ready, history, cfg, args.sims, fits)
    before, after = grade(without), grade(with_)
    table = before.join(after, lsuffix=" without", rsuffix=" with")
    print(table.round(4).T.to_string())
    print("played, simulated minus real, by level:")
    print(pd.DataFrame({"without": by_level(without), "with": by_level(with_)}).round(3).to_string())
    d = with_[(with_.team_games >= 4) & with_.status.isin(["", "Q"])]
    d = d.assign(share=d.games / d.team_games, band=pd.cut(d.mult, [0, 0.8, 1.2, 1.6, 9]))
    print(d.groupby("band", observed=True).agg(n=("share", "size"), played=("share", "mean"),
                                               simulated=("sim_avail", "mean")).round(3).to_string())
    better = after["avail sq err"].mean() < before["avail sq err"].mean()
    worse = int((after["avail sq err"] > before["avail sq err"]).sum())
    print(f"availability: {'better' if better else 'not better'} overall, worse in {worse} of {len(after)} seasons")

    final = fit(ev)
    print(json.dumps(final))
    if args.dry_run or not better or worse > 1:
        return 0
    model = outcome.load()
    model["futures"]["hazard_history"] = final
    outcome.MODEL_PATH.write_text(json.dumps(model, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {outcome.MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
