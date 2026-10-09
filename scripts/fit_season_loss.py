"""Refit who is about to lose his season, for the season simulator.

    python scripts/fit_season_loss.py [--dry-run] [--sims 1000] [--cache ready.pkl]

Replays 2022-2025 through the production projection (see
:mod:`streamer.roster.ros_backtest`) and simulates every checkpoint with each
player at the base share of absences that end a season. For every healthy or
questionable skill player, it fits a logistic on "plays under half of his
team's remaining games", over the simulator's own chance of that, on
exactly the inputs production reads (:func:`futures.season_loss_inputs`).
Each season is scored with the fit from the other three (log loss and AUC
against the simulator's), then the fit on all four is written into
``outcome_model.json`` (``futures.season_loss``): the coefficients, centred
so the base share holds on average, and the evidence.

The hazard by level was matched to what happened with this in place; after
a refit here, run ``scripts/fit_ros.py`` to match it again. DECISIONS.md,
"Who loses his season".
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
warnings.filterwarnings("ignore")
os.environ.setdefault("STREAMER_CREDIT_FREE", "1")

#: Ridge penalty on the standardised coefficients.
LAM = 1.0


def table(ready: dict, history: pd.DataFrame, cfg, sims: int) -> pd.DataFrame:
    """One row per healthy or questionable player and checkpoint: the inputs,
    the simulator's chance he plays under half of what is left (base share),
    and whether he did."""
    from streamer.roster import ros_backtest as rb
    from streamer.roster.futures import season_loss_applies, season_loss_inputs
    from streamer.roster.projections import season_loss_counts

    rows = []
    for (season, week), (players, weeks, bye, real) in sorted(ready.items()):
        for p in players:
            p.nfl_id = p.player_id               # the replay's ids are nflverse's
            p.season_loss = None                 # the base share: what the fit is over
        counts = season_loss_counts(players, history, season, week)
        cp = rb.checkpoint(None, cfg, season, week, None, n_sims=sims, ready=(players, weeks, bye, real))
        for p in players:
            if p.player_id not in counts or p.player_id not in real.index \
                    or (p.player_id, "avail") not in cp.simulated or not season_loss_applies(p, counts[p.player_id][1]):
                continue
            r = real.loc[p.player_id]
            if int(r["team_games"]) <= 0:
                continue
            avail = cp.simulated[(p.player_id, "avail")]
            rows.append({"season": season, "week": week, "player_id": p.player_id,
                         "p_half": float((avail < 0.5).mean()),
                         "half": float(int(r["games"]) / int(r["team_games"]) < 0.5),
                         **season_loss_inputs(p, *counts[p.player_id])})
        print(f"{season} week {week}: {len(rows)} rows", flush=True)
    return pd.DataFrame(rows)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def fit(X: pd.DataFrame, y: np.ndarray, off: np.ndarray) -> dict:
    """Ridge logistic with an offset, on standardised inputs."""
    from scipy.optimize import minimize

    mu, sd = X.mean(), X.std().replace(0, 1)
    Z = ((X - mu) / sd).to_numpy()

    def loss(b):
        eta = off + b[0] + Z @ b[1:]
        p = 1 / (1 + np.exp(-eta))
        val = np.sum(np.logaddexp(0, eta) - y * eta) + LAM * np.sum(b[1:] ** 2)
        grad = np.concatenate([[np.sum(p - y)], Z.T @ (p - y) + 2 * LAM * b[1:]])
        return val, grad

    b = minimize(loss, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B").x
    beta = pd.Series(b[1:], X.columns) / sd          # per unit of each input
    return {"b0": float(b[0] - (beta * mu).sum()), "beta": beta}


def _eta(m: dict, X: pd.DataFrame, off: np.ndarray) -> np.ndarray:
    return off + m["b0"] + X.to_numpy() @ m["beta"].to_numpy()


def log_loss(y: np.ndarray, eta: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0, eta) - y * eta))


def auc(y: np.ndarray, score: np.ndarray) -> float:
    ranks = pd.Series(score).rank().to_numpy()
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--cache", help="pickle of prepared checkpoints (made if missing)")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    from streamer.config import get_config
    from streamer.roster import outcome
    from streamer.roster import ros_backtest as rb
    from streamer.roster.projections import load_history

    cfg = get_config().for_profile("espn")
    if args.cache and Path(args.cache).exists():
        ready = pickle.loads(Path(args.cache).read_bytes())
    else:
        ready = rb.prepare_all(cfg)
        if args.cache:
            Path(args.cache).write_bytes(pickle.dumps(ready))
    history = load_history(cfg)
    d = table(ready, history, cfg, args.sims)
    inputs = [c for c in d.columns if c not in ("season", "week", "player_id", "p_half", "half")]
    X, y, off = d[inputs], d["half"].to_numpy(), _logit(d["p_half"].to_numpy())

    graded, held_auc, sim_auc = [], [], []
    for s in rb.SEASONS:
        tr, te = (d["season"] != s).to_numpy(), (d["season"] == s).to_numpy()
        m = fit(X[tr], y[tr], off[tr])
        eta = _eta(m, X[te], off[te])
        held_auc.append(round(auc(y[te], eta), 3))
        sim_auc.append(round(auc(y[te], off[te]), 3))
        graded.append({"season": s, "cases": int(te.sum()), "real": y[te].mean(), "simulated": d["p_half"][te].mean(),
                       "log loss, simulator": log_loss(y[te], off[te]), "log loss, fit": log_loss(y[te], eta),
                       "AUC, simulator": sim_auc[-1], "AUC, fit": held_auc[-1]})
    print(pd.DataFrame(graded).round(4).to_string(index=False))

    m = fit(X, y, off)
    feat = X.to_numpy() @ m["beta"].to_numpy()
    centre = float(np.log(np.mean(np.exp(feat))))
    model = outcome.load()
    old = (model.get("futures") or {}).get("season_loss") or {}
    new = {"base": old.get("base", 0.12), "cap": old.get("cap", 0.9), "centre": round(centre, 4),
           "coef": {k: round(float(v), 4) for k, v in m["beta"].items()},
           "evidence": {"cases": len(d), "seasons": list(rb.SEASONS),
                        "target": "plays under half of his team's remaining games",
                        "held_out_auc": held_auc, "simulator_auc": sim_auc}}
    print(pd.DataFrame({"before": pd.Series(old.get("coef") or {}), "after": pd.Series(new["coef"])}).round(3)
          .to_string())
    print(json.dumps({k: new[k] for k in ("centre", "evidence")}))
    if args.dry_run:
        return 0
    model.setdefault("futures", {})["season_loss"] = new
    outcome.MODEL_PATH.write_text(json.dumps(model, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {outcome.MODEL_PATH}; now run scripts/fit_ros.py to match the hazard by level again")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
