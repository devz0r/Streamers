"""Refit how the market values players, from Yahoo's "% rostered".

    python scripts/fit_perception.py [--profile yahoo] [--dry-run]

Every player in the Yahoo snapshot (rostered and free agents) carries the
share of all Yahoo leagues that roster him: the market's revealed opinion.
It is regressed (logit, position intercepts) on the things a manager can
see -- the platform projection, last seasons' points a game, this season's,
and, when FANTASYPROS_API_KEY is set, the FantasyPros consensus rank (as the
market value of that rank) -- built exactly as
:mod:`streamer.roster.perception` builds them, and the normalised
coefficients become the blend's weights. Prints the
weights with 90% bootstrap intervals and the out-of-sample R^2 of the blend
against our own projection alone, then writes ``perception.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

os.environ.setdefault("STREAMER_CREDIT_FREE", "1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="yahoo")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.disable(logging.WARNING)

    from streamer.config import get_config
    from streamer.league.store import load_snapshot
    from streamer.rankings import rank_week
    from streamer.roster import perception
    from streamer.roster.projections import load_history, project_snapshot

    cfg = get_config().for_profile(args.profile)
    snap = load_snapshot(cfg)
    project_snapshot(snap, cfg, rank_week(snap.week, snap.season, cfg, allow_network=False),
                     allow_network=False)
    from streamer.roster import consensus

    conf = dict(perception.load())
    lines = perception.season_lines(load_history(cfg), snap.season)
    has_consensus = consensus.attach(snap, cfg) >= 100
    players = [p for p in snap.all_players()
               if p.position in ("QB", "RB", "WR", "TE") and p.percent_owned is not None
               and p.ros_value is not None and p.nfl_id]
    seen = perception.perceive(players, lines, snap.season, conf)
    ecr = consensus.market_values(snap, seen) if has_consensus else {}
    if has_consensus:
        players = [p for p in players if p.player_id in ecr]
    d = pd.DataFrame([{"pos": p.position, "own": p.percent_owned, "ours": p.ros_value,
                       "platform": seen[p.player_id].platform, "reputation": seen[p.player_id].reputation,
                       "season": seen[p.player_id].season, "consensus": ecr.get(p.player_id, np.nan)}
                      for p in players])
    own = d["own"].clip(1, 99) / 100
    y = np.log(own / (1 - own)).to_numpy()
    pos = pd.get_dummies(d["pos"]).astype(float).to_numpy()
    cols = ["platform", "reputation", "season"] + (["consensus"] if has_consensus else [])
    X = d[cols].to_numpy()

    def weights(idx):
        A = np.column_stack([pos[idx], X[idx]])
        b, *_ = np.linalg.lstsq(A, y[idx], rcond=None)
        b = b[pos.shape[1]:]
        return b / b.sum()

    rng = np.random.default_rng(0)
    full = weights(np.arange(len(y)))
    boot = np.array([weights(rng.integers(0, len(y), len(y))) for _ in range(500)])

    def cv(features):
        idx = rng.permutation(len(y))
        err = 0.0
        for fold in np.array_split(idx, 5):
            train = np.setdiff1d(idx, fold)
            A = lambda ii: np.column_stack([pos[ii], features[ii]])  # noqa: E731
            b, *_ = np.linalg.lstsq(A(train), y[train], rcond=None)
            err += ((y[fold] - A(fold) @ b) ** 2).sum()
        return 1 - err / ((y - y.mean()) ** 2).sum()

    print(f"{len(y)} players, week {snap.week} of {snap.season}")
    for c, w, lo, hi in zip(cols, full, *np.percentile(boot, [5, 95], axis=0)):
        print(f"  {c:11} {w:5.2f}  (90% interval {lo:5.2f} to {hi:5.2f})")
    print(f"  out-of-sample R^2: blend {cv(X):.3f}, our projection alone {cv(d[['ours']].to_numpy()):.3f}")
    clipped = np.clip(full, 0.0, None)
    shares = {c: float(w / clipped.sum()) for c, w in zip(cols, clipped)}
    if has_consensus:
        # The consensus is folded in after the other three (perception.with_consensus).
        conf["consensus_weight"] = round(shares.pop("consensus"), 3)
        rest = sum(shares.values()) or 1.0
        shares = {c: w / rest for c, w in shares.items()}
    conf["weights"] = {c: round(w, 3) for c, w in shares.items()}
    conf["fitted"] = {"season": snap.season, "week": snap.week, "players": int(len(y))}
    if args.dry_run:
        print(json.dumps(conf["weights"]))
        return 0
    perception.MODEL_PATH.write_text(json.dumps(conf, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {perception.MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
