"""Refit the season simulator's availability model from replayed seasons.

    python scripts/fit_ros.py [--dry-run] [--sims 600]

Replays 2022-2025 through the production projection (see
:mod:`streamer.roster.ros_backtest`), then fits, from what happened after
each checkpoint:

* ``ir``: for a player on a reserve list (IR, PUP, NFI), the chance he never
  plays again that season -- by season value and games already missed --
  and, if he does, how many of his team's games it takes;

and grades the simulator before and after on seasons the fit did not see
(fitted on 2022-2023, scored on 2024-2025, then the reverse). Writes the
parameters into ``outcome_model.json`` (``futures``).
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

IR_LEVELS = [6.0, 12.0]
HAZARD_LEVELS = [4.0, 8.0, 12.0, 16.0]
#: Rounds of matching simulated availability to what happened.
ROUNDS = 3
#: Two structural choices the replay settled (see DECISIONS.md, "Rest of
#: season, graded"): the absence hazard reads the projection, not the hidden
#: true level, and the persistent error is multiplicative.
STRUCTURE = {"hazard_on": "projection", "error_shape": "lognormal"}
#: Fitted alongside the calibrations, and written with them.
FITTED = ("ir", "hazard_scale", "level_offset", "error_scale", "visibility", "projection_noise")
#: Weeks between the two reads of the projection (checkpoints 3-5 apart).
MOVE_GAP = 4
IR_MISSED = [3, 6]
WAIT_GAMES = 12
#: Pseudo-cases pulling a thin cell of the gone table toward the whole.
PRIOR = 20.0


def returns(ready: dict, history: pd.DataFrame, seasons) -> pd.DataFrame:
    """Every reserve-list player at a checkpoint: his level, games missed, and
    his team games until he played again (NaN: never that season)."""
    played = history.groupby(["player_id", "season"])["week"].apply(set).to_dict()
    rows = []
    for (season, _week), (players, weeks, bye, _real) in ready.items():
        if season not in seasons:
            continue
        for p in players:
            if p.status not in ("IR", "INJURY_RESERVE", "PUP", "PUP-R", "NFI", "IR-R"):
                continue
            tw = [w for w in weeks if w not in bye.get(p.team or "", set())]
            ws = played.get((p.player_id, season), set())
            back = next((i for i, w in enumerate(tw) if w in ws), np.nan)    # team games he misses
            rows.append({"level": float(p.ros_value or 0.0), "missed": int(p.games_missed or 0), "back": back})
    return pd.DataFrame(rows)


def fit_ir(r: pd.DataFrame) -> dict:
    overall = float(r["back"].isna().mean())
    li = np.digitize(r["level"], IR_LEVELS)
    mi = np.digitize(r["missed"], IR_MISSED)
    gone = []
    for a in range(len(IR_LEVELS) + 1):
        row = []
        for b in range(len(IR_MISSED) + 1):
            cell = r[(li == a) & (mi == b)]
            row.append(round((float(cell["back"].isna().sum()) + PRIOR * overall) / (len(cell) + PRIOR), 3))
        gone.append(row)
    back = r["back"].dropna().clip(upper=WAIT_GAMES).astype(int)
    counts = back.value_counts().reindex(range(0, WAIT_GAMES + 1), fill_value=0)
    return {"levels": IR_LEVELS, "missed": IR_MISSED, "gone": gone,
            "wait": [round(float(x), 4) for x in counts / counts.sum()],
            "evidence": {"cases": int(len(r)), "never_back": round(overall, 3)}}


CAL_LEVELS = [4.0, 7.0, 10.0, 14.0, 18.0]
#: Games played for a player's points per game to be graded (the replay's
#: own threshold, :data:`streamer.roster.ros_backtest.PG_GAMES`).
PG_GAMES = 4
POSITIONS = ("QB", "RB", "WR", "TE")


def calibrate(simulated, conf: dict, seasons, rounds: int = 6) -> dict:
    """Three calibrations fitted together, a round at a time on the same
    simulated seasons, each nudged toward what happened:

    * ``hazard_scale`` -- how often healthy players at each level played;
    * ``level_offset`` -- what they scored per game played, by position and
      level;
    * ``error_scale`` -- how far their level runs from the projection: the
      middle half of the simulated points per game played should hold half
      of what players who kept playing scored. Points per team game would
      mix in absences, which have their own fit; a level error widened to
      cover them made early form far too persistent (a QB's weeks 5-8
      predicted his weeks 9-17 at 0.77 simulated, 0.16 real).
    """
    from scipy.stats import norm

    conf = dict(conf)
    conf.setdefault("visibility", {pos: 1.0 for pos in POSITIONS})
    conf.setdefault("projection_noise", {pos: 0.0 for pos in POSITIONS})
    nh, nc = len(HAZARD_LEVELS) + 1, len(CAL_LEVELS) + 1
    conf["hazard_scale"] = {"levels": HAZARD_LEVELS, "scale": [1.2] * nh}
    conf["level_offset"] = {"levels": CAL_LEVELS, **{pos: [0.0] * nc for pos in POSITIONS}}
    conf["error_scale"] = {"levels": CAL_LEVELS, **{pos: [1.0] * nc for pos in POSITIONS}}
    for _ in range(rounds):
        d = simulated(conf, seasons)
        d = d[(d.team_games >= 4) & d.status.isin(["", "Q"])]
        hs = list(conf["hazard_scale"]["scale"])
        b = np.digitize(d.ros_value, HAZARD_LEVELS)
        for i in range(nh):
            g = d[b == i]
            if len(g) >= 50:
                miss_sim, miss_act = 1.0 - g.sim_avail.mean(), 1.0 - (g.games / g.team_games).mean()
                hs[i] = float(np.clip(hs[i] * miss_act / max(miss_sim, 1e-3), 0.2, 6.0))
        conf["hazard_scale"] = {"levels": HAZARD_LEVELS, "scale": [round(x, 3) for x in hs]}
        off = {pos: list(conf["level_offset"][pos]) for pos in POSITIONS}
        played = d[d.games >= 2]
        lp = np.digitize(played.ros_value, CAL_LEVELS)
        for pos in POSITIONS:
            for i in range(nc):
                g = played[(played.position == pos) & (lp == i)]
                if len(g) >= 40:
                    w = g.games
                    sim_pg = g.sim_mean / g.sim_avail.clip(lower=1e-3)
                    off[pos][i] = round(off[pos][i] + float((w * (g.actual_per_game - sim_pg)).sum() / w.sum()), 3)
        conf["level_offset"] = {"levels": CAL_LEVELS, **off}
        es = {pos: list(conf["error_scale"][pos]) for pos in POSITIONS}
        kept = d[(d.games >= PG_GAMES) & d.pg_q25.notna()]
        lk = np.digitize(kept.ros_value, CAL_LEVELS)
        for pos in POSITIONS:
            for i in range(nc):
                # A thin cell borrows its level's whole-league evidence.
                g = kept[(lk == i) & (kept.position == pos)]
                if len(g) < 80:
                    g = kept[lk == i]
                if len(g) >= 80:
                    in50 = float(((g.actual_per_game >= g.pg_q25) & (g.actual_per_game <= g.pg_q75)).mean())
                    ratio = 0.6745 / norm.ppf(0.5 + min(max(in50, 0.05), 0.95) / 2)
                    es[pos][i] = round(float(np.clip(es[pos][i] * ratio ** 0.7, 0.05, 2.5)), 3)
        conf["error_scale"] = {"levels": CAL_LEVELS, **es}
        # What a manager sees, against what a real projection did a few weeks
        # on: how far it moved, and how much of the move held up in the points
        # that followed. A simulated move is ``v`` times the player's real
        # change (covariance C0 with what follows, variance V0) plus the
        # projection's own noise (variance N), so the two real numbers give
        # both: v = slope * sd^2 / C0, N = sd^2 - v^2 V0.
        from streamer.roster.futures import PROJECTION_NOISE_PERSIST as rho
        from streamer.roster.ros_backtest import moves

        vis, noise = dict(conf["visibility"]), dict(conf["projection_noise"])
        spread = (1 - rho ** (2 * MOVE_GAP)) / (1 - rho ** 2)       # noise variance over the gap, per unit step
        for pos in POSITIONS:
            mv = moves(d[(d.position == pos) & (d.ros_value >= 6) & d.status.isin(["", "Q"])])
            if mv.get("n", 0) < 100:
                continue
            v0, n0 = max(vis[pos], 0.05), spread * noise[pos] ** 2
            c_true = mv["sim_slope"] * mv["sim_sd"] ** 2 / v0
            v_true = max(mv["sim_sd"] ** 2 - n0, 1e-6) / v0 ** 2
            target = mv["real_sd"] ** 2
            v_new = float(np.clip(max(mv["real_slope"], 0.0) * target / max(c_true, 1e-6), 0.1, 1.0))
            n_new = max(target - v_new ** 2 * v_true, 0.0)
            vis[pos] = round(0.5 * (vis[pos] + v_new), 3)
            noise[pos] = round(0.5 * (noise[pos] + float(np.sqrt(n_new / spread))), 3)
        conf["visibility"], conf["projection_noise"] = vis, noise
    return conf


def fit_hazard_scale(simulated, conf: dict, seasons, start: float = 1.2) -> dict:
    """A multiplier on the absence hazard per projection level, chosen so the
    simulated share of remaining games played matches what healthy players
    actually played, level by level."""
    scale = [start] * (len(HAZARD_LEVELS) + 1)
    for _ in range(ROUNDS):
        trial = {**conf, "hazard_scale": {"levels": HAZARD_LEVELS, "scale": scale}}
        d = simulated(trial, seasons)
        d = d[(d.team_games >= 4) & d.status.isin(["", "Q"])]
        b = np.digitize(d.ros_value, HAZARD_LEVELS)
        for i in range(len(scale)):
            g = d[b == i]
            if len(g) < 50:
                continue
            miss_sim, miss_act = 1.0 - g.sim_avail.mean(), 1.0 - (g.games / g.team_games).mean()
            scale[i] = float(np.clip(scale[i] * miss_act / max(miss_sim, 1e-3), 0.2, 6.0))
    return {"levels": HAZARD_LEVELS, "scale": [round(x, 3) for x in scale]}


def grade(d: pd.DataFrame) -> pd.Series:
    d = d[d.team_games >= 4]
    ir = d[d.status == "IR"]
    lvl = pd.cut(d.ros_value, [-1, 4, 7, 10, 14, 18, 99], labels=["<4", "4-7", "7-10", "10-14", "14-18", "18+"])
    out = {"n": len(d), "bias": (d.sim_mean - d.actual).mean(), "miss": (d.sim_mean - d.actual).abs().mean(),
           "in80": ((d.actual >= d.q10) & (d.actual <= d.q90)).mean(), "below10": (d.actual < d.q10).mean(),
           "above90": (d.actual > d.q90).mean(), "in50": ((d.actual >= d.q25) & (d.actual <= d.q75)).mean(),
           "IR bias": (ir.sim_mean - ir.actual).mean() if len(ir) else np.nan}
    kept = d[(d.games >= PG_GAMES) & d.pg_q25.notna() & (d.status != "IR")] if "pg_q25" in d else d.iloc[:0]
    for pos, g in kept.groupby("position"):
        out[f"pg in50 {pos}"] = ((g.actual_per_game >= g.pg_q25) & (g.actual_per_game <= g.pg_q75)).mean()
    from streamer.roster.ros_backtest import moves, persistence

    for pos in POSITIONS:
        g = d[(d.position == pos) & (d.ros_value >= 6) & d.status.isin(["", "Q"])]
        if "e_mean" in g:
            out[f"early->late {pos} real"], out[f"early->late {pos} sim"], _n = persistence(g)
        if "mv_mean" in g:
            mv = moves(g)
            if mv.get("n", 0) >= 20:
                out[f"move holds {pos} real"], out[f"move holds {pos} sim"] = mv["real_slope"], mv["sim_slope"]
                out[f"move sd {pos} real"], out[f"move sd {pos} sim"] = mv["real_sd"], mv["sim_sd"]
    for k, g in d.groupby(lvl, observed=True):
        out[f"bias {k}"] = (g.sim_mean - g.actual).mean()
    for k, g in d[d.status != "IR"].groupby(lvl, observed=True):
        out[f"played {k}"] = (g.sim_avail - g.games / g.team_games).mean()
    return pd.Series(out)


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
    if args.cache and Path(args.cache).exists():
        ready = pickle.loads(Path(args.cache).read_bytes())
    else:
        ready = rb.prepare_all(cfg)
        if args.cache:
            Path(args.cache).write_bytes(pickle.dumps(ready))
    history = load_history(cfg)
    model = outcome.load()

    def simulated(futures_conf: dict, seasons) -> pd.DataFrame:
        trial = json.loads(json.dumps(model))
        trial["futures"] = futures_conf
        saved = outcome.load
        outcome.load = lambda: trial
        try:
            return rb.run(cfg, ready={k: v for k, v in ready.items() if k[0] in seasons}, n_sims=args.sims,
                          history=history)
        finally:
            outcome.load = saved

    def fitted(seasons) -> dict:
        conf = {**model["futures"], **STRUCTURE}
        conf["ir"] = fit_ir(returns(ready, history, seasons))
        return calibrate(simulated, conf, seasons)

    halves = [((2022, 2023), (2024, 2025)), ((2024, 2025), (2022, 2023))]
    rows = {}
    for train, test in halves:
        rows[f"before, {test[0]}-{test[1]}"] = grade(simulated(dict(model["futures"]), test))
        rows[f"after, {test[0]}-{test[1]}"] = grade(simulated(fitted(train), test))
    print(pd.DataFrame(rows).round(3).to_string())

    final = fitted(rb.SEASONS)
    print(json.dumps({k: final[k] for k in (*FITTED, *STRUCTURE)}))
    if args.dry_run:
        return 0
    model["futures"] = final
    outcome.MODEL_PATH.write_text(json.dumps(model, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {outcome.MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
