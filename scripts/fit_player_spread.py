"""Should a player's own style widen or narrow his weekly range?

    python scripts/fit_player_spread.py [--cache rows.parquet]

Replays 2022-2025, weeks 3-18, through the production volume model and its
range: the spread of misses by position and projection level
(``projections.sd_table``) shaped by ``outcome.shape``. Every player's
spread is then multiplied by ``exp(c + sum(beta x z))`` over his last 17
games before the week (standardised within position):

* ``td``   -- touchdown dependence: 6 x expected TDs / expected points;
* ``rec``  -- targets / (targets + carries): catches are steadier in PPR;
* ``vol``  -- his own swing: sd of (points - expected points), shrunk to
  the position's with 6 games;
* ``env``  -- his team's implied total this week over 23.

``(c, beta)`` per position is fitted on three seasons (quantile loss at the
10th and 90th percentiles) and scored on the fourth, with the Brier score of
his chance of a top-12 week at the position as a second grade; kept only if
both are better overall and quantile loss is worse in at most one season.
DECISIONS.md, "Each player's own range".
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
POSITIONS = ("QB", "RB", "WR", "TE")
FEATURES = ("td", "rec", "vol", "env")
VOL_PRIOR = 6.0
TOP = 12


def player_features(prior: pd.DataFrame) -> pd.DataFrame:
    """Per player: the style features over his last 17 games (production's)."""
    from streamer.roster.projections import style_features

    return style_features(prior, VOL_PRIOR)


def build(cfg) -> pd.DataFrame:
    from streamer.data.nflverse import games_frame
    from streamer.roster import outcome
    from streamer.roster import projections as pj
    from streamer.teams import normalize_team

    h = pj.load_history(cfg)
    h = h.assign(_team=h["team"].map(normalize_team))
    games = games_frame(cfg)
    games = games[games["game_type"] == "REG"].assign(team=lambda g: g["team"].map(normalize_team))
    implied = games.set_index(["season", "week", "team"])["team_implied_total"]
    bars = (h[h["position"].isin(POSITIONS)].groupby(["season", "week", "position"])["fantasy_points_ppr"]
            .apply(lambda x: x.nlargest(TOP).min() if len(x) >= TOP else np.nan).groupby("position").mean())
    rows = []
    for season in (2022, 2023, 2024, 2025):
        sds = pj.sd_table(h[h["season"] < season], cfg)
        for week in range(3, 19):
            cur = h[(h["season"] == season) & (h["week"] == week)]
            if cur.empty:
                continue
            prior = h[(h["season"] < season) | ((h["season"] == season) & (h["week"] < week))]
            feats = player_features(prior[prior["season"] >= season - 2])
            t = pj.player_table(h, season, week, cfg)
            got = cur.groupby("player_id")["fantasy_points_ppr"].sum()
            team_now = cur.groupby("player_id")["_team"].first()
            t = t[t["player_id"].isin(got.index) & t["position"].isin(POSITIONS) & t["blend"].notna()
                  & t["player_id"].isin(feats.index)]
            for r in t.itertuples():
                m = float(r.blend)
                p, z = outcome.shape(r.position, m)
                f = feats.loc[r.player_id]
                rows.append({"season": season, "week": week, "position": r.position, "pred": m,
                             "sd": pj.lookup_sd(sds, r.position, m), "pts": float(got[r.player_id]),
                             "z10": float(np.interp(0.1, p, z)), "z90": float(np.interp(0.9, p, z)),
                             "p_grid": p, "z_grid": z, "bar": float(bars[r.position]),
                             "td": f["td"], "rec": f["rec"], "vol": f["vol"],
                             "env": float(implied.get((season, week, team_now[r.player_id]), 23.0)) / 23.0})
        print(season, len(rows), flush=True)
    return pd.DataFrame(rows)


def standardise(d: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
    mu, sd = ref[list(FEATURES)].mean(), ref[list(FEATURES)].std().replace(0, 1)
    return ((d[list(FEATURES)] - mu) / sd).fillna(0.0).to_numpy()


def pinball(d: pd.DataFrame, mult: np.ndarray) -> float:
    s = d["sd"].to_numpy() * mult
    y = d["pts"].to_numpy()
    lo, hi = d["pred"].to_numpy() + d["z10"].to_numpy() * s, d["pred"].to_numpy() + d["z90"].to_numpy() * s
    return float((np.maximum(0.1 * (y - lo), 0.9 * (lo - y)) + np.maximum(0.9 * (y - hi), 0.1 * (hi - y))).mean())


def brier(d: pd.DataFrame, mult: np.ndarray) -> float:
    s = d["sd"].to_numpy() * mult
    zb = (d["bar"].to_numpy() - d["pred"].to_numpy()) / np.maximum(s, 1e-6)
    pb = np.array([1.0 - float(np.interp(z, zg, pg, left=0.0, right=1.0))
                   for z, zg, pg in zip(zb, d["z_grid"], d["p_grid"])])
    hit = (d["pts"].to_numpy() >= d["bar"].to_numpy()).astype(float)
    return float(np.mean((pb - hit) ** 2))


def fit(d: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    def loss(theta):
        return pinball(d, np.exp(theta[0] + x @ theta[1:]))
    return minimize(loss, np.zeros(x.shape[1] + 1), method="Nelder-Mead",
                    options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6}).x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache")
    ap.add_argument("--write", nargs="*", default=None,
                    help="positions to fit on every season and write to outcome_model.json")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    from streamer.config import get_config

    cfg = get_config().for_profile("espn")
    if args.cache and Path(args.cache).exists():
        d = pd.read_pickle(args.cache)
    else:
        d = build(cfg)
        if args.cache:
            d.to_pickle(args.cache)
    for pos in POSITIONS:
        p = d[d["position"] == pos].reset_index(drop=True)
        print(f"\n{pos}: {len(p)} player-weeks")
        for label, use in (("width only", []), ("width + style", list(FEATURES))):
            wins, l0, l1, b0, b1, thetas = 0, 0.0, 0.0, 0.0, 0.0, []
            for s in sorted(p["season"].unique()):
                tr, te = p[p["season"] != s], p[p["season"] == s]
                xtr, xte = standardise(tr, tr), standardise(te, tr)
                cols = [FEATURES.index(f) for f in use]
                th = fit(tr, xtr[:, cols])
                mult = np.exp(th[0] + xte[:, cols] @ th[1:])
                a, b = pinball(te, np.ones(len(te))), pinball(te, mult)
                wins += b < a
                l0 += a * len(te)
                l1 += b * len(te)
                b0 += brier(te, np.ones(len(te))) * len(te)
                b1 += brier(te, mult) * len(te)
                thetas.append(np.round(th, 3))
            n = len(p)
            keep = l1 < l0 and b1 < b0 and wins >= 3
            print(f"  {label:14} qloss {l0 / n:.4f} -> {l1 / n:.4f} ({wins}/4)  top-{TOP} Brier {b0 / n:.5f} -> "
                  f"{b1 / n:.5f}  {'KEEP' if keep else 'drop'}  theta (c, {', '.join(use)}): {thetas[-1].tolist()}")
    if args.write:
        from streamer.roster import outcome

        model = outcome.load()
        fitted = {}
        for pos in args.write:
            p = d[d["position"] == pos].reset_index(drop=True)
            th = fit(p, standardise(p, p))
            fitted[pos] = {"c": round(float(th[0]), 4),
                           "beta": {f: round(float(b), 4) for f, b in zip(FEATURES, th[1:])},
                           "mean": {f: round(float(p[f].mean()), 4) for f in FEATURES},
                           "sd": {f: round(float(p[f].std()), 4) for f in FEATURES}}
        model["player_spread"] = {"positions": fitted, "vol_prior_games": VOL_PRIOR,
                                  "evidence": "scripts/fit_player_spread.py, 2022-2025, held out by season"}
        outcome.MODEL_PATH.write_text(__import__("json").dumps(model, indent=1) + "\n", encoding="utf-8")
        print("wrote", outcome.MODEL_PATH, fitted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
