"""Fit the outcome model the lineup simulator and waivers read.

    python scripts/fit_outcome_model.py [--profile espn]

Writes ``src/streamer/roster/outcome_model.json`` from the completed training
seasons, using the production projection (so residuals are measured against
exactly what the tools project):

* ``shape``: standardised residual quantiles by position and projection
  level, re-centred to mean 0 and scaled to sd 1, so a player's simulated
  mean and spread stay his projection and sd while the skew is kept.
* ``correlation``: residual correlation between roles in the same game, same
  team and opponents, D/ST and K included. Pairs with |r| < 0.03 or fewer
  than 400 games are left out (treated as independent).
* ``drift``: sd of the change in a player's per-game projection four games
  later, by position and level, with rookie and second-year multipliers and
  their average rise relative to veterans.

Rerun after a season is added to ``train_seasons``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from streamer.config import get_config  # noqa: E402
from streamer.roster import outcome  # noqa: E402
from streamer.roster.projections import (  # noqa: E402
    SD_BUCKETS,
    _blend,
    _trailing_features,
    load_history,
    position_rates,
)
from streamer.teams import normalize_team_series  # noqa: E402

PROBS = np.array([0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995])
MIN_PAIRS = 400
MIN_ABS_R = 0.03


def _standardised_shape(z: np.ndarray) -> dict:
    """Quantiles of ``z`` on PROBS, re-centred and re-scaled so sampling by
    interpolation reproduces mean 0 and sd 1."""
    q = np.quantile(z, PROBS)
    fine = np.linspace(0.0005, 0.9995, 4000)
    draw = np.interp(fine, PROBS, q)
    q = (q - draw.mean()) / draw.std()
    return {"p": [round(float(x), 4) for x in PROBS], "z": [round(float(x), 3) for x in q], "n": int(len(z))}


def skill_frame(cfg) -> pd.DataFrame:
    conf = cfg.raw["roster"]
    hist = load_history(cfg)
    hist = hist[hist["season"].isin(cfg.train_seasons)].copy()
    feats = _trailing_features(hist, int(conf["trailing_games"]), int(conf.get("long_games", 17)),
                               float(conf.get("volume_halflife_games", 2.5)))
    pos_mean, pos_eff = position_rates(hist)
    feats["proj"] = _blend(feats, feats["position"].map(pos_mean), feats["position"].map(pos_eff), conf)
    feats = feats[feats["proj"].notna()].copy()
    feats["resid"] = feats["fantasy_points_ppr"] - feats["proj"]
    feats["level"] = np.digitize(feats["proj"], SD_BUCKETS[1:-1])
    sd = feats.groupby(["position", "level"])["resid"].transform("std")
    feats["z"] = feats["resid"] / sd
    # nflverse player_stats carries the game id; the opportunity frame does not.
    ids = []
    for season in sorted(feats["season"].unique()):
        ps = pd.read_parquet(cfg.raw_dir / f"player_stats_{season}.parquet",
                             columns=["player_id", "season", "week", "game_id", "season_type"])
        ids.append(ps[ps["season_type"].eq("REG")].drop(columns="season_type"))
    ids = pd.concat(ids).drop_duplicates(["player_id", "season", "week"])
    ids["season"] = ids["season"].astype(int)
    ids["week"] = ids["week"].astype(int)
    return feats.merge(ids, on=["player_id", "season", "week"], how="left")


def team_units(cfg) -> pd.DataFrame:
    """D/ST and K residual z-scores per team-game, against a linear fit on the
    implied totals (the Vegas anchor both streaming models start from)."""
    from streamer.data.nflverse import games_frame, load_pbp
    from streamer.features.build import build_dst_features, build_kicker_features

    pbp = load_pbp(list(cfg.train_seasons))
    games = games_frame(cfg)
    out = []
    for kind, frame in (("DST", build_dst_features(pbp, games, cfg)),
                        ("K", build_kicker_features(pbp, games, cfg))):
        d = frame.dropna(subset=["fantasy_points"]).copy()
        d["team"] = normalize_team_series(d["team"])
        x = np.column_stack([np.ones(len(d)),
                             d["team_implied_total"].fillna(d["team_implied_total"].mean()),
                             d["opp_implied_total"].fillna(d["opp_implied_total"].mean())])
        y = d["fantasy_points"].to_numpy(float)
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        r = y - x @ coef
        d["z"] = r / r.std()
        d["role"] = kind
        d["position"] = kind
        out.append(d[["game_id", "season", "team", "role", "position", "z"]])
    return pd.concat(out, ignore_index=True)


def correlations(skill: pd.DataFrame, units: pd.DataFrame) -> dict:
    sk = skill[(skill["proj"] >= 3) & skill["game_id"].notna()].copy()
    sk["rk"] = sk.groupby(["game_id", "team", "position"])["proj"].rank(ascending=False, method="first")
    pos, rk = sk["position"], sk["rk"]
    sk["role"] = np.select(
        [(pos == "QB") & (rk == 1), (pos == "RB") & (rk <= 2), (pos == "WR") & (rk <= 3), (pos == "TE") & (rk == 1)],
        ["QB", "RB" + rk.astype(int).astype(str), "WR" + rk.astype(int).astype(str), "TE"], default="")
    sk = sk[sk["role"] != ""]
    u = pd.concat([sk[["game_id", "team", "role", "z"]], units[["game_id", "team", "role", "z"]]])
    u = u.dropna().drop_duplicates(["game_id", "team", "role"])
    m = u.merge(u, on="game_id", suffixes=("_a", "_b"))
    # Each unordered pair once: opponents by team order, teammates by role.
    m = m[(m["team_a"] < m["team_b"]) | ((m["team_a"] == m["team_b"]) & (m["role_a"] < m["role_b"]))].copy()
    order = {r: i for i, r in enumerate(outcome.ROLES)}
    flip = m["role_a"].map(order) > m["role_b"].map(order)
    m.loc[flip, ["role_a", "role_b", "z_a", "z_b"]] = m.loc[flip, ["role_b", "role_a", "z_b", "z_a"]].values
    m["rel"] = np.where(m["team_a"] == m["team_b"], "same", "opp")
    out: dict[str, dict[str, float]] = {"same": {}, "opp": {}}
    for (rel, ra, rb), g in m.groupby(["rel", "role_a", "role_b"]):
        if len(g) < MIN_PAIRS:
            continue
        r = float(np.corrcoef(g["z_a"].astype(float), g["z_b"].astype(float))[0, 1])
        if abs(r) >= MIN_ABS_R:
            out[rel][f"{ra}|{rb}"] = round(r, 3)
    return out


def drift(skill: pd.DataFrame) -> dict:
    f = skill.sort_values(["player_id", "season", "week"]).copy()
    first = f.groupby("player_id")["season"].transform("min")
    earliest = int(f["season"].min())
    f["exp"] = np.where(first > earliest, f["season"] - first, 9)
    g = f.groupby("player_id")
    f["proj4"] = g["proj"].shift(-4)
    ok = (g["season"].shift(-4) == f["season"]) & (g["week"].shift(-4) - f["week"] <= 6)
    f = f[ok & (f["week"] <= 12) & (f["c_long"] >= 2)].copy()
    f["d"] = f["proj4"] - f["proj"]
    vets = f[f["exp"] > 1]
    sd = {}
    for (pos, lvl), gg in vets.groupby(["position", "level"]):
        if len(gg) >= 40:
            sd[f"{pos}|{int(lvl)}"] = round(float(gg["d"].std()), 3)
    for pos, gg in vets.groupby("position"):
        sd[f"{pos}|all"] = round(float(gg["d"].std()), 3)

    def ratio(mask) -> float:
        # Compare like with like: each young player's spread against veterans
        # at the same position and level.
        young = f[mask]
        exp_var = young.apply(lambda r: sd.get(f"{r['position']}|{int(r['level'])}",
                                               sd.get(f"{r['position']}|all")) ** 2, axis=1)
        return round(float(np.sqrt((young["d"] ** 2).mean() / exp_var.mean())), 3)

    vet_mean = float(vets["d"].mean())
    return {
        "sd": sd,
        "youth_multiplier": {"rookie": ratio(f["exp"] == 0), "second": ratio(f["exp"] == 1)},
        "youth_mean": {"rookie": round(float(f[f["exp"] == 0]["d"].mean()) - vet_mean, 3),
                       "second": round(float(f[f["exp"] == 1]["d"].mean()) - vet_mean, 3)},
        "n": int(len(f)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", default="espn", help="profile whose D/ST and K scoring to use")
    ap.add_argument("--out", default=str(outcome.MODEL_PATH))
    args = ap.parse_args()
    cfg = get_config(args.profile)

    skill = skill_frame(cfg)
    units = team_units(cfg)
    shapes = {}
    for (pos, lvl), g in skill.groupby(["position", "level"]):
        if len(g) >= 300:
            shapes[f"{pos}|{int(lvl)}"] = _standardised_shape(g["z"].dropna().to_numpy())
    for pos, g in skill.groupby("position"):
        shapes[f"{pos}|all"] = _standardised_shape(g["z"].dropna().to_numpy())
    for kind, g in units.groupby("position"):
        shapes[kind] = _standardised_shape(g["z"].dropna().to_numpy())

    model = {
        "fitted_on": [int(s) for s in cfg.train_seasons],
        "shape": shapes,
        "correlation": correlations(skill, units),
        "drift": drift(skill),
    }
    Path(args.out).write_text(json.dumps(model, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    for rel in ("same", "opp"):
        top = sorted(model["correlation"][rel].items(), key=lambda kv: -abs(kv[1]))[:8]
        print(f"  {rel}: " + ", ".join(f"{k} {v:+.2f}" for k, v in top))
    print("  drift youth:", model["drift"]["youth_multiplier"], model["drift"]["youth_mean"])


if __name__ == "__main__":
    main()
