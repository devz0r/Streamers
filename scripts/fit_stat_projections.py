"""Projecting each stat (catches, yards, touchdowns) against projecting points.

    python scripts/fit_stat_projections.py

The skill projection's core is volume x efficiency on fantasy points
(``projections._volume_efficiency``): volume an exponentially weighted mean of
nflverse expected points anchored to the ~17-game volume, efficiency points
per expected point shrunk toward the position. nflverse's ``ff_opportunity``
carries the same expectation for every stat, so the same core can run per
stat -- receptions, receiving / rushing / passing yards, touchdowns of each
kind, interceptions, two-point conversions, fumbles -- and the stats be summed
on each league's sheet.

Walk-forward over 2022-2025 (each week sees only earlier games; position rates
from earlier seasons), on the same player-weeks for both (a projection of 5+
points, three games of history), REG season. Per-stat efficiency priors are a
multiple of production's (150 expected points, in that stat's units, the same
number of games' worth), fitted on three seasons and scored on the fourth.
DECISIONS.md, "Projecting each stat".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SEASONS = range(2018, 2026)
TEST = (2022, 2023, 2024, 2025)
POS = ("QB", "RB", "WR", "TE")
# stat -> (actual column, expected column, sheet key)
STATS = {
    "pass_yards": ("pass_yards_gained", "pass_yards_gained_exp", "pass_yards"),
    "pass_td": ("pass_touchdown", "pass_touchdown_exp", "pass_td"),
    "interception": ("pass_interception", "pass_interception_exp", "interception"),
    "rush_yards": ("rush_yards_gained", "rush_yards_gained_exp", "rush_yards"),
    "rush_td": ("rush_touchdown", "rush_touchdown_exp", "rush_td"),
    "reception": ("receptions", "receptions_exp", "reception"),
    "rec_yards": ("rec_yards_gained", "rec_yards_gained_exp", "reception_yards"),
    "rec_td": ("rec_touchdown", "rec_touchdown_exp", "reception_td"),
    "two_point": ("two_pt", "two_pt_exp", "two_point"),
}
MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)
PROD = {"k": 3.0, "k_vol": 2.0, "m_eff": 150.0, "m_rec": 60.0, "w_rec": 0.25, "half": 2.5, "short": 4, "long": 17}


def load() -> pd.DataFrame:
    frames = []
    for s in SEASONS:
        p = Path(f"data/raw/ff_opportunity_{s}.parquet")
        if p.exists():
            frames.append(pd.read_parquet(p))
    d = pd.concat(frames, ignore_index=True)
    d["season"] = pd.to_numeric(d["season"]).astype(int)
    d["week"] = pd.to_numeric(d["week"]).astype(int)
    d = d[d["position"].isin(POS) & (d["week"] <= 18)].copy()
    d["two_pt"] = d[["pass_two_point_conv", "rec_two_point_conv", "rush_two_point_conv"]].fillna(0).sum(axis=1)
    d["two_pt_exp"] = d[["pass_two_point_conv_exp", "rec_two_point_conv_exp", "rush_two_point_conv_exp"]].fillna(0).sum(axis=1)
    d["fumbles"] = d[["rec_fumble_lost", "rush_fumble_lost"]].fillna(0).sum(axis=1)
    for a, e, _k in STATS.values():
        d[a], d[e] = d[a].fillna(0.0).astype(float), d[e].fillna(0.0).astype(float)
    return d.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def trailing(d: pd.DataFrame, a: str, e: str, p=PROD) -> pd.DataFrame:
    """The production core's leak-free windows for one (actual, expected) pair."""
    pid = d["player_id"]
    pa, pe = d.groupby(pid, sort=False)[a].shift(1), d.groupby(pid, sort=False)[e].shift(1)

    def roll(x, n, how):
        return getattr(x.groupby(pid, sort=False).rolling(n, min_periods=1), how)().reset_index(level=0, drop=True).sort_index()

    alpha = 1.0 - 0.5 ** (1.0 / p["half"])
    out = pd.DataFrame({
        "a_short": roll(pa, p["short"], "sum"), "e_short": roll(pe, p["short"], "sum"),
        "c_short": roll(pa, p["short"], "count"), "a_long": roll(pa, p["long"], "sum"),
        "e_long": roll(pe, p["long"], "sum"), "c_long": roll(pa, p["long"], "count"),
        "v_recent": pe.groupby(pid, sort=False).ewm(alpha=alpha, adjust=True).mean().reset_index(level=0, drop=True).sort_index(),
    })
    return out


def core(t: pd.DataFrame, pos_mean, pos_eff, m_eff, m_rec, p=PROD) -> pd.Series:
    """``projections._volume_efficiency`` x, verbatim."""
    c_long, e_long = t["c_long"].fillna(0), t["e_long"].fillna(0)
    vol_long = (e_long + p["k"] * pos_mean) / (c_long + p["k"])
    c_short = t["c_short"].fillna(0)
    vol = (c_short * t["v_recent"] + p["k_vol"] * vol_long) / (c_short + p["k_vol"])
    vol = vol.where(t["v_recent"].notna(), vol_long)
    eff_long = (t["a_long"].fillna(0) + m_eff * pos_eff) / (e_long + m_eff)
    eff_rec = (t["a_short"].fillna(0) + m_rec * eff_long) / (t["e_short"].fillna(0) + m_rec)
    eff = (1 - p["w_rec"]) * eff_long + p["w_rec"] * eff_rec
    return (vol * eff).where(c_long > 0)


def rates(d: pd.DataFrame, season: int, a: str, e: str):
    """Position mean expected a game and actual per expected, from seasons before ``season``."""
    h = d[d["season"] < season]
    g = h.groupby("position")
    return g[e].mean(), g[a].sum() / g[e].sum().where(g[e].sum() > 0)


def main() -> None:
    from streamer.config import get_config

    d = load()
    sheets = {lg: get_config().for_profile(lg)._profile["offense_scoring"] for lg in ("espn", "yahoo")}
    out = {}
    for lg, sc in sheets.items():
        d[f"pts_{lg}"] = sum(float(sc[k]) * d[a] for a, _e, k in STATS.values()) + float(sc["fumble_lost"]) * d["fumbles"]
        d[f"exp_{lg}"] = sum(float(sc[k]) * d[e] for _a, e, k in STATS.values())
    # Points (production core) per league, and every stat once (stats do not depend on the sheet).
    tr_pts = {lg: trailing(d, f"pts_{lg}", f"exp_{lg}") for lg in sheets}
    tr_stat = {s: trailing(d, a, e) for s, (a, e, _k) in STATS.items()}
    fum = d.groupby("player_id", sort=False)["fumbles"].shift(1).groupby(d["player_id"], sort=False).rolling(17, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
    pos_fum = d.groupby("position")["fumbles"].mean()
    for season in TEST:
        here = d["season"] == season
        res = {}
        # points
        for lg in sheets:
            pm, pe = rates(d, season, f"pts_{lg}", f"exp_{lg}")
            res[f"pts_{lg}"] = core(tr_pts[lg][here], d.loc[here, "position"].map(pm), d.loc[here, "position"].map(pe),
                                    PROD["m_eff"], PROD["m_rec"])
            # production's 150 points of prior is this many games' worth of expected points at the position
            res[f"games_{lg}"] = PROD["m_eff"] / d.loc[here, "position"].map(pm)
        # stats, at each multiple of production's prior (in games)
        for s, (a, e, _k) in STATS.items():
            pm, pe = rates(d, season, a, e)
            g = res["games_espn"]
            for m in MULTS:
                res[(s, m)] = core(tr_stat[s][here], d.loc[here, "position"].map(pm), d.loc[here, "position"].map(pe).fillna(1.0),
                                   m * g * d.loc[here, "position"].map(pm), m * g * d.loc[here, "position"].map(pm) * PROD["m_rec"] / PROD["m_eff"])
        n = tr_stat["reception"].loc[here, "c_long"].fillna(0)
        res["fum"] = ((fum[here].fillna(0) * n + 3 * d.loc[here, "position"].map(pos_fum)) / (n + 3))
        out[season] = (pd.DataFrame({k if isinstance(k, str) else f"{k[0]}|{k[1]}": v for k, v in res.items()}), here)
    report(d, out, sheets)


def report(d, out, sheets) -> None:
    def rmse(p, y):
        return float(np.sqrt(np.mean((p - y) ** 2)))

    print("Per stat, held-out RMSE by season (prior multiple fitted on the other three):")
    chosen = {}
    for s, (a, _e, _k) in STATS.items():
        errs = {}
        for season, (r, here) in out.items():
            y = d.loc[here, a]
            mask = r["pts_espn"].fillna(0) >= 5
            errs[season] = {m: (r[f"{s}|{m}"][mask], y[mask]) for m in MULTS}
        picks = {}
        for season in TEST:
            train = [x for x in TEST if x != season]
            def tot(m, train=train, errs=errs):
                return sum(((errs[t][m][0] - errs[t][m][1]) ** 2).sum() for t in train)
            picks[season] = min(MULTS, key=tot)
        chosen[s] = picks
        line = "  ".join(f"{t}: {rmse(*errs[t][picks[t]]):.3f} (x{picks[t]}; prod x1 {rmse(*errs[t][1.0]):.3f})" for t in TEST)
        print(f"  {s:12s} {line}")
    for lg, sc in sheets.items():
        print(f"\n{lg}: fantasy points, held-out, same player-weeks (production core on points vs per-stat summed)")
        tot_p, tot_s, n_tot = 0.0, 0.0, 0
        for season, (r, here) in out.items():
            y = d.loc[here, f"pts_{lg}"]
            mask = r["pts_espn"].fillna(0) >= 5
            stat_sum = sum(float(sc[k]) * r[f"{s}|{chosen[s][season]}"] for s, (_a, _e, k) in STATS.items()) + float(sc["fumble_lost"]) * r["fum"]
            stat_x1 = sum(float(sc[k]) * r[f"{s}|1.0"] for s, (_a, _e, k) in STATS.items()) + float(sc["fumble_lost"]) * r["fum"]
            p, s_, s1, yy = r[f"pts_{lg}"][mask], stat_sum[mask], stat_x1[mask], y[mask]
            ok = p.notna() & s_.notna()
            p, s_, s1, yy = p[ok], s_[ok], s1[ok], yy[ok]
            print(f"  {season}: n {len(yy)}  points RMSE {rmse(p, yy):.3f} MAE {np.mean(np.abs(p - yy)):.3f} | per-stat RMSE "
                  f"{rmse(s_, yy):.3f} MAE {np.mean(np.abs(s_ - yy)):.3f} | per-stat at production's prior {rmse(s1, yy):.3f}")
            tot_p += ((p - yy) ** 2).sum()
            tot_s += ((s_ - yy) ** 2).sum()
            n_tot += len(yy)
        print(f"  all: points {np.sqrt(tot_p / n_tot):.3f}  per-stat {np.sqrt(tot_s / n_tot):.3f}")


if __name__ == "__main__":
    main()
