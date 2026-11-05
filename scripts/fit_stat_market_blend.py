# ruff: noqa: E702, E731, B023, E501
"""Per-stat market blend against one weight, 2026 weeks held out (DECISIONS.md, "Projecting each stat").

    python scripts/fit_stat_market_blend.py [anytime-TD price over fair, default 1.0]
"""
import json
import sys

sys.path.insert(0, "scripts")
import fit_stat_projections as F
import numpy as np
import pandas as pd

from streamer.config import get_config

F.SEASONS = range(2018, 2027)
d = F.load()
season = 2026
here = d["season"] == season
MULT = {"pass_yards": 1, "pass_td": 4, "interception": 4, "rush_yards": 1, "rush_td": 4, "reception": 1, "rec_yards": 1, "rec_td": 4, "two_point": 4}
proj = {}
sc0 = get_config().for_profile("espn")._profile["offense_scoring"]
d["pts_espn"] = sum(float(sc0[k]) * d[a] for a, _e, k in F.STATS.values()) + float(sc0["fumble_lost"]) * d["fumbles"]
d["exp_espn"] = sum(float(sc0[k]) * d[e] for _a, e, k in F.STATS.values())
pm_pts, _ = F.rates(d, season, "pts_espn", "exp_espn")
games = F.PROD["m_eff"] / d.loc[here, "position"].map(pm_pts)
for s, (a, e, _k) in F.STATS.items():
    t = F.trailing(d, a, e)
    pm, pe = F.rates(d, season, a, e)
    m = MULT[s] * games * d.loc[here, "position"].map(pm)
    proj[s] = F.core(t[here], d.loc[here, "position"].map(pm), d.loc[here, "position"].map(pe).fillna(1.0), m, m * 0.4)
P = pd.DataFrame(proj); P["player_id"] = d.loc[here, "player_id"].values; P["week"] = d.loc[here, "week"].values
for s, (a, _e, _k) in F.STATS.items():
    P["act_" + s] = d.loc[here, a].values
P["fumbles"] = d.loc[here, "fumbles"].values
# market per-stat means from the projection log
VIG = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
MK = {"passing_yards": "pass_yards", "passing_tds": "pass_td", "passing_interceptions": "interception",
      "rushing_yards": "rush_yards", "receptions": "reception", "receiving_yards": "rec_yards"}
for lg in ("espn", "yahoo"):
    sc = get_config().for_profile(lg)._profile["offense_scoring"]
    s = pd.read_parquet(f"results/{lg}/skill_log.parquet")
    s = s.sort_values("logged_at").groupby(["season", "week", "player_id"]).tail(1)
    s = s[(s.season == season) & (s.vegas_version.fillna(0) >= 2) & (s.vegas_stats.fillna("") != "")]
    rows = []
    for r in s.itertuples():
        st = json.loads(r.vegas_stats); filled = set((r.vegas_filled or "").split(","))
        row = {"player_id": r.nfl_id, "week": r.week, "pos": r.position, "mkt_pts": r.vegas_points, "prod": r.projection}
        for k, v in MK.items():
            if k in st and k not in filled:
                row["m_" + v] = float(st[k])
        if "anytime_td" in st and "anytime_td" not in filled:
            row["m_td"] = -np.log(max(1 - float(st["anytime_td"]) / VIG, 1e-3))      # expected TDs (Poisson)
        rows.append(row)
    M = pd.DataFrame(rows)
    X = P.merge(M, on=["player_id", "week"])
    X["ours_td"] = X.rush_td + X.rec_td; X["act_td"] = X.act_rush_td + X.act_rec_td
    def pts(get):
        return (sc["pass_yards"] * get("pass_yards") + sc["pass_td"] * get("pass_td") + sc["interception"] * get("interception")
                + sc["rush_yards"] * get("rush_yards") + sc["reception"] * get("reception") + sc["reception_yards"] * get("rec_yards")
                + 6.0 * get("td") + sc["two_point"] * get("two_point") + sc["fumble_lost"] * X.fumbles.mean() * 0 )
    ours = lambda k: X["ours_td"] if k == "td" else X[k]
    act = lambda k: X["act_td"] if k == "td" else X["act_" + k]
    X["ours_pts"] = pts(ours); X["act_pts"] = pts(act) + sc["fumble_lost"] * X.fumbles
    ratio = (X["prod"] / X["ours_pts"]).clip(0.3, 3.0).where(X["ours_pts"] > 1, 1.0)
    for k in ["pass_yards", "pass_td", "interception", "rush_yards", "reception", "rec_yards", "two_point", "ours_td"]:
        X[k] = X[k] * ratio
    X["ours_pts"] = pts(ours)
    X = X[X.ours_pts >= 5]
    stats = ["pass_yards", "pass_td", "interception", "rush_yards", "reception", "rec_yards", "td"]
    weeks = sorted(X.week.unique())
    grid = np.round(np.arange(0, 1.01, 0.05), 2)
    rmse = lambda p, y: float(np.sqrt(np.mean((p - y) ** 2)))
    resA, resB, res0 = [], [], []
    for wk in weeks:
        tr, te = X[X.week != wk], X[X.week == wk]
        # A: one weight on fantasy points (ours core vs the market's fantasy points)
        wA = min(grid, key=lambda w: (((1 - w) * tr.ours_pts + w * tr.mkt_pts - tr.act_pts) ** 2).mean())
        pA = (1 - wA) * te.ours_pts + wA * te.mkt_pts
        # B: a weight per stat, where the market priced it
        ws = {}
        for k in stats:
            mcol = "m_" + k
            g = tr[tr[mcol].notna()] if mcol in tr else tr.iloc[:0]
            o = g["ours_td"] if k == "td" else g[k]
            y = g["act_td"] if k == "td" else g["act_" + k]
            ws[k] = min(grid, key=lambda w: (((1 - w) * o + w * g[mcol] - y) ** 2).mean()) if len(g) > 30 else 0.0
        def blended(k, te=te, ws=ws):
            o = te["ours_td"] if k == "td" else te[k]
            m = te["m_" + k] if ("m_" + k) in te else pd.Series(np.nan, index=te.index)
            return ((1 - ws[k]) * o + ws[k] * m).where(m.notna(), o)
        pB = (sc["pass_yards"] * blended("pass_yards") + sc["pass_td"] * blended("pass_td") + sc["interception"] * blended("interception")
              + sc["rush_yards"] * blended("rush_yards") + sc["reception"] * blended("reception") + sc["reception_yards"] * blended("rec_yards")
              + 6.0 * blended("td") + sc["two_point"] * te.two_point)
        resA.append((wk, rmse(pA, te.act_pts), wA)); resB.append((wk, rmse(pB, te.act_pts), ws)); res0.append((wk, rmse(te.ours_pts, te.act_pts)))
    print(lg, "n", len(X), "weeks", weeks)
    for (wk, a, wA), (_, b, ws), (_, o) in zip(resA, resB, res0):
        print(f"  wk{wk}: ours alone {o:.3f} | one weight {a:.3f} (w={wA}) | per stat {b:.3f}  w={ {k: v for k, v in ws.items()} }")
    nA = sum(a for _, a, _ in resA) / len(resA); nB = sum(b for _, b, _ in resB) / len(resB)
    print(f"  mean: one weight {nA:.3f}  per stat {nB:.3f}")
