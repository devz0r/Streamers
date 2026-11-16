"""Season audit, part 1: how accurate the inputs to P(title) were this season.

Weekly projections by position against the platform, props and Sleeper; bias
by week; injury tags; the starters on your own rosters; D/ST and kickers; and
the rest-of-season values against the consensus (from the scorecard the
weekly job writes). Run: PYTHONPATH=src python scripts/audit_inputs.py
"""
import json, subprocess, numpy as np, pandas as pd
from streamer.config import load_config
from streamer.roster.projections import load_history
from streamer.roster.scorecard import graded_log

def mae(a,b): return float(np.abs(a-b).mean())
for lg in ("espn","yahoo"):
    cfg = load_config().for_profile(lg)
    hist = load_history(cfg); hist = hist[hist.season == 2026]
    last = int(hist.week.max())
    g = graded_log(cfg, hist, include_inactive=True)
    g = g[g.week <= last]
    print(f"\n===== {lg}: weeks {g.week.min()}-{g.week.max()} graded, {len(g)} player-weeks (history through week {last})")
    # A. by position, the players who mattered: top-N by our number each week
    starters = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}
    rk = g.groupby(["week","position"])["blended_projection"].rank(ascending=False, method="first")
    s = g[rk <= g.position.map(starters).fillna(0)]
    rows = []
    for pos, w in s.groupby("position"):
        r = {"pos": pos, "n": len(w), "ours_mae": mae(w.blended_projection, w.actual),
             "bias": float((w.blended_projection - w.actual).mean())}
        for col, lab in (("platform_projection","platform"), ("vegas_points","props"), ("sleeper_projection","sleeper"), ("model_projection","model")):
            c = w[w[col].notna() & (w[col] > 0)]
            if len(c) > 30:
                r[lab+"_mae"] = mae(c[col], c.actual); r["ours_same"+lab] = mae(c.blended_projection, c.actual); r[lab+"_n"] = len(c)
        rows.append(r)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    # bias by week (all likely starters)
    print("bias by week (ours - actual, likely starters):", {int(k): round(float((w.blended_projection - w.actual).mean()), 2) for k, w in s.groupby("week")})
    # B. injury tags: what happened to tagged players
    t = g.copy(); t["zero"] = t.actual <= 0
    tag = t.groupby(t.status.replace("", "none")).agg(n=("zero","size"), played=("played","mean"), proj=("blended_projection","mean"), act=("actual","mean"))
    print("by injury tag (played = had a box score):"); print(tag.round(2).to_string())
    # C. my rosters: started players, final lineup of each week's snapshot in git
    mine_rows = []
    for wk in sorted(g.week.unique()):
        path = f"data/leagues/{lg}/week_{wk}.json"
        try: raw = json.loads(subprocess.run(["git","show",f"HEAD:{path}"],capture_output=True,text=True,check=True).stdout)
        except Exception: continue
        me = next(t for t in raw["teams"] if t.get("is_mine"))
        for p in me["roster"]:
            mine_rows.append({"week": wk, "player_id": p["player_id"], "slot": p.get("slot"), "name": p["name"]})
    m = pd.DataFrame(mine_rows)
    if not m.empty:
        j = m.merge(g, on=["week","player_id"], how="inner", suffixes=("","_l"))
        st = j[~j.slot.isin(["BN","IR","Bench","bench"])]
        print(f"your starters graded: {len(st)}; ours total {st.blended_projection.sum():.0f} vs scored {st.actual.sum():.0f} "
              f"(bias {(st.blended_projection-st.actual).mean():+.2f}/player-week); platform {st.platform_projection.sum():.0f}")
        st = st.assign(err=st.actual - st.blended_projection)
        print("biggest misses on your starters:"); print(st.reindex(st.err.abs().sort_values(ascending=False).index)[["week","name","position","status","blended_projection","platform_projection","actual","err"]].head(10).round(1).to_string(index=False))
        by = st.groupby("name").agg(wks=("week","size"), proj=("blended_projection","sum"), act=("actual","sum"))
        by["gap"] = by.act - by.proj
        print("your players, season so far (started weeks):"); print(by.sort_values("gap").round(1).to_string())
    # D. D/ST and K
    h = pd.read_parquet(f"results/{lg}/history.parquet"); h = h[(h.season==2026) & h.actual_points.notna()]
    for pos, w in h.groupby("position"):
        top = w[w.model_rank <= 12]
        print(f"{pos}: {len(w)} unit-weeks, ours MAE {mae(w.expected_points,w.actual_points):.2f} bias {(w.expected_points-w.actual_points).mean():+.2f}; "
              f"baseline MAE {mae(w.baseline_points,w.actual_points):.2f}; top-12 by us: ours bias {(top.expected_points-top.actual_points).mean():+.2f}, "
              f"rank corr ours {w.expected_points.corr(w.actual_points, method='spearman'):.3f} vs baseline {w.baseline_points.corr(w.actual_points, method='spearman'):.3f}")
    sc = json.load(open(f"results/{lg}/scorecard.json"))
    print("scorecard (runner, scope=played):"); [print("  ", x["source"], "pairs", round(x["pairs"],3) if x["pairs"] else None, "miss", round(x["miss"],2) if x["miss"] else None, "| ours on same", round(x["ours_pairs"],3), round(x["ours_miss"],2)) for x in sc["grades"] if x["scope"]=="played"]
    print("ROS benchmark:", sc.get("ros"))
