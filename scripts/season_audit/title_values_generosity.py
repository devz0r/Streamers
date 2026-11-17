"""Season audit, part 3: do the production season values (consensus blended in) rate your roster more generously
than other rosters, against what the same players have scored this season?
Run: PYTHONPATH=src python scripts/season_audit/title_values_generosity.py
"""
import json, glob, numpy as np, pandas as pd
from streamer.config import load_config
from streamer.roster.projections import load_history
from streamer.league.model import SLOT_ELIGIBILITY
OUTS = {"OUT", "INJURY_RESERVE", "IR", "O", "IR-R", "PUP", "SUSPENSION", "NA"}
def lineup(players, slots, key):
    order = sorted(slots.items(), key=lambda kv: (len(SLOT_ELIGIBILITY.get(kv[0], (kv[0],))), kv[0]))
    ranked = sorted(players, key=lambda p: -key(p)); used, tot = set(), 0.0
    for slot, n in order:
        for p in ranked:
            if n <= 0: break
            if p["id"] in used or p["pos"] not in SLOT_ELIGIBILITY.get(slot, (slot,)): continue
            used.add(p["id"]); tot += key(p); n -= 1
    return tot
rng = np.random.default_rng(3)
for lg in ("espn", "yahoo"):
    cfg = load_config().for_profile(lg)
    hist = load_history(cfg); hist = hist[hist.season == 2026]
    ppg = hist.groupby("player_id").fantasy_points_ppr.agg(["mean", "count"])
    v = pd.read_parquet(f"results/{lg}/value_history.parquet"); v["player_id"] = v.player_id.astype(str)
    wk = int(v.week.max()); vals = v[v.week == wk].drop_duplicates("player_id", keep="last").set_index("player_id").ros_value
    log = pd.read_parquet(f"results/{lg}/skill_log.parquet").dropna(subset=["nfl_id"])
    idmap = dict(zip(log.player_id.astype(str), log.nfl_id))
    f = sorted(glob.glob(f"data/leagues/{lg}/week_*.json"), key=lambda x: int(x.rsplit("_", 1)[1][:-5]))[-1]
    snap = json.load(open(f)); slots = {k: c for k, c in snap["slots"].items() if k in ("QB", "RB", "WR", "TE", "FLEX", "WR/RB", "WR/TE", "SUPERFLEX")}
    rows, players = [], []
    for t in snap["teams"]:
        ps = []
        for p in t["roster"]:
            pid = str(p["player_id"]); nfl = idmap.get(pid)
            if p["position"] not in ("QB", "RB", "WR", "TE") or p.get("slot") == "IR" or pid not in vals.index: continue
            hurt = (p.get("status") or "").upper() in OUTS
            r = ppg.loc[nfl] if nfl in ppg.index else None
            d = {"id": pid, "pos": p["position"], "value": 0.0 if hurt else float(vals[pid]),
                 "ppg": 0.0 if hurt else (float(r["mean"]) if r is not None and r["count"] >= 3 else float(vals[pid])),
                 "has_ppg": r is not None and r["count"] >= 3 and not hurt}
            ps.append(d)
            if d["has_ppg"] and d["value"] >= 6: players.append({"mine": bool(t.get("is_mine")), "team": t["name"], "gap": d["value"] - d["ppg"], "id": pid})
        rows.append({"team": t["name"][:22], "mine": bool(t.get("is_mine")), "by_value": lineup(ps, slots, lambda p: p["value"]),
                     "by_ppg": lineup(ps, slots, lambda p: p["ppg"])})
    d = pd.DataFrame(rows); d["gap"] = d.by_value - d.by_ppg
    print(f"\n================ {lg}: production season values (week {wk} log), current rosters, healthy players")
    print(d.sort_values("gap").round(1).to_string(index=False))
    me = d[d.mine].iloc[0]; others = d[~d.mine]
    z = (me.gap - others.gap.mean()) / others.gap.std(ddof=1)
    print(f"  your lineup: values minus 2026 points a game {me.gap:+.1f}, the others {others.gap.mean():+.1f} (sd {others.gap.std(ddof=1):.1f}); "
          f"yours is {me.gap - others.gap.mean():+.1f} a week more generous, {z:+.1f} sd, rank {int((d.gap > me.gap).sum()) + 1} of {len(d)} most generous")
    pl = pd.DataFrame(players)
    print(f"  player level (worth 6+, 3+ games): yours {pl[pl.mine].gap.mean():+.2f} a game (n={pl.mine.sum()}), others {pl[~pl.mine].gap.mean():+.2f} (n={(~pl.mine).sum()})")
