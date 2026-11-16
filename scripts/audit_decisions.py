"""Season audit, part 2: the decisions, replayed.

Lineups: the best lineup on each source's projections, scored on what happened,
against the one played and against hindsight. Moves: every add and drop seen
between week-end rosters, what the adds scored while starting, what the drops
did afterwards, and the roster of week 3 left alone as the do-nothing
comparison. Kicker and defence streaming, and the one trade.
Run: PYTHONPATH=src python scripts/audit_decisions.py
"""
import json, subprocess, numpy as np, pandas as pd
from streamer.config import load_config
from streamer.roster.projections import load_history
from streamer.roster.scorecard import graded_log
from streamer.league.model import SLOT_ELIGIBILITY


def week_file(lg, wk):
    return json.loads(subprocess.run(["git","show",f"HEAD:data/leagues/{lg}/week_{wk}.json"],capture_output=True,text=True,check=True).stdout)

def best(players, slots, key):
    """greedy: narrow slots first, then flex; players: list of dicts with pos."""
    order = sorted(slots.items(), key=lambda kv: (len(SLOT_ELIGIBILITY.get(kv[0], (kv[0],))), kv[0]))
    ranked = sorted(players, key=lambda p: -key(p))
    used, out = set(), []
    for slot, n in order:
        for p in ranked:
            if n <= 0: break
            if p["id"] in used or p["pos"] not in SLOT_ELIGIBILITY.get(slot, (slot,)): continue
            used.add(p["id"]); out.append((slot, p)); n -= 1
    return out

for lg in ("espn", "yahoo"):
    cfg = load_config().for_profile(lg)
    hist = load_history(cfg); hist = hist[hist.season == 2026]
    g = graded_log(cfg, hist, include_inactive=True)
    log = pd.read_parquet(f"results/{lg}/skill_log.parquet").dropna(subset=["nfl_id"])
    idmap = dict(zip(log.player_id.astype(str), log.nfl_id))
    full = pd.read_parquet(f"results/{lg}/skill_log.parquet").assign(player_id=lambda x: x.player_id.astype(str))
    lastlog = full.sort_values("logged_at").drop_duplicates(["season","week","player_id"], keep="last")
    lastlog = lastlog[lastlog.season == 2026]
    act_skill = hist.groupby(["player_id","week"])["fantasy_points_ppr"].sum()
    units = pd.read_parquet(f"results/{lg}/history.parquet"); units = units[units.season == 2026]
    upred = pd.read_parquet(f"results/{lg}/predictions.parquet"); upred = upred[upred.season == 2026]
    res = []
    for wk in range(3, int(hist.week.max()) + 1):
        try: raw = week_file(lg, wk)
        except Exception: continue
        slots = {k: v for k, v in raw["slots"].items() if k not in ("BN", "IR", "Bench")}
        me = next(t for t in raw["teams"] if t.get("is_mine"))
        gw = lastlog[lastlog.week == wk].set_index("player_id")
        players = []
        for p in me["roster"]:
            if p.get("slot") == "IR": continue
            pid, pos = p["player_id"], p["position"]
            d = {"id": pid, "name": p["name"], "pos": pos, "slot": p.get("slot")}
            if pos in ("QB","RB","WR","TE"):
                r = gw.loc[pid] if pid in gw.index else None
                if isinstance(r, pd.DataFrame): r = r.iloc[-1]
                bp = pd.to_numeric(r.blended_projection, errors="coerce") if r is not None else np.nan
                if bp != bp and r is not None: bp = pd.to_numeric(r.projection, errors="coerce")
                d["ours"] = float(bp) if bp == bp else 0.0
                for c, k in (("platform_projection","platform"),("vegas_points","props"),("sleeper_projection","sleeper")):
                    v = pd.to_numeric(r[c], errors="coerce") if r is not None else np.nan
                    d[k] = float(v) if v == v and v > 0 else d["ours"]
                nfl = p.get("nfl_id") or idmap.get(str(pid))
                a = act_skill.get((nfl, wk), np.nan) if nfl else np.nan
                d["actual"] = float(a) if a == a else 0.0
                d["known"] = r is not None
            else:
                team = p.get("team")
                u = units[(units.week == wk) & (units.position == pos) & (units.team == team)]
                d["ours"] = float(u.expected_points.iloc[0]) if len(u) else 0.0
                d["platform"] = d["props"] = d["sleeper"] = float(u.baseline_points.iloc[0]) if len(u) else d["ours"]
                d["actual"] = float(u.actual_points.iloc[0]) if len(u) and u.actual_points.notna().iloc[0] else (float(p["actual_points"]) if p.get("actual_points") is not None else 0.0)
            players.append(d)
        row = {"week": wk}
        played = [p for p in players if p["slot"] not in ("BN", "IR", "Bench", None)]
        row["played"] = sum(p["actual"] for p in played)
        for k in ("ours", "platform", "props", "sleeper"):
            row[k] = sum(p["actual"] for _s, p in best(players, slots, lambda p: p[k]))
        row["hindsight"] = sum(p["actual"] for _s, p in best(players, slots, lambda p: p["actual"]))
        ours_ids = {p["id"] for _s, p in best(players, slots, lambda p: p["ours"])}
        row["unknown"] = sum(1 for p in players if p["pos"] in ("QB","RB","WR","TE") and not p.get("known"))
        row["followed"] = ours_ids == {p["id"] for p in played}
        row["diffs"] = "; ".join(f"started {p['name']}" for p in played if p["id"] not in ours_ids) + \
                       ("" if row["followed"] else " | ours: " + ", ".join(p["name"] for _s, p in best(players, slots, lambda p: p["ours"]) if p["id"] not in {q["id"] for q in played}))
        res.append(row)
    t = pd.DataFrame(res)
    print(f"\n===== {lg}: lineups, weeks {t.week.min()}-{t.week.max()} (points scored by the best lineup on each source's projections)")
    print(t[["week","played","ours","platform","props","sleeper","hindsight","followed","unknown"]].round(1).to_string(index=False))
    s = t[["played","ours","platform","props","sleeper","hindsight"]].sum()
    print("totals:", s.round(1).to_dict(), "| weeks matching ours:", int(t.followed.sum()), "of", len(t))
    for _i, r in t[~t.followed].iterrows(): print("   week", r.week, r.diffs)

# ---------------------------------------------------------------- moves, streaming, the trade
import json, subprocess, numpy as np, pandas as pd
from streamer.config import load_config
from streamer.roster.projections import load_history
from streamer.league.model import SLOT_ELIGIBILITY
SK = ("QB","RB","WR","TE")
def week_file(lg, wk):
    return json.loads(subprocess.run(["git","show",f"HEAD:data/leagues/{lg}/week_{wk}.json"],capture_output=True,text=True,check=True).stdout)
def best(players, slots, key):
    order = sorted(slots.items(), key=lambda kv: (len(SLOT_ELIGIBILITY.get(kv[0], (kv[0],))), kv[0]))
    ranked = sorted(players, key=lambda p: -key(p)); used, out = set(), []
    for slot, n in order:
        for p in ranked:
            if n <= 0: break
            if p["id"] in used or p["pos"] not in SLOT_ELIGIBILITY.get(slot, (slot,)): continue
            used.add(p["id"]); out.append(p); n -= 1
    return out
for lg, first in (("espn", 1), ("yahoo", 3)):
    cfg = load_config().for_profile(lg)
    hist = load_history(cfg); hist = hist[hist.season == 2026]; last = int(hist.week.max())
    act = hist.groupby(["player_id","week"])["fantasy_points_ppr"].sum()
    full = pd.read_parquet(f"results/{lg}/skill_log.parquet").assign(player_id=lambda x: x.player_id.astype(str))
    full = full[full.season == 2026]
    idmap = dict(zip(full.dropna(subset=["nfl_id"]).player_id, full.dropna(subset=["nfl_id"]).nfl_id))
    lastlog = full.sort_values("logged_at").drop_duplicates(["week","player_id"], keep="last").set_index(["week","player_id"])
    units = pd.read_parquet(f"results/{lg}/history.parquet"); units = units[units.season == 2026].set_index(["week","position","team"])
    files = {}
    for wk in range(first, last + 1):
        try: files[wk] = week_file(lg, wk)
        except Exception: pass
    def info(p, wk):
        d = {"id": str(p["player_id"]), "name": p["name"], "pos": p["position"], "team": p.get("team"), "slot": p.get("slot")}
        if d["pos"] in SK:
            nfl = p.get("nfl_id") or idmap.get(d["id"]); a = act.get((nfl, wk), np.nan) if nfl else np.nan
            d["actual"] = float(a) if a == a else 0.0; d["nfl"] = nfl
            r = lastlog.loc[(wk, d["id"])] if (wk, d["id"]) in lastlog.index else None
            bp = pd.to_numeric(r.blended_projection, errors="coerce") if r is not None else np.nan
            if bp != bp and r is not None: bp = pd.to_numeric(r.projection, errors="coerce")
            d["ours"] = float(bp) if bp == bp else 0.0
        else:
            k = (wk, d["pos"], d["team"])
            u = units.loc[k] if k in units.index else None
            if isinstance(u, pd.DataFrame): u = u.iloc[0]
            d["ours"] = float(u.expected_points) if u is not None else 0.0
            d["actual"] = float(u.actual_points) if u is not None and u.actual_points == u.actual_points else np.nan
        return d
    mine = {wk: next(t for t in raw["teams"] if t.get("is_mine")) for wk, raw in files.items()}
    print(f"\n================ {lg} ================")
    # --- games: margin each week
    games = {wk: (raw["matchup"] or {}) for wk, raw in files.items()}
    print("games (your points, opponent's):", {wk: (round(m.get("my_points") or 0, 1), round(m.get("opp_points") or 0, 1)) for wk, m in games.items()})
    # --- moves ledger
    wks = sorted(files)
    rows = []
    for a, b in zip(wks, wks[1:]):
        ra = {str(p["player_id"]): p for p in mine[a]["roster"]}; rb = {str(p["player_id"]): p for p in mine[b]["roster"]}
        for pid in rb.keys() - ra.keys():
            p = rb[pid]; held = [w for w in wks if w >= b and pid in {str(q["player_id"]) for q in mine[w]["roster"]}]
            started = [w for w in held if next(q for q in mine[w]["roster"] if str(q["player_id"]) == pid).get("slot") not in ("BN","IR")]
            pts_started = sum((info(p, w)["actual"] or 0) for w in started if info(p, w)["actual"] == info(p, w)["actual"])
            rows.append({"week": b, "move": "add", "name": p["name"], "pos": p["position"], "held": len(held), "starts": len(started), "pts_started": pts_started})
        for pid in ra.keys() - rb.keys():
            p = ra[pid]
            if p["position"] in SK:
                after = [info(p, w)["actual"] for w in range(b, last + 1)]
                rows.append({"week": b, "move": "drop", "name": p["name"], "pos": p["position"], "pts_after": sum(after), "games_after": sum(1 for x in after if x > 0),
                             "best_after": max(after) if after else 0})
            else:
                rows.append({"week": b, "move": "drop", "name": p["name"], "pos": p["position"]})
    m = pd.DataFrame(rows)
    adds, drops = m[m.move == "add"], m[m.move == "drop"]
    sk_adds = adds[adds.pos.isin(SK)]
    print(f"moves seen between week-end rosters (weeks {wks[0]}-{wks[-1]}): {len(adds)} adds ({len(sk_adds)} skill, {len(adds)-len(sk_adds)} K/DST), {len(drops)} drops")
    print("skill adds: started at least once:", int((sk_adds.starts > 0).sum()), "of", len(sk_adds), "| total points while starting:", round(sk_adds.pts_started.sum(), 1),
          "| per start:", round(sk_adds.pts_started.sum() / max(sk_adds.starts.sum(), 1), 1))
    print("  adds that started:"); print(sk_adds[sk_adds.starts > 0].sort_values("pts_started", ascending=False)[["week","name","pos","held","starts","pts_started"]].round(1).to_string(index=False))
    sd = drops[drops.pos.isin(SK)].sort_values("pts_after", ascending=False)
    print("  skill drops, what they did afterwards (through week %d):" % last); print(sd[["week","name","pos","games_after","pts_after","best_after"]].head(12).round(1).to_string(index=False))
    print("  drops who then averaged 10+ a game over 2+ games:", sd[(sd.games_after >= 2) & (sd.pts_after / sd.games_after.clip(lower=1) >= 10)].name.tolist())
    # --- frozen roster: what the earliest logged roster would have scored, lineup by our numbers
    base_wk = max(first, 3)
    frozen = [p for p in mine[base_wk]["roster"]]
    comp = []
    for wk in range(base_wk + 1, last + 1):
        if wk not in files: continue
        slots = {k: v for k, v in files[wk]["slots"].items() if k not in ("BN","IR","Bench")}
        now = [info(p, wk) for p in mine[wk]["roster"] if p.get("slot") != "IR"]
        fr = [info(p, wk) for p in frozen]
        if any(p["actual"] != p["actual"] for p in now + fr):       # a unit with no final score yet
            skill_only = True
            slots = {k: v for k, v in slots.items() if k not in ("K","DST")}
            now = [p for p in now if p["pos"] in SK]; fr = [p for p in fr if p["pos"] in SK]
        else: skill_only = False
        played = sum(p["actual"] for p in now if p["slot"] not in ("BN","IR") and (not skill_only or p["pos"] in SK))
        f = sum(p["actual"] for p in best(fr, slots, lambda p: p["ours"]))
        g = games[wk]; my, opp = g.get("my_points"), g.get("opp_points")
        comp.append({"week": wk, "played": played, "frozen": f, "gain": played - f, "skill_only": skill_only, "my": my, "opp": opp,
                     "won": None if my is None or opp is None else my > opp,
                     "frozen_wins": None if my is None or opp is None else (my - played + f) > opp})
    c = pd.DataFrame(comp)
    print(f"your moves since the week-{base_wk} roster: what you scored against that roster left alone (lineup by our numbers)")
    print(c.round(1).to_string(index=False))
    print("  total gain %.1f over %d weeks (%.1f a week); record %d-%d, frozen roster %d-%d" % (
        c.gain.sum(), len(c), c.gain.mean(), int(c.won.sum()), int((~c.won.astype(bool)).sum()), int(c.frozen_wins.sum()), int((~c.frozen_wins.astype(bool)).sum())))
    # --- kicker and defence streaming
    u = units.reset_index()
    for pos in ("K","DST"):
        got = []
        for wk in wks:
            for p in mine[wk]["roster"]:
                if p["position"] == pos and p.get("slot") not in ("BN","IR"):
                    d = info(p, wk)
                    if d["actual"] == d["actual"]:
                        w = u[(u.week == wk) & (u.position == pos)]
                        got.append({"week": wk, "mine": d["actual"], "all": w.actual_points.mean(), "top12_platform": w.nlargest(12, "baseline_points").actual_points.mean(),
                                    "top12_ours": w.nlargest(12, "expected_points").actual_points.mean(), "best3_ours": w.nlargest(3, "expected_points").actual_points.mean()})
        s = pd.DataFrame(got)
        if len(s): print(f"{pos} you started, {len(s)} weeks: you {s.mine.mean():.1f} a week; all units {s['all'].mean():.1f}; platform's top 12 {s.top12_platform.mean():.1f}; our top 12 {s.top12_ours.mean():.1f}; our top 3 {s.best3_ours.mean():.1f}")
if True:
    # --- the ESPN trade (Oct 31): Hockenson to Kevin, Rachaad White to you
    cfg = load_config().for_profile("espn"); hist = load_history(cfg); hist = hist[hist.season == 2026]
    for name in ("T.J. Hockenson", "Rachaad White"):
        h = hist[hist.player_display_name == name].sort_values("week")
        print(name, "by week:", {int(w): round(float(p), 1) for w, p in zip(h.week, h.fantasy_points_ppr)})
