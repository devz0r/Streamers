"""Season audit, part 3: the optimizer's curse on this season's real leagues.
Run: PYTHONPATH=src python scripts/season_audit/title_curse.py
(1) season values: realized points a game after each checkpoint minus the
season value at it, by where the player was (your roster, another roster, the wire);
(2) weekly projections of every team's starters, by team."""
import json, subprocess, numpy as np, pandas as pd
from streamer.config import load_config
from streamer.roster.projections import load_history

SK = ("QB", "RB", "WR", "TE")
rng = np.random.default_rng(7)

def week_file(lg, wk):
    try:
        return json.loads(subprocess.run(["git", "show", f"HEAD:data/leagues/{lg}/week_{wk}.json"],
                                         capture_output=True, text=True, check=True).stdout)
    except Exception:
        return None

def boot_mean(values, groups, n=2000):
    """Mean and a 95% interval, resampling players (a player's checkpoints move together)."""
    df = pd.DataFrame({"v": values, "g": groups})
    per = df.groupby("g").v.agg(["sum", "count"])
    s, c = per["sum"].to_numpy(), per["count"].to_numpy()
    idx = rng.integers(0, len(per), size=(n, len(per)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

for lg in ("espn", "yahoo"):
    cfg = load_config().for_profile(lg)
    hist = load_history(cfg); hist = hist[hist.season == 2026]
    games_per_week = hist.groupby("week").team.nunique()
    last = int(games_per_week[games_per_week >= 26].index.max())        # weeks with (nearly) every team's box scores
    log = pd.read_parquet(f"results/{lg}/skill_log.parquet").assign(player_id=lambda x: x.player_id.astype(str))
    log = log[log.season == 2026]
    lastlog = log.sort_values("logged_at").drop_duplicates(["week", "player_id"], keep="last")
    files = {wk: week_file(lg, wk) for wk in range(1, last + 2)}
    print(f"\n================ {lg} (box scores complete through week {last})")

    # ---- (1) season values at checkpoints 3..last-2, realized after
    rows = []
    for w in range(3, last - 1):
        f = files.get(w)
        if f is None: continue
        where = {}
        for t in f["teams"]:
            for p in t["roster"]:
                where[str(p["player_id"])] = "yours" if t.get("is_mine") else "other team"
        snap = lastlog[(lastlog.week == w) & lastlog.position.isin(SK) & lastlog.ros_value.notna() & lastlog.nfl_id.notna()]
        later = hist[(hist.week > w) & (hist.week <= last)]
        ppg = later.groupby("player_id").fantasy_points_ppr.agg(["mean", "count"])
        j = snap.merge(ppg, left_on="nfl_id", right_index=True)
        j = j[(j["count"] >= 2) & (j.ros_value >= 4)]
        j = j.assign(group=j.player_id.map(where).fillna("free agent"), w=w)
        rows.append(j[["w", "player_id", "name", "position", "group", "ros_value", "mean", "count"]])
    d = pd.concat(rows)
    d["gap"] = d["mean"] - d["ros_value"]
    print("season value vs points a game since (players worth 4+, 2+ games after; checkpoints weeks 3-%d)" % (last - 2))
    for gname in ("yours", "other team", "free agent"):
        x = d[d.group == gname]
        m, lo, hi = boot_mean(x.gap.to_numpy(), x.player_id.to_numpy())
        print(f"  {gname:11s} {len(x):5d} player-checkpoints, {x.player_id.nunique():4d} players: realized minus value {m:+.2f} a game (95% {lo:+.2f} to {hi:+.2f})")
    # starters-level: the players a team would start (top of the roster by value), yours against others
    top = d[d.group != "free agent"].copy()
    top["rk"] = top.groupby(["w", "group"]).ros_value.rank(ascending=False)
    for gname in ("yours", "other team"):
        x = d[(d.group == gname)]
        x = x[x.ros_value >= x.groupby("w").ros_value.transform(lambda s: s.quantile(0.4))]
        m, lo, hi = boot_mean(x.gap.to_numpy(), x.player_id.to_numpy())
        print(f"  {gname:11s} (top 60% of each roster's values): {m:+.2f} a game (95% {lo:+.2f} to {hi:+.2f})")
    mine = d[d.group == "yours"].groupby("name").agg(checks=("w", "size"), value=("ros_value", "mean"), since=("mean", "mean")).assign(gap=lambda x: x.since - x.value)
    print("  your players (mean over checkpoints):"); print(mine.sort_values("gap").round(1).to_string())

    # ---- (2) every team's starters, our weekly projection against what they scored
    out = []
    for wk in range(3, last + 1):
        f = files.get(wk)
        if f is None: continue
        lw = lastlog[lastlog.week == wk].set_index("player_id")
        for t in f["teams"]:
            for p in t["roster"]:
                if p.get("slot") in ("BN", "IR", None) or p["position"] not in SK: continue
                pid = str(p["player_id"])
                if pid not in lw.index: continue
                r = lw.loc[pid]
                if isinstance(r, pd.DataFrame): r = r.iloc[-1]
                proj = pd.to_numeric(r.blended_projection, errors="coerce")
                if proj != proj or r.nfl_id is None: continue
                a = hist[(hist.player_id == r.nfl_id) & (hist.week == wk)].fantasy_points_ppr.sum()
                out.append({"week": wk, "team": t["name"][:22], "mine": bool(t.get("is_mine")), "pid": pid,
                            "proj": float(proj), "plat": pd.to_numeric(r.platform_projection, errors="coerce"), "actual": float(a)})
    s = pd.DataFrame(out)
    s["gap"] = s.actual - s.proj
    s["plat_gap"] = s.actual - s.plat
    by = s.groupby(["team", "mine"]).agg(starts=("gap", "size"), ours=("gap", "mean"), platform=("plat_gap", "mean")).reset_index().sort_values("ours")
    print("every team's starters, weeks 3-%d: scored minus projected, a start" % last)
    print(by.round(2).to_string(index=False))
    for gname, x in (("yours", s[s.mine]), ("the others", s[~s.mine])):
        m, lo, hi = boot_mean(x.gap.to_numpy(), x.pid.to_numpy())
        mp, lop, hip = boot_mean(x.plat_gap.dropna().to_numpy(), x[x.plat_gap.notna()].pid.to_numpy())
        print(f"  {gname:10s}: ours {m:+.2f} (95% {lo:+.2f} to {hi:+.2f}); platform's {mp:+.2f} (95% {lop:+.2f} to {hip:+.2f}), {len(x)} starts")
