"""Season audit, part 3: team strength forecasts in the odds log against what teams then scored.
Run: PYTHONPATH=src python scripts/season_audit/title_strength_log.py
"""
import json, subprocess, numpy as np, pandas as pd

def week_file(lg, wk):
    try:
        return json.loads(subprocess.run(["git", "show", f"HEAD:data/leagues/{lg}/week_{wk}.json"],
                                         capture_output=True, text=True, check=True).stdout)
    except Exception:
        return None

for lg, first in (("espn", 1), ("yahoo", 3)):
    files = {wk: week_file(lg, wk) for wk in range(first, 13)}
    files = {k: v for k, v in files.items() if v}
    pf = pd.DataFrame({wk: {t["team_id"]: t["points_for"] for t in f["teams"]} for wk, f in files.items()})
    games = pd.DataFrame({wk: {t["team_id"]: t["wins"] + t["losses"] + t["ties"] for t in f["teams"]} for wk, f in files.items()})
    names = {t["team_id"]: t["name"][:20] for t in files[max(files)]["teams"]}
    mine = next(t["team_id"] for t in files[max(files)]["teams"] if t.get("is_mine"))
    # a week's score: points-for after it minus before it (a file's standings count the weeks before it)
    weekly = {}
    for wk in sorted(files):
        if wk + 1 in files and (games[wk + 1] - games[wk]).min() == 1:
            weekly[wk] = pf[wk + 1] - pf[wk]
    weekly = pd.DataFrame(weekly)
    done = max(weekly.columns)
    print(f"\n================ {lg}: weekly scores rebuilt for weeks {min(weekly.columns)}-{done}")
    sd_within = (weekly.sub(weekly.mean(axis=1), axis=0)).stack().std(ddof=1)
    print(f"  a team's week, spread around its own average: sd {sd_within:.1f}; average {weekly.stack().mean():.1f}")
    o = pd.read_parquet(f"results/{lg}/odds_log.parquet"); o["team_id"] = o.team_id.astype(str)
    rows = []
    for w in sorted(o.week.unique()):
        if w + 1 > done: continue
        lw = o[o.week == w].set_index("team_id")
        for tid, r in lw.iterrows():
            # was week w still in the simulation? (the log's points-for counts weeks before it)
            through = None
            if r.points_for != r.points_for:          # not logged before week 7: logged at the week's end, like the rest
                through = w
            for k in (w, w + 1):
                if k in pf.columns and abs(pf.loc[tid, k] - r.points_for) < 0.6: through = k
            if through is None: continue
            future = r.exp_points - (weekly.loc[tid, w] if through == w and w in weekly.columns else 0.0)
            n_future = 14 - w
            realized = weekly.loc[tid, [k for k in weekly.columns if w < k <= done]].mean()
            n_played = games.loc[tid, w + 1] if (w + 1) in games.columns else None
            naive = pf.loc[tid, w + 1] / max(games.loc[tid, w + 1], 1) if (w + 1) in pf.columns else np.nan
            rows.append({"log": w, "team": tid, "mine": tid == mine, "model": future / n_future, "naive": naive,
                         "realized": realized, "weeks_after": done - w})
    d = pd.DataFrame(rows)
    gone = d[d[["model", "naive", "realized"]].isna().any(axis=1)]
    if len(gone): print(f"  {len(gone)} team-logs without a forecast or a score left out: logs {sorted(gone.log.unique())}")
    d = d.dropna(subset=["model", "naive", "realized"])
    # Each log week, against the league's own level that week (the model's level runs a little high from streaming)
    for c in ("model", "naive", "realized"):
        d[c + "_rel"] = d[c] - d.groupby("log")[c].transform("mean")
    print("  level: model's expected a week %.1f, realized %.1f (all teams, logs %s)" % (d.model.mean(), d.realized.mean(), sorted(d.log.unique())))
    for c in ("model", "naive"):
        slope = np.polyfit(d[c + "_rel"], d.realized_rel, 1)[0]
        r = np.corrcoef(d[c + "_rel"], d.realized_rel)[0, 1]
        mae = (d[c + "_rel"] - d.realized_rel).abs().mean()
        print(f"  {c:6s}: teams told apart (realized gap on forecast gap, 1 = right) {slope:.2f}; correlation {r:.2f}; miss {mae:.1f} a week")
    both = np.linalg.lstsq(np.c_[d.model_rel, d.naive_rel], d.realized_rel, rcond=None)[0]
    print(f"  realized on both: model {both[0]:.2f}, points so far {both[1]:.2f}")
    m = d[d.mine]
    print("  your team, against the league's level that week:")
    print(m[["log", "model_rel", "naive_rel", "realized_rel", "weeks_after"]].round(1).to_string(index=False))
    print("  your team, mean: forecast %+.1f, points-so-far %+.1f, realized %+.1f a week vs the league" % (m.model_rel.mean(), m.naive_rel.mean(), m.realized_rel.mean()))
