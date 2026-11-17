"""Season audit, part 3: the page's own season simulation, captured from an offline page build.

The rules it plays and its sums; every team's title and playoff odds against two baselines
(points so far, coin flips); how far your odds move if your team is weaker than simulated;
and where each team's simulated strength comes from (our season values against this
season's points a game). Run it in a scratch copy of the repository -- the build writes
the page and the logs: cd <copy> && PYTHONPATH=src python scripts/season_audit/title_sim.py
"""
import os, sys, runpy, numpy as np, pandas as pd
os.environ["STREAMER_GAME_DAY"] = "0"
from streamer.roster import title_moves

CAUGHT = {}
_init = title_moves.TitleEngine.__init__
def init(self, snapshot, cfg, *a, **k):
    _init(self, snapshot, cfg, *a, **k)
    if k.get("model") is None and snapshot.profile not in CAUGHT and self.confirm:
        CAUGHT[snapshot.profile] = (snapshot, self.model, cfg)
title_moves.TitleEngine.__init__ = init
week = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--week=")), None)
sys.argv = ["streamer.cli", "--offline", "publish"] + (["--week", week] if week else [])
if not week:
    import glob, re as _re
    sys.argv += ["--week", str(max(int(_re.search(r"week_(\d+)", p).group(1)) for p in glob.glob("data/leagues/*/week_*.json")))]
try:
    runpy.run_module("streamer.cli", run_name="__main__")
except SystemExit:
    pass

def table(rows, cols):
    w = [max(len(str(c)), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    print("  " + "  ".join(str(c).rjust(w[i]) for i, c in enumerate(cols)))
    for r in rows: print("  " + "  ".join(str(x).rjust(w[i]) for i, x in enumerate(r)))

rng = np.random.default_rng(11)
for prof, (snap, m, _cfg) in CAUGHT.items():
    print(f"\n================ {prof}: snapshot week {snap.week}, simulated weeks {m.weeks[0]}-{m.weeks[-1]} "
          f"(regular season to {m.reg_weeks}, {m.playoff_teams} playoff teams, {m.rounds} rounds of {m.round_weeks} week, "
          f"byes {2 ** m.rounds - m.playoff_teams}, reseed {m.reseed}, division winners first {m.divisions is not None}, "
          f"playoff tie -> {'points' if m.playoff_points_tiebreak else 'higher seed'}, median game {m.median}, "
          f"standings lag {m.lag}, {m.n} seasons)")
    teams = m.teams; me = next(i for i, t in enumerate(teams) if t.is_mine)
    left = {t.team_id: sum(1 for w, a, b in m.schedule if t.team_id in (a, b)) for t in teams}
    print("  regular-season games left per team (from the platform's schedule):", sorted(set(left.values())))
    base = m.odds()
    print(f"  sums: P(playoffs) {base.p_playoffs.sum():.3f}, P(title) {base.p_title.sum():.3f}, byes {base.p_bye.sum():.3f}")
    # The week under way is (nearly) settled; everything after it is drawn three ways.
    k0 = 1 if m.weeks[0] <= m.reg_weeks and m.lag + 0 == 0 or m.lag else 1
    later = slice(1, None)
    games = np.array([max(t.wins + t.losses + t.ties, 1) for t in teams], float)
    ppg = np.array([t.points_for for t in teams]) / games
    sims_mean = m.base_scores[:, :, 1:1 + max(m.reg_weeks - m.weeks[0], 1)].mean(axis=(1, 2))
    sd = 21.5
    def with_draws(mean_by_team):
        s = m.base_scores.copy()
        n_w = s.shape[2] - 1
        s[:, :, later] = mean_by_team[:, None, None] + sd * rng.standard_normal((len(teams), m.n, n_w))
        return s
    out = {}
    for label, mean in (("points so far", ppg), ("coin flips", np.full(len(teams), ppg.mean()))):
        saved = m.base_scores
        m.base_scores = with_draws(mean)
        out[label] = m.odds(retention=1.0)
        m.base_scores = saved
    order = np.argsort(-base.p_title)
    rows = []
    for i in order:
        t = teams[i]
        rows.append([t.name[:20] + (" *" if i == me else ""), f"{t.wins}-{t.losses}", round(ppg[i], 1), round(sims_mean[i], 1),
                     f"{base.p_playoffs[i]:.0%}", f"{out['points so far'].p_playoffs[i]:.0%}", f"{out['coin flips'].p_playoffs[i]:.0%}",
                     f"{base.p_title[i]:.1%}", f"{out['points so far'].p_title[i]:.1%}", f"{out['coin flips'].p_title[i]:.1%}"])
    table(rows, ["team", "rec", "pts/wk so far", "sim pts/wk ahead", "PO model", "PO pts", "PO coin", "title model", "title pts", "title coin"])
    # How far your odds move if your team is weaker than the simulation says, every week from next week on.
    rows = []
    for d in (0, 2.5, 5, 7.5, 10):
        s = m.base_scores[me].copy(); s[:, later] -= d
        o = m.odds(override={teams[me].team_id: s})
        rows.append([f"-{d}", f"{o.p_playoffs[me]:.1%}", f"{o.p_title[me]:.2%}", f"{o.exp_wins[me]:.2f}"])
    print("  your team weaker than simulated by this many points a week, from next week:")
    table(rows, ["pts/wk", "P(playoffs)", "P(title)", "exp wins"])
    # Where the title chance comes from: making it, then winning it from each seed.
    print(f"  your title chance if you make the playoffs: {base.p_title[me] / max(base.p_playoffs[me], 1e-9):.1%} "
          f"(league average {1 / m.playoff_teams:.1%})")


# ---- where each team's simulated strength comes from
from streamer.roster.projections import load_history
for prof, (snap, m, cfg) in CAUGHT.items():
    hist = load_history(cfg); hist = hist[hist.season == snap.season]
    ppg = hist.groupby("player_id").fantasy_points_ppr.agg(["mean", "count"])
    ahead = slice(1, 1 + max(m.reg_weeks - m.weeks[0], 1))
    sc = m.futures.scores[:, :, ahead]; pl = m.futures.played[:, :, ahead] if m.futures.played is not None else sc > 0
    rows = []
    for ti, t in enumerate(m.teams):
        for p in t.roster:
            j = m.pid.get(p.player_id)
            r = ppg.loc[p.nfl_id] if p.nfl_id in ppg.index else None
            rows.append({"team": t.name[:22], "mine": t.is_mine, "player": p.name, "pos": p.position, "slot": p.slot,
                         "status": p.status or "", "ros_value": p.ros_value, "this_week": p.projection,
                         "sim_mean": float(sc[:, j, :].mean()) if j is not None else np.nan,
                         "sim_played": float(pl[:, j, :].mean()) if j is not None else np.nan,
                         "ppg_2026": float(r["mean"]) if r is not None else np.nan, "games_2026": int(r["count"]) if r is not None else 0,
                         "ecr_ros": getattr(p, "ecr_ros_pos_rank", None), "reported_back": str(getattr(p, "reported_back", None) or "")})
    d = pd.DataFrame(rows)
    team_ahead = m.base_scores[:, :, ahead].mean(axis=(1, 2))
    games = np.array([max(t.wins + t.losses + t.ties, 1) for t in m.teams], float)
    summ = pd.DataFrame({"team": [t.name[:22] for t in m.teams], "mine": [t.is_mine for t in m.teams],
                         "so_far": [t.points_for / g for t, g in zip(m.teams, games)], "sim_ahead": team_ahead})
    # the same roster valued on what its players have scored this season (points a game, games played), best lineup
    from streamer.roster.waivers import lineup_slots
    def lineup_total(t, key):
        act = [p for p in t.roster if not p.in_ir_slot]
        a = lineup_slots(act, snap.starting_slots, key)
        return sum(key(p) for p in act if p.player_id in a)
    valmap = {p.player_id: (float(p.ros_value or 0.0)) for t in m.teams for p in t.roster}
    pp = {p.player_id: (float(ppg.loc[p.nfl_id]["mean"]) if p.nfl_id in ppg.index and ppg.loc[p.nfl_id]["count"] >= 3 else float(p.ros_value or 0.0))
          for t in m.teams for p in t.roster}
    summ["lineup_by_value"] = [lineup_total(t, lambda p: valmap[p.player_id] if p.status not in ("OUT", "INJURY_RESERVE", "IR", "O") else 0.0) for t in m.teams]
    summ["lineup_by_2026_ppg"] = [lineup_total(t, lambda p: pp[p.player_id] if p.status not in ("OUT", "INJURY_RESERVE", "IR", "O") else 0.0) for t in m.teams]
    summ = summ.sort_values("so_far", ascending=False)
    print(f"\n================ {prof}")
    print(summ.round(1).to_string(index=False))
    print("  correlation with points so far: sim ahead %.2f, lineup by our values %.2f, lineup by 2026 points a game %.2f" % (
        summ.so_far.corr(summ.sim_ahead), summ.so_far.corr(summ.lineup_by_value), summ.so_far.corr(summ.lineup_by_2026_ppg)))
    print("  sim ahead against lineup by our values: correlation %.2f" % summ.sim_ahead.corr(summ.lineup_by_value))
    sk = d[d.pos.isin(["QB", "RB", "WR", "TE"]) & d.ros_value.notna() & (d.games_2026 >= 4) & (d.ros_value >= 6)]
    print("  rostered players worth 6+ with 4+ games: season value minus 2026 points a game, by position:")
    print("   ", sk.assign(gap=sk.ros_value - sk.ppg_2026).groupby("pos").gap.agg(["mean", "count"]).round(2).to_dict())
    print("  value on 2026 points a game, slope (1 = values follow this season fully): %.2f" % np.polyfit(sk.ppg_2026, sk.ros_value, 1)[0])
