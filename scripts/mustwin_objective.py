"""Must-win objective test (DECISIONS.md, "Ranking moves by playoff odds when the title is out of reach: not adopted").

    python scripts/mustwin_objective.py SNAPSHOTS_DIR   # SNAPSHOTS_DIR/snaps_w12.pkl: {profile: (projected snapshot, cfg)}

Prices 12 pickups for every team on one simulation and on six held-out ones, then scores
ranking by title odds, playoff odds and this week's win chance by the held-out title odds of each pick.
"""
import os, sys, pickle, time, logging, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import numpy as np
from streamer.roster.season import SeasonModel
from streamer.roster.lineup import use_profile
S = sys.argv[1]
CAUGHT = pickle.load(open(os.path.join(S, "snaps_w12.pkl"), "rb"))
SKILL = ("QB", "RB", "WR", "TE")
N_CAND, SEEDS_TRUTH = 12, 6
rows = []
for prof, (snap, cfg) in CAUGHT.items():
    use_profile(cfg)
    fas = sorted([p for p in snap.free_agents if p.position in SKILL and not p.is_out and (p.ros_value or 0) > 0],
                  key=lambda p: -float(p.ros_value or 0))[:N_CAND]
    t0 = time.time()
    models = [SeasonModel(snap, cfg, n_sims=6000, seed=101 + s, extra_players=fas) for s in range(1 + SEEDS_TRUTH)]
    print(prof, "models built", round(time.time() - t0), "s", flush=True)
    wk = int(snap.week)
    for j, team in enumerate(snap.teams):
        ros = [p for p in team.roster if p.position in SKILL and not p.in_ir_slot]
        if not ros:
            continue
        drop = min(ros, key=lambda p: float(p.ros_value or 0))
        base_ids = [p.player_id for p in team.roster]
        opp = next((b if a == team.team_id else a for w, a, b in models[0].schedule
                    if w == wk and team.team_id in (a, b)), None)
        for x in fas:
            ids = [i for i in base_ids if i != drop.player_id] + [x.player_id]
            rec = {"prof": prof, "team": team.name, "add": x.name}
            for s, m in enumerate(models):
                sc = m.team_scores(ids, owner=team.team_id)
                o = m.odds(override={team.team_id: sc})
                if s == 0:
                    b = m.odds()
                    rec.update(title_a=o.p_title[j], po_a=o.p_playoffs[j], base_title_a=b.p_title[j],
                               base_po_a=b.p_playoffs[j])
                    if opp is not None and wk in m.weeks:
                        k = m.weeks.index(wk)
                        oi = m.team_index[opp]
                        rec["win_a"] = float((sc[:, k] > m.base_scores[oi, :, k]).mean())
                else:
                    rec.setdefault("truth", []).append(o.p_title[j])
                    rec.setdefault("truth_po", []).append(o.p_playoffs[j])
            rec["truth"] = float(np.mean(rec["truth"])); rec["truth_po"] = float(np.mean(rec["truth_po"]))
            rows.append(rec)
        print(" ", team.name, "done", round(time.time() - t0), "s", flush=True)
pickle.dump(rows, open(os.path.join(S, "mustwin_rows.pkl"), "wb"))


if True:
    import subprocess
    subprocess.run([sys.executable, __file__.replace("mustwin_objective.py", "mustwin_objective_score.py"), os.path.join(S, "mustwin_rows.pkl")])
