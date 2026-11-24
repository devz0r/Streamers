import sys, pickle, numpy as np, pandas as pd
d = pd.DataFrame(pickle.load(open(sys.argv[1], "rb")))
out = []
for (prof, team), g in d.groupby(["prof", "team"]):
    best = g.truth.max()
    r = {"prof": prof, "team": team, "po": g.base_po_a.iloc[0], "title": g.base_title_a.iloc[0], "spread": best - g.truth.min()}
    for rule, col in (("title", "title_a"), ("playoffs", "po_a"), ("this week", "win_a")):
        if col in g and g[col].notna().all():
            pick = g.loc[g[col].idxmax()]
            r[rule] = best - pick.truth                       # regret in true title odds
            r[rule + "_po"] = g.truth_po.max() - pick.truth_po
    out.append(r)
o = pd.DataFrame(out)
o["band"] = np.where(o.po < 0.10, "playoffs <10%", np.where(o.po < 0.5, "10-50%", "50%+"))
rules = [c for c in ("title", "playoffs", "this week") if c in o]
print("Regret: true title odds (36,000 held-out seasons) lost against the best of the 12 moves, in points; lower is better")
for key, g in [("all", o)] + list(o.groupby("prof")) + list(o.groupby("band")):
    print(f"  {key:14} n={len(g):2}  " + "  ".join(f"{r} {g[r].mean() * 100:.3f}" for r in rules)
          + f"  | spread of the 12 {g.spread.mean() * 100:.3f}")
print("Regret in true playoff odds (points):")
for key, g in [("all", o)] + list(o.groupby("band")):
    print(f"  {key:14} " + "  ".join(f"{r} {g[r + '_po'].mean() * 100:.2f}" for r in rules))
low = o[o.po < 0.10]
print("\nlow-odds teams:", low[["prof", "team", "po", "title"] + rules].round(4).to_string(index=False))
