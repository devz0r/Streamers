"""Refit how much of an absent lead's work the next man up takes, by how much
of his position's other snaps he already had.

    python scripts/fit_next_man_up.py [--dry-run]

Every 2021-2025 absence of a lead at RB and TE (the team plays, he does
not): the lead is the most expected points a game at the position over the
team's previous four games (two or more played), the next man up the
teammate with the next most. For each: both players' model volume the week
the absence began, the next man up's efficiency, his concentration (his
share of the position's offensive snaps that did not go to the lead, same
four games; nflverse snap counts) and what he scored in his first game.

The weekly projection gives him ``(volume + share x gap) x efficiency``. The
share is fitted two ways, each season held out in turn: one share for the
position (today's config), and ``base + slope x (concentration - mean)``.
The slope is kept only if it predicts the held-out first games better
overall and is worse in at most one season. Shares are kept inside
``depth.SHARE_BOUNDS`` (the clearest backups averaged 0.80 of the gap over a
whole absence). Writes ``src/streamer/roster/next_man_up.json``.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

POSITIONS = ("RB", "TE")


def spells(h: pd.DataFrame) -> pd.DataFrame:
    h = h[h.position.isin(POSITIONS)].copy()
    h["exp"] = h["total_fantasy_points_exp"].fillna(h["fantasy_points_ppr"])
    out = []
    for (season, team, pos), g in h.groupby(["season", "team", "position"]):
        weeks = sorted(h[(h.season == season) & (h.team == team)].week.unique())
        by_week = {w: d.set_index("player_id") for w, d in g.groupby("week")}
        i = 4
        while i < len(weeks):
            prev = weeks[i - 4:i]
            vols = {}
            for pid in set().union(*[by_week.get(x, pd.DataFrame()).index for x in prev]):
                vals = [by_week[x].loc[pid, "exp"] for x in prev if x in by_week and pid in by_week[x].index]
                if len(vals) >= 2:
                    vols[pid] = float(np.mean(vals))
            if len(vols) < 2:
                i += 1
                continue
            lead = max(vols, key=vols.get)
            if lead in by_week.get(weeks[i], pd.DataFrame()).index:
                i += 1
                continue
            heir = max((p for p in vols if p != lead), key=vols.get)
            j, during = i, []
            while j < len(weeks) and lead not in by_week.get(weeks[j], pd.DataFrame()).index:
                d = by_week.get(weeks[j], pd.DataFrame())
                if heir in d.index:
                    during.append((float(d.loc[heir, "exp"]), float(d.loc[heir, "fantasy_points_ppr"])))
                j += 1
            if during:
                out.append({"season": int(season), "team": team, "position": pos, "week": int(weeks[i]),
                            "lead": lead, "heir": heir, "first_pts": during[0][1],
                            "exps": [e for e, _p in during], "games": len(during)})
            i = j + 1
    return pd.DataFrame(out)


def build(cfg) -> pd.DataFrame:
    from streamer.data.nflverse import load_snap_counts
    from streamer.roster import depth
    from streamer.roster.projections import load_history, player_table

    h = load_history(cfg)
    names = h.drop_duplicates("player_id").set_index("player_id")["player_display_name"]
    snaps = depth.prepare(load_snap_counts([2021, 2022, 2023, 2024, 2025], cfg))
    sp = spells(h[h.season <= 2025])
    rows = []
    for (season, week), g in sp.groupby(["season", "week"]):
        tab = player_table(h, int(season), int(week), cfg).set_index("player_id")
        for r in g.itertuples():
            if r.lead not in tab.index or r.heir not in tab.index:
                continue
            vl, vh = float(tab.loc[r.lead, "vol"]), float(tab.loc[r.heir, "vol"])
            eff = float(tab.loc[r.heir, "eff"])
            if vl - vh < 3.0:
                continue
            conc = depth.concentration(snaps, int(season), int(week), r.team, r.position,
                                       names.get(r.heir, ""), names.get(r.lead, ""))
            if conc is None:
                continue
            rows.append({"season": int(season), "position": r.position, "vl": vl, "vh": vh, "eff": eff,
                         "gap": vl - vh, "conc": conc, "first_pts": r.first_pts,
                         "share": float(np.mean([(e - vh) / (vl - vh) for e in r.exps])), "games": r.games})
    return pd.DataFrame(rows)


def fit(d: pd.DataFrame, slope: bool, mean: float) -> tuple[float, float]:
    """Least squares on first-game points: pts - vh*eff = eff*gap*(base + slope*(conc - mean))."""
    y = (d.first_pts - d.vh * d.eff).to_numpy()
    x1 = (d.eff * d.gap).to_numpy()
    if not slope:
        return float((x1 * y).sum() / (x1 * x1).sum()), 0.0
    x2 = x1 * (d.conc - mean).to_numpy()
    (b, s), *_ = np.linalg.lstsq(np.column_stack([x1, x2]), y, rcond=None)
    return float(b), float(s)


def predict(d: pd.DataFrame, base: float, slope: float, mean: float) -> np.ndarray:
    from streamer.roster.depth import SHARE_BOUNDS

    share = np.clip(base + slope * (d.conc - mean), *SHARE_BOUNDS)
    return ((d.vh + share * d.gap) * d.eff).to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    from streamer.config import get_config
    from streamer.roster import depth

    d = build(get_config().for_profile("espn"))
    out = {}
    for pos, g in d.groupby("position"):
        mean = float(g.conc.mean())
        wins, rows = 0, []
        for season in sorted(g.season.unique()):
            tr, te = g[g.season != season], g[g.season == season]
            b0, _ = fit(tr, False, mean)
            b1, s1 = fit(tr, True, mean)
            e0 = te.first_pts.to_numpy() - predict(te, b0, 0.0, mean)
            e1 = te.first_pts.to_numpy() - predict(te, b1, s1, mean)
            rows.append((season, len(te), np.sqrt((e0 ** 2).mean()), np.sqrt((e1 ** 2).mean()), s1))
            wins += np.sqrt((e1 ** 2).mean()) < np.sqrt((e0 ** 2).mean())
        n = sum(r[1] for r in rows)
        rmse0 = np.sqrt(sum(r[1] * r[2] ** 2 for r in rows) / n)
        rmse1 = np.sqrt(sum(r[1] * r[3] ** 2 for r in rows) / n)
        base, slope = fit(g, True, mean)
        print(f"{pos}: {len(g)} absences, concentration mean {mean:.2f}; held-out RMSE one share {rmse0:.3f}, "
              f"with concentration {rmse1:.3f}; better in {wins} of {len(rows)} seasons; "
              f"full fit base {base:.2f} slope {slope:+.2f}")
        for season, k, r0, r1, s1 in rows:
            print(f"   {season}: n={k:3} {r0:.2f} -> {r1:.2f} (slope {s1:+.2f})")
        for lo, hi in ((0.0, 0.5), (0.5, 0.75), (0.75, 1.01)):
            b = g[(g.conc >= lo) & (g.conc < hi)]
            if len(b):
                w = b.games * b.gap
                print(f"   concentration {lo:.2f}-{hi:.2f}: n={len(b):3} share over the absence "
                      f"{np.average(b.share, weights=w):.2f}")
        # Kept only when it is better overall and worse in at most one season:
        # a relationship that flips from season to season is not one to use.
        if rmse1 < rmse0 and wins >= len(rows) - 1:
            out[pos] = {"base": round(base, 3), "slope": round(slope, 3), "mean": round(mean, 3),
                        "evidence": {"absences": int(len(g)), "rmse_one_share": round(float(rmse0), 3),
                                     "rmse_by_concentration": round(float(rmse1), 3),
                                     "seasons_better": f"{wins} of {len(rows)}"}}
    text = json.dumps(out, indent=2) + "\n"
    if args.dry_run:
        print(text)
        return 0
    depth.MODEL_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {depth.MODEL_PATH} ({', '.join(out) or 'no position kept'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
