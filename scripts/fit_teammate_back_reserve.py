"""Should a teammate back from a season-start reserve list count as back?

    python scripts/fit_teammate_back_reserve.py

Walk-forward over 2023-2025, weeks 3-18, as ``fit_teammate_back.py``: every
player projected by the production volume model on the games before the
week, with the teammate-return correction at config.yaml's shares, once as
it was (a returning teammate must have played for the team this season) and
once also counting one who has not, when the team carried him on a reserve
list this season before the week (:func:`projections.prior_absences` for the
season, earlier weeks only). Nothing is fitted: on/off at the shares already
chosen. Scored on the points each
player scored that week, on the player-weeks it moves, per season; kept
only if better overall and worse in at most one season (DECISIONS.md, "Last
season's stand-in games" and "Back from a season-start reserve list").
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
POSITIONS = ("RB", "WR", "TE")


def build(cfg) -> pd.DataFrame:
    from streamer.roster import projections as pj

    h = pj.load_history(cfg)
    conf = dict(cfg.raw["roster"])
    rows = []
    for season in (2023, 2024, 2025):
        out_prior = pj.prior_absences(cfg, season - 1)
        out_now = pj.prior_absences(cfg, season)
        for week in range(3, 19):
            cur = h[(h["season"] == season) & (h["week"] == week)]
            if cur.empty:
                continue
            prior = h[(h["season"] < season) | ((h["season"] == season) & (h["week"] < week))]
            pos_mean = pj.position_rates(prior)[0]
            base = pj.player_table(h, season, week, cfg)
            back = {pid: (1.0, 1.0) for pid in cur["player_id"]}
            got = cur.set_index("player_id")["fantasy_points_ppr"]
            keep = base[base["player_id"].isin(got.index) & (base["vol"] >= 3.0)
                        & base["position"].isin(POSITIONS)].set_index("player_id")
            out = keep[["position", "vol", "eff"]].assign(season=season, week=week,
                                                         pts=got.reindex(keep.index).to_numpy())
            before = {x for x in out_now if x[0] < week}
            for name, season_out in (("off", None), ("on", before)):
                conf["teammate_back"] = {**conf["teammate_back"], "reserve_return": list(POSITIONS)}
                t = pj._returning(prior, base, back, conf, pos_mean, season, out_prior, season_out)
                out[name] = t.set_index("player_id")["vol"].reindex(keep.index).to_numpy()
            rows.append(out.reset_index())
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    logging.disable(logging.WARNING)
    from streamer.config import get_config

    d = build(get_config().for_profile("espn"))

    def err(g, c):
        return (g["pts"] - g[c] * g["eff"]).to_numpy()

    def rmse(e):
        return float(np.sqrt((e ** 2).mean()))

    for pos in POSITIONS:
        allp = d[d["position"] == pos]
        g = allp[(allp["on"] - allp["off"]).abs() > 1e-9]
        print(f"== {pos}: {len(g)} of {len(allp)} player-weeks moved")
        wins, n_s = 0, 0
        for season, te in g.groupby("season"):
            e0, e1 = err(te, "off"), err(te, "on")
            wins += rmse(e1) < rmse(e0)
            n_s += 1
            print(f"   {season}: n={len(te):4d} RMSE {rmse(e0):.3f} -> {rmse(e1):.3f}  "
                  f"bias {e0.mean():+.2f} -> {e1.mean():+.2f}")
        if len(g):
            r0, r1 = rmse(err(g, "off")), rmse(err(g, "on"))
            keep = r1 < r0 and wins >= n_s - 1
            print(f"   overall RMSE {r0:.3f} -> {r1:.3f}, better in {wins} of {n_s}: {'keep' if keep else 'drop'}")
        print(f"   every {pos} player-week: MAE {np.abs(err(allp, 'off')).mean():.4f} -> "
              f"{np.abs(err(allp, 'on')).mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
