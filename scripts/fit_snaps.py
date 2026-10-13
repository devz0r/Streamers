"""Do snap-share trends predict a role change before the targets do?

    python scripts/fit_snaps.py [--cache rows.parquet]

Replays 2022-2025, weeks 3-18, through the production projection for backs,
receivers and tight ends (volume model x damped implied-total scaling, as
production). Snap counts (Pro Football Reference via nflverse) joined by
team, week and name. Before each week, per player at his current team this
season: ``s2`` his offensive snap share over his last 2 games, ``s_prior``
over his earlier games (2+ of them), ``delta = s2 - s_prior``; and ``gap`` --
his snap share over the last 2 games against the share his opportunity
(expected points) is of his position group's: a player on the field more
than he is used may be about to be used more.

Forms tried, per position: ``pred x exp(a x delta)``, ``pred x exp(b x gap)``
and both; ``(a, b)`` picked on three seasons, scored on the fourth; kept
only if better overall and worse in at most one season. DECISIONS.md,
"Snap share as an early sign".
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
POSITIONS = ("RB", "WR", "TE")
A = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
B = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)


def build(cfg) -> pd.DataFrame:
    from streamer.data.nflverse import games_frame
    from streamer.roster import projections as pj
    from streamer.roster import snaps as sn
    from streamer.teams import normalize_team

    h = pj.load_history(cfg)
    raw = pd.concat([pd.read_parquet(cfg.raw_dir / f"snap_counts_{s}.parquet") for s in range(2022, 2027)
                     if (cfg.raw_dir / f"snap_counts_{s}.parquet").exists()])
    joined = sn.join(h, sn.tidy(raw))
    games = games_frame(cfg)
    games = games[games["game_type"] == "REG"].assign(team=lambda g: g["team"].map(normalize_team))
    damp = float(cfg.raw["roster"]["vegas_damping"])
    rows = []
    for season in (2022, 2023, 2024, 2025):
        for week in range(3, 19):
            cur = joined[(joined["season"] == season) & (joined["week"] == week)]
            if cur.empty:
                continue
            past = games[(games["season"] < season) | ((games["season"] == season) & (games["week"] < week))]
            trail = (past.dropna(subset=["team_implied_total"]).sort_values(["season", "week"])
                     .groupby("team")["team_implied_total"].apply(lambda s: float(s.tail(6).mean())))
            now = games[(games["season"] == season) & (games["week"] == week)].drop_duplicates("team") \
                .set_index("team")["team_implied_total"]
            t = pj.player_table(h, season, week, cfg)
            got = cur.groupby("player_id")["fantasy_points_ppr"].sum()
            team_now = cur.groupby("player_id")["_team"].first()
            sig = sn.signals(joined, season, week, team_now.to_dict())
            t = t[t["player_id"].isin(got.index) & t["position"].isin(sn.POSITIONS) & t["blend"].notna()
                  & t["player_id"].isin(sig.index)]
            for r in t.itertuples():
                team = team_now[r.player_id]
                if team not in trail or not np.isfinite(now.get(team, np.nan)):
                    continue
                scale = float(np.clip(now[team] / trail[team], 0.7, 1.3)) ** damp
                g = sig.loc[r.player_id]
                rows.append({"season": season, "week": week, "position": r.position,
                             "pred": float(r.blend) * scale, "pts": float(got[r.player_id]),
                             "delta": g["delta"], "gap": g["gap"]})
        print(season, len(rows), flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    from streamer.config import get_config

    cfg = get_config().for_profile("espn")
    if args.cache and Path(args.cache).exists():
        d = pd.read_parquet(args.cache)
    else:
        d = build(cfg)
        if args.cache:
            d.to_parquet(args.cache)

    def pred(g, a, b):
        return g["pred"] * np.exp(a * g["delta"] + b * g["gap"])

    def rmse(g, a, b):
        return float(np.sqrt(((g["pts"] - pred(g, a, b)) ** 2).mean()))

    forms = {"trend": (A, (0.0,)), "snaps vs use": ((0.0,), B), "both": (A, B)}
    for pos in POSITIONS:
        p = d[d["position"] == pos]
        big = p[p["delta"].abs() >= 0.15]
        print(f"\n{pos}: {len(p)} weeks; snap share up 15+ pts: bias {(big[big.delta > 0].pts - big[big.delta > 0].pred).mean():+.2f} "
              f"(n={int((big.delta > 0).sum())}), down 15+: {(big[big.delta < 0].pts - big[big.delta < 0].pred).mean():+.2f} "
              f"(n={int((big.delta < 0).sum())}), all {(p.pts - p.pred).mean():+.2f}")
        for name, (ag, bg) in forms.items():
            wins, t0, t1, picks = 0, 0.0, 0.0, []
            for s in sorted(p["season"].unique()):
                tr, te = p[p["season"] != s], p[p["season"] == s]
                a, b = min(itertools.product(ag, bg), key=lambda ab: rmse(tr, *ab))
                r0, r1 = rmse(te, 0, 0), rmse(te, a, b)
                wins += r1 < r0
                t0 += r0 ** 2 * len(te)
                t1 += r1 ** 2 * len(te)
                picks.append(f"{s}:({a},{b}) {r0:.3f}->{r1:.3f}")
            o0, o1 = np.sqrt(t0 / len(p)), np.sqrt(t1 / len(p))
            keep = o1 < o0 and wins >= 3
            print(f"  {name:13} {o0:.4f} -> {o1:.4f}, better {wins}/4 {'KEEP' if keep else 'drop'} | " + "  ".join(picks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
