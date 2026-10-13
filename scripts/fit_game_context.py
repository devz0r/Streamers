"""This week's opponent and game script, on top of the implied total?

    python scripts/fit_game_context.py [--cache rows.parquet]

Replays 2022-2025, weeks 3-18, through the production projection for every
QB, RB, WR and TE who played: the volume model times this week's implied
team total over his team's trailing one, damped as in production -- the
spread already moves that total, so what is tested is what is left. Two
signals, both known before kickoff:

* **allowed** -- the opponent's points allowed to the position per game,
  this season and last at half weight, shrunk toward the league by
  ``allowed_k`` games (:func:`schedule._allowed`, as the season simulator
  uses it for the weeks to come);
* **spread** -- his team's point spread (positive: favoured), scaled by 7.

The week becomes ``pred * allowed ** a * exp(b * spread / 7)``; ``(a, b)``
per position picked on three seasons and scored on the fourth, each signal
alone and together; kept only if better overall and worse in at most one
season. DECISIONS.md, "This week's defence and game script".
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
POSITIONS = ("QB", "RB", "WR", "TE")
A = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
B = (-0.15, -0.1, -0.06, -0.03, 0.0, 0.03, 0.06, 0.1, 0.15)


def build(cfg) -> pd.DataFrame:
    from streamer.data.nflverse import games_frame
    from streamer.roster import projections as pj
    from streamer.roster.schedule import _allowed
    from streamer.teams import normalize_team

    h = pj.load_history(cfg)
    h = h.assign(_team=h["team"].map(normalize_team))
    games = games_frame(cfg)
    games = games[games["game_type"] == "REG"].assign(team=lambda g: g["team"].map(normalize_team),
                                                       opponent=lambda g: g["opponent"].map(normalize_team))
    conf = cfg.raw["roster"]
    damp, k = float(conf["vegas_damping"]), float((conf.get("schedule") or {}).get("allowed_k", 6.0))
    rows = []
    for season in (2022, 2023, 2024, 2025):
        for week in range(3, 19):
            cur = h[(h["season"] == season) & (h["week"] == week)]
            if cur.empty:
                continue
            past = games[(games["season"] < season) | ((games["season"] == season) & (games["week"] < week))]
            trail = (past.dropna(subset=["team_implied_total"]).sort_values(["season", "week"])
                     .groupby("team")["team_implied_total"].apply(lambda s: float(s.tail(6).mean())))
            now = games[(games["season"] == season) & (games["week"] == week)].set_index("team")
            allowed = {pos: _allowed(h, games, season, week, pos, k, 0.5) for pos in POSITIONS}
            t = pj.player_table(h, season, week, cfg)
            got = cur.groupby("player_id")["fantasy_points_ppr"].sum()
            team_now = cur.groupby("player_id")["_team"].first()
            t = t[t["player_id"].isin(got.index) & t["position"].isin(POSITIONS) & t["blend"].notna()]
            for r in t.itertuples():
                team = team_now[r.player_id]
                if team not in now.index or team not in trail:
                    continue
                g = now.loc[team]
                if isinstance(g, pd.DataFrame):
                    g = g.iloc[0]
                if not np.isfinite(g["team_implied_total"]):
                    continue
                scale = float(np.clip(g["team_implied_total"] / trail[team], 0.7, 1.3)) ** damp
                rows.append({"season": season, "week": week, "position": r.position,
                             "pred": float(r.blend) * scale, "pts": float(got[r.player_id]),
                             "allowed": allowed[r.position].get(g["opponent"], 1.0),
                             "spread": float(g["team_spread"]) if np.isfinite(g["team_spread"]) else 0.0})
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
        return g["pred"] * g["allowed"] ** a * np.exp(b * g["spread"] / 7.0)

    def rmse(g, a, b):
        return float(np.sqrt(((g["pts"] - pred(g, a, b)) ** 2).mean()))

    forms = {"allowed only": (A, (0.0,)), "spread only": ((0.0,), B), "both": (A, B)}
    for pos in POSITIONS:
        p = d[d["position"] == pos]
        print(f"\n{pos}: {len(p)} player-weeks, flat RMSE {rmse(p, 0, 0):.4f}")
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
            print(f"  {name:12} {o0:.4f} -> {o1:.4f}, better {wins}/4 {'KEEP' if keep else 'drop'} | " + "  ".join(picks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
