"""Do D/ST and kickers follow their playoff-week opponents? A held-out test.

    python scripts/fit_schedule_dstk.py

Replays 2023-2025 at checkpoints (weeks 3-10): every unit's season value then
(our walk-forward ``baseline_points``) against its points in each later week
it played, with the week scaled by ``environment ** a * allowed ** b``:

* **environment** -- the game's implied total predicted from past closing
  lines (offence + defence + home, ridge; :mod:`streamer.roster.schedule`):
  for a defence the *opponent's* total over the league mean (fewer points
  for them, more for the defence: ``a`` should come out negative), for a
  kicker his own team's over its trailing average, as for quarterbacks;
* **allowed** -- points the opponent has given up to the position per game,
  shrunk toward the league mean by ``allowed_k`` games.

``a, b`` are fitted on two seasons (grid, squared error) and scored on the
third; the rule is the one every other change met: better overall and worse
in at most one held-out season. DECISIONS.md, "Playoff-week matchups for
D/ST and kickers".
"""

from __future__ import annotations

import itertools
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

WALK = "data/backtest/dstk_walkforward.parquet"
CHECKPOINTS = range(3, 11)
A_GRID = np.round(np.arange(-2.0, 2.01, 0.1), 2)
B_GRID = np.round(np.arange(0.0, 1.21, 0.1), 2)


def signals(games: pd.DataFrame, walk: pd.DataFrame, lam: float, prev_w: float, k: float) -> pd.DataFrame:
    """One row per (checkpoint, unit, future week): baseline, actual, env, allowed."""
    from streamer.roster.schedule import _allowed, _ratings

    hist = walk.rename(columns={"fantasy_points": "fantasy_points_ppr"})
    rows = []
    for season in sorted(walk["season"].unique()):
        for c in CHECKPOINTS:
            predict = _ratings(games, season, c, lam, prev_w)
            if predict is None:
                continue
            past = games[(games["season"] < season) | ((games["season"] == season) & (games["week"] < c))]
            past = past.dropna(subset=["team_implied_total"]).sort_values(["season", "week"])
            trail = past.groupby("team")["team_implied_total"].apply(lambda s: float(s.tail(6).mean()))
            mean_total = float(past[past["season"] >= season - 1]["team_implied_total"].mean())
            fut = games[(games["season"] == season) & (games["week"] > c)]
            opp = {(r.week, r.team): (r.opponent, int(r.is_home)) for r in fut.itertuples()}
            at_c = walk[(walk["season"] == season) & (walk["week"] == c)].groupby(["position", "team"])["baseline_points"].max()
            for pos in ("DST", "K"):
                allowed = _allowed(hist, games, season, c, pos, k, prev_w)
                later = walk[(walk["season"] == season) & (walk["week"] > c) & (walk["position"] == pos)]
                for r in later.itertuples():
                    base = at_c.get((pos, r.team))
                    g = opp.get((r.week, r.team))
                    if base is None or g is None or not np.isfinite(base):
                        continue
                    o, home = g
                    if pos == "DST":
                        env = predict(o, r.team, 1 - home) / mean_total
                    else:
                        env = predict(r.team, o, home) / trail.get(r.team, np.nan)
                    rows.append({"season": season, "checkpoint": c, "week": r.week, "position": pos,
                                 "team": r.team, "base": float(base), "actual": float(r.fantasy_points),
                                 "env": env, "allowed": allowed.get(o, 1.0)})
    d = pd.DataFrame(rows)
    return d[np.isfinite(d["env"]) & (d["env"] > 0)]


def sse(d: pd.DataFrame, a: float, b: float) -> float:
    pred = d["base"] * d["env"] ** a * d["allowed"] ** b
    return float(((pred - d["actual"]) ** 2).sum())


def main() -> int:
    logging.disable(logging.WARNING)
    from streamer.config import get_config
    from streamer.data.nflverse import games_frame
    from streamer.teams import normalize_team

    cfg = get_config().for_profile("espn")
    conf = (cfg.raw.get("roster") or {}).get("schedule") or {}
    lam, prev_w, k = float(conf.get("ridge", 4.0)), float(conf.get("previous_season_weight", 0.5)), \
        float(conf.get("allowed_k", 6.0))
    games = games_frame(cfg)
    games = games[games["game_type"] == "REG"].assign(
        team=lambda g: g["team"].map(normalize_team), opponent=lambda g: g["opponent"].map(normalize_team))
    walk = pd.read_parquet(WALK).assign(team=lambda w: w["team"].map(normalize_team))
    d = signals(games, walk, lam, prev_w, k)
    seasons = sorted(d["season"].unique())
    print(f"{len(d)} future unit-weeks, seasons {seasons}")
    for pos in ("DST", "K"):
        p = d[d["position"] == pos]
        forms = {"implied only": (A_GRID, [0.0]), "allowed only": ([0.0], B_GRID), "both": (A_GRID, B_GRID)}
        print(f"\n{pos}: RMSE per game, held-out season (flat -> fitted)")
        for name, (ag, bg) in forms.items():
            better, out = 0, []
            tot_flat = tot_fit = 0.0
            for s in seasons:
                tr, te = p[p["season"] != s], p[p["season"] == s]
                a, b = min(itertools.product(ag, bg), key=lambda ab: sse(tr, *ab))
                flat, fit = sse(te, 0.0, 0.0), sse(te, a, b)
                tot_flat, tot_fit = tot_flat + flat, tot_fit + fit
                better += fit < flat
                out.append(f"{s}: {np.sqrt(flat / len(te)):.3f} -> {np.sqrt(fit / len(te)):.3f} (a {a:+.1f}, b {b:.1f})")
            a, b = min(itertools.product(ag, bg), key=lambda ab: sse(p, *ab))
            print(f"  {name:13} {better}/{len(seasons)} better; overall {np.sqrt(tot_flat / len(p)):.3f} -> "
                  f"{np.sqrt(tot_fit / len(p)):.3f}; all seasons a {a:+.1f} b {b:.1f}")
            for line in out:
                print("     ", line)
        late = p[p["week"].isin([15, 16, 17])]
        if len(late):
            a, b = min(itertools.product(A_GRID, B_GRID), key=lambda ab: sse(p[p["week"] < 15], *ab))
            print(f"  weeks 15-17 ({len(late)}), fit on earlier weeks: {np.sqrt(sse(late, 0, 0) / len(late)):.3f} -> "
                  f"{np.sqrt(sse(late, a, b) / len(late)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
