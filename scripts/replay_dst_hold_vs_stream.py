"""Hold one strong defence, or stream the best matchup each week? A replay.

    python scripts/replay_dst_hold_vs_stream.py

2023-2025, from the walk-forward D/ST projections the rankings would have
published (``data/backtest/dstk_walkforward.parquet``: ``expected_points``
for the week, ``baseline_points`` the season level, both from strictly earlier
games), scored on each league's own D/ST sheet from the play-by-play. Nothing
is fitted, so there is nothing to hold out: each season is reported on its own.

At a checkpoint week ``c`` (3-10, when a manager might pick a defence to keep)
two managers play out weeks ``c+1``-17:

* **hold** -- keeps the unit ranked ``k`` by season level at ``c`` (k = 1: the
  best one; 2, 3, 5: one of the good ones) and starts it every week it plays,
  streaming only its bye;
* **stream** -- starts the best-projected defence on the wire every week.

The wire is what the other managers leave. Two pictures of it:

* **realistic** -- as the 2026 leagues look: rosters hold the top ~60% of the
  team count by ``baseline_points`` (ESPN 6 of 10, Yahoo 8 of 14), and the
  other managers who stream take the week's two (three) best matchups before
  you do;
* **first pick** -- the top N-1 are held but nobody else streams: you get
  the week's best matchup on the wire.

Caveat found doing this: ``baseline_points`` is *this week's* Vegas-only
projection, not a season level, so "the best unit" here is the best matchup
at the checkpoint week.

DECISIONS.md, "Hold a defence or stream it".
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

WALK = "data/backtest/dstk_walkforward.parquet"
CHECKPOINTS = range(3, 11)
LAST = 17
LEAGUES = {"espn": 10, "yahoo": 14}


def actual_points(cfg, seasons: list[int]) -> pd.DataFrame:
    """(season, week, team) -> D/ST points on this league's sheet."""
    from streamer.actuals import dst_game_lines
    from streamer.data.nflverse import cached_games_frame, load_pbp

    games = cached_games_frame()
    out = []
    for s in seasons:
        pbp = load_pbp([s], cfg)
        pbp = pbp[pbp["season_type"].eq("REG")] if "season_type" in pbp.columns else pbp
        g = games[(games["season"] == s) & games["game_type"].eq("REG")]
        lines = dst_game_lines(pbp, g, cfg)
        out.append(lines[["season", "week", "team", "fantasy_points"]])
    return pd.concat(out, ignore_index=True)


def replay(walk: pd.DataFrame, pts: pd.DataFrame, n_teams: int, picture: str, k: int) -> pd.DataFrame:
    held_n = n_teams - 1 if picture == "first pick" else round(0.6 * n_teams)
    ahead = 0 if picture == "first pick" else round(0.2 * n_teams)
    act = pts.set_index(["season", "week", "team"])["fantasy_points"]
    rows = []
    for season in sorted(walk["season"].unique()):
        ws = walk[walk["season"] == season]
        by_week = {w: g.set_index("team") for w, g in ws.groupby("week")}
        for c in CHECKPOINTS:
            if c not in by_week:
                continue
            mine = by_week[c]["baseline_points"].sort_values(ascending=False).index[k - 1]
            hold_pts, stream_pts = [], []
            for w in range(c + 1, LAST + 1):
                g = by_week.get(w)
                if g is None:
                    continue
                level = g["baseline_points"].sort_values(ascending=False)

                def wire(exclude: set, level=level, g=g) -> pd.Series:
                    others = [t for t in level.index if t not in exclude][:held_n]
                    pool = g.drop(index=[t for t in others if t in g.index] + [t for t in exclude if t in g.index])
                    pool = pool["expected_points"].sort_values(ascending=False)
                    return pool.iloc[ahead:]          # the other streamers' picks go first

                # Stream: you hold nobody; the best of what is left this week.
                pool = wire(set())
                pick = pool.index[0]
                stream_pts.append(float(act.get((season, w, pick), np.nan)))
                # Hold: your unit when it plays (others hold the rest of the best), else stream its bye.
                if mine in g.index:
                    hold_pts.append(float(act.get((season, w, mine), np.nan)))
                else:
                    pool = wire({mine})
                    hold_pts.append(float(act.get((season, w, pool.index[0]), np.nan)))
            h, s_ = np.array(hold_pts), np.array(stream_pts)
            ok = np.isfinite(h) & np.isfinite(s_)
            rows.append({"season": season, "checkpoint": c, "unit": mine, "weeks": int(ok.sum()),
                         "hold": h[ok].mean(), "stream": s_[ok].mean(), "diff": (h[ok] - s_[ok]).mean()})
    return pd.DataFrame(rows)


def main() -> None:
    from streamer.config import get_config

    walk = pd.read_parquet(WALK)
    walk = walk[walk["position"] == "DST"]
    seasons = sorted(walk["season"].unique().tolist())
    for lg, n in LEAGUES.items():
        cfg = get_config().for_profile(lg)
        pts = actual_points(cfg, seasons)
        print(f"\n== {lg} ({n} teams), points a week, weeks after the checkpoint to 17 ==")
        for picture in ("realistic", "first pick"):
            for k in (1, 2, 3, 5):
                r = replay(walk, pts, n, picture, k)
                per = r.groupby("season")["diff"].mean()
                print(f"{picture:9s} hold #{k}: hold {r['hold'].mean():5.2f}  stream {r['stream'].mean():5.2f}  "
                      f"hold-stream {r['diff'].mean():+5.2f}  by season "
                      + "  ".join(f"{s}:{v:+.2f}" for s, v in per.items()))


if __name__ == "__main__":
    main()
