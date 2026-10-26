"""Does a change of quarterback move his receivers more than the line says?

    python scripts/fit_qb_change.py [--cache rows.parquet]

Replays 2022-2025, weeks 3-18, through the production projection for backs,
receivers and tight ends: the volume model (``projections.player_table``)
times this week's implied team total over the team's trailing one, damped
as in production (``vegas_damping``) -- so a backup's lower team total is
already priced. Each team-week's starter is the quarterback with the most
pass attempts. A player's own quarterback is the starter of his recent games
for the team (up to six, this season and last); ``q`` is the share of those
started by this week's starter.

Quarterback quality is passing EPA per attempt over every earlier game,
shrunk toward the league mean by ``SHRINK`` attempts. The factor tried is
``(1 + gap) ** b`` with ``gap`` this week's starter's quality less his recent
quarterbacks' (per attempt, scaled to the league spread), weighted by the
games he did not start: ``(1 - q)``. ``b`` per position is picked on three
seasons and scored on the fourth, on the player-weeks with a change; kept
only if better overall and worse in at most one season. DECISIONS.md, "A
new quarterback for the receivers".
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
POSITIONS = ("WR", "TE", "RB")
SHRINK = 200.0
RECENT = 6
BS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def starters(cfg) -> pd.DataFrame:
    """(season, week, team) -> starter id, his attempts and passing EPA."""
    from streamer.teams import normalize_team

    frames = []
    for s in range(2021, 2026):
        st = pd.read_parquet(cfg.raw_dir / f"player_stats_{s}.parquet")
        st = st[(st["position"] == "QB") & (st["season_type"] == "REG") & (st["attempts"] > 0)]
        frames.append(st[["season", "week", "team", "player_id", "player_display_name", "attempts", "passing_epa"]])
    q = pd.concat(frames)
    q["team"] = q["team"].map(normalize_team)
    return q


def quality(q: pd.DataFrame, season: int, week: int) -> tuple[dict[str, float], float, float]:
    """QB id -> shrunk EPA per attempt before (season, week); league mean and spread."""
    past = q[(q["season"] < season) | ((q["season"] == season) & (q["week"] < week))]
    g = past.groupby("player_id")[["attempts", "passing_epa"]].sum()
    mu = float(g["passing_epa"].sum() / g["attempts"].sum())
    rate = (g["passing_epa"] + SHRINK * mu) / (g["attempts"] + SHRINK)
    spread = float(rate[g["attempts"] >= 150].std())
    return rate.to_dict(), mu, spread


def build(cfg) -> pd.DataFrame:
    from streamer.data.nflverse import games_frame
    from streamer.roster import projections as pj
    from streamer.teams import normalize_team

    h = pj.load_history(cfg)
    h = h.assign(_team=h["team"].map(normalize_team))
    q = starters(cfg)
    start = (q.sort_values("attempts").groupby(["season", "week", "team"]).tail(1)
             .set_index(["season", "week", "team"])["player_id"])
    games = games_frame(cfg)
    games = games[games["game_type"] == "REG"].assign(team=lambda g: g["team"].map(normalize_team))
    damp = float(cfg.raw["roster"]["vegas_damping"])
    rows = []
    for season in (2022, 2023, 2024, 2025):
        for week in range(3, 19):
            cur = h[(h["season"] == season) & (h["week"] == week)]
            if cur.empty:
                continue
            rate, mu, spread = quality(q, season, week)
            past_g = games[(games["season"] < season) | ((games["season"] == season) & (games["week"] < week))]
            trail = (past_g.dropna(subset=["team_implied_total"]).sort_values(["season", "week"])
                     .groupby("team")["team_implied_total"].apply(lambda s: float(s.tail(6).mean())))
            now = games[(games["season"] == season) & (games["week"] == week)].set_index("team")["team_implied_total"]
            t = pj.player_table(h, season, week, cfg)
            got = cur.groupby("player_id")["fantasy_points_ppr"].sum()
            team_now = cur.groupby("player_id")["_team"].first()
            t = t[t["player_id"].isin(got.index) & t["position"].isin(POSITIONS) & t["blend"].notna()]
            prior = h[((h["season"] == season) & (h["week"] < week)) | (h["season"] == season - 1)]
            for r in t.itertuples():
                team = team_now[r.player_id]
                qb = start.get((season, week, team))
                if qb is None or not np.isfinite(now.get(team, np.nan)) or team not in trail:
                    continue
                mine = prior[(prior["player_id"] == r.player_id) & (prior["_team"] == team)] \
                    .sort_values(["season", "week"]).tail(RECENT)
                qbs = [start.get((s, w, team)) for s, w in zip(mine["season"], mine["week"])]
                qbs = [x for x in qbs if x is not None]
                if not qbs:
                    continue
                share = sum(x == qb for x in qbs) / len(qbs)
                old_q = float(np.mean([rate.get(x, mu) for x in qbs]))
                scale = float(np.clip(now[team] / trail[team], 0.7, 1.3)) ** damp
                rows.append({"season": season, "week": week, "position": r.position, "player_id": r.player_id,
                             "team": team, "pred": float(r.blend) * scale, "vol": float(r.blend), "pts": float(got[r.player_id]),
                             "q": share, "gap": (rate.get(qb, mu) - old_q) / spread,
                             "new_attempts": float(q[(q["player_id"] == qb) & ((q["season"] < season) | (
                                 (q["season"] == season) & (q["week"] < week)))]["attempts"].sum())})
        print(season, len(rows), flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache")
    ap.add_argument("--no-line", action="store_true",
                    help="grade against the volume model alone, without this week's implied total: "
                         "the whole effect, for the season simulation's weeks to come")
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
    if args.no_line:
        d["pred"] = d["vol"]
    d["x"] = (1.0 - d["q"]) * d["gap"]

    def pred(g, b):
        return g["pred"] * np.clip(1.0 + g["x"], 0.3, 2.0) ** b

    def rmse(g, b):
        return float(np.sqrt(((g["pts"] - pred(g, b)) ** 2).mean()))

    ch = d[d["q"] < 0.5]
    print(f"\n{len(d)} player-weeks; {len(ch)} with a new starter (most recent games by someone else)")
    for pos in POSITIONS:
        g = ch[ch["position"] == pos]
        worse = g[g["gap"] < -0.5]
        print(f"\n{pos}: {len(g)} changed weeks; bias (pts - pred) all changed {(g['pts'] - g['pred']).mean():+.2f}, "
              f"worse QB by 0.5+ sd: {(worse['pts'] - worse['pred']).mean():+.2f} (n={len(worse)}), "
              f"unchanged {(d[(d['position'] == pos) & (d['q'] >= 0.99)]['pts'] - d[(d['position'] == pos) & (d['q'] >= 0.99)]['pred']).mean():+.2f}")
        moved = d[(d["position"] == pos) & (d["x"].abs() > 1e-9)]
        wins, t0, t1 = 0, 0.0, 0.0
        for s in sorted(moved["season"].unique()):
            tr, te = moved[moved["season"] != s], moved[moved["season"] == s]
            b = min(BS, key=lambda b: rmse(tr, b))
            a0, a1 = rmse(te, 0.0), rmse(te, b)
            wins += a1 < a0
            t0 += a0 ** 2 * len(te)
            t1 += a1 ** 2 * len(te)
            print(f"   {s}: n={len(te)} b={b} RMSE {a0:.3f} -> {a1:.3f}  bias {(te['pts'] - pred(te, 0)).mean():+.2f} -> "
                  f"{(te['pts'] - pred(te, b)).mean():+.2f}")
        o0, o1 = np.sqrt(t0 / len(moved)), np.sqrt(t1 / len(moved))
        best = min(BS, key=lambda b: rmse(moved, b))
        keep = o1 < o0 and wins >= 3
        print(f"   overall {o0:.3f} -> {o1:.3f}, better in {wins} of 4, all-season b={best}: {'KEEP' if keep else 'drop'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
