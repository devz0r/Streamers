"""Grade the weekly forecast logs (team_weeks_log, game_odds_log) against real scores.

    python scripts/grade_forecasts.py [--profile espn|yahoo|both]

Only the last forecast made before each week's first kickoff is graded (every run is kept
in the logs). A team's score for week w is its season points-for in the snapshot taken in
week w+1 less the one taken in week w. Prints, by how many weeks ahead the
forecast was made, the miss of each team's mean (and the spread of
miss / sd, 1 when the spread is right) and each game's log loss, Brier score
and calibration of the win chance. Aggregates only. DECISIONS.md, "Weekly
forecasts are logged".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="both", choices=("espn", "yahoo", "both"))
    args = ap.parse_args()
    from streamer.config import get_config
    from streamer.league.store import load_snapshot, snapshot_dir
    from streamer.roster import odds_log

    for prof in ("espn", "yahoo") if args.profile == "both" else (args.profile,):
        cfg = get_config().for_profile(prof)
        weeks = sorted(int(p.stem.split("_")[1]) for p in snapshot_dir(cfg).glob("week_*.json")
                       if p.stem.split("_")[1].isdigit())
        if not weeks:
            print(f"{prof}: no snapshots")
            continue
        season = load_snapshot(cfg, weeks[-1]).season
        pts = {}
        for w in weeks:
            s = load_snapshot(cfg, w)
            if s.season == season:
                pts[w] = {str(t.team_id): float(t.points_for) for t in s.teams}
        g = odds_log.grade_weeks(cfg, odds_log.week_scores(pts), int(season), odds_log.first_kickoffs(cfg, int(season)))
        if g is None:
            print(f"{prof}: nothing logged has been played yet (needs the snapshot after a forecast's week)")
            continue
        print(f"{prof} {season}")
        for t in g["teams"]:
            z = "" if t["z_sd"] is None else f", sd of miss/sd {t['z_sd']:.2f}"
            print(f"  team weeks, {t['ahead']} ahead (n={t['n']}): mean off by {t['bias']:+.1f} on average, "
                  f"{t['mae']:.1f} absolute{z}")
        for x in g["games"]:
            print(f"  games, {x['ahead']} ahead (n={x['n']}): log loss {x['log_loss']:.4f} (coin flip 0.6931), "
                  f"Brier {x['brier']:.4f}; favourites said {x['favourite_said']:.1%}, won {x['favourite_won']:.1%}")
        for b in g["buckets"]:
            print(f"    favourite {b['range']} (n={b['n']}): said {b['said']:.1%}, happened {b['happened']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
