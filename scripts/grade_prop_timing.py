"""Are fresher prop prices better? Early against late, from the log's history.

    python scripts/grade_prop_timing.py [week ...]

The projection log (``results/<league>/skill_log.parquet``) keeps one row per
player-week, overwritten until kickoff, but every refresh commits it, so git
history holds each pull. For every player-week priced at least 12 hours apart,
the first and last market number before his kickoff are scored against his
points on the league's sheet (RMSE, with a bootstrap interval on the gain).
DECISIONS.md, "When to buy the props".
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pulls(league: str) -> pd.DataFrame:
    path = f"results/{league}/skill_log.parquet"
    hashes = subprocess.run(["git", "log", "--format=%h", "--", path], capture_output=True, text=True,
                            check=True).stdout.split()
    frames = []
    for h in hashes:
        blob = subprocess.run(["git", "show", f"{h}:{path}"], capture_output=True, check=False).stdout
        if blob:
            d = pd.read_parquet(io.BytesIO(blob))
            frames.append(d[d["vegas_points"].notna()])
    a = pd.concat(frames, ignore_index=True)
    return a.drop_duplicates(["season", "week", "player_id", "logged_at"])


def kickoffs(season: int, week: int) -> dict:
    from streamer.teams import normalize_team

    s = pd.read_parquet("data/raw/schedules.parquet")
    s = s[(s["season"] == season) & (s["week"] == week)]
    out = {}
    for r in s.itertuples():
        t = pd.Timestamp(f"{r.gameday} {r.gametime}").tz_localize("America/New_York").tz_convert("UTC")
        out[normalize_team(r.home_team)] = out[normalize_team(r.away_team)] = t
    return out


def main(weeks: list[int]) -> None:
    from streamer.config import get_config
    from streamer.roster.projections import load_history
    from streamer.teams import normalize_team

    for lg in ("espn", "yahoo"):
        a = pulls(lg)
        hist = load_history(get_config().for_profile(lg))
        for (season, week), d in a.groupby(["season", "week"]):
            if weeks and week not in weeks:
                continue
            ko = kickoffs(int(season), int(week))
            d = d.assign(ko=d["team"].map(lambda t, ko=ko: ko.get(normalize_team(t))),
                         ts=pd.to_datetime(d["logged_at"], utc=True)).dropna(subset=["ko"])
            d = d[d["ts"] < d["ko"]].assign(hrs=lambda x: (x["ko"] - x["ts"]).dt.total_seconds() / 3600)
            h = hist[(hist["season"] == season) & (hist["week"] == week)][["player_id", "fantasy_points_ppr"]]
            d = d.merge(h.rename(columns={"player_id": "nfl_id"}), on="nfl_id").sort_values("ts")
            m = d.groupby("player_id").head(1).merge(d.groupby("player_id").tail(1), on="player_id",
                                                      suffixes=("_f", "_l"))
            m = m[(m["hrs_f"] - m["hrs_l"]) >= 12]
            if len(m) < 30:
                continue
            y = m["fantasy_points_ppr_l"].to_numpy()
            ef, el = (m["vegas_points_f"].to_numpy() - y) ** 2, (m["vegas_points_l"].to_numpy() - y) ** 2
            rng = np.random.default_rng(0)
            boot = [np.sqrt(ef[i].mean()) - np.sqrt(el[i].mean())
                    for i in (rng.integers(0, len(m), len(m)) for _ in range(2000))]
            print(f"{lg} {season} wk{week}: n {len(m)}, hours before kickoff {m['hrs_f'].median():.0f} -> "
                  f"{m['hrs_l'].median():.0f}; market RMSE {np.sqrt(ef.mean()):.2f} -> {np.sqrt(el.mean()):.2f} "
                  f"(gain {np.sqrt(ef.mean()) - np.sqrt(el.mean()):+.2f}, 90% {np.percentile(boot, 5):+.2f} to "
                  f"{np.percentile(boot, 95):+.2f})")


if __name__ == "__main__":
    main([int(w) for w in sys.argv[1:]])
