"""Check the page's live scoring against a finished week.

    python scripts/live_check.py [--week N]

Reads every game of a completed week from ESPN's public box scores (the same
endpoints the page reads), scores every player with ``static/live.js`` (run
in Node) under the ESPN profile's rules, and compares: skill players against
nflverse's PPR points (the same offensive scoring), kickers and defences
against what ``streamer update`` recorded for that week. Prints agreement
only -- counts and differences, never a name or a score. Run as the
``live-check`` job (ESPN cannot be reached from everywhere).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/"
LIVE_JS = Path(__file__).resolve().parents[1] / "src" / "streamer" / "static" / "live.js"

DRIVER = r"""
const L = require(process.argv[1]);
const fs = require('fs');
const {sc, teams, games} = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = {skill: {}, k: {}, dst: {}};
for (const g of games) {
  const box = L.boxStats(g.summary, teams), plays = L.scoringPlays(g.summary, teams), two = L.twoPoints(plays);
  for (const tm of Object.keys(box)) {
    for (const [id, s] of Object.entries(box[tm].ath)) {
      const name = Object.keys(box[tm].names).find(n => box[tm].names[n] === id) || '';
      out.skill[id] = L.skillPoints(s, sc.off, two[name]);
      if (s['kicking.fieldGoalsMade/fieldGoalAttempts'] !== undefined) out.k[tm] = L.kickerPoints(s, plays, tm, sc.k);
    }
    const opp = g.home === tm ? g.away : g.home;
    out.dst[tm] = L.dstPoints(box, plays, tm, opp, g.score[opp] || 0, sc.dst);
  }
}
console.log(JSON.stringify(out));
"""


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def agree(ours: pd.Series, theirs: pd.Series, label: str) -> None:
    both = pd.concat([ours.rename("ours"), theirs.rename("theirs")], axis=1).dropna()
    if both.empty:
        print(f"{label}: nothing to compare")
        return
    d = (both.ours - both.theirs).abs()
    print(f"{label}: {len(both)} compared, exact (within 0.05) {np.mean(d <= 0.05):.1%}, "
          f"within 1 point {np.mean(d <= 1.0):.1%}, mean miss {d.mean():.2f}, worst {d.max():.1f}")


def main() -> int:
    from streamer.config import get_config
    from streamer.teams import normalize_team

    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int)
    args = ap.parse_args()
    cfg = get_config().for_profile("espn")
    season = int(cfg.raw["season"]["current"]) if "current" in cfg.raw["season"] else 2026
    hist = pd.read_parquet(cfg.results_dir / "history.parquet")
    week = args.week or int(hist[hist.season == season].week.max())
    sb = get(f"{BASE}scoreboard?seasontype=2&week={week}&dates={season}")
    alias = {"LAR": "LA", "WSH": "WAS"}
    games, names = [], set()
    for e in sb.get("events", []):
        if (e.get("status", {}).get("type", {}) or {}).get("state") != "post":
            continue
        comp = e["competitions"][0]
        g = {"id": e["id"], "score": {}}
        for c in comp["competitors"]:
            tm = alias.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
            g["score"][tm] = float(c.get("score") or 0)
            g["home" if c["homeAway"] == "home" else "away"] = tm
        g["summary"] = get(f"{BASE}summary?event={e['id']}")
        for t in g["summary"].get("boxscore", {}).get("teams", []):
            names.update(s.get("name") for s in t.get("statistics", []))
        games.append(g)
    print(f"week {week}: {len(games)} finished games")
    print("team stat names present:", sorted(n for n in names if n in ("fumblesLost", "totalYards", "fourthDownEff")),
          "of fumblesLost, totalYards, fourthDownEff")
    prof = cfg.profiles["espn"]
    sc = {"off": prof["offense_scoring"], "k": prof["kicker_scoring"], "dst": prof["dst_scoring"]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"sc": sc, "teams": alias, "games": games}, f)
    run = subprocess.run(["node", "-e", DRIVER, str(LIVE_JS), f.name], capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        print("node failed:", run.stderr[-500:])
        return 1
    ours = json.loads(run.stdout)

    # Skill players: nflverse PPR, matched through the roster table's ESPN ids.
    ps = pd.read_parquet(cfg.raw_dir / f"player_stats_{season}.parquet")
    ps = ps[(ps.week == week) & (ps.season_type == "REG") & ps.position.isin(["QB", "RB", "WR", "TE"])]
    path = cfg.raw_dir / f"rosters_weekly_{season}.parquet"
    if path.exists():
        ro = pd.read_parquet(path, columns=["gsis_id", "espn_id"])
    else:
        import nflreadpy

        ro = nflreadpy.load_rosters_weekly(seasons=[season]).to_pandas()[["gsis_id", "espn_id"]]
    ro = ro.dropna()
    espn = dict(zip(ro.gsis_id, ro.espn_id.astype(str).str.split(".").str[0]))
    theirs = pd.Series({espn[g]: v for g, v in zip(ps.player_id, ps.fantasy_points_ppr) if g in espn})
    agree(pd.Series(ours["skill"], dtype=float), theirs, "skill players vs nflverse PPR")

    wk = hist[(hist.season == season) & (hist.week == week)].copy()
    wk["team"] = wk.team.map(normalize_team)
    for pos, key in (("K", "k"), ("DST", "dst")):
        rec = wk[wk.position == pos].set_index("team").actual_points
        agree(pd.Series(ours[key], dtype=float), rec, f"{pos} vs our recorded actuals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
