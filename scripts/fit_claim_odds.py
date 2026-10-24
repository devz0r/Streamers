"""How often a rival picks up an unowned player within a week, by league.

    python scripts/fit_claim_odds.py [--dry-run]

From the stored weekly snapshots (data/leagues/<profile>/week_*.json): every
unowned QB/RB/WR/TE in week w, and whether another team had him in week w+1.
A logistic on the platform-wide % rostered, per league (how busy its
managers are, and how much they follow the market, differ: ESPN adds 2% of
its unowned players a week and follows % rostered closely, Yahoo 10% and
barely). Whether he would
start for rivals -- the old guess, 0.5 per rival he would start for -- said
nothing: ESPN players who would start for 3-5 rivals were added 1.6% of the
time. Writes ``src/streamer/roster/claim_odds.json``. DECISIONS.md, "Who
actually gets picked up".
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROFILES = ("espn", "yahoo")
OUT = ROOT / "src" / "streamer" / "roster" / "claim_odds.json"


def events() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from streamer.league.model import LeagueSnapshot

    x, y, k = [], [], []
    for i, prof in enumerate(PROFILES):
        snaps = {}
        for f in glob.glob(str(ROOT / "data" / "leagues" / prof / "week_*.json")):
            s = LeagueSnapshot.load(Path(f))
            snaps[int(s.week)] = s
        for w in sorted(snaps):
            if w + 1 not in snaps:
                continue
            a, b = snaps[w], snaps[w + 1]
            me = a.my_team.team_id if a.my_team else None
            owner = {p.player_id: t.team_id for t in b.teams for p in t.roster}
            for p in a.free_agents:
                if p.position not in ("QB", "RB", "WR", "TE") or not p.team:
                    continue
                x.append(float(p.percent_owned or 0.0) / 100.0)
                y.append(owner.get(p.player_id) not in (None, me))
                k.append(i)
    return np.array(x), np.array(y, float), np.array(k)


def fit(x, y) -> tuple[float, float]:
    from scipy.optimize import minimize

    def nll(b):
        z = b[0] + b[1] * x
        return np.sum(np.logaddexp(0, z) - y * z)

    b = minimize(nll, np.array([-3.0, 3.0])).x
    return float(b[0]), float(b[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    x, y, k = events()
    out = {"leagues": {}, "evidence": {"player_weeks": int(len(x))}}
    for i, prof in enumerate(PROFILES):
        m = k == i
        a, b = fit(x[m], y[m])
        p = 1 / (1 + np.exp(-(a + b * x[m])))
        base = y[m].mean()
        ll = lambda q, yy=y[m]: float(-np.mean(yy * np.log(q) + (1 - yy) * np.log(1 - q)))  # noqa: E731
        print(f"{prof}: {m.sum()} player-weeks, {base:.1%} added; log loss {ll(p):.4f} vs flat "
              f"{ll(np.full(m.sum(), base)):.4f}")
        out["leagues"][prof] = {"intercept": round(a, 3), "slope": round(b, 3), "player_weeks": int(m.sum())}
    print(json.dumps(out))
    if not args.dry_run:
        OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
