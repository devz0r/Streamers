"""Fit and grade the scale on this week's absence risk for untagged starters.

    python scripts/fit_this_week_absence.py

2022-2025: every skill player who played a week and was not listed Out,
Doubtful or Questionable on his next game's final injury report; whether he
played that game, against the season simulation's weekly absence hazard at
his level (his trailing four-game points). For players at ``from_level`` and
up the hazard is scaled by a factor fitted on three seasons and scored on the
fourth (log loss), for three levels. DECISIONS.md, "This week's absences for
players with no tag"; ``futures.this_week`` in outcome_model.json.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
SEASONS = (2022, 2023, 2024, 2025)


def cases(cfg) -> pd.DataFrame:
    from streamer.roster.projections import load_history

    h = load_history(cfg)
    h = h[h.season.isin(SEASONS) & h.position.isin(["QB", "RB", "WR", "TE"])].sort_values(
        ["player_id", "season", "week"]).copy()
    h["trail"] = h.groupby(["player_id", "season"]).fantasy_points_ppr.transform(lambda s: s.rolling(4, 1).mean())
    sch = pd.read_parquet(cfg.raw_dir / "schedules.parquet")
    sch = sch[sch.season.isin(SEASONS) & (sch.game_type == "REG")]
    weeks: dict = {}
    for r in sch.itertuples():
        for t in (r.home_team, r.away_team):
            weeks.setdefault((r.season, t), []).append(r.week)
    inj = pd.concat([pd.read_parquet(cfg.raw_dir / f"injuries_{s}.parquet") for s in SEASONS])
    tagged = {(r.season, str(r.gsis_id), r.week) for r in inj.itertuples()
              if r.report_status in ("Out", "Doubtful", "Questionable")}
    played = {(r.season, str(r.player_id), r.week) for r in h.itertuples()}
    rows = []
    for r in h.itertuples():
        nxt = [w for w in sorted(weeks.get((r.season, r.team), [])) if w > r.week]
        if not nxt or (r.season, str(r.player_id), nxt[0]) in tagged:
            continue
        rows.append((r.season, r.position, r.trail, float((r.season, str(r.player_id), nxt[0]) not in played)))
    return pd.DataFrame(rows, columns=["season", "pos", "trail", "y"])


def main() -> int:
    logging.disable(logging.WARNING)
    from streamer.config import get_config
    from streamer.roster import futures as fut

    d = cases(get_config().for_profile("espn"))
    conf = {k: v for k, v in fut._params().items() if k != "this_week"}
    d["model"] = 0.0
    for p, g in d.groupby("pos"):
        d.loc[g.index, "model"] = np.minimum(fut._hazard(p, g.trail.to_numpy(), conf), 1)

    def ll(p, y):
        p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
        y = np.asarray(y, float)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    grid = np.round(np.arange(0.2, 1.21, 0.05), 2)
    for cut in (8.0, 10.0, 12.0):
        rows = []
        for s in SEASONS:
            tr, te = d[(d.season != s) & (d.trail >= cut)], d[d.season == s]
            m = grid[int(np.argmin([ll(tr.model * k, tr.y) for k in grid]))]
            new = np.where(te.trail >= cut, te.model * m, te.model)
            rows.append((s, m, ll(te.model, te.y), ll(new, te.y)))
        r = pd.DataFrame(rows, columns=["season", "scale", "as_is", "scaled"])
        hi = d[d.trail >= cut]
        m_all = grid[int(np.argmin([ll(hi.model * k, hi.y) for k in grid]))]
        print(f"from {cut:g} a game: better in {int((r.scaled < r.as_is).sum())} of 4; log loss {r.as_is.mean():.5f} -> "
              f"{r.scaled.mean():.5f}; scale by fold {list(r.scale)}; on all four {m_all}; "
              f"missed {hi.y.mean():.3%} against {hi.model.mean():.3%} drawn (n={len(hi)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
