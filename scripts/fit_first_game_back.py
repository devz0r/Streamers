"""First game back from an injury: is the projection too high?

    python scripts/fit_first_game_back.py

Players coming off an absence often play limited snaps in their first game.
Replays the production weekly projection (every week 3-15 of 2023-2025) and
splits the players who played by how many of their team's games in a row
they missed before it (0, 1, 2-3, 4+; only those who played earlier in the
season, so a rookie or a new signing is not a "return"). If the projection
runs high for the 2+ groups, a fitted multiplier (on the projection, by
group, picked on two seasons and scored on the third) is tested on the usual
rule: better overall and worse in at most one season. DECISIONS.md, "First
game back".
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

SEASONS = (2023, 2024, 2025)
WEEKS = tuple(range(3, 16))
GROUPS = [("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4+", 4, 99)]


def group_of(n: int) -> str:
    return next(g for g, lo, hi in GROUPS if lo <= n <= hi)


def main() -> int:
    logging.disable(logging.WARNING)
    from streamer.config import get_config
    from streamer.roster import ros_backtest as rb
    from streamer.roster.projections import _missed_in_a_row, games_missed, load_history

    cfg = get_config().for_profile("espn")
    # The baseline is the projection without the correction it now applies.
    cfg.raw.setdefault("roster", {})["first_game_back"] = {}
    hist = load_history(cfg)
    actual = hist.groupby(["season", "week", "player_id"])["fantasy_points_ppr"].sum().rename("actual").reset_index()
    from streamer.teams import normalize_team

    rows = []
    for s in SEASONS:
        rost, inj = rb.rosters(cfg, s), rb.injuries(cfg, s)
        for w in WEEKS:
            team_weeks, played = games_missed(hist, s, w)
            ps = rb.project(hist, cfg, s, w, rb.statuses(rost, inj, w))
            for p in ps:
                if p.projection is None or p.projection < 3 or not p.team:
                    continue
                pw = played.get(str(p.player_id))
                if not pw:
                    continue                      # no game yet this season: not a return
                n = _missed_in_a_row(team_weeks.get(normalize_team(p.team), []), pw)
                rows.append({"season": s, "week": w, "player_id": p.player_id, "position": p.position,
                             "status": p.status, "proj": p.projection,
                             "q": float(p.play_probability or 1.0), "missed": n, "group": group_of(n)})
    d = pd.DataFrame(rows).merge(actual, on=["season", "week", "player_id"])
    # Graded on players who played, so against the number if he plays (the
    # injury-tag discount would otherwise mask the comparison).
    d = d[d.q > 0.05]
    d["proj_if"] = d.proj / d.q
    d["err"] = d.proj_if - d.actual
    print(f"player-weeks that played, projected 3+: {len(d)}")
    print(d.groupby("group").agg(n=("err", "size"), proj=("proj_if", "mean"), actual=("actual", "mean"),
                                 bias=("err", "mean"), mae=("err", lambda e: e.abs().mean())).round(2))
    print("\nby season (bias):")
    print(d.groupby(["group", "season"]).err.mean().unstack().round(2))
    print("\nby position, 2+ missed (bias, n):")
    back = d[d.missed >= 2]
    print(back.groupby("position").agg(bias=("err", "mean"), n=("err", "size")).round(2))
    print("\nratio actual/projection-if-plays (2+ missed):", round(back.actual.sum() / back.proj_if.sum(), 3))
    print("ratio by missed, clean (no tag):", {g: round(x.actual.sum() / x.proj_if.sum(), 3)
                                               for g, x in d[d.status == ""].groupby("group")})

    # Leave one season out: a multiplier for one missed game and one for 2+,
    # each from the other seasons' ratio of actual to the if-he-plays number.
    def mults(frame):
        out = {}
        for key, sel in (("1", frame.missed == 1), ("2+", frame.missed >= 2)):
            g = frame[sel]
            out[key] = float(g.actual.sum() / g.proj_if.sum())
        return out

    out, picks = [], []
    for s in SEASONS:
        m = mults(d[d.season != s])
        picks.append({k: round(v, 3) for k, v in m.items()})
        te = d[d.season == s]
        f = np.where(te.missed >= 2, m["2+"], np.where(te.missed == 1, m["1"], 1.0))
        out.append(te.assign(new=te.proj_if * f))
    o = pd.concat(out)
    by = o.groupby("season").apply(lambda g: pd.Series({"base": g.err.abs().mean(), "new": (g.new - g.actual).abs().mean()}))
    better = int((by.new < by.base).sum())
    ret = o[o.missed >= 1]
    print(f"\nmultipliers per held-out season: {picks}   all seasons: { {k: round(v, 3) for k, v in mults(d).items()} }")
    print(f"MAE {by.base.mean():.4f} -> {by.new.mean():.4f} ({better}/{len(by)} better): "
          + "  ".join(f"{int(s)}: {r.base:.3f}->{r.new:.3f}" for s, r in by.iterrows()))
    print(f"players back from an absence ({len(ret)}): MAE {ret.err.abs().mean():.3f} -> "
          f"{(ret.new - ret.actual).abs().mean():.3f}, bias {ret.err.mean():+.2f} -> {(ret.new - ret.actual).mean():+.2f}")
    print(f"VERDICT: {'KEEP' if by.new.mean() < by.base.mean() and better >= len(by) - 1 else 'DROP'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
