"""Does counting games a player was ON THE FIELD but recorded nothing improve
the production model? Held out, 2023-2025.

    python scripts/check_active_zero.py

nflverse's weekly player stats have no row for a skill player who played
offensive snaps and caught, carried and scored nothing, so every trailing
average the model uses skips those games and runs high for backups and
rotational players (weeks 3-5 of 2026: players projected under 3 points were
projected 2.65 by the model alone and scored 1.79). This replays the
production model (``ros_backtest.project``) on the games before each week with
``roster.idle_games`` off and on (the feature switches itself off for the
baseline), and grades both on every snap-active player, a zero counting as
a zero.

Prints aggregates only. ``roster/idle.py``; DECISIONS.md, "Games on the field
with no stat line".
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

SEASONS = (2023, 2024, 2025)
WEEKS = (4, 6, 8, 10, 12, 14)


def main() -> int:
    logging.disable(logging.WARNING)
    from streamer.config import get_config
    from streamer.roster import idle
    from streamer.roster import ros_backtest as rb
    from streamer.roster.projections import load_history

    cfg = get_config().for_profile("espn")
    roster = cfg.raw["roster"]
    was = roster.get("idle_games", False)
    h0 = load_history(cfg)
    roster["idle_games"] = True
    full = idle.with_idle_games(h0, cfg, allow_network=False)
    print(f"history rows {len(h0):,} -> {len(full):,}: {len(full) - len(h0):,} games on the field with no "
          f"stat line ({(len(full) - len(h0)) / len(full):.0%})")
    actual = full.groupby(["season", "week", "player_id"])["fantasy_points_ppr"].sum().rename("actual").reset_index()
    act = idle.active_rows(h0, idle.snap_rows(cfg, SEASONS, allow_network=False)) \
        .drop_duplicates(["player_id", "season", "week"])[["player_id", "season", "week"]]
    rost = {s: rb.rosters(cfg, s) for s in SEASONS}
    inj = {s: rb.injuries(cfg, s) for s in SEASONS}
    rows = []
    try:
        for s in SEASONS:
            for w in WEEKS:
                st = rb.statuses(rost[s], inj[s], w)
                for tag, on in (("old", False), ("new", True)):
                    roster["idle_games"] = on
                    try:
                        ps = rb.project(h0, cfg, s, w, st)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  {tag} {s} wk{w}: {type(exc).__name__}")
                        continue
                    rows.append(pd.DataFrame([{"tag": tag, "season": s, "week": w, "player_id": p.player_id,
                                               "position": p.position, "proj": p.projection, "status": p.status}
                                              for p in ps if p.projection is not None]))
            print(f"  replayed {s}", flush=True)
    finally:
        roster["idle_games"] = was
    d = pd.concat(rows)
    d = d[d["status"].fillna("") == ""]
    d = d.merge(act, on=["player_id", "season", "week"]).merge(actual, on=["season", "week", "player_id"], how="left")
    d["actual"] = d["actual"].fillna(0.0)
    both = d.pivot_table(index=["season", "week", "player_id", "position", "actual"], columns="tag",
                         values="proj").dropna().reset_index()
    print(f"\ngraded: {len(both)} snap-active player-weeks projected both ways")
    for name in ("old", "new"):
        e = both[name] - both["actual"]
        print(f"  {name}: MAE {e.abs().mean():.4f}  bias {e.mean():+.3f}")
    wins = 0
    for s, g in both.groupby("season"):
        wins += (g.new - g.actual).abs().mean() < (g.old - g.actual).abs().mean()
        print(f"    {s}: MAE {(g.old - g.actual).abs().mean():.3f} -> {(g.new - g.actual).abs().mean():.3f}"
              f"   bias {(g.old - g.actual).mean():+.2f} -> {(g.new - g.actual).mean():+.2f}")
    both["bucket"] = pd.cut(both["old"], [0, 3, 6, 9, 12, 60])
    t = both.groupby("bucket", observed=True).apply(lambda g: pd.Series({
        "n": len(g), "old": g.old.mean(), "new": g.new.mean(), "actual": g.actual.mean(),
        "MAE old": (g.old - g.actual).abs().mean(), "MAE new": (g.new - g.actual).abs().mean()}))
    print("\nby the old projection:\n" + t.round(2).to_string())
    for pos, g in both.groupby("position"):
        print(f"  {pos}: n {len(g)}  MAE {(g.old - g.actual).abs().mean():.3f} -> {(g.new - g.actual).abs().mean():.3f}")
    rk = lambda c: both.groupby(["season", "week"]).apply(  # noqa: E731
        lambda g: g[c].rank().corr(g["actual"].rank())).mean()
    print(f"\nrank correlation (mean over weeks): old {rk('old'):.4f}  new {rk('new'):.4f}")
    print(f"VERDICT: better in {int(wins)}/3 seasons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
