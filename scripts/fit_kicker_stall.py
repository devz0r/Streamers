"""Where drives stall: does it add to the kicker model? A held-out test.

    python scripts/fit_kicker_stall.py

A kicker scores when drives die in field-goal range, not just when his team
scores. The model already carries field-goal attempts per drive, red-zone
touchdown rate and field goals' share of scores. This splits attempts per
drive into the three steps behind them, each estimated on its own (shrunk,
leak-free, as every team prior) and against the opponent's defence:

* **range**  -- drives that reach the opponent's 35 (a 52-yard try or
  shorter), per drive;
* **stall**  -- of those, drives that end without a touchdown;
* **kick**   -- of those stalls, how many end in a field-goal try: the
  coach's fourth-down habits and how far he trusts the kicker.

and a **long stall** rate: drives that die between the opponent's 25 and
35, where most 40- and 50-yard tries come from (ESPN scores those at 4-5).

Variants, walk-forward over 2023-2025 exactly as the backtest scores the
shipped model (MAE, rank correlation, top-5 hit rate):

* ``production``  -- today's features;
* ``composite``   -- plus expected tries per game, built from the three
  steps (offence x opposing defence over the league) and the drives;
* ``components``  -- plus the steps themselves and the long-stall rate.

Kept only on the rule every change meets: better overall and worse in at
most one season. DECISIONS.md, "Kicker drive stalls".
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

RANGE = 35          # yardline_100 at or inside: a 52-yard try or shorter
LONG = (25, 35)     # stalls here are 42- to 52-yard tries
KEYS = ["season", "week", "game_id", "team"]
SNAPS = ("pass", "run", "qb_kneel", "qb_spike", "field_goal", "punt")


def drive_counts(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: drives, range drives, range stalls, kicks, long stalls."""
    from streamer.features.team_week import drive_table

    d = drive_table(pbp).rename(columns={"posteam": "team"})
    # Kickoffs (from the 35) and extra points ride on the drive in nflverse;
    # only snaps, tries and punts say where the ball got to.
    plays = pbp[pbp["posteam"].notna() & pbp["fixed_drive"].notna() & pbp["play_type"].isin(SNAPS)]
    plays = plays.assign(y=pd.to_numeric(plays["yardline_100"], errors="coerce")).dropna(subset=["y"])
    g = plays.groupby(["game_id", "posteam", "fixed_drive"])["y"]
    ends = pd.DataFrame({"min_y": g.min(), "end_y": g.last()}).reset_index().rename(columns={"posteam": "team"})
    d = d.merge(ends, on=["game_id", "team", "fixed_drive"], how="inner")
    d["min_yardline"] = d["min_y"]
    kicked = (d["is_fg"] + d["is_missed_fg"]).clip(upper=1)
    reached = (d["min_yardline"] <= RANGE).astype(int)
    stalled = reached * (1 - d["is_td"])
    long_stall = stalled * d["end_y"].between(LONG[0] + 0.5, LONG[1]).astype(int)
    d = d.assign(st_range=reached, st_stall=stalled, st_kick=kicked * stalled, st_long=long_stall,
                 st_drive=1)
    return d.groupby(KEYS, dropna=False)[["st_drive", "st_range", "st_stall", "st_kick", "st_long"]].sum().reset_index()


def stall_features(pbp: pd.DataFrame, games: pd.DataFrame, cfg) -> pd.DataFrame:
    """Leak-free shrunk rates for each team's offence and what its defence allows."""
    from streamer.features.build import team_priors
    from streamer.features.rolling import RateSpec, add_shrunk_rate

    tw = team_priors(pbp, games, cfg, position="K")[[*KEYS, "opponent", "off_drives_per_game",
                                                     "def_drives_per_game_allowed"]]
    c = drive_counts(pbp)
    tw = tw.merge(c, on=KEYS, how="left")
    faced = c.rename(columns={"team": "opponent", **{k: f"f_{k}" for k in c.columns if k.startswith("st_")}})
    tw = tw.merge(faced, on=["season", "week", "game_id", "opponent"], how="left")
    tw = tw.fillna({k: 0.0 for k in tw.columns if k.startswith(("st_", "f_st_"))})
    tw = tw.sort_values(KEYS).reset_index(drop=True)
    k = cfg.team_prior_games("K")
    specs = [("st_range_rate", "st_range", "st_drive"), ("st_stall_rate", "st_stall", "st_range"),
             ("st_kick_rate", "st_kick", "st_stall"), ("st_long_rate", "st_long", "st_drive")]
    for name, num, den in specs:
        tw = add_shrunk_rate(tw, RateSpec(name, num, den, k), ["team"])
        tw = add_shrunk_rate(tw, RateSpec(f"d_{name}", f"f_{num}", f"f_{den}", k), ["team"])
    return tw


def attach(kicker: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Offence from the kicker's team row, the opposing defence from the opponent's."""
    off = st[[*KEYS, "st_range_rate", "st_stall_rate", "st_kick_rate", "st_long_rate"]]
    dfn = st[[*KEYS, "d_st_range_rate", "d_st_stall_rate", "d_st_long_rate"]].rename(columns={"team": "opponent"})
    out = kicker.merge(off, on=KEYS, how="left").merge(dfn, on=["season", "week", "game_id", "opponent"], how="left")
    # League level to scale offence x defence (log5-style): leak-free enough
    # as a constant, since it only rescales one feature the ridge standardises.
    lg = {c: float(st[c].mean()) for c in ("st_range_rate", "st_stall_rate", "st_long_rate")}
    rng = out["st_range_rate"] * out["d_st_range_rate"] / lg["st_range_rate"]
    stall = out["st_stall_rate"] * out["d_st_stall_rate"] / lg["st_stall_rate"]
    drives = (out["off_drives_per_game"] + out["def_drives_per_game_allowed"]) / 2.0
    out["stall_exp_fga"] = drives * rng * stall * out["st_kick_rate"]
    out["stall_long"] = drives * out["st_long_rate"] * out["d_st_long_rate"] / lg["st_long_rate"] * out["st_kick_rate"]
    out["stall_implied_x_fga"] = out["team_implied_total"] * out["stall_exp_fga"]
    return out


def main() -> int:
    logging.disable(logging.WARNING)
    from streamer.backtest import score_predictions, walk_forward
    from streamer.config import get_config
    from streamer.data.nflverse import games_frame, load_pbp
    from streamer.features.build import KICKER_FEATURES, build_kicker_features
    from streamer.models.positions import KickerModel
    from streamer.pipeline import training_seasons

    cfg = get_config().for_profile("espn")
    pbp, games = load_pbp(training_seasons(cfg), cfg), games_frame(cfg)
    kicker = build_kicker_features(pbp, games, cfg)
    if "def_drives_per_game_allowed" not in kicker.columns or "off_drives_per_game" not in kicker.columns:
        raise SystemExit("kicker frame lacks drive columns")
    k = attach(kicker, stall_features(pbp, games, cfg))
    miss = k[["stall_exp_fga", "st_kick_rate"]].isna().mean()
    print(f"kicker rows {len(k)}; missing stall features {miss.max():.3%}")
    played = k[k["fantasy_points"].notna()]
    for col in ("off_fga_per_drive", "stall_exp_fga", "stall_long", "st_kick_rate", "st_range_rate"):
        print(f"  corr with points: {col:20} {played[col].corr(played['fantasy_points']):+.4f}")
    print(f"  corr with fg tries: stall_exp_fga {played['stall_exp_fga'].corr(played['fg_att']):+.4f}  "
          f"off_fga_per_drive {played['off_fga_per_drive'].corr(played['fg_att']):+.4f}")

    variants = {
        "production": KICKER_FEATURES,
        "composite": (*KICKER_FEATURES, "stall_exp_fga", "stall_implied_x_fga"),
        "components": (*KICKER_FEATURES, "stall_exp_fga", "stall_implied_x_fga", "stall_long",
                       "st_range_rate", "st_stall_rate", "st_kick_rate",
                       "d_st_range_rate", "d_st_stall_rate"),
    }
    k = k.fillna({c: k[c].mean() for c in variants["components"] if c.startswith(("st_", "d_st_", "stall_"))})
    res = {}
    for name, feats in variants.items():
        KickerModel.FEATURES = tuple(feats)
        preds = walk_forward(k, "K", [2023, 2024, 2025], cfg)
        _weekly, summary = score_predictions(preds, cfg)
        res[name] = summary.set_index("season")
        print(f"\n{name:11} MAE {summary.mae.mean():.4f}  rank corr {summary.rank_corr.mean():.4f}  "
              f"top-5 hit {summary.top5_hit_rate.mean():.4f}")
        for r in summary.itertuples():
            print(f"     {r.season}: MAE {r.mae:.4f}  rank corr {r.rank_corr:.4f}  top5 {r.top5_hit_rate:.3f}")
    KickerModel.FEATURES = KICKER_FEATURES
    base = res["production"]
    for name in ("composite", "components"):
        r = res[name]
        mae_better = int((r["mae"] < base["mae"]).sum())
        rc_better = int((r["rank_corr"] > base["rank_corr"]).sum())
        overall = r["mae"].mean() < base["mae"].mean()
        ok = overall and mae_better >= len(r) - 1 and r["rank_corr"].mean() >= base["rank_corr"].mean()
        print(f"VERDICT {name}: MAE better {mae_better}/{len(r)}, rank corr better {rc_better}/{len(r)} "
              f"-> {'KEEP' if ok else 'DROP'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
