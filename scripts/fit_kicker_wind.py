"""Kicker wind: what the model is given for a game whose weather is unknown.

    python scripts/fit_kicker_wind.py

The kicker model trains on the wind actually recorded at each game, but a
game still to be played has no recorded wind: production fills it with
``weather.default_wind_mph`` (0, calm). Walk-forward over 2023-2025, the
model predicts each week with its test games' wind set to

* ``actual``  -- what was recorded (not knowable ahead; the ceiling),
* ``zero``    -- calm, production's fill today,
* ``typical`` -- the mean wind of the home team's outdoor games over the
  previous two seasons and this one so far (known ahead),

domes neutral in all. Scored as the backtest does (MAE, rank correlation,
top-5 hit rate), and on outdoor games only. DECISIONS.md, "Kicker wind".
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")


def typical_wind(frame: pd.DataFrame, train: pd.DataFrame, season: int, week: int) -> pd.Series:
    """Home team -> mean recorded wind at its outdoor games in earlier games."""
    past = train[(train["season"] >= season - 2) & (train["is_dome"] == 0) & train["wind"].notna()]
    one = past.drop_duplicates(["season", "week", "home_team"])
    return one.groupby("home_team")["wind"].mean()


def main() -> int:
    logging.disable(logging.WARNING)
    from streamer.backtest import score_predictions, walk_forward
    from streamer.config import get_config
    from streamer.data.nflverse import games_frame, load_pbp
    from streamer.features.build import build_kicker_features
    from streamer.pipeline import training_seasons

    cfg = get_config().for_profile("espn")
    kicker = build_kicker_features(load_pbp(training_seasons(cfg), cfg), games_frame(cfg), cfg)
    print(f"kicker rows {len(kicker)}; outdoor mean wind {kicker[kicker.is_dome == 0].wind.mean():.1f} mph")

    def variant(name):
        def fn(test, train):
            t = test.copy()
            season, week = int(t["season"].iloc[0]), int(t["week"].iloc[0])
            outdoor = t["is_dome"] == 0
            if name == "zero":
                t.loc[outdoor, "wind"] = 0.0
            elif name == "typical":
                typ = typical_wind(t, train, season, week)
                t.loc[outdoor, "wind"] = t.loc[outdoor, "home_team"].map(typ).fillna(typ.mean()).to_numpy()
            t["wind_x_long_share"] = t["wind"] * t["k_long_att_share"]
            return t
        return fn

    results = {}
    for name in ("actual", "zero", "typical"):
        preds = walk_forward(kicker, "K", [2023, 2024, 2025], cfg, transform_test=variant(name))
        weekly, summary = score_predictions(preds, cfg)
        results[name] = (preds, weekly, summary)
        out = preds[preds["is_dome"] == 0]
        err = (out["expected_points"] - out["fantasy_points"])
        print(f"\n{name:8} all games: MAE {summary.mae.mean():.4f}  rank corr {summary.rank_corr.mean():.4f}  "
              f"top-5 hit {summary.top5_hit_rate.mean():.4f}   outdoor bias {err.mean():+.3f} (n={len(out)})")
        for r in summary.itertuples():
            print(f"     {r.season}: MAE {r.mae:.4f}  rank corr {r.rank_corr:.4f}  top5 {r.top5_hit_rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
