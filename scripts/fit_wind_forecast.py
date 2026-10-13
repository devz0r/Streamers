"""Calibrate and test a weather-forecast wind for kickers. Runs where Open-Meteo is reachable.

    python scripts/fit_wind_forecast.py [--write]

For every outdoor game 2022-2025 with a recorded wind, the forecast Open-Meteo
issued for the kickoff hours (its historical-forecast archive: forecasts as
issued, not hindsight; one request per stadium and season), averaged over
three hours from kickoff. Then:

1. recorded = a + b x forecast, fitted leave-one-season-out;
2. kickers, walk-forward 2023-2025, the test games' wind set to the stadium's
   typical wind (production today) against the calibrated forecast (blended
   with typical as for a game a day away, since the archive holds only
   near-term forecasts), plus the same on windy games (typical 8+ mph or
   forecast 12+);
3. the verdict by the usual rule: forecast kept only if better than typical
   on MAE and rank correlation in at least two of three seasons.

Prints aggregates only. ``--write`` stores ``a``, ``b`` (fitted on all four
seasons) in ``streamer/data/wind_forecast.json`` when the rule passes.
DECISIONS.md, "Kicker wind forecast".
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")


def forecasts(games: pd.DataFrame) -> pd.DataFrame:
    """Archived kickoff-hour forecast for each outdoor game."""
    from streamer.data import forecast_wind as fw

    rows = []
    for (season, home), g in games.groupby(["season", "home_team"]):
        where = fw.venue(home, g["stadium"].iloc[0])
        if where is None:
            continue
        kos = {r.game_id: fw.kickoff_utc(str(r.gameday), r.gametime) for r in g.itertuples()}
        start, end = min(kos.values()).strftime("%Y-%m-%d"), (max(kos.values()) + pd.Timedelta(hours=4)).strftime("%Y-%m-%d")
        try:
            payload = fw.fetch(where[0], where[1], start, end, archive=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  {season} {home}: {type(exc).__name__}", flush=True)
            continue
        for r in g.itertuples():
            got = fw.kickoff_wind(payload, kos[r.game_id])
            if got:
                rows.append({"game_id": r.game_id, "season": season, "week": r.week, "home_team": home,
                             "fc_wind": got[0], "fc_gust": got[1]})
        time.sleep(0.3)
    return pd.DataFrame(rows)


def main() -> int:
    logging.disable(logging.WARNING)
    from streamer.backtest import score_predictions, walk_forward
    from streamer.config import get_config
    from streamer.data import forecast_wind as fw
    from streamer.data.nflverse import games_frame, load_pbp
    from streamer.features.build import build_kicker_features
    from streamer.pipeline import training_seasons

    cfg = get_config().for_profile("espn")
    g = games_frame(cfg)
    g = g[(g["game_type"] == "REG") & (g["season"] >= 2022) & (g["season"] <= 2025)].drop_duplicates("game_id")
    g = g[g["is_dome"].fillna(0).astype(int).eq(0) & g["wind"].notna()]
    print(f"{len(g)} outdoor games with recorded wind")
    fc = forecasts(g).merge(g[["game_id", "wind"]], on="game_id")
    print(f"{len(fc)} with an archived forecast; recorded mean {fc.wind.mean():.2f} mph, forecast mean {fc.fc_wind.mean():.2f}; "
          f"correlation {fc.wind.corr(fc.fc_wind):.3f}")

    def fit(d):
        b, a = np.polyfit(d["fc_wind"], d["wind"], 1)
        return float(a), float(b)

    fc["cal"] = np.nan
    for s in sorted(fc.season.unique()):
        a, b = fit(fc[fc.season != s])
        fc.loc[fc.season == s, "cal"] = a + b * fc.loc[fc.season == s, "fc_wind"]
    print(f"leave-one-season-out RMSE vs recorded: raw forecast {np.sqrt(((fc.fc_wind - fc.wind) ** 2).mean()):.2f}, "
          f"calibrated {np.sqrt(((fc.cal - fc.wind) ** 2).mean()):.2f}")

    kicker = build_kicker_features(load_pbp(training_seasons(cfg), cfg), games_frame(cfg), cfg)
    look = fc.set_index(["season", "week", "home_team"])["cal"]
    cal_all = dict(zip(("a", "b"), fit(fc)))

    def variant(use_forecast):
        def fn(test, train):
            t = test.copy()
            season, week = int(t["season"].iloc[0]), int(t["week"].iloc[0])
            outdoor = t["is_dome"] == 0
            past = train[(train["season"] >= season - 2) & (train["is_dome"] == 0) & train["wind"].notna()]
            typ = past.drop_duplicates(["season", "week", "home_team"]).groupby("home_team")["wind"].mean()
            wind = t.loc[outdoor, "home_team"].map(typ).fillna(typ.mean())
            if use_forecast:
                w = fw.lead_weight(1.0, {})
                got = pd.Series([look.get((season, week, h), np.nan) for h in t.loc[outdoor, "home_team"]], index=wind.index)
                wind = (w * got + (1 - w) * wind).where(got.notna(), wind)
            t.loc[outdoor, "wind"] = wind.to_numpy()
            t["wind_x_long_share"] = t["wind"] * t["k_long_att_share"]
            return t
        return fn

    res = {}
    for name, use in (("typical", False), ("forecast", True)):
        preds = walk_forward(kicker, "K", [2023, 2024, 2025], cfg, transform_test=variant(use))
        _, summary = score_predictions(preds, cfg)
        res[name] = (preds, summary.set_index("season"))
        print(f"{name:9} MAE {summary.mae.mean():.4f}  rank corr {summary.rank_corr.mean():.4f}  top-5 hit {summary.top5_hit_rate.mean():.4f}")
    a, b = res["typical"][1], res["forecast"][1]
    wins = 0
    for s in a.index:
        ok = b.loc[s, "mae"] < a.loc[s, "mae"] and b.loc[s, "rank_corr"] > a.loc[s, "rank_corr"]
        wins += ok
        print(f"   {s}: MAE {a.loc[s, 'mae']:.4f} -> {b.loc[s, 'mae']:.4f}   rank corr {a.loc[s, 'rank_corr']:.4f} -> {b.loc[s, 'rank_corr']:.4f}")
    windy = (kicker[["season", "week", "home_team", "is_dome"]].drop_duplicates()
             .merge(fc[["season", "week", "home_team", "cal"]], on=["season", "week", "home_team"]))
    windy = windy[(windy.cal >= 12)][["season", "week", "home_team"]]
    for name in ("typical", "forecast"):
        p = res[name][0].merge(windy.assign(w=1), on=["season", "week", "home_team"], how="inner")
        print(f"windy games (calibrated forecast 12+ mph) {name:9}: n={len(p)}, MAE {(p.expected_points - p.fantasy_points).abs().mean():.3f}, "
              f"bias {(p.fantasy_points - p.expected_points).mean():+.3f}")
    keep = wins >= 2
    print(f"VERDICT: forecast better than typical in {wins} of 3 seasons on both measures -> {'KEEP' if keep else 'DROP'}")
    print("CALIBRATION", json.dumps({**{k: round(v, 4) for k, v in cal_all.items()}, "full_days": 2.0, "zero_days": 7.0}))
    if "--write" in sys.argv and keep:
        fw.CALIBRATION.write_text(json.dumps({**{k: round(v, 4) for k, v in cal_all.items()}, "full_days": 2.0, "zero_days": 7.0,
                                              "evidence": "scripts/fit_wind_forecast.py, 2022-2025"}, indent=1) + "\n")
        print("wrote", fw.CALIBRATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
