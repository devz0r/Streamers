"""Does weather move skill players' points, held out by season? (screening test)

    python scripts/grade_weather_skill.py [--no-rain]

Outdoor games 2022-2025. For each QB/RB/WR/TE game, the base is his own
trailing five-game average (any season) scaled by his team's implied total (from the closing
line); the question is whether adding the game's weather -- precipitation,
wind above 12 mph, cold below 40F -- brings the prediction closer to what he
scored, fitted on the other three seasons and scored on the fourth. If it does
not here it will not in the full model, which already has everything this
base has and more. Weather is what Open-Meteo's archive of forecasts as issued
held for the kickoff hours (``--no-rain`` uses the recorded wind and
temperature only, where Open-Meteo is unreachable). The verdict is the usual
rule: better overall, worse in at most one season. Aggregates only.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

SEASONS = (2022, 2023, 2024, 2025)
POS = ("QB", "RB", "WR", "TE")


def forecasts(games: pd.DataFrame) -> pd.DataFrame:
    """Kickoff-window precipitation (mm), wind (mph) and temperature (F), as forecast."""
    import requests

    from streamer.data import forecast_wind as fw

    rows = []
    for (season, home), g in games.groupby(["season", "home_team"]):
        where = fw.venue(home, g["stadium"].iloc[0])
        if where is None:
            continue
        kos = {r.game_id: fw.kickoff_utc(str(r.gameday), r.gametime) for r in g.itertuples()}
        start, end = min(kos.values()).strftime("%Y-%m-%d"), (max(kos.values()) + pd.Timedelta(hours=4)).strftime("%Y-%m-%d")
        try:
            r = requests.get(fw.ARCHIVE_URL, timeout=30, params={
                "latitude": where[0], "longitude": where[1], "hourly": "precipitation,wind_speed_10m,temperature_2m",
                "wind_speed_unit": "mph", "temperature_unit": "fahrenheit", "timezone": "UTC",
                "start_date": start, "end_date": end})
            r.raise_for_status()
            hourly = r.json().get("hourly") or {}
        except Exception as exc:  # noqa: BLE001
            print(f"  {season} {home}: {type(exc).__name__}", flush=True)
            continue
        times = hourly.get("time") or []
        for game_id, ko in kos.items():
            want = {(ko + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H:00") for h in range(4)}
            idx = [i for i, t in enumerate(times) if t in want]
            if not idx:
                continue
            pr = [hourly["precipitation"][i] for i in idx if hourly["precipitation"][i] is not None]
            rows.append({"game_id": game_id, "precip": float(np.sum(pr)) if pr else np.nan,
                         "fc_wind": float(np.nanmean([hourly["wind_speed_10m"][i] for i in idx])),
                         "fc_temp": float(np.nanmean([hourly["temperature_2m"][i] for i in idx]))})
        time.sleep(0.3)
    return pd.DataFrame(rows)


def panel(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    from streamer.data.nflverse import games_frame

    stats = []
    for s in range(2019, 2026):
        d = pd.read_parquet(cfg.raw_dir / f"player_stats_{s}.parquet")
        stats.append(d[(d["season_type"] == "REG") & d["position"].isin(POS)][
            ["player_id", "season", "week", "position", "team", "opponent_team", "fantasy_points_ppr", "carries",
             "attempts", "targets"]])
    st = pd.concat(stats, ignore_index=True).sort_values(["player_id", "season", "week"])
    st["trail"] = st.groupby("player_id")["fantasy_points_ppr"].transform(lambda x: x.shift(1).rolling(5, min_periods=4).mean())
    g = games_frame(cfg)
    g = g[(g["game_type"] == "REG") & (g["season"].isin(SEASONS))].drop_duplicates("game_id")
    return st, g


def attach(st: pd.DataFrame, g: pd.DataFrame, wx: pd.DataFrame | None) -> pd.DataFrame:
    sc = pd.read_parquet(Path(__file__).resolve().parents[1] / "data/raw/schedules.parquet")
    sc = sc[(sc["game_type"] == "REG") & sc["season"].isin(SEASONS)][
        ["game_id", "season", "week", "home_team", "away_team", "total_line", "spread_line", "temp", "wind", "roof"]]
    rows = []
    for r in sc.itertuples():
        for team, imp in ((r.home_team, (r.total_line + r.spread_line) / 2), (r.away_team, (r.total_line - r.spread_line) / 2)):
            rows.append({"season": r.season, "week": r.week, "team": team, "game_id": r.game_id, "implied": imp,
                         "rec_wind": r.wind, "rec_temp": r.temp, "roof": r.roof})
    tg = pd.DataFrame(rows)
    d = st.merge(tg, on=["season", "week", "team"], how="inner")
    dome = d["roof"].isin(["dome", "closed"])
    if wx is not None:
        d = d.merge(wx, on="game_id", how="left")
        d["rain"], d["wind_x"], d["temp_x"] = d["precip"], d["fc_wind"], d["fc_temp"]
    else:
        d["rain"], d["wind_x"], d["temp_x"] = np.nan, d["rec_wind"], d["rec_temp"]
    d["dome"] = dome
    d["f_rain"] = np.where(dome, 0.0, np.clip(d["rain"].fillna(0.0), 0, 6.0))
    d["f_wind"] = np.where(dome, 0.0, np.clip(d["wind_x"].fillna(8.0) - 12.0, 0, None) / 5.0)
    d["f_cold"] = np.where(dome, 0.0, np.clip(40.0 - d["temp_x"].fillna(60.0), 0, None) / 10.0)
    league = d.groupby("season")["implied"].transform("mean")
    d["f_imp"] = d["implied"] - league
    d = d[d["trail"].notna() & (d["trail"] >= 6.0)]
    d["resid"] = d["fantasy_points_ppr"] - d["trail"]
    return d


def fit(train: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = np.column_stack([np.ones(len(train))] + [train[c].to_numpy(float) for c in cols])
    return np.linalg.solve(X.T @ X + 20.0 * np.diag([0.0] + [1.0] * len(cols)), X.T @ train["resid"].to_numpy(float))


def predict(b: np.ndarray, test: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return b[0] + np.column_stack([test[c].to_numpy(float) for c in cols]) @ b[1:]


def grade(d: pd.DataFrame, label: str, weather: list[str]) -> None:
    base = ["f_imp"]
    print(f"== {label}: {len(d)} player-games")
    for pos in POS:
        x = d[d["position"] == pos]
        per, tot = [], [0.0, 0.0, 0]
        for s in SEASONS:
            tr, te = x[x["season"] != s], x[x["season"] == s]
            if len(te) < 50:
                continue
            e0 = te["resid"].to_numpy(float) - predict(fit(tr, base), te, base)
            e1 = te["resid"].to_numpy(float) - predict(fit(tr, base + weather), te, base + weather)
            per.append((s, len(te), np.abs(e0).mean(), np.abs(e1).mean()))
            tot[0] += np.abs(e0).sum(); tot[1] += np.abs(e1).sum(); tot[2] += len(te)
        worse = sum(a1 > a0 for _s, _n, a0, a1 in per)
        coef = fit(x, base + weather)
        eff = " ".join(f"{c} {v:+.2f}" for c, v in zip(weather, coef[2:]))
        print(f"  {pos}: MAE {tot[0] / tot[2]:.3f} -> {tot[1] / tot[2]:.3f} (worse in {worse} of {len(per)} seasons)  "
              f"effects per unit: {eff}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rain", action="store_true")
    args = ap.parse_args()
    from streamer.config import get_config

    cfg = get_config()
    st, g = panel(cfg)
    wx = None
    if not args.no_rain:
        out = g[g["is_dome"].fillna(0).astype(int).eq(0)]
        wx = forecasts(out)
        print(f"{len(wx)} of {len(out)} outdoor games with an archived forecast; rain share (>0.5 mm) "
              f"{(wx.precip > 0.5).mean():.1%}", flush=True)
    d = attach(st, g, wx)
    wet = d[(~d["dome"]) & (d["f_rain"] > 0.5)] if wx is not None else d.iloc[:0]
    if len(wet):
        dry = d[(~d["dome"]) & (d["f_rain"] <= 0.5)]
        print("rain games vs dry, mean points above trailing average (all positions): "
              + " ".join(f"{p} {wet[wet.position == p].resid.mean():+.2f}/{dry[dry.position == p].resid.mean():+.2f}" for p in POS))
        t = d.groupby(["game_id", "team"]).agg(att=("attempts", "sum"), car=("carries", "sum"),
                                                 rain=("f_rain", "max"), dome=("dome", "max")).reset_index()
        t = t[~t["dome"]]
        print(f"team pass attempts / carries, rain vs dry: {t[t.rain > 0.5].att.mean():.1f}/{t[t.rain > 0.5].car.mean():.1f} "
              f"vs {t[t.rain <= 0.5].att.mean():.1f}/{t[t.rain <= 0.5].car.mean():.1f}")
    if wx is not None:
        grade(d, "rain, wind, cold", ["f_rain", "f_wind", "f_cold"])
    else:
        grade(d, "recorded wind and cold only", ["f_wind", "f_cold"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
