"""Weather for kicker projections.

Wind is the only weather variable with a robust, well-documented effect on
field-goal accuracy and attempt distance, so it is the one we model; temp is
carried for reporting. No free forecast API is reliable enough (and key-free)
to depend on for an automated weekly job, so the design is:

1. ``data/weather_week_N.csv`` -- manual override, committed from a Mac.
2. Dome / retractable-roof-closed games -> forced to neutral automatically.
3. Everything else -> neutral defaults from ``config.yaml``.

Historical rows use the observed ``temp``/``wind`` that nflverse ships on the
schedule, so the model is fit on real weather and applied with forecast (or
neutral) weather.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import Config, get_config
from ..teams import normalize_team

log = logging.getLogger(__name__)


def manual_weather_path(week: int, cfg: Config | None = None) -> Path:
    cfg = cfg or get_config()
    return cfg.data_dir / f"weather_week_{week}.csv"


def read_manual_weather(week: int, cfg: Config | None = None) -> pd.DataFrame:
    """Read ``data/weather_week_N.csv`` (columns: ``home_team,wind,temp``)."""
    cfg = cfg or get_config()
    path = manual_weather_path(week, cfg)
    if not path.exists():
        return pd.DataFrame(columns=["home_team", "wind", "temp"])
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {"home": "home_team", "team": "home_team",
              "wind_mph": "wind", "temp_f": "temp"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "home_team" not in df.columns:
        raise ValueError(f"{path} must have a home_team column")
    df["home_team"] = df["home_team"].map(normalize_team)
    for col in ("wind", "temp"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    return df[df["home_team"].notna()][["home_team", "wind", "temp"]]


def typical_wind(cfg: Config, season: int, week: int, years: int = 2) -> pd.Series:
    """Home team -> mean recorded wind (mph) at its outdoor games over the
    last ``years`` seasons and this one so far: what to expect where no
    forecast exists. Empty when there is no schedule history."""
    from .nflverse import games_frame

    g = games_frame(cfg)
    g = g[(g["game_type"] == "REG") & (g["season"] >= season - years)
          & ((g["season"] < season) | (g["week"] < week))]
    g = g[g["is_dome"].fillna(0).astype(int).eq(0) & g["wind"].notna()].drop_duplicates("game_id")
    g = g.assign(home_team=g["home_team"].map(normalize_team))
    return g.groupby("home_team")["wind"].mean()


def resolve_weather(
    slate: pd.DataFrame, week: int, cfg: Config | None = None
) -> pd.DataFrame:
    """Attach ``wind``, ``temp``, ``is_dome`` and ``weather_source`` to a slate.

    ``slate`` needs ``home_team`` and either ``roof`` or ``is_dome``.
    """
    cfg = cfg or get_config()
    conf = cfg.weather
    out = slate.copy()
    if "is_dome" not in out.columns:
        out["is_dome"] = out.get("roof", pd.Series(index=out.index, dtype=object)).isin(
            ["dome", "closed"]
        ).astype(int)

    out["wind"] = pd.to_numeric(out.get("wind"), errors="coerce")
    out["temp"] = pd.to_numeric(out.get("temp"), errors="coerce")
    earlier = out["weather_source"] if "weather_source" in out.columns else pd.Series("schedule", index=out.index)
    out["weather_source"] = earlier.where(out["wind"].notna(), None).where(
        out["wind"].isna() | earlier.notna(), "schedule")

    manual = read_manual_weather(week, cfg)
    if not manual.empty:
        lookup = manual.set_index("home_team")
        for idx, row in out.iterrows():
            if row["home_team"] not in lookup.index:
                continue
            entry = lookup.loc[row["home_team"]]
            if isinstance(entry, pd.DataFrame):
                entry = entry.iloc[0]
            if not pd.isna(entry.get("wind")):
                out.at[idx, "wind"] = float(entry["wind"])
                out.at[idx, "weather_source"] = "manual"
            if not pd.isna(entry.get("temp")):
                out.at[idx, "temp"] = float(entry["temp"])
                out.at[idx, "weather_source"] = "manual"

    # Indoors is always neutral, whatever the CSV says.
    dome = out["is_dome"] == 1
    out.loc[dome, "wind"] = 0.0
    out.loc[dome, "temp"] = 72.0
    out.loc[dome, "weather_source"] = "dome"

    # No wind recorded or given: a game still to be played. The model trains
    # on recorded wind, so calm is not neutral -- filling 0 (about 7 mph under
    # the outdoor average) cost kickers 0.03 of MAE and 0.012 of rank
    # correlation in every held-out season; the home stadium's typical wind
    # recovers most of it (scripts/fit_kicker_wind.py; DECISIONS.md, "Kicker
    # wind").
    missing = out["wind"].isna()
    if missing.any() and "season" in out.columns and len(out):
        try:
            typ = typical_wind(cfg, int(out["season"].iloc[0]), week)
            usual = out["home_team"].map(typ)
            fill = missing & usual.notna()
            out.loc[fill, "wind"] = usual[fill]
            out.loc[fill, "weather_source"] = "typical"
            missing = out["wind"].isna()
        except Exception as exc:  # noqa: BLE001 - a prior; fall back to the neutral default
            log.warning("typical wind unavailable: %s", exc)
    out.loc[missing, "wind"] = float(conf["default_wind_mph"])
    out.loc[missing, "weather_source"] = out.loc[missing, "weather_source"].fillna("neutral")
    out["temp"] = out["temp"].fillna(float(conf["default_temp_f"]))
    out["weather_source"] = out["weather_source"].fillna("neutral")
    out["high_wind"] = (out["wind"] >= float(conf["high_wind_threshold"])).astype(int)
    return out
