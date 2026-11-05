"""Each player's projection split into the stats props are written on.

The skill projection is built on fantasy points. nflverse's ``ff_opportunity``
carries an expectation for every stat, so the same core (volume x efficiency,
as in ``projections._volume_efficiency``) runs per stat: receptions, receiving /
rushing / passing yards, touchdowns of each kind, interceptions, two-point
conversions. Walk-forward 2022-2025 the stats summed on each sheet matched the
points core (ESPN RMSE 7.061 vs 7.064, Yahoo 7.401 vs 7.403; two seasons each
way), touchdowns, interceptions and two-pointers wanting four times the
efficiency prior (``scripts/fit_stat_projections.py``).

Each player's stats are scaled together to his projection before the market
(``pre_market_projection``), so everything production adds on top -- the
game's lines, injuries, the next man up -- carries through, and the split
only says where his points come from. Logged with the projection log for the
per-stat market blend and the betting grade. DECISIONS.md, "Projecting each
stat".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..league.model import LeagueSnapshot

log = logging.getLogger(__name__)

# stat -> (actual column, expected column, sheet key, efficiency prior multiple)
STATS = {
    "pass_yards": ("pass_yards_gained", "pass_yards_gained_exp", "pass_yards", 1.0),
    "pass_td": ("pass_touchdown", "pass_touchdown_exp", "pass_td", 4.0),
    "interception": ("pass_interception", "pass_interception_exp", "interception", 4.0),
    "rush_yards": ("rush_yards_gained", "rush_yards_gained_exp", "rush_yards", 1.0),
    "rush_td": ("rush_touchdown", "rush_touchdown_exp", "rush_td", 4.0),
    "reception": ("receptions", "receptions_exp", "reception", 1.0),
    "rec_yards": ("rec_yards_gained", "rec_yards_gained_exp", "reception_yards", 1.0),
    "rec_td": ("rec_touchdown", "rec_touchdown_exp", "reception_td", 4.0),
}
POS = ("QB", "RB", "WR", "TE")


def _history(cfg: Config, season: int, week: int, years: int = 3) -> pd.DataFrame:
    frames = []
    for s in range(season - years, season + 1):
        p = cfg.raw_dir / f"ff_opportunity_{s}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["season"], d["week"] = pd.to_numeric(d["season"]).astype(int), pd.to_numeric(d["week"]).astype(int)
    d = d[d["position"].isin(POS) & (d["week"] <= 18)]
    d = d[(d["season"] < season) | (d["week"] < week)]
    for a, e, _k, _m in STATS.values():
        d[a], d[e] = d[a].fillna(0.0).astype(float), d[e].fillna(0.0).astype(float)
    return d.sort_values(["player_id", "season", "week"])


def per_game(cfg: Config, season: int, week: int) -> pd.DataFrame:
    """Player (nflverse id) x stat: the core's projection a game, from games before ``week``."""
    d = _history(cfg, season, week)
    if d.empty:
        return pd.DataFrame()
    conf = cfg.raw["roster"]
    long_n, short_n = int(conf.get("long_games", 17)), int(conf["trailing_games"])
    alpha = 1.0 - 0.5 ** (1.0 / float(conf.get("volume_halflife_games", 2.5)))
    prior = d[d["season"] < season]
    sc = cfg._profile["offense_scoring"]
    exp_pts = sum(float(sc[k]) * prior[e] for _a, e, k, _m in STATS.values())
    games = float(conf.get("efficiency_prior_points", 150.0)) / exp_pts.groupby(prior["position"]).mean()
    out = {}
    for stat, (a, e, _k, mult) in STATS.items():
        g = d.groupby("player_id", sort=False)
        last = g.tail(long_n).copy()
        last["age"] = last.groupby("player_id").cumcount(ascending=False)
        w = (1.0 - alpha) ** last["age"]
        agg = pd.DataFrame({"pid": last["player_id"], "pos": last["position"], "a": last[a], "e": last[e],
                            "ew": w, "ewx": w * last[e], "short": (last["age"] < short_n).astype(float)})
        agg["a_s"], agg["e_s"] = agg["a"] * agg["short"], agg["e"] * agg["short"]
        s = agg.groupby("pid").agg(pos=("pos", "last"), a_long=("a", "sum"), e_long=("e", "sum"), c_long=("a", "size"),
                                   a_short=("a_s", "sum"), e_short=("e_s", "sum"), c_short=("short", "sum"),
                                   ew=("ew", "sum"), ewx=("ewx", "sum"))
        s["v_recent"] = s["ewx"] / s["ew"]
        pm = prior.groupby("position")[e].mean()
        pe = (prior.groupby("position")[a].sum() / prior.groupby("position")[e].sum().where(lambda x: x > 0)).fillna(1.0)
        m_eff = mult * s["pos"].map(games) * s["pos"].map(pm)
        k, k_vol = float(conf["shrink_games"]), float(conf.get("volume_prior_games", 2.0))
        vol_long = (s["e_long"] + k * s["pos"].map(pm)) / (s["c_long"] + k)
        vol = (s["c_short"] * s["v_recent"] + k_vol * vol_long) / (s["c_short"] + k_vol)
        eff_long = (s["a_long"] + m_eff * s["pos"].map(pe)) / (s["e_long"] + m_eff)
        eff_rec = (s["a_short"] + 0.4 * m_eff * eff_long) / (s["e_short"] + 0.4 * m_eff)
        w_rec = float(conf.get("recent_efficiency_weight", 0.25))
        out[stat] = vol * ((1 - w_rec) * eff_long + w_rec * eff_rec)
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)


def attach(snapshot: LeagueSnapshot, cfg: Config) -> int:
    """Put ``stat_projection`` (stat -> this week's number, scaled to his
    projection before the market) on every skill player with an nflverse id."""
    table = per_game(cfg, int(snapshot.season), int(snapshot.week))
    if table.empty:
        return 0
    sc = cfg._profile["offense_scoring"]
    n = 0
    for p in snapshot.all_players():
        if p.position not in POS or not p.nfl_id or str(p.nfl_id) not in table.index:
            continue
        row = table.loc[str(p.nfl_id)].fillna(0.0)
        pts = sum(float(sc[STATS[s][2]]) * float(v) for s, v in row.items())
        target = p.pre_market_projection if p.pre_market_projection is not None else p.projection
        if target is None or pts <= 0.5:
            continue
        f = float(target) / pts
        p.stat_projection = {s: round(float(v) * f, 3) for s, v in row.items()}
        n += 1
    return n
