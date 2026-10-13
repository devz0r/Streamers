"""Does the opposing quarterback himself help the D/ST model?

    python scripts/fit_dst_opp_qb.py

The D/ST model knows the opponent's *team* sack, interception and pressure
rates, which describe whoever played quarterback before -- a backup starting
this week keeps his starter's numbers. Here, each game's opposing starter
(the passer with the most dropbacks) is rated from his own earlier dropbacks
only -- sack rate, interception rate and EPA per dropback, each shrunk toward
the league by ``SHRINK`` dropbacks -- and given to the model as
``opp_qb_sack_rate``, ``opp_qb_int_rate``, ``opp_qb_epa`` and the gap between
his sack and interception rates and his team's. Walk-forward 2023-2025, the
shipped D/ST model against the same model with these features, scored as
the backtest does. The starter is taken from the game itself, as production
would take him from the depth chart and injury report. DECISIONS.md, "D/ST
and the quarterback across from it".
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
SHRINK = 150.0
NEW = ("opp_qb_sack_rate", "opp_qb_int_rate", "opp_qb_epa", "opp_qb_sack_gap", "opp_qb_int_gap")


def qb_ratings(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (game, offence): the starter's rates from his earlier dropbacks."""
    from streamer.teams import normalize_team

    d = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna()]
    if "season_type" in d.columns:
        d = d[d["season_type"] == "REG"]
    g = (d.groupby(["season", "week", "game_id", "posteam", "passer_player_id"])
         .agg(db=("qb_dropback", "size"), sacks=("sack", "sum"), ints=("interception", "sum"),
              epa=("qb_epa", "sum")).reset_index().sort_values(["season", "week"]))
    league = g[["db", "sacks", "ints", "epa"]].sum()
    mu = {"sack": league.sacks / league.db, "int": league.ints / league.db, "epa": league.epa / league.db}
    for col in ("db", "sacks", "ints", "epa"):
        g[f"prev_{col}"] = g.groupby("passer_player_id")[col].transform(lambda x: x.shift(1).cumsum()).fillna(0.0)
    starter = g.sort_values("db").groupby(["game_id", "posteam"]).tail(1).sort_values(["season", "week"])
    prev = starter.groupby("posteam")["passer_player_id"].shift(1)
    out = pd.DataFrame({
        "opp_qb_changed": (prev.notna() & (prev != starter["passer_player_id"])).astype(int),
        "game_id": starter["game_id"], "opponent": starter["posteam"].map(normalize_team),
        "opp_qb_sack_rate": (starter["prev_sacks"] + SHRINK * mu["sack"]) / (starter["prev_db"] + SHRINK),
        "opp_qb_int_rate": (starter["prev_ints"] + SHRINK * mu["int"]) / (starter["prev_db"] + SHRINK),
        "opp_qb_epa": (starter["prev_epa"] + SHRINK * mu["epa"]) / (starter["prev_db"] + SHRINK)})
    return out


def main() -> int:
    logging.disable(logging.WARNING)
    from streamer import backtest
    from streamer.config import get_config
    from streamer.data.nflverse import games_frame, load_pbp
    from streamer.features.build import DST_FEATURES, build_dst_features
    from streamer.models.positions import DstModel
    from streamer.pipeline import training_seasons

    cfg = get_config().for_profile("espn")
    pbp = load_pbp(training_seasons(cfg), cfg)
    dst = build_dst_features(pbp, games_frame(cfg), cfg)
    cols = ["game_id", "season", "week", "posteam", "passer_player_id", "qb_dropback", "sack", "interception",
            "qb_epa", "season_type"]
    raw = pd.concat([pd.read_parquet(cfg.raw_dir / f"pbp_{s}.parquet", columns=cols)
                     for s in training_seasons(cfg) if (cfg.raw_dir / f"pbp_{s}.parquet").exists()])
    dst = dst.merge(qb_ratings(raw), on=["game_id", "opponent"], how="left")
    for col in ("opp_qb_sack_rate", "opp_qb_int_rate", "opp_qb_epa"):
        dst[col] = dst[col].fillna(dst[col].mean())
    dst["opp_qb_sack_gap"] = dst["opp_qb_sack_rate"] - dst["opp_sack_rate_allowed"]
    dst["opp_qb_int_gap"] = dst["opp_qb_int_rate"] - dst["opp_int_rate"]
    print(f"D/ST rows {len(dst)}; starter rated on {dst['opp_qb_epa'].notna().mean():.0%}")
    res = {}
    for name, feats in (("shipped", tuple(DST_FEATURES)), ("with opposing QB", tuple(DST_FEATURES) + NEW),
                        ("new-starter flag", tuple(DST_FEATURES) + ("opp_qb_changed",))):
        DstModel.FEATURES = feats
        backtest._features_for = lambda pos, f=feats: list(f)
        preds = backtest.walk_forward(dst, "DST", [2023, 2024, 2025], cfg, kind="ridge")
        _, summary = backtest.score_predictions(preds, cfg)
        res[name] = summary.set_index("season")
        preds = preds.merge(dst[["game_id", "team", "opp_qb_changed"]], on=["game_id", "team"], how="left",
                            suffixes=("", "_x"))
        ch = preds[preds["opp_qb_changed"] == 1]
        res[name + " changed"] = ch
        print(f"     games facing a different starter than last week: n={len(ch)}, "
              f"MAE {(ch.expected_points - ch.fantasy_points).abs().mean():.4f}, "
              f"bias {(ch.fantasy_points - ch.expected_points).mean():+.3f}")
        print(f"  {name:18} MAE {summary.mae.mean():.4f}  rank corr {summary.rank_corr.mean():.4f}  "
              f"top-5 hit {summary.top5_hit_rate.mean():.4f}  top-5 points {summary.top5_mean_points.mean():.3f}")
    a, b = res["shipped"], res["new-starter flag"]
    for s in a.index:
        print(f"     {s}: MAE {a.loc[s, 'mae']:.4f} -> {b.loc[s, 'mae']:.4f}   rank corr {a.loc[s, 'rank_corr']:.4f} -> "
              f"{b.loc[s, 'rank_corr']:.4f}   top5 pts {a.loc[s, 'top5_mean_points']:.2f} -> {b.loc[s, 'top5_mean_points']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
