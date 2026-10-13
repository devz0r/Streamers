"""Does the factor ledger earn its place in the D/ST and K ridge?

    python scripts/test_ledger.py

The ledger scales each standardised feature by ``clip(|r_blend| / |r_hist|,
0.25, 3)``, which changes the effective ridge penalty by up to (1/0.25)^2 =
16x down to (1/3)^2 and so up to 144x between two features. Walk-forward
2023-2025 with the shipped settings, the ledger on against a plain
standardised ridge (every multiplier 1), scored as the backtest does.
DECISIONS.md, "The factor ledger".
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")


def main() -> int:
    logging.disable(logging.WARNING)
    from streamer.backtest import score_predictions, walk_forward
    from streamer.config import get_config
    from streamer.data.nflverse import games_frame, load_pbp
    from streamer.features.build import build_dst_features, build_kicker_features
    from streamer.pipeline import training_seasons

    cfg = get_config().for_profile("espn")
    pbp, games = load_pbp(training_seasons(cfg), cfg), games_frame(cfg)
    frames = {"K": build_kicker_features(pbp, games, cfg), "DST": build_dst_features(pbp, games, cfg)}
    for pos, feats in frames.items():
        print(f"\n{pos}")
        res = {}
        for name, ledger in (("ledger", True), ("plain ridge", False)):
            preds = walk_forward(feats, pos, [2023, 2024, 2025], cfg, kind="ridge", use_ledger=ledger)
            _, summary = score_predictions(preds, cfg)
            res[name] = summary.set_index("season")
            print(f"  {name:12} MAE {summary.mae.mean():.4f}  rank corr {summary.rank_corr.mean():.4f}  "
                  f"top-5 hit {summary.top5_hit_rate.mean():.4f}  top-5 points {summary.top5_mean_points.mean():.3f}")
        a, b = res["ledger"], res["plain ridge"]
        for s in a.index:
            print(f"     {s}: MAE {a.loc[s, 'mae']:.4f} vs {b.loc[s, 'mae']:.4f}   rank corr {a.loc[s, 'rank_corr']:.4f} "
                  f"vs {b.loc[s, 'rank_corr']:.4f}   top5 pts {a.loc[s, 'top5_mean_points']:.2f} vs "
                  f"{b.loc[s, 'top5_mean_points']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
