import numpy as np
import pandas as pd

from streamer.roster import comps


def _hist(n_players=40, weeks=8, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_players):
        level = rng.uniform(8, 24)
        for w in range(1, weeks + 1):
            rows.append({"player_id": f"p{i}", "position": "QB", "season": 2024, "week": w, "team": "KC",
                         "total_fantasy_points_exp": level, "fantasy_points_ppr": level + rng.normal(0, 1),
                         "targets": 0.0, "carries": 3.0, "target_share": 0.0})
    return pd.DataFrame(rows)


def _games(weeks=9):
    return pd.DataFrame([{"season": 2024, "week": w, "team": "KC", "game_type": "REG", "spread_line": 3.0,
                          "total_line": 47.0, "is_home": 1} for w in range(1, weeks + 1)])


def test_features_use_only_earlier_games():
    f = comps.features(_hist(), _games())
    first = f.sort_values(["player_id", "week"]).groupby("player_id").head(1)
    assert (first["week"] == 4).all()                       # three earlier games needed
    assert np.allclose(f["implied"], 25.0) and np.allclose(f["spread"], 3.0)


def test_current_features_and_nearest_mean_track_the_player():
    h = _hist()
    pool = comps.features(h, _games())
    hi = h[h.player_id == "p0"].total_fantasy_points_exp.iloc[0]
    cur = comps.current(h, _games(), {"p0": "KC"}, 2024, 9)
    assert len(cur) == 1 and abs(float(cur.xfp4.iloc[0]) - hi) < 1e-9
    got = comps.nearest_mean(pool, cur, k=10)
    assert abs(got[0] - hi) < 2.0


def test_current_skips_players_without_enough_games_or_a_game():
    h = _hist()
    assert comps.current(h, _games(), {"p0": "KC"}, 2024, 3).empty      # two games before week 3
    assert comps.current(h, _games(), {"p0": "BUF"}, 2024, 9).empty     # no game for his team


def test_settings_cover_quarterbacks_only(cfg):
    conf = comps.settings(cfg)
    assert set(conf) == {"QB"} and 0 < float(conf["QB"]["weight"]) <= 1


def test_for_week_without_wanted_positions_is_empty(cfg):
    assert comps.for_week(cfg, _hist(), {"p0": ("KC", "WR")}, 2024, 9) == {}
