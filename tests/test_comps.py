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


def _season_hist(seasons=(2023, 2024), n=30, seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        lvl = rng.uniform(4, 20)
        for s in seasons:
            for w in range(1, 18):
                rows.append({"player_id": f"p{i}", "position": "WR", "season": s, "week": w, "team": "KC",
                             "total_fantasy_points_exp": lvl, "fantasy_points_ppr": lvl + rng.normal(0, 1),
                             "targets": lvl / 2, "carries": 0.0, "target_share": 0.2})
    return pd.DataFrame(rows)


def test_ros_rows_use_games_before_the_week_and_score_the_rest():
    h = _season_hist()
    r = comps.ros_rows(h, 2024, 6).set_index("player_id")
    p0 = h[(h.player_id == "p0") & (h.season == 2024)]
    assert r.loc["p0", "gp"] == 5
    assert abs(r.loc["p0", "ppg"] - p0[p0.week < 6].fantasy_points_ppr.mean()) < 1e-9
    assert abs(r.loc["p0", "ros_pg"] - p0[p0.week >= 6].fantasy_points_ppr.mean()) < 1e-9
    assert r.loc["p0", "has_prev"] == 1.0 and r.loc["p0", "career"] == 17


def test_ros_nearest_tracks_the_level():
    h = _season_hist()
    pool = comps.ros_table(h[h.season == 2023], [2023], range(3, 11)).dropna(subset=["ros_pg"])
    rows = comps.ros_rows(h, 2024, 6)
    got = comps.ros_nearest(pool, rows, k=10)
    assert np.corrcoef(got, rows["ppg"])[0, 1] > 0.9


def test_ros_only_inside_the_tested_weeks(cfg):
    assert comps.ros_for_week(cfg, {"x"}, 2026, 2) == {}
    assert comps.ros_for_week(cfg, {"x"}, 2026, 12) == {}
    assert comps.ros_settings(cfg) is not None
