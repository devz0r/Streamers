import numpy as np
import pandas as pd

from streamer.models import analogs


def _pool(n=300, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(15, 30, n)
    return pd.DataFrame({"team_implied_total": x, "fantasy_points": x / 3.0})


def test_analog_mean_averages_the_nearest_games():
    pool = _pool()
    slate = pd.DataFrame({"team_implied_total": [16.0, 29.0]})
    got = analogs.analog_mean(pool, slate, ["team_implied_total"], k=20)
    assert got[0] < 6.0 < 9.0 < got[1]
    assert abs(got[0] - 16.0 / 3) < 0.6 and abs(got[1] - 29.0 / 3) < 0.6


def test_analog_mean_is_nan_without_features_or_pool():
    pool = _pool(10)
    slate = pd.DataFrame({"team_implied_total": [20.0, np.nan]})
    assert np.isnan(analogs.analog_mean(pool, slate, ["team_implied_total"], k=50)).all()
    got = analogs.analog_mean(_pool(), slate, ["team_implied_total"], k=50)
    assert np.isfinite(got[0]) and np.isnan(got[1])


def test_blend_falls_back_to_the_model():
    out = analogs.blend(np.array([8.0, 8.0]), np.array([6.0, np.nan]), 0.5)
    assert out.tolist() == [7.0, 8.0]


def test_settings_only_for_configured_positions(cfg):
    assert analogs.settings(cfg, "DST") is None
    k = analogs.settings(cfg, "K")
    assert k is not None and 0 < float(k["weight"]) <= 1 and "team_implied_total" in k["features"]
