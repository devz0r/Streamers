"""The live-odds grading (scripts/grade_live_odds.py) rescoring draws as the page does."""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import grade_live_odds as G  # noqa: E402


def test_points_so_far_are_read_play_by_play():
    base = {"game_id": "g", "qtr": 1, "passer_player_id": None, "receiver_player_id": None, "rusher_player_id": None,
            "fumbled_1_player_id": None, "td_player_id": None, "passing_yards": 0, "receiving_yards": 0,
            "rushing_yards": 0, "complete_pass": 0, "pass_touchdown": 0, "rush_touchdown": 0, "interception": 0,
            "fumble_lost": 0}
    plays = pd.DataFrame([
        {**base, "game_seconds_remaining": 3000, "passer_player_id": "qb", "receiver_player_id": "wr", "td_player_id": "wr",
         "passing_yards": 20, "receiving_yards": 20, "complete_pass": 1, "pass_touchdown": 1},      # first quarter
        {**base, "game_seconds_remaining": 1000, "qtr": 3, "rusher_player_id": "rb", "rushing_yards": 30},
        {**base, "game_seconds_remaining": 0, "qtr": 5, "rusher_player_id": "rb", "td_player_id": "rb",
         "rushing_yards": 5, "rush_touchdown": 1},                                                    # overtime: at the end
    ])
    t = G.skill_so_far(plays).set_index("pid")
    assert t.loc["wr", ["q25", "q100"]].tolist() == [9.0, 9.0]             # a catch, 20 yards, the touchdown
    assert abs(t.loc["qb", "q25"] - 4.8) < 1e-9
    assert t.loc["rb", ["q50", "q75", "q100"]].tolist() == [0.0, 3.0, 9.5]


def test_the_draws_are_rescored_as_the_page_does():
    d = np.array([[10.0, 9.0], [20.0, 9.0]])
    m, now, dst = np.array([15.0, 9.0]), np.array([8.0, 20.0]), np.array([False, True])
    half = np.array([0.5, 0.5])
    # A receiver at the half: 8 + 0.5 x 15 + sqrt(0.5) (draw - 15). A defence with 20 on a sheet that starts it at 11
    # keeps its 20 and gives back half the base: 20 + 0.5 (9 - 11).
    got = G.totals(d, m, now, half, dst, dst_base=11.0)
    want = [8 + 7.5 + np.sqrt(0.5) * (x - 15) + 19.0 for x in (10.0, 20.0)]
    assert np.allclose(got, want)
    assert np.allclose(G.totals(d, m, now, np.array([0.0, 0.0]), dst, dst_base=11.0), [28.0, 28.0])      # all final
    assert np.allclose(G.totals(d, m, now, np.array([1.0, 1.0]), dst, dst_base=11.0), [19.0, 29.0])      # none started


def test_the_page_uses_the_exponent_the_grading_reports():
    js = (ROOT / "src" / "streamer" / "static" / "live.js").read_text(encoding="utf-8")
    got = re.search(r"var K_END = ([0-9.]+), K_POWER = ([0-9.]+);", js)
    assert got and (float(got.group(1)), float(got.group(2))) == (G.K_END, G.K_POWER)
    from streamer.roster.lineup import WIN_SHRINK

    assert G.K0 == WIN_SHRINK
    assert float(G.k_live(1.0)) == WIN_SHRINK and float(G.k_live(0.0)) == G.K_END
