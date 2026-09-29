"""Deferring to the market where our season value has overreacted."""

from __future__ import annotations

from streamer.roster import market_calibration as mc

CONF = {"cells": [{"position": "WR", "sign": "up", "weeks": [3, 6], "k": 0.25, "note": "early surge"},
                  {"position": "QB", "sign": "down", "weeks": [3, 6], "k": 0.4, "note": "slow start"}]}
LINES = {("w1", 2025): (13.0, 17), ("w1", 2026): (13.4, 3), ("q1", 2025): (22.0, 17), ("q1", 2026): (12.0, 3)}


def test_an_early_receiver_surge_is_tempered_toward_his_name_value():
    adj = mc.calibrate("WR", 4, 16.1, "w1", 2026, LINES, CONF)
    assert adj is not None and adj.market < 16.1
    assert abs(adj.value - (adj.market + 0.25 * (16.1 - adj.market))) < 1e-9
    assert adj.market < adj.value < 16.1


def test_a_markdown_of_a_receiver_stands():
    # A London-shaped case: we are below his name value; history says we were right.
    lines = {("w2", 2025): (16.8, 12), ("w2", 2026): (14.3, 3)}
    assert mc.calibrate("WR", 4, 14.6, "w2", 2026, lines, CONF) is None


def test_only_early_in_the_season_and_only_with_a_name():
    assert mc.calibrate("WR", 9, 16.1, "w1", 2026, LINES, CONF) is None
    assert mc.calibrate("WR", 4, 16.1, "rookie", 2026, LINES, CONF) is None
    q = mc.calibrate("QB", 4, 15.0, "q1", 2026, LINES, CONF)
    assert q is not None and q.value > 15.0                  # a slow-start QB is marked back up


def test_the_fitted_file_only_shrinks():
    for cell in mc.load()["cells"]:
        assert 0 <= cell["k"] < 1 and cell["evidence"]["miss_calibrated"] < cell["evidence"]["miss_ours"]
