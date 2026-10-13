"""Snap share as an early sign of a role change."""

from __future__ import annotations

import pandas as pd
import pytest

from streamer.roster import snaps


def _history_and_snaps():
    hist, sn = [], []
    for w in range(1, 6):
        for pid, name, pct, exp in (("rising", "Rising Back", 0.3 if w <= 3 else 0.8, 6.0),
                                    ("falling", "Falling Back", 0.8 if w <= 3 else 0.3, 12.0),
                                    ("steady", "Steady Back", 0.5, 8.0)):
            hist.append({"player_id": pid, "player_display_name": name, "position": "RB", "season": 2026,
                         "week": w, "team": "KC", "fantasy_points_ppr": exp, "total_fantasy_points_exp": exp})
            sn.append({"season": 2026, "week": w, "game_type": "REG", "player": name, "position": "RB",
                       "team": "KC", "offense_pct": pct})
    return pd.DataFrame(hist), snaps.tidy(pd.DataFrame(sn))


def test_a_back_whose_snaps_jumped_is_raised_and_one_whose_snaps_fell_is_cut(cfg):
    hist, sn = _history_and_snaps()
    sig = snaps.signals(snaps.join(hist, sn), 2026, 6)
    assert sig.loc["rising", "delta"] == pytest.approx(0.8 - 0.3)
    assert sig.loc["falling", "delta"] == pytest.approx(0.3 - 0.8)
    f = snaps.factors(sig, cfg)
    assert f["rising"][0] > 1.0 > f["falling"][0]
    assert "on the field more lately" in f["rising"][1] or "more than he is used" in f["rising"][1]


def test_too_few_games_or_no_snaps_change_nothing(cfg):
    hist, sn = _history_and_snaps()
    assert snaps.signals(snaps.join(hist, sn), 2026, 4).empty          # 3 games: too few
    assert snaps.signals(snaps.join(hist, pd.DataFrame()), 2026, 6).empty
    sig = snaps.signals(snaps.join(hist.assign(position="WR"), sn.iloc[:0]), 2026, 6)
    assert snaps.factors(sig, cfg) == {}
