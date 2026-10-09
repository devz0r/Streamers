"""Who a player faces in the weeks to come, priced into the season simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from streamer.roster import schedule


def _games():
    """Four teams, three weeks played and two to come: KC scores freely and BUF
    concedes freely, by the closing lines."""
    rows, wk = [], 0
    pairs = [("KC", "BUF"), ("SF", "DAL"), ("KC", "SF"), ("BUF", "DAL"), ("KC", "DAL"), ("BUF", "SF")]
    strength = {"KC": (4, 0), "BUF": (0, 4), "SF": (0, 0), "DAL": (0, -2)}       # (offence, points conceded)
    for i, (home, away) in enumerate(pairs):
        wk = 1 + i // 2
        for t, o, h in ((home, away, 1), (away, home, 0)):
            rows.append({"season": 2026, "week": wk, "game_type": "REG", "team": t, "opponent": o, "is_home": h,
                         "team_implied_total": 22 + strength[t][0] + strength[o][1] + h})
    for wk, (home, away) in ((4, ("SF", "BUF")), (5, ("SF", "DAL")), (5, ("KC", "BUF"))):
        for t, o, h in ((home, away, 1), (away, home, 0)):
            rows.append({"season": 2026, "week": wk, "game_type": "REG", "team": t, "opponent": o, "is_home": h,
                         "team_implied_total": np.nan})
    return pd.DataFrame(rows)


def test_a_soft_defence_ahead_lifts_the_week_and_a_tough_one_lowers_it(cfg):
    history = pd.DataFrame({"season": [2026] * 6, "week": [1, 1, 2, 2, 3, 3], "team": ["KC", "BUF", "KC", "SF",
                            "BUF", "DAL"], "position": "QB", "fantasy_points_ppr": [20.0] * 6, "player_id": list("abcdef")})
    import copy
    import dataclasses

    raw = copy.deepcopy(cfg.raw)
    raw["roster"]["schedule"]["ridge"] = 0.05                 # six games: let them speak
    f = schedule.factors(dataclasses.replace(cfg, raw=raw), 2026, 4, history, games=_games())
    assert set(f[("SF", "QB")]) == {5}                         # week 4 is this week: its own lines price it
    assert not any(pos == "WR" for _t, pos in f)               # receivers are not priced
    # Week 5: KC visits the soft BUF defence, SF hosts the stingy DAL one.
    assert f[("BUF", "QB")][5] < f[("KC", "QB")][5] and f[("SF", "QB")][5] < 1.0 < f[("KC", "QB")][5]


def test_the_card_note_says_easier_or_harder_and_the_playoffs():
    assert schedule.describe({}) == ""
    note = schedule.describe({6: 1.02, 15: 1.10, 16: 1.08, 17: 1.09})
    assert note.startswith("easier schedule the rest of the way (+7%)") and "playoff weeks 15-17 +9%" in note
    assert schedule.describe({6: 1.01, 7: 0.99}) == ""


def test_the_simulator_plays_each_week_at_his_schedule(monkeypatch):
    from streamer.league.model import PlayerRow
    from streamer.roster import futures

    p = PlayerRow(player_id="q", name="Q", position="QB", team="SF", ros_value=20.0, projection=20.0)
    flat = futures.simulate([p], [6, 7], {}, n_sims=4000, seed=5).scores[:, 0, 1].mean()
    p.schedule = {7: 1.2}
    hard = futures.simulate([p], [6, 7], {}, n_sims=4000, seed=5).scores[:, 0, 1].mean()
    assert hard == pytest.approx(flat * 1.2, rel=0.08)
