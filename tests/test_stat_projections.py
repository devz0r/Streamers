"""Per-stat projections are scaled together to the player's projection."""

from types import SimpleNamespace as NS

import pandas as pd

from streamer.config import get_config
from streamer.league.model import PlayerRow
from streamer.roster import stat_projections as S


def test_the_split_adds_up_to_his_projection_on_the_leagues_sheet(monkeypatch):
    cfg = get_config().for_profile("yahoo")
    table = pd.DataFrame([{"pass_yards": 0.0, "pass_td": 0.0, "interception": 0.0, "rush_yards": 5.0, "rush_td": 0.0,
                           "reception": 6.0, "rec_yards": 80.0, "rec_td": 0.5}], index=["00-1"])
    monkeypatch.setattr(S, "per_game", lambda *_a: table)
    wr = PlayerRow(player_id="1", name="Wide Out", position="WR", team="CIN", projection=20.0)
    wr.nfl_id, wr.pre_market_projection = "00-1", 18.0
    snap = NS(season=2026, week=9, all_players=lambda: [wr])
    assert S.attach(snap, cfg) == 1
    sc = cfg._profile["offense_scoring"]
    total = sum(float(sc[S.STATS[k][2]]) * v for k, v in wr.stat_projection.items())
    assert abs(total - 18.0) < 0.01                         # the pre-market projection, not the blended one
    assert wr.stat_projection["reception"] > wr.stat_projection["rec_td"]
