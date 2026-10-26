"""Projection logging, blend weights and Vegas lines."""

from __future__ import annotations

import pytest


def test_the_whole_wire_is_logged_but_not_players_who_cannot_play(tmp_cfg):
    import pandas as pd

    from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow
    from streamer.roster.vegas import log_projections

    def p(pid, **kw):
        return PlayerRow(player_id=pid, name=pid, position="WR", team=kw.pop("team", "KC"), projection=5.0,
                         ros_value=5.0, **kw)

    snap = LeagueSnapshot(
        platform="espn", profile="espn", league_id="1", league_name="L", season=2026, week=5, slots={},
        bench_size=0, teams=[TeamRow(team_id="0", name="T", is_mine=True, roster=[p("mine")])],
        free_agents=[p("wire"), p("props", ), p("hurt", status="INJURY_RESERVE"), p("nobody", team="")],
        matchup=None, synced_at="")
    snap.free_agents[1].vegas_points = 4.0
    assert log_projections(snap, tmp_cfg) == 3
    got = set(pd.read_parquet(tmp_cfg.results_dir / "skill_log.parquet")["player_id"])
    assert got == {"mine", "wire", "props"}
    # The final season value (after the consensus blend) joins this week's rows.
    from streamer.roster.vegas import log_season_values

    snap.teams[0].roster[0].ros_value = 7.5
    assert log_season_values(snap, tmp_cfg) == 3
    logged = pd.read_parquet(tmp_cfg.results_dir / "skill_log.parquet").set_index("player_id")["ros_value"]
    assert logged["mine"] == 7.5 and logged["wire"] == 5.0


def _graded(weights_by_pos, weeks=6, per=60, seed=3):
    """Logged rows where the market is worth ``weights_by_pos[pos]`` of the way."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for pos, w in weights_by_pos.items():
        for wk in range(weeks):
            ours = rng.uniform(5, 15, per)
            market = ours + rng.normal(0, 3, per)
            y = ours + w * (market - ours) + rng.normal(0, 1, per)
            rows.append(pd.DataFrame({"season": 2026, "week": wk + 1, "position": pos, "projection": ours,
                                      "fantasy_points_ppr": y}).assign(_m=market - ours))
    j = pd.concat(rows, ignore_index=True)
    return j, j["_m"].to_numpy(), np.zeros(len(j))


def test_position_weights_switch_on_only_when_they_beat_the_pooled_pair():
    from streamer.roster.vegas import _position_weights

    j, dm, dc = _graded({"QB": 0.9, "TE": 0.0, "WR": 0.45})
    got, note = _position_weights(j, dm, dc, (0.45, 0.0), k=100, need=50)
    assert got["QB"][0] > got["WR"][0] > got["TE"][0] and "weights for" in note
    j, dm, dc = _graded({"QB": 0.4, "TE": 0.4, "WR": 0.4})                 # no real difference
    got, note = _position_weights(j, dm, dc, (0.4, 0.0), k=100, need=50)
    assert got == {} and "pooled" in note
    j, dm, dc = _graded({"QB": 0.9, "TE": 0.0}, weeks=2)                   # too few weeks to trust
    assert _position_weights(j, dm, dc, (0.45, 0.0), k=100, need=50)[0] == {}


def test_blend_market_uses_a_positions_own_weight_when_given(tmp_cfg):
    from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow
    from streamer.roster.vegas import blend_market

    def p(pid, pos):
        return PlayerRow(player_id=pid, name=pid, position=pos, team="KC", projection=10.0, play_probability=1.0,
                         vegas_points=20.0, vegas_books=3)

    snap = LeagueSnapshot(platform="espn", profile="espn", league_id="1", league_name="L", season=2026, week=5,
                          slots={}, bench_size=0, teams=[TeamRow(team_id="0", name="T", is_mine=True,
                                                                 roster=[p("qb", "QB"), p("te", "TE")])],
                          free_agents=[], matchup=None, synced_at="")
    blend_market(snap, tmp_cfg, weight=0.2, by_position={"QB": (0.8, 0.0)})
    qb, te = snap.teams[0].roster
    assert qb.projection == 18.0 and te.projection == 12.0          # QB 0.8 of the way, TE the pooled 0.2


def test_an_unposted_market_is_filled_from_his_own_games_not_counted_as_zero(cfg):
    import pandas as pd

    from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow
    from streamer.roster.vegas import complete

    back = PlayerRow(player_id="b", name="Back", position="RB", team="KC", projection=15.0,
                     vegas_points=13.0, vegas_stats={"rushing_yards": 70.0, "anytime_td": 0.6})
    back.nfl_id = "00-1"
    snap = LeagueSnapshot(platform="espn", profile="espn", league_id="1", league_name="L", season=2026, week=5,
                          slots={}, bench_size=0, teams=[TeamRow(team_id="0", name="T", is_mine=True, roster=[back])],
                          free_agents=[], matchup=None, synced_at="")
    recent = pd.DataFrame({"rushing_yards": [65.0], "receptions": [3.0], "receiving_yards": [22.0],
                           "anytime_td": [0.5]}, index=["00-1"])
    espn = cfg.for_profile("espn")
    assert complete(snap, espn, recent) == 1
    assert sorted(back.vegas_filled) == ["receiving_yards", "receptions"]          # posted lines kept
    assert back.vegas_stats["rushing_yards"] == 70.0
    assert back.vegas_points == pytest.approx(70 * 0.1 + 0.6 * 6 + 3 * 1 + 22 * 0.1)
