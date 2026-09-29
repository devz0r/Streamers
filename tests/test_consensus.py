"""FantasyPros consensus: parsing, matching, second opinions, and grading."""

from __future__ import annotations

import numpy as np
import pandas as pd

from streamer.data import fantasypros as fp
from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow
from streamer.roster import consensus, perception, scorecard

ROS = {"players": [
    {"player_name": "Josh Starter", "player_team_id": "ARI", "player_position_id": "QB", "rank_ecr": 10, "pos_rank": "QB1"},
    {"player_name": "Jacoby Backup", "player_team_id": "ARI", "player_position_id": "QB", "rank_ecr": 90, "pos_rank": "QB27"},
    {"player_name": "Drake Maybe", "player_team_id": "NE", "player_position_id": "QB", "rank_ecr": 50, "pos_rank": "QB11"},
]}
PROJ = {"players": [{"name": "Jacoby Backup", "team_id": "ARI", "position_id": "QB", "stats": {"points_ppr": 15.8}}]}


def _fake_get(path, params, cfg, timeout=20.0):
    if "projections" in path:
        return PROJ if params["position"] == "QB" else {"players": []}
    return ROS if params["position"] == "QB" else {"players": []}


def _qb(pid, name, team, ros):
    return PlayerRow(player_id=pid, name=name, position="QB", team=team, ros_value=ros, projection=ros,
                     nfl_id=f"n-{pid}")


def _snap():
    mine = TeamRow(team_id="1", name="Mine", is_mine=True, roster=[_qb("m", "Drake Maybe", "NE", 16.4)])
    other = TeamRow(team_id="2", name="Other", roster=[_qb("s", "Josh Starter", "ARI", 24.0)])
    return LeagueSnapshot(platform="espn", profile="espn", league_id="1", league_name="L", season=2026,
                          week=4, slots={"QB": 1}, bench_size=1, teams=[mine, other],
                          free_agents=[_qb("b", "Jacoby Backup", "ARI", 17.0)], matchup=None, synced_at="")


def test_rankings_and_projections_parse_the_documented_fields(monkeypatch, cfg):
    monkeypatch.setattr(fp, "_get", _fake_get)
    r = fp.rankings(cfg, 2026, 4, "ros")
    assert list(r.pos_rank) == [1, 27, 11] and r.team.tolist() == ["ARI", "ARI", "NE"]
    p = fp.projections(cfg, 2026, 4)
    assert p.points.tolist() == [15.8]


def test_consensus_attaches_by_name_and_team_and_flags_a_big_gap(monkeypatch, cfg):
    monkeypatch.setattr(fp, "_get", _fake_get)
    monkeypatch.setenv("FANTASYPROS_API_KEY", "test")
    snap = _snap()
    assert consensus.attach(snap, cfg) == 3
    b = snap.free_agents[0]
    assert b.ecr_ros_pos_rank == 27 and b.fp_projection == 15.8
    ours = consensus.our_ranks(snap)
    # We have the backup above Maye: on the consensus's own scale he is our #11, not #27.
    assert ours["b"] == 11 and ours["m"] == 27
    assert "lower" in consensus.note(b, ours, cfg) and "QB27" not in consensus.note(b, ours, cfg)


def test_the_market_view_moves_toward_the_consensus(monkeypatch, cfg):
    monkeypatch.setattr(fp, "_get", _fake_get)
    monkeypatch.setenv("FANTASYPROS_API_KEY", "test")
    snap = _snap()
    consensus.attach(snap, cfg)
    base = perception.perceive(snap.all_players(), {}, 2026)
    blended = perception.with_consensus(base, snap, {"consensus_weight": 0.5})
    # The consensus has the backup last of the three: the market sees him lower than before.
    assert blended["b"].value < base["b"].value
    assert blended["s"].value == base["s"].value            # the consensus agrees on the top


def test_start_sit_accuracy_counts_only_same_position_pairs():
    pred = np.array([10.0, 5.0, 8.0, 1.0])
    actual = np.array([12.0, 3.0, 2.0, 9.0])
    pos = np.array(["WR", "WR", "RB", "RB"])
    right, total = scorecard.pair_accuracy(pred, actual, pos)
    assert (right, total) == (1, 2)                          # WR pair right, RB pair wrong


def test_the_scorecard_grades_each_source_beside_ours(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(type(cfg), "results_dir", property(lambda self: tmp_path))
    rng = np.random.default_rng(0)
    n = 60
    actual = rng.normal(12, 5, n)
    log = pd.DataFrame({
        "season": 2026, "week": 3, "player_id": [str(i) for i in range(n)], "nfl_id": [f"n{i}" for i in range(n)],
        "position": ["WR", "RB"] * (n // 2), "blended_projection": actual + rng.normal(0, 3, n),
        "model_projection": actual + rng.normal(0, 4, n), "platform_projection": actual + rng.normal(0, 3.5, n),
        "vegas_points": np.nan, "logged_at": "2026-09-27T16:00:00Z"})
    log.to_parquet(tmp_path / "skill_log.parquet")
    history = pd.DataFrame({"player_id": log.nfl_id, "season": 2026, "week": 3, "fantasy_points_ppr": actual})
    card = scorecard.build(cfg, history, "ESPN")
    names = [g.source for g in card.grades]
    assert names[:3] == ["Our final number", "Our model alone", "ESPN's projection"]
    final = card.grades[0]
    assert final.players >= n - 3 and 0.5 < final.pairs <= 1 and final.miss == final.ours_miss
    assert card.per_week and card.per_week[0]["week"] == 3
