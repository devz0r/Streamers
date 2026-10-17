"""Kicker kicking-points lines and the opposing QB's interception line:
logged beside our unit projections, graded before they move anything."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from streamer.data.props import to_fantasy_points
from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow
from streamer.roster import unit_props

TEAMS = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND",
         "JAX", "KC", "LV", "LAC", "LA", "MIA"]


def _snap(week=7):
    k = PlayerRow(player_id="k1", name="Kay Kicker", position="K", team="KC",
                  vegas_stats={"kicking_points": 8.6}, vegas_books=3)
    q = PlayerRow(player_id="q1", name="Que Back", position="QB", team="DEN", projection=18.0,
                  vegas_stats={"passing_interceptions": 0.95})
    q2 = PlayerRow(player_id="q2", name="Backup Q", position="QB", team="DEN", projection=4.0,
                   vegas_stats={"passing_interceptions": 0.2})
    filled = PlayerRow(player_id="q3", name="Filled Q", position="QB", team="LV", projection=15.0,
                       vegas_stats={"passing_interceptions": 0.7}, vegas_filled=["passing_interceptions"])
    dst = PlayerRow(player_id="d1", name="Chiefs D/ST", position="DST", team="KC")
    me = TeamRow(team_id="1", name="Me", is_mine=True, roster=[k, q, dst])
    return LeagueSnapshot(platform="espn", profile="espn", league_id="1", league_name="L", season=2026,
                          week=week, slots={"K": 1}, bench_size=5, teams=[me], free_agents=[q2, filled],
                          matchup=None, synced_at="2026-10-17T00:00:00Z")


def _rankings():
    return SimpleNamespace(
        kicker=pd.DataFrame([{"team": "KC", "opponent": "DEN", "expected_points": 8.0, "baseline_points": 7.5},
                             {"team": "LV", "opponent": "KC", "expected_points": 6.0, "baseline_points": 6.0}]),
        dst=pd.DataFrame([{"team": "KC", "opponent": "DEN", "expected_points": 7.0, "baseline_points": 6.5},
                          {"team": "KC2", "opponent": "LV", "expected_points": 5.0, "baseline_points": 5.0}]))


def test_kicking_points_are_kept_but_not_scored_as_fantasy_points(cfg):
    cons = pd.DataFrame([{"player": "Kay Kicker", "stat": "kicking_points", "mean": 8.6, "books": 2}])
    got = to_fantasy_points(cons, cfg.for_profile("espn"))
    assert got.iloc[0]["vegas_points"] == 0.0 and got.iloc[0]["stats"] == {"kicking_points": 8.6}


def test_rows_pair_each_unit_with_its_line():
    rows = unit_props.rows(_snap(), _rankings())
    k = [r for r in rows if r["position"] == "K"]
    d = [r for r in rows if r["position"] == "DST"]
    assert [(r["team"], r["kicker"], r["kicking_points"]) for r in k] == [("KC", "Kay Kicker", 8.6)]
    # The starter's posted line, not the backup's; a filled-in line does not count.
    assert [(r["team"], r["opp_qb"], r["opp_qb_int"]) for r in d] == [("KC", "Que Back", 0.95)]


def test_cards_show_the_line_and_projections_do_not_move():
    snap = _snap()
    assert unit_props.annotate(snap, _rankings()) == 2
    k = next(p for p in snap.all_players() if p.position == "K")
    d = next(p for p in snap.all_players() if p.position == "DST")
    assert any("kicking-points line: 8.6" in s for s in k.signals)
    assert any("Que Back" in s and "0.95" in s for s in d.signals)
    assert k.projection is None and d.projection is None


def test_the_log_is_replaced_until_kickoff(tmp_cfg):
    snap = _snap()
    unit_props.log_week(snap, tmp_cfg, _rankings(), now="t1")
    snap.all_players()[0].vegas_stats = {"kicking_points": 9.1}
    unit_props.log_week(snap, tmp_cfg, _rankings(), now="t2")
    lg = pd.read_parquet(tmp_cfg.results_dir / "unit_log.parquet")
    assert len(lg) == 2 and lg[lg.position == "K"].kicking_points.tolist() == [9.1]
    # His game kicks off: the last pre-game line stands.
    snap.all_players()[0].locked = True
    snap.all_players()[0].vegas_stats = {"kicking_points": 4.0}
    unit_props.log_week(snap, tmp_cfg, _rankings(), now="t3")
    lg = pd.read_parquet(tmp_cfg.results_dir / "unit_log.parquet")
    assert lg[lg.position == "K"].kicking_points.tolist() == [9.1]


def test_grade_waits_for_data_then_keeps_a_line_that_helps(tmp_cfg):
    assert unit_props.grade(tmp_cfg)["K"]["verdict"].startswith("waiting")
    rng = np.random.default_rng(0)
    rows, hist = [], []
    for week in range(1, 7):
        for t in range(20):
            line = rng.uniform(5, 11)
            truth = 0.95 * line
            ours = truth + rng.normal(0, 2.0)           # ours is noisy; the line is the truth here
            team = TEAMS[t]
            rows.append({"season": 2026, "week": week, "position": "K", "team": team, "expected_points": ours,
                         "kicking_points": line, "opp_qb_int": None})
            hist.append({"season": 2026, "week": week, "position": "K", "team": team,
                         "actual_points": truth + rng.normal(0, 1.0)})
    pd.DataFrame(rows).to_parquet(tmp_cfg.results_dir / "unit_log.parquet", index=False)
    pd.DataFrame(hist).to_parquet(tmp_cfg.results_dir / "history.parquet", index=False)
    res = unit_props.grade(tmp_cfg)["K"]
    assert res["verdict"] == "KEEP" and res["mae_with_line"] < res["mae_ours"]
