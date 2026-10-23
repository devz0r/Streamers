"""The league's most valuable players: value over a waiver pickup."""

from __future__ import annotations

import json

import pytest

from streamer.league.model import PlayerRow
from streamer.roster import season as season_mod
from streamer.roster import top_players
from streamer.roster.matchup import MatchupReport
from streamer.roster.page import _top_players
from streamer.roster.season import SeasonModel
from test_season import _league


@pytest.fixture(autouse=True)
def _no_byes(monkeypatch):
    monkeypatch.setattr(season_mod, "nfl_byes", lambda *a, **k: {})


def _fa(pid, pos, value, **kw):
    return PlayerRow(player_id=pid, name=pid, position=pos, team="KC", projection=value, projection_sd=6.0,
                     outcome_sd=6.0, ros_value=value, **kw)


def test_value_is_points_over_a_pickup_and_a_reported_absence_costs_it(cfg):
    snap = _league([1.0, 1.2, 0.9, 1.0])
    snap.free_agents = [_fa("fa-rb1", "RB", 6.0), _fa("fa-rb2", "RB", 5.0), _fa("fa-wr1", "WR", 6.0),
                        _fa("fa-wr2", "WR", 5.0), _fa("fa-qb1", "QB", 12.0), _fa("fa-qb2", "QB", 11.0),
                        _fa("fa-te1", "TE", 4.0), _fa("fa-te2", "TE", 3.0)]
    hurt = next(p for p in snap.teams[1].roster if p.player_id == "1-WR0")
    hurt.reported_back = [99, 99]                           # out for the season
    model = SeasonModel(snap, cfg, n_sims=300, extra_players=snap.free_agents)
    data = top_players.build(snap, model)
    rows = {r["name"]: r for r in data["rows"]}
    assert data["repl"]["RB"] == 5.0                        # the second-best free agent
    assert [r["vor"] for r in data["rows"]] == sorted((r["vor"] for r in data["rows"]), reverse=True)
    assert rows["1-WR0"]["vor"] == 0.0 and rows["1-WR0"]["games"] == 0.0
    assert rows["1-RB0"]["vor"] > rows["0-RB0"]["vor"] > 0   # the stronger team's lead back first
    assert rows["fa-rb1"]["owner"] == "FA" and rows["0-RB0"]["mine"]
    assert all(r["up"] >= r["vor"] for r in data["rows"])          # his best tenth of seasons

    report = MatchupReport(snapshot=snap, optimisation=None, top_players=data)
    html = _top_players(report)
    assert 'data-pos="FLEX"' in html and 'data-k="up"' in html
    payload = html.split('class="tp-data">')[1].split("</script>")[0]
    assert len(json.loads(payload)["rows"]) == len(data["rows"])


def test_a_final_score_shows_to_the_hundredth():
    from streamer.roster.page import _proj_cell

    done = PlayerRow(player_id="q", name="q", position="QB", team="KC", projection=20.0, actual_points=18.46)
    pending = PlayerRow(player_id="r", name="r", position="QB", team="KC", projection=18.46)
    assert _proj_cell(done) == "18.46"
    assert _proj_cell(pending) == "18.5"
