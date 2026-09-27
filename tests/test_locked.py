"""Games already played: scores locked in, lineup spots frozen."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from fixtures_league import SLOTS, my_roster, opponent_roster
from streamer.league.model import LeagueSnapshot, Matchup, PlayerRow, TeamRow
from streamer.roster import lineup
from streamer.roster.locked import kickoff_utc, lock_played


def _games():
    return pd.DataFrame([
        {"game_id": "g1", "team": "ATL", "gameday": "2026-09-24", "gametime": "20:15", "team_score": 35.0},
        {"game_id": "g1", "team": "GB", "gameday": "2026-09-24", "gametime": "20:15", "team_score": 14.0},
        {"game_id": "g2", "team": "KC", "gameday": "2026-09-27", "gametime": "13:00", "team_score": np.nan},
        {"game_id": "g2", "team": "MIA", "gameday": "2026-09-27", "gametime": "13:00", "team_score": np.nan},
    ])


def _snap():
    me = TeamRow(team_id="1", name="me", is_mine=True, roster=[
        PlayerRow(player_id="lon", name="Drake London", position="WR", team="ATL", slot="WR", nfl_id="n-lon"),
        PlayerRow(player_id="kel", name="Travis Kelce", position="TE", team="KC", slot="TE", nfl_id="n-kel"),
        PlayerRow(player_id="gbk", name="Tee Smack", position="K", team="GB", slot="K"),
    ])
    return LeagueSnapshot(platform="espn", profile="espn", league_id="1", league_name="L", season=2026,
                          week=3, slots={"WR": 1, "TE": 1, "K": 1}, bench_size=0, teams=[me],
                          free_agents=[], matchup=None, synced_at="")


def _history(with_week3: bool = True):
    rows = [{"player_id": "n-lon", "season": 2026, "week": 2, "team": "ATL", "fantasy_points_ppr": 8.9}]
    if with_week3:
        rows.append({"player_id": "n-lon", "season": 2026, "week": 3, "team": "ATL", "fantasy_points_ppr": 28.4})
    return pd.DataFrame(rows)


def test_kickoff_is_eastern_time():
    assert kickoff_utc("2026-09-27", "13:00") == datetime(2026, 9, 27, 17, 0, tzinfo=UTC)


def test_final_scores_lock_and_unstarted_games_do_not(cfg):
    snap = _snap()
    pbp = pd.DataFrame(columns=["week"])
    notes = lock_played(snap, cfg, _history(), now=datetime(2026, 9, 27, 15, 0, tzinfo=UTC),
                        games=_games(), pbp=pbp)
    by = {p.player_id: p for p in snap.all_players()}
    assert by["lon"].locked and by["lon"].actual_points == 28.4
    assert not by["kel"].locked and by["kel"].actual_points is None
    assert by["gbk"].locked                      # kicked off, no kicker line available
    assert any("locked into P(win)" in n for n in notes)


def test_a_final_game_without_published_stats_stays_simulated(cfg):
    snap = _snap()
    notes = lock_played(snap, cfg, _history(with_week3=False), now=datetime(2026, 9, 27, 15, 0, tzinfo=UTC),
                        games=_games(), pbp=pd.DataFrame(columns=["week"]))
    lon = snap.my_team.roster[0]
    assert lon.locked and lon.actual_points is None
    assert any("not published" in n for n in notes)


def test_a_final_score_is_a_constant_in_the_simulation():
    p = PlayerRow(player_id="x", name="x", position="WR", projection=10.0, projection_sd=7.0, actual_points=28.4)
    s = lineup.sample_points([p], 1000, np.random.default_rng(0))
    assert (s[:, 0] == 28.4).all()


def test_locked_starters_stay_and_locked_bench_players_cannot_come_in():
    roster = my_roster()
    by = {p.player_id: p for p in roster}
    by["6"].locked, by["6"].actual_points = True, 2.0     # started, played badly
    by["7"].locked, by["7"].actual_points = True, 30.0    # benched, went off
    res = lineup.optimise(roster, SLOTS, opponent_roster(), n_sims=4000)
    for lu in (res.best_win, res.best_ev):
        assert "6" in lu.player_ids and "7" not in lu.player_ids


def test_the_editor_payload_carries_all_three_lineups(cfg):
    import json
    import re

    from fixtures_league import snapshot
    from streamer.roster.matchup import build_report
    from streamer.roster.page import render_my_team

    snap = snapshot(week=5)
    snap.matchup = snap.matchup or Matchup(week=5, my_team_id="1", opponent_team_id="2")
    report = build_report(snap, cfg)
    html = render_my_team(snap, report, [], cfg)
    blob = re.search(r'class="ed-data">(.*?)</script>', html).group(1)
    data = json.loads(blob)
    assert set(data["presets"]) == {"set", "win", "pts"}
    assert len(data["presets"]["win"]) == len(data["slots"])
    assert data["n"] * data["m"] * 2 * 4 / 3 <= len(data["s"]) + 4
    assert "Best P(win)" in html and "Most points" in html
