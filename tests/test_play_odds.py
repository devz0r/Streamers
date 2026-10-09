"""FantasyPros' chance to play: attached, logged, graded, and given a say
only as far as it has earned one."""

from __future__ import annotations

import pandas as pd
import pytest

from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow
from streamer.roster import play_odds


def _league(players):
    return LeagueSnapshot(platform="espn", profile="espn", league_id="1", league_name="L", season=2026, week=5,
                          slots={}, bench_size=5, teams=[TeamRow(team_id="1", name="Me", roster=players, is_mine=True)],
                          free_agents=[], matchup=None, synced_at="")


def test_their_chance_is_put_on_tagged_players_by_name(cfg, monkeypatch):
    from streamer.data import fantasypros as fp

    monkeypatch.setattr(fp, "api_key", lambda: "k")
    monkeypatch.setattr(fp, "players", lambda c: pd.DataFrame(
        {"name": ["Mike Evans", "Drake Maye"], "team": ["SF", "NE"], "position": ["WR", "QB"], "fp_id": [11, 22]}))
    monkeypatch.setattr(fp, "injuries", lambda c: pd.DataFrame(
        {"fp_id": ["11", "22"], "status": ["Q", ""], "play": [0.89, 0.99], "updated": None}))
    evans = PlayerRow(player_id="e", name="Mike Evans", position="WR", team="SF", status="QUESTIONABLE")
    maye = PlayerRow(player_id="m", name="Drake Maye", position="QB", team="NE")
    snap = _league([evans, maye])
    assert play_odds.attach(snap, cfg) == 1
    assert evans.fp_play == pytest.approx(0.89) and maye.fp_play is None          # untagged: ours stands at 100%


def test_ours_moves_toward_theirs_only_by_the_earned_weight(cfg, monkeypatch):
    from streamer.roster.projections import _play_probability, tag_probability

    p = PlayerRow(player_id="e", name="E", position="WR", team="SF", status="QUESTIONABLE", practice="Limited",
                  fp_play=0.95)
    ours = tag_probability(p, cfg)
    monkeypatch.setattr(play_odds, "WEIGHT", 0.0)
    assert _play_probability(p, cfg) == pytest.approx(ours)
    monkeypatch.setattr(play_odds, "WEIGHT", 0.5)
    assert _play_probability(p, cfg) == pytest.approx(ours + 0.5 * (0.95 - ours))
    p.status = "OUT"
    assert _play_probability(p, cfg) == 0.0                                       # an out tag is not blended


def test_logged_reads_are_graded_on_who_played_and_the_weight_follows(cfg, tmp_path, monkeypatch):
    import types

    bound = types.SimpleNamespace(raw=cfg.raw)
    monkeypatch.setattr(play_odds, "log_path", lambda c: tmp_path / "espn" / play_odds.LOG)
    rows = []
    for i in range(60):                       # theirs right, ours flat: half of them sit
        plays = i % 2 == 0
        rows.append({"season": 2026, "week": 4, "nfl_id": f"p{i}", "name": f"P{i}", "position": "WR",
                     "status": "Q", "practice": "Limited", "ours": 0.7, "theirs": 0.9 if plays else 0.2,
                     "at": "2026-10-01T12:00:00+00:00"})
    (tmp_path / "espn").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "espn" / play_odds.LOG)
    history = pd.DataFrame({"season": 2026, "week": 4, "player_id": [f"p{i}" for i in range(0, 60, 2)]})
    g = play_odds.graded(bound, history)
    assert len(g) == 60 and g["played"].sum() == 30
    w, n = play_odds.fit_weight(bound, history)
    assert n == 60 and 0.5 < w < 1.0                                              # earned, shrunk toward 0
    few = history.iloc[:0]
    assert play_odds.fit_weight(bound, few) == (0.0, 0)                           # nothing graded: no say
