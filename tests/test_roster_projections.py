"""Projection engine on synthetic history, so it runs fast and offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow
from streamer.roster.projections import (
    _status_adjust,
    load_history,
    lookup_sd,
    player_table,
    sd_table,
)


def _history(n_players: int = 40, weeks: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_players):
        pos = ["QB", "RB", "WR", "TE"][i % 4]
        talent = rng.uniform(5, 20)
        for season in (2024, 2025):
            for week in range(1, weeks + 1):
                exp = talent + rng.normal(0, 2)
                rows.append({
                    "player_id": f"p{i}", "player_display_name": f"Player {i}",
                    "position": pos, "season": season, "week": week, "team": "KC",
                    "fantasy_points_ppr": max(0.0, exp + rng.normal(0, 6)),
                    "total_fantasy_points_exp": max(0.0, exp),
                })
    return pd.DataFrame(rows).sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def test_player_table_is_leak_free(cfg):
    """A monster game in the target week must not move that week's projection."""
    hist = _history()
    quiet = player_table(hist, 2025, 8, cfg).set_index("player_id")["blend"]
    spiked = hist.copy()
    mask = (spiked.player_id == "p0") & (spiked.season == 2025) & (spiked.week == 8)
    spiked.loc[mask, "fantasy_points_ppr"] = 80.0
    spiked.loc[mask, "total_fantasy_points_exp"] = 80.0
    assert player_table(spiked, 2025, 8, cfg).set_index("player_id")["blend"]["p0"] == pytest.approx(quiet["p0"])


def test_player_table_uses_only_prior_weeks(cfg):
    hist = _history()
    early = player_table(hist, 2025, 2, cfg)
    late = player_table(hist, 2025, 12, cfg)
    assert not early.empty and not late.empty
    # More history -> less shrinkage -> projections spread further from the mean.
    assert late["blend"].std() >= early["blend"].std() * 0.8


def test_player_table_before_any_history_is_empty(cfg):
    hist = _history()
    assert player_table(hist[hist.season >= 2026], 2026, 1, cfg).empty


def test_sd_table_grows_with_projection_level(cfg):
    table = sd_table(_history(n_players=120, weeks=17), cfg)
    assert table
    for pos in ("QB", "RB", "WR", "TE"):
        assert (pos, -1) in table                     # the per-position fallback
        assert lookup_sd(table, pos, 3.0) > 0
    # An unknown position falls back to a sane default rather than raising.
    assert lookup_sd(table, "LB", 10.0) == pytest.approx(7.5)


def test_status_adjustment_is_a_mixture(cfg):
    healthy = PlayerRow(player_id="1", name="a", position="WR")
    q = PlayerRow(player_id="2", name="b", position="WR", status="Q")
    d = PlayerRow(player_id="3", name="c", position="WR", status="D")
    out = PlayerRow(player_id="4", name="d", position="WR", status="OUT")
    m, s = _status_adjust(12.0, 6.0, healthy, cfg)
    assert (m, s) == (12.0, 6.0)
    mq, sq = _status_adjust(12.0, 6.0, q, cfg)
    assert mq == pytest.approx(12.0 * 0.75)
    assert sq > 6.0                                       # the coin flip widens it
    md, _ = _status_adjust(12.0, 6.0, d, cfg)
    assert md == pytest.approx(12.0 * 0.25)
    assert _status_adjust(12.0, 6.0, out, cfg) == (0.0, 0.0)


def test_project_snapshot_fills_every_matched_player(cfg, monkeypatch):
    """End-to-end on synthetic history, with lines and rankings stubbed out."""
    from streamer.roster import projections

    hist = _history()
    monkeypatch.setattr(projections, "_implied_scale", lambda *a, **k: ({"KC": 1.1}, "stub"))
    roster = [
        PlayerRow(player_id="x1", name="Player 0", position="QB", team="KC", slot="QB"),
        PlayerRow(player_id="x2", name="Player 1", position="RB", team="KC", slot="RB",
                  platform_projection=9.0),
        PlayerRow(player_id="x3", name="Nobody Known", position="WR", team="KC"),
        PlayerRow(player_id="x4", name="Chiefs D/ST", position="DST", team="KC", platform_projection=8.0),
    ]
    snap = LeagueSnapshot(
        platform="espn", profile="espn", league_id="1", league_name="t", season=2025, week=9,
        slots={"QB": 1}, bench_size=0,
        teams=[TeamRow(team_id="1", name="me", roster=roster, is_mine=True)],
        free_agents=[], matchup=None, synced_at="",
    )
    report = projections.project_snapshot(snap, cfg, rankings=None, allow_network=False, history=hist)
    by = {p.player_id: p for p in roster}
    assert by["x1"].projection_source == "model" and by["x1"].projection > 0
    assert by["x2"].projection_source == "model+platform"
    assert by["x3"].projection_source == "prior" and "Nobody Known (WR)" in report.unmatched
    assert by["x4"].projection_source == "platform" and by["x4"].projection == 8.0
    assert all(p.projection_sd is not None for p in roster)
    assert report.projected == 4


def test_load_history_returns_the_expected_columns(cfg):
    hist = load_history(cfg)
    for col in ("player_id", "player_display_name", "position", "season", "week",
                "team", "fantasy_points_ppr", "total_fantasy_points_exp"):
        assert col in hist.columns
