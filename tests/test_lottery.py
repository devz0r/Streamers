"""Game-time lottery tickets."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from fixtures_league import snapshot
from streamer.league.model import PlayerRow
from streamer.roster.lottery import (
    enabled,
    kickoff_windows,
    lottery_tickets,
    p_at_least,
    window_label,
)


def _games():
    rows = []
    for team, day, time in (("KC", "2026-09-24", "20:15"), ("BUF", "2026-09-27", "13:00"),
                            ("SEA", "2026-09-27", "16:05"), ("LA", "2026-09-27", "16:25"),
                            ("DEN", "2026-09-27", "20:20"), ("CHI", "2026-09-28", "20:15")):
        rows.append({"team": team, "gameday": day, "gametime": time})
    return pd.DataFrame(rows)


def test_windows_group_the_late_slots_and_skip_started_games():
    labels = [w.label for w in kickoff_windows(_games(), now=datetime(2026, 9, 25, 12, tzinfo=UTC))]
    assert labels == ["Sun 1:00 PM", "Sun 4 PM", "Sun 8:20 PM", "Mon 8:15 PM"]
    assert window_label(datetime(2026, 9, 25, 0, 15, tzinfo=UTC)) == "Thu 8:15 PM"


def test_boom_chance_falls_with_the_bar_and_respects_injury_tags():
    p = PlayerRow(player_id="w", name="w", position="WR", projection=9.0, projection_sd=6.0, outcome_sd=6.0)
    assert p_at_least(p, 12.0) > p_at_least(p, 18.0) > p_at_least(p, 25.0) > 0
    q = PlayerRow(player_id="q", name="q", position="WR", projection=9.0 * 0.72, projection_sd=6.0,
                  outcome_sd=6.0, play_probability=0.72, status="QUESTIONABLE")
    assert abs(p_at_least(q, 18.0) - 0.72 * p_at_least(p, 18.0)) < 1e-6


def test_tickets_only_at_positions_where_a_big_week_would_matter(cfg):
    snap = snapshot(week=3)
    snap.platform = "yahoo"
    hist = pd.DataFrame({"player_id": [f"x{i}" for i in range(40)], "season": 2025, "week": 1,
                         "position": ["QB"] * 10 + ["WR"] * 30,
                         "fantasy_points_ppr": [30.0] * 10 + list(range(30))})
    wr = PlayerRow(player_id="t1", name="Ticket WR", position="WR", team="SEA", slot="FA",
                   eligible_slots=["WR", "FLEX"], projection=8.0, projection_sd=6.0, outcome_sd=6.0)
    qb = PlayerRow(player_id="t2", name="Ticket QB", position="QB", team="SEA", slot="FA",
                   eligible_slots=["QB"], projection=14.0, projection_sd=7.0, outcome_sd=7.0)
    snap.free_agents = [wr, qb]
    windows = lottery_tickets(snap, cfg.for_profile("yahoo"), games=_games(),
                              now=datetime(2026, 9, 25, tzinfo=UTC), history=hist)
    names = [t.player.name for w in windows for t in w.tickets]
    assert "Ticket WR" in names
    assert "Ticket QB" not in names            # would not beat the 19-point starter at QB
    assert enabled(snap, cfg.for_profile("yahoo"))
    snap.platform = "espn"
    assert not enabled(snap, cfg.for_profile("espn"))
