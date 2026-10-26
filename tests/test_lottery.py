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


def test_a_fragile_lead_back_makes_his_handcuff_a_better_ticket():
    from streamer.roster.lottery import lead_miss_rate
    from streamer.roster.waivers import LEAD_BACK_MISS_RATE

    lead = PlayerRow(player_id="l", name="Lead", position="RB", team="SEA")
    assert lead_miss_rate(lead) == LEAD_BACK_MISS_RATE                   # no record: the league rate
    lead.hazard_mult = 2.0                                               # missed games lately: odds doubled
    assert 0.15 < lead_miss_rate(lead) < 0.16
    lead.hazard_mult = 0.5
    assert 0.04 < lead_miss_rate(lead) < 0.05


def test_the_quarterback_big_week_chance_is_scaled_down_by_config(cfg):
    snap = snapshot(week=3)
    snap.platform = "yahoo"
    hist = pd.DataFrame({"player_id": [f"x{i}" for i in range(40)], "season": 2025, "week": 1,
                         "position": ["QB"] * 40, "fantasy_points_ppr": list(range(10, 50))})
    qb = PlayerRow(player_id="t2", name="Ticket QB", position="QB", team="SEA", slot="FA",
                   eligible_slots=["QB"], projection=12.0, projection_sd=7.0, outcome_sd=7.0)
    snap.free_agents = [qb]
    snap.my_team.roster = [p for p in snap.my_team.roster if p.position != "QB"]
    c = cfg.for_profile("yahoo")
    kw = {"games": _games(), "now": datetime(2026, 9, 25, tzinfo=UTC), "history": hist}
    conf = c.raw["roster"]["lottery_tickets"]
    saved = dict(conf)
    conf.update(boom_scale={"QB": 0.65}, min_keep=0.0)
    scaled = lottery_tickets(snap, c, **kw)[0].tickets[0].p_boom
    conf.update(boom_scale={})
    try:
        unscaled = lottery_tickets(snap, c, **kw)[0].tickets[0].p_boom
    finally:
        conf.clear()
        conf.update(saved)
    assert scaled == __import__("pytest").approx(0.65 * unscaled)


def test_tickets_are_logged_with_their_inputs_and_graded_against_the_week(cfg, tmp_cfg):
    from streamer.roster.lottery import grade_tickets, log_tickets

    snap = snapshot(week=3)
    snap.platform = "yahoo"
    hist = pd.DataFrame({"player_id": [f"x{i}" for i in range(40)], "season": 2025, "week": 1,
                         "position": ["WR"] * 40, "fantasy_points_ppr": list(range(30)) + [0] * 10})
    wr = PlayerRow(player_id="t1", name="Ticket WR", position="WR", team="SEA", slot="FA", nfl_id="n1",
                   eligible_slots=["WR", "FLEX"], projection=8.0, projection_sd=6.0, outcome_sd=6.0,
                   fp_projection=9.5, vegas_points=11.0, vegas_stats={"rec_yards": 55.0})
    snap.free_agents = [wr]
    c = cfg.for_profile("yahoo")
    windows = lottery_tickets(snap, c, games=_games(), now=datetime(2026, 9, 25, tzinfo=UTC), history=hist)
    assert windows and windows[0].bars["WR"] > 0
    assert log_tickets(snap, tmp_cfg, windows, windows[0].bars) == 1
    assert log_tickets(snap, tmp_cfg, windows, windows[0].bars) == 1             # same window: replaced
    logged = pd.read_parquet(tmp_cfg.results_dir / "ticket_log.parquet")
    assert len(logged) == 1 and logged.iloc[0]["fp_projection"] == 9.5 and logged.iloc[0]["vegas_points"] == 11.0
    assert grade_tickets(tmp_cfg, pd.DataFrame(columns=["season", "week", "player_id", "fantasy_points_ppr"])) is None
    played = pd.DataFrame({"player_id": ["n1"], "season": [2026], "week": [3], "fantasy_points_ppr": [30.0]})
    g = grade_tickets(tmp_cfg, played)
    assert g["n"] == 1 and g["bands"][0]["happened"] == 1.0
