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
    assert mq == pytest.approx(12.0 * 0.72)               # measured: Q skill players play 72%
    assert sq > 6.0                                       # the coin flip widens it
    qb = PlayerRow(player_id="5", name="e", position="QB", status="Q")
    assert _status_adjust(12.0, 6.0, qb, cfg)[0] == pytest.approx(12.0 * 0.42)
    md, _ = _status_adjust(12.0, 6.0, d, cfg)
    assert md == pytest.approx(12.0 * 0.02)               # doubtful almost never plays
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


def _streak_history(recent_points: float, recent_exp: float, usual_points: float = 8.0,
                    usual_exp: float = 8.5) -> pd.DataFrame:
    rows = []
    for season in (2024, 2025):
        for week in range(1, 18):
            recent = season == 2025 and week >= 13
            rows.append({"player_id": "p", "player_display_name": "P", "position": "WR",
                         "season": season, "week": week, "team": "CAR",
                         "fantasy_points_ppr": recent_points if recent else usual_points,
                         "total_fantasy_points_exp": recent_exp if recent else usual_exp})
    return pd.DataFrame(rows)


def test_a_scoring_streak_without_more_opportunity_is_mostly_discounted(cfg):
    """Four 24-point games on the same 8.5 expected points: touchdowns and big
    plays, which regress. The projection moves, but stays near his usual."""
    proj = player_table(_streak_history(24.0, 8.5), 2025, 18, cfg).set_index("player_id")["blend"]["p"]
    assert 8.5 < proj < 13.0


def test_more_opportunity_moves_the_projection_most_of_the_way(cfg):
    """Four games at 22 expected points is a new role (a starter hurt, a
    promotion): opportunity is sticky, so it is believed."""
    proj = player_table(_streak_history(20.0, 22.0), 2025, 18, cfg).set_index("player_id")["blend"]["p"]
    assert proj > 15.0


def test_a_proven_player_in_a_cold_spell_stays_above_his_slump(cfg):
    proj = player_table(_streak_history(6.0, 16.0, usual_points=18.0, usual_exp=17.0),
                        2025, 18, cfg).set_index("player_id")["blend"]["p"]
    assert proj > 12.0


def _stale_snapshot(status: str = "", platform=None, team: str | None = "LV"):
    from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow

    ghost = PlayerRow(player_id="g1", name="Gone Guy", position="WR", team=team,
                      slot="FA", status=status, platform_projection=platform)
    me = TeamRow(team_id="1", name="Me", roster=[], is_mine=True)
    return LeagueSnapshot(platform="espn", profile="espn", league_id="1", league_name="L",
                          season=2026, week=3, slots={"WR": 2}, bench_size=5, teams=[me],
                          free_agents=[ghost], matchup=None, synced_at="2026-09-23T00:00:00+00:00")


def _stale_history():
    rows = [{"player_id": "nfl-g1", "player_display_name": "Gone Guy", "position": "WR",
             "season": 2024, "week": w, "team": "BUF", "fantasy_points_ppr": 15.0,
             "total_fantasy_points_exp": 14.0} for w in range(1, 18)]
    # Plenty of other receivers so the league has a history at all.
    for i in range(30):
        for season in (2025, 2026):
            for w in range(1, 3 if season == 2026 else 18):
                rows.append({"player_id": f"nfl-o{i}", "player_display_name": f"Other {i}",
                             "position": "WR", "season": season, "week": w, "team": "KC",
                             "fantasy_points_ppr": 8.0, "total_fantasy_points_exp": 8.0})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("status,platform,team,expect", [
    ("", None, "LV", "inactive"),     # out of the league since 2024: not projected
    ("", 11.0, "LV", "platform"),     # returning, and the platform confirms it
    ("OUT", None, "LV", "inactive"),  # tagged OUT but gone since 2024 (Aiyuk): not an injury
    ("OUT", None, None, "inactive"),  # unsigned, whatever tag he still carries
])
def test_stale_history_needs_something_current(cfg, monkeypatch, status, platform, team, expect):
    import streamer.roster.projections as pj

    snap = _stale_snapshot(status, platform, team)
    monkeypatch.setattr(pj, "_implied_scale", lambda *a, **k: ({}, "test"))
    monkeypatch.setattr(pj, "match_players",
                        lambda rows, index: type("M", (), {"mapping": {"g1": "nfl-g1"}, "unmatched": []})())
    pj.project_snapshot(snap, cfg.for_profile("espn"), rankings=None, allow_network=False,
                        history=_stale_history())
    ghost = snap.free_agents[0]
    assert ghost.projection_source.startswith(expect), ghost.projection_source
    if expect == "inactive":
        assert ghost.projection == 0.0 and ghost.ros_value == 0.0
    if expect == "model":
        assert ghost.ros_value and ghost.ros_value > 5.0 and ghost.projection == 0.0


def test_hurt_partway_through_last_season_keeps_his_value(cfg, monkeypatch):
    """Out since week 8 of last season, tagged OUT: an injury, not a departure."""
    import streamer.roster.projections as pj

    snap = _stale_snapshot("OUT", None, "LV")
    history = _stale_history()
    gone = history.player_id == "nfl-g1"
    history.loc[gone, "season"] = 2025
    history = history[~gone | (history.week <= 8)]
    monkeypatch.setattr(pj, "_implied_scale", lambda *a, **k: ({}, "test"))
    monkeypatch.setattr(pj, "match_players",
                        lambda rows, index: type("M", (), {"mapping": {"g1": "nfl-g1"}, "unmatched": []})())
    pj.project_snapshot(snap, cfg.for_profile("espn"), rankings=None, allow_network=False, history=history)
    hurt = snap.free_agents[0]
    assert hurt.projection_source.startswith("model") and hurt.ros_value > 5.0 and hurt.projection == 0.0


def _coverage_snapshot(jacobs_proj, others_proj=15.0, n_others=24):
    from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow

    roster = [PlayerRow(player_id="jj", name="Josh Jacobs", position="RB", team="GB",
                        slot="RB", status="DAY_TO_DAY", platform_projection=jacobs_proj)]
    roster += [PlayerRow(player_id=f"o{i}", name=f"Other {i}", position="WR", team="KC",
                         slot="BN", platform_projection=others_proj) for i in range(n_others)]
    me = TeamRow(team_id="1", name="Me", roster=roster, is_mine=True)
    return LeagueSnapshot(platform="espn", profile="espn", league_id="1", league_name="L",
                          season=2026, week=3, slots={"RB": 2, "WR": 2}, bench_size=6, teams=[me],
                          free_agents=[], matchup=None, synced_at="2026-09-24T00:00:00+00:00")


def _history_with_jacobs():
    rows = []
    for pid, name, pos, pts in (("nfl-jj", "Josh Jacobs", "RB", 18.0),) + tuple(
            (f"nfl-o{i}", f"Other {i}", "WR", 10.0) for i in range(24)):
        for season in (2025, 2026):
            for w in range(1, 18 if season == 2025 else 3):
                rows.append({"player_id": pid, "player_display_name": name, "position": pos,
                             "season": season, "week": w, "team": "GB" if pid == "nfl-jj" else "KC",
                             "fantasy_points_ppr": pts, "total_fantasy_points_exp": pts})
    return pd.DataFrame(rows)


def _project(snap, cfg, monkeypatch):
    import streamer.roster.projections as pj

    monkeypatch.setattr(pj, "_implied_scale", lambda *a, **k: ({}, "test"))
    mapping = {p.player_id: f"nfl-{p.player_id}" for p in snap.my_team.roster}
    monkeypatch.setattr(pj, "match_players",
                        lambda rows, index: type("M", (), {"mapping": mapping, "unmatched": []})())
    return pj.project_snapshot(snap, cfg.for_profile("espn"), rankings=None,
                               allow_network=False, history=_history_with_jacobs())


def test_platform_zero_means_not_playing_this_week(cfg, monkeypatch):
    """Josh Jacobs, commissioner-exempt: ESPN says DAY_TO_DAY and projects 0.0."""
    snap = _coverage_snapshot(jacobs_proj=0.0)
    rep = _project(snap, cfg, monkeypatch)
    jj = snap.my_team.roster[0]
    assert jj.projection == 0.0 and jj.model_projection == 0.0
    assert jj.ros_value and jj.ros_value > 10.0          # still a real player going forward
    assert "sits" in jj.projection_source
    assert any("Josh Jacobs" in n and "not playing" in n for n in rep.notes)

    # And the optimiser starts a healthy backup over him.
    from streamer.league.model import PlayerRow
    from streamer.roster.lineup import optimise
    backup = PlayerRow(player_id="bk", name="Backup Back", position="RB", team="GB",
                       slot="BN", eligible_slots=["RB", "FLEX"], projection=5.0, projection_sd=3.0)
    opt = optimise(snap.my_team.roster + [backup], {"RB": 1, "WR": 2}, None, n_sims=300)
    assert "jj" not in opt.best_win.player_ids and "bk" in opt.best_win.player_ids


def test_platform_zero_ignored_when_the_feed_is_not_populated(cfg, monkeypatch):
    """If the platform has projected almost nobody yet, a zero means nothing."""
    snap = _coverage_snapshot(jacobs_proj=0.0, others_proj=0.0)
    _project(snap, cfg, monkeypatch)
    jj = snap.my_team.roster[0]
    assert jj.projection and jj.projection > 5.0


# ---------------------------------------------------------------------------
# The next man up
# ---------------------------------------------------------------------------
def _depth_history() -> pd.DataFrame:
    """KC: a 16-point lead back and a 4-point backup; two receivers likewise."""
    rows = []
    for season in (2025, 2026):
        for week in range(1, 18 if season == 2025 else 3):
            for pid, name, pos, exp in (("nfl-lead", "Lead Back", "RB", 16.0),
                                        ("nfl-cuff", "Cuff Back", "RB", 4.0),
                                        ("nfl-wr1", "Top Receiver", "WR", 15.0),
                                        ("nfl-wr3", "Deep Receiver", "WR", 4.0)):
                rows.append({"player_id": pid, "player_display_name": name, "position": pos,
                             "season": season, "week": week, "team": "KC",
                             "fantasy_points_ppr": exp, "total_fantasy_points_exp": exp})
    return pd.DataFrame(rows).sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def _depth_snapshot(lead_status: str = "", wr_status: str = ""):
    roster = [
        PlayerRow(player_id="lead", name="Lead Back", position="RB", team="KC", status=lead_status),
        PlayerRow(player_id="wr1", name="Top Receiver", position="WR", team="KC", status=wr_status),
    ]
    fas = [
        PlayerRow(player_id="cuff", name="Cuff Back", position="RB", team="KC", slot="FA"),
        PlayerRow(player_id="wr3", name="Deep Receiver", position="WR", team="KC", slot="FA"),
    ]
    return LeagueSnapshot(
        platform="espn", profile="espn", league_id="1", league_name="t", season=2026, week=3,
        slots={"RB": 1, "WR": 1}, bench_size=2,
        teams=[TeamRow(team_id="1", name="me", roster=roster, is_mine=True)],
        free_agents=fas, matchup=None, synced_at="",
    )


def _project_depth(snap, cfg):
    from streamer.roster import projections

    projections.project_snapshot(snap, cfg, rankings=None, allow_network=False, history=_depth_history())
    return {p.player_id: p for p in snap.all_players()}


def test_backup_inherits_part_of_an_injured_starters_work(cfg):
    healthy = _project_depth(_depth_snapshot(), cfg)
    hurt = _project_depth(_depth_snapshot(lead_status="INJURY_RESERVE"), cfg)
    gap = healthy["lead"].model_projection - healthy["cuff"].model_projection
    gain = hurt["cuff"].model_projection - healthy["cuff"].model_projection
    assert 0.25 * gap < gain < 0.45 * gap                 # ~0.35 of the gap
    assert hurt["cuff"].ros_value > healthy["cuff"].ros_value   # IR: rest of season too
    assert any("next man up: Lead Back is on IR" in s for s in hurt["cuff"].signals)
    assert not healthy["cuff"].signals or all("next man up" not in s for s in healthy["cuff"].signals)


def test_a_one_week_absence_lifts_this_week_more_than_the_season(cfg):
    healthy = _project_depth(_depth_snapshot(), cfg)
    out = _project_depth(_depth_snapshot(lead_status="OUT"), cfg)
    week_gain = out["cuff"].model_projection - healthy["cuff"].model_projection
    ros_gain = out["cuff"].ros_value - healthy["cuff"].ros_value
    assert week_gain > 0 and 0 < ros_gain < week_gain


def test_receivers_do_not_inherit_a_missing_receivers_targets(cfg):
    """Measured: a missing WR's targets spread too thinly to lift any one
    teammate, so none is projected to."""
    healthy = _project_depth(_depth_snapshot(), cfg)
    hurt = _project_depth(_depth_snapshot(wr_status="OUT"), cfg)
    assert hurt["wr3"].model_projection == pytest.approx(healthy["wr3"].model_projection)


def test_roles_and_opponents_are_attached(cfg):
    got = _project_depth(_depth_snapshot(), cfg)
    assert got["lead"].role == "RB1" and got["cuff"].role == "RB2"
    assert got["wr1"].role == "WR1"
    assert got["lead"].ros_sd is not None and got["lead"].ros_sd > 0
