

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
