import json

from fixtures_league import snapshot
from streamer.data import sleeper
from streamer.roster import sleeper_proj


def _cache(cfg, season, week, rows, db):
    d = cfg.raw_dir
    (d / "sleeper_proj").mkdir(parents=True, exist_ok=True)
    (d / "sleeper_proj" / f"{season}_{week}.json").write_text(json.dumps(rows), encoding="utf-8")
    (d / "sleeper_players.json").write_text(json.dumps(db), encoding="utf-8")


def test_projections_map_through_gsis_and_skip_empty(tmp_cfg):
    _cache(tmp_cfg, 2026, 7,
           [{"player_id": "1", "stats": {"pts_ppr": 14.2}}, {"player_id": "2", "stats": {"pts_ppr": 0}},
            {"player_id": "3", "stats": {}}, {"player_id": "9", "stats": {"pts_ppr": 5.0}}],
           {"1": {"gsis_id": "00-A"}, "2": {"gsis_id": "00-B"}, "3": {"gsis_id": "00-C"}})
    got = sleeper.projections(tmp_cfg, 2026, 7, allow_network=False)
    assert got == {"00-A": 14.2}


def test_attach_matches_on_the_nflverse_id(tmp_cfg):
    snap = snapshot(week=7)
    a = snap.my_team.roster[4]
    a.nfl_id = "00-A"
    _cache(tmp_cfg, int(snap.season), 7, [{"player_id": "1", "stats": {"pts_ppr": 11.5}}],
           {"1": {"gsis_id": "00-A"}})
    assert sleeper_proj.attach(snap, tmp_cfg, allow_network=False) == 1
    assert a.sleeper_projection == 11.5
    assert all(p.sleeper_projection is None for p in snap.all_players() if p is not a)


def test_attach_without_data_is_a_no_op(tmp_cfg):
    assert sleeper_proj.attach(snapshot(week=7), tmp_cfg, allow_network=False) == 0


def test_projections_without_a_gsis_id_match_on_name_team_and_position(tmp_cfg):
    snap = snapshot(week=7)
    a = next(p for p in snap.all_players() if p.position == "TE" and p.team)
    a.nfl_id = "00-NOT-IN-SLEEPER"
    _cache(tmp_cfg, int(snap.season), 7,
           [{"player_id": "7", "stats": {"pts_ppr": 8.4}}],
           {"7": {"full_name": a.name + " Jr.", "team": a.team, "position": "TE"}})
    assert sleeper_proj.attach(snap, tmp_cfg, allow_network=False) == 1
    assert a.sleeper_projection == 8.4


def test_name_match_needs_the_same_team_and_is_not_guessed_when_ambiguous():
    rows = [{"gsis": None, "name": "Sam Smith", "team": "TB", "position": "WR", "points": 5.0},
            {"gsis": None, "name": "Sam Smith", "team": "TB", "position": "WR", "points": 9.0},
            {"gsis": None, "name": "Ann Lee", "team": "LAR", "position": "TE", "points": 7.0}]
    got = sleeper.match(rows, [("a", None, "Sam Smith", "TB", "WR"), ("b", None, "Ann Lee", "LA", "TE"),
                               ("c", None, "Ann Lee", "TB", "TE")])
    assert got == {"b": 7.0}
