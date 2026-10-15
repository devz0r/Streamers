"""Every team's best lineup this week, against its opponent as set or at its best."""

from __future__ import annotations

from fixtures_league import snapshot
from streamer.roster import all_lineups, lineup


def test_pairings_come_from_the_schedule_and_the_matchup():
    snap = snapshot(week=3)
    snap.rules = dict(snap.rules or {}, schedule=[[3, "a", "b"], [4, "a", "c"]])
    opp = all_lineups.pairings(snap)
    assert opp["a"] == "b" and opp["b"] == "a" and "c" not in opp
    me = snap.my_team.team_id
    assert opp[me] == snap.matchup.opponent_team_id


def test_both_views_for_every_paired_team_and_mine_first():
    snap = snapshot(week=3)
    data = all_lineups.build(snap, None, sims=2000, other_sims=1000)
    assert data["teams"][0]["id"] == snap.my_team.team_id == data["mine"]
    for t in data["teams"]:
        assert set(t["views"]) == {"current", "ideal"}
        for v in t["views"].values():
            assert 0 <= v["win"] <= 100 and v["lineup"]
            assert sum(snap.starting_slots.values()) >= len(v["lineup"])


def test_ideal_faces_the_opponents_best_projected_lineup():
    snap = snapshot(week=3)
    me, opp = snap.my_team, snap.opponent
    cur = lineup.optimise(me.roster, snap.starting_slots, opp.roster, n_sims=2000)
    ideal = lineup.optimise(me.roster, snap.starting_slots, opp.roster, n_sims=2000, opponent_mode="ideal")
    assert ideal.opponent.expected >= cur.opponent.expected - 1e-9
