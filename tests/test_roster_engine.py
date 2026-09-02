"""Lineup optimiser, waiver engine and matchup report on the synthetic league.

These assert *properties* rather than numbers: an out player never starts, the
underdog leans on variance, a genuine upgrade is recommended and a fake one is
not. The numbers themselves come from Monte Carlo and are only stable to the
tolerance the tests allow.
"""

from __future__ import annotations

from fixtures_league import SLOTS, my_roster, opponent_roster, snapshot
from streamer.league.model import PlayerRow
from streamer.roster import lineup, waivers
from streamer.roster.matchup import build_report

STARTING = {k: v for k, v in SLOTS.items()}


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
def test_enumeration_fills_every_slot_and_dedupes():
    lineups = lineup.enumerate_lineups(my_roster(), STARTING)
    assert lineups
    for lu in lineups:
        for slot, n in STARTING.items():
            assert len(lu[slot]) == n, slot
    # Same starting set in different slot arrangements counts once.
    sets = [frozenset(p.player_id for ps in lu.values() for p in ps) for lu in lineups]
    assert len(sets) == len(set(sets))


def test_out_and_bye_players_never_start():
    lineups = lineup.enumerate_lineups(my_roster(), STARTING)
    for lu in lineups:
        for ps in lu.values():
            for p in ps:
                assert not p.is_out, p.name


def test_slot_eligibility_is_respected():
    lineups = lineup.enumerate_lineups(my_roster(), STARTING)
    for lu in lineups:
        assert all(p.position == "QB" for p in lu["QB"])
        assert all(p.position in ("RB", "WR", "TE") for p in lu["FLEX"])
        assert all(p.position == "DST" for p in lu["DST"])


def test_a_short_roster_still_produces_a_lineup():
    thin = [p for p in my_roster() if p.position != "TE"]   # no tight end at all
    lineups = lineup.enumerate_lineups(thin, STARTING)
    assert lineups
    assert all(len(lu["TE"]) == 0 for lu in lineups)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
def test_optimiser_beats_or_matches_the_current_lineup():
    result = lineup.optimise(my_roster(), STARTING, opponent_roster(), n_sims=8000)
    assert result.current is not None
    assert result.best_win.win_probability >= result.current.win_probability - 0.01
    assert result.best_ev.expected >= result.current.expected - 1e-9
    assert result.n_lineups > 1


def test_win_probability_responds_to_opponent_strength():
    weak = lineup.optimise(my_roster(), STARTING, opponent_roster(0.6), n_sims=8000)
    strong = lineup.optimise(my_roster(), STARTING, opponent_roster(1.5), n_sims=8000)
    assert weak.best_win.win_probability > 0.75
    assert strong.best_win.win_probability < 0.25


def test_underdog_prefers_the_high_variance_receiver():
    """WR 7 (mean 11, sd 11) vs WR 6 (mean 11, sd 6): identical EV.

    Against a much stronger opponent only a boom wins, so the optimiser should
    lean on the volatile one; against a weak opponent it should not.
    """
    def picked_boom(strength: float) -> bool:
        res = lineup.optimise(my_roster(), STARTING, opponent_roster(strength), n_sims=12000)
        return "7" in res.best_win.player_ids

    assert picked_boom(1.6) is True
    assert picked_boom(0.5) is False


def test_optimiser_without_an_opponent_still_returns_lineups():
    res = lineup.optimise(my_roster(), STARTING, None, n_sims=2000)
    assert res.opponent is None
    assert res.best_ev.expected > 0


def test_changes_pair_benched_with_started():
    res = lineup.optimise(my_roster(), STARTING, opponent_roster(1.6), n_sims=6000)
    for slot, benched, started in res.changes:
        assert slot in STARTING
        assert started.player_id in res.best_win.player_ids
        if benched is not None:
            assert benched.player_id not in res.best_win.player_ids


def test_sampling_floors_skill_players_but_not_dst():
    import numpy as np

    rng = np.random.default_rng(1)
    wr = PlayerRow(player_id="w", name="w", position="WR", projection=2.0, projection_sd=10.0)
    dst = PlayerRow(player_id="d", name="d", position="DST", projection=2.0, projection_sd=10.0)
    out = PlayerRow(player_id="o", name="o", position="RB", projection=15.0, projection_sd=5.0, status="OUT")
    s = lineup.sample_points([wr, dst, out], 5000, rng)
    assert s[:, 0].min() >= 0.0
    assert s[:, 1].min() < 0.0
    assert (s[:, 2] == 0.0).all()


# ---------------------------------------------------------------------------
# Waivers
# ---------------------------------------------------------------------------
def test_horizon_weight_rises_through_the_season():
    assert waivers.horizon_weight(1) < waivers.horizon_weight(9) < waivers.horizon_weight(17)
    assert 0.3 < waivers.horizon_weight(1) < 0.45
    assert waivers.horizon_weight(18) <= 0.9


def test_recommends_the_clear_upgrades_and_drops_the_dead_spot():
    moves = waivers.recommend(snapshot(week=5))
    adds = {m.add.player_id: m for m in moves}
    assert "201" in adds                     # RB 11.5 over the 3.0 OUT receiver
    assert "205" in adds                     # better D/ST stream
    assert moves[0].score > 0
    # The first drop offered should be the dead roster spot, not a starter.
    first_drop_ids = [m.drop.player_id for m in moves[:3] if m.drop]
    assert "14" in first_drop_ids


def test_never_drops_the_only_quarterback_for_a_receiver():
    snap = snapshot(week=5)
    moves = waivers.recommend(snap)
    for m in moves:
        if m.drop and m.drop.position == "QB":
            assert m.add.position == "QB", m


def test_never_drops_the_best_player_at_a_position():
    snap = snapshot(week=5)
    moves = waivers.recommend(snap)
    best_rb = max((p for p in snap.my_team.roster if p.position == "RB"),
                  key=lambda p: p.projection or 0)
    assert all(not (m.drop and m.drop.player_id == best_rb.player_id) for m in moves)


def _only_stash(week: int):
    snap = snapshot(week=week)
    snap.free_agents = [p for p in snap.free_agents if p.player_id == "207"]
    return snap


def test_bye_week_stash_is_tagged_and_kept_late_season_out():
    early = {m.add.player_id: m for m in waivers.recommend(_only_stash(3))}
    assert "207" in early and early["207"].tag == "stash"
    assert "on bye" in early["207"].reason
    # Late in the season a bye-week body is worth far less.
    late = {m.add.player_id: m for m in waivers.recommend(_only_stash(17))}
    assert early["207"].score > late.get("207", early["207"]).score - 1e-9


def test_moves_use_distinct_roster_spots():
    moves = waivers.recommend(snapshot(week=3))
    drops = [m.drop.player_id for m in moves]
    assert len(drops) == len(set(drops))
    # Each move is priced on top of the previous ones, so the list is a
    # sequence that can all be made together, in descending marginal value.
    scores = [m.score for m in moves]
    assert scores == sorted(scores, reverse=True)


def test_backup_quarterback_is_worth_little_when_the_wire_is_deep():
    """A 16-point QB on the bench of a one-QB league is not a +16 move."""
    snap = snapshot(week=5)
    extra = [p for p in snap.free_agents if p.player_id == "204"][0]
    twin = PlayerRow(**{**extra.__dict__, "player_id": "299", "name": "QB Player 299",
                        "projection": 15.5, "ros_value": 15.5})
    snap.free_agents.append(twin)
    moves = {m.add.player_id: m for m in waivers.recommend(snap)}
    assert "204" not in moves or moves["204"].score < 2.0


def test_ir_slot_players_are_neither_dropped_nor_started():
    snap = snapshot(week=5)
    stashed = PlayerRow(player_id="50", name="RB Player 50", position="RB", team="KC",
                        status="INJURY_RESERVE", slot="IR", eligible_slots=["RB", "FLEX"],
                        projection=14.0, projection_sd=7.0, ros_value=14.0)
    snap.my_team.roster.append(stashed)
    moves = waivers.recommend(snap)
    assert all(m.drop.player_id != "50" for m in moves)
    assert "50" not in {p.player_id for p in waivers.drop_watch(snap, n=20)}
    opt = lineup.optimise(snap.my_team.roster, snap.starting_slots, None, n_sims=500)
    assert "50" not in opt.best_win.player_ids
    # A day-to-day player parked on IR needs a roster move, and says so.
    stashed.status = "DAY_TO_DAY"
    notes = waivers.ir_notes(snap)
    assert len(notes) == 1 and "RB Player 50" in notes[0] and "DAY_TO_DAY" in notes[0]
    assert stashed.is_out is False and stashed.is_questionable is True


def test_junk_free_agents_are_not_recommended():
    moves = waivers.recommend(snapshot(week=5))
    for m in moves:
        assert (m.add.projection or 0) > 3.0 or (m.add.ros_value or 0) > 3.0


def test_drop_watch_surfaces_the_worst_spots():
    watch = waivers.drop_watch(snapshot(week=5), n=2)
    assert {p.player_id for p in watch} == {"14", "8"}


# ---------------------------------------------------------------------------
# Matchup report
# ---------------------------------------------------------------------------
def test_matchup_report_assembles(cfg):
    report = build_report(snapshot(week=5), cfg)
    assert report.opponent_name == "Rival"
    assert 0.0 <= report.win_probability <= 1.0
    assert report.verdict() in ("favoured", "slight favourite", "coin flip", "slight underdog", "underdog")
    assert len(report.swing_players(2)) == 2
    assert report.current_win_probability is not None


def test_matchup_without_opponent_notes_it(cfg):
    snap = snapshot(week=5)
    snap.matchup = None
    report = build_report(snap, cfg)
    assert any("no matchup" in n for n in report.notes)


# ---------------------------------------------------------------------------
# Page panel
# ---------------------------------------------------------------------------
def test_my_team_panel_renders(cfg):
    from streamer.roster.page import render_my_team

    snap = snapshot(week=5)
    report = build_report(snap, cfg)
    moves = waivers.recommend(snap)
    html = render_my_team(snap, report, moves, cfg)
    assert "<h2>My team</h2>" in html
    assert "Rival" in html
    assert "P(win)" in html
    # The best pickup and the dead roster spot both appear in the waiver list.
    assert moves and moves[0].add.name in html
    assert "<script" not in html
