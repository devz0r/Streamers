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


def _two_for_one_slot() -> list[PlayerRow]:
    """WR 6 (mean 11, sd 6) and WR 7 (mean 11, sd 11) contest the last WR
    spot: the flex is taken by a 12-point back, so only one of them starts."""
    roster = my_roster()
    for p in roster:
        if p.player_id == "11":
            p.projection, p.status = 12.0, ""
    return roster


def test_underdog_prefers_the_high_variance_receiver():
    """Identical EV. Against a much stronger opponent only a boom wins, so the
    optimiser leans on the volatile one; against a weak opponent it does not."""
    def picked_boom(strength: float) -> bool:
        res = lineup.optimise(_two_for_one_slot(), STARTING, opponent_roster(strength), n_sims=12000)
        return "7" in res.best_win.player_ids

    assert picked_boom(1.4) is True
    assert picked_boom(0.8) is False


def test_points_are_not_given_up_for_an_edge_the_simulation_cannot_see():
    """In a blowout the steadier player gains a sliver of P(win) at the cost
    of a projected point; the projection wins that tie."""
    res = lineup.optimise(my_roster(), STARTING, opponent_roster(0.5), n_sims=12000)
    assert res.best_win.expected == res.best_ev.expected
    assert res.reasons == []


def _stack_roster(opp_qb_team: str = "BUF") -> tuple[list[PlayerRow], list[PlayerRow]]:
    """Two receivers with the same projection and spread; one catches passes
    from our quarterback (KC), the other from the opponent's (BUF)."""
    roster = _two_for_one_slot()
    for p in roster:
        p.role = {"QB": "QB", "K": "K", "DST": "DST"}.get(p.position, p.position + "2")
        if p.player_id == "6":
            p.team, p.projection_sd, p.role = opp_qb_team, 8.0, "WR1"
        if p.player_id == "7":
            p.team, p.projection_sd, p.role = "KC", 8.0, "WR1"
    opp = opponent_roster(1.0)
    for p in opp:
        p.role = {"QB": "QB", "K": "K", "DST": "DST"}.get(p.position, p.position + "2")
    return roster, opp


def test_correlation_widens_a_stack_and_narrows_a_hedge():
    import numpy as np

    roster, opp = _stack_roster()
    rng = np.random.default_rng(3)
    qb, stack, hedge = roster[0], roster[6], roster[5]
    opp_qb = opp[0]
    s = lineup.sample_points([qb, stack, hedge, opp_qb], 40000, rng)
    assert np.corrcoef(s[:, 0], s[:, 1])[0, 1] > 0.25       # QB and his WR1
    assert np.corrcoef(s[:, 2], s[:, 3])[0, 1] > 0.25       # opponent's QB and his WR1
    assert abs(np.corrcoef(s[:, 0], s[:, 2])[0, 1]) < 0.05  # unrelated games


def test_underdog_stacks_and_favourite_hedges():
    """Same projection and spread: the underdog starts the receiver who moves
    with its own quarterback; the favourite the one who moves with the
    opponent's. The explanation names the reason."""
    def pick(strength: float) -> tuple[str, list[str]]:
        roster, opp = _stack_roster()
        for p in opp:
            if p.position != "QB":
                p.projection = (p.projection or 0) * strength
        res = lineup.optimise(roster, STARTING, opp, n_sims=20000)
        return ("7" if "7" in res.best_win.player_ids else "6"), res.reasons

    under, _ = pick(1.5)
    fav, _ = pick(0.6)
    assert under == "7"
    assert fav == "6"


def test_a_questionable_player_is_a_coin_flip_in_the_simulation():
    import numpy as np

    q = PlayerRow(player_id="q", name="q", position="WR", status="QUESTIONABLE",
                  projection=7.5, projection_sd=8.0, outcome_sd=6.0, play_probability=0.75)
    s = lineup.sample_points([q], 40000, np.random.default_rng(2))[:, 0]
    assert 0.2 < (s == 0.0).mean() < 0.3          # sits about a quarter of the time
    assert abs(s.mean() - 7.5) < 0.4              # the mixture mean is kept


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
    # The comparison works without script; only the lineup editor uses it.
    assert "Best P(win)" in html and "Most points" in html
    assert html.count("<script") == 2 and 'class="ed-data"' in html


def test_sync_failure_panel_says_what_went_wrong(cfg):
    from streamer.roster.page import render_sync_failure

    status = {"platform": "yahoo", "week": 2, "ok": False,
              "at": "2026-09-15T13:27:31+00:00",
              "error": "Yahoo rejected the refresh token. Run `streamer yahoo-auth` again"}
    html = render_sync_failure(status, cfg)
    assert "YAHOO sync failed" in html
    assert "refresh token" in html
    assert "no team panel" in html
    # A stale snapshot is a different message: there IS a panel, just an old one.
    stale = render_sync_failure(status, cfg, stale_week=1)
    assert "week 1" in stale and "no team panel" not in stale


# ---------------------------------------------------------------------------
# Like-for-like waiver pricing
# ---------------------------------------------------------------------------
def test_waivers_use_model_basis_when_the_wire_has_no_platform_projection():
    """Roster blended with Yahoo, wire model-only: compare on the model for both."""
    snap = snapshot(week=5)
    for p in snap.my_team.roster:
        p.platform_projection = (p.projection or 0) + 5.0
        p.model_projection = p.projection
        p.projection = (p.projection or 0) + 2.5            # the blend is inflated
    for p in snap.free_agents:
        p.platform_projection = None
        p.model_projection = p.projection
    assert waivers.next_week_basis(snap) is waivers._next_week_model

    # With every side on the model, the moves are exactly the unblended ones.
    fair = {(m.add.player_id, m.drop.player_id): round(m.score, 6) for m in waivers.recommend(snap)}
    for p in snap.my_team.roster:
        p.projection = p.model_projection
    base = {(m.add.player_id, m.drop.player_id): round(m.score, 6) for m in waivers.recommend(snap)}
    assert fair == base


def test_waivers_keep_the_blend_when_both_sides_have_it():
    snap = snapshot(week=5)
    for p in [*snap.my_team.roster, *snap.free_agents]:
        p.platform_projection = p.projection
    assert waivers.next_week_basis(snap) is waivers._next_week


def test_model_projection_falls_back_to_projection():
    from streamer.league.model import PlayerRow

    p = PlayerRow(player_id="1", name="x", position="WR", projection=9.0)
    assert waivers._next_week_model(p) == 9.0
    p.model_projection = 7.0
    assert waivers._next_week_model(p) == 7.0


def test_a_pickup_that_would_start_names_who_it_benches():
    """'+3.5 this week' for a second QB is baffling until it says whose job it takes."""
    snap = snapshot(week=5)
    hot = [p for p in snap.free_agents if p.player_id == "204"][0]     # the FA QB
    hot.projection = hot.ros_value = 25.0                               # now beats QB1 (19.0)
    moves = {m.add.player_id: m for m in waivers.recommend(snap)}
    assert "204" in moves
    assert "would start over QB Player 1" in moves["204"].reason
    assert "25.0 vs 19.0" in moves["204"].reason

    # A pickup that only adds depth says nothing about starting.
    for m in moves.values():
        if m.add.player_id != "204" and m.tag == "stash":
            assert "would start over" not in m.reason


# ---------------------------------------------------------------------------
# Upside
# ---------------------------------------------------------------------------
def test_upside_is_an_option_worth_more_with_more_room_to_move():
    assert waivers.upside(8.0, 3.0, 12.0) > waivers.upside(8.0, 1.0, 12.0) > 0.0
    assert waivers.upside(8.0, 0.0, 12.0) == 0.0
    assert waivers.upside(14.0, 0.0, 12.0) == 2.0


def _wr(pid: str, ros: float, spread: float, slot: str = "BN") -> PlayerRow:
    return PlayerRow(player_id=pid, name=pid, position="WR", team="KC", slot=slot,
                     eligible_slots=["WR", "FLEX"], projection=ros, projection_sd=6.0,
                     ros_value=ros, ros_sd=spread)


def test_bench_upside_only_counts_jobs_he_could_take():
    """A bench receiver's upside is measured against the receiver he could
    replace, never against a weak tight end in a slot he cannot fill."""
    slots = {"WR": 1, "TE": 1}
    te = PlayerRow(player_id="te", name="te", position="TE", team="KC", slot="TE",
                   eligible_slots=["TE"], projection=3.0, ros_value=3.0, ros_sd=1.0)
    roster = [te, _wr("wr", 15.0, 2.0, slot="WR"), _wr("bench", 10.0, 2.0)]
    key = waivers._ros
    flat = waivers.roster_value(roster, slots, key, {})
    with_up = waivers.roster_value(roster, slots, key, {}, with_upside=True)
    assert 0.0 <= with_up - flat < 0.05                   # 10 vs 15: little chance


def test_the_riskier_stash_wins_at_the_same_projection():
    base = snapshot(week=4)
    for p in base.my_team.roster:
        p.ros_sd = 1.5
    steady = _wr("steady", 10.5, 1.0, slot="FA")
    rookie = _wr("rookie", 10.5, 3.5, slot="FA")
    rookie.experience = 0
    base.free_agents = [steady, rookie]
    moves = {m.add.player_id: m for m in waivers.recommend(base, min_gain=-99)}
    assert moves["rookie"].score > moves["steady"].score
    assert "rookie" in moves["rookie"].reason


def test_stash_list_names_a_handcuff():
    """Behind a bell-cow, the backup would start for you if he got the job."""
    snap = snapshot(week=4)
    for p in snap.my_team.roster:
        p.ros_sd = 1.5
    lead = PlayerRow(player_id="lead", name="Lead Back", position="RB", team="SEA", role="RB1",
                     projection=24.0, ros_value=24.0, ros_sd=2.0)
    cuff = PlayerRow(player_id="cuff", name="Cuff Back", position="RB", team="SEA", role="RB2",
                     slot="FA", eligible_slots=["RB", "FLEX"], projection=8.0, ros_value=8.0, ros_sd=1.6)
    snap.teams[1].roster.append(lead)
    snap.free_agents = [cuff]
    names = {s.player.player_id: s for s in waivers.stashes(snap, [])}
    assert "cuff" in names
    assert any("handcuff to Lead Back" in r for r in names["cuff"].reasons)


def test_a_handcuff_who_could_not_start_for_you_is_not_a_stash():
    snap = snapshot(week=4)
    for p in snap.my_team.roster:
        p.ros_sd = 1.5
    lead = PlayerRow(player_id="lead", name="Lead Back", position="RB", team="SEA", role="RB1",
                     projection=13.0, ros_value=13.0, ros_sd=2.0)
    cuff = PlayerRow(player_id="cuff", name="Cuff Back", position="RB", team="SEA", role="RB2",
                     slot="FA", eligible_slots=["RB", "FLEX"], projection=3.0, ros_value=3.0, ros_sd=1.2)
    snap.teams[1].roster.append(lead)
    snap.free_agents = [cuff]
    assert waivers.stashes(snap, []) == []
