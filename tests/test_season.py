"""Rest-of-season simulation: playoffs and title odds."""

from __future__ import annotations

import numpy as np
import pytest

from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow
from streamer.roster import season as season_mod
from streamer.roster.season import SeasonModel, bracket_order


def test_bracket_order_is_the_standard_fixed_bracket():
    order = bracket_order(8)
    pairs = [tuple(sorted(order[i:i + 2])) for i in range(0, 8, 2)]
    assert pairs == [(1, 8), (4, 5), (2, 7), (3, 6)]          # 1 meets the 4/5 winner
    assert set(order[:4]) == {1, 8, 4, 5}
    assert bracket_order(4) == [1, 4, 2, 3]


def _team(tid: str, strength: float, mine: bool = False, wins: int = 0) -> TeamRow:
    roster = []
    for pos, n in (("QB", 1), ("RB", 3), ("WR", 3), ("TE", 1)):
        for k in range(n):
            pid = f"{tid}-{pos}{k}"
            base = strength * {"QB": 18, "RB": 12, "WR": 12, "TE": 8}[pos] * (1 - 0.15 * k)
            roster.append(PlayerRow(player_id=pid, name=pid, position=pos, team="KC", projection=base,
                                    projection_sd=6.0, outcome_sd=6.0, ros_value=base,
                                    eligible_slots=[pos, "FLEX"] if pos != "QB" else ["QB"]))
    return TeamRow(team_id=tid, name=f"Team {tid}", roster=roster, is_mine=mine, wins=wins, losses=2 - wins)


def _league(strengths, median=False) -> LeagueSnapshot:
    teams = [_team(str(i), s, mine=(i == 0)) for i, s in enumerate(strengths)]
    ids = [t.team_id for t in teams]
    schedule = []
    for w in range(3, 11):                     # round robin, rotating
        rot = ids[:1] + ids[1:][w % (len(ids) - 1):] + ids[1:][:w % (len(ids) - 1)]
        for i in range(len(rot) // 2):
            schedule.append([w, rot[i], rot[-1 - i]])
    return LeagueSnapshot(
        platform="espn", profile="espn", league_id="1", league_name="L", season=2026, week=3,
        slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}, bench_size=2, teams=teams,
        free_agents=[], matchup=None, synced_at="",
        rules={"regular_season_weeks": 10, "playoff_teams": 4, "playoff_round_weeks": 1,
               "median_game": median, "schedule": schedule})


@pytest.fixture(autouse=True)
def _no_byes(monkeypatch):
    monkeypatch.setattr(season_mod, "nfl_byes", lambda *a, **k: {})


def test_odds_add_up_and_the_strongest_team_is_the_favourite(cfg):
    snap = _league([1.0, 1.3, 0.9, 0.8, 1.0, 0.95])
    odds = SeasonModel(snap, cfg, n_sims=1500).odds()
    assert odds.p_title.sum() == pytest.approx(1.0)
    assert odds.p_playoffs.sum() == pytest.approx(4.0)
    assert odds.p_title.argmax() == 1
    assert odds.p_title[1] > 2 * odds.p_title[3]


def test_a_better_roster_raises_your_title_odds_on_the_same_futures(cfg):
    snap = _league([1.0, 1.1, 1.0, 1.0])
    model = SeasonModel(snap, cfg, n_sims=1500)
    base = model.odds().p_title[0]
    me = snap.teams[0]
    better = [p.player_id for p in me.roster if p.position != "WR"] + \
             [p.player_id for p in snap.teams[1].roster if p.position == "WR"]
    boosted = model.odds(override={"0": model.team_scores(better)}).p_title[0]
    assert boosted > base


def test_espn_tiebreak_key_in_rules(cfg):
    from types import SimpleNamespace as NS

    from streamer.league import espn

    league = NS(teams=[], settings=NS(reg_season_count=14, playoff_team_count=6,
                                      playoff_matchup_period_length=1,
                                      playoff_seed_tie_rule="H2H_RECORD", median_scoring=False,
                                      faab=False, acquisition_budget=0))
    assert espn.league_rules(league)["tiebreak"] == "h2h"


def test_reseeding_pairs_the_best_seed_left_with_the_worst(cfg):
    import numpy as np

    snap = _league([1.0, 1.1, 1.0, 1.0, 0.9, 0.95])
    snap.rules.update(playoff_teams=6, reseed=True)
    model = SeasonModel(snap, cfg, n_sims=4)
    # After round one of a six-team bracket: seeds 1 and 2 had byes; in sim 0
    # seeds 4 and 3 won, in sim 1 seeds 5 and 6 did (team index = seed * 10).
    alive = [np.array([10, 10, 10, 10]), np.array([40, 50, 40, 50]),
             np.array([30, 60, 60, 30]), np.array([20, 20, 20, 20])]
    seeds = [1, np.array([4, 5, 4, 5]), np.array([3, 6, 6, 3]), 2]
    teams, s = model._reseeded(alive, seeds)
    pairs = [sorted((int(s[i][k]), int(s[i + 1][k]))) for k in range(4) for i in (0, 2)]
    assert pairs[:2] == [[1, 4], [2, 3]]          # 1 meets the worst seed left
    assert pairs[2:4] == [[1, 6], [2, 5]]
    assert all(int(t[k]) == int(sd[k]) * 10 for t, sd in zip(teams, s) for k in range(4))
    odds = model.odds()
    assert odds.p_title.sum() == pytest.approx(1.0) and odds.p_playoffs.sum() == pytest.approx(6.0)


def test_a_mid_week_pickup_cannot_use_points_already_scored(cfg):
    import numpy as np

    snap = _league([1.0, 1.1, 1.0, 1.0])
    star = snap.teams[1].roster[1]                 # someone else's back, game already played
    star.locked, star.actual_points = True, 40.0
    mine = snap.teams[0]
    mine.roster[1].locked, mine.roster[1].actual_points = True, 3.0   # my back, played, scored 3
    model = SeasonModel(snap, cfg, n_sims=500)
    swapped = [p.player_id for p in mine.roster if p is not mine.roster[1]] + [star.player_id]
    naive = model.team_scores(swapped)
    settled = model.team_scores(swapped, owner=mine.team_id)
    base = model.team_scores([p.player_id for p in mine.roster], owner=mine.team_id)
    # This week: my own finished score stands, the newcomer's 40 does not count.
    assert np.allclose(settled[:, 0], base[:, 0])
    assert naive[:, 0].mean() > settled[:, 0].mean() + 20
    # From next week he plays for me.
    assert np.allclose(settled[:, 1:], naive[:, 1:])


def test_a_week_the_standings_have_not_counted_is_played_out(cfg):
    snap = _league([1.0, 1.1, 1.0, 1.0])
    snap.week = 4                                  # records show 2 games: week 3 not counted yet
    model = SeasonModel(snap, cfg, n_sims=500)
    assert model.lag == 1 and model.weeks[0] == 3 and model.notes
    odds = model.odds()
    # Everyone is 0-2; weeks 3-10 are all played, week 3 included: 8 weeks x 2 games.
    assert odds.exp_wins.sum() == pytest.approx(8 * len(snap.teams) / 2)


def test_winning_a_game_never_hurts_and_more_wins_get_in_more(cfg):
    snap = _league([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    snap.rules["playoff_teams"] = 3
    model = SeasonModel(snap, cfg, n_sims=2000)
    st = model.stakes("0")
    assert st.games and all(g.playoffs_win >= g.playoffs_loss for g in st.games)
    assert all(0 <= g.p_win <= 1 for g in st.games)
    assert abs(st.p_playoffs - model.odds().p_playoffs[0]) < 1e-9
    # More wins, better odds.
    ks = sorted(k for k, (_p, n) in st.by_wins.items() if n >= 30)
    ps = [st.by_wins[k][0] for k in ks]
    assert ps == sorted(ps)


def test_the_stakes_table_renders(cfg):
    from types import SimpleNamespace

    from streamer.roster.page import _stakes

    snap = _league([1.0, 1.1, 0.9, 1.0])
    st = SeasonModel(snap, cfg, n_sims=500).stakes("0")
    html = _stakes(SimpleNamespace(stakes=st))
    assert "Must-win weeks" in html and html.count("<tr") == len(st.games) + 1
    assert _stakes(SimpleNamespace(stakes=None)) == ""


def test_who_loses_his_season_is_read_from_his_season_so_far():
    from streamer.roster import outcome
    from streamer.roster.futures import season_loss

    conf = outcome.load().get("futures", {})
    star = PlayerRow(player_id="s", name="S", position="WR", team="SEA", role="WR1", ros_value=16.0)
    fringe = PlayerRow(player_id="f", name="F", position="TE", team="SEA", role="TE3", ros_value=4.0, games_missed=2)
    a = season_loss(star, played=5, team_games=5, prev_games=17, conf=conf)
    b = season_loss(fringe, played=1, team_games=5, prev_games=3, conf=conf)
    assert 0 < a < 0.12 < b <= 0.9                  # an every-week starter less than the base, a fringe TE more
    star.on_bye = True
    assert season_loss(star, played=5, team_games=5, prev_games=17, conf=conf) == a   # a bye is not an absence
    star.status = "OUT"
    assert season_loss(star, played=5, team_games=5, prev_games=17, conf=conf) is None


def test_season_loss_reads_this_season_and_last_from_the_box_scores():
    import pandas as pd

    from streamer.roster.projections import season_loss_counts

    rows = [{"player_id": "a", "team": "SEA", "season": 2026, "week": w} for w in (1, 2, 4)] + \
        [{"player_id": "b", "team": "SEA", "season": 2026, "week": w} for w in (1, 2, 3, 4)] + \
        [{"player_id": "a", "team": "SEA", "season": 2025, "week": w} for w in range(1, 7)]
    history = pd.DataFrame(rows)
    a = PlayerRow(player_id="ea", name="A", position="WR", team="SEA")
    b = PlayerRow(player_id="eb", name="B", position="WR", team="SEA")
    gone = PlayerRow(player_id="ec", name="C", position="WR", team=None)
    a.nfl_id, b.nfl_id, gone.nfl_id = "a", "b", "c"
    got = season_loss_counts([a, b, gone], history, 2026, 5)
    assert got == {"ea": (3, 4, 6), "eb": (4, 4, None)}      # b: no games last season; c: on no team


def test_a_backup_quarterback_takes_the_job_or_sits_not_a_scattering_of_games(monkeypatch):

    from streamer.roster import futures

    conf = futures._params()
    qb = PlayerRow(player_id="q", name="Q", position="QB", team="SEA", ros_value=6.0, projection=6.0)

    def seasons(extra):
        monkeypatch.setattr(futures, "_params", lambda: {**conf, "role_spread": {}, **extra})
        played = futures.simulate([qb], list(range(6, 18)), {}, n_sims=4000, seed=3).played[:, 0, :].mean(axis=1)
        return played.mean(), (played >= 0.9).mean() + (played <= 0.1).mean()

    flat_mean, flat_ends = seasons({})
    held_mean, held_ends = seasons({k: conf[k] for k in ("role_spread", "role_scale", "role_corr")})
    assert held_ends > flat_ends + 0.05                       # all or nothing more often
    assert abs(held_mean - flat_mean) < 0.05                  # about as many games on average


# ---------------------------------------------------------------------------
# The wire, week by week
# ---------------------------------------------------------------------------
def _fa(pid, pos, value, team):
    return PlayerRow(player_id=pid, name=pid, position=pos, team=team, projection=value,
                     projection_sd=5.0, outcome_sd=5.0, ros_value=value,
                     eligible_slots=[pos, "FLEX"] if pos != "QB" else ["QB"])


def test_a_free_agent_on_bye_fills_nothing_that_week(cfg, monkeypatch):
    snap = _league([1.0, 1.0, 1.0, 1.0])
    snap.free_agents = [_fa("te1", "TE", 9.0, "BUF"), _fa("te2", "TE", 8.0, "BUF"), _fa("te3", "TE", 4.0, "MIA")]
    monkeypatch.setattr(season_mod, "nfl_byes", lambda *a, **k: {"BUF": {5}})
    model = SeasonModel(snap, cfg, n_sims=200)
    te = model._rep_table["TE"]
    assert te[model.weeks.index(4), 0] == 8.0          # the second best, as before
    assert te[model.weeks.index(5), 0] == 4.0          # both BUF tight ends on bye
    assert model.slot_replacement("TE") == 8.0


def test_the_flex_level_without_competition_is_the_old_one(cfg):
    snap = _league([1.0, 1.0, 1.0, 1.0])
    snap.free_agents = [_fa("w1", "WR", 10.0, "KC"), _fa("w2", "WR", 7.0, "KC"),
                        _fa("t1", "TE", 9.0, "KC"), _fa("t2", "TE", 6.5, "KC")]
    model = SeasonModel(snap, cfg, n_sims=200)
    assert model._rep_table["FLEX"][0, 0] == model.slot_replacement("FLEX") == 7.0


def test_rivals_with_the_same_hole_leave_a_worse_fill(cfg):
    snap = _league([1.0, 1.0, 1.0, 1.0])
    snap.free_agents = [_fa(f"te{k}", "TE", 30.0 - 4 * k, "KC") for k in range(8)]   # far above any rostered TE
    model = SeasonModel(snap, cfg, n_sims=300)
    alone = model._wire_level("TE", 0, compete=False)
    shared = model._wire_level("TE", 0)
    assert model._demand["TE"].mean() == pytest.approx(4.0)                 # every team streams the TE
    assert (shared <= alone).all() and shared.mean() < alone.mean() - 2.0


def test_thinning_off_is_the_old_flat_level(cfg):
    snap = _league([1.0, 1.1, 0.9, 1.0])
    snap.free_agents = [_fa("te1", "TE", 9.0, "BUF"), _fa("te2", "TE", 8.0, "BUF")]
    saved = dict(cfg.raw["roster"])
    try:
        cfg.raw["roster"]["wire_thinning"] = False
        model = SeasonModel(snap, cfg, n_sims=200)
        assert not model._rep_table and model._wire_level("TE", 0) == model.slot_replacement("TE") == 8.0
    finally:
        cfg.raw["roster"].clear()
        cfg.raw["roster"].update(saved)


def test_division_winners_are_seeded_first_and_the_rest_by_record():
    import numpy as np

    from streamer.roster.season import division_seeds

    # Six teams, two divisions; one season. Division B's best (team 5) has
    # the fourth-best record but is seeded second, ahead of teams 1 and 2.
    key = np.array([[90.0], [80.0], [70.0], [10.0], [20.0], [60.0]])
    divs = ["A", "A", "A", "B", "B", "B"]
    seeds = division_seeds(key, divs, 4)[:, 0].tolist()
    assert seeds == [0, 5, 1, 2]


def test_a_weak_division_winner_gets_the_bye(cfg):
    snap = _league([1.3, 1.25, 1.2, 1.15, 0.8, 0.75, 0.7, 0.65])
    snap.rules["playoff_teams"] = 6
    for i, t in enumerate(snap.teams):
        t.division = "strong" if i < 4 else "weak"
    league = cfg.profiles[cfg.profile].setdefault("league", {})
    was = league.get("division_byes")
    try:
        league["division_byes"] = False
        plain = SeasonModel(snap, cfg, n_sims=800).odds()
        league["division_byes"] = True
        divided = SeasonModel(snap, cfg, n_sims=800).odds()
    finally:
        league["division_byes"] = was
    weak = [4, 5, 6, 7]
    assert divided.p_bye[weak].sum() == pytest.approx(1.0)        # exactly one weak-division team each season
    assert plain.p_bye[weak].sum() < 0.5


# ---------------------------------------------------------------------------
# The bracket, exactly: hand-set scores, one known answer each
# ---------------------------------------------------------------------------
def _bracket_model(cfg, teams: int, playoff_teams: int, round_weeks: int = 1, reseed: bool = False,
                   playoff_tiebreak: str = "higher seed"):
    import numpy as np

    snap = _league([1.0] * teams)
    snap.rules.update(playoff_teams=playoff_teams, playoff_round_weeks=round_weeks, reseed=reseed,
                      playoff_tiebreak=playoff_tiebreak)
    m = SeasonModel(snap, cfg, n_sims=2)
    # Seed k (1-based) is team k-1: wins fall with the index.
    wins = np.array([[float(teams - i)] * 2 for i in range(teams)])
    points = np.zeros((teams, 2))
    scores = np.zeros((teams, 2, len(m.weeks)))
    first = m.weeks.index(m.reg_weeks + 1)
    return m, wins, points, scores, first


def _champ(m, wins, points, scores) -> list[int]:
    made, byes, champ = m._playoffs(scores, wins, points)
    assert made.sum(axis=0).tolist() == [m.playoff_teams] * m.n          # exactly the field, every season
    return [int(c) for c in champ]                                     # exactly one champion each


def test_a_fixed_bracket_and_a_reseeded_one_differ_as_they_should(cfg):
    for reseed, expected in ((False, 7), (True, 2)):
        m, wins, points, scores, k = _bracket_model(cfg, 8, 8, reseed=reseed)
        r1 = [150, 140, 130, 120, 110, 100, 90, 200]        # seed 8 upsets seed 1; elsewhere the higher seed
        r2 = [0, 120, 110, 90, 0, 0, 0, 100]
        r3 = [0, 50, 250, 0, 0, 0, 0, 300]
        for t in range(8):
            scores[t, :, k], scores[t, :, k + 1], scores[t, :, k + 2] = r1[t], r2[t], r3[t]
        # Fixed: 8 meets the 4/5 winner (beats 4), 2 beats 3; 8 beats 2 in the final.
        # Reseeded: the best left (2) meets the worst (8) and wins; 3 beats 4; 3 beats 2.
        assert _champ(m, wins, points, scores) == [expected] * 2


def test_a_tied_playoff_game_goes_to_the_higher_seed_or_season_points(cfg):
    m, wins, points, scores, k = _bracket_model(cfg, 4, 2)
    scores[0, :, k] = scores[1, :, k] = 100.0
    assert _champ(m, wins, points, scores) == [0, 0]                      # seed 1 on the tie
    m, wins, points, scores, k = _bracket_model(cfg, 4, 2, playoff_tiebreak="points")
    scores[0, :, k] = scores[1, :, k] = 100.0
    points[1] = 900.0                                                   # seed 2 scored more all season
    assert _champ(m, wins, points, scores) == [1, 1]


def test_a_two_week_round_is_decided_on_both_weeks(cfg):
    m, wins, points, scores, k = _bracket_model(cfg, 4, 2, round_weeks=2)
    scores[0, :, k], scores[1, :, k] = 130.0, 100.0                     # seed 1 wins week one ...
    scores[0, :, k + 1], scores[1, :, k + 1] = 80.0, 120.0              # ... seed 2 the round, 220 to 210
    assert _champ(m, wins, points, scores) == [1, 1]


def test_a_move_that_changes_nothing_moves_the_title_odds_by_exactly_zero(cfg):
    import numpy as np

    from streamer.roster.title_moves import TitleEngine

    snap = _league([1.0, 1.1, 1.0, 1.05, 0.95, 1.0])
    engine = TitleEngine(snap, cfg, n_sims=500, candidates=4)
    same = engine._won([p.player_id for p in engine.me.roster])
    assert np.array_equal(same, engine.base_won)
    odds = engine.model.odds()
    assert odds.p_title.sum() == pytest.approx(1.0) and np.all(odds.champion >= 0)


def test_the_simulation_noise_interval_is_wilsons():
    from streamer.roster.season import wilson

    lo, hi = wilson(0.10, 6000)
    assert lo == pytest.approx(0.0927, abs=2e-4) and hi == pytest.approx(0.1078, abs=2e-4)   # about +-0.76 points
    lo, hi = wilson(0.10, 1500)
    assert hi - lo == pytest.approx(0.0305, abs=0.001)            # about +-1.5 at 1,500
    assert wilson(0.0, 6000)[0] == pytest.approx(0.0, abs=1e-12) and wilson(1.0, 6000)[1] == pytest.approx(1.0)


def test_the_season_simulation_draws_each_nfl_game_together(cfg, monkeypatch):
    """With season_correlation on, SeasonModel hands the simulator this
    season's NFL games."""
    seen = {}
    real = season_mod.fut.simulate

    def spy(*a, **k):
        seen["opponents"] = k.get("opponents")
        return real(*a, **k)

    monkeypatch.setattr(season_mod, "nfl_opponents", lambda *a, **k: {"KC": {3: "BUF"}, "BUF": {3: "KC"}})
    monkeypatch.setattr(season_mod.fut, "simulate", spy)
    SeasonModel(_league([1.0, 1.0, 1.0, 1.0]), cfg, n_sims=200)
    assert seen["opponents"] == {"KC": {3: "BUF"}, "BUF": {3: "KC"}}


def test_a_fading_edge_pulls_the_favourite_back_but_keeps_what_a_move_adds(cfg):
    """strength_fade: a team's edge over the league shrinks week by week ahead
    (claims and trades level rosters); this week's games are untouched, and a
    roster change's own points pass through whole."""
    snap = _league([1.0, 1.3, 0.9, 0.8, 1.0, 0.95])
    model = SeasonModel(snap, cfg, n_sims=1500)
    full, faded = model.odds(retention=1.0), model.odds(retention=0.9)
    assert faded.p_title[1] < full.p_title[1]                      # the favourite
    assert faded.p_title.sum() == pytest.approx(1.0) and faded.p_playoffs.sum() == pytest.approx(4.0)
    fade = model._fade(0.9)
    assert np.allclose(fade[:, 0, model.lag], 0.0)                 # this week as projected
    assert np.allclose(fade.sum(axis=0), 0.0)                      # the league as a whole keeps its points
    # A move's own points are not faded: the same shift applies to any override.
    i = model.team_index["0"]
    better = model.base_scores[i] + 5.0
    gained = (model._scores({"0": better}, 0.9) - model._scores(None, 0.9))[i]
    assert np.allclose(gained, 5.0)


def test_power_is_record_blind_and_orders_teams_by_how_often_they_outscore_the_rest(cfg):
    from streamer.roster.season import all_play

    scores = np.zeros((3, 4, 2))
    scores[0] = 100.0
    scores[1] = 90.0
    scores[2] = 80.0
    assert list(all_play(scores)) == [1.0, 0.5, 0.0]
    assert list(all_play(np.full((3, 4, 2), 70.0))) == [0.5, 0.5, 0.5]            # ties are half
    snap = _league([1.0, 1.3, 0.9, 0.8, 1.0, 0.95])
    snap.teams[3].wins, snap.teams[3].losses = 7, 0                                # a hot record, a weak roster
    odds = SeasonModel(snap, cfg, n_sims=800).odds()
    assert odds.power is not None and odds.power.argmax() == 1                     # the strongest roster
    assert odds.power[3] < odds.power[0]
    assert odds.power.mean() == pytest.approx(0.5, abs=1e-6)


def test_points_a_week_ahead_and_so_far_are_reported(cfg):
    snap = _league([1.0, 1.3, 0.9, 0.8, 1.0, 0.95])
    for i, t in enumerate(snap.teams):
        t.wins, t.losses, t.points_for = 1, 1, 200.0 + 10 * i
    odds = SeasonModel(snap, cfg, n_sims=300).odds()
    assert list(odds.pts_so_far) == [100.0, 105.0, 110.0, 115.0, 120.0, 125.0]
    assert odds.pts_ahead is not None and odds.pts_ahead.argmax() == 1             # the strongest roster
    assert odds.pts_ahead.sum() == pytest.approx((odds.exp_points / (snap.rules["regular_season_weeks"] - 2)).sum())
