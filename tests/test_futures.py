"""Simulated futures: how a player's outlook moves week by week."""

from __future__ import annotations

from streamer.league.model import PlayerRow
from streamer.roster.futures import simulate


def _p(pid, pos="WR", ros=12.0, team="KC", status="", sd=7.0):
    return PlayerRow(player_id=pid, name=pid, position=pos, team=team, status=status,
                     projection=ros, projection_sd=sd, outcome_sd=sd, ros_value=ros)


def test_uncertainty_about_his_level_grows_with_time():
    f = simulate([_p("a")], weeks=list(range(4, 16)), byes={}, n_sims=4000)
    lv = f.levels[:, 0, :]
    playing = lv > 0
    early = lv[:, 1][playing[:, 1]].std()
    late = lv[:, 10][playing[:, 10]].std()
    # Persistent error (~c*sqrt(12) = 5.2) is there from day one; weekly drift
    # (~0.95/week) adds to it: sqrt(5.2^2 + 10*0.95^2) = 6.0.
    assert late > early * 1.1


def test_byes_score_nothing_and_absences_happen_at_a_plausible_rate():
    f = simulate([_p("a", team="KC")], weeks=[5, 6, 7], byes={"KC": {6}}, n_sims=4000)
    assert (f.scores[:, 0, 1] == 0).all()
    missing = (f.levels[:, 0, 2] == 0).mean()
    assert 0.03 < missing < 0.25


def test_injured_reserve_starts_him_out_for_weeks():
    f = simulate([_p("a", status="INJURY_RESERVE")], weeks=[4, 5, 6, 7, 8], byes={}, n_sims=2000)
    assert (f.scores[:, 0, :4] == 0).all()


def test_the_next_man_up_gains_while_the_lead_is_out():
    lead, cuff = _p("lead", "RB", 18.0), _p("cuff", "RB", 5.0)
    f = simulate([lead, cuff], weeks=list(range(4, 12)), byes={}, n_sims=4000, heirs=[("lead", "cuff")])
    lead_out = f.levels[:, 0, :] == 0
    cuff_on = f.levels[:, 1, :] > 0
    with_job = f.levels[:, 1, :][lead_out & cuff_on].mean()
    without = f.levels[:, 1, :][~lead_out & cuff_on].mean()
    assert with_job > without + 2.0


def test_the_average_future_stays_near_todays_projection():
    f = simulate([_p("a", ros=12.0)], weeks=list(range(4, 10)), byes={}, n_sims=6000)
    assert 9.5 < f.scores[:, 0, :].mean() < 12.5          # absences pull it a little below


def test_a_rookies_outlook_moves_more_than_a_veterans():
    rookie, vet = _p("rookie"), _p("vet")
    rookie.experience, vet.experience = 0, 6
    f = simulate([rookie, vet], weeks=list(range(4, 16)), byes={}, n_sims=6000)
    k = 11
    spread = [f.levels[:, j, k][f.levels[:, j, k] > 0].std() for j in (0, 1)]
    assert spread[0] > spread[1] * 1.03


def test_a_takeover_varies_by_absence_and_can_stick():
    """Hall out now, Allen next up: how much of the job Allen gets is a draw
    for each absence, and when Hall is back Allen keeps part of it."""
    import numpy as np

    lead = _p("lead", "RB", 16.0, status="OUT")
    cuff = _p("cuff", "RB", 6.0)
    cuff.inherited_ros = 1.0                      # already in his season value: played out here instead
    f = simulate([lead, cuff], weeks=list(range(4, 14)), byes={}, n_sims=6000, heirs=[("lead", "cuff")],
                 takeover={"RB": 0.55}, takeover_sd={"RB": 0.8}, kept=0.26)
    first = f.levels[:, 1, 0][f.levels[:, 1, 0] > 0]
    # Week one: Hall is out in every future. Allen's level is 5 (6 less the
    # inherited point) plus a drawn share of the ~11-point gap.
    assert 9.5 < first.mean() < 12.5
    assert first.std() > 5.0                                  # bell cow in some futures, a bystander in others
    assert np.quantile(first, 0.9) > 16.0
    # Futures where Hall has been back for a while: Allen is above where he started.
    lead_back = (f.levels[:, 0, 0] == 0) & (f.levels[:, 0, 6:] > 0).all(axis=1)
    after = f.levels[lead_back, 1, 9]
    assert after[after > 0].mean() > 5.0 + 0.4
