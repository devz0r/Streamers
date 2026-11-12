"""The return model: when a player who is out plays again, read off his
injury report, and drawn in the season simulation in place of the average
absence."""

from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from streamer.league.model import PlayerRow
from streamer.roster import return_model as rm
from streamer.roster.futures import GONE, simulate

WEEKS = list(range(10, 18))


def _ctx(**kw):
    base = {"m": 2, "reserve": 0, "status": "O", "practice": "dnp", "stale": 0, "injury": "hamstring", "again": 0,
            "level": 12.0, "pos": "WR", "bye_first": 0}
    return {**base, **kw}


def test_the_fitted_model_is_shipped_and_reads_the_report_the_way_the_record_does():
    assert rm.load() is not None
    nxt = lambda **kw: float(rm.hazards(_ctx(**kw), [12])[0])  # noqa: E731
    assert nxt(practice="limited") > nxt(practice="dnp") + 0.05         # back at practice: likelier to play
    assert nxt(practice="limited", status="Q") > nxt() + 0.15           # ... and listed questionable, far likelier
    assert nxt(bye_first=1) > nxt() + 0.05                              # a bye is a week of healing
    assert nxt(reserve=1, m=1) < 0.05 < nxt()                           # a reserve list sits him at least four games
    assert nxt(again=1) < nxt()                                         # the same injury again comes back slower
    by = rm.back_by(_ctx(), list(range(12, 18)))
    assert (np.diff(by) > 0).all() and by[-1] < 1.0                     # some absences end the season


def test_injuries_and_practice_are_grouped_from_the_reports_wording():
    assert rm.injury_group("Right Hamstring") == "hamstring" and rm.injury_group("Thumb") == "hand"
    assert rm.injury_group(None) == "unknown" and rm.injury_group("Spleen") == "other"
    assert rm.practice_tag("Did Not Participate In Practice") == "dnp" and rm.practice_tag(None) == "none"


def test_the_draw_follows_the_chances_and_is_gone_past_the_last_game():
    c, targets = _ctx(), list(range(12, 18))
    got = rm.draw(c, targets, 20000, np.random.default_rng(1))
    by = rm.back_by(c, targets)
    assert abs((got == 0).mean() - by[0]) < 0.015 and abs((got <= 2).mean() - by[2]) < 0.015
    assert abs((got == rm.GONE).mean() - (1 - by[-1])) < 0.015
    assert (rm.draw(c, [], 10, np.random.default_rng(1)) == rm.GONE).all() and rm.GONE == GONE


def _p(status="OUT", on_bye=False, missed=2, ctx=None):
    p = PlayerRow(player_id="a", name="a", position="WR", team="LA", status=status, slot="BN", projection=0.0,
                  projection_sd=5.0, outcome_sd=5.0, ros_value=12.0)
    p.on_bye, p.games_missed, p.injury_ctx = on_bye, missed, ctx
    return p


def test_a_game_this_week_is_missed_for_certain_and_a_bye_is_not_a_game():
    info = {"injury": "hamstring", "again": 1, "report_week": 9, "practice": "dnp", "status": "O"}
    c, targets, certain = rm.checkpoint(_p(ctx=info), 10, [10, 11, 12])
    assert (certain, targets, c["m"], c["stale"], c["bye_first"]) == (1, [11, 12], 3, 1, 0)   # last week's practice: a week old
    c, targets, certain = rm.checkpoint(_p(on_bye=True, ctx=info), 10, [11, 12])
    assert (certain, targets, c["m"], c["stale"], c["bye_first"]) == (0, [11, 12], 2, 0, 1)   # the game after the bye is in question
    assert c["injury"] == "hamstring" and c["again"] == 1 and c["status"] == "O"
    c, _t, _n = rm.checkpoint(_p(status="IR", ctx={}), 10, [10, 11])
    assert c["reserve"] == 1 and c["practice"] == "none" and c["injury"] == "unknown"


def test_the_simulation_draws_his_return_from_the_report_when_it_was_read():
    limited = _p(ctx={"injury": "ankle", "again": 0, "report_week": 10, "practice": "limited", "status": "Q"})
    sat = _p(ctx={"injury": "ankle", "again": 0, "report_week": 10, "practice": "dnp", "status": "O"})
    sat.player_id = sat.name = "b"
    unread = _p(ctx=None)
    unread.player_id = unread.name = "c"
    out = simulate([limited, sat, unread], weeks=WEEKS, byes={}, n_sims=4000, seed=3).played
    assert not out[:, :2, 0].any()                                       # tagged out: this week is missed
    assert out[:, 0, 1].mean() > out[:, 1, 1].mean() + 0.1               # limited in practice: back sooner
    want = rm.back_by(rm.checkpoint(sat, 10, WEEKS)[0], WEEKS[1:])[0]
    assert abs(out[:, 1, 1].mean() - want) < 0.05                        # at the model's chance (less a new injury)


def test_contexts_read_only_the_reports_so_far_and_spot_the_same_injury_again():
    rows = [("x", 3, "Hamstring", "Out", "Did Not Participate In Practice"),
            ("x", 6, "Hamstring", "Out", "Limited Participation in Practice"),
            ("x", 7, "Knee", "Out", "Did Not Participate In Practice")]
    inj = pd.DataFrame(rows, columns=["gsis_id", "week", "report_primary_injury", "report_status", "practice_status"])
    inj["practice_primary_injury"] = None
    got = rm.contexts(inj, {"LA": [1, 2, 3, 4, 5, 6, 7]}, {"x": {1, 2, 4, 5}}, 7, {"x": "LA"})["x"]
    assert got == {"injury": "hamstring", "again": 1, "report_week": 6, "practice": "limited", "status": "O"} or \
        (got["injury"], got["again"], got["report_week"]) == ("knee", 0, 7)
    early = rm.contexts(inj, {"LA": [1, 2, 3, 4, 5, 6, 7]}, {"x": {1, 2, 4, 5}}, 6, {"x": "LA"})["x"]
    assert (early["injury"], early["again"], early["report_week"], early["practice"]) == ("hamstring", 1, 6, "limited")
