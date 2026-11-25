"""Why a defence's number moved, and whether a swap is worth making."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace as NS

import pandas as pd

from streamer.league.model import PlayerRow
from streamer.roster import unit_drift
from streamer.roster.lineup import StreamOption


def _ranks(exp, imp, opp="PIT"):
    dst = pd.DataFrame([{"team": "DEN", "opponent": opp, "expected_points": exp,
                         "rationale": f"opp implied {imp}, 33% to hold them under 14"}])
    return NS(dst=dst, kicker=None)


def test_the_first_reading_stands_and_the_note_says_what_moved(tmp_cfg):
    snap = NS(season=2026, week=12)
    unit_drift.record(tmp_cfg, snap, _ranks(9.1, 17.8), now=datetime(2026, 11, 24, 6, 37, tzinfo=UTC))
    unit_drift.record(tmp_cfg, snap, _ranks(9.7, 16.5), now=datetime(2026, 11, 25, 9, 0, tzinfo=UTC))
    u = unit_drift.load(tmp_cfg, 2026, 12)["DST:DEN"]
    assert u["first"]["exp"] == 9.1 and u["last"]["exp"] == 9.7
    text = unit_drift.note(u, "DST")
    assert "9.1 to 9.7" in text and "since Tue 1:37 AM ET" in text and "16.5 (was 17.8)" in text
    assert unit_drift.load(tmp_cfg, 2026, 13) == {}                       # another week: nothing
    unit_drift.record(tmp_cfg, NS(season=2026, week=13), _ranks(8.0, 20.0), now=datetime(2026, 12, 1, tzinfo=UTC))
    assert unit_drift.load(tmp_cfg, 2026, 12) == {}                        # the old week is dropped


def test_a_wobble_is_not_reported_and_a_changed_opponent_is():
    base = {"first": {"exp": 9.1, "imp": 17.8, "opp": "PIT", "at": "2026-11-24T06:37:00+00:00"}}
    assert unit_drift.note({**base, "last": {"exp": 9.2, "imp": 17.7, "opp": "PIT"}}, "DST") == ""
    assert "opponent changed" in unit_drift.note({**base, "last": {"exp": 9.2, "imp": 17.7, "opp": "BAL"}}, "DST")


def test_the_streaming_table_calls_a_small_gap_the_same_tier_and_notes_what_moved():
    from streamer.roster.page import _streams

    def dst(pid, team, proj):
        return PlayerRow(player_id=pid, name=f"{team} D/ST", position="DST", team=team, projection=proj, nfl_opponent="X")

    mine = StreamOption(dst("m", "CLE", 9.1), 0.31, True, low=3, high=14)
    near = StreamOption(dst("a", "WAS", 9.4), 0.31, False, low=3, high=14)
    far = StreamOption(dst("b", "DEN", 10.3), 0.33, False, low=3, high=15)
    drift = {"DST:DEN": {"first": {"exp": 9.1, "imp": 17.8, "opp": "PIT", "at": "2026-11-24T06:37:00+00:00"},
                         "last": {"exp": 10.3, "imp": 16.5, "opp": "PIT"}}}
    html = _streams(NS(streams={"DST": [far, near, mine]}, unit_drift=drift))
    assert "same tier" in html and "+1.2: worth it" in html and "<th>Move?</th>" in html
    assert "the books now expect their opponent to score 16.5 (was 17.8)" in html
