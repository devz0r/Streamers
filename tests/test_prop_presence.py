"""Props still posted on a questionable player, as a read on whether he plays:
read from the pull, logged, graded, and given no say until it has earned it."""

import pandas as pd
import pytest

from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow
from streamer.roster import prop_presence as pp


def _p(pid, name, team, status=""):
    p = PlayerRow(player_id=pid, name=name, position="WR", team=team, status=status, slot="WR", projection=8.0)
    p.nfl_id = pid
    return p


def _snap(*players):
    return LeagueSnapshot(platform="espn", profile="espn", league_id="1", league_name="L", season=2026, week=10,
                          slots={"WR": 2}, bench_size=2, free_agents=[], matchup=None, synced_at="",
                          teams=[TeamRow(team_id="1", name="T", roster=list(players), is_mine=True)])


def _frame(event, names, per=2):
    rows = [{"event_id": event, "player": n, "bookmaker": b, "stat": "receiving_yards"}
            for n in names for b in ("fanduel", "draftkings")[:per]]
    return pd.DataFrame(rows)


def _game(snap_players):
    return _frame("e1", [p.name for p in snap_players])


def test_a_tagged_player_with_lines_in_a_pulled_game_reads_as_posted_and_without_as_taken_down():
    mates = [_p(str(i), f"Mate {i}", "LA") for i in range(5)] + [_p(str(10 + i), f"Foe {i}", "SF") for i in range(5)]
    posted, down = _p("a", "Posted Q", "LA", "Q"), _p("b", "Down Q", "SF", "Q")
    snap = _snap(*mates, posted, down)
    got = pp.attach(snap, _game(mates + [posted]))
    assert got == 2 and posted.prop_lines == 2 and down.prop_lines == 0


def test_a_game_that_was_not_pulled_says_nothing():
    mates = [_p(str(i), f"Mate {i}", "LA") for i in range(5)]
    q = _p("a", "Elsewhere Q", "DAL", "Q")
    snap = _snap(*mates, q)
    assert pp.attach(snap, _game(mates)) == 0 and q.prop_lines is None
    assert pp.attach(snap, None) == 0


def test_no_say_below_the_cases_and_a_step_that_follows_the_record_above_them(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(type(cfg), "path", lambda self, name: tmp_path)
    cfg.raw.setdefault("roster", {})["prop_presence"] = {"prior_cases": 40, "min_cases": 30}
    rows = []
    for i in range(60):
        rows.append({"season": 2026, "week": 3 + i % 6, "nfl_id": f"h{i}", "name": "x", "position": "WR", "status": "Q",
                     "practice": "", "ours": 0.6, "lines": 3, "at": "2026-10-01"})           # has lines: plays 90%
        rows.append({"season": 2026, "week": 3 + i % 6, "nfl_id": f"n{i}", "name": "x", "position": "WR", "status": "Q",
                     "practice": "", "ours": 0.6, "lines": 0, "at": "2026-10-01"})           # none: plays 20%
    pd.DataFrame(rows[:20]).to_parquet(tmp_path / pp.LOG)
    hist = pd.DataFrame([{"season": 2026, "week": w, "player_id": f"{k}{i}"} for w in range(3, 9) for k in "hn"
                         for i in range(60) if (k == "h" and i % 10) or (k == "n" and i % 5 == 0)])
    steps, counts = pp.fit_steps(cfg, hist)
    assert steps == {"has": 0.0, "none": 0.0} and counts["has"] + counts["none"] == 20       # too few graded: no say
    pd.DataFrame(rows).to_parquet(tmp_path / pp.LOG)
    steps, counts = pp.fit_steps(cfg, hist)
    assert counts == {"has": 60, "none": 60}
    assert steps["has"] > 0.5 and steps["none"] < -0.5                                       # posted: likelier; taken down: unlikelier


def test_adjust_moves_only_a_pulled_game_in_the_direction_of_its_step():
    step = {"has": 1.0, "none": -1.0}
    assert pp.adjust(0.6, 3, step) > 0.6 > pp.adjust(0.6, 0, step)
    assert pp.adjust(0.6, None, step) == 0.6 and pp.adjust(0.0, 3, step) == 0.0 and pp.adjust(1.0, 0, step) == 1.0
    assert pp.adjust(0.6, 0, {"has": 0.0, "none": 0.0}) == 0.6
    assert pp.STEP == {"has": 0.0, "none": 0.0}                                              # no say until fitted
