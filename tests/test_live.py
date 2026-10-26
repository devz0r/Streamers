"""Live scores: the box score -> fantasy points in static/live.js (run in
Node), the draws rescored with the time left, and the page's live block."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

LIVE_JS = Path(__file__).resolve().parents[1] / "src" / "streamer" / "static" / "live.js"
ESPN_OFF = {"pass_yards": 0.04, "pass_td": 4.0, "interception": -2.0, "rush_yards": 0.1, "rush_td": 6.0,
            "reception": 1.0, "reception_yards": 0.1, "reception_td": 6.0, "fumble_lost": -2.0, "two_point": 2.0}
ESPN_K = {"fg_0_39": 3.0, "fg_40_49": 4.0, "fg_50_59": 5.0, "fg_60_plus": 6.0, "fg_missed_0_39": -1.0,
          "fg_missed_40_49": -1.0, "fg_missed_50_59": -1.0, "fg_missed_60_plus": -1.0, "pat_made": 1.0,
          "pat_missed": -0.5}
ESPN_DST = {"sack": 2.5, "interception": 2.5, "fumble_recovery": 2.0, "safety": 2.0, "defensive_td": 6.0,
            "return_td": 6.0, "blocked_kick_td": 6.0, "fourth_down_stop": 0.0,
            "points_allowed_tiers": [[0, 0, 6.0], [1, 6, 4.0], [7, 13, 3.0], [14, 17, 1.0], [18, 27, 0.0],
                                     [28, 34, -1.0], [35, 45, -3.0], [46, None, -4.0]],
            "yards_allowed_tiers": [[0, 99, 5.0], [100, 199, 3.0], [200, 299, 2.0], [300, 349, 1.0],
                                    [350, 399, -1.0], [400, 449, -2.0], [450, 499, -3.0], [500, 549, -4.0],
                                    [550, None, -5.0]]}
SC = {"off": ESPN_OFF, "k": ESPN_K, "dst": ESPN_DST}

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")


def _ath(pid, name, stats):
    return {"athlete": {"id": pid, "displayName": name}, "stats": stats}


SUMMARY = {
    "boxscore": {
        "players": [
            {"team": {"abbreviation": "KC"}, "statistics": [
                {"name": "passing", "keys": ["completions/passingAttempts", "passingYards", "yardsPerPassAttempt",
                                             "passingTouchdowns", "interceptions", "sacks-sackYardsLost", "adjQBR",
                                             "QBRating"],
                 "athletes": [_ath("1", "Patrick Mahomes", ["20/30", "250", "8.3", "2", "1", "2-14", "70", "100"])]},
                {"name": "rushing", "keys": ["rushingAttempts", "rushingYards", "yardsPerRushAttempt",
                                             "rushingTouchdowns", "longRushing"],
                 "athletes": [_ath("2", "Isiah Pacheco", ["15", "80", "5.3", "1", "20"])]},
                {"name": "receiving", "keys": ["receptions", "receivingYards", "yardsPerReception",
                                               "receivingTouchdowns", "longReception", "receivingTargets"],
                 "athletes": [_ath("3", "Rashee Rice", ["6", "90", "15.0", "1", "40", "9"])]},
                {"name": "fumbles", "keys": ["fumbles", "fumblesLost", "fumblesRecovered"],
                 "athletes": [_ath("2", "Isiah Pacheco", ["1", "1", "0"])]},
                {"name": "kicking", "keys": ["fieldGoalsMade/fieldGoalAttempts", "fieldGoalPct", "longFieldGoalMade",
                                             "extraPointsMade/extraPointAttempts", "totalKickingPoints"],
                 "athletes": [_ath("4", "Harrison Butker", ["2/3", "66.7", "47", "3/3", "9"])]},
            ]},
            {"team": {"abbreviation": "DEN"}, "statistics": [
                {"name": "defensive", "keys": ["totalTackles", "soloTackles", "sacks", "tacklesForLoss",
                                               "passesDefended", "QBHits", "defensiveTouchdowns"],
                 "athletes": [_ath("9", "Nik Bonitto", ["5", "4", "1.5", "1", "0", "2", "0"]),
                              _ath("10", "Zach Allen", ["3", "2", "0.5", "1", "0", "1", "0"])]},
                {"name": "interceptions", "keys": ["interceptions", "interceptionYards", "interceptionTouchdowns"],
                 "athletes": [_ath("11", "Pat Surtain II", ["1", "20", "0"])]},
            ]},
        ],
        "teams": [
            {"team": {"abbreviation": "KC"}, "statistics": [{"name": "fumblesLost", "displayValue": "1"},
                                                            {"name": "totalYards", "displayValue": "380"},
                                                            {"name": "fourthDownEff", "displayValue": "1-3"}]},
            {"team": {"abbreviation": "DEN"}, "statistics": [{"name": "fumblesLost", "displayValue": "0"},
                                                             {"name": "totalYards", "displayValue": "250"}]},
        ],
    },
    "scoringPlays": [
        {"team": {"abbreviation": "KC"}, "type": {"text": "Field Goal Good"}, "text": "Harrison Butker 47 Yd Field Goal"},
        {"team": {"abbreviation": "KC"}, "type": {"text": "Field Goal Good"}, "text": "Harrison Butker 32 Yd Field Goal"},
        {"team": {"abbreviation": "KC"}, "type": {"text": "Passing Touchdown"},
         "text": "Rashee Rice 12 Yd pass from Patrick Mahomes (Patrick Mahomes Pass to Rashee Rice for Two-Point Conversion)"},
        {"team": {"abbreviation": "DEN"}, "type": {"text": "Safety"}, "text": "Isiah Pacheco tackled in end zone, SAFETY"},
    ],
}
GAME = {"id": "1", "home": "KC", "away": "DEN", "score": {"KC": 27, "DEN": 10}, "state": "post", "left": 0}


def _node(script: str):
    out = subprocess.run(["node", "-e", f"const L=require({json.dumps(str(LIVE_JS))});{script}"],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _points(player):
    return _node(f"console.log(JSON.stringify(L.pointsFor({json.dumps(player)}, {json.dumps(GAME)}, "
                 f"{json.dumps(SUMMARY)}, {json.dumps(SC)}, {{}})))")


def test_skill_players_score_as_espn_does():
    assert _points({"n": "Patrick Mahomes", "pos": "QB", "tm": "KC", "eid": "1"}) == pytest.approx(18.0)  # 10+8-2 +2pt
    assert _points({"n": "Isiah Pacheco", "pos": "RB", "tm": "KC", "eid": "2"}) == pytest.approx(12.0)    # 8+6-2
    assert _points({"n": "Rashee Rice", "pos": "WR", "tm": "KC", "eid": "3"}) == pytest.approx(23.0)      # 6+9+6 +2pt


def test_a_player_is_found_by_name_when_the_id_is_missing():
    assert _points({"n": "Rashee Rice", "pos": "WR", "tm": "KC", "eid": None}) == pytest.approx(23.0)


def test_a_kicker_scores_by_distance_with_misses_and_extra_points():
    # 47 (4) + 32 (3), one miss (-1), three extra points (3)
    assert _points({"n": "Harrison Butker", "pos": "K", "tm": "KC", "eid": "4"}) == pytest.approx(9.0)


def test_a_defence_scores_its_plays_and_what_it_allowed():
    # 2 sacks (5), 1 INT (2.5), 1 fumble recovered (2), safety (2), 27 allowed (0), 380 yards (-1)
    assert _points({"n": "Broncos D/ST", "pos": "DST", "tm": "DEN"}) == pytest.approx(10.5)


def test_the_share_of_the_game_left():
    left = _node("console.log(JSON.stringify([L.remaining({type:{state:'pre'}}), L.remaining({type:{state:'post'}}),"
                 "L.remaining({type:{state:'in',name:'STATUS_IN_PROGRESS'},period:3,clock:450}),"
                 "L.remaining({type:{state:'in',name:'STATUS_HALFTIME'},period:2,clock:0})]))")
    assert left == [1, 0, pytest.approx(0.375), 0.5]


def _b64(rows):
    return base64.b64encode((np.asarray(rows) * 10).astype("<i2").tobytes()).decode()


def test_the_draws_are_rescored_with_what_has_happened():
    """Before kickoff the draws decide; once a game is final its score does."""
    n = 4
    D = {"n": n, "sc": SC, "teams": {},
         "me": [{"n": "Rashee Rice", "pos": "WR", "tm": "KC", "eid": "3", "m": 15.0, "slot": "WR"}],
         "opp": [{"n": "Other Guy", "pos": "WR", "tm": "BUF", "eid": "99", "m": 18.0, "slot": "WR"}],
         "S": _b64([10, 12, 14, 16]), "O": _b64([20, 20, 20, 20])}
    games = {"KC": GAME, "BUF": {"id": "2", "home": "BUF", "away": "MIA", "score": {}, "state": "pre", "left": 1,
                                 "label": "Sun 4:25 PM"}}
    r = _node(f"console.log(JSON.stringify(L.evaluate({json.dumps(D)}, {json.dumps(games)}, "
              f"{{'1': {json.dumps(SUMMARY)}}})))")
    assert r["p0"] == 0.0                       # every draw behind before kickoff
    assert r["p"] == 1.0                        # 23 already in beats every 20
    assert r["nowMe"] == pytest.approx(23.0) and r["nowOpp"] == 0.0
    assert r["projMe"] == pytest.approx(23.0) and r["projOpp"] == pytest.approx(20.0)


def test_the_page_carries_both_sides_draws_and_the_scoring(cfg):
    from streamer.league.model import PlayerRow
    from streamer.roster import lineup
    from streamer.roster.live import payload

    def p(pid, pos, proj, slot, team="KC"):
        return PlayerRow(player_id=pid, name=f"P{pid}", position=pos, team=team, slot=slot, projection=proj,
                         projection_sd=5.0, outcome_sd=5.0)

    mine = [p("101", "WR", 14.0, "WR"), p("102", "WR", 9.0, "BN")]
    theirs = [p("201", "WR", 12.0, "WR", "BUF")]
    opt = lineup.optimise(mine, {"WR": 1}, theirs, n_sims=3000)

    class Snap:
        platform, profile, season, week = "espn", "espn", 2026, 7

    d = payload(Snap(), opt, cfg.for_profile("espn"))
    assert [r["n"] for r in d["me"]] == ["P101"] and [r["n"] for r in d["opp"]] == ["P201"]
    assert d["me"][0]["eid"] == "101"                      # ESPN's league ids are ESPN's athlete ids
    assert d["n"] == 2000 and len(base64.b64decode(d["S"])) == 2 * 2000
    assert d["sc"]["off"]["reception"] == 1.0 and "points_allowed_tiers" in d["sc"]["dst"]


def test_blocked_kicks_and_the_units_muffed_punts_come_from_the_play_by_play():
    summary = dict(SUMMARY, drives={"previous": [
        {"team": {"abbreviation": "KC"}, "plays": [
            {"type": {"text": "Punt"}, "text": "T.Townsend punt is BLOCKED by N.Bonitto, recovered by DEN"},
            {"type": {"text": "Punt"}, "text": "T.Townsend punts 45 yards, M.Mims MUFFS catch, RECOVERED by KC-J.Smith"}]},
        {"team": {"abbreviation": "DEN"}, "plays": [
            {"type": {"text": "Rush"}, "text": "J.Williams up the middle for 4 yards"}]},
    ]})
    unit = _node(f"console.log(JSON.stringify(L.unitPlays({json.dumps(summary)}, {{}})))")
    assert unit == {"DEN": {"blocked": 1, "lost": 1}}
    pts = _node(f"console.log(JSON.stringify(L.pointsFor({{'n':'Broncos D/ST','pos':'DST','tm':'DEN'}}, "
                f"{json.dumps(GAME)}, {json.dumps(summary)}, {json.dumps(dict(SC, dst=dict(ESPN_DST, blocked_kick=2.0, fumble_lost=-2.0)))}, {{}})))")
    assert pts == pytest.approx(10.5 + 2.0 - 2.0)


def test_the_season_odds_move_along_the_win_loss_line_with_the_live_p_win():
    s = {"po": 0.5, "ti": 0.08, "po_w": 0.7, "po_l": 0.3, "ti_w": 0.14, "ti_l": 0.02}
    at_kickoff = _node(f"console.log(JSON.stringify(L.seasonOdds({json.dumps(s)}, 0)))")
    assert at_kickoff["po"] == 0.5 and at_kickoff["ti"] == 0.08           # nothing moves before kickoff
    up = _node(f"console.log(JSON.stringify(L.seasonOdds({json.dumps(s)}, 0.25)))")
    assert up["po"] == pytest.approx(0.6) and up["ti"] == pytest.approx(0.11)
    # Certain win or loss lands on the end of the line, and never past it.
    win = _node(f"console.log(JSON.stringify(L.seasonOdds({json.dumps(s)}, 0.5)))")
    assert win["po"] == pytest.approx(0.7) and win["ti"] == pytest.approx(0.14)
    huge = _node(f"console.log(JSON.stringify(L.seasonOdds({json.dumps(s)}, 9)))")
    assert huge["po"] == 1.0
    assert _node("console.log(JSON.stringify(L.seasonOdds(null, 0.1)))") is None


def test_the_live_card_carries_this_weeks_game_stakes_when_there_are_some():
    import numpy as np

    from streamer.roster.live import season_stakes
    from streamer.roster.season import GameStakes, SeasonOdds, Stakes

    class Snap:
        week = 7

    odds = SeasonOdds(team_ids=["a", "b"], names=["A", "B"], records=["", ""], p_playoffs=np.array([0.5, 0.4]),
                      p_bye=np.zeros(2), p_title=np.array([0.08, 0.05]), exp_wins=np.zeros(2), n_sims=10, mine=0)
    game = GameStakes(week=7, opponent="B", opponent_id="b", p_win=0.55, playoffs_win=0.7, playoffs_loss=0.3,
                      title_win=0.14, title_loss=0.02)
    stakes = Stakes(games=[game], by_wins={}, p_playoffs=0.5, exp_wins=7.0, n_sims=10)
    got = season_stakes(Snap(), odds, stakes)
    assert got == {"po": 0.5, "ti": 0.08, "po_w": 0.7, "po_l": 0.3, "ti_w": 0.14, "ti_l": 0.02}
    Snap.week = 9                                                       # no such game left
    assert season_stakes(Snap(), odds, stakes) is None
    assert season_stakes(Snap(), None, stakes) is None
