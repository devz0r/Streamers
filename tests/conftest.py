"""Shared fixtures.

The suite runs entirely offline: nothing here touches the network or the
``data/raw`` cache. Data-loader tests exercise the parsing and normalisation
logic against hand-built frames instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamer.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "config.yaml")


@pytest.fixture
def tmp_cfg(tmp_path, cfg):
    """A config rooted in a temp dir, so writes never touch the repo."""
    from dataclasses import replace

    for name in ("data/raw", "results", "reports", "docs"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    return replace(cfg, root=tmp_path)


@pytest.fixture
def toy_pbp() -> pd.DataFrame:
    """A two-game play-by-play covering every scoring event we care about.

    Game 1 (BUF at NYJ): NYJ kicker hits 3 field goals across all three
    distance buckets and misses one; BUF's defense records a sack, an
    interception, a fumble recovery, a safety and a blocked field goal.

    Game 2 (KC at DEN): a defensive touchdown, a punt-return touchdown and a
    kickoff-return touchdown, which are the three attribution edge cases.
    """
    rows = []

    def play(**kw):
        base = dict(
            season=2024, week=1, game_id="2024_01_BUF_NYJ", home_team="NYJ",
            away_team="BUF", season_type="REG", posteam="NYJ", defteam="BUF",
            play_type="pass", play_id=len(rows) + 1, drive=1.0, fixed_drive=1.0,
            fixed_drive_result="Field goal", yardline_100=25.0, down=1.0, ydstogo=10.0,
            qtr=1, field_goal_attempt=0.0, field_goal_result=None, kick_distance=None,
            extra_point_attempt=0.0, extra_point_result=None,
            kicker_player_id=None, kicker_player_name=None,
            sack=0.0, qb_hit=0.0, interception=0.0, pass_attempt=1.0, rush_attempt=0.0,
            qb_dropback=1.0, fumble=0.0, fumble_lost=0.0, fumble_forced=0.0,
            fumble_recovery_1_team=None, fumble_recovery_2_team=None,
            fumbled_1_team=None, fumbled_2_team=None,
            safety=0.0, touchdown=0.0, td_team=None, return_touchdown=0.0,
            punt_blocked=0.0, punt_attempt=0.0, kickoff_attempt=0.0,
            defensive_extra_point_conv=0.0, epa=0.0, wp=0.5,
            aborted_play=0.0, penalty=0.0,
        )
        base.update(kw)
        rows.append(base)

    # --- Game 1: kicking ---
    for distance, result in ((30.0, "made"), (45.0, "made"), (55.0, "made"), (42.0, "missed")):
        play(play_type="field_goal", field_goal_attempt=1.0, field_goal_result=result,
             kick_distance=distance, kicker_player_id="K1", kicker_player_name="T.Boot",
             qb_dropback=0.0, pass_attempt=0.0)
    play(play_type="extra_point", extra_point_attempt=1.0, extra_point_result="good",
         kicker_player_id="K1", kicker_player_name="T.Boot", qb_dropback=0.0, pass_attempt=0.0)
    play(play_type="extra_point", extra_point_attempt=1.0, extra_point_result="failed",
         kicker_player_id="K1", kicker_player_name="T.Boot", qb_dropback=0.0, pass_attempt=0.0)

    # --- Game 1: BUF defense ---
    play(sack=1.0, qb_hit=1.0)
    play(interception=1.0)
    play(fumble=1.0, fumble_lost=1.0, fumbled_1_team="NYJ", fumble_recovery_1_team="BUF")
    play(safety=1.0)
    play(play_type="field_goal", field_goal_attempt=1.0, field_goal_result="blocked",
         kick_distance=48.0, kicker_player_id="K1", kicker_player_name="T.Boot",
         qb_dropback=0.0, pass_attempt=0.0)

    # BUF's own offense, so the team-week join has both sides of the game.
    for _ in range(3):
        play(posteam="BUF", defteam="NYJ", fixed_drive=2.0, fixed_drive_result="Punt")
    play(posteam="BUF", defteam="NYJ", fixed_drive=3.0, fixed_drive_result="Touchdown",
         yardline_100=5.0, touchdown=1.0, td_team="BUF")

    # --- Game 2: the three touchdown attribution cases ---
    g2 = dict(game_id="2024_01_KC_DEN", home_team="DEN", away_team="KC")
    # Defensive TD: DEN's defense scores while KC has the ball.
    play(posteam="KC", defteam="DEN", touchdown=1.0, td_team="DEN",
         return_touchdown=1.0, interception=1.0, **g2)
    # Punt return TD: posteam is the punting team, so the returner is defteam.
    play(posteam="KC", defteam="DEN", play_type="punt", punt_attempt=1.0,
         touchdown=1.0, td_team="DEN", return_touchdown=1.0,
         qb_dropback=0.0, pass_attempt=0.0, **g2)
    # Kickoff return TD: posteam is the RECEIVING team on a kickoff.
    play(posteam="DEN", defteam="KC", play_type="kickoff", kickoff_attempt=1.0,
         touchdown=1.0, td_team="DEN", return_touchdown=1.0,
         qb_dropback=0.0, pass_attempt=0.0, **g2)

    return pd.DataFrame(rows)


@pytest.fixture
def toy_games() -> pd.DataFrame:
    """Team-game spine matching :func:`toy_pbp`."""
    rows = [
        dict(season=2024, week=1, game_id="2024_01_BUF_NYJ", team="NYJ", opponent="BUF",
             is_home=1, team_score=13, opp_score=6, team_spread=-2.0, total_line=44.0),
        dict(season=2024, week=1, game_id="2024_01_BUF_NYJ", team="BUF", opponent="NYJ",
             is_home=0, team_score=6, opp_score=13, team_spread=2.0, total_line=44.0),
        dict(season=2024, week=1, game_id="2024_01_KC_DEN", team="DEN", opponent="KC",
             is_home=1, team_score=21, opp_score=0, team_spread=-3.0, total_line=42.0),
        dict(season=2024, week=1, game_id="2024_01_KC_DEN", team="KC", opponent="DEN",
             is_home=0, team_score=0, opp_score=21, team_spread=3.0, total_line=42.0),
    ]
    frame = pd.DataFrame(rows)
    frame["team_implied_total"] = frame["total_line"] / 2 + frame["team_spread"] / 2
    frame["opp_implied_total"] = frame["total_line"] / 2 - frame["team_spread"] / 2
    frame["is_dome"] = 0
    frame["roof"] = "outdoors"
    frame["temp"] = 60.0
    frame["wind"] = 5.0
    frame["gameday"] = "2024-09-08"
    return frame
