"""A realistic synthetic league snapshot for offline tests.

Ten teams, ESPN-shaped slots, a full free-agent pool, and one deliberately
lopsided matchup so the optimiser has something to optimise. Projections are
attached directly so the optimiser and waiver tests do not need nflverse.
"""

from __future__ import annotations

import random

from streamer.league.model import LeagueSnapshot, Matchup, PlayerRow, TeamRow

SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1}
TEAMS_NFL = ["KC", "BUF", "SF", "DAL", "PHI", "DET", "MIA", "BAL", "CIN", "GB",
             "LAC", "NYJ", "SEA", "LA", "MIN", "HOU", "JAX", "PIT", "CLE", "ATL"]


def _player(pid: int, pos: str, proj: float, sd: float, team: str, slot: str = "BN",
            status: str = "", bye: bool = False, pct: float | None = None) -> PlayerRow:
    eligible = {"QB": ["QB"], "RB": ["RB", "FLEX"], "WR": ["WR", "FLEX"],
                "TE": ["TE", "FLEX"], "K": ["K"], "DST": ["DST"]}[pos]
    return PlayerRow(
        player_id=str(pid), name=f"{pos} Player {pid}", position=pos, team=team,
        status=status, slot=slot, eligible_slots=eligible, on_bye=bye,
        projection=proj, projection_sd=sd, ros_value=proj, projection_source="test",
        percent_owned=pct,
    )


def my_roster() -> list[PlayerRow]:
    """A roster with real decisions in it: a Q-tag, a bye, a bust, a boom WR."""
    return [
        _player(1, "QB", 19.0, 8.0, "KC", slot="QB"),
        _player(2, "RB", 15.0, 7.0, "SF", slot="RB"),
        _player(3, "RB", 12.0, 7.0, "DET", slot="RB"),
        _player(4, "RB", 9.0, 6.0, "MIA"),                          # bench RB
        _player(5, "WR", 14.0, 7.5, "PHI", slot="WR"),
        _player(6, "WR", 11.0, 6.0, "DAL", slot="WR"),               # steady
        _player(7, "WR", 11.0, 11.0, "CIN"),                         # boom-or-bust, benched
        _player(8, "WR", 7.0, 5.0, "GB", bye=True),                  # on bye
        _player(9, "TE", 9.0, 6.0, "BAL", slot="TE"),
        _player(10, "TE", 5.0, 4.0, "NYJ"),
        _player(11, "RB", 10.0, 6.5, "LAC", slot="FLEX", status="Q"),
        _player(12, "K", 8.0, 3.5, "KC", slot="K"),
        _player(13, "DST", 7.0, 5.0, "SF", slot="DST"),
        _player(14, "WR", 3.0, 3.0, "JAX", status="OUT"),            # dead roster spot
    ]


def opponent_roster(strength: float = 1.0) -> list[PlayerRow]:
    return [
        _player(101, "QB", 18.0 * strength, 8.0, "BUF", slot="QB"),
        _player(102, "RB", 14.0 * strength, 7.0, "PHI", slot="RB"),
        _player(103, "RB", 11.0 * strength, 7.0, "SEA", slot="RB"),
        _player(104, "WR", 13.0 * strength, 7.5, "MIA", slot="WR"),
        _player(105, "WR", 12.0 * strength, 7.5, "MIN", slot="WR"),
        _player(106, "TE", 8.0 * strength, 6.0, "KC", slot="TE"),
        _player(107, "WR", 10.0 * strength, 7.0, "HOU", slot="FLEX"),
        _player(108, "K", 8.0 * strength, 3.5, "BAL", slot="K"),
        _player(109, "DST", 7.0 * strength, 5.0, "PIT", slot="DST"),
        _player(110, "RB", 6.0, 5.0, "ATL"),
    ]


def free_agents() -> list[PlayerRow]:
    rng = random.Random(3)
    out = [
        _player(201, "RB", 11.5, 7.0, "CLE", slot="FA", pct=18.0),      # clear upgrade on RB4
        _player(202, "WR", 8.0, 6.0, "ATL", slot="FA", pct=30.0),
        _player(203, "TE", 8.5, 6.0, "HOU", slot="FA", pct=12.0),       # upgrade on TE2
        _player(204, "QB", 16.0, 8.0, "JAX", slot="FA", pct=40.0),
        _player(205, "DST", 9.5, 5.0, "DEN", slot="FA", pct=35.0),      # better stream
        _player(206, "K", 8.5, 3.5, "DAL", slot="FA", pct=20.0),
        _player(207, "WR", 12.0, 7.0, "LA", slot="FA", pct=8.0, bye=True),  # stash: good ROS, bye now
    ]
    out[-1].ros_value = 12.0
    for i in range(208, 230):
        pos = rng.choice(["RB", "WR", "TE"])
        out.append(_player(i, pos, rng.uniform(1.0, 6.0), 4.0, rng.choice(TEAMS_NFL), slot="FA", pct=rng.uniform(0, 10)))
    out[-1].projection_source = "test"
    return out


def snapshot(week: int = 5, opp_strength: float = 1.0) -> LeagueSnapshot:
    me = TeamRow(team_id="1", name="My Team", owner="me", wins=2, losses=2,
                 roster=my_roster(), is_mine=True)
    opp = TeamRow(team_id="2", name="Rival", owner="them", wins=3, losses=1,
                  roster=opponent_roster(opp_strength))
    others = [TeamRow(team_id=str(i), name=f"Team {i}", roster=[]) for i in range(3, 11)]
    return LeagueSnapshot(
        platform="espn", profile="espn", league_id="999", league_name="Test League",
        season=2026, week=week, slots={**SLOTS, "BN": 7, "IR": 1}, bench_size=7,
        teams=[me, opp, *others], free_agents=free_agents(),
        matchup=Matchup(week=week, my_team_id="1", opponent_team_id="2",
                        my_platform_projection=100.0, opp_platform_projection=98.0),
        synced_at="2026-10-01T12:00:00+00:00",
    )
