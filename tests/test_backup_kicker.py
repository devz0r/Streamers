"""A team's kicking line goes to the kicker kicking for it, not a second one listed."""

from types import SimpleNamespace

from streamer.league.model import PlayerRow
from streamer.roster.projections import _backup_kickers


def _k(pid, name, team, status=""):
    return PlayerRow(player_id=pid, name=name, position="K", team=team, status=status)


def test_a_second_kicker_listed_on_the_team_is_a_backup():
    rows = {"KC": SimpleNamespace(player_name="H.Butker"), "PIT": SimpleNamespace(player_name="C.Boswell")}
    players = [_k("1", "Harrison Butker", "KC"), _k("2", "Justin Tucker", "KC"), _k("3", "Chris Boswell", "PIT")]
    assert _backup_kickers(players, rows) == {"2": "Harrison Butker"}


def test_a_team_that_replaced_its_kicker_keeps_the_line_for_whoever_it_lists():
    rows = {"KC": SimpleNamespace(player_name="H.Butker")}
    hurt = [_k("1", "Harrison Butker", "KC", status="OUT"), _k("2", "Justin Tucker", "KC")]
    assert _backup_kickers(hurt, rows) == {}
    released = [_k("2", "Justin Tucker", "KC"), _k("4", "Spencer Shrader", "KC")]
    assert _backup_kickers(released, rows) == {}
