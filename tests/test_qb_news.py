"""A quarterback named to start or benched in this week's news moves who is in line to start."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS

from streamer.data import news
from streamer.league.model import PlayerRow
from streamer.roster.qb_change import this_weeks_starters


def test_the_news_reads_who_starts_and_who_is_benched():
    assert news.starter_call("Joe Fagnano (ankle) will start Thursday's game against the Jaguars.", "Fagnano") == "start"
    assert news.starter_call("Jacoby Brissett was benched in favor of Gardner Minshew.", "Brissett") == "benched"
    assert news.starter_call("Gardner Minshew could start if Brissett cannot go.", "Minshew") is None     # a hedge
    assert news.starter_call("Brissett is out, and Minshew will start Sunday.", "Brissett") is None        # someone else
    assert news.starter_call("Kyler Murray has been named the starter.", "Murray") == "start"


def test_a_benched_starter_and_a_named_backup_swap_places(tmp_path):
    ids = {"b": "00-b", "m": "00-m"}
    brissett = PlayerRow(player_id="b", name="Jacoby Brissett", position="QB", team="ARI")
    minshew = PlayerRow(player_id="m", name="Gardner Minshew II", position="QB", team="ARI")
    brissett.depth_order, minshew.depth_order = 1, 2
    assert this_weeks_starters([brissett, minshew], ids)["ARI"][1] == "Jacoby Brissett"
    now = datetime(2026, 11, 6, 15, tzinfo=UTC)
    item = NS(name="Jacoby Brissett", published=now - timedelta(hours=3),
              headline="Jacoby Brissett was benched in favor of Gardner Minshew.")
    snap = NS(_news=[item], all_players=lambda: [brissett, minshew])
    cfg = NS(path=lambda _d: tmp_path)
    assert news.qb_calls(snap, cfg, now=now) == ["Jacoby Brissett: benched (Fri Nov 06)"]
    assert this_weeks_starters([brissett, minshew], ids)["ARI"][1] == "Gardner Minshew II"
    # Kept for the week: with no fresh news on Sunday, the call still stands.
    snap._news = []
    news.qb_calls(snap, cfg, now=now + timedelta(days=2))
    assert brissett.starter_call == "benched"
