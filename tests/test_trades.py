"""Trades: offered only when the other side gains on projections, priced in title odds."""

from __future__ import annotations

import pytest

from streamer.league.model import LeagueSnapshot, PlayerRow, TeamRow
from streamer.roster import season as season_mod
from streamer.roster.season import SeasonModel
from streamer.roster.trades import CONSOLIDATION_GAP, TradeFinder, own_view


def _p(pid, pos, v):
    return PlayerRow(player_id=pid, name=pid, position=pos, team="KC", projection=v, projection_sd=6.0,
                     outcome_sd=6.0, ros_value=v, ros_sd=2.0,
                     eligible_slots=[pos, "FLEX"] if pos != "QB" else ["QB"])


def _team(tid, spec, mine=False):
    roster = [_p(f"{tid}-{pos}{i}", pos, v) for pos, vals in spec.items() for i, v in enumerate(vals)]
    return TeamRow(team_id=tid, name=f"Team {tid}", roster=roster, is_mine=mine, wins=1, losses=1)


def _league():
    # You: four good backs, weak receivers. Team 1: the mirror image.
    me = _team("0", {"QB": [18], "RB": [16, 15, 14, 13], "WR": [8, 7, 6], "TE": [8]}, mine=True)
    mirror = _team("1", {"QB": [18], "RB": [8, 7, 6], "WR": [16, 15, 14, 13], "TE": [8]})
    others = [_team(str(i), {"QB": [18], "RB": [12, 11, 7], "WR": [12, 11, 7], "TE": [8]}) for i in (2, 3)]
    teams = [me, mirror, *others]
    ids = [t.team_id for t in teams]
    schedule = [[w, a, b] for w in range(3, 11) for a, b in ((ids[0], ids[1 + w % 3]),
                                                             tuple(i for i in ids[1:] if i != ids[1 + w % 3]))]
    fa = [_p("fa-rb", "RB", 5), _p("fa-rb2", "RB", 4.5), _p("fa-wr", "WR", 5), _p("fa-wr2", "WR", 4.5),
          _p("fa-qb", "QB", 12), _p("fa-qb2", "QB", 11), _p("fa-te", "TE", 4), _p("fa-te2", "TE", 3.5)]
    return LeagueSnapshot(
        platform="espn", profile="espn", league_id="1", league_name="L", season=2026, week=3,
        slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}, bench_size=4, teams=teams,
        free_agents=fa, matchup=None, synced_at="",
        rules={"regular_season_weeks": 10, "playoff_teams": 2, "playoff_round_weeks": 1,
               "schedule": schedule})


@pytest.fixture(autouse=True)
def _no_byes(monkeypatch):
    monkeypatch.setattr(season_mod, "nfl_byes", lambda *a, **k: {})


@pytest.fixture
def finder(cfg):
    snap = _league()
    return TradeFinder(snap, SeasonModel(snap, cfg, n_sims=1500))


def test_surplus_for_need_is_found_and_both_views_are_filled(finder):
    board = finder.find(5)
    assert board.top, "a back-for-receiver swap with the mirror team should be on offer"
    best = board.top[0]
    assert best.partner.team_id == "1"
    assert "RB" in {p.position for p in best.give} and "WR" in {p.position for p in best.get}
    assert best.gain > 0 and best.p_accept > 0.25 and best.their_seen > 0
    assert best.why_you and best.why_them


def test_the_three_lists_are_ordered_by_what_they_promise(finder):
    board = finder.find(5)
    top = [t.expected for t in board.top]
    assert top == sorted(top, reverse=True)
    assert [t.p_accept for t in board.likely] == sorted((t.p_accept for t in board.likely), reverse=True)
    assert all(t.p_accept >= 0.15 for t in board.best)
    assert [t.gain for t in board.best] == sorted((t.gain for t in board.best), reverse=True)


def test_a_trade_never_leaves_either_side_without_a_quarterback(finder):
    for t in finder.screen():
        mine = [p for p in finder.me.roster if p not in t.give] + t.get
        theirs = [p for p in t.partner.roster if p not in t.get] + t.give
        assert any(p.position == "QB" for p in mine) and any(p.position == "QB" for p in theirs)


def test_selling_a_player_the_market_overrates_is_called_out(cfg):
    snap = _league()
    star = next(p for p in snap.my_team.roster if p.player_id == "0-RB0")
    star.platform_projection = 21.0            # the market sees a star; we project 16
    board = TradeFinder(snap, SeasonModel(snap, cfg, n_sims=1500)).find(5)
    selling = [t for t in board.top + board.best + board.likely if star in t.give]
    assert selling and all(any("sell high" in r for r in t.why_you) for t in selling)


def test_the_page_shows_four_views_one_per_team(finder):
    from types import SimpleNamespace

    from streamer.roster.page import _trades

    board = finder.find(3)
    html = _trades(SimpleNamespace(trades=board, snapshot=finder.snapshot))
    for label in ("Top trades", "Best for you", "Most likely yes", "By team"):
        assert label in html
    assert html.count('type="radio"') == 4 and 'id="trades-espn-top"' in html
    others = [t for t in finder.snapshot.teams if not t.is_mine]
    assert set(board.by_team) == {t.team_id for t in others}
    assert all(html.count(f"<strong>{t.name}</strong>") >= 1 for t in others)
    assert all(p.name in html for t in board.top for p in t.give + t.get)
    assert _trades(SimpleNamespace(trades=None, snapshot=finder.snapshot)) == ""


def test_the_best_player_bonus_needs_a_clearly_best_player():
    """Worthy for Sadiq + Hill: Worthy 0.4 a game ahead is not "the best
    player in the deal" the way a star is."""
    value = {"worthy": 9.35, "sadiq": 8.95, "hill": 4.0, "star": 15.0}
    his = lambda p: value[p.player_id]                                   # noqa: E731
    P = {k: _p(k, "WR", v) for k, v in value.items()}
    near = TradeFinder._consolidation([P["worthy"]], [P["sadiq"], P["hill"]], his)
    assert near == pytest.approx(0.4 / CONSOLIDATION_GAP)
    assert TradeFinder._consolidation([P["star"]], [P["sadiq"], P["hill"]], his) == 1.0
    assert TradeFinder._consolidation([P["sadiq"], P["hill"]], [P["star"]], his) == -1.0
    assert TradeFinder._consolidation([P["hill"]], [P["sadiq"], P["worthy"]], his) == 0.0
    assert TradeFinder._consolidation([P["worthy"]], [P["sadiq"]], his) == 0.0


def test_his_lineup_raises_the_players_he_starts_never_lowers_his_bench():
    """He starts Sadiq in his flex over Brian Thomas Jr.: to him Sadiq is worth
    at least what the market gives Thomas. Thomas is not marked down -- a
    lineup can be a week old."""
    sadiq = PlayerRow(player_id="s", name="Kenyon Sadiq", position="TE", team="NYJ", slot="FLEX")
    btj = PlayerRow(player_id="b", name="Brian Thomas Jr.", position="WR", team="JAX", slot="BN")
    hurt = PlayerRow(player_id="h", name="Hurt Guy", position="WR", team="JAX", slot="BN", status="O")
    qb = PlayerRow(player_id="q", name="Backup QB", position="QB", team="JAX", slot="BN")
    starter_wr = PlayerRow(player_id="w", name="Starter", position="WR", team="KC", slot="WR")
    team = TeamRow(team_id="3", name="Punisher", roster=[sadiq, btj, hurt, qb, starter_wr])
    market = {"s": 8.95, "b": 9.6, "h": 14.0, "q": 12.0, "w": 12.0}
    values, over = own_view(team, lambda p: market[p.player_id])
    assert values == {"s": 9.6} and over == {"s": "Brian Thomas Jr."}


def test_each_team_gets_its_own_best_offers_beyond_the_league_lists(finder):
    board = finder.find(3)
    listed = {t.key for t in board.top + board.best + board.likely}
    for team_id, trades in board.by_team.items():
        assert all(t.partner.team_id == team_id for t in trades)
        assert [t.expected for t in trades] == sorted((t.expected for t in trades), reverse=True)
        assert len(trades) <= 5
    extra = [t for ts in board.by_team.values() for t in ts if t.key not in listed]
    assert extra or all(len(ts) <= 2 for ts in board.by_team.values())


def test_two_for_twos_are_searched(finder):
    shapes = {(len(t.give), len(t.get)) for t in finder.screen()}
    assert (2, 2) in shapes


def test_a_freed_roster_spot_is_filled_from_the_wire_and_priced_in(cfg):
    from streamer.roster.trades import FILL_POOL

    snap = _league()
    snap.free_agents = [_p("fa_wr", "WR", 11.0)]
    model = SeasonModel(snap, cfg, n_sims=400, extra_players=snap.free_agents)
    f = TradeFinder(snap, model)
    assert [p.player_id for p in f.fills] == ["fa_wr"] and FILL_POOL >= 1
    two_for_one = [t for t in f.screen() if len(t.give) == 2 and len(t.get) == 1]
    assert two_for_one and all(t.my_add is not None and t.my_add.player_id == "fa_wr" for t in two_for_one)
    t = f.price(two_for_one[0])
    assert "fa_wr" in f._rosters(t)[0]
    f._explain(t)
    assert any("then add fa_wr" in w for w in t.why_you)


def test_the_deadline_counts_down_and_closes(cfg):
    from datetime import date

    from streamer.roster.trades import days_to_deadline, deadline_text

    snap = _league()
    assert deadline_text(snap) == ""
    snap.rules["trade_deadline"] = "2026-11-18"
    assert days_to_deadline(snap, date(2026, 11, 6)) == 12
    assert deadline_text(snap, date(2026, 11, 6)) == "Trades close in 12 days (Nov 18)"
    assert deadline_text(snap, date(2026, 11, 18)) == "Trades close today (Nov 18)"
    assert deadline_text(snap, date(2026, 11, 19)) == "Trading closed on Nov 18"
