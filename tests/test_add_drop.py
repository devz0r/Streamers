"""The "Try a pickup" table: every add priced against its sensible drops."""

from __future__ import annotations

from types import SimpleNamespace

from streamer.roster import add_drop, page
from streamer.roster.title_moves import TitleEngine
from test_season import _league
from test_title_moves import _fa, _wire


def _table(cfg):
    snap = _league([1.0, 1.1, 1.0, 1.05, 0.95, 1.0])
    _wire(snap, [_fa("star", "RB", 22.0), _fa("ok", "WR", 9.0)])
    engine = TitleEngine(snap, cfg, n_sims=400, extra=add_drop.candidates(snap))
    report = SimpleNamespace(snapshot=snap, optimisation=None)
    return snap, add_drop.build(engine, report, n_drops=3)


def test_every_add_has_priced_drops_from_your_roster(cfg):
    snap, data = _table(cfg)
    mine = {p.player_id for p in snap.my_team.roster}
    assert data["adds"] and data["base"]["win"] is None          # no opponent: no weekly odds
    for a in data["adds"]:
        assert 1 <= len(a["drops"]) <= 3
        assert all(d["id"] in mine and d["win"] is None for d in a["drops"])
        titles = [d["title"] for d in a["drops"]]
        assert titles == sorted(titles, reverse=True)
    # The star is the add worth most, and is listed first.
    assert data["adds"][0]["id"] == "star" and data["adds"][0]["drops"][0]["title"] > 0


def test_the_page_section_carries_the_table(cfg):
    snap, data = _table(cfg)
    html = page._try_pickup(SimpleNamespace(add_drop=data, snapshot=snap))
    assert "Try a pickup" in html and 'class="ad-pick"' in html and '"star"' in html
    assert page._try_pickup(SimpleNamespace(add_drop=None, snapshot=snap)) == ""
