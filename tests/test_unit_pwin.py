"""P(win) beside each unit in the full K and D/ST lists, and released kickers."""

from __future__ import annotations

from types import SimpleNamespace as NS

from streamer.league.model import PlayerRow
from streamer.roster.lineup import StreamOption


def _k(pid, name, team, proj, status=""):
    return PlayerRow(player_id=pid, name=name, position="K", team=team, projection=proj, status=status)


def test_a_released_kicker_is_not_shown_as_available(monkeypatch):
    from streamer import publish
    from streamer.league import store

    carlson_y = _k("1", "Daniel Carlson", "NO", 9.2, status="NA")       # Yahoo: last team, "NA"
    little = _k("2", "Cam Little", "JAX", 8.9)
    snap = NS(free_agents=[carlson_y, little], teams=[], platform="yahoo")
    monkeypatch.setattr(store, "load_snapshot", lambda cfg, week=None: snap)
    out = publish.unit_availability(None, 9, {"K": ["NO", "JAX"]}, {"NO": "D.Carlson", "JAX": "C.Little"})
    assert out[("K", "NO")] == ("released", "")
    assert out[("K", "JAX")] == ("available", "")
    carlson_e = _k("1", "Daniel Carlson", None, 9.2)                    # ESPN: no team at all
    snap.free_agents = [carlson_e, little]
    out = publish.unit_availability(None, 9, {"K": ["NO", "JAX"]}, {"NO": "D.Carlson", "JAX": "C.Little"})
    assert out[("K", "NO")] == ("released", "")


def test_each_unit_gets_its_p_win_and_why_it_differs_from_the_projection():
    from streamer.roster.page import unit_pwin

    mine = StreamOption(_k("m", "Evan McPherson", "CIN", 8.6), 0.661, True, low=4, high=13, noise=0.0)
    steady = StreamOption(_k("a", "Cam Little", "JAX", 8.4), 0.664, False, low=4, high=13, noise=0.002)
    swingy = StreamOption(_k("b", "Big Leg", "NO", 9.2), 0.655, False, low=2, high=16, noise=0.001)
    tied = StreamOption(_k("c", "Tied Up", "PHI", 8.5), 0.650, False, low=4, high=13, noise=0.001,
                        links=["moves with their Jalen Hurts (+0.30)"])
    out = unit_pwin(NS(streams={"K": [steady, mine, swingy, tied]}))["K"]
    assert out["CIN"]["mine"] and out["CIN"]["why"] == "yours"
    assert "within simulation noise" in out["JAX"]["why"]
    assert "wider range" in out["NO"]["why"] and "costs a favourite" in out["NO"]["why"]
    assert "Jalen Hurts" in out["PHI"]["why"]
    assert round(out["NO"]["d"], 1) == -0.6
