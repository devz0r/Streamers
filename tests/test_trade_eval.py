"""The page's trade evaluator: its data, its estimate, and the script that
must compute exactly what the Python reference does."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from streamer.roster import season as season_mod
from streamer.roster import trade_eval
from streamer.roster.season import SeasonModel
from streamer.roster.trades import TradeFinder
from test_trades import _league


@pytest.fixture(autouse=True)
def _no_byes(monkeypatch):
    monkeypatch.setattr(season_mod, "nfl_byes", lambda *a, **k: {})


@pytest.fixture(scope="module")
def league():
    from streamer.config import get_config

    mp = pytest.MonkeyPatch()
    mp.setattr(season_mod, "nfl_byes", lambda *a, **k: {})
    snap = _league()
    finder = TradeFinder(snap, SeasonModel(snap, get_config(), n_sims=1500))
    finder.find(5)
    data = trade_eval.export(snap, finder.model, finder, finder.priced)
    mp.undo()
    return snap, finder, data


def test_random_numbers_are_the_same_everywhere():
    assert trade_eval.fnv1a("a") == 0xE40C292C
    u = trade_eval.uniforms(12345, 3, 4)
    assert u.shape == (3, 4) and ((u >= 0) & (u < 1)).all()
    assert (trade_eval.uniforms(12345, 3, 4) == u).all()
    z = trade_eval.normals(7, 20000)
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.05


def test_a_team_is_likelier_to_win_the_title_with_more_points(league):
    _snap, _finder, data = league
    for j, curve in enumerate(data["title"]):
        own = [row[j] for row in curve]
        assert own == sorted(own), "a team's title odds rise with its own weekly points"


def test_a_lopsided_gift_helps_the_receiver_and_hurts_the_giver(league):
    snap, _finder, data = league
    me = next(t for t in data["teams"] if t["me"])
    mirror = next(t for t in data["teams"] if t["id"] == "1")
    star = max((pid for pid in mirror["r"] if data["players"][pid]["pos"] == "WR"),
               key=lambda pid: data["players"][pid]["o"])
    scrub = min((pid for pid in me["r"] if data["players"][pid]["pos"] == "WR"),
                key=lambda pid: data["players"][pid]["o"])
    r = trade_eval.evaluate(data, "1", [scrub], [star])
    assert r["shift_me"] > 0 > r["shift_them"]
    assert r["title_me"][1] > r["title_me"][0] and r["title_them"][1] < r["title_them"][0]
    assert r["p_accept"] < 0.2 and r["legal"]
    assert r["why_you"] and r["why_them"]


def test_receiving_more_than_you_give_forces_a_drop_you_can_choose(league):
    _snap, _finder, data = league
    me = next(t for t in data["teams"] if t["me"])
    other = next(t for t in data["teams"] if t["id"] == "2")
    get = [pid for pid in other["r"] if data["players"][pid]["pos"] in ("RB", "WR")][:2]
    give = [next(pid for pid in me["r"] if data["players"][pid]["pos"] == "RB")]
    auto = trade_eval.evaluate(data, "2", give, get)
    assert len(auto["my_drops"]) == 1 and auto["my_drops"][0] not in get
    mine = next(pid for pid in me["r"] if pid not in give and pid not in auto["my_drops"]
                and data["players"][pid]["pos"] == "WR")
    chosen = trade_eval.evaluate(data, "2", give, get, my_drops=[mine])
    assert chosen["my_drops"] == [mine] and chosen["exact"] is None


def test_the_estimate_is_checked_against_the_full_simulation(league):
    _snap, finder, data = league
    assert data["check"]["n"] == len(finder.priced)
    e = data["exact"][0]
    r = trade_eval.evaluate(data, e[0], e[1], e[2])
    assert r["exact"] == e[3:]


_PARITY = r"""
const fs = require('fs');
const ST = require(process.argv[2]);
const D = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const cases = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
const sim = new ST.Sim(D);
process.stdout.write(JSON.stringify(cases.map(c => ST.evaluate(D, c.partner, c.give, c.get, c.drops, sim))));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_page_script_computes_exactly_what_python_does(league, tmp_path):
    _snap, _finder, data = league
    me = next(t for t in data["teams"] if t["me"])
    skill = lambda ids: [i for i in ids if data["players"][i]["pos"] in ("QB", "RB", "WR", "TE")]  # noqa: E731
    cases = [{"partner": e[0], "give": e[1], "get": e[2], "drops": None} for e in data["exact"][:4]]
    for t in data["teams"]:
        if t["me"]:
            continue
        cases.append({"partner": t["id"], "give": skill(me["r"])[:2], "get": skill(t["r"])[1:3], "drops": None})
        cases.append({"partner": t["id"], "give": skill(me["r"])[3:4], "get": skill(t["r"])[:3],
                      "drops": [skill(me["r"])[-1]]})
    (tmp_path / "data.json").write_text(json.dumps(data))
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    (tmp_path / "parity.js").write_text(_PARITY)
    run = subprocess.run(["node", str(tmp_path / "parity.js"), str(trade_eval.SCRIPT),
                          str(tmp_path / "data.json"), str(tmp_path / "cases.json")],
                         capture_output=True, text=True, timeout=120, check=True)
    sim = trade_eval.Sim(data)
    for c, js in zip(cases, json.loads(run.stdout)):
        py = trade_eval.evaluate(data, c["partner"], c["give"], c["get"], c["drops"], sim=sim)
        for k in ("shift_me", "shift_them", "lineup_me", "seen", "consolidation", "p_accept"):
            assert js[k] == pytest.approx(py[k], abs=1e-9), k
        for k in ("title_me", "title_them", "playoffs_me", "playoffs_them"):
            assert js[k] == pytest.approx(list(py[k]), abs=1e-9), k
        assert js["my_drops"] == py["my_drops"] and js["their_drops"] == py["their_drops"]
        assert js["why_you"] == py["why_you"] and js["why_them"] == py["why_them"]
        assert js["legal"] == py["legal"] and js["exact"] == py["exact"]


def test_the_page_carries_the_evaluator(league):
    from types import SimpleNamespace

    from streamer.roster.page import _trade_evaluator

    snap, _finder, data = league
    html = _trade_evaluator(SimpleNamespace(trade_eval=data, snapshot=snap))
    assert 'id="te-espn"' in html and "StreamerTrade.mount('te-espn')" in html
    payload = html.split('class="te-data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(payload.replace("<\\/", "</"))["ns"] == trade_eval.NS
    assert _trade_evaluator(SimpleNamespace(trade_eval=None, snapshot=snap)) == ""
