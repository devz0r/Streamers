"""The golden-output script's helpers: what is masked, and what is reported as moved."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("golden_page", Path(__file__).resolve().parents[1] / "scripts" / "golden_page.py")
gp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gp)


def test_times_and_build_durations_are_masked_but_numbers_are_not():
    text = '{"generated_at": "2026-11-14T11:41:51+00:00", "p": 0.4321}\nview as X (light) in 13s at 2026-11-14 11:42:03Z'
    out = gp.masked(text)
    assert "2026-11-14" not in out and "in <n>s" in out and "0.4321" in out
    assert gp.masked(text) == gp.masked(text.replace("11:41:51", "09:00:00").replace("in 13s", "in 99s"))


def test_compare_reports_added_removed_and_changed_files(capsys):
    a = {"index.html": "aaa", "week_10.html": "bbb", "gone.json": "ccc"}
    b = {"index.html": "aaa", "week_10.html": "zzz", "new.json": "ddd"}
    assert gp.compare(a, b) == 3
    out = capsys.readouterr().out
    assert "changed  week_10.html" in out and "removed  gone.json" in out and "added    new.json" in out
    assert gp.compare(a, dict(a)) == 0


def test_the_fingerprint_moves_with_an_input_and_not_with_a_source_file(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.json").write_text("1")
    (tmp_path / "config.yaml").write_text("a: 1")
    (tmp_path / "src").mkdir()
    monkeypatch.setattr(gp, "ROOT", tmp_path)
    before = gp.fingerprint()
    (tmp_path / "src" / "code.py").write_text("x = 1")
    assert gp.fingerprint() == before                           # code is what a check changes: not an input
    (tmp_path / "data" / "x.json").write_text("2")
    assert gp.fingerprint() != before


def test_building_overlays_the_inputs_of_another_tree(tmp_path, monkeypatch):
    src, mine = tmp_path / "src_tree", tmp_path / "workspace"
    for root, tag in ((src, "ref"), (mine, "workspace")):
        (root / "data").mkdir(parents=True)
        (root / "data" / "x.json").write_text(tag)
        (root / "config.yaml").write_text(tag)
        (root / "src").mkdir()
        (root / "src" / "m.py").write_text(tag)
    (mine / "data" / "leagues" / "espn").mkdir(parents=True)
    (mine / "data" / "leagues" / "espn" / "week_3.json").write_text("{}")
    seen = {}

    def fake_run(cmd, cwd, env, capture_output, text):
        seen["data"] = (Path(cwd) / "data" / "x.json").read_text()
        seen["config"] = (Path(cwd) / "config.yaml").read_text()
        seen["code"] = (Path(cwd) / "src" / "m.py").read_text()
        seen["workers"] = env.get("STREAMER_VIEW_WORKERS")

        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(gp.subprocess, "run", fake_run)
    monkeypatch.setattr(gp, "ROOT", mine)
    work, _secs = gp.build("t", src=src, overlay=mine, workers="4")
    assert seen == {"data": "workspace", "config": "workspace", "code": "ref", "workers": "4"}   # the ref's code on the workspace's data
    gp.shutil.rmtree(work, ignore_errors=True)


def test_every_time_format_the_page_writes_is_masked_and_the_diff_points_at_what_differs():
    for a, b in (('updated Sat 14 Nov 2026, 16:08 UTC', 'updated Sat 14 Nov 2026, 16:16 UTC'),
                 ('"built_at":"2026-11-14T16:08+00:00"', '"built_at":"2026-11-14T16:16+00:00"'),
                 ('synced 2026-11-14 09:23 UTC', 'synced 2026-11-14 10:00 UTC'),
                 ('at 2026-11-14T16:08:03.5Z', 'at 2026-11-14T17:00:00Z')):
        assert gp.masked(a) == gp.masked(b), a
    assert gp.masked("p 0.4321 Nov 28") != gp.masked("p 0.4322 Nov 28")                # a real number still shows
    assert gp._where("a" * 200 + "X" + "b" * 200, "a" * 200 + "Y" + "b" * 200) == [
        "..." + "a" * 70 + "[-X-]{+Y+}" + "b" * 70 + "..."]
    assert gp._where("same", "same") == []


def test_the_diff_of_two_megabyte_lines_is_quick():
    import random
    import time

    rng = random.Random(1)
    a = "".join(rng.choice('abcdefgh <>"=/') for _ in range(1_000_000))
    b = a[:400_000] + "X" + a[400_001:900_000] + "YY" + a[900_000:]
    t = time.time()
    got = gp._where(a, b)
    assert time.time() - t < 3 and got and "[-" in got[0]            # a full sequence match here took most of an hour


def test_text_relative_to_the_moment_of_the_build_is_masked():
    for a, b in (("yours &middot; 14h ago</span>", "yours &middot; 15h ago</span>"),
                 ("read Sat 1:29 PM ET (10h before this page was built)", "read Sat 1:29 PM ET (11h before this page was built)"),
                 ("<b>Trades close in 14 days (Nov 28).</b>", "<b>Trades close in 13 days (Nov 28).</b>"),
                 ("trades close in 14 days (nov 28)", "trades close in 13 days (nov 28)"),
                 ("(WR, 88% to play): inactives about Sunday", "(WR, 88% to play): inactives about 11:30 AM ET")):
        assert gp.masked(a) == gp.masked(b), a
    assert gp.masked("88% to play") != gp.masked("87% to play")
