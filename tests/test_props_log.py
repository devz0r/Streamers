"""The raw prop lines log keeps each line's first and latest pull, never a live price."""

import pandas as pd

from streamer.config import get_config
from streamer.data import props_log


def _frame(line, over):
    return pd.DataFrame([{"event_id": "e1", "bookmaker": "pinnacle", "market": "player_rush_yds", "stat": "rushing_yards",
                          "kind": "normal", "player": "Derrick Henry", "line": line, "over_price": over,
                          "under_price": -110, "yes_price": None}])


def test_first_and_latest_pull_are_kept_and_a_started_game_is_not_touched(tmp_path, monkeypatch):
    cfg = get_config()
    monkeypatch.setattr(props_log, "path", lambda _c: tmp_path / "prop_lines.parquet")
    n = lambda f, t, **k: props_log.record(f, cfg, 2026, 9, now=t, **k)       # noqa: E731
    assert n(_frame(70.5, -115), "2026-11-05T12:00:00+00:00") == 1
    assert n(_frame(72.5, -120), "2026-11-06T12:00:00+00:00") == 2            # first + latest
    assert n(_frame(73.5, -125), "2026-11-07T12:00:00+00:00") == 2            # the middle pull is dropped
    assert n(_frame(40.5, 300), "2026-11-08T18:00:00+00:00", started={"derrick henry"}) == 0
    held = pd.read_parquet(tmp_path / "prop_lines.parquet")
    assert list(held.line) == [70.5, 73.5]
