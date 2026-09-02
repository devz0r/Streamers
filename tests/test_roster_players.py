"""Matching platform player names to nflverse."""

from __future__ import annotations

import pandas as pd

from streamer.league.model import PlayerRow
from streamer.roster.players import build_index, match_players, normalize_name


def test_normalize_name_strips_punctuation_and_suffixes():
    assert normalize_name("D.J. Moore") == "dj moore"
    assert normalize_name("Kenneth Walker III") == "kenneth walker"
    assert normalize_name("Marvin Harrison Jr.") == "marvin harrison"
    assert normalize_name("Ja'Marr Chase") == "jamarr chase"
    assert normalize_name("  Amon-Ra   St. Brown ") == "amon ra st brown"
    assert normalize_name(None) == ""


def _index():
    return build_index(pd.DataFrame({
        "player_id": ["00-1", "00-2", "00-3", "00-4"],
        "player_display_name": ["DJ Moore", "Kenneth Walker", "Josh Allen", "Josh Allen"],
        "position": ["WR", "RB", "QB", "LB"],
        "team": ["CHI", "SEA", "BUF", "JAX"],
    }))


def test_matches_on_name_and_position():
    rows = [
        PlayerRow(player_id="a", name="D.J. Moore", position="WR", team="CHI"),
        PlayerRow(player_id="b", name="Kenneth Walker III", position="RB", team="SEA"),
        PlayerRow(player_id="c", name="Josh Allen", position="QB", team="BUF"),
    ]
    result = match_players(rows, _index())
    assert result.mapping == {"a": "00-1", "b": "00-2", "c": "00-3"}
    assert result.unmatched == []


def test_position_disambiguates_shared_names():
    rows = [PlayerRow(player_id="c", name="Josh Allen", position="QB", team="BUF")]
    assert match_players(rows, _index()).mapping["c"] == "00-3"


def test_defenses_and_unknowns_are_reported():
    rows = [
        PlayerRow(player_id="d", name="Chiefs D/ST", position="DST", team="KC"),
        PlayerRow(player_id="e", name="Nobody Real", position="WR", team="KC"),
    ]
    result = match_players(rows, _index())
    assert "d" not in result.mapping                 # defences are matched by team elsewhere
    assert result.unmatched == ["Nobody Real (WR)"]
