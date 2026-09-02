"""Matching platform players to nflverse players.

ESPN and Yahoo identify players by their own ids; nflverse by GSIS id. The
only shared key is the name, which the platforms format differently
("D.J. Moore" / "DJ Moore", "Kenneth Walker III" / "Kenneth Walker"). Names are
normalised aggressively and matched on (name, position), with the NFL team as
a tie-breaker. Defences are matched by team.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

from .._names import strip_suffix  # noqa: F401  (re-exported for callers)

_SUFFIXES = ("jr", "sr", "ii", "iii", "iv", "v")


def normalize_name(name: object) -> str:
    """Lower-case, strip accents, punctuation and generational suffixes."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = text.lower().replace("'", "").replace(".", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    parts = [p for p in text.split() if p]
    while parts and parts[-1] in _SUFFIXES:
        parts.pop()
    return " ".join(parts)


@dataclass(frozen=True)
class MatchResult:
    """Platform player id -> nflverse player id, plus what could not be matched."""

    mapping: dict[str, str]
    unmatched: list[str]


def build_index(nfl: pd.DataFrame) -> dict[tuple[str, str], list[tuple[str, str | None]]]:
    """Index nflverse players by (normalised name, position) -> [(id, team)].

    ``nfl`` needs ``player_id``, ``player_display_name``, ``position`` and
    ``team`` (the most recent team is best).
    """
    index: dict[tuple[str, str], list[tuple[str, str | None]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in nfl.itertuples(index=False):
        key = (normalize_name(row.player_display_name), str(row.position))
        team = getattr(row, "team", None)
        entry = (str(row.player_id), team if isinstance(team, str) else None)
        dedupe = (*key, entry[0])
        if dedupe in seen:
            continue
        seen.add(dedupe)
        index.setdefault(key, []).append(entry)
    return index


def match_players(
    platform: list, index: dict[tuple[str, str], list[tuple[str, str | None]]]
) -> MatchResult:
    """Match each platform :class:`PlayerRow` to an nflverse id.

    Skips defences (matched by team elsewhere). When several nflverse players
    share a name and position, the one on the same NFL team wins; failing that,
    the first is taken and the ambiguity is logged in ``unmatched`` as a note.
    """
    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    for p in platform:
        if p.position == "DST":
            continue
        key = (normalize_name(p.name), p.position)
        candidates = index.get(key, [])
        if not candidates:
            # Try first-initial + last name, which catches "DJ" vs "D.J." splits.
            parts = key[0].split()
            if len(parts) >= 2:
                loose = [
                    (k, v) for k, v in index.items()
                    if k[1] == p.position and k[0].split()[-1:] == parts[-1:]
                    and k[0][:1] == parts[0][:1]
                ]
                if len(loose) == 1:
                    candidates = loose[0][1]
        if not candidates:
            unmatched.append(f"{p.name} ({p.position})")
            continue
        chosen = candidates[0]
        if len(candidates) > 1 and p.team:
            same_team = [c for c in candidates if c[1] == p.team]
            if same_team:
                chosen = same_team[0]
        mapping[p.player_id] = chosen[0]
    return MatchResult(mapping=mapping, unmatched=unmatched)
