"""Tiny name helpers shared by the benchmark matcher and the roster matcher."""

from __future__ import annotations

_SUFFIXES = ("jr", "sr", "ii", "iii", "iv", "v")


def strip_suffix(name: str) -> str:
    parts = str(name).replace(".", "").split()
    while parts and parts[-1].lower().strip(",") in _SUFFIXES:
        parts.pop()
    return " ".join(parts)
