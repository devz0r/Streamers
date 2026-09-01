"""Configuration loading.

Every scoring rule, shrinkage constant and modelling knob lives in
``config.yaml`` at the repository root. Nothing else in the codebase is
allowed to hard-code a scoring value -- tests assert the ESPN defaults come
from here.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Directory containing ``config.yaml``.

    Resolution order: ``STREAMER_ROOT`` env var, then the first ancestor of
    this file that contains a ``config.yaml``, then the current directory.
    """
    env = os.environ.get("STREAMER_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config.yaml").exists():
            return parent
    return Path.cwd()


@dataclass(frozen=True)
class Config:
    """Parsed ``config.yaml`` with convenience accessors."""

    raw: dict[str, Any]
    root: Path = field(default_factory=repo_root)

    # -- section accessors -------------------------------------------------
    @property
    def season(self) -> dict[str, Any]:
        return self.raw["season"]

    @property
    def league(self) -> dict[str, Any]:
        return self.raw["league"]

    @property
    def kicker_scoring(self) -> dict[str, float]:
        return self.raw["kicker_scoring"]

    @property
    def dst_scoring(self) -> dict[str, Any]:
        return self.raw["dst_scoring"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def shrinkage(self) -> dict[str, float]:
        return self.raw["shrinkage"]

    @property
    def ledger(self) -> dict[str, Any]:
        return self.raw["ledger"]

    @property
    def odds(self) -> dict[str, Any]:
        return self.raw["odds"]

    @property
    def weather(self) -> dict[str, Any]:
        return self.raw["weather"]

    @property
    def publish(self) -> dict[str, Any]:
        return self.raw["publish"]

    @property
    def current_season(self) -> int:
        return int(self.season["current"])

    @property
    def train_seasons(self) -> list[int]:
        return [int(s) for s in self.season["train_seasons"]]

    @property
    def startable_rank(self) -> int:
        return int(self.league["startable_rank"])

    def team_prior_games(self, position: str) -> float:
        """Shrinkage strength for team rate priors, which is position-specific.

        Accepts either a scalar (applied everywhere) or a ``{position: value}``
        mapping in ``config.yaml``.
        """
        value = self.shrinkage["team_rate_prior_games"]
        if isinstance(value, dict):
            if position in value:
                return float(value[position])
            return float(next(iter(value.values())))
        return float(value)

    # -- paths -------------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a configured directory, creating it if absent."""
        rel = self.raw["paths"][key]
        p = self.root / rel
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_dir(self) -> Path:
        p = self.root / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def raw_dir(self) -> Path:
        return self.path("raw")

    @property
    def results_dir(self) -> Path:
        return self.path("results")

    @property
    def reports_dir(self) -> Path:
        return self.path("reports")

    @property
    def docs_dir(self) -> Path:
        return self.path("docs")


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from ``path`` (default: ``<repo root>/config.yaml``)."""
    root = repo_root()
    cfg_path = Path(path) if path is not None else root / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path} did not parse to a mapping")
    return Config(raw=raw, root=cfg_path.resolve().parent)


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide cached configuration."""
    return load_config()
