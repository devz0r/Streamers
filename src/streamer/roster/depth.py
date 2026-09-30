"""Who else is at the position: how much of an absent lead's work his backup takes.

A league-wide average share of the lead's work, applied to every backup,
cannot tell the Jets' backfield -- Braelon Allen with every non-Hall snap,
Isaiah Davis with none -- from a three-way committee. Box scores cannot
either: a back who never touches the ball has no row. Snap counts can
(Pro Football Reference, via nflverse: free).

**Concentration**: the next man up's share of the offensive snaps at his
position that did not go to the lead, over the team's last four games. On
2021-2025 absences it predicts how much of the job he takes, and so how
much he keeps once the lead is back. The fit (``scripts/fit_next_man_up.py``,
each season held out in turn) is ``next_man_up.json``: the share at the
average concentration, and how it moves with concentration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .players import normalize_name

MODEL_PATH = Path(__file__).with_name("next_man_up.json")
#: Team games looked back over.
GAMES = 4
#: The expected share is kept inside this range: even the clearest backups
#: (75%+ of the other snaps) averaged 0.80 of the gap over a whole absence.
SHARE_BOUNDS = (0.1, 0.9)


def load() -> dict:
    try:
        return json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def prepare(snaps: pd.DataFrame) -> pd.DataFrame:
    """Snap counts with a name key, offensive snaps only."""
    if snaps is None or snaps.empty:
        return pd.DataFrame(columns=["season", "week", "team", "position", "key", "offense_snaps"])
    out = snaps[["season", "week", "team", "position", "player", "offense_snaps"]].copy()
    out["key"] = out["player"].map(normalize_name)
    out["offense_snaps"] = out["offense_snaps"].fillna(0.0).astype(float)
    return out


def concentration(snaps: pd.DataFrame, season: int, week: int, team: str, position: str,
                  player: str, lead: str | None, games: int = GAMES) -> float | None:
    """``player``'s share of the offensive snaps at ``position`` that did not
    go to ``lead``, over ``team``'s last ``games`` games before ``week``.
    Names are matched normalised. None when there is nothing to go on."""
    s = snaps[(snaps["season"] == season) & (snaps["team"] == team) & (snaps["week"] < week)
              & (snaps["position"] == position)]
    if s.empty:
        return None
    weeks = sorted(s["week"].unique())[-games:]
    s = s[s["week"].isin(weeks)]
    lead_key = normalize_name(lead) if lead else None
    rest = s[s["key"] != lead_key] if lead_key else s
    total = float(rest["offense_snaps"].sum())
    mine = float(rest.loc[rest["key"] == normalize_name(player), "offense_snaps"].sum())
    # A next man up has touches, so he has snaps: none found means his name
    # did not match the snap data, not that he never plays.
    if total <= 0 or mine <= 0:
        return None
    return mine / total


def share(position: str, conc: float | None, default: float, model: dict | None = None) -> float:
    """The expected share of the lead's work for a next man up with this
    concentration: the position's fitted share at the average, moved by the
    fitted slope. The position default when there is no fit or no snaps."""
    fit = (model if model is not None else load()).get(position)
    if not fit or conc is None:
        return default
    value = float(fit["base"]) + float(fit["slope"]) * (float(conc) - float(fit["mean"]))
    return float(min(max(value, SHARE_BOUNDS[0]), SHARE_BOUNDS[1]))
