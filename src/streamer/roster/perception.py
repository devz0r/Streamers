"""How the other manager values a player -- without this tool.

A trade is accepted on what the other side *sees*, not on what our model
projects. What people see was measured, not assumed: Yahoo's "% rostered"
is the whole market's revealed opinion of every player, and on 237 players
at week 3 of 2026 (logit of % rostered, position intercepts) it is explained
by three things a manager can see, in these proportions:

* **the platform's own projection** -- 41% (90% interval 18-73%);
* **reputation**, last season's points a game (the season before at half
  weight), shrunk toward the projection for players with few games -- 46%
  (24-64%);
* **this season so far**, shrunk the same way -- 13% (-2 to 27%): three
  weeks in, a hot start has only begun to move the market.

Out of sample that blend explains 68% of the variation in rostership; our own
projection alone explains 60% and adds nothing once the three are in. The
gap between the two is where trades live: players the market rates above
our projection are worth more to sell than to hold, and the reverse.

Two smaller measured effects: a player on IR reads about 1.4 points a game
lower, and each game missed last season about 0.08 lower (injury history
barely registers). Refit with ``scripts/fit_perception.py``; weights live in
``perception.json``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..league.model import LeagueSnapshot, PlayerRow

MODEL_PATH = Path(__file__).with_name("perception.json")

DEFAULTS = {
    "weights": {"platform": 0.41, "reputation": 0.46, "season": 0.13},
    "reputation_prior_games": 4.0,
    "season_prior_games": 2.0,
    "previous_season_weight": 0.5,
    "ir_points": 1.4,
    "missed_game_points": 0.08,
    "full_season_games": 14,
}


def load() -> dict:
    try:
        raw = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(DEFAULTS)
    return {**DEFAULTS, **raw}


@dataclass
class Perceived:
    """A player as the rest of the league sees him."""

    value: float              # the blend, points a game
    platform: float           # what the platform projects (our projection when it shows none)
    reputation: float         # last seasons, shrunk
    season: float             # this season so far, shrunk
    last_ppg: float | None = None
    last_games: int = 0
    now_ppg: float | None = None
    now_games: int = 0


def _ours(p: PlayerRow) -> float:
    return float(p.ros_value if p.ros_value is not None else (p.projection or 0.0))


def _platform(p: PlayerRow) -> float:
    """The platform's projection, unless this week says nothing about him (a
    bye, an absence) -- then our projection stands in for its ranking."""
    v = p.platform_projection
    if v is None or v <= 0 or p.on_bye or p.is_out:
        return _ours(p)
    return float(v)


def season_lines(history: pd.DataFrame | None, season: int) -> dict[tuple[str, int], tuple[float, int]]:
    """(nflverse id, season) -> (points a game, games) for this season and the two before."""
    if history is None or history.empty:
        return {}
    h = history[history["season"].between(season - 2, season)]
    g = h.groupby(["player_id", "season"])["fantasy_points_ppr"].agg(["mean", "count"])
    return {(str(pid), int(s)): (float(r["mean"]), int(r["count"])) for (pid, s), r in g.iterrows()}


def perceive(players: list[PlayerRow], lines: dict, season: int, conf: dict | None = None) -> dict[str, Perceived]:
    conf = conf or load()
    w = conf["weights"]
    total = sum(w.values()) or 1.0
    out: dict[str, Perceived] = {}
    for p in players:
        plat = _platform(p)
        last = lines.get((str(p.nfl_id), season - 1)) if p.nfl_id else None
        prev = lines.get((str(p.nfl_id), season - 2)) if p.nfl_id else None
        now = lines.get((str(p.nfl_id), season)) if p.nfl_id else None
        gl, gp = (last[1] if last else 0), (prev[1] if prev else 0) * conf["previous_season_weight"]
        k = conf["reputation_prior_games"]
        rep = ((last[0] * gl if last else 0.0) + (prev[0] * gp if prev else 0.0) + k * plat) / (gl + gp + k)
        gn = now[1] if now else 0
        cur = ((now[0] * gn if now else 0.0) + conf["season_prior_games"] * plat) / (gn + conf["season_prior_games"])
        value = (w["platform"] * plat + w["reputation"] * rep + w["season"] * cur) / total
        if p.is_long_term_out or p.in_ir_slot:
            value -= conf["ir_points"]
        if last and gl:
            value -= conf["missed_game_points"] * max(conf["full_season_games"] - gl, 0)
        out[p.player_id] = Perceived(value=max(value, 0.0), platform=plat, reputation=rep, season=cur,
                                     last_ppg=last[0] if last else None, last_games=gl,
                                     now_ppg=now[0] if now else None, now_games=gn)
    return out


def for_snapshot(snapshot: LeagueSnapshot, history: pd.DataFrame | None) -> dict[str, Perceived]:
    return perceive(snapshot.all_players(), season_lines(history, snapshot.season), snapshot.season)


# ---------------------------------------------------------------------------
# Will he say yes?
# ---------------------------------------------------------------------------
#: A trade has to look clearly better to the other manager: people value
#: what they already own above what they are offered. At this perceived gain
#: (points a game of his roster value) he is even money to accept.
ACCEPT_CENTER = 1.0
#: How fast acceptance rises with the perceived gain.
ACCEPT_SCALE = 0.8
#: In an uneven deal the side getting the best player tends to feel it won.
CONSOLIDATION = 0.75


def engagement(acquisitions: int | None, weeks: int) -> float:
    """Managers who work the waiver wire answer trade offers; ones who never
    touch their roster mostly do not. 0.6 for a team with no moves, rising to
    1 at a pickup a week; 0.8 when the platform does not say."""
    if acquisitions is None:
        return 0.8
    return 0.6 + 0.4 * min(acquisitions / max(weeks, 1), 1.0)


def p_accept(perceived_gain: float, consolidation: int, engaged: float) -> float:
    """A rough chance he accepts. ``consolidation`` is +1 when he gets the
    best player of an uneven deal, -1 when he gives it up.

    The shape (a logistic in perceived gain, centred at a clear +1 point a
    game) is an assumption: offers and refusals are not published, so it
    cannot be fitted. It ranks offers; do not read the percentage as a
    measured rate."""
    x = perceived_gain + consolidation * CONSOLIDATION
    return engaged / (1.0 + math.exp(-(x - ACCEPT_CENTER) / ACCEPT_SCALE))
