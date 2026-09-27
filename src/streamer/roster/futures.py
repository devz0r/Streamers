"""How a player's outlook can move: simulated futures, week by week.

Rest-of-season decisions are not about today's projection; they are about
where a player's projection could go -- a role that grows, an injury, a
starter ahead of him going down -- and what that is worth to your team.
Each simulated future of a player is built from what was measured on
2021-2025 (see DECISIONS.md, "How a player's outlook can move"):

* **His true level** starts at today's per-game projection plus a
  persistent error (the projection is wrong by the same amount every week),
  ``PERSISTENT_ERROR * sqrt(level)``, then drifts as a random walk -- no
  momentum, spread growing with the square root of time, ~0.8-1.15 points a
  game per week by position.
* **Absences** fire at his position-and-level rate (a backup QB 46% a week,
  a 16+ back 6%), scaled for role loss, and last a measured number of games
  (half are one game; a quarter four or more).
* **The next man up** inherits part of a lead's opportunity while he is out
  (RB 0.35, QB 0.30, TE 0.10 of the gap).
* **Byes** score zero; each week's score follows the measured outcome shape.

Validated before use: from any week of 2024-25, 82% of players' actual next
six-game averages fell inside the simulated 80% band (target 80%), 52% inside
the 50% band, misses split 9% below / 9% above, and the simulated mean was
within 0.12 points of the actual.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..league.model import LONG_TERM_OUT_STATUSES, PlayerRow
from . import outcome

SKILL = ("QB", "RB", "WR", "TE")

#: Games of evidence at which a manager has learned half of how wrong the
#: preseason-style projection was about a player (the production projection
#: moves volume on a 2.5-game half-life and anchors to ~17 games).
LEARN_GAMES = 4.0

#: Share of a missing lead's opportunity gap the next man up inherits.
TAKEOVER = {"RB": 0.35, "QB": 0.30, "TE": 0.10}


@dataclass
class Futures:
    """Simulated weekly scores: ``scores[sim, player, week]``, plus the
    per-game level a manager would *see* each week (0 when out) -- what
    lineup and claim decisions are made on, which lags the true level."""

    player_ids: list[str]
    weeks: list[int]
    scores: np.ndarray
    levels: np.ndarray        # the projection a manager would see that week (0 when out)

    def index(self, player_id: str) -> int:
        return self.player_ids.index(player_id)


def _params() -> dict:
    return outcome.load().get("futures", {})


def _hazard(pos: str, level: np.ndarray, conf: dict) -> np.ndarray:
    edges = conf.get("hazard_levels", [8.0, 12.0, 16.0])
    idx = np.digitize(level, edges)
    table = conf.get("hazard", {})
    base = np.array([table.get(f"{pos}|{i}", table.get(f"{pos}|1", 0.08)) for i in range(len(edges) + 1)])
    return base[idx] * float(conf.get("absence_scale", 1.2))


def _durations(pos: str, n: int, rng: np.random.Generator, conf: dict) -> np.ndarray:
    pmf = np.asarray(conf.get("duration", {}).get(pos) or [0.45, 0.19, 0.11, 0.08, 0.05, 0.04, 0.03, 0.05])
    pmf = pmf / pmf.sum()
    d = rng.choice(np.arange(1, len(pmf) + 1), size=n, p=pmf)
    # The last bucket is "8 or more": spread it over the rest of a season.
    long = d == len(pmf)
    d[long] = rng.integers(len(pmf), len(pmf) + 8, size=int(long.sum()))
    return d


def _initial_absence(p: PlayerRow, n: int, rng: np.random.Generator, conf: dict) -> np.ndarray:
    """Games still to miss from today's status."""
    if p.status in LONG_TERM_OUT_STATUSES:
        return np.maximum(_durations(p.position if p.position in SKILL else "WR", n, rng, conf), 4)
    if p.is_out and not p.on_bye:
        return _durations(p.position if p.position in SKILL else "WR", n, rng, conf)
    q = p.play_probability if p.play_probability is not None else 1.0
    if q < 1.0:
        sits = rng.random(n) >= q
        return np.where(sits, _durations(p.position if p.position in SKILL else "WR", n, rng, conf), 0)
    return np.zeros(n, int)


def simulate(
    players: list[PlayerRow],
    weeks: list[int],
    byes: dict[str, set[int]],
    n_sims: int = 2000,
    seed: int = 17,
    heirs: Iterable[tuple[str, str]] = (),
) -> Futures:
    """Simulate every player's next ``weeks``.

    ``heirs`` pairs (lead id, next-man-up id) on the same NFL team: while the
    lead is out, the heir's level gains the takeover share of the gap.
    """
    conf = _params()
    rng = np.random.default_rng(seed)
    n_p, n_w = len(players), len(weeks)
    scores = np.zeros((n_sims, n_p, n_w))
    levels = np.zeros((n_sims, n_p, n_w))
    c = float(conf.get("persistent_error", 1.5))
    step_sd = conf.get("step", {})

    base = np.array([max(float(p.ros_value if p.ros_value is not None else (p.projection or 0.0)), 0.0)
                     for p in players])
    # Two parts to a player's level: ``walk`` -- where his projection is and
    # will move to, visible to everyone as it happens -- and ``err``, how
    # wrong today's projection is about him, which nobody sees at first and
    # which games reveal (a manager's read after k games carries
    # k / (k + LEARN_GAMES) of it). Scores come from the true level; lineups
    # and claims can only use the visible one.
    walk = np.repeat(base[None, :], n_sims, axis=0)
    err = c * np.sqrt(base)[None, :] * rng.standard_normal((n_sims, n_p))
    for j, p in enumerate(players):
        if p.position not in SKILL:
            err[:, j] = 0.0                             # D/ST and K are streamed, not held
    seen = np.zeros((n_sims, n_p))
    out_left = np.stack([_initial_absence(p, n_sims, rng, conf) for p in players], axis=1) \
        if players else np.zeros((n_sims, 0), int)
    heir_of = {players.index(next(q for q in players if q.player_id == h)): players.index(
        next(q for q in players if q.player_id == lead))
        for lead, h in heirs
        if any(q.player_id == lead for q in players) and any(q.player_id == h for q in players)}

    sd0 = np.array([float(p.outcome_sd if p.outcome_sd is not None else (p.projection_sd or 6.0)) for p in players])
    for k, week in enumerate(weeks):
        on_bye = np.array([bool(p.team) and week in byes.get(p.team, set()) for p in players])
        # New absences (not on bye, not already out).
        level = np.maximum(walk + err, 0.0)
        visible = np.maximum(walk + err * seen / (seen + LEARN_GAMES), 0.0)
        for j, p in enumerate(players):
            if p.position not in SKILL or on_bye[j]:
                continue
            fresh = (out_left[:, j] == 0) & (rng.random(n_sims) < _hazard(p.position, level[:, j], conf))
            if fresh.any():
                out_left[fresh, j] = _durations(p.position, int(fresh.sum()), rng, conf)
        playing = (out_left == 0) & ~on_bye[None, :]
        eff, eff_vis = level.copy(), visible.copy()
        for h, lead in heir_of.items():
            share = TAKEOVER.get(players[h].position, 0.0)
            lead_out = out_left[:, lead] > 0
            eff[:, h] = np.where(lead_out, level[:, h] + share * np.maximum(level[:, lead] - level[:, h], 0.0),
                                 level[:, h])
            eff_vis[:, h] = np.where(lead_out, visible[:, h] + share * np.maximum(visible[:, lead] - visible[:, h], 0.0),
                                     visible[:, h])
        levels[:, :, k] = np.where(playing, eff_vis, 0.0)
        finals = [j for j, p in enumerate(players) if k == 0 and p.actual_points is not None]
        for j, p in enumerate(players):
            probs, zq = outcome.shape(p.position, float(base[j]))
            sd = sd0[j] * np.sqrt(np.maximum(eff[:, j], 1.0) / max(base[j], 1.0))
            u = rng.random(n_sims)
            s = eff[:, j] + sd * np.interp(u, probs, zq)
            if p.position in SKILL:
                s = np.maximum(s, 0.0)
            scores[:, j, k] = np.where(playing[:, j], s, 0.0)
        for j in finals:
            # This week's game is over: his score is a fact.
            scores[:, j, k] = float(players[j].actual_points)
        # A game passes: absences tick down (a bye does not use one up), and
        # the true level drifts.
        out_left = np.where(on_bye[None, :], out_left, np.maximum(out_left - 1, 0))
        seen += playing
        for j, p in enumerate(players):
            if p.position in SKILL:
                walk[:, j] = walk[:, j] + float(step_sd.get(p.position, 0.95)) * rng.standard_normal(n_sims)
    return Futures(player_ids=[p.player_id for p in players], weeks=list(weeks), scores=scores, levels=levels)
