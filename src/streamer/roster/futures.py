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
  game per week by position, 23% wider for a rookie (4% for a second-year
  player), as their projections were measured to move.
* **Absences** fire at his position-and-level rate (a backup QB 46% a week,
  a 16+ back 6%), scaled for role loss, and last a measured number of games
  (half are one game; a quarter four or more).
* **The next man up** inherits part of a lead's opportunity while he is out:
  on average RB 0.55, QB 0.60, TE 0.35 of the gap, but drawn afresh for each
  absence with a wide spread (sd 0.8 at RB) -- some backups become the bell
  cow, some watch a third man get the work. When the lead comes back the
  backup keeps a quarter of what he took over, and the lead loses it: a
  takeover can stick (2021-2025 absences; see DECISIONS.md, "The next man
  up, speculated on").
* **Byes** score zero; each week's score follows the measured outcome shape.

Graded on replayed seasons (``streamer.roster.ros_backtest``,
``scripts/fit_ros.py``): every rostered player at weeks 3-10 of 2022-2025,
projected by the production code on the games before that week, simulated to
week 17 and scored against what holding him returned. Four calibrations came
out of it, each fitted on two seasons and checked on the other two:

* **On a reserve list, most players are gone**: 63% never played again that
  season; the rest took about four games. The old rule (back after four)
  had them worth 1.3-1.4 points a team game too much.
* **The absence hazard reads the projection**, scaled by level: fringe
  players miss twice as often as their injury rate says (roles and roster
  spots vanish), stars rarely miss. Reading the hidden true level instead
  gave a star drawn below his number a backup's absence rate.
* **The persistent error is multiplicative** (a level cannot go below zero,
  so a normal error floored there lifted a low player's average), with a
  per-position, per-level scale -- a star's outlook is narrower than
  sqrt(level) said, a fringe back's wider.
* **A per-position, per-level offset** so a simulated player scores per game
  what players like him did.

Average miss on rest-of-season points per team game fell 9-10% on the
held-out seasons (2.76 to 2.48, 2.64 to 2.40), bias from +0.34/+0.45 to
-0.17/+0.08.

**What a manager sees** decides what holding similar players is worth: you
can only start the better one if you can tell which he is. Lineups and
claims read a player through a lens fitted to how our real projection
behaved (DECISIONS.md, "Form that fades, and a read that lags"):

* the persistent error is sized to the spread of points per game played,
  not per team game, so early form lasts as long as it really did;
* a read after k games carries k / (k + LEARN_GAMES) of what they showed,
  the luck of those games included;
* only a share of a player's real change shows (``visibility``, about a
  third; RB 0.46), and the read also moves on its own noise
  (``projection_noise``), fading at the projection's 2.5-game half-life.

Fitted that way, a second quarterback of the same standing is worth what
it was on 2022-2025 (+0.6 a week, not +2.1), and a third +0.25.
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

#: Week-to-week persistence of the projection's own noise -- moves in what a
#: manager sees that are not moves in the player -- matching the production
#: projection's 2.5-game half-life on recent volume.
PROJECTION_NOISE_PERSIST = 0.5 ** (1 / 2.5)

#: Share of a missing lead's opportunity gap the next man up inherits, on
#: average (the config's ``roster.next_man_up`` overrides).
TAKEOVER = {"RB": 0.55, "QB": 0.60, "TE": 0.35}
#: Spread of that share from one absence to the next, beyond game noise.
TAKEOVER_SD = {"RB": 0.80, "QB": 0.50, "TE": 0.45}
#: Bounds on a drawn share: a third man can take some of the backup's own
#: work, and a backup can out-touch the man he replaced, but not by much.
SHARE_RANGE = (-0.3, 1.3)
#: Share of his takeover the next man up keeps once the lead is back.
KEPT = 0.26


@dataclass
class Futures:
    """Simulated weekly scores: ``scores[sim, player, week]``, plus the
    per-game level a manager would *see* each week (0 when out) -- what
    lineup and claim decisions are made on, which lags the true level."""

    player_ids: list[str]
    weeks: list[int]
    scores: np.ndarray
    levels: np.ndarray        # the projection a manager would see that week (0 when out)
    played: np.ndarray | None = None   # whether he played that week (not out, not on bye)

    def index(self, player_id: str) -> int:
        return self.player_ids.index(player_id)


def _params() -> dict:
    return outcome.load().get("futures", {})


def _hazard(pos: str, level: np.ndarray, conf: dict) -> np.ndarray:
    edges = conf.get("hazard_levels", [8.0, 12.0, 16.0])
    idx = np.digitize(level, edges)
    table = conf.get("hazard", {})
    base = np.array([table.get(f"{pos}|{i}", table.get(f"{pos}|1", 0.08)) for i in range(len(edges) + 1)])
    scale = conf.get("hazard_scale")
    if scale:
        # Fitted to how often players at each level actually played the
        # rest of the season (fringe players lose roles and get cut; stars
        # rarely miss), replacing one flat scale for every level.
        mult = np.asarray(scale["scale"], dtype=float)[np.digitize(level, scale["levels"])]
        return np.minimum(base[idx] * mult, 0.95)
    return base[idx] * float(conf.get("absence_scale", 1.2))


def _durations(pos: str, n: int, rng: np.random.Generator, conf: dict, already: int = 0) -> np.ndarray:
    """Games an absence lasts (from its start), or -- for one that has
    already run ``already`` games -- the games still to go: the measured
    lengths conditioned on lasting longer than that. Half of all absences
    last one game, but one that has already run three is a long one."""
    pmf = np.asarray(conf.get("duration", {}).get(pos) or [0.45, 0.19, 0.11, 0.08, 0.05, 0.04, 0.03, 0.05])
    pmf = pmf / pmf.sum()
    # The last bucket is "8 or more": spread it over the rest of a season.
    full = np.concatenate([pmf[:-1], np.full(8, pmf[-1] / 8)])
    lengths = np.arange(1, len(full) + 1)
    if already > 0:
        full = np.where(lengths > already, full, 0.0)
        if full.sum() <= 0:                  # out longer than any measured absence
            return rng.integers(1, 9, size=n)
    d = rng.choice(lengths, size=n, p=full / full.sum())
    return d - already


#: Games left "out" for a player who does not come back this season.
GONE = 99


def ir_gone(ir: dict, level: float, missed: int) -> float:
    """Chance a player on a reserve list does not play again this season, by
    his season value and how many games in a row he has already missed."""
    li = int(np.digitize(level, ir.get("levels", [6.0, 12.0])))
    mi = int(np.digitize(missed, ir.get("missed", [3, 6])))
    return float(ir["gone"][li][mi])


def _initial_absence(p: PlayerRow, n: int, rng: np.random.Generator, conf: dict) -> np.ndarray:
    """Games still to miss from today's status, given how many of his team's
    games in a row he has already missed."""
    pos = p.position if p.position in SKILL else "WR"
    already = int(getattr(p, "games_missed", 0) or 0)
    if p.status in LONG_TERM_OUT_STATUSES:
        ir = conf.get("ir")
        if ir:
            # Measured from every player on a reserve list at a checkpoint,
            # 2022-2025: most never play again that season, and the ones who
            # do take about four games (see DECISIONS.md, "Rest of season,
            # graded").
            level = float(p.ros_value if p.ros_value is not None else (p.projection or 0.0))
            gone = rng.random(n) < ir_gone(ir, level, already)
            pmf = np.asarray(ir["wait"], dtype=float)           # games he misses: 0, 1, 2, ...
            wait = rng.choice(np.arange(len(pmf)), size=n, p=pmf / pmf.sum())
            return np.where(gone, GONE, wait)
        return np.maximum(_durations(pos, n, rng, conf, already), 4)
    if p.is_out and not p.on_bye:
        return _durations(pos, n, rng, conf, already)
    q = p.play_probability if p.play_probability is not None else 1.0
    if q < 1.0:
        sits = rng.random(n) >= q
        return np.where(sits, _durations(pos, n, rng, conf, already), 0)
    return np.zeros(n, int)


def _by_level(table: dict | None, positions: list[str], base: np.ndarray) -> np.ndarray:
    """Per player: the table's value for his position at his level (0 where
    the table has none)."""
    if not table:
        return np.zeros(len(base))
    idx = np.digitize(base, table["levels"])
    return np.array([float(table[pos][i]) if pos in table else 0.0 for pos, i in zip(positions, idx)])


def _scale_by_level(table: dict | None, base: np.ndarray, positions: list[str] | None = None) -> np.ndarray:
    """Per player: the table's multiplier at his level -- his position's row
    where it has one, else the shared ``scale``."""
    if not table:
        return np.ones(len(base))
    idx = np.digitize(base, table["levels"])
    shared = table.get("scale")
    out = []
    for j, i in enumerate(idx):
        row = table.get(positions[j]) if positions else None
        row = row if row is not None else shared
        out.append(float(row[i]) if row is not None else 1.0)
    return np.asarray(out)


def _takeover(heir: PlayerRow, take: dict[str, float]) -> float:
    """The share of the lead's work this next man up is expected to take: his
    own, from how much of the position's other snaps he has had (depth.py),
    else the position's."""
    own = getattr(heir, "takeover_share", None)
    return float(own) if own is not None else float(take.get(heir.position, 0.0))


def _if_plays(p: PlayerRow) -> float | None:
    """This week's projection given he plays (the platform folds a
    questionable tag in as a mixture; the simulation flips that coin
    itself). None when he is out, on a bye, or has no projection."""
    q = float(p.play_probability if p.play_probability is not None else 1.0)
    if p.projection is None or p.is_out or q <= 0.0:
        return None
    return float(p.projection) / q if q < 1.0 else float(p.projection)


def _chance_out(p: PlayerRow) -> float:
    """Chance he misses the week in progress, as :func:`_initial_absence` draws it."""
    if p.status in LONG_TERM_OUT_STATUSES or (p.is_out and not p.on_bye):
        return 1.0
    return 1.0 - float(p.play_probability if p.play_probability is not None else 1.0)


def _this_week(players: list[PlayerRow], base: np.ndarray, heir_of: dict[int, int],
               take: dict[str, float]) -> np.ndarray:
    """Per player: this week's projection (if he plays) less what the
    simulation would expect of him this week from his season level -- the
    offset that makes the week in progress play at this week's numbers. For
    a next man up, the expected share of his absent lead's work is part of
    the simulation's side, so it is not counted twice."""
    out = np.zeros(len(players))
    for j, p in enumerate(players):
        week = _if_plays(p)
        if week is None:
            continue
        expect = float(base[j])
        if j in heir_of:
            lead = heir_of[j]
            expect += (_chance_out(players[lead]) * _takeover(p, take)
                       * max(float(base[lead]) - float(base[j]), 0.0))
        out[j] = week - expect
    return out


def simulate(
    players: list[PlayerRow],
    weeks: list[int],
    byes: dict[str, set[int]],
    n_sims: int = 2000,
    seed: int = 17,
    heirs: Iterable[tuple[str, str]] = (),
    current: int = 0,
    takeover: dict[str, float] | None = None,
    takeover_sd: dict[str, float] | None = None,
    kept: float | None = None,
) -> Futures:
    """Simulate every player's next ``weeks``.

    ``heirs`` pairs (lead id, next-man-up id) on the same NFL team: while the
    lead is out, the heir's level gains a share of the gap between them,
    drawn for each absence around ``takeover`` with spread ``takeover_sd``;
    when the lead is back, the heir keeps ``kept`` of what he took and the
    lead loses it. The heir's ``inherited_ros`` -- the absence already priced
    into his season value -- is taken back out, since it is played out here.
    ``current`` is the index of the week in progress, whose final scores
    are facts (earlier weeks are ones the platform has not counted yet). That
    week plays at each player's projection *for it* -- his matchup, the
    platform's and the betting market's numbers, a starter ahead of him out
    -- rather than his season level: a back starting this week because the
    lead is out is worth his start this week, not his season average.
    """
    conf = _params()
    rng = np.random.default_rng(seed)
    n_p, n_w = len(players), len(weeks)
    scores = np.zeros((n_sims, n_p, n_w))
    levels = np.zeros((n_sims, n_p, n_w))
    played = np.zeros((n_sims, n_p, n_w), bool)
    c = float(conf.get("persistent_error", 1.5))
    step_sd = conf.get("step", {})

    take = {**TAKEOVER, **(takeover or {})}
    take_sd = {**TAKEOVER_SD, **(takeover_sd or {})}
    keep = KEPT if kept is None else float(kept)
    heir_of = {players.index(next(q for q in players if q.player_id == h)): players.index(
        next(q for q in players if q.player_id == lead))
        for lead, h in heirs
        if any(q.player_id == lead for q in players) and any(q.player_id == h for q in players)}
    base = np.array([max(float(p.ros_value if p.ros_value is not None else (p.projection or 0.0))
                         - (float(p.inherited_ros or 0.0) if j in heir_of else 0.0), 0.0)
                     for j, p in enumerate(players)])
    # Two parts to a player's level: ``walk`` -- where his projection is and
    # will move to, visible to everyone as it happens -- and ``err``, how
    # wrong today's projection is about him, which nobody sees at first and
    # which games reveal: a manager's read after k games carries
    # k / (k + LEARN_GAMES) of what those games showed -- his error plus the
    # luck of those k games, which nobody can separate from it. Scores come
    # from the true level; lineups and claims can only use the visible one.
    # Two calibrations fitted on replayed seasons (scripts/fit_ros.py): an
    # offset by position and level, so a simulated player scores per game
    # played what players like him actually did (zero floors on levels and
    # scores otherwise lift a low player's average), and a scale on the
    # persistent error by level, so the bands hold what they claim (a
    # star's outlook is narrower than sqrt(level) says).
    shift = _by_level(conf.get("level_offset"), [p.position for p in players], base)
    cs = c * _scale_by_level(conf.get("error_scale"), base, [p.position for p in players])
    start = np.maximum(base + shift, 0.0)
    walk = np.repeat(start[None, :], n_sims, axis=0)
    z = rng.standard_normal((n_sims, n_p))
    if conf.get("error_shape") == "lognormal":
        # Multiplicative, mean-preserving: a level cannot go below zero, so a
        # normal error floored at zero would raise a low player's mean.
        rel = cs / np.sqrt(np.maximum(base, 0.25))
        sig = np.sqrt(np.log1p(rel ** 2))[None, :]
        err = start[None, :] * (np.exp(sig * z - 0.5 * sig ** 2) - 1.0)
    else:
        err = (cs * np.sqrt(base))[None, :] * z
    for j, p in enumerate(players):
        if p.position not in SKILL:
            err[:, j] = 0.0                             # D/ST and K are streamed, not held
    seen = np.zeros((n_sims, n_p))
    luck = np.zeros((n_sims, n_p))          # summed game-to-game noise in what was seen
    # The projection's own noise: what a manager sees moves by more than the
    # player does (fitted per position on how much of a real projection move
    # held up; scripts/fit_ros.py). Zero at the start: today's projection is
    # the reference.
    noise_sd = np.array([float((conf.get("projection_noise") or {}).get(p.position, 0.0))
                         if p.position in SKILL else 0.0 for p in players])
    # How much of the change in a player a manager's read catches: the
    # projection is slow and partial, so only this share of where he has
    # really moved (drift, and what games revealed) shows (fitted per
    # position, 1 where not).
    vis_share = np.array([float((conf.get("visibility") or {}).get(p.position, 1.0)) for p in players])[None, :]
    anchor = start.copy()
    shade = np.zeros((n_sims, n_p))
    out_left = np.stack([_initial_absence(p, n_sims, rng, conf) for p in players], axis=1) \
        if players else np.zeros((n_sims, 0), int)
    share = np.zeros((n_sims, n_p))          # the heir's share of this absence
    was_out = np.zeros((n_sims, n_p), bool)
    # Against the calibrated start, so the week in progress still plays at
    # this week's projection (validated on its own) and the offset is about
    # the weeks after it.
    week_delta = _this_week(players, start, heir_of, take)

    sd0 = np.array([float(p.outcome_sd if p.outcome_sd is not None else (p.projection_sd or 6.0)) for p in players])
    # Young players' outlooks move more: a rookie's projection drifts 23%
    # wider over a month than a veteran's at the same level (measured with
    # the drift, see outcome_model.json). Their expected rise is already in
    # today's projection, so only the spread is widened here.
    youth = outcome.load().get("drift", {}).get("youth_multiplier", {})
    walk_scale = np.array([float(youth.get("rookie", 1.0)) if p.experience == 0
                           else float(youth.get("second", 1.0)) if p.experience == 1 else 1.0
                           for p in players])
    for k, week in enumerate(weeks):
        on_bye = np.array([bool(p.team) and week in byes.get(p.team, set()) for p in players])
        # A lead back from an absence: the heir keeps part of his takeover,
        # and the lead has lost it.
        for h, lead in heir_of.items():
            back = was_out[:, lead] & (out_left[:, lead] == 0)
            if back.any():
                moved = np.where(back, keep * np.maximum(share[:, h], 0.0)
                                 * np.maximum(walk[:, lead] - walk[:, h], 0.0), 0.0)
                walk[:, h] += moved
                walk[:, lead] -= moved
        # New absences (not on bye, not already out).
        level = np.maximum(walk + err, 0.0)
        shown = err + luck / np.maximum(seen, 1.0)
        read = walk - anchor + shown * seen / (seen + LEARN_GAMES)
        visible = np.maximum(anchor + vis_share * read + shade, 0.0)
        for j, p in enumerate(players):
            if p.position not in SKILL or on_bye[j]:
                continue
            # The hazard was measured against projections, so it reads the
            # projection (where it has drifted to), not the hidden truth: a
            # star simulated to be worse than his number is not a backup.
            risk_at = np.maximum(walk[:, j], 0.0) if conf.get("hazard_on") == "projection" else level[:, j]
            fresh = (out_left[:, j] == 0) & (rng.random(n_sims) < _hazard(p.position, risk_at, conf))
            if fresh.any():
                out_left[fresh, j] = _durations(p.position, int(fresh.sum()), rng, conf)
        playing = (out_left == 0) & ~on_bye[None, :]
        out_now = out_left > 0
        eff, eff_vis = level.copy(), visible.copy()
        for h, lead in heir_of.items():
            pos = players[h].position
            start = out_now[:, lead] & ~was_out[:, lead]
            if start.any():
                draw = rng.normal(_takeover(players[h], take), take_sd.get(pos, 0.0), size=int(start.sum()))
                share[start, h] = np.clip(draw, *SHARE_RANGE)
            lead_out = out_now[:, lead]
            eff[:, h] = np.where(lead_out, level[:, h] + share[:, h] * np.maximum(level[:, lead] - level[:, h], 0.0),
                                 level[:, h])
            eff_vis[:, h] = np.where(lead_out, visible[:, h] + share[:, h]
                                     * np.maximum(visible[:, lead] - visible[:, h], 0.0), visible[:, h])
        was_out = out_now
        if k == current:
            # The week in progress at this week's projection, not the season's.
            eff = np.maximum(eff + week_delta[None, :], 0.0)
            eff_vis = np.maximum(eff_vis + week_delta[None, :], 0.0)
        levels[:, :, k] = np.where(playing, eff_vis, 0.0)
        played[:, :, k] = playing
        finals = [j for j, p in enumerate(players) if k == current and p.actual_points is not None]
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
        # What each game showed beyond his level is read as part of it.
        luck += np.where(playing, scores[:, :, k] - eff, 0.0)
        # A game passes: absences tick down (a bye does not use one up), and
        # the true level drifts.
        out_left = np.where(on_bye[None, :], out_left, np.maximum(out_left - 1, 0))
        seen += playing
        for j, p in enumerate(players):
            if p.position in SKILL:
                walk[:, j] = walk[:, j] + float(step_sd.get(p.position, 0.95)) * walk_scale[j] \
                    * rng.standard_normal(n_sims)
        if noise_sd.any():
            shade = PROJECTION_NOISE_PERSIST * shade + (noise_sd * walk_scale)[None, :] \
                * rng.standard_normal((n_sims, n_p))
    return Futures(player_ids=[p.player_id for p in players], weeks=list(weeks), scores=scores, levels=levels,
                   played=played)
