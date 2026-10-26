"""Replayed leagues: weekly P(win) and the season odds, graded on seasons
already played.

Each part of the season engine is validated on its own (projections, the
simulated futures, the bracket). This grades the whole: leagues are drafted
and played out on 2022-2025 with real players and real scores, and every
number the page shows a manager -- this week's P(win), P(playoffs), P(title),
expected wins -- is computed as the production code computes it, from what
was known that week, and set against what happened.

A replayed league:

* **Teams and rules**: ``N_TEAMS`` teams, QB/RB/RB/WR/WR/TE/FLEX and a
  ``BENCH`` bench, full PPR, weeks 1-14 the regular season, six teams in
  the playoffs over weeks 15-17 (the top two seeds on a bye), as in both of
  our leagues. Skill positions only: the replay has no D/ST or K projections.
* **Draft**: a snake draft at week 1 on each manager's read of the season
  values (``ros_value`` times his own lognormal noise, ``DRAFT_NOISE``),
  with position caps and the starting spots filled.
* **Each week**: projections from :func:`ros_backtest.project` on the games
  before it; in waiver order (worst record first), each team fills any
  starting spot it cannot field with the best free agent who plays, then
  makes at most one upgrade claim when a free agent's season value beats
  its weakest player's by ``UPGRADE``. Lineups start the best projected
  players who play; a started player scores what he really scored.
* **Graded**: every head-to-head game's P(win) (:func:`lineup.sample_points`,
  the lineup tab's correlated draws) against the result; at ``CHECKPOINTS``
  the :class:`SeasonModel` odds for every team against who made the
  playoffs and who won, beside a baseline that knows only the standings
  (every remaining game a coin flip).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..league.model import LONG_TERM_OUT_STATUSES, OUT_STATUSES, LeagueSnapshot, PlayerRow, TeamRow
from . import lineup
from .season import SeasonModel, bracket_order

SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
#: Streamed every week from the free units, in waiver order, when the replay
#: is given defence and kicker projections (:func:`unit_rows`).
UNIT_SLOTS = ("DST", "K")
FILL_ORDER = ("QB", "TE", "RB", "WR", "FLEX")
ELIGIBLE = {"QB": ("QB",), "RB": ("RB",), "WR": ("WR",), "TE": ("TE",), "FLEX": ("RB", "WR", "TE")}
BENCH = 6
N_TEAMS = 10
REG_WEEKS = 14
PLAYOFF_TEAMS = 6
LAST_WEEK = 17
#: Most a team holds at a position (draft and claims).
CAPS = {"QB": 2, "RB": 6, "WR": 6, "TE": 2}
#: Fewest it keeps: the starting spots, flex aside.
NEEDS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
DRAFT_NOISE = 0.15
#: Points a game of season value a free agent must add over a team's
#: weakest player to be claimed.
UPGRADE = 1.5
CHECKPOINTS = (1, 4, 7, 10, 13)


@dataclass
class Replay:
    """One league played out: who held whom each week, the standings, the
    graded games and the season odds at each checkpoint."""
    season: int
    seed: int
    games: list[dict] = field(default_factory=list)
    odds: list[dict] = field(default_factory=list)
    finals: dict = field(default_factory=dict)


def round_robin(n: int, weeks: int) -> list[tuple[int, int, int]]:
    """(week, a, b) for ``weeks`` weeks of a circle-method round robin."""
    idx = list(range(n))
    out = []
    for w in range(1, weeks + 1):
        for i in range(n // 2):
            out.append((w, idx[i], idx[n - 1 - i]))
        idx = [idx[0], idx[-1], *idx[1:-1]]
    return out


def _value(p: PlayerRow) -> float:
    return float(p.ros_value if p.ros_value is not None else (p.projection or 0.0))


def _plays(p: PlayerRow | None) -> bool:
    return (p is not None and not p.on_bye and p.status not in OUT_STATUSES
            and p.status not in LONG_TERM_OUT_STATUSES and float(p.projection or 0.0) > 0.0)


def draft(players: list[PlayerRow], n_teams: int, rng: np.random.Generator,
          bench: int = BENCH, noise: float = DRAFT_NOISE) -> list[list[str]]:
    """Snake draft on each manager's noisy read of season value."""
    pool = {p.player_id: p for p in players if p.position in NEEDS and _value(p) > 0}
    pos_of = {pid: p.position for pid, p in pool.items()}
    ids = list(pool)
    tilt = np.exp(noise * rng.standard_normal((n_teams, len(ids))) - 0.5 * noise ** 2)
    seen = {pid: tilt[:, j] * _value(pool[pid]) for j, pid in enumerate(ids)}
    size = sum(SLOTS.values()) + bench
    rosters: list[list[str]] = [[] for _ in range(n_teams)]
    for r in range(size):
        order = range(n_teams) if r % 2 == 0 else range(n_teams - 1, -1, -1)
        for t in order:
            have = {pos: sum(pos_of[q] == pos for q in rosters[t]) for pos in NEEDS}
            left = size - len(rosters[t])
            short = {pos for pos, k in NEEDS.items() if have[pos] < k}
            need_now = sum(NEEDS[pos] - have[pos] for pos in short) >= left
            ok = [pid for pid in pool if have[pool[pid].position] < CAPS[pool[pid].position]
                  and (not need_now or pool[pid].position in short)]
            if not ok:
                continue
            pick = max(ok, key=lambda pid: seen[pid][t])
            rosters[t].append(pick)
            del pool[pick]
    return rosters


def best_lineup(rows: list[PlayerRow]) -> list[PlayerRow]:
    """The best projected lineup from the players who play this week (and
    the team's streamed defence and kicker, when it has them)."""
    left = sorted((p for p in rows if _plays(p)), key=lambda p: -float(p.projection or 0.0))
    out = []
    for slot in FILL_ORDER:
        for _ in range(SLOTS[slot]):
            pick = next((p for p in left if p.position in ELIGIBLE[slot]), None)
            if pick is not None:
                out.append(pick)
                left.remove(pick)
    for pos in UNIT_SLOTS:
        pick = next((p for p in left if p.position == pos), None)
        if pick is not None:
            out.append(pick)
    return out


def unit_rows(frame: pd.DataFrame, season: int) -> tuple[dict[int, list[PlayerRow]], dict[tuple[int, str], float]]:
    """Defences and kickers from the walk-forward backtest
    (:func:`streamer.backtest.walk_forward`: each week projected from earlier
    games only) as streamable rows, week -> units, and their real points."""
    from ..teams import normalize_team

    out: dict[int, list[PlayerRow]] = {}
    pts: dict[tuple[int, str], float] = {}
    for r in frame[frame["season"] == season].itertuples():
        team = normalize_team(r.team)
        pid = f"{r.position}:{team}"
        sd = max((float(r.ceiling) - float(r.floor)) / 2.07, 1.0)
        out.setdefault(int(r.week), []).append(PlayerRow(
            player_id=pid, name=pid, position=r.position, team=team, status="", slot="BE",
            eligible_slots=[r.position], projection=float(r.expected_points), projection_sd=sd, outcome_sd=sd,
            ros_value=float(r.baseline_points), nfl_opponent=normalize_team(r.opponent), role=r.position))
        pts[(int(r.week), pid)] = float(r.fantasy_points)
    return out, pts


def _holes(rows: list[PlayerRow]) -> list[str]:
    """Starting spots this roster cannot field this week."""
    left = sorted((p for p in rows if _plays(p)), key=lambda p: -float(p.projection or 0.0))
    holes = []
    for slot in FILL_ORDER:
        for _ in range(SLOTS[slot]):
            pick = next((p for p in left if p.position in ELIGIBLE[slot]), None)
            if pick is None:
                holes.append(slot)
            else:
                left.remove(pick)
    return holes


def _droppable(roster: list[PlayerRow], adding: PlayerRow, protect=()) -> PlayerRow | None:
    """The weakest player the team can let go for ``adding`` and still have
    its starting spots and stay under the caps; never one in ``protect``."""
    counts = {pos: sum(p.position == pos for p in roster) for pos in NEEDS}
    counts[adding.position] += 1
    for p in sorted(roster, key=_value):
        if p in protect:
            continue
        after = dict(counts)
        after[p.position] -= 1
        if all(after[pos] >= k for pos, k in NEEDS.items()) and all(after[pos] <= CAPS[pos] for pos in CAPS):
            return p
    return None


def _stub(pid: str, last: PlayerRow | None) -> PlayerRow:
    """A held player with no projection this week (released, no NFL team)."""
    return PlayerRow(player_id=pid, name=last.name if last else pid, position=last.position if last else "WR",
                     team=None, status="", slot="BE", eligible_slots=[], projection=0.0, projection_sd=1.0,
                     outcome_sd=1.0, ros_value=0.0)


def standings_key(wins: np.ndarray, points: np.ndarray) -> np.ndarray:
    return wins * 1e6 + points


def realized_bracket(wins: np.ndarray, points: np.ndarray, playoff_scores: np.ndarray,
                     playoff_teams: int = PLAYOFF_TEAMS) -> tuple[set[int], int]:
    """Who made the playoffs and who won, from real results: seeds by record
    then points, a fixed bracket with byes for the top seeds, a tie to the
    higher seed -- the production bracket (:meth:`SeasonModel._playoffs`)
    without reseeding. ``playoff_scores``: (teams, rounds)."""
    order = list(np.argsort(-standings_key(wins, points), kind="stable"))
    seeds = order[:playoff_teams]
    rounds = max(int(np.ceil(np.log2(max(playoff_teams, 2)))), 1)
    alive = [(s, seeds[s - 1]) if s <= playoff_teams else None for s in bracket_order(2 ** rounds)]
    for r in range(rounds):
        nxt = []
        for i in range(0, len(alive), 2):
            a, b = alive[i], alive[i + 1]
            if a is None or b is None:
                nxt.append(a if b is None else b)
                continue
            sa, ta = a
            sb, tb = b
            ra, rb = playoff_scores[ta, r], playoff_scores[tb, r]
            nxt.append(a if (ra > rb or (ra == rb and sa < sb)) else b)
        alive = nxt
    return set(int(t) for t in seeds), int(alive[0][1])


def snapshot(season: int, week: int, rosters: list[list[PlayerRow]], free: list[PlayerRow],
             wins: np.ndarray, losses: np.ndarray, points: np.ndarray, waiver: list[int],
             schedule: list[tuple[int, int, int]], slots: dict[str, int] | None = None) -> LeagueSnapshot:
    teams = [TeamRow(team_id=str(i), name=f"T{i}", wins=int(wins[i]), losses=int(losses[i]),
                     points_for=float(points[i]), roster=list(r), is_mine=(i == 0),
                     waiver_rank=waiver.index(i) + 1) for i, r in enumerate(rosters)]
    return LeagueSnapshot(
        platform="espn", profile="replay", league_id="replay", league_name="replay", season=season, week=week,
        slots={**(slots or SLOTS), "BE": BENCH}, bench_size=BENCH, teams=teams, free_agents=list(free), matchup=None,
        synced_at="", rules={"regular_season_weeks": REG_WEEKS, "playoff_teams": PLAYOFF_TEAMS,
                             "playoff_round_weeks": 1, "median_game": False,
                             "schedule": [[w, str(a), str(b)] for w, a, b in schedule]})


def coin_flip_odds(model: SeasonModel, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """P(playoffs) and P(title) knowing only the standings: every team's
    remaining weeks drawn alike, so each game left is a coin flip."""
    shape = model.base_scores.shape
    scores = rng.normal(100.0, 25.0, shape)
    wins, pts, _ = model._regular_season(scores)
    made, _byes, champ = model._playoffs(scores, wins, pts)
    title = np.bincount(np.asarray(champ).ravel(), minlength=shape[0]) / model.n
    return made.mean(axis=1), title


def play(season: int, seed: int, weekly: dict[int, list[PlayerRow]], points: dict[tuple[int, str], float],
         byes: dict[str, set[int]], cfg: Config, n_sims: int = 2000, win_sims: int = 4000,
         checkpoints=CHECKPOINTS, log=None, upgrade: float | None = UPGRADE,
         retentions: tuple[float, ...] = (), units: dict[int, list[PlayerRow]] | None = None) -> Replay:
    """Draft and play one league through week 17, grading as it goes.
    ``weekly``: week -> projected players (games before it only);
    ``points``: (week, player id) -> real PPR points; ``retentions``: other
    ``strength_fade`` values whose odds are recorded beside the model's;
    ``units``: week -> defences and kickers (:func:`unit_rows`), streamed
    each week in waiver order, best projected first (their points in
    ``points``)."""
    rng = np.random.default_rng(seed)
    out = Replay(season, seed)
    rows = {w: {p.player_id: p for p in ps} for w, ps in weekly.items()}
    for w, ps in weekly.items():
        for p in ps:
            p.on_bye = bool(p.team) and w in byes.get(p.team, set())
    last_seen: dict[str, PlayerRow] = {}
    held = [list(r) for r in draft(weekly[1], N_TEAMS, rng)]
    schedule = round_robin(N_TEAMS, REG_WEEKS)
    wins, losses, pts = np.zeros(N_TEAMS), np.zeros(N_TEAMS), np.zeros(N_TEAMS)
    waiver = list(range(N_TEAMS))[::-1]
    playoff_scores = np.zeros((N_TEAMS, LAST_WEEK - REG_WEEKS))
    weekly_scores: dict[int, np.ndarray] = {}
    for week in range(1, LAST_WEEK + 1):
        here = rows[week]
        for pid, p in here.items():
            last_seen[pid] = p

        def row(pid: str, here=here) -> PlayerRow:
            return here.get(pid) or _stub(pid, last_seen.get(pid))

        # Waivers, worst record first.
        if week > 1:
            waiver = sorted(range(N_TEAMS), key=lambda t: (wins[t], pts[t]))
        owned = {pid for r in held for pid in r}
        free = [p for pid, p in here.items() if pid not in owned and p.position in NEEDS]
        for t in waiver:
            roster = [row(pid) for pid in held[t]]
            for slot in _holes(roster):
                cands = sorted((p for p in free if _plays(p) and p.position in ELIGIBLE[slot]),
                               key=lambda p: -float(p.projection or 0.0))
                for fa in cands:
                    drop = _droppable(roster, fa, protect=best_lineup(roster))
                    if drop is None:
                        continue
                    held[t].remove(drop.player_id)
                    held[t].append(fa.player_id)
                    free.remove(fa)
                    roster = [row(pid) for pid in held[t]]
                    break
            healthy = [p for p in free if p.status not in LONG_TERM_OUT_STATUSES]
            if healthy and upgrade is not None:
                fa = max(healthy, key=_value)
                drop = _droppable(roster, fa, protect=best_lineup(roster))
                if drop is not None and _value(fa) >= _value(drop) + upgrade:
                    held[t].remove(drop.player_id)
                    held[t].append(fa.player_id)
                    free.remove(fa)
        rosters = [[row(pid) for pid in r] for r in held]
        owned = {pid for r in held for pid in r}
        free = [p for pid, p in here.items() if pid not in owned and p.position in NEEDS]
        if units:
            # Stream a defence and a kicker, worst record first.
            pool = sorted(units.get(week, []), key=lambda p: -float(p.projection or 0.0))
            for t in waiver:
                for pos in UNIT_SLOTS:
                    pick = next((p for p in pool if p.position == pos), None)
                    if pick is not None:
                        rosters[t].append(pick)
                        pool.remove(pick)
            free = free + pool
        # The season odds, before this week's games.
        if week in checkpoints and week <= REG_WEEKS:
            snap = snapshot(season, week, rosters, free, wins, losses, pts, waiver, schedule,
                            slots={**SLOTS, **{u: 1 for u in UNIT_SLOTS}} if units else SLOTS)
            model = SeasonModel(snap, cfg, n_sims=n_sims, seed=seed * 100 + week)
            odds = model.odds()
            faded = {r: model.odds(retention=r) for r in retentions}
            base_po, base_title = coin_flip_odds(model, rng)
            # P(win) this week as the season model has it (its own draws).
            k0 = 0
            sm_win = {}
            for w, a, b in schedule:
                if w == week:
                    sa, sb = model.base_scores[a, :, k0], model.base_scores[b, :, k0]
                    sm_win[(a, b)] = float((sa > sb).mean() + 0.5 * (sa == sb).mean())
            reg = [k for k, w in enumerate(model.weeks) if w <= REG_WEEKS]
            rest = model.base_scores[:, :, reg].mean(axis=1).sum(axis=1)
            for i in range(N_TEAMS):
                out.odds.append({"season": season, "seed": seed, "week": week, "team": i,
                                 "exp_points": float(rest[i]), "roster": list(held[i]),
                                 "wins": float(wins[i]), "p_playoffs": float(odds.p_playoffs[i]),
                                 "p_title": float(odds.p_title[i]), "p_bye": float(odds.p_bye[i]),
                                 "exp_wins": float(odds.exp_wins[i]), "base_playoffs": float(base_po[i]),
                                 "base_title": float(base_title[i]),
                                 **{f"p_playoffs_r{r:g}": float(o.p_playoffs[i]) for r, o in faded.items()},
                                 **{f"p_title_r{r:g}": float(o.p_title[i]) for r, o in faded.items()}})
        else:
            sm_win = {}
        # The week's games.
        lineups = [best_lineup(r) for r in rosters]
        scored = np.array([sum(points.get((week, p.player_id), 0.0) for p in lu) for lu in lineups])
        weekly_scores[week] = scored
        if week <= REG_WEEKS:
            for w, a, b in schedule:
                if w != week:
                    continue
                both = lineups[a] + lineups[b]
                draws = lineup.sample_points(both, win_sims, np.random.default_rng(seed * 1000 + week * 10 + a))
                ta, tb = draws[:, :len(lineups[a])].sum(axis=1), draws[:, len(lineups[a]):].sum(axis=1)
                p_win = lineup.calibrated((ta > tb).mean() + 0.5 * (ta == tb).mean())
                margin = ta - tb
                real = scored[a] - scored[b]
                out.games.append({"season": season, "seed": seed, "week": week, "a": a, "b": b,
                                  "p_win": p_win, "p_win_season": sm_win.get((a, b), np.nan),
                                  "proj_a": float(ta.mean()), "proj_b": float(tb.mean()),
                                  "margin_sd": float(margin.std()), "pit": float((margin < real).mean()),
                                  "score_a": float(scored[a]), "score_b": float(scored[b]),
                                  "won": 1.0 if real > 0 else 0.5 if real == 0 else 0.0})
                wins[a] += real > 0
                wins[b] += real < 0
                losses[a] += real < 0
                losses[b] += real > 0
                wins[[a, b]] += 0.5 * (real == 0)
            pts += scored
        else:
            playoff_scores[:, week - REG_WEEKS - 1] = scored
        if log:
            log(f"{season} league {seed} week {week}")
    made, champ = realized_bracket(wins, pts, playoff_scores)
    out.finals = {"wins": wins.tolist(), "made": sorted(made), "champion": champ}
    for r in out.odds:
        r["made"] = float(r["team"] in made)
        r["won_title"] = float(r["team"] == champ)
        r["final_wins"] = float(wins[r["team"]])
        r["rest_points"] = float(sum(weekly_scores[w][r["team"]] for w in range(r["week"], REG_WEEKS + 1)))
    return out


def points_by_week(history: pd.DataFrame, season: int) -> dict[tuple[int, str], float]:
    h = history[history["season"] == season]
    return {(int(w), str(p)): float(v) for (w, p), v in
            h.groupby(["week", "player_id"])["fantasy_points_ppr"].sum().items()}
