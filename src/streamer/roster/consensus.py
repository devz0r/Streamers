"""FantasyPros consensus, attached to league players and logged for grading.

The consensus is the market that matters most for trades -- managers read
these rankings in their apps -- the benchmark our season values have to
beat, and a second forecast of them: every season value moves part of the
way toward the consensus's view, as far as the log shows that view earns
(:func:`season_weight`). Its numbers are FantasyPros' data: they live on the player objects for
one publish and in a runner-only log (``data/raw/fantasypros/log.parquet``),
never in a committed file or on the public page, where only analysis derived
from them appears, credited to FantasyPros. Showing their ranks or
projections themselves would be redistribution, which their API licence
reserves for a Commercial agreement.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..data import fantasypros as fp
from ..league.model import OUT_STATUSES, LeagueSnapshot, PlayerRow
from .players import normalize_name
from .waivers import _ros

log = logging.getLogger(__name__)

SKILL = ("QB", "RB", "WR", "TE")


def _key(name, team, position) -> tuple:
    from ..teams import normalize_team

    return normalize_name(name), normalize_team(team) if team else None, str(position or "").upper()


def _index(frame: pd.DataFrame) -> tuple[dict, dict]:
    """Rows by (name, team, position), and by (name, position) when unique."""
    exact, loose, seen = {}, {}, {}
    for r in frame.itertuples():
        k = _key(r.name, r.team, r.position)
        exact.setdefault(k, r)
        seen[(k[0], k[2])] = seen.get((k[0], k[2]), 0) + 1
        loose[(k[0], k[2])] = r
    loose = {k: v for k, v in loose.items() if seen[k] == 1}
    return exact, loose


def _find(p: PlayerRow, idx: tuple[dict, dict]):
    exact, loose = idx
    k = _key(p.name, p.team, p.position)
    return exact.get(k) or loose.get((k[0], k[2]))


def attach(snapshot: LeagueSnapshot, cfg: Config) -> int:
    """Put this week's consensus on every skill player it covers. Returns how
    many players matched (0 with no key, no data, or no match)."""
    if not fp.api_key() and not any(fp.cache_dir(cfg).glob("*.json")):
        return 0
    season, week = int(snapshot.season), int(snapshot.week)
    try:
        ros = fp.rankings(cfg, season, week, "ros")
        weekly = fp.rankings(cfg, season, week, "weekly")
        proj = fp.projections(cfg, season, week)
    except Exception as exc:  # noqa: BLE001 - the consensus is a bonus input
        log.warning("FantasyPros pull failed: %s", type(exc).__name__)
        return 0
    idx = {name: _index(f) for name, f in (("ros", ros), ("weekly", weekly), ("proj", proj)) if not f.empty}
    matched = 0
    for p in snapshot.all_players():
        if p.position not in SKILL:
            continue
        hit = False
        if "ros" in idx and (r := _find(p, idx["ros"])) is not None and r.pos_rank:
            p.ecr_ros_pos_rank, hit = int(r.pos_rank), True
        if "weekly" in idx and (r := _find(p, idx["weekly"])) is not None and r.pos_rank:
            p.ecr_week_pos_rank, hit = int(r.pos_rank), True
        if "proj" in idx and (r := _find(p, idx["proj"])) is not None and r.points is not None:
            p.fp_projection, hit = float(r.points), True
        matched += hit
    log.info("FantasyPros consensus matched %d players", matched)
    if "ros" in idx:
        log.info("out for the season by the consensus: %d players", season_over(snapshot))
    return matched


def season_over(snapshot: LeagueSnapshot) -> int:
    """Injured players the rest-of-season consensus no longer ranks: out for
    the season, whatever the platform's tag says.

    ESPN and Yahoo tag a torn ACL and a four-week hamstring alike (IR), and
    the model gave both his full value back after a few games -- De'Von
    Achane, done for the season, still carried 13.7 a game. Expert rankings
    drop a player who will not play again this season and keep one who will
    be back. Only players on a long-term tag whose value would put him in
    the top half of the players the consensus ranks at his position are
    judged, and only when the consensus is seen ranking some such injured
    players at all (so a feed that leaves out every injured player cannot
    zero them all). Returns how many were marked."""
    judged = []
    for pos in SKILL:
        ranked = [p for p in snapshot.all_players() if p.position == pos and p.ecr_ros_pos_rank]
        if len(ranked) < 8:
            continue
        bar = float(np.median([float(p.ros_value or 0.0) for p in ranked]))
        judged += [p for p in snapshot.all_players() if p.position == pos and p.is_long_term_out
                   and p.team and float(p.ros_value or 0.0) >= bar]
    kept_ranked = [p for p in judged if p.ecr_ros_pos_rank]
    if not judged or len(kept_ranked) < max(2, 0.25 * len(judged)):
        return 0
    marked = 0
    for p in judged:
        if p.ecr_ros_pos_rank:
            continue
        p.signals.append(f"out for the season by the FantasyPros rest-of-season consensus, which does not "
                         f"rank him: no value this season, whatever the {p.status.lower()} tag says "
                         f"(was {float(p.ros_value or 0.0):.1f} a game)")
        p.ros_value, p.inherited_ros = 0.0, 0.0
        marked += 1
    return marked


def our_ranks(snapshot: LeagueSnapshot) -> dict[str, int]:
    """Our position rank for every player the consensus ranks, among the
    players it ranks -- so the two ranks are on the same list."""
    out: dict[str, int] = {}
    for pos in SKILL:
        ranked = [p for p in snapshot.all_players() if p.position == pos and p.ecr_ros_pos_rank]
        ecr_order = sorted(ranked, key=lambda p: p.ecr_ros_pos_rank)
        ours = sorted(ranked, key=lambda p: -_ros(p))
        # Our rank, placed on the consensus's scale: the consensus rank of
        # whoever holds that spot in its own order.
        for i, p in enumerate(ours):
            out[p.player_id] = ecr_order[i].ecr_ros_pos_rank
    return out


def note(p: PlayerRow, ours: dict[str, int], cfg: Config) -> str:
    """A second opinion worth reading: where the consensus is well off our view."""
    if not p.ecr_ros_pos_rank or p.player_id not in ours:
        return ""
    mine, theirs = ours[p.player_id], p.ecr_ros_pos_rank
    gap = theirs - mine                         # positive: consensus ranks him lower
    if abs(gap) < max(5, 0.3 * min(mine, theirs)):
        return ""
    side = "lower" if gap > 0 else "higher"
    # Analysis only: their ranks themselves are never shown. Publishing
    # FantasyPros data is redistribution, which needs a Commercial API
    # agreement; analysis based on it is allowed with credit to them.
    return f"FantasyPros consensus is much {side} on him than we are"


def market_values(snapshot: LeagueSnapshot, seen: dict) -> dict[str, float]:
    """What the consensus rank says a player is worth, in the market's own
    points: the perceived value of whoever sits at that position rank."""
    out = {}
    for pos in SKILL:
        ranked = [p for p in snapshot.all_players() if p.position == pos and p.ecr_ros_pos_rank
                  and p.player_id in seen]
        ladder = sorted((seen[p.player_id].value for p in ranked), reverse=True)
        for p in sorted(ranked, key=lambda p: p.ecr_ros_pos_rank):
            i = min(sum(1 for q in ranked if q.ecr_ros_pos_rank < p.ecr_ros_pos_rank), len(ladder) - 1)
            out[p.player_id] = ladder[i]
    return out


# ---------------------------------------------------------------------------
# The consensus as a second forecast of season value
# ---------------------------------------------------------------------------
def _clean(p: PlayerRow) -> bool:
    """A player the consensus prices on the same footing as our season value.
    An injured player's rest-of-season rank also counts the games he will
    miss, which our per-game value leaves to the season simulation."""
    return (p.position in SKILL and bool(p.ecr_ros_pos_rank) and bool(p.team) and p.ros_value is not None
            and p.ros_value > 0 and p.status not in OUT_STATUSES and not p.in_ir_slot)


def implied(frame: pd.DataFrame, by: list[str]) -> pd.Series:
    """The consensus's order on our scale: each player gets our season value
    of whoever holds his consensus rank among the same players (same
    position, and the same ``by`` groups). Levels stay ours; only the order is
    theirs, so the blend moves players past each other, not the whole list."""
    out = pd.Series(np.nan, index=frame.index)
    for _key, g in frame.groupby([*by, "position"]):
        ladder = np.sort(g["our_ros"].to_numpy(float))[::-1]
        out.loc[g.sort_values("ecr_ros_pos_rank", kind="stable").index] = ladder
    return out


def season_weight(cfg: Config, history: pd.DataFrame | None = None) -> tuple[float, int]:
    """How much of a season value the consensus gets, and the graded games
    that rests on.

    From the consensus log, each week's season values -- ours, and the
    consensus's order on our scale -- against what each player scored that
    week; the weight minimising squared error, shrunk toward the prior by
    ``season_prior_games``. Below ``season_min_games`` the prior stands.
    There is no free archive of expert rankings to backtest, so the evidence
    comes from this season's own logs, the same way the betting-prop weight
    does.
    """
    conf = fp.conf(cfg)
    prior = float(conf.get("season_weight_prior", 0.3))
    k = float(conf.get("season_prior_games", 300))
    need = int(conf.get("season_min_games", 150))
    path = fp.cache_dir(cfg) / "log.parquet"
    if not path.exists():
        return prior, 0
    c = pd.read_parquet(path).dropna(subset=["ecr_ros_pos_rank", "our_ros"])
    c = c[c.our_ros > 0]
    if "clean" in c:
        c = c[c["clean"].fillna(True).astype(bool)]
    if c.empty:
        return prior, 0
    c = c.assign(implied=implied(c, ["season", "week"]))
    if history is None:
        from .projections import load_history

        history = load_history(cfg)
    actual = history[["player_id", "season", "week", "fantasy_points_ppr"]].rename(columns={"player_id": "nfl_id"})
    j = c.merge(actual, on=["nfl_id", "season", "week"], how="inner")
    n = len(j)
    if n < need:
        return prior, n
    ours, theirs, y = (j["our_ros"].to_numpy(float), j["implied"].to_numpy(float),
                       j["fantasy_points_ppr"].to_numpy(float))
    grid = np.linspace(0.0, 1.0, 21)
    errs = [float(np.mean(((1 - w) * ours + w * theirs - y) ** 2)) for w in grid]
    w_fit = float(grid[int(np.argmin(errs))])
    return (n * w_fit + k * prior) / (n + k), n


def blend_season(snapshot: LeagueSnapshot, weight: float) -> int:
    """Move every season value part of the way toward the consensus's view of
    the player, ``weight`` of the gap. Runs after :func:`log_week`, so the log
    keeps our own number to grade. Returns how many moved 0.3+ points."""
    clean = [p for p in snapshot.all_players() if _clean(p)]
    if not clean or weight <= 0:
        return 0
    frame = pd.DataFrame({"position": [p.position for p in clean],
                          "ecr_ros_pos_rank": [p.ecr_ros_pos_rank for p in clean],
                          "our_ros": [float(p.ros_value) for p in clean]})
    moved = 0
    for p, target in zip(clean, implied(frame, [])):
        before = float(p.ros_value)
        after = before + weight * (float(target) - before)
        p.ros_value = round(after, 2)
        if abs(after - before) >= 0.3:
            # Analysis only: the direction of their view, never their rank.
            p.signals.append(f"season value {'raised' if after > before else 'lowered'} from {before:.1f} to "
                             f"{after:.1f}: the FantasyPros expert consensus rates him "
                             f"{'higher' if after > before else 'lower'} than our model does")
            moved += 1
    return moved


def log_week(snapshot: LeagueSnapshot, cfg: Config) -> int:
    """Record the consensus beside our numbers, before kickoff, for grading.
    Runner-only: this file holds FantasyPros' numbers and is never committed."""
    rows = []
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    for p in snapshot.all_players():
        if p.position not in SKILL or p.locked or not p.nfl_id:
            continue
        if p.fp_projection is None and not p.ecr_week_pos_rank and not p.ecr_ros_pos_rank:
            continue
        rows.append({"season": int(snapshot.season), "week": int(snapshot.week), "nfl_id": p.nfl_id,
                     "position": p.position, "fp_projection": p.fp_projection,
                     "ecr_week_pos_rank": p.ecr_week_pos_rank, "ecr_ros_pos_rank": p.ecr_ros_pos_rank,
                     "our_ros": _ros(p), "clean": _clean(p), "logged_at": stamp})
    if not rows:
        return 0
    path = fp.cache_dir(cfg) / "log.parquet"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_parquet(path)
        key = ["season", "week", "nfl_id"]
        old = old.merge(new[key], on=key, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(path, index=False)
    return len(rows)
