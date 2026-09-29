"""FantasyPros consensus, attached to league players and logged for grading.

The consensus is the market that matters most for trades -- managers read
these rankings in their apps -- and the benchmark our season values have to
beat. Its numbers are FantasyPros' data: they live on the player objects for
one publish and in a runner-only log (``data/raw/fantasypros/log.parquet``),
never in a committed file or on the public page, where only analysis derived
from them appears, credited to FantasyPros. Showing their ranks or
projections themselves would be redistribution, which their API licence
reserves for a Commercial agreement.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import Config
from ..data import fantasypros as fp
from ..league.model import LeagueSnapshot, PlayerRow
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
    return matched


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
                     "our_ros": _ros(p), "logged_at": stamp})
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
