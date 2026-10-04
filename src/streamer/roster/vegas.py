"""Attach sportsbook-implied points to a roster, and start the lineup they imply.

Two projections disagreeing is information. Ours leans on trailing production
and opportunity; the market's leans on everything a book knows, including
beat-reporter noise about a snap count that has not shown up in a box score
yet. Where they disagree the market is usually, though not always, right --
so the panel shows both and says which players the disagreement is about.

Props do not cover everybody. Kickers and defences are rarely posted, deep
bench players never are, and a player the books have not priced simply has no
Vegas number -- which is shown as a blank rather than a zero, because "no
market" is not "no points".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import Config, get_config
from ..league.model import LeagueSnapshot, PlayerRow
from .players import normalize_name

log = logging.getLogger(__name__)


@dataclass
class VegasReport:
    """What the props pull produced, for the page and the CLI."""

    matched: int = 0
    #: Players who could start and should therefore have had a price.
    eligible: int = 0
    matched_startable: int = 0
    unmatched: list[str] = field(default_factory=list)
    events: int = 0
    credits_remaining: int | None = None
    warnings: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def is_usable(self) -> bool:
        return self.matched > 0


def teams_on(snapshot: LeagueSnapshot, include_opponent: bool = False) -> dict[str, int]:
    """NFL teams worth paying for props on, and how many players each is worth.

    Only players who could start for *you*: props are billed per market per
    event, the opponent's implied points are never shown, and P(win) scores
    both sides with our own projections. Players already ruled out for the
    week cost credits and change nothing. The counts let the fetch spend its
    event budget on the games holding the most of your lineup.
    """
    # Kickers and defences have no player props, so they buy nothing.
    rows = [p for p in snapshot.my_team.roster
            if not p.in_ir_slot and not p.on_bye and not p.is_out
            and p.position not in ("K", "DST")]
    if include_opponent and snapshot.opponent is not None:
        rows += list(snapshot.opponent.roster)
    out: dict[str, int] = {}
    for p in rows:
        if p.team:
            out[p.team] = out.get(p.team, 0) + 1
    return out


def attach(
    snapshot: LeagueSnapshot,
    cfg: Config | None = None,
    allow_network: bool = True,
    payload=None,
    prefetched=None,
) -> VegasReport:
    """Fetch props for the matchup's teams and attach them to the players.

    ``prefetched`` is a :class:`~streamer.data.props.PropsResult` pulled once
    for several leagues (see :func:`teams_for_all`); ``payload`` injects a raw
    payload instead of calling out, for tests.
    """
    from ..data.props import fetch_props, props_to_points

    cfg = cfg or get_config()
    report = VegasReport()
    if payload is not None:
        points = props_to_points(payload, cfg)
    elif not allow_network:
        report.warnings.append("player props skipped: --offline")
        return report
    else:
        result = prefetched if prefetched is not None else fetch_props(cfg, teams=teams_on(snapshot))
        report.events = result.events
        report.credits_remaining = result.credits_remaining
        report.warnings.extend(result.warnings)
        report.description = result.describe()
        snapshot._vegas_credits = result.credits_remaining
        if not result.is_usable:
            return report
        from ..data.props import consensus, implied_means, to_fantasy_points

        points = to_fantasy_points(consensus(implied_means(result.frame, cfg), cfg), cfg)

    if points.empty:
        return report
    by_name = {normalize_name(r.player): r for r in points.itertuples()}
    everyone = list(snapshot.all_players())
    for p in everyone:
        row = by_name.get(normalize_name(p.name))
        if row is None:
            continue
        p.vegas_points = float(row.vegas_points)
        p.vegas_stats = dict(row.stats)
        p.vegas_books = int(row.books)
        report.matched += 1
    startable = [p for p in snapshot.my_team.roster
                 if p.position not in ("K", "DST") and not p.on_bye
                 and not p.is_out and not p.in_ir_slot]
    report.eligible = len(startable)
    report.matched_startable = sum(1 for p in startable if p.vegas_points is not None)
    report.unmatched = sorted(p.name for p in startable if p.vegas_points is None)
    snapshot._vegas_report = report
    return report


def teams_for_all(snapshots: list[LeagueSnapshot]) -> dict[str, int]:
    """Combined team weights across leagues, for one shared props pull.

    Pulling per league bought the same games twice. Summing the weights means
    the event budget goes to the games holding the most of your players
    across every league you play in.
    """
    out: dict[str, int] = {}
    for snap in snapshots:
        for team, n in teams_on(snap).items():
            out[team] = out.get(team, 0) + n
    return out


def vegas_lineup(snapshot: LeagueSnapshot, cfg: Config | None = None):
    """The lineup the market's numbers would start, and what it projects.

    Falls back to our own projection for players with no posted prop, because
    a lineup that benches every kicker is not a lineup. Respects kickoffs the
    way the optimiser does: a starter whose game has begun keeps his slot, a
    benched one cannot come in, and a final score counts as scored.
    """
    from .lineup import LineupResult, best_by_key

    cfg = cfg or get_config()
    roster = [p for p in snapshot.my_team.roster if not p.in_ir_slot]
    slots = snapshot.starting_slots

    def key(p: PlayerRow) -> float:
        if p.actual_points is not None:
            return float(p.actual_points)
        if p.is_out:
            return 0.0
        if p.vegas_points is not None:
            return float(p.vegas_points)
        return float(p.pre_market_projection if p.pre_market_projection is not None
                     else (p.projection or 0.0))

    fixed: dict[str, list[PlayerRow]] = {}
    for p in roster:
        if p.locked and p.starting and p.slot in slots:
            fixed.setdefault(p.slot, []).append(p)
    free_slots = {s: n - len(fixed.get(s, [])) for s, n in slots.items()}
    free_slots = {s: n for s, n in free_slots.items() if n > 0}
    free = [p for p in roster if not p.locked]
    rest = best_by_key(free, free_slots, key) if free_slots else None
    starters = {s: list(fixed.get(s, [])) + (list(rest.starters.get(s, [])) if rest else []) for s in slots}
    total = sum(key(p) for ps in starters.values() for p in ps)
    return LineupResult(starters=starters, expected=round(total, 2), sd=0.0, win_probability=0.0)


# ---------------------------------------------------------------------------
# Blending the market into the projection
# ---------------------------------------------------------------------------
#: A consensus from fewer books is noisier: the share of the full weight a
#: player gets by how many books priced him.
BOOK_DEPTH = {1: 0.6, 2: 0.85}


@dataclass
class WeekWeights:
    """How far this week's projection moves toward each second forecast, and
    the graded games behind each weight (below the minimum, the prior)."""

    market: float
    market_games: int = 0
    consensus: float = 0.0
    consensus_games: int = 0


#: A consensus projection this low says he will not play; the platform's tag
#: carries that, so it is not blended as a points forecast.
NO_PROJECTION = 0.5


def _shrunk(n: int, fitted: float, prior: float, k: float, need: int) -> float:
    return prior if n < need else (n * fitted + k * prior) / (n + k)


def fit_week_weights(cfg: Config, history=None) -> WeekWeights:
    """The weights this week's projection gives the betting market and the
    FantasyPros consensus projection, fitted together.

    Joins the projection log (ours before either blend, and the market's
    number) and the consensus log (their projection) to actual PPR points,
    and finds the pair minimising squared error of
    ``ours + w_m (market - ours) + w_c (consensus - ours)`` -- a source a
    player has no number from adds nothing for him -- then shrinks each
    toward its prior by its own graded games. Below each minimum the prior
    stands: there is no free archive of props or expert projections, so the
    evidence accumulates from this season's own logs.
    """
    import numpy as np
    import pandas as pd

    from ..data import fantasypros as fp

    pconf = cfg.odds.get("props") or {}
    fconf = fp.conf(cfg)
    pm, km, nm = (float(pconf.get("blend_weight_prior", 0.4)), float(pconf.get("blend_prior_games", 300)),
                  int(pconf.get("blend_min_games", 150)))
    pc, kc, nc = (float(fconf.get("week_weight_prior", 0.3)), float(fconf.get("week_prior_games", 300)),
                  int(fconf.get("week_min_games", 150)))
    prior = WeekWeights(market=pm, consensus=pc)
    frames = []
    for d in sorted({cfg.results_dir.parent / name for name in ("espn", "yahoo")} | {cfg.results_dir}):
        path = d / "skill_log.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return prior
    log_ = pd.concat(frames).dropna(subset=["projection", "nfl_id"])
    log_ = log_.drop_duplicates(["season", "week", "nfl_id"])
    if log_.empty:
        return prior
    if "vegas_points" not in log_:
        log_["vegas_points"] = np.nan
    cpath = fp.cache_dir(cfg) / "log.parquet"
    if cpath.exists():
        c = pd.read_parquet(cpath)
        if "fp_projection" in c:
            c = c.dropna(subset=["fp_projection"]).drop_duplicates(["season", "week", "nfl_id"])
            log_ = log_.merge(c[["season", "week", "nfl_id", "fp_projection"]],
                              on=["season", "week", "nfl_id"], how="left")
    if "fp_projection" not in log_:
        log_["fp_projection"] = np.nan
    if history is None:
        from .projections import load_history

        history = load_history(cfg)
    actual = history[["player_id", "season", "week", "fantasy_points_ppr"]].rename(columns={"player_id": "nfl_id"})
    j = log_.merge(actual, on=["nfl_id", "season", "week"], how="inner")
    has_m = j["vegas_points"].notna().to_numpy()
    has_c = (j["fp_projection"].notna() & (j["fp_projection"].astype(float) > NO_PROJECTION)).to_numpy()
    keep = has_m | has_c
    j, has_m, has_c = j[keep], has_m[keep], has_c[keep]
    n_m, n_c = int(has_m.sum()), int(has_c.sum())
    if n_m < nm and n_c < nc:
        return WeekWeights(pm, n_m, pc, n_c)
    ours, y = j["projection"].to_numpy(float), j["fantasy_points_ppr"].to_numpy(float)
    dm = np.where(has_m, j["vegas_points"].to_numpy(float) - ours, 0.0)
    dc = np.where(has_c, j["fp_projection"].to_numpy(float) - ours, 0.0)
    grid = np.linspace(0.0, 1.0, 21)
    best, fit = np.inf, (pm, pc)
    for wm in grid:
        for wc in grid:
            if wm + wc > 1.0 + 1e-9:
                continue
            err = float(np.mean((ours + wm * dm + wc * dc - y) ** 2))
            if err < best:
                best, fit = err, (float(wm), float(wc))
    return WeekWeights(market=_shrunk(n_m, fit[0], pm, km, nm), market_games=n_m,
                       consensus=_shrunk(n_c, fit[1], pc, kc, nc), consensus_games=n_c)


def fit_market_weight(cfg: Config, history=None) -> tuple[float, int]:
    """The market's blend weight, and how many finished games it rests on
    (:func:`fit_week_weights`)."""
    w = fit_week_weights(cfg, history)
    return w.market, w.market_games


def blend_market(snapshot: LeagueSnapshot, cfg: Config, weight: float | None = None,
                 consensus_weight: float = 0.0) -> float:
    """Blend the second forecasts into every player's projection for this
    week: sportsbook-implied points by ``weight`` (less where few books
    priced him) and the FantasyPros consensus projection by
    ``consensus_weight``, each where the player has one.

    Both assume the player suits up, so the blend is done on the
    if-he-plays number and the injury discount re-applied. Players already
    locked, out, or whom the platform has ruled out are left alone. Returns
    the market weight used.
    """
    if weight is None:
        weight, _n = fit_market_weight(cfg)
    for p in snapshot.all_players():
        p.pre_market_projection = p.projection
        p.market_weight = 0.0
        p.consensus_weight = 0.0
        if (p.projection is None or p.locked or p.is_out
                or p.play_probability <= 0 or "sits" in (p.projection_source or "")):
            continue
        wm = weight * BOOK_DEPTH.get(int(p.vegas_books or 0), 1.0) if p.vegas_points is not None else 0.0
        wc = consensus_weight if p.fp_projection is not None and p.fp_projection > NO_PROJECTION else 0.0
        if wm <= 0 and wc <= 0:
            continue
        q = p.play_probability if p.play_probability else 1.0
        if_plays = float(p.projection) / q
        blended = if_plays
        if wm > 0:
            blended += wm * (float(p.vegas_points) - if_plays)
        if wc > 0:
            blended += wc * (float(p.fp_projection) - if_plays)
        p.projection = round(q * blended, 2)
        p.market_weight = round(wm, 3)
        p.consensus_weight = round(wc, 3)
    return weight


def disagreements(
    snapshot: LeagueSnapshot, n: int = 3, among: set[str] | None = None
) -> list[tuple[PlayerRow, float]]:
    """Players our projection and the market disagree about most, biggest first.

    ``among`` restricts to decision-relevant players -- the ones either lineup
    would start. A three-point disagreement about the last man on the bench is
    not a thing anyone needs to read.
    """
    gaps = []
    for p in snapshot.my_team.roster:
        if p.vegas_points is None or p.projection is None or p.in_ir_slot or p.is_out:
            continue
        if among is not None and p.player_id not in among:
            continue
        ours = p.pre_market_projection if p.pre_market_projection is not None else p.projection
        gaps.append((p, float(p.vegas_points) - float(ours)))
    gaps.sort(key=lambda g: -abs(g[1]))
    return gaps[:n]


def log_projections(snapshot: LeagueSnapshot, cfg: Config) -> int:
    """Record ours, the platform's and the market's number for every skill
    player the market priced (plus everyone rostered), before kickoff.

    There is no free history of player props, so whether the market beats
    our model -- and by how much to blend them -- can only be measured going
    forward. ``results/<profile>/skill_log.parquet`` accumulates one row per
    (season, week, player), overwritten until his game kicks off; actual
    points join on later from nflverse by ``nfl_id``.
    """
    import pandas as pd

    rostered = {p.player_id for t in snapshot.teams for p in t.roster}
    rows = []
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    for p in snapshot.all_players():
        if p.position not in ("QB", "RB", "WR", "TE") or p.locked or p.projection is None:
            continue
        if p.vegas_points is None and p.player_id not in rostered:
            continue
        rows.append({
            "season": snapshot.season, "week": snapshot.week, "player_id": p.player_id,
            "nfl_id": p.nfl_id, "name": p.name, "position": p.position, "team": p.team,
            "status": p.status,
            "projection": p.pre_market_projection if p.pre_market_projection is not None else p.projection,
            "blended_projection": p.projection, "market_weight": p.market_weight,
            "consensus_weight": p.consensus_weight,
            "model_projection": p.model_projection,
            "platform_projection": p.platform_projection, "vegas_points": p.vegas_points,
            "vegas_books": p.vegas_books, "logged_at": stamp,
        })
    if not rows:
        return 0
    path = cfg.results_dir / "skill_log.parquet"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_parquet(path)
        key = ["season", "week", "player_id"]
        # Keep rows for players now locked (their pre-kickoff numbers stand).
        old = old.merge(new[key], on=key, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return len(rows)
