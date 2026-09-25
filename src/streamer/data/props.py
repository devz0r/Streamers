"""Sportsbook player props -> implied fantasy points.

A book's player props are the closest thing to a market consensus projection
that exists: real money, priced by people whose living depends on being right.
Turning them into fantasy points takes three steps, and the middle one is the
part most implementations get wrong.

**De-vig.** A two-way price carries the book's margin, so the raw implied
probabilities sum to more than 1. Normalising them back to 1 recovers the
market's actual view. Pinnacle's margin is the thinnest, which is why it is
weighted highest.

**Line to mean.** A prop line is roughly the market's *median*, not its mean,
and fantasy scoring wants the mean. Given P(X > L) = p from the de-vigged
prices and a normal approximation, the mean is ``L + sd * z(p)`` -- the line
plus however far the market leans over or under it. That needs the spread of
the stat at that level, which is fitted from five seasons of nflverse weekly
data (see ``odds.props.spread`` in config.yaml and DECISIONS.md). Counting
markets (touchdowns) use a Poisson instead, where the whole distribution
follows from one parameter.

**Score it.** Multiply through by the league's own scoring sheet. Anytime-TD
prices give an expected touchdown count of ``-ln(1 - p)``: the Poisson rate
whose chance of at least one is p, which is not the same as p once a second
touchdown is possible.

Everything here is pure except :func:`fetch_props`, so the maths is tested
offline against fixtures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config, get_config

log = logging.getLogger(__name__)

#: Market key -> (stat it prices, how to model it).
#: ``normal`` markets are yardage/count lines with an over/under; ``poisson``
#: markets are touchdown counts; ``anytime`` is a one-sided yes/no price.
MARKETS: dict[str, tuple[str, str]] = {
    "player_pass_yds": ("passing_yards", "normal"),
    "player_pass_tds": ("passing_tds", "poisson"),
    "player_pass_interceptions": ("passing_interceptions", "poisson"),
    "player_rush_yds": ("rushing_yards", "normal"),
    "player_receptions": ("receptions", "normal"),
    "player_reception_yds": ("receiving_yards", "normal"),
    "player_anytime_td": ("anytime_td", "anytime"),
}

#: Stat -> the scoring key in a profile's ``offense_scoring`` block.
SCORING_KEY: dict[str, str] = {
    "passing_yards": "pass_yards",
    "passing_tds": "pass_td",
    "passing_interceptions": "interception",
    "rushing_yards": "rush_yards",
    "receptions": "reception",
    "receiving_yards": "reception_yards",
}


@dataclass
class PropsResult:
    """Prop lines for one week, plus provenance for the page."""

    frame: pd.DataFrame
    fetched_at: datetime
    events: int = 0
    credits_remaining: int | None = None
    warnings: list[str] = field(default_factory=list)
    #: Of ``events``, how many were paid for now versus reused from the cache.
    requested: int = 0
    reused: int = 0

    @property
    def is_usable(self) -> bool:
        return not self.frame.empty

    def describe(self) -> str:
        if self.frame.empty:
            return "no player props"
        books = sorted(self.frame["bookmaker"].unique())
        paid = f", {self.requested} bought and {self.reused} reused" if self.reused else ""
        return (f"{self.frame['player'].nunique()} players across {self.events} games "
                f"from {', '.join(books)}{paid}")


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
def american_to_prob(price: float | int | None) -> float | None:
    """Implied probability of an American price, vig included."""
    if price is None or (isinstance(price, float) and np.isnan(price)):
        return None
    price = float(price)
    if price == 0:
        return None
    if price > 0:
        return 100.0 / (price + 100.0)
    return -price / (-price + 100.0)


def devig(over_price, under_price) -> float | None:
    """De-vigged P(over) from a two-way price. None if either side is missing."""
    p_over = american_to_prob(over_price)
    p_under = american_to_prob(under_price)
    if p_over is None or p_under is None:
        return None
    total = p_over + p_under
    if total <= 0:
        return None
    return float(np.clip(p_over / total, 1e-6, 1 - 1e-6))


def devig_one_sided(price, margin: float = 0.04) -> float | None:
    """De-vig a yes-only price (anytime TD) by removing a typical margin.

    Without the other side of the market the overround cannot be measured, so
    a nominal margin is taken out. It is a small correction on a small number.
    """
    p = american_to_prob(price)
    if p is None:
        return None
    return float(np.clip(p / (1.0 + margin), 1e-6, 1 - 1e-6))


def spread_for(stat: str, line: float, cfg: Config) -> float:
    """sd of ``stat`` at ``line``, from the fitted intercept/slope."""
    conf = (cfg.odds.get("props") or {}).get("spread") or {}
    row = conf.get(stat)
    if not row:
        return max(float(line) * 0.5, 1.0)
    sd = float(row["intercept"]) + float(row["slope"]) * float(line)
    return max(sd, float(row.get("floor", 1.0)))


def mean_from_normal_line(line: float, p_over: float, sd: float) -> float:
    """Mean of a normal whose P(X > line) is ``p_over``."""
    return float(line + sd * stats.norm.ppf(p_over))


def mean_from_poisson_line(line: float, p_over: float) -> float:
    """Rate of a Poisson whose P(X > line) is ``p_over``.

    Prop lines sit on a half (over 1.5 touchdowns), so "over the line" means
    at least ``ceil(line)`` events. Solved by bisection: the survival function
    rises monotonically in the rate, so there is exactly one answer.
    """
    k = int(np.ceil(line))            # smallest count that beats the line
    p_over = float(np.clip(p_over, 1e-6, 1 - 1e-6))
    lo, hi = 1e-6, 25.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if stats.poisson.sf(k - 1, mid) < p_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def mean_from_anytime(p: float) -> float:
    """Expected touchdowns behind a P(scores at least one) of ``p``.

    ``-ln(1 - p)`` is the Poisson rate whose chance of at least one is p. Using
    p itself would quietly discard every multi-touchdown game.
    """
    return float(-np.log(1.0 - float(np.clip(p, 1e-6, 1 - 1e-6))))


# ---------------------------------------------------------------------------
# Payload -> tidy frame
# ---------------------------------------------------------------------------
def parse_props_payload(payload: dict | list, cfg: Config | None = None) -> pd.DataFrame:
    """Flatten The Odds API event-odds JSON into one row per player/market/book.

    Separate from the HTTP call so it is tested against a fixture.
    """
    cfg = cfg or get_config()
    events = payload if isinstance(payload, list) else [payload]
    rows: list[dict] = []
    for event in events or []:
        event_id = event.get("id")
        for book in event.get("bookmakers", []) or []:
            book_key = book.get("key")
            for market in book.get("markets", []) or []:
                key = market.get("key")
                if key not in MARKETS:
                    continue
                stat, kind = MARKETS[key]
                # Outcomes pair up per player: Over/Under, or a single Yes.
                by_player: dict[str, dict] = {}
                for outcome in market.get("outcomes", []) or []:
                    player = (outcome.get("description") or outcome.get("participant") or "").strip()
                    if not player:
                        continue
                    side = str(outcome.get("name") or "").strip().lower()
                    slot = by_player.setdefault(player, {"line": outcome.get("point")})
                    slot[side] = outcome.get("price")
                    if outcome.get("point") is not None:
                        slot["line"] = outcome.get("point")
                for player, slot in by_player.items():
                    rows.append({
                        "event_id": event_id, "bookmaker": book_key, "market": key,
                        "stat": stat, "kind": kind, "player": player,
                        "line": slot.get("line"),
                        "over_price": slot.get("over"),
                        "under_price": slot.get("under"),
                        "yes_price": slot.get("yes"),
                    })
    return pd.DataFrame(rows, columns=["event_id", "bookmaker", "market", "stat", "kind",
                                       "player", "line", "over_price", "under_price", "yes_price"])


def implied_means(frame: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Per player/stat/book: the market's implied mean for that stat."""
    cfg = cfg or get_config()
    if frame.empty:
        return pd.DataFrame(columns=["player", "stat", "bookmaker", "mean"])
    out = []
    for row in frame.itertuples():
        mean = None
        if row.kind == "anytime":
            p = devig_one_sided(row.yes_price)
            if p is not None:
                mean = mean_from_anytime(p)
        elif row.line is not None and not pd.isna(row.line):
            p_over = devig(row.over_price, row.under_price)
            if p_over is not None:
                if row.kind == "poisson":
                    mean = mean_from_poisson_line(float(row.line), p_over)
                else:
                    sd = spread_for(row.stat, float(row.line), cfg)
                    mean = mean_from_normal_line(float(row.line), p_over, sd)
        if mean is not None and np.isfinite(mean):
            out.append({"player": row.player, "stat": row.stat,
                        "bookmaker": row.bookmaker, "mean": float(max(mean, 0.0))})
    return pd.DataFrame(out, columns=["player", "stat", "bookmaker", "mean"])


def consensus(means: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Weighted average across books, per player and stat."""
    cfg = cfg or get_config()
    if means.empty:
        return pd.DataFrame(columns=["player", "stat", "mean", "books"])
    weights = (cfg.odds.get("props") or {}).get("bookmaker_weights") or {}
    means = means.copy()
    means["w"] = means["bookmaker"].map(lambda b: float(weights.get(b, 1.0)))
    means["wm"] = means["mean"] * means["w"]
    g = means.groupby(["player", "stat"], as_index=False).agg(
        wm=("wm", "sum"), w=("w", "sum"), books=("bookmaker", "nunique"))
    g["mean"] = g["wm"] / g["w"]
    return g[["player", "stat", "mean", "books"]]


def to_fantasy_points(cons: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Score each player's implied stat line with the profile's own sheet.

    Anytime-TD is a rushing/receiving touchdown; a quarterback's passing
    touchdowns come from their own market, so the two do not overlap.
    """
    scoring = cfg._profile.get("offense_scoring") or {}
    if cons.empty:
        return pd.DataFrame(columns=["player", "vegas_points", "stats", "books"])
    rows = []
    for player, grp in cons.groupby("player"):
        points = 0.0
        detail: dict[str, float] = {}
        for r in grp.itertuples():
            detail[r.stat] = round(float(r.mean), 2)
            if r.stat == "anytime_td":
                # Non-passing touchdowns: rushing and receiving score the same
                # in both configured leagues, so one rate covers both.
                points += float(r.mean) * float(scoring.get("rush_td", 6.0))
            else:
                points += float(r.mean) * float(scoring.get(SCORING_KEY[r.stat], 0.0))
        rows.append({"player": player, "vegas_points": round(points, 2),
                     "stats": detail, "books": int(grp["books"].max())})
    return pd.DataFrame(rows).sort_values("vegas_points", ascending=False).reset_index(drop=True)


def props_to_points(payload, cfg: Config) -> pd.DataFrame:
    """The whole pipeline: raw payload -> per-player implied fantasy points."""
    return to_fantasy_points(consensus(implied_means(parse_props_payload(payload, cfg), cfg), cfg), cfg)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Per-event cache
#
# Props are billed per market per event, and both leagues -- and any re-run
# shortly after a scheduled one -- want largely the same games. Each event's
# payload is kept in memory for the life of the process and on disk under
# data/raw/props for ``odds.props.cache_minutes``; the weekly job restores
# data/raw between runs, so a manual re-run an hour later pays nothing.
# ---------------------------------------------------------------------------
_MEMORY: dict[str, tuple[datetime, dict]] = {}


def _cache_dir(cfg: Config):
    path = cfg.raw_dir / "props"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached_event(cfg: Config, event_id: str, max_age_minutes: float) -> dict | None:
    """A payload for ``event_id`` younger than ``max_age_minutes``, if one exists."""
    import json

    now = datetime.now(UTC)
    hit = _MEMORY.get(event_id)
    if hit and (now - hit[0]).total_seconds() <= max_age_minutes * 60:
        return hit[1]
    if max_age_minutes <= 0:
        return None
    path = _cache_dir(cfg) / f"{event_id}.json"
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        at = datetime.fromisoformat(blob["fetched_at"])
    except (OSError, ValueError, KeyError):
        return None
    if (now - at).total_seconds() > max_age_minutes * 60:
        return None
    _MEMORY[event_id] = (at, blob["payload"])
    return blob["payload"]


def store_event(cfg: Config, event_id: str, payload: dict) -> None:
    import json

    now = datetime.now(UTC)
    _MEMORY[event_id] = (now, payload)
    try:
        (_cache_dir(cfg) / f"{event_id}.json").write_text(
            json.dumps({"fetched_at": now.isoformat(), "payload": payload}), encoding="utf-8")
    except OSError as exc:  # a cache that cannot be written is just a cache miss
        log.debug("could not cache props for %s: %s", event_id, exc)


def fetch_props(
    cfg: Config | None = None,
    teams: set[str] | dict[str, int] | None = None,
    timeout: float = 20.0,
) -> PropsResult:
    """Pull player props for this week's events.

    Props are billed per market per event, so ``teams`` narrows the pull to
    games involving players that matter and ``odds.props.max_events`` caps it.
    A game already fetched recently is served from the cache for free.
    """
    import requests

    from ..teams import normalize_team
    from .odds import odds_api_key

    cfg = cfg or get_config()
    conf = cfg.odds
    pconf = conf.get("props") or {}
    now = datetime.now(UTC)
    if not pconf.get("enabled", True):
        return PropsResult(pd.DataFrame(), now, warnings=["player props are disabled in config"])
    key = odds_api_key(cfg)
    if not key:
        return PropsResult(pd.DataFrame(), now, warnings=["no ODDS_API_KEY configured"])

    base = f"{conf['api_base']}/sports/{conf['sport']}"
    try:
        resp = requests.get(f"{base}/events", params={"apiKey": key}, timeout=timeout)
        resp.raise_for_status()
        events = resp.json() or []
    except Exception as exc:  # noqa: BLE001
        return PropsResult(pd.DataFrame(), now, warnings=[f"event list unavailable: {exc}"])

    # Books post a game's props a day or two before kickoff, so asking on a
    # Tuesday about Sunday buys nothing -- the request either comes back empty
    # or errors. Events beyond the window are skipped, which is why a midweek
    # run costs almost nothing and the Sunday-morning run pays for real
    # coverage exactly when the lineup is being set.
    window = float(pconf.get("window_hours") or 0)
    if window > 0:
        cutoff = now + timedelta(hours=window)
        kept = []
        for e in events:
            ts = e.get("commence_time")
            if not ts:
                kept.append(e)
                continue
            try:
                start = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except ValueError:
                kept.append(e)
                continue
            if start <= cutoff:
                kept.append(e)
        events = kept

    # Spend the cap where it buys the most: rank events by how many of the
    # players we care about are in them, not by kickoff time. Taking the
    # earliest games instead would buy Thursday night and miss the lineup.
    if teams:
        weights = teams if isinstance(teams, dict) else {t: 1 for t in teams}
        scored = []
        for e in events:
            home = normalize_team(e.get("home_team"))
            away = normalize_team(e.get("away_team"))
            n = weights.get(home, 0) + weights.get(away, 0)
            if n:
                scored.append((n, e))
        scored.sort(key=lambda ne: -ne[0])
        events = [e for _n, e in scored]
    cap = int(pconf.get("max_events") or 0)
    if cap > 0:
        events = events[:cap]

    markets = ",".join(pconf.get("markets") or [])
    books = ",".join(pconf.get("bookmakers") or [])
    payloads, warnings, remaining = [], [], None
    max_age = float(pconf.get("cache_minutes", 180))
    requested = reused = 0
    for event in events:
        hit = cached_event(cfg, str(event["id"]), max_age)
        if hit is not None:
            payloads.append(hit)
            reused += 1
            continue
        params = {"apiKey": key, "regions": conf["regions"], "markets": markets,
                  "oddsFormat": conf["odds_format"]}
        if books:
            params["bookmakers"] = books
        try:
            r = requests.get(f"{base}/events/{event['id']}/odds", params=params, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            payloads.append(payload)
            store_event(cfg, str(event["id"]), payload)
            requested += 1
            left = r.headers.get("x-requests-remaining")
            if left is not None:
                remaining = int(float(left))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"props for {event.get('away_team')} @ {event.get('home_team')}: {exc}")
    frame = parse_props_payload(payloads, cfg)
    if frame.empty and not warnings:
        if window > 0 and not events:
            warnings.append(
                f"no games kick off within {window:.0f}h, so no props are posted yet")
        else:
            warnings.append("the books returned no player props for these games")
    return PropsResult(frame, now, events=len(payloads), credits_remaining=remaining,
                       warnings=warnings, requested=requested, reused=reused)
