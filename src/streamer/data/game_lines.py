"""Game lines -- moneyline, spread, total -- across books, and the ones that look wrong.

The same idea as the prop board (:mod:`streamer.data.betting`), on the main
markets: each book's two prices, margin taken out, are its read of the game;
a book whose price pays more than the *other* books' read says it is worth is
a price to take. No projection of ours is involved -- we have no edge on game
outcomes -- so this is about prices, not opinions.

Spreads and totals at different numbers are priced on the measured chance of
each final margin and combined score (nflverse games since 1999): the margin
piles up on 3 and 7, so half a point around 3 is worth far more than half a
point around 5. Given a line ``m`` the market expects, the chance a game ends
on each margin ``k`` is read from games whose closing line was near ``m``.

SportsGameOdds bills per game, so its game lines come with the same pull as
the props. DECISIONS.md, "Game lines".
"""

from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)

KEY = ["event_id", "market"]
COLUMNS = ["event_id", "bookmaker", "market", "home", "away", "starts", "line", "price_a", "price_b"]
#: price_a / price_b: home / away (moneyline and spread), over / under (total).
#: ``line`` is the home team's spread (-3: home gives three) or the total.
STALE_MINUTES = 90
ARB_MINUTES = 30        # a sure bet needs both prices read this close together
#: Moneyline sides the books' margin-free read puts under this are not flagged. Scaling a book's two
#: prices to sum to one overstates a long shot: 2006-2025, home or away sides read at 15% or less
#: won 8.9% of 271 games against the 12.0% read (a third too high, far more than any edge here).
#: Other ways of removing the margin (power, Shin) were closer there but not better in every
#: stretch of seasons, so the read stands and the long shots are left out.
LONGSHOT = 0.15
MIN_OTHERS = 2
MIN_EV = 0.02
WAY_OFF = 0.10
GRID = np.round(np.arange(-24.0, 24.01, 0.25), 2)
TOTAL_GRID = np.round(np.arange(30.0, 60.01, 0.25), 2)


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------
def _price(v):
    try:
        return float(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return None


def _team(event: dict, side: str) -> str:
    from ..teams import normalize_team

    t = ((event.get("teams") or {}).get(side) or {})
    names = t.get("names") or {}
    for v in (names.get("long"), names.get("medium"), names.get("short"), t.get("teamID")):
        if v:
            got = normalize_team(str(v).replace("_NFL", "").replace("_", " "))
            if got:
                return got
    return ""


def parse(events: list[dict]) -> pd.DataFrame:
    """SportsGameOdds events -> one row per game, market and book. Odds keys
    read ``points-{home|away|all}-game-{ml|sp|ou}-{home|away|over|under}``;
    a spread's number is in ``spread`` (or ``line``), a total's in ``overUnder``."""
    rows = []
    for e in events or []:
        eid = str(e.get("eventID") or "")
        home, away = _team(e, "home"), _team(e, "away")
        starts = str((e.get("status") or {}).get("startsAt") or "")
        quotes: dict[tuple, dict] = {}
        for odd_id, odd in (e.get("odds") or {}).items():
            parts = str(odd_id).split("-")
            if len(parts) != 5:
                continue
            stat, entity, period, bet, side = parts
            if stat != "points" or period != "game" or bet not in ("ml", "sp", "ou"):
                continue
            # The game total is the "all" entity's over/under; "home"/"away" over/unders are each
            # team's own points (team totals), a different market. Moneyline and spread: the team's own.
            if (bet == "ou" and entity != "all") or (bet != "ou" and entity != side):
                continue
            for book, b in (odd.get("byBookmaker") or {}).items():
                if not isinstance(b, dict) or b.get("available") is False:
                    continue
                q = quotes.setdefault((bet, book), {})
                q[side] = _price(b.get("odds"))
                if bet == "sp":
                    sp = _price(b.get("spread", b.get("line")))
                    if sp is not None:
                        q[f"line_{side}"] = sp
                if bet == "ou":
                    ou = _price(b.get("overUnder", b.get("line")))
                    if ou is not None:
                        q["line"] = ou
        for (bet, book), q in quotes.items():
            if bet == "ml":
                a, b_, line = q.get("home"), q.get("away"), None
            elif bet == "sp":
                a, b_ = q.get("home"), q.get("away")
                line = q.get("line_home", -q["line_away"] if q.get("line_away") is not None else None)
            else:
                a, b_, line = q.get("over"), q.get("under"), q.get("line")
            if a is None or b_ is None or (bet != "ml" and line is None):
                continue
            rows.append({"event_id": eid, "bookmaker": book, "market": {"ml": "moneyline", "sp": "spread", "ou": "total"}[bet],
                         "home": home, "away": away, "starts": starts, "line": line, "price_a": a, "price_b": b_})
    return pd.DataFrame(rows, columns=COLUMNS)


def path(cfg: Config):
    return cfg.path("results") / "game_lines.parquet"


def record(frame: pd.DataFrame, cfg: Config, now: str | None = None) -> int:
    """Each book's first and latest quote per game and market (as the prop log)."""
    if frame is None or frame.empty:
        return 0
    from .props_log import quote_times

    new = frame.copy()
    new["logged_at"] = quote_times(new, now)       # when the feed was read, not when this run happened
    new = new.drop(columns=[c for c in ("quoted_at",) if c in new])
    p = path(cfg)
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=new.columns)
    k = KEY + ["bookmaker"]
    both = pd.concat([old, new], ignore_index=True).sort_values("logged_at")
    out = pd.concat([both.groupby(k, sort=False).head(1), both.groupby(k, sort=False).tail(1)]).drop_duplicates(
        k + ["logged_at"]).sort_values(k + ["logged_at"]).reset_index(drop=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False, compression="zstd")
    return len(out)


# ---------------------------------------------------------------------------
# The chance of each final margin and total
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=2)
def _games(raw_dir: str) -> pd.DataFrame:
    s = pd.read_parquet(f"{raw_dir}/schedules.parquet", columns=["season", "spread_line", "total_line", "result", "total"])
    return s[s["result"].notna() & s["spread_line"].notna()]


@functools.lru_cache(maxsize=4096)
def _dist(raw_dir: str, kind: str, m: float) -> tuple[np.ndarray, np.ndarray]:
    """Outcomes and their chances for games the market had at ``m`` (the home
    team's expected margin, or the total), from games whose closing line was
    near it -- widened until there are 300 of them."""
    g = _games(raw_dir)
    col, out = ("spread_line", "result") if kind == "spread" else ("total_line", "total")
    if kind == "total":
        g = g[g["season"] >= 2015]          # scoring has moved; totals from the last decade
    for w in (1.0, 1.5, 2.0, 3.0, 5.0):
        near = g[(g[col] - m).abs() <= w]
        if len(near) >= 300:
            break
    # Shift each game by how far its line sat from m, rounded: keeps the spikes on 3 and 7.
    x = (near[out] + np.round(m - near[col])).astype(int)
    vc = x.value_counts(normalize=True).sort_index()
    return vc.index.to_numpy(), vc.to_numpy()


def chances(cfg: Config, kind: str, m: float, threshold: float) -> tuple[float, float, float]:
    """(above, push, below) for outcome vs ``threshold`` when the market expects ``m``."""
    k, p = _dist(str(cfg.raw_dir), kind, float(np.round(m * 4) / 4))
    return float(p[k > threshold].sum()), float(p[k == threshold].sum()), float(p[k < threshold].sum())


def _devig(a: float, b: float) -> float:
    ia = -a / (-a + 100) if a < 0 else 100 / (a + 100)
    ib = -b / (-b + 100) if b < 0 else 100 / (b + 100)
    return ia / (ia + ib)


def _decimal(p: float) -> float:
    return 1.0 + (100.0 / -p if p < 0 else p / 100.0)


def centre(cfg: Config, kind: str, line: float, fair_a: float) -> float:
    """The expected margin (home) or total that makes ``fair_a`` the chance of
    side A beating ``line``, pushes set aside: home covering ``line`` (it wins
    by more than ``-line``), or the over."""
    grid = GRID if kind == "spread" else TOTAL_GRID
    t = -line if kind == "spread" else line
    # Near the number only: the measured chances are not perfectly monotone in m (sampling noise
    # where few games had that line), and a far-off m can match a price by accident.
    grid = grid[np.abs(grid - t) <= 4.0]
    best, err = t, 9.0
    for m in grid:
        up, push, down = chances(cfg, kind, m, t)
        p = up / max(up + down, 1e-9)
        if abs(p - fair_a) < err:
            best, err = m, abs(p - fair_a)
    return float(best)


# ---------------------------------------------------------------------------
# The board
# ---------------------------------------------------------------------------
def latest(lines: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Each book's latest quote, games not started, books in the latest pull only."""
    now = now or pd.Timestamp.now(tz="UTC")
    d = lines.sort_values("logged_at").groupby(KEY + ["bookmaker"], dropna=False).tail(1).copy()
    starts = pd.to_datetime(d["starts"], utc=True, errors="coerce")
    d = d[starts.isna() | (starts > now)]
    newest = pd.to_datetime(d.groupby(KEY, dropna=False)["logged_at"].transform("max"), utc=True)
    return d[pd.to_datetime(d["logged_at"], utc=True) >= newest - pd.Timedelta(minutes=STALE_MINUTES)]


def board(cfg: Config, lines: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per game, market and side: the best price across books, against
    the other books' read (moneyline: their margin-free chance; spread and total:
    the margin or total they expect, priced at this book's number with pushes)."""
    if lines is None or lines.empty:
        return pd.DataFrame()
    d = latest(lines, now)
    rows = []
    for (event, market), g in d.groupby(KEY):
        if len(g) < MIN_OTHERS + 1:
            continue
        g = g.copy()
        g["fair_a"] = [_devig(r.price_a, r.price_b) for r in g.itertuples()]
        if market != "moneyline":
            g["m"] = [centre(cfg, market, r.line, r.fair_a) for r in g.itertuples()]
        best: dict[str, dict] = {}
        for r in g.itertuples():
            others = g[g["bookmaker"] != r.bookmaker]
            if len(others) < MIN_OTHERS:
                continue
            if market == "moneyline":
                pa = float(others["fair_a"].mean())
                legs = (("a", pa, 0.0, 1 - pa, r.price_a), ("b", 1 - pa, 0.0, pa, r.price_b))
                cons = None
            else:
                m = float(others["m"].mean())
                t = -r.line if market == "spread" else r.line
                up, push, down = chances(cfg, market, m, t)
                legs = (("a", up, push, down, r.price_a), ("b", down, push, up, r.price_b))
                cons = float(others["line"].median())
            for side, win, push, lose, price in legs:
                ev = win * (_decimal(price) - 1.0) - lose
                row = {"event_id": event, "market": market, "home": r.home, "away": r.away, "starts": r.starts,
                       "side": side, "line": None if r.line is None or pd.isna(r.line) else float(r.line),
                       "consensus": cons, "book": r.bookmaker, "price": float(price), "win": win, "push": push,
                       "ev": ev, "books": int(len(g)), "logged_at": str(r.logged_at)}
                if market == "moneyline" and win < LONGSHOT:
                    continue
                if side not in best or ev > best[side]["ev"]:
                    best[side] = row
        rows += list(best.values())
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["kind"] = np.where(out["ev"] >= WAY_OFF, "way off", np.where(out["ev"] >= MIN_EV, "market", ""))
    return out.sort_values("ev", ascending=False).reset_index(drop=True)


def side_text(r) -> str:
    """"BUF -3.5", "LV +150", "over 47.5"."""
    if r["market"] == "total":
        return f"{'over' if r['side'] == 'a' else 'under'} {r['line']:g}"
    team = r["home"] if r["side"] == "a" else r["away"]
    if r["market"] == "moneyline":
        return f"{team} to win"
    line = r["line"] if r["side"] == "a" else -r["line"]
    return f"{team} {line:+g}"


def hedges(cfg: Config, lines: pd.DataFrame, now: pd.Timestamp | None = None) -> dict:
    """Sure bets (the best of each side at different books summing under one,
    same number) and middles (spread or total at different numbers, both able to
    win), priced on the margin and total distributions."""
    if lines is None or lines.empty:
        return {"arbs": [], "middles": []}
    d = latest(lines, now)
    d = d.assign(da=[_decimal(x) for x in d["price_a"]], db=[_decimal(x) for x in d["price_b"]])
    arbs, middles = [], []
    for (event, market), g in d.groupby(KEY):
        if len(g) < 2:
            continue
        for line, at in g.groupby(g["line"].fillna(0.0)):
            # The best pair at two different books, either way round, read within ARB_MINUTES of each other.
            pair = None
            for a in at.itertuples():
                for b in at.itertuples():
                    if a.bookmaker == b.bookmaker:
                        continue
                    apart = abs((pd.Timestamp(a.logged_at) - pd.Timestamp(b.logged_at)).total_seconds()) / 60.0
                    s = 1 / a.da + 1 / b.db
                    if apart <= ARB_MINUTES and (pair is None or s < pair[0]):
                        pair = (s, a, b)
            if pair is not None and pair[0] < 1:
                s, a, b = pair
                arbs.append({"event_id": event, "market": market, "home": a.home, "away": a.away,
                             "line": None if market == "moneyline" else float(line),
                             "book_a": a.bookmaker, "price_a": float(a.price_a),
                             "book_b": b.bookmaker, "price_b": float(b.price_b),
                             "profit": 1 / s - 1, "share_a": (1 / a.da) / s,
                             "logged_at": str(min(a.logged_at, b.logged_at))})
        if market == "moneyline":
            continue
        fair = [_devig(r.price_a, r.price_b) for r in g.itertuples()]
        m = float(np.mean([centre(cfg, market, r.line, f) for r, f in zip(g.itertuples(), fair)]))
        top = None
        for x in g.itertuples():           # side A at x's number
            for y in g.itertuples():       # side B at y's number
                if x.bookmaker == y.bookmaker or x.line == y.line:
                    continue
                if market == "spread":
                    ta, tb = -x.line, -y.line      # home covers above ta; away covers below tb
                    if not ta < tb:
                        continue
                else:
                    ta, tb = x.line, y.line        # over above ta; under below tb
                    if not ta < tb:
                        continue
                k, p = _dist(str(cfg.raw_dir), market, float(np.round(m * 4) / 4))
                wa = np.where(k > ta, x.da - 1, np.where(k == ta, 0.0, -1.0))
                wb = np.where(k < tb, y.db - 1, np.where(k == tb, 0.0, -1.0))
                ev = float((p * (wa + wb)).sum()) / 2.0
                between = float(p[(k > ta) & (k < tb)].sum())
                if top is None or ev > top["ev"]:
                    top = {"event_id": event, "market": market, "home": x.home, "away": x.away,
                           "line_a": float(x.line), "book_a": x.bookmaker, "price_a": float(x.price_a),
                           "line_b": float(y.line), "book_b": y.bookmaker, "price_b": float(y.price_b),
                           "between": between, "ev": ev}
        if top and top["ev"] > 0:
            middles.append(top)
    arbs.sort(key=lambda r: -r["profit"])
    middles.sort(key=lambda r: -r["ev"])
    return {"arbs": arbs, "middles": middles}


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
def log_path(cfg: Config):
    return cfg.path("results") / "game_bet_log.parquet"


def flag(rows: pd.DataFrame, cfg: Config) -> int:
    if rows is None or rows.empty:
        return 0
    new = rows[rows["ev"] >= MIN_EV].copy()
    k = ["event_id", "market", "side", "line", "book"]
    p = log_path(cfg)
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    if not old.empty:
        seen = set(map(tuple, old[k].astype(str).to_numpy().tolist()))
        new = new[[tuple(r) not in seen for r in new[k].astype(str).to_numpy().tolist()]]
    if new.empty:
        return 0
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([old, new], ignore_index=True).to_parquet(p, index=False, compression="zstd")
    return len(new)


def settle(cfg: Config) -> pd.DataFrame:
    """The flags with their result from the final score (nflverse schedules)."""
    from ..teams import normalize_team

    p = log_path(cfg)
    if not p.exists():
        return pd.DataFrame()
    bets = pd.read_parquet(p)
    s = pd.read_parquet(cfg.raw_dir / "schedules.parquet", columns=["gameday", "home_team", "away_team", "result", "total"])
    s = s[s["result"].notna()].assign(home=lambda x: x["home_team"].map(normalize_team),
                                      away=lambda x: x["away_team"].map(normalize_team))
    s["day"] = s["gameday"].astype(str)
    res = []
    for b in bets.itertuples():
        day = str(pd.Timestamp(b.starts).tz_convert("America/New_York").date()) if b.starts else ""
        g = s[(s["home"] == b.home) & (s["away"] == b.away) & (s["day"] == day)]
        if g.empty:
            res.append(None)
            continue
        margin, total = float(g.iloc[0]["result"]), float(g.iloc[0]["total"])
        if b.market == "moneyline":
            won = (margin > 0) if b.side == "a" else (margin < 0)
            res.append("push" if margin == 0 else "win" if won else "loss")
        elif b.market == "spread":
            x = margin + b.line                     # home's margin against its spread
            x = x if b.side == "a" else -x
            res.append("push" if x == 0 else "win" if x > 0 else "loss")
        else:
            x = total - b.line
            x = x if b.side == "a" else -x
            res.append("push" if x == 0 else "win" if x > 0 else "loss")
    bets["result"] = res
    return bets


def summary(bets: pd.DataFrame) -> dict:
    if bets is None or bets.empty:
        return {"flagged": 0}
    done = bets[bets["result"].isin(["win", "loss"])]
    profit = sum((_decimal(float(r.price)) - 1.0) if r.result == "win" else -1.0 for r in done.itertuples())
    return {"flagged": int(len(bets)), "settled": int(len(done)), "wins": int((done["result"] == "win").sum()),
            "losses": int((done["result"] == "loss").sum()), "profit": float(profit), "mean_ev": float(bets["ev"].mean())}


def build(cfg: Config, frame: pd.DataFrame, now: pd.Timestamp | None = None) -> dict:
    """Each refresh: log this pull, build the board and the hedges, flag, grade."""
    if frame is not None and not frame.empty:
        record(frame, cfg)
    p = path(cfg)
    if not p.exists():
        return {}
    lines = pd.read_parquet(p)
    b = board(cfg, lines, now)
    h = hedges(cfg, lines, now)
    new = flag(b[b["kind"] != ""] if len(b) else b, cfg)
    rows = b[b["kind"] != ""].head(25) if len(b) else b
    import json as _json

    return {"rows": _json.loads(rows.to_json(orient="records")) if len(rows) else [],
            "arbs": h["arbs"][:10], "middles": h["middles"][:10], "new_flags": new,
            "record": summary(settle(cfg)), "quotes": int(len(latest(lines, now))),
            "games": int(lines["event_id"].nunique()),
            "quoted_at": str(latest(lines, now)["logged_at"].max()) if len(latest(lines, now)) else None}
