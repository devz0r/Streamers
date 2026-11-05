"""A best-price board over the books' prop lines, and a record of what it flagged.

For each player and stat, the books quote an over/under at a line and two
prices. Each book's two prices carry its margin; scaled to sum to one they give
that book's own read of the chance of the over. A book whose price pays more
than the *other* books' average read says the over (or under) is worth -- a
price a bettor can beat on arithmetic alone, with no projection of ours in it.
Pinnacle, the sharpest book, counts double.

* :func:`board` -- the current best price on each side of each line, with its
  expected profit per unit staked against the other books' fair chance.
* :func:`flag` / :func:`settle` -- every price flagged (once, at the first pull
  that showed it) goes in ``results/bet_log.parquet`` and is graded later on
  two things: whether it won, and whether the market then moved its way
  (closing-line value -- the early test of a real edge, which needs weeks and
  not a season).

One-sided markets (anytime touchdown) cannot be scaled without the other
side's price and are left out. DECISIONS.md, "A best-price board".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)

SHARP = {"pinnacle": 2.0}
MIN_OTHERS = 2          # other books quoting the player and stat before a price is judged
MIN_EV = 0.02           # expected profit per unit a flag needs
KEY = ["season", "week", "event_id", "stat", "player"]
#: nflverse player_stats column a stat settles on.
SETTLE = {"passing_yards": "passing_yards", "passing_tds": "passing_tds", "passing_interceptions": "passing_interceptions",
          "rushing_yards": "rushing_yards", "receiving_yards": "receiving_yards", "receptions": "receptions"}


def _implied(price: pd.Series) -> np.ndarray:
    a = pd.to_numeric(price, errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a < 0, -a / (-a + 100.0), 100.0 / (a + 100.0))


def _decimal(price: float) -> float:
    return 1.0 + (100.0 / -price if price < 0 else price / 100.0)


def latest(lines: pd.DataFrame) -> pd.DataFrame:
    """Each book's latest two-sided quote, with its own fair chance of the over."""
    d = lines[lines["line"].notna() & lines["over_price"].notna() & lines["under_price"].notna()].copy()
    d = d.sort_values("logged_at").groupby(KEY[2:] + ["bookmaker", "season", "week"], dropna=False).tail(1)
    po, pu = _implied(d["over_price"]), _implied(d["under_price"])
    d["fair_over"] = po / (po + pu)
    d["hold"] = po + pu - 1.0
    d["w"] = d["bookmaker"].map(SHARP).fillna(1.0)
    return d


#: Stats that are small whole counts settled on a half (catches too: a smooth
#: curve misprices 0.5 and 1.5): Poisson; the rest on the
#: shape measured from 2021-2025 player-games (``prop_shape.json``).
POISSON = {"passing_tds", "passing_interceptions", "receptions"}
#: Our per-stat projection's key for each prop stat (``roster.stat_projections``).
OURS = {"passing_yards": "pass_yards", "passing_tds": "pass_td", "passing_interceptions": "interception",
        "rushing_yards": "rush_yards", "receiving_yards": "rec_yards", "receptions": "reception"}
#: How far the combined number sits toward the market's, per stat: the per-stat
#: market blend fitted on weeks 6-8 of 2026 (DECISIONS.md, "Projecting each
#: stat"); provisional, refitted as weeks are graded.
MARKET_WEIGHT = {"passing_yards": 0.9, "passing_tds": 0.75, "passing_interceptions": 0.9, "rushing_yards": 0.8,
                 "receptions": 0.6, "receiving_yards": 0.4}
WAY_OFF = 0.10          # expected profit a line needs, on both the market's and our read, to be "way off"


def _rows(shape: dict, stat: str, level: float):
    rows = (shape.get("stats") or {}).get(stat)
    if not rows:
        return None
    meds = np.array([r["median"] for r in rows], dtype=float)
    i = int(np.clip(np.searchsorted(meds, level), 1, len(rows) - 1))
    lo, hi = rows[i - 1], rows[i]
    t = float(np.clip((level - lo["median"]) / max(hi["median"] - lo["median"], 1e-9), 0.0, 1.0))
    q = (1 - t) * np.asarray(lo["q"], dtype=float) + t * np.asarray(hi["q"], dtype=float)
    gap = (1 - t) * lo["mean_minus_median"] + t * hi["mean_minus_median"]
    return q, float(gap)


def center(stat: str, line: float, p_over: float, shape: dict) -> float | None:
    """The distribution's centre that puts ``p_over`` above ``line``: the median
    on the measured shape, the rate for a Poisson stat."""
    from .props import mean_from_poisson_line

    if stat in POISSON:
        return mean_from_poisson_line(line, p_over)
    got = _rows(shape, stat, line)
    if got is None:
        return None
    q, _gap = got
    grid = np.asarray(shape["grid"], dtype=float)
    return float(line) - float(np.interp(float(np.clip(1 - p_over, grid[0], grid[-1])), grid, q))


def p_over(stat: str, c: float, line: float, shape: dict) -> float | None:
    """The chance of beating ``line`` for a player centred at ``c`` (median, or rate)."""
    from scipy import stats as st

    if stat in POISSON:
        return float(st.poisson.sf(int(np.ceil(line)) - 1, max(c, 1e-6)))
    got = _rows(shape, stat, c)
    if got is None:
        return None
    q, _gap = got
    grid = np.asarray(shape["grid"], dtype=float)
    u = float(np.interp(float(line) - c, q, grid, left=0.0, right=1.0))
    return float(np.clip(1.0 - u, 0.005, 0.995))


def median_of_mean(stat: str, mean: float, shape: dict) -> float:
    """Our projection is a mean; the shape is about the median."""
    if stat in POISSON:
        return float(mean)
    got = _rows(shape, stat, mean)
    return float(mean) - (got[1] if got else 0.0)


def board(lines: pd.DataFrame, exclude: set[str] | None = None, min_ev: float = 0.0,
          ours: dict | None = None) -> pd.DataFrame:
    """Every book's quote on every line, priced three ways: the other books'
    read (their quotes, margin out, fitted to one curve for the player so any
    line can be priced -- a lone 30.5 against a 72.5 consensus included), our
    projection's (``ours``: normalised name -> {stat: mean}), and the two
    combined. One row per player, stat and side: the book paying most on the
    combined read, with its expected profit per unit on each."""
    from ..roster.players import normalize_name
    from .props import load_shape

    if lines is None or lines.empty:
        return pd.DataFrame()
    shape = load_shape()
    d = latest(lines)
    d["_n"] = d["player"].map(normalize_name)
    if exclude:
        d = d[~d["_n"].isin(exclude)]
    rows = []
    for (season, week, event, stat, player), g in d.groupby(["season", "week", "event_id", "stat", "player"], dropna=False):
        if len(g) < MIN_OTHERS + 1:
            continue
        g = g.assign(c=[center(stat, r.line, r.fair_over, shape) for r in g.itertuples()])
        g = g[g["c"].notna()]
        mine = (ours or {}).get(normalize_name(player), {}).get(OURS.get(stat, ""))
        c_ours = median_of_mean(stat, mine, shape) if mine is not None and mine > 0 else None
        best: dict[str, dict] = {}
        for r in g.itertuples():
            others = g[g["bookmaker"] != r.bookmaker]
            if len(others) < MIN_OTHERS:
                continue
            c_mkt = float(np.average(others["c"], weights=others["w"]))
            same = others[others["line"] == r.line]
            # Books on the same line say it directly; the curve prices a line nobody else quotes.
            pm = (float(np.average(same["fair_over"], weights=same["w"])) if len(same) >= MIN_OTHERS
                  else p_over(stat, c_mkt, r.line, shape))
            po = p_over(stat, c_ours, r.line, shape) if c_ours is not None else None
            w = MARKET_WEIGHT.get(stat, 1.0)
            c_comb = c_mkt if c_ours is None else w * c_mkt + (1 - w) * c_ours
            pc = p_over(stat, c_comb, r.line, shape)
            if pm is None or pc is None:
                continue
            for side, price in (("over", r.over_price), ("under", r.under_price)):
                flip = (lambda x: x) if side == "over" else (lambda x: None if x is None else 1.0 - x)
                dec = _decimal(float(price))
                ev_c, ev_m = flip(pc) * dec - 1.0, flip(pm) * dec - 1.0
                ev_o = flip(po) * dec - 1.0 if po is not None else None
                row = {"season": season, "week": week, "event_id": event, "stat": stat, "player": player,
                       "side": side, "line": float(r.line), "book": r.bookmaker, "price": float(price),
                       "fair": flip(pc), "p_market": flip(pm), "p_ours": flip(po), "ev": ev_c, "ev_market": ev_m,
                       "ev_ours": ev_o, "books": int(len(g)), "consensus": float(np.average(g["line"], weights=g["w"])),
                       "ours_mean": None if mine is None else float(mine),
                       "way_off": bool(ev_m >= WAY_OFF and (ev_o is None or ev_o >= WAY_OFF)),
                       "logged_at": str(g["logged_at"].max())}
                if side not in best or ev_c > best[side]["ev"]:
                    best[side] = row
        rows += [b for b in best.values() if b["ev"] >= min_ev]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # What kind of edge: the books' own read beats the price ("market", "way off" when ours agrees by 10%+),
    # or only our projection says so ("model": unproven until the record says otherwise).
    out["kind"] = np.where(out["way_off"], "way off", np.where(out["ev_market"] >= MIN_EV, "market", "model"))
    order = {"way off": 0, "market": 1, "model": 2}
    out["_k"] = out["kind"].map(order)
    return out.sort_values(["_k", "ev_market", "ev"], ascending=[True, False, False]).drop(columns="_k").reset_index(drop=True)


def hedges(lines: pd.DataFrame, exclude: set[str] | None = None, min_middle_ev: float = 0.0) -> dict:
    """Bets on the same player and stat at different books that lock in a profit
    (``arbs``) or can win both ways (``middles``).

    * **Arb**: the best over and the best under, at different books on the same
      line, whose implied chances add to under one. Stake the over share
      ``(1 / over odds) / sum`` of the total and either side returns the same.
    * **Middle**: over at a lower line at one book, under at a higher line at
      another. Both win if the result lands between; otherwise one wins and one
      loses (about the books' margin, 4% or so). The chance of landing between
      comes from the same curve as the board (the books' consensus), and the
      expected profit per unit staked is priced on it.
    """
    from ..roster.players import normalize_name
    from .props import load_shape

    if lines is None or lines.empty:
        return {"arbs": [], "middles": [], "checked": 0}
    shape = load_shape()
    d = latest(lines)
    if exclude:
        d = d[~d["player"].map(normalize_name).isin(exclude)]
    d = d.assign(do=[_decimal(float(x)) for x in d["over_price"]], du=[_decimal(float(x)) for x in d["under_price"]])
    arbs, middles = [], []
    for (_season, _week, _event, stat, player), g in d.groupby(["season", "week", "event_id", "stat", "player"], dropna=False):
        if len(g) < 2:
            continue
        for line, at in g.groupby("line"):
            o, u = at.loc[at["do"].idxmax()], at.loc[at["du"].idxmax()]
            if o["bookmaker"] == u["bookmaker"]:
                alt = at[at["bookmaker"] != o["bookmaker"]]
                if alt.empty:
                    continue
                u = alt.loc[alt["du"].idxmax()]
            total = 1.0 / o["do"] + 1.0 / u["du"]
            if total < 1.0:
                arbs.append({"player": player, "stat": stat, "line": float(line), "over_book": o["bookmaker"],
                             "over_price": float(o["over_price"]), "under_book": u["bookmaker"],
                             "under_price": float(u["under_price"]), "profit": 1.0 / total - 1.0,
                             "over_share": (1.0 / o["do"]) / total, "logged_at": str(min(o["logged_at"], u["logged_at"]))})
        centres = [center(stat, r.line, r.fair_over, shape) for r in g.itertuples()]
        ok = [c for c in centres if c is not None]
        if not ok:
            continue
        c_mkt = float(np.average(ok, weights=g["w"].to_numpy()[[c is not None for c in centres]]))
        best = None
        for lo in g.itertuples():
            for hi in g.itertuples():
                if lo.line >= hi.line or lo.bookmaker == hi.bookmaker:
                    continue
                pa, pb = p_over(stat, c_mkt, lo.line, shape), p_over(stat, c_mkt, hi.line, shape)
                if pa is None or pb is None:
                    continue
                gap, below, above = max(pa - pb, 0.0), 1.0 - pa, pb
                ev = (below * (hi.du - 2.0) + gap * (lo.do + hi.du - 2.0) + above * (lo.do - 2.0)) / 2.0
                if best is None or ev > best["ev"]:
                    best = {"player": player, "stat": stat, "over_line": float(lo.line), "over_book": lo.bookmaker,
                            "over_price": float(lo.over_price), "under_line": float(hi.line), "under_book": hi.bookmaker,
                            "under_price": float(hi.under_price), "gap": float(gap), "ev": float(ev),
                            "logged_at": str(min(lo.logged_at, hi.logged_at))}
        if best and best["ev"] > min_middle_ev:
            middles.append(best)
    arbs.sort(key=lambda x: -x["profit"])
    middles.sort(key=lambda x: -x["ev"])
    return {"arbs": arbs, "middles": middles, "checked": int(len(d))}


def path(cfg: Config):
    return cfg.path("results") / "bet_log.parquet"


def flag(flagged: pd.DataFrame, cfg: Config) -> int:
    """Add the flagged prices not already logged; returns how many were new.
    A bet is keyed by player, stat, side, line and book, so a price that stays
    flagged is one bet, recorded at the first pull that showed it."""
    if flagged is None or flagged.empty:
        return 0
    k = KEY + ["side", "line", "book"]
    new = flagged[(flagged["ev"] >= MIN_EV) | (flagged["ev_market"] >= MIN_EV)].copy()
    p = path(cfg)
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    if not old.empty:
        seen = set(map(tuple, old[k].to_numpy().tolist()))
        new = new[[tuple(r) not in seen for r in new[k].to_numpy().tolist()]]
    if new.empty:
        return 0
    out = pd.concat([old, new], ignore_index=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False, compression="zstd")
    return len(new)


def settle(cfg: Config, lines: pd.DataFrame) -> pd.DataFrame:
    """The bet log with each bet's result (win / loss / push, once the game is in
    nflverse's stats) and its closing-line value (the closing fair chance of its
    side, from the latest pull at the same line, less the chance its price
    implied; positive: the market came to it)."""
    from ..roster.players import normalize_name

    p = path(cfg)
    if not p.exists():
        return pd.DataFrame()
    bets = pd.read_parquet(p)
    if bets.empty:
        return bets
    # Flags logged before the board told the kinds apart were all on the books' read alone.
    bets["kind"] = bets["kind"].fillna("market") if "kind" in bets else "market"
    lt = latest(lines)
    close = lt.groupby(KEY[2:] + ["line"]).apply(
        lambda g: pd.Series({"close_over": float(np.average(g["fair_over"], weights=g["w"])),
                             "close_at": str(g["logged_at"].max())}), include_groups=False).reset_index()
    bets = bets.merge(close, on=KEY[2:] + ["line"], how="left")
    bets["close_fair"] = np.where(bets["side"] == "over", bets["close_over"], 1.0 - bets["close_over"])
    # Only a later pull says where the market went; the flagging pull itself proves nothing.
    later = pd.to_datetime(bets["close_at"], utc=True, errors="coerce") > pd.to_datetime(bets["logged_at"], utc=True)
    bets.loc[~later, "close_fair"] = np.nan
    bets["implied"] = _implied(bets["price"])
    bets["clv"] = bets["close_fair"] - bets["implied"]
    actual = {}
    for season in sorted(bets["season"].unique()):
        f = cfg.raw_dir / f"player_stats_{int(season)}.parquet"
        if f.exists():
            st = pd.read_parquet(f)
            st = st[st["season_type"].eq("REG")]
            st["_n"] = st["player_display_name"].map(normalize_name)
            actual[int(season)] = st.drop_duplicates(["_n", "week"]).set_index(["_n", "week"]).sort_index()
    res = []
    for b in bets.itertuples():
        st = actual.get(int(b.season))
        col = SETTLE.get(b.stat)
        v = None
        if st is not None and col in st.columns:
            try:
                v = float(st.loc[(normalize_name(b.player), int(b.week)), col])
            except (KeyError, TypeError, ValueError):
                v = None
        if v is None or np.isnan(v):
            res.append(None)
        elif v == b.line:
            res.append("push")
        else:
            res.append("win" if (v > b.line) == (b.side == "over") else "loss")
    bets["result"] = res
    return bets


def summary_by_kind(bets: pd.DataFrame) -> dict:
    if bets is None or bets.empty or "kind" not in bets:
        return {}
    return {k: summary(g) for k, g in bets.groupby("kind")}


def summary(bets: pd.DataFrame) -> dict:
    """The record: flagged, settled, wins-losses, flat one-unit profit, mean
    expected profit, and how often the market moved toward the bet."""
    if bets is None or bets.empty:
        return {"flagged": 0}
    done = bets[bets["result"].isin(["win", "loss"])]
    profit = sum((_decimal(float(r.price)) - 1.0) if r.result == "win" else -1.0 for r in done.itertuples())
    c = bets["clv"].dropna()
    return {"flagged": int(len(bets)), "settled": int(len(done)), "wins": int((done["result"] == "win").sum()),
            "losses": int((done["result"] == "loss").sum()), "profit": float(profit),
            "mean_ev": float(bets["ev"].mean()), "clv_n": int(len(c)),
            "clv_hit": float((c > 0).mean()) if len(c) else None, "clv_mean": float(c.mean()) if len(c) else None}


def build(cfg: Config, lines: pd.DataFrame, started: set[str] | None = None, now: str | None = None,
          ours: dict | None = None) -> dict:
    """Each refresh: the board (games not yet started), this pull's new flags
    logged, the record graded; written to ``results/betting_board.json`` for
    the page. Returns what was written."""
    import json

    flagged = board(lines, exclude=started, min_ev=0.0, ours=ours)
    new = flag(flagged, cfg)
    graded = settle(cfg, lines)
    record = summary(graded)
    by_kind = summary_by_kind(graded)
    stamp = now or pd.Timestamp.now(tz="UTC").isoformat()
    rows = pd.concat([flagged[flagged["kind"] != "model"].head(30),
                      flagged[(flagged["kind"] == "model") & (flagged["ev"] > 0)].head(20)]) if len(flagged) else flagged
    hedge = hedges(lines, exclude=started)
    out = {"hedges": {"arbs": hedge["arbs"][:15], "middles": hedge["middles"][:15], "checked": hedge["checked"],
                      "n_arbs": len(hedge["arbs"]), "n_middles": len(hedge["middles"])},
           "built_at": stamp, "new_flags": new, "record": record, "sides": int(len(flagged)),
           "way_off": int(flagged["way_off"].sum()) if len(flagged) else 0, "by_kind": by_kind,
           "rows": json.loads(rows.to_json(orient="records"))}
    p = cfg.path("results") / "betting_board.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return out
