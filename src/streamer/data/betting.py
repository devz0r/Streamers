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
MIN_OTHERS = 2          # other books at the same line before a price is judged
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


def board(lines: pd.DataFrame, exclude: set[str] | None = None, min_ev: float = 0.0) -> pd.DataFrame:
    """One row per side of each line worth a look: the best price across books at the
    line most books quote, its fair chance (the other books' weighted average),
    expected profit per unit, how many books, and when it was pulled."""
    from ..roster.players import normalize_name

    if lines is None or lines.empty:
        return pd.DataFrame()
    d = latest(lines)
    if exclude:
        d = d[~d["player"].map(normalize_name).isin(exclude)]
    rows = []
    for (season, week, event, stat, player), g in d.groupby(["season", "week", "event_id", "stat", "player"], dropna=False):
        by_line = g.groupby("line")["w"].sum()
        line = by_line.idxmax()
        at = g[g["line"] == line]
        if len(at) < MIN_OTHERS + 1:
            continue
        for side in ("over", "under"):
            best = None
            for r in at.itertuples():
                others = at[at["bookmaker"] != r.bookmaker]
                p_over = float(np.average(others["fair_over"], weights=others["w"]))
                fair = p_over if side == "over" else 1.0 - p_over
                price = float(r.over_price if side == "over" else r.under_price)
                ev = fair * _decimal(price) - 1.0
                if best is None or ev > best["ev"]:
                    best = {"season": season, "week": week, "event_id": event, "stat": stat, "player": player,
                            "side": side, "line": float(line), "book": r.bookmaker, "price": price, "fair": fair,
                            "ev": ev, "books": int(len(at)), "spread": float(others["fair_over"].max() - others["fair_over"].min()),
                            "logged_at": str(at["logged_at"].max())}
            if best and best["ev"] >= min_ev:
                rows.append(best)
    out = pd.DataFrame(rows)
    return out.sort_values("ev", ascending=False).reset_index(drop=True) if not out.empty else out


def path(cfg: Config):
    return cfg.path("results") / "bet_log.parquet"


def flag(flagged: pd.DataFrame, cfg: Config) -> int:
    """Add the flagged prices not already logged; returns how many were new.
    A bet is keyed by player, stat, side, line and book, so a price that stays
    flagged is one bet, recorded at the first pull that showed it."""
    if flagged is None or flagged.empty:
        return 0
    k = KEY + ["side", "line", "book"]
    new = flagged[flagged["ev"] >= MIN_EV].copy()
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
    close = latest(lines).groupby(KEY[2:] + ["line"]).apply(
        lambda g: float(np.average(g["fair_over"], weights=g["w"])), include_groups=False).rename("close_over").reset_index()
    bets = bets.merge(close, on=KEY[2:] + ["line"], how="left")
    bets["close_fair"] = np.where(bets["side"] == "over", bets["close_over"], 1.0 - bets["close_over"])
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


def build(cfg: Config, lines: pd.DataFrame, started: set[str] | None = None, now: str | None = None) -> dict:
    """Each refresh: the board (games not yet started), this pull's new flags
    logged, the record graded; written to ``results/betting_board.json`` for
    the page. Returns what was written."""
    import json

    flagged = board(lines, exclude=started, min_ev=0.0)
    new = flag(flagged, cfg)
    record = summary(settle(cfg, lines))
    stamp = now or pd.Timestamp.now(tz="UTC").isoformat()
    rows = flagged[flagged["ev"] > 0].head(40)
    out = {"built_at": stamp, "new_flags": new, "record": record, "sides": int(len(flagged)),
           "rows": json.loads(rows.to_json(orient="records"))}
    p = cfg.path("results") / "betting_board.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return out
