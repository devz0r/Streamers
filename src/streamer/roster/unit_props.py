"""Betting lines for kickers and defences: logged now, graded as weeks finish.

The kicker and D/ST models already lean on Vegas through the game lines
(team and opponent implied totals). Two more lines say something those do
not, and are already in, or cheap to add to, the props we buy:

* **A kicker's kicking points** (``player_kicking_points``: field goals x 3
  plus extra points) -- whether his team's drives stall into field goals,
  his range, the weather.
* **The opposing quarterback's interceptions** (``player_pass_interceptions``,
  bought for the skill-player blend anyway) -- the part of a defence's
  big-play points the game line knows least about.

There is no free history of player props, so neither can be backtested. Each
week's lines are logged next to our projection for that unit
(``results/<profile>/unit_log.parquet``, overwritten until kickoff like the
skill log) and graded against what the units scored (:func:`grade`,
``scripts/grade_unit_props.py``). Neither moves a projection until the grade
says it helps, held out, by the usual rule; until then the cards show the
line beside ours. DECISIONS.md, "Kicker props and the opposing QB's
interception line".
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..league.model import LeagueSnapshot
from ..teams import normalize_team

log = logging.getLogger(__name__)

#: Graded rows needed before :func:`grade` fits anything.
MIN_ROWS = 60


def kicker_lines(snapshot: LeagueSnapshot) -> dict[str, tuple[str, float, int]]:
    """Team -> (kicker, his kicking-points mean, books) for every kicker the
    books priced."""
    out = {}
    for p in snapshot.all_players():
        kp = (p.vegas_stats or {}).get("kicking_points") if p.position == "K" else None
        if kp is not None and p.team:
            out[normalize_team(p.team)] = (p.name, float(kp), int(p.vegas_books or 0))
    return out


def qb_int_lines(snapshot: LeagueSnapshot) -> dict[str, tuple[str, float]]:
    """Team -> (quarterback, his posted interceptions mean): the team's QB
    with the highest projection among those a book priced for interceptions
    (a line filled in from his recent games does not count)."""
    best: dict[str, tuple[float, str, float]] = {}
    for p in snapshot.all_players():
        if p.position != "QB" or not p.team:
            continue
        ints = (p.vegas_stats or {}).get("passing_interceptions")
        if ints is None or "passing_interceptions" in (getattr(p, "vegas_filled", None) or []):
            continue
        team = normalize_team(p.team)
        proj = float(p.projection or 0.0)
        if team not in best or proj > best[team][0]:
            best[team] = (proj, p.name, float(ints))
    return {t: (name, ints) for t, (_proj, name, ints) in best.items()}


def rows(snapshot: LeagueSnapshot, rankings) -> list[dict]:
    """This week's log rows: every ranked kicker the books priced, and every
    ranked defence whose opponent's quarterback has an interceptions line."""
    out = []
    ks, ints = kicker_lines(snapshot), qb_int_lines(snapshot)
    for pos, frame in (("K", getattr(rankings, "kicker", None)), ("DST", getattr(rankings, "dst", None))):
        if frame is None or frame.empty:
            continue
        for r in frame.itertuples():
            team, opp = normalize_team(r.team), normalize_team(getattr(r, "opponent", "") or "")
            row = {"season": int(snapshot.season), "week": int(snapshot.week), "position": pos, "team": team,
                   "opponent": opp, "expected_points": float(r.expected_points),
                   "baseline_points": float(getattr(r, "baseline_points", np.nan)),
                   "kicker": None, "kicking_points": None, "books": None, "opp_qb": None, "opp_qb_int": None}
            if pos == "K" and team in ks:
                row["kicker"], row["kicking_points"], row["books"] = ks[team]
            elif pos == "DST" and opp in ints:
                row["opp_qb"], row["opp_qb_int"] = ints[opp]
            else:
                continue
            out.append(row)
    return out


def annotate(snapshot: LeagueSnapshot, rankings) -> int:
    """Put the line on each kicker's and defence's card (information only,
    until graded). Returns how many cards got one."""
    league_int = _league_int(snapshot)
    by = {(r["position"], r["team"]): r for r in rows(snapshot, rankings)}
    n = 0
    for p in snapshot.all_players():
        if p.position not in ("K", "DST") or not p.team:
            continue
        r = by.get((p.position, normalize_team(p.team)))
        if r is None:
            continue
        if p.position == "K":
            p.signals.append(f"the books' kicking-points line: {r['kicking_points']:.1f} (field goals x 3 plus "
                             "extra points; logged and graded, not yet in his projection)")
        else:
            rel = "" if league_int is None else (
                f", {'above' if r['opp_qb_int'] > league_int else 'below'} this week's average of {league_int:.2f}")
            p.signals.append(f"the books expect {r['opp_qb']} to throw {r['opp_qb_int']:.2f} interceptions{rel} "
                             "(logged and graded, not yet in the projection)")
        n += 1
    return n


def _league_int(snapshot: LeagueSnapshot) -> float | None:
    lines = [v for _n, v in qb_int_lines(snapshot).values()]
    return float(np.mean(lines)) if len(lines) >= 4 else None


def log_week(snapshot: LeagueSnapshot, cfg: Config, rankings, now=None) -> int:
    """Write this week's rows to ``unit_log.parquet``, replacing earlier rows
    for the same unit and week unless his game has started (those stand)."""
    new = pd.DataFrame(rows(snapshot, rankings))
    if new.empty:
        return 0
    new["logged_at"] = pd.Timestamp.now(tz="UTC").isoformat() if now is None else str(now)
    path = cfg.results_dir / "unit_log.parquet"
    key = ["season", "week", "position", "team"]
    if path.exists():
        old = pd.read_parquet(path)
        started = _started(snapshot)
        old_keys = set(map(tuple, old[key].to_numpy().tolist()))
        new_keys = set(map(tuple, new[key].to_numpy().tolist()))
        # A unit whose game has started keeps the line logged before it.
        new = new[[not (k in old_keys and k[3] in started) for k in map(tuple, new[key].to_numpy().tolist())]]
        replaced = {k for k in map(tuple, new[key].to_numpy().tolist())} & new_keys
        old = old[[k not in replaced for k in map(tuple, old[key].to_numpy().tolist())]]
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return len(new)


def _started(snapshot: LeagueSnapshot) -> set[str]:
    """Teams whose game this week has kicked off (their logged lines stand)."""
    return {normalize_team(p.team) for p in snapshot.all_players() if p.team and (p.locked or p.actual_points is not None)}


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def graded(cfg: Config) -> pd.DataFrame:
    """Logged rows joined to what the units scored (``history.parquet``)."""
    path = cfg.results_dir / "unit_log.parquet"
    hist = cfg.results_dir / "history.parquet"
    if not path.exists() or not hist.exists():
        return pd.DataFrame()
    lg = pd.read_parquet(path)
    lg["team"] = lg["team"].map(normalize_team)
    h = pd.read_parquet(hist)[["season", "week", "position", "team", "actual_points"]].dropna()
    h["team"] = h["team"].map(normalize_team)
    return lg.merge(h.drop_duplicates(["season", "week", "position", "team"]),
                    on=["season", "week", "position", "team"])


def grade(cfg: Config) -> dict:
    """Does each line add to our projection? Leave one week out: the blend
    (kickers: ``(1 - w) * ours + w * c * line``; defences: ``ours + b *
    (int line - that week's mean)``) is fitted on the other weeks and scored
    on the held-out one. ``keep`` follows the usual rule: better overall,
    worse in at most one week in four."""
    d = graded(cfg)
    out: dict = {}
    for pos, col in (("K", "kicking_points"), ("DST", "opp_qb_int")):
        g = d[(d.position == pos)].dropna(subset=[col, "expected_points", "actual_points"]) if not d.empty else d
        res = {"rows": int(len(g)), "weeks": int(g.week.nunique()) if len(g) else 0}
        if len(g) < MIN_ROWS or res["weeks"] < 3:
            res["verdict"] = f"waiting: {len(g)} graded rows over {res['weeks']} weeks (need {MIN_ROWS}, 3 weeks)"
            out[pos] = res
            continue
        g = g.copy()
        if pos == "DST":
            g["x"] = g[col] - g.groupby(["season", "week"])[col].transform("mean")
        preds, base = [], []
        for wk, te in g.groupby(["season", "week"]):
            tr = g[(g.season != wk[0]) | (g.week != wk[1])]
            if pos == "K":
                c = float((tr.actual_points * tr[col]).sum() / max((tr[col] ** 2).sum(), 1e-9))
                grid = np.round(np.arange(0, 1.01, 0.05), 2)
                w = min(grid, key=lambda v: float(((1 - v) * tr.expected_points + v * c * tr[col]
                                                   - tr.actual_points).abs().mean()))
                p = (1 - w) * te.expected_points + w * c * te[col]
            else:
                resid = tr.actual_points - tr.expected_points
                b = float((resid * tr.x).sum() / max((tr.x ** 2).sum(), 1e-9))
                p = te.expected_points + b * te.x
            preds.append((te.actual_points - p).abs().mean())
            base.append((te.actual_points - te.expected_points).abs().mean())
        preds, base = np.array(preds), np.array(base)
        worse = int((preds > base + 1e-9).sum())
        res.update({"mae_ours": round(float(base.mean()), 3), "mae_with_line": round(float(preds.mean()), 3),
                    "weeks_worse": worse,
                    "keep": bool(preds.mean() < base.mean() and worse <= max(1, len(base) // 4))})
        res["verdict"] = "KEEP" if res["keep"] else "DROP"
        out[pos] = res
    return out


def describe(result: dict) -> str:
    return json.dumps(result, indent=2, sort_keys=True)

