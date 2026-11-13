"""Which waiver breakouts would the page have priced the week before? (2022-2025)

    python scripts/audit_breakouts.py [--teams 10] [--sims 1000]

Every week 2-12 of 2022-2025 is replayed through the production projection
code (``ros_backtest``: only what was known that week). At each:

* **Waiver-level**: a skill player a league of ``--teams`` would plausibly
  have left on the wire. There is no archive of who was rostered, so a
  player counts as rostered if he is in the top N at his position (10 teams:
  QB 15, RB 50, WR 55, TE 18) by any of: our season value, last season's
  points a game (name value: a star we rate low, or one on a reserve list),
  this season's points a game so far (a hot start is picked up), or he is a
  rookie drafted in the first 50 picks. Everyone else is waiver-level.
* **Breakout**: over his team's next five games he played at least three and
  averaged a startable score: at or above the Nth best at his position over
  the same games (QB 10, RB 25, WR 25, TE 10, scaled with the league).
* **Priced**: whether the waiver engine's own selection would have put him in
  front of the title simulation that week -- the best few by value at each
  position, the upside classes (stepping in, next in line, rookie, role
  growing; :meth:`TitleEngine._upside_candidates`) and the IR stashes.

Reports how many breakouts were priced, by which route; what the missed ones
looked like (a tag the quota cut, the starter ahead going down afterwards, a
return from injury, a new team, no sign); how often each route's players
break out against the base rate; and whether the season simulation's own
chance of a startable rest of season is honest for waiver-level players.
Aggregates and a few names, for reading; nothing is fitted.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from streamer.config import get_config  # noqa: E402
from streamer.league.model import LONG_TERM_OUT_STATUSES, LeagueSnapshot, TeamRow  # noqa: E402
from streamer.roster import ros_backtest as rb  # noqa: E402
from streamer.roster.title_moves import STASH_EACH, TitleEngine  # noqa: E402

SEASONS = (2022, 2023, 2024, 2025)
WEEKS = tuple(range(2, 13))
POS = ("QB", "RB", "WR", "TE")
ROSTERED_10 = {"QB": 15, "RB": 50, "WR": 55, "TE": 18}
STARTABLE_10 = {"QB": 10, "RB": 25, "WR": 25, "TE": 10}
QUOTA = {"RB": 10, "WR": 15, "QB": 3, "TE": 10}     # the engine's (title_moves.TitleEngine)
WINDOW, MIN_PLAYED = 5, 3


def tags_of(p) -> list[str]:
    """Every upside sign on the player, whether or not a quota kept him."""
    out = []
    if any(s.startswith("next man up: ") for s in p.signals):
        out.append("stepping in")
    if p.experience == 0:
        out.append("rookie")
    if any(s.startswith("opportunity up") for s in p.signals):
        out.append("role growing")
    if any(s.startswith("on the field more") for s in p.signals):
        out.append("snaps up")
    if any(s.startswith("new team") for s in p.signals):
        out.append("new team")
    return out


def audit(ready: dict, history: pd.DataFrame, teams: int, draft: dict) -> pd.DataFrame:
    scale = teams / 10.0
    rostered = {k: int(round(v * scale)) for k, v in ROSTERED_10.items()}
    startable = {k: int(round(v * scale)) for k, v in STARTABLE_10.items()}
    rows = []
    for (season, week), (players, weeks, bye, _real) in sorted(ready.items()):
        h = history[(history["season"] == season) & history["week"].isin(weeks)]
        pts = h.groupby(["player_id", "week"])["fantasy_points_ppr"].sum().to_dict()
        skill = [p for p in players if p.position in POS and p.team and p.ros_value is not None]
        # The next five team games: points a game played, games played.
        win = {}
        for p in skill:
            tw = [w for w in weeks if w not in bye.get(p.team, set())][:WINDOW]
            got = [pts[(p.player_id, w)] for w in tw if (p.player_id, w) in pts]
            win[p.player_id] = (float(np.mean(got)) if got else 0.0, len(got), len(tw))
        bar = {}
        for pos in POS:
            ok = sorted((win[p.player_id][0] for p in skill if p.position == pos and win[p.player_id][1] >= MIN_PLAYED),
                        reverse=True)
            bar[pos] = ok[min(startable[pos], len(ok)) - 1] if ok else np.inf
        by_pos = {pos: sorted((p for p in skill if p.position == pos), key=lambda p: -float(p.ros_value)) for pos in POS}
        before = history[(history["season"] == season) & (history["week"] < week)]
        sofar = before.groupby("player_id")["fantasy_points_ppr"].agg(["mean", "size"])
        last = history[history["season"] == season - 1].groupby("player_id")["fantasy_points_ppr"].agg(["mean", "size"])
        named: set[str] = set()
        for pos in POS:
            ids = [p.player_id for p in by_pos[pos]]
            named |= set(ids[:rostered[pos]])
            for table, need in ((last, 6), (sofar, min(2, week - 1))):
                t = table[table.index.isin(ids) & (table["size"] >= need)]
                named |= set(t.sort_values("mean", ascending=False).index[:rostered[pos]])
        named |= {p.player_id for p in skill if p.experience == 0 and draft.get((season, p.player_id), 999) <= 50}
        held = [p for p in skill if p.player_id in named]
        pool_all = [p for p in skill if p.player_id not in named]
        # What his recent games looked like, from the box scores before this week.
        recent = {}
        for pid, g in before.sort_values("week").groupby("player_id"):
            l2 = g.tail(2)
            recent[pid] = (float(g["fantasy_points_ppr"].iloc[-1]), float(l2["fantasy_points_ppr"].mean()),
                           float(l2["total_fantasy_points_exp"].mean()), float(g["fantasy_points_ppr"].mean()),
                           float(g["total_fantasy_points_exp"].mean()), len(g))
        snap = LeagueSnapshot(platform="espn", profile="espn", league_id="audit", league_name="audit", season=season,
                              week=week, slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}, bench_size=7,
                              teams=[TeamRow(team_id="1", name="league", roster=held, is_mine=True)],
                              free_agents=pool_all, matchup=None, synced_at="")
        pool = [p for p in pool_all if not p.is_long_term_out]
        by_value: set[str] = set()
        for pos, n in QUOTA.items():
            at = sorted((p for p in pool if p.position == pos),
                        key=lambda p: -(float(p.ros_value or 0.0) + float(p.ros_sd or 0.0)))
            by_value |= {p.player_id for p in at[:n]}
        upside = TitleEngine._upside_candidates(snap, pool)
        stash = {p.player_id for p in sorted((p for p in pool_all if p.status in LONG_TERM_OUT_STATUSES),
                                             key=lambda p: -float(p.ros_value or 0.0))[:STASH_EACH]}
        mates: dict[tuple[str, str], list] = {}
        for p in skill:
            mates.setdefault((p.team, p.position), []).append(p)
        for p in pool_all:
            pg, played, tg = win[p.player_id]
            ahead = [q for q in mates[(p.team, p.position)] if float(q.ros_value) > float(p.ros_value)]
            lead = max(ahead, key=lambda q: float(q.ros_value)) if ahead else None
            lead_played = win[lead.player_id][1] if lead is not None else None
            why = upside.get(p.player_id, "").split(":")[0]
            why = "next in line" if why.startswith("next in line") else why
            route = ("value" if p.player_id in by_value else "upside: " + why
                     if p.player_id in upside else "stash" if p.player_id in stash else "")
            rc = recent.get(p.player_id, (0.0, 0.0, 0.0, 0.0, 0.0, 0))
            rows.append({"season": season, "week": week, "player_id": p.player_id, "name": p.name, "position": p.position,
                         "team": p.team, "status": p.status or "", "ros": float(p.ros_value), "role": p.role or "",
                         "experience": p.experience, "missed": int(p.games_missed or 0),
                         "ros_sd": float(p.ros_sd or 0.0), "proj": float(p.projection or 0.0),
                         "pick": float(draft.get((season, p.player_id), 260)),
                         "last": rc[0], "last2": rc[1], "last2_exp": rc[2], "ppg": rc[3], "exp_pg": rc[4], "games": rc[5],
                         "tags": ", ".join(tags_of(p)), "route": route, "pg": pg, "played": played, "team_games": tg,
                         "breakout": bool(tg >= WINDOW - 1 and played >= MIN_PLAYED and pg >= bar[p.position]),
                         "bar": bar[p.position],
                         "lead_out_now": bool(lead is not None and lead.is_out),
                         "lead_down_after": bool(lead is not None and not lead.is_out and lead_played is not None
                                                 and lead_played <= 2)})
    return pd.DataFrame(rows)


def cause(r) -> str:
    """Why a breakout was not priced, from what was and was not visible."""
    if r.tags:
        return "had a sign, cut by the per-class quota: " + r.tags.split(", ")[0]
    if r.status in ("OUT", "D", "Q") or r.status in LONG_TERM_OUT_STATUSES or r.missed >= 1:
        return "back from an injury or absence"
    if r.lead_out_now:
        return "starter ahead already out, not read as stepping in"
    if r.lead_down_after:
        return "starter ahead went down afterwards"
    return "no sign we read"


def report(d: pd.DataFrame, label: str) -> None:
    b = d[d["breakout"]]
    print(f"\n== {label}: {len(d)} waiver-level player-weeks, {len(b)} breakouts ({len(b) / len(d):.1%}); "
          f"{b[['season', 'player_id']].drop_duplicates().shape[0]} different player-seasons")
    priced = b[b["route"] != ""]
    print(f"priced the week before: {len(priced)} of {len(b)} ({len(priced) / len(b):.0%})")
    for pos in POS:
        x = b[b["position"] == pos]
        if len(x):
            print(f"  {pos}: {int((x.route != '').sum())} of {len(x)} ({(x.route != '').mean():.0%})")
    print("by season: " + " ".join(f"{s}: {(x.route != '').mean():.0%} of {len(x)}" for s, x in b.groupby("season")))
    print("how the priced ones got there: " + ", ".join(f"{k} {v}" for k, v in priced["route"].value_counts().items()))
    missed = b[b["route"] == ""].copy()
    missed["cause"] = [cause(r) for r in missed.itertuples()]
    print(f"missed ({len(missed)}), by what was visible:")
    for k, v in missed["cause"].value_counts().items():
        print(f"  {v:4d} ({v / len(missed):.0%})  {k}")
    print("how often each group broke out (base rate %.1f%%):" % (100 * d["breakout"].mean()))
    for k, x in d.groupby("route"):
        print(f"  {'not priced' if k == '' else 'priced, ' + k:36s} {len(x):6d} player-weeks, broke out {x.breakout.mean():.1%}")
    un = d[d["route"] == ""]
    for tag in ("stepping in", "rookie", "role growing", "snaps up", "new team"):
        x = un[un["tags"].str.contains(tag)]
        if len(x):
            print(f"  not priced, sign '{tag}':{'':8s} {len(x):6d} player-weeks, broke out {x.breakout.mean():.1%}")
    for name, m in (("not priced, starter ahead already out", un["lead_out_now"] & (un["tags"] == "")),
                    ("not priced, back from absence, no sign", ((un["status"] != "") | (un["missed"] >= 1)) & (un["tags"] == "")),
                    ("not priced, no sign at all", (un["tags"] == "") & (un["status"] == "") & (un["missed"] == 0) & ~un["lead_out_now"])):
        x = un[m]
        print(f"  {name:36s} {len(x):6d} player-weeks, broke out {x.breakout.mean():.1%}")


FEATURES = ["ros", "ros_sd", "proj", "last", "last2", "last2_exp", "ppg", "exp_pg", "games", "rookie", "early_pick",
            "lead_out_now", "hurt", "missed", "t_step", "t_role", "t_snaps", "t_team"]


def signs(d: pd.DataFrame) -> None:
    """Among the players the engine did not price: how often simple reads of
    the box scores were followed by a breakout."""
    un = d[d["route"] == ""]
    print(f"\nnot priced ({len(un)} player-weeks, broke out {un.breakout.mean():.1%}); how often each read was followed by one:")
    reads = [("expected points, last 2 games, 10+", un["last2_exp"] >= 10),
             ("expected points, last 2 games, 8-10", (un["last2_exp"] >= 8) & (un["last2_exp"] < 10)),
             ("scored 15+ last game", un["last"] >= 15),
             ("last 2 games 4+ above his season average", (un["last2"] - un["ppg"] >= 4) & (un["games"] >= 3)),
             ("expected points last 2 games 3+ above his season's", (un["last2_exp"] - un["exp_pg"] >= 3) & (un["games"] >= 3)),
             ("unlucky: expected 3+ a game above actual this season", (un["exp_pg"] - un["ppg"] >= 3) & (un["games"] >= 2)),
             ("starter ahead already out", un["lead_out_now"]),
             ("starter ahead out and expected 8+ last 2 games", un["lead_out_now"] & (un["last2_exp"] >= 8)),
             ("rookie drafted in the first 100 picks", (un["experience"] == 0) & (un["pick"] <= 100))]
    for name, m in reads:
        x = un[m]
        if len(x):
            print(f"  {name:52s} {len(x):5d} player-weeks, broke out {x.breakout.mean():5.1%}, {int(x.breakout.sum()):3d} breakouts")


def screen(d: pd.DataFrame) -> None:
    """If the same number of players were priced each week but chosen by one
    fitted score, how many breakouts would have been in front of the
    simulation? Fitted on three seasons, scored on the fourth."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    x = d.assign(rookie=(d["experience"] == 0).astype(float), early_pick=(d["pick"] <= 100).astype(float),
                 hurt=(d["status"] != "").astype(float), lead_out_now=d["lead_out_now"].astype(float),
                 t_step=d["tags"].str.contains("stepping in").astype(float), t_role=d["tags"].str.contains("role growing").astype(float),
                 t_snaps=d["tags"].str.contains("snaps up").astype(float), t_team=d["tags"].str.contains("new team").astype(float))
    x["score"] = np.nan
    for s in SEASONS:
        tr, te = x[x["season"] != s], x[x["season"] == s]
        for pos in POS:
            a, b = tr[tr["position"] == pos], te[te["position"] == pos]
            if a["breakout"].sum() < 10 or b.empty:
                continue
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=120, learning_rate=0.06, min_samples_leaf=40,
                                               l2_regularization=1.0, random_state=0)
            m.fit(a[FEATURES], a["breakout"])
            x.loc[b.index, "score"] = m.predict_proba(b[FEATURES])[:, 1]
    print("\nsame number priced each week, chosen by one fitted score (held out by season):")
    tot = [0, 0, 0]
    for pos in POS:
        got_e = got_s = n = 0
        for (_s, _w), g in x[x["position"] == pos].groupby(["season", "week"]):
            k = int((g["route"] != "").sum())
            top = g.nlargest(k, "score") if k else g.iloc[:0]
            got_e += int(g.loc[g["route"] != "", "breakout"].sum())
            got_s += int(top["breakout"].sum())
            n += int(g["breakout"].sum())
        tot = [tot[0] + got_e, tot[1] + got_s, tot[2] + n]
        print(f"  {pos}: of {n} breakouts, the engine's choice had {got_e} ({got_e / max(n, 1):.0%}), the score's {got_s} ({got_s / max(n, 1):.0%})")
    print(f"  all: engine {tot[0]} of {tot[2]} ({tot[0] / tot[2]:.0%}), score {tot[1]} ({tot[1] / tot[2]:.0%})")
    per = []
    for s in SEASONS:
        g = x[x["season"] == s]
        e = int(g.loc[g["route"] != "", "breakout"].sum())
        sc = sum(int(q.nlargest(int((q["route"] != "").sum()), "score")["breakout"].sum())
                 for _k, q in g.groupby(["week", "position"]))
        per.append(f"{s}: {e} -> {sc}")
    print("  by season (engine -> score): " + "  ".join(per))


def calibration(ready: dict, cfg, d: pd.DataFrame, teams: int, n_sims: int) -> None:
    """Is the simulation's chance of a startable rest of season honest for
    waiver-level players, and does it tell them apart?"""
    scale = teams / 10.0
    rows = []
    pool = {(r.season, r.week, r.player_id) for r in d.itertuples()}
    for (season, week), prep in sorted(ready.items()):
        cp = rb.checkpoint(None, cfg, season, week, None, n_sims=n_sims, ready=prep)
        real = cp.realized
        for pos in POS:
            ps = [p for p in cp.players if p.position == pos and p.player_id in real.index and p.player_id in cp.simulated]
            got = sorted((float(real.loc[p.player_id, "per_team_game"]) for p in ps), reverse=True)
            n = int(round(STARTABLE_10[pos] * scale))
            if len(got) < n:
                continue
            bar = got[n - 1]
            for p in ps:
                if (season, week, p.player_id) in pool:
                    rows.append({"season": season, "position": pos, "said": float((cp.simulated[p.player_id] >= bar).mean()),
                                 "did": float(real.loc[p.player_id, "per_team_game"] >= bar), "ros": float(p.ros_value)})
    c = pd.DataFrame(rows)
    print(f"\n== the simulation's chance of a startable rest of season, waiver-level players ({len(c)} player-weeks)")
    print(f"said {c.said.mean():.2%} on average, happened {c.did.mean():.2%}")
    c["bin"] = pd.cut(c["said"], [-1, 0.005, 0.02, 0.05, 0.10, 0.20, 1.0], labels=["<0.5%", "0.5-2%", "2-5%", "5-10%", "10-20%", "20%+"])
    t = c.groupby("bin", observed=True).agg(n=("did", "size"), said=("said", "mean"), happened=("did", "mean"))
    print(t.round(3).to_string())

    def auc(score, y):
        order = np.argsort(score)
        ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
        pos = y.sum()
        return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))) if 0 < pos < len(y) else np.nan

    y = c["did"].to_numpy()
    print(f"telling them apart (AUC; 0.5 is a coin flip): the simulation's chance {auc(c['said'].to_numpy(), y):.3f}, "
          f"season value alone {auc(c['ros'].to_numpy(), y):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", type=int, default=10)
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--cache", default="")
    args = ap.parse_args()
    cfg = get_config()
    from streamer.roster.projections import load_history

    history = load_history(cfg)
    if args.cache and Path(args.cache).exists():
        ready = pickle.load(open(args.cache, "rb"))
    else:
        ready = rb.prepare_all(cfg, SEASONS, WEEKS, log=lambda *a: None)
        if args.cache:
            pickle.dump(ready, open(args.cache, "wb"))
    draft = {}
    for s_ in SEASONS:
        r = rb.rosters(cfg, s_)
        if "draft_number" in r.columns:
            for gid, n in zip(r["gsis_id"], r["draft_number"]):
                if n == n:
                    draft[(s_, str(gid))] = float(n)
    d = audit(ready, history, args.teams, draft)
    d.to_parquet(cfg.raw_dir / f"breakout_audit_{args.teams}.parquet")
    report(d, f"{args.teams}-team league, weeks {WEEKS[0]}-{WEEKS[-1]}, {SEASONS[0]}-{SEASONS[-1]}")
    flex = d[d["position"] != "QB"]
    report(flex, "backs, receivers and tight ends only")
    signs(flex)
    screen(d)
    first = d[d["breakout"]].sort_values(["season", "player_id", "week"]).drop_duplicates(["season", "player_id"])
    report_first = first.assign(priced=first["route"] != "")
    print(f"\nfirst week each breakout began (one row a player-season): priced {report_first.priced.mean():.0%} of {len(first)}")
    miss = first[first["route"] == ""].assign(cause=[cause(r) for r in first[first["route"] == ""].itertuples()])
    for k, x in miss.groupby("cause"):
        ex = ", ".join(f"{r.name} ({r.season} wk {r.week}, {r.pg:.0f}/g)" for r in x.sort_values("pg", ascending=False).head(5).itertuples())
        print(f"  {len(x):3d}  {k}: {ex}")
    if not args.no_sim:
        calibration(ready, cfg, d, args.teams, args.sims)
    return 0


if __name__ == "__main__":
    sys.exit(main())
