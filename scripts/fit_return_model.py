"""Fit and grade the return model: the chance a hurt player plays each coming
game, from his injury report (2015-2025).

    python scripts/fit_return_model.py            # build the panel, grade held out by season
    python scripts/fit_return_model.py --write    # ... and write roster/return_model.json

A *checkpoint* is a team game a skill player missed while on the injury
report (out, doubtful, questionable) or a reserve list. From it, each later
team game he has not yet returned for is a row: did he play it? What is known
at the checkpoint -- games missed in a row, the injury, his last practice,
the report's status, reserve list or not, his level, the week, a bye right
after -- is what the page knows about a player tagged out today. The fit is
a logistic on those; the grade is against the tables the simulation draws
from now (absence lengths by position, and the reserve-list table).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamer.config import get_config  # noqa: E402
from streamer.roster import return_model as rm  # noqa: E402
from streamer.teams import normalize_team  # noqa: E402

SEASONS = tuple(range(2015, 2026))
SKILL = ("QB", "RB", "WR", "TE")
LAST = 18
#: How much less the reserve-list practice terms are penalised (see :func:`fit`).
LIGHT = 4.0


def _load(cfg, name: str, season: int) -> pd.DataFrame:
    import nflreadpy

    path = cfg.raw_dir / f"{name}_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    loader = {"injuries": nflreadpy.load_injuries, "rosters_weekly": nflreadpy.load_rosters_weekly,
              "player_stats": nflreadpy.load_player_stats}[name]
    frame = loader(seasons=[season]).to_pandas()
    frame.to_parquet(path)
    return frame


def season_panel(cfg, season: int, prev_ppg: dict[str, float]) -> tuple[pd.DataFrame, dict[str, float]]:
    """Checkpoint x horizon rows for one season, and each player's points a
    game that season (the next season's fallback level)."""
    stats = _load(cfg, "player_stats", season)
    stats = stats[(stats["season_type"] == "REG") & stats["position"].isin(SKILL) & (stats["week"] <= LAST)]
    pts = {(str(p), int(w)): float(v or 0.0) for p, w, v in zip(stats["player_id"], stats["week"], stats["fantasy_points_ppr"])}
    sched = pd.read_parquet(cfg.raw_dir / "schedules.parquet")
    sched = sched[(sched["season"] == season) & (sched["game_type"] == "REG") & (sched["week"] <= LAST)]
    team_weeks: dict[str, list[int]] = {}
    for h, a, w in zip(sched["home_team"], sched["away_team"], sched["week"]):
        for t in (h, a):
            team_weeks.setdefault(normalize_team(t), []).append(int(w))
    team_weeks = {t: sorted(set(w)) for t, w in team_weeks.items()}
    rost = _load(cfg, "rosters_weekly", season)
    rost = rost[(rost["game_type"] == "REG") & rost["position"].isin(SKILL) & rost["gsis_id"].notna() & (rost["week"] <= LAST)]
    rost = rost.assign(team=rost["team"].map(normalize_team)).drop_duplicates(["gsis_id", "week"], keep="last")
    inj = _load(cfg, "injuries", season)
    inj = inj[(inj["game_type"] == "REG") & inj["gsis_id"].notna()]
    if "date_modified" in inj.columns:
        inj = inj.sort_values("date_modified")
    inj = inj.drop_duplicates(["gsis_id", "week"], keep="last")
    report = {(str(g), int(w)): (rs, ps, pi if isinstance(pi, str) and pi else qi)
              for g, w, rs, ps, pi, qi in zip(inj["gsis_id"], inj["week"], inj["report_status"], inj["practice_status"],
                                              inj["report_primary_injury"], inj["practice_primary_injury"])}
    rows = []
    ppg_out: dict[str, float] = {}
    for gid, g in rost.groupby("gsis_id"):
        gid = str(gid)
        status = dict(zip(g["week"].astype(int), g["status"]))
        # The 2020-2021 COVID and opt-out lists are reserve lists that no longer exist: not an injury absence.
        if "status_description_abbr" in g.columns:
            for w, a in zip(g["week"].astype(int), g["status_description_abbr"]):
                if a in ("R59", "R62"):
                    status[w] = "COVID"
        team = dict(zip(g["week"].astype(int), g["team"]))
        pos = g["position"].iloc[0]
        played_weeks = sorted(w for (p, w) in pts if p == gid) if False else None
        mine = {w: v for (p, w), v in pts.items() if p == gid} if False else None
        # (filled below from the per-player index)
        rows.append((gid, pos, status, team))
    by_player: dict[str, dict[int, float]] = {}
    for (p, w), v in pts.items():
        by_player.setdefault(p, {})[w] = v
    out = []
    for gid, pos, status, team in rows:
        mine = by_player.get(gid, {})
        if mine:
            ppg_out[gid] = float(np.mean(list(mine.values())))
        weeks = sorted(status)
        if not weeks:
            continue
        # His team games in order (the team he was with that week), played or not.
        games = []
        last_team = None
        for w in range(1, LAST + 1):
            t = team.get(w)
            if t is None and last_team is not None and w not in team_weeks.get(last_team, ()):
                continue                                           # his team's bye (no roster row that week)
            last_team = t or last_team
            if t is None:
                # Not on a roster that week: keep his last team's game if it had one (cut / practice squad ends the spell).
                games.append((w, None, None))
                continue
            if w not in team_weeks.get(t, ()):
                continue                                           # bye
            games.append((w, t, status.get(w)))
        seen_types: set[str] = set()                               # injuries of earlier absences this season
        spell_type = None
        m = 0
        for i, (w, t, st) in enumerate(games):
            if t is None:
                m, spell_type = 0, None
                continue
            if w in mine:
                if m and spell_type:
                    seen_types.add(spell_type)
                m, spell_type = 0, None
                continue
            rep = report.get((gid, w))
            if st == "COVID":
                m, spell_type = 0, None
                continue
            reserve = st == "RES"
            tagged = reserve or (rep is not None and rep[0] in ("Out", "Doubtful", "Questionable"))
            if not tagged and m == 0:
                continue                                           # a healthy scratch, not an injury absence
            m += 1
            if rep is not None and rep[2]:
                spell_type = rm.injury_group(rep[2])
            if not tagged:
                continue                                           # mid-spell without a tag: not a checkpoint the page would see
            before = [v for ww, v in mine.items() if ww < w]
            level = float(np.mean(before)) if len(before) >= 2 else float(prev_ppg.get(gid, np.mean(before) if before else 0.0))
            prev = report.get((gid, games[i - 1][0])) if i > 0 and m > 1 else None
            base = {"season": season, "gid": gid, "pos": pos, "week": w, "m": m, "reserve": int(reserve),
                    "status": "RES" if reserve else rep[0][0], "injury": spell_type or "unknown",
                    "again": int(bool(spell_type) and spell_type in seen_types), "level": level,
                    "practice": rm.practice_tag(rep[1]) if rep is not None else "none",
                    "practice_prev": rm.practice_tag(prev[1]) if prev is not None else "none"}
            # Later team games until he plays, leaves the roster, or the season ends.
            later = games[i + 1:]
            bye_first = int(bool(later) and later[0][0] > w + 1) if later else 0
            for k, (w2, t2, _st2) in enumerate(later, start=1):
                if t2 is None:
                    break
                played = int(w2 in mine)
                out.append({**base, "k": k, "target_week": w2, "bye_first": bye_first, "played": played})
                if played:
                    break
    return pd.DataFrame(out), ppg_out


def panel(cfg, seasons=SEASONS) -> pd.DataFrame:
    cache = cfg.raw_dir / "return_panel.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    prev: dict[str, float] = {}
    try:
        first = _load(cfg, "player_stats", seasons[0] - 1)
        first = first[first["season_type"] == "REG"]
        prev = first.groupby("player_id")["fantasy_points_ppr"].mean().to_dict()
    except Exception:  # noqa: BLE001 - the first season then has no fallback level
        prev = {}
    for s in seasons:
        f, prev = season_panel(cfg, s, prev)
        print(f"{s}: {len(f)} rows, {f[['gid', 'week']].drop_duplicates().shape[0] if len(f) else 0} checkpoints", flush=True)
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    d.to_parquet(cache)
    return d


# ---------------------------------------------------------------------------
# Fit and grade
# ---------------------------------------------------------------------------
def design(d: pd.DataFrame, stale: int) -> list[list[str]]:
    out = []
    for r in d.itertuples():
        c = {"m": r.m, "reserve": r.reserve, "old_rules": int(r.season < 2020), "status": r.status, "practice": r.practice_prev if stale else r.practice,
             "stale": stale, "injury": r.injury, "again": r.again, "level": r.level, "pos": r.pos, "bye_first": r.bye_first}
        out.append(rm.keys(c, int(r.k), int(r.target_week)))
    return out


def fit(train_keys: list[list[str]], y: np.ndarray, w: np.ndarray, C: float = 1.0) -> dict:
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import LogisticRegression

    vocab = sorted({k for ks in train_keys for k in ks})
    idx = {k: i for i, k in enumerate(vocab)}
    rows = np.repeat(np.arange(len(train_keys)), [len(ks) for ks in train_keys])
    cols = np.array([idx[k] for ks in train_keys for k in ks])
    # A reserve-list player back at practice is rare and decisive (he plays
    # next 22% of the time against 3%): its terms are penalised less, or the
    # shrinkage that steadies the rest would flatten it.
    scale = np.array([LIGHT if (k.startswith("pr") and "|r1" in k and "prdnp" not in k) else 1.0 for k in vocab])
    X = csr_matrix((scale[cols], (rows, cols)), shape=(len(train_keys), len(vocab)))
    lr = LogisticRegression(C=C, max_iter=3000)
    lr.fit(X, y, sample_weight=w)
    return {"intercept": float(lr.intercept_[0]), "coef": {k: float(v) for k, v in zip(vocab, lr.coef_[0] * scale)}}


def predict(model: dict, keys: list[list[str]]) -> np.ndarray:
    coef = model["coef"]
    z = np.array([model["intercept"] + sum(coef.get(k, 0.0) for k in ks) for ks in keys])
    return 1.0 / (1.0 + np.exp(-z))


def baseline(d: pd.DataFrame, conf: dict) -> np.ndarray:
    """The chance he plays each row's game under the tables in use now: an
    absence's length by position, given the games already missed; for a
    reserve list, gone-for-the-season by level and games missed, else the
    measured wait."""
    dur, ir = conf["duration"], conf["ir"]
    wait = np.asarray(ir["wait"], dtype=float)
    out = np.zeros(len(d))
    for i, r in enumerate(d.itertuples()):
        k = int(r.k)
        if r.reserve:
            gone = ir["gone"][int(np.digitize(r.level, ir["levels"]))][int(np.digitize(r.m - 1, ir["missed"]))]
            p = (1.0 - gone) * wait / wait.sum()
            before = p[:k].sum()
            out[i] = (p[k] if k < len(p) else 0.0) / max(1.0 - before, 1e-9)
        else:
            pmf = np.asarray(dur.get(r.pos) or dur["WR"], dtype=float)
            pmf = pmf / pmf.sum()
            full = np.concatenate([pmf[:-1], np.full(8, pmf[-1] / 8)])
            length = int(r.m) + k - 1                                  # he is back for this game: the absence ran this long
            tail = full[length - 1:].sum() if length <= len(full) else 0.0
            out[i] = full[length - 1] / tail if tail > 0 else 1.0 / 8.0
    return np.clip(out, 1e-4, 1 - 1e-4)


def logloss(y, p, w=None) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(ll, weights=w))


def grade(d: pd.DataFrame, conf: dict, C: float, seasons, log=print) -> pd.DataFrame:
    y = d["played"].to_numpy(float)
    k0, k1 = design(d, 0), design(d, 1)
    base = baseline(d, conf)
    rows = []
    preds0, preds1 = np.zeros(len(d)), np.zeros(len(d))
    for s in seasons:
        tr, te = (d["season"] != s).to_numpy(), (d["season"] == s).to_numpy()
        keys_tr = [k for k, m in zip(k0, tr) if m] + [k for k, m in zip(k1, tr) if m]
        model = fit(keys_tr, np.concatenate([y[tr], y[tr]]), np.full(2 * tr.sum(), 0.5), C)
        preds0[te] = predict(model, [k for k, m in zip(k0, te) if m])
        preds1[te] = predict(model, [k for k, m in zip(k1, te) if m])
    d = d.assign(base=base, fresh=preds0, stale=preds1)
    return d


def report(d: pd.DataFrame, label: str, mask, log=print) -> None:
    g = d[mask]
    y = g["played"].to_numpy(float)
    per = []
    for s, x in g.groupby("season"):
        yy = x["played"].to_numpy(float)
        per.append((s, len(x), logloss(yy, x["base"]), logloss(yy, x["fresh"]), logloss(yy, x["stale"])))
    wins_f = sum(a[3] < a[2] for a in per)
    wins_s = sum(a[4] < a[2] for a in per)
    log(f"{label}: {len(g)} rows, return rate {y.mean():.3f}; log loss tables {logloss(y, g['base']):.4f}  "
        f"model {logloss(y, g['fresh']):.4f} ({wins_f}/{len(per)} seasons)  a week stale {logloss(y, g['stale']):.4f} ({wins_s}/{len(per)})")
    log("    " + " ".join(f"{s}:{b:.3f}>{f:.3f}/{st:.3f}" for s, _n, b, f, st in per))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--panel-only", action="store_true")
    ap.add_argument("--C", type=float, nargs="+", default=[0.1], help="inverse L2 strength(s) to grade; the last is written")
    args = ap.parse_args()
    cfg = get_config()
    d = panel(cfg)
    print(len(d), "rows")
    if args.panel_only:
        sys.exit(0)
    conf = json.loads((ROOT / "src/streamer/roster/outcome_model.json").read_text())["futures"]
    d = d[d["season"] >= 2016].reset_index(drop=True)
    seasons = sorted(d["season"].unique())
    for C in args.C:
        g = grade(d, conf, C, seasons)
        print(f"== C={C}")
        report(g, "all", np.ones(len(g), bool))
        report(g, "level 6+", (g["level"] >= 6).to_numpy())
        report(g, "level 6+, not on reserve", ((g["level"] >= 6) & (g["reserve"] == 0)).to_numpy())
        report(g, "level 6+, reserve", ((g["level"] >= 6) & (g["reserve"] == 1)).to_numpy())
        report(g, "level 6+, next game only", ((g["level"] >= 6) & (g["k"] == 1)).to_numpy())
        rp = g[(g["reserve"] == 1) & g["practice"].isin(["limited", "full"]) & (g["season"] >= 2020) & (g["k"] == 1)]
        print(f"    reserve, back at practice, next game: {len(rp)} rows, played {rp['played'].mean():.3f}, model {rp['fresh'].mean():.3f}, tables {rp['base'].mean():.3f}")
    g.to_parquet(cfg.raw_dir / "return_graded.parquet")
    if args.write:
        C = args.C[-1]
        y = d["played"].to_numpy(float)
        model = fit(design(d, 0) + design(d, 1), np.concatenate([y, y]), np.full(2 * len(y), 0.5), C)
        g = grade(d, conf, C, seasons)
        hi = g[g["level"] >= 6]
        yy = hi["played"].to_numpy(float)
        out = {"fitted_on": f"{seasons[0]}-{seasons[-1]} injury reports, reserve lists and games played (nflverse)",
               "C": C, "intercept": round(model["intercept"], 4),
               "coef": {k: round(v, 4) for k, v in model["coef"].items()},
               "evidence": {"rows": int(len(d)), "checkpoints": int(d[["season", "gid", "week"]].drop_duplicates().shape[0]),
                            "held_out_log_loss_level6": {"tables": round(logloss(yy, hi["base"]), 4),
                                                         "model": round(logloss(yy, hi["fresh"]), 4),
                                                         "model_week_old_practice": round(logloss(yy, hi["stale"]), 4)}}}
        path = ROOT / "src/streamer/roster/return_model.json"
        path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(out['coef'])} terms)")
