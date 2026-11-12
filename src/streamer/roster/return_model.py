"""When a hurt player is back: the chance he plays each coming game.

(Stub while the model is fitted: the injury and practice vocabularies.)
"""

from __future__ import annotations

import re

#: Injury groups, by a word in the report's injury ("Right Hamstring" -> hamstring).
GROUPS = (
    ("achilles", r"achilles"),
    ("acl", r"\bacl\b"),
    ("hamstring", r"hamstring"),
    ("concussion", r"concussion|head\b"),
    ("ankle", r"ankle"),
    ("knee", r"knee|mcl|pcl|meniscus"),
    ("foot", r"foot|toe|heel|plantar"),
    ("calf", r"calf|shin|tibia|fibula|lower leg"),
    ("thigh", r"quad|thigh|groin|hip|glute|adductor"),
    ("shoulder", r"shoulder|collarbone|clavicle|pectoral|\bpec\b"),
    ("hand", r"hand|thumb|finger|wrist|forearm|elbow|\barm\b|tricep|bicep"),
    ("core", r"rib|chest|oblique|abdom|core|back\b|neck|spine|stinger"),
    ("illness", r"illness|covid|personal|not injury|rest\b|coach"),
)
_GROUPS = [(name, re.compile(pat, re.I)) for name, pat in GROUPS]


def injury_group(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    for name, pat in _GROUPS:
        if pat.search(text):
            return name
    return "other"


def practice_tag(text) -> str:
    t = str(text or "")
    return "dnp" if "Did Not" in t else "limited" if "Limited" in t else "full" if "Full" in t else "none"


def _bucket(v: float, edges) -> int:
    n = 0
    for e in edges:
        if v >= e:
            n += 1
    return n


T_EDGES = (2, 3, 4, 5, 7, 9, 12)          # games missed before the game in question
K_EDGES = (2, 3, 4, 6)                    # games ahead of what is known
LEVEL_EDGES = (3.0, 6.0, 10.0, 15.0)
WEEK_EDGES = (9, 13, 16)


def keys(c: dict, k: int, target_week: int) -> list[str]:
    """The active terms for one coming game: ``c`` is what is known at the
    last game he missed (``m`` games missed in a row through it, ``reserve``,
    ``status`` O/D/Q, ``practice`` and whether it is a week old (``stale``),
    ``injury``, ``again``, ``level``, ``pos``, ``bye_first``), ``k`` how many
    team games after it this one is."""
    # A reserve list before 2020 let two players a team back all season: its
    # own terms, so today's read comes from the seasons under today's rules.
    res = ("1x" if c.get("old_rules") else "1") if int(c.get("reserve", 0)) else 0
    t = int(c["m"]) + k - 1
    tb, kb = _bucket(t, T_EDGES), _bucket(k, K_EDGES)
    kc = min(k, 3)
    tc = 0 if t <= 2 else 1 if t <= 5 else 2
    out = [f"t{tb}|r{res}", f"k{kb}|r{res}", f"pos{c.get('pos', 'WR')}|r{res}",
           f"lvl{_bucket(float(c.get('level') or 0.0), LEVEL_EDGES)}|r{res}",
           f"wk{_bucket(int(target_week), WEEK_EDGES)}|r{res}",
           f"inj{c.get('injury', 'unknown')}|t{tc}"]
    if c.get("bye_first"):
        out.append(f"bye|k{min(k, 2)}|r{res}")
    if res:
        if c.get("practice", "none") != "none":           # back at practice off a reserve list: his window is open
            out.append(f"pr{c['practice']}|s{int(c.get('stale', 0))}|r{res}|k{kc}")
    else:
        out += [f"st{c.get('status', 'O')}|k{kc}",
                f"pr{c.get('practice', 'none')}|s{int(c.get('stale', 0))}|k{kc}",
                f"again{int(c.get('again', 0))}|k{kc}"]
    return out


# ---------------------------------------------------------------------------
# The fitted model at run time
# ---------------------------------------------------------------------------
_MODEL: dict | None = None
_LOADED = False
#: Games "missed" by a player who is not back in the weeks simulated.
GONE = 99
STATUS = {"Out": "O", "Doubtful": "D", "Questionable": "Q"}


def load() -> dict | None:
    """The fitted terms (``return_model.json``, written by
    ``scripts/fit_return_model.py``), or None before it has been fitted."""
    global _MODEL, _LOADED
    if not _LOADED:
        import json
        from pathlib import Path

        path = Path(__file__).with_name("return_model.json")
        _MODEL = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        _LOADED = True
    return _MODEL


def hazards(c: dict, target_weeks, model: dict | None = None):
    """For each coming team game (its NFL week, in order): the chance he
    plays it given he has not played since the checkpoint ``c`` describes."""
    import numpy as np

    model = model or load()
    coef, b0 = model["coef"], float(model["intercept"])
    z = np.array([b0 + sum(coef.get(key, 0.0) for key in keys(c, k, int(w)))
                  for k, w in enumerate(target_weeks, start=1)], dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def back_by(c: dict, target_weeks, model: dict | None = None):
    """The chance he has played by each coming game (cumulative)."""
    import numpy as np

    h = hazards(c, target_weeks, model)
    return 1.0 - np.cumprod(1.0 - h)


def draw(c: dict, target_weeks, n: int, rng, model: dict | None = None):
    """Per simulated season: how many of the coming games he misses before
    he is back (0: he plays the first), :data:`GONE` when he is not back in
    any of them."""
    import numpy as np

    if not len(target_weeks):
        return np.full(n, GONE)
    cum = back_by(c, target_weeks, model)
    idx = np.searchsorted(cum, rng.random(n), side="right")      # first game by which he is back
    return np.where(idx >= len(cum), GONE, idx)


def contexts(inj, team_weeks: dict[str, list[int]], played_weeks: dict[str, set[int]], week: int,
             team_of: dict[str, str]) -> dict[str, dict]:
    """nflverse id -> what this season's injury reports say about a player,
    as of ``week`` (no later report is read): the injury's group, his last
    practice and the report's status with the week they are from, and
    whether he already had an absence with the same injury this season."""
    out: dict[str, dict] = {}
    if inj is None or len(inj) == 0 or "gsis_id" not in inj.columns:
        return out
    inj = inj[inj["gsis_id"].notna() & (inj["week"] <= int(week))]
    if "game_type" in inj.columns:
        inj = inj[inj["game_type"] == "REG"]
    if "date_modified" in inj.columns:
        inj = inj.sort_values("date_modified")
    inj = inj.drop_duplicates(["gsis_id", "week"], keep="last")
    for gid, g in inj.groupby("gsis_id"):
        gid = str(gid)
        rows = {int(r.week): r for r in g.itertuples()}
        played = played_weeks.get(gid, set())
        seen: set[str] = set()
        cur, m = None, 0
        for w in team_weeks.get(team_of.get(gid, ""), []):
            if w >= int(week):
                break
            r = rows.get(w)
            if w in played:
                if m and cur:
                    seen.add(cur)
                cur, m = None, 0
            elif m or (r is not None and r.report_status in STATUS):
                m += 1
                kind = _injury_of(r)
                cur = kind if kind != "unknown" else cur
        last_week = max(rows)
        if last_week < int(week) - 2:
            # Off the report for weeks: nothing current to read, but an earlier absence still counts.
            out[gid] = {"injury": cur or "unknown", "again": int(bool(cur) and cur in seen), "report_week": None}
            continue
        r = rows[last_week]
        kind = _injury_of(r)
        kind = kind if kind != "unknown" else (cur or "unknown")
        out[gid] = {"injury": kind, "again": int(kind != "unknown" and kind in seen), "report_week": last_week,
                    "practice": practice_tag(r.practice_status), "status": STATUS.get(r.report_status)}
    return out


def _injury_of(r) -> str:
    if r is None:
        return "unknown"
    for col in ("report_primary_injury", "practice_primary_injury"):
        kind = injury_group(getattr(r, col, None))
        if kind != "unknown":
            return kind
    return "unknown"


def checkpoint(p, week: int, ahead: list[int], sat: bool = True, status: str | None = None) -> tuple[dict, list[int], int]:
    """What the model is asked for a player who is out: ``(context, the team
    games it is asked about, games he misses for certain first)``.

    ``ahead``: his team's coming games from this week on (NFL weeks, byes
    left out). A player tagged out with a game this week misses it -- that
    game is the checkpoint, and the model is asked about the ones after. On
    his bye there is no game to miss: the checkpoint is the last game he
    missed, and the first game back from the bye is in question, with the
    week of rest the bye gave him."""
    from ..league.model import LONG_TERM_OUT_STATUSES

    info = getattr(p, "injury_ctx", None) or {}
    missed = int(getattr(p, "games_missed", 0) or 0)
    on_bye = bool(getattr(p, "on_bye", False))
    reserve = int(p.status in LONG_TERM_OUT_STATUSES)
    if on_bye and (missed >= 1 or not reserve):
        certain, targets, m, bye_first = 0, list(ahead), max(missed, 1), 1
    else:
        certain, targets, m = 1, list(ahead[1:]), missed + 1
        bye_first = int(bool(targets) and bool(ahead) and targets[0] > ahead[0] + 1)
    rw = info.get("report_week")
    fresh = rw is not None and (rw == int(week) or (on_bye and rw >= int(week) - 1))
    known = rw is not None and rw >= int(week) - (2 if on_bye else 1)
    c = {"m": m, "reserve": reserve, "pos": p.position, "level": float(p.ros_value or 0.0),
         "injury": info.get("injury", "unknown"), "again": int(info.get("again", 0)),
         "practice": info.get("practice", "none") if known else "none", "stale": 0 if fresh else 1,
         "status": status or (info.get("status") if fresh and info.get("status") else "O"), "bye_first": bye_first}
    return c, targets, certain
