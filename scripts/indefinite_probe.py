"""How long "out indefinitely" lasts, against an ordinary "Out" designation.

    python scripts/indefinite_probe.py

For every skill player who was listed Out on an official injury report in
2025 or 2026, reads his ESPN player news (public; up to 50 items) and finds
reports that he is out "indefinitely" -- a sentence naming him (or an item
about him) with the word, not hedged ("could", "might", "if"). For each, the
team games he then missed before playing again, from the weekly game logs,
against the same count from every first Out designation on the injury report
(what the season simulation draws from today when no timeline is reported).
Prints counts and distributions only. Run as the ``indefinite-probe`` job.
DECISIONS.md, "Out indefinitely".
"""

from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SEASONS = (2025, 2026)
SKILL = ("QB", "RB", "WR", "TE")
WORD = re.compile(r"\bindefinite(?:ly)?\b", re.I)


def norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(name).lower()))


def team_games(sch: pd.DataFrame) -> dict[tuple[int, str], list[tuple[int, date]]]:
    """(season, team) -> [(week, game date)] in order, regular season."""
    out = defaultdict(list)
    for r in sch.itertuples():
        if getattr(r, "game_type", "REG") != "REG" or not isinstance(r.gameday, str):
            continue
        d = date.fromisoformat(r.gameday[:10])
        out[(int(r.season), r.home_team)].append((int(r.week), d))
        out[(int(r.season), r.away_team)].append((int(r.week), d))
    return {k: sorted(v) for k, v in out.items()}


def missed(games: list[tuple[int, date]], played: set[int], from_week: int, last_week: int) -> tuple[int, bool]:
    """Team games from ``from_week`` on that he missed before his next game;
    and whether he came back (False: the season ended, or has not yet)."""
    n = 0
    for w, _d in games:
        if w < from_week or w > last_week:
            continue
        if w in played:
            return n, True
        n += 1
    return n, False


def summary(label: str, rows: list[tuple[int, bool]]) -> None:
    if not rows:
        print(f"  {label}: none")
        return
    m = np.array([r[0] for r in rows])
    back = np.array([r[1] for r in rows])
    print(f"  {label}: n={len(rows)}; came back {back.mean():.0%}; games missed median {np.median(m):.0f}, "
          f"mean {m.mean():.1f}; missed 2+ {np.mean(m >= 2):.0%}, 4+ {np.mean(m >= 4):.0%}, "
          f"6+ {np.mean(m >= 6):.0%}, 8+ {np.mean(m >= 8):.0%}")


def main() -> int:
    from streamer.config import get_config
    from streamer.data import news
    from streamer.roster.projections import load_history

    cfg = get_config().for_profile("espn")
    hist = load_history(cfg)
    played = defaultdict(set)
    for r in hist[hist["season"].isin(SEASONS)][["season", "week", "player_id"]].itertuples(index=False):
        played[(int(r.season), str(r.player_id))].add(int(r.week))
    sch = pd.read_parquet(cfg.raw_dir / "schedules.parquet")
    games = team_games(sch[sch["season"].isin(SEASONS)])
    done = {s: int(hist[hist["season"] == s]["week"].max()) for s in SEASONS}

    inj = pd.concat([pd.read_parquet(cfg.raw_dir / f"injuries_{s}.parquet") for s in SEASONS], ignore_index=True)
    inj = inj[inj["position"].isin(SKILL) & (inj.get("game_type", "REG") == "REG")]
    out = inj[inj["report_status"] == "Out"]

    # Baseline: every first Out designation of an absence (not Out the week before).
    flagged = {(int(r.season), str(r.gsis_id), int(r.week)) for r in out.itertuples()}
    base = []
    for r in out.itertuples():
        s, g, w = int(r.season), str(r.gsis_id), int(r.week)
        if (s, g, w - 1) in flagged:
            continue
        tg = games.get((s, r.team))
        if tg:
            base.append(missed(tg, played[(s, g)], w, done[s]))

    # The news: players listed Out at least once, by ESPN id from the latest snapshot.
    from streamer.league.store import load_snapshot

    snap = load_snapshot(cfg)
    espn = {norm(p.name): p.player_id for p in snap.all_players() if p.position in SKILL}
    who = out.drop_duplicates(["gsis_id"])[["gsis_id", "full_name", "team", "position"]]
    found, fetched, errors = [], 0, 0
    for r in who.itertuples():
        eid = espn.get(norm(r.full_name))
        if not eid:
            continue
        try:
            resp = news._get(news.ESPN_PLAYER_NEWS, params={"playerId": eid, "limit": 50})
            resp.raise_for_status()
            items = resp.json().get("feed") or []
            fetched += 1
        except Exception:  # noqa: BLE001 - a probe: count and go on
            errors += 1
            continue
        time.sleep(0.1)
        last = str(r.full_name).split()[-1]
        seen_dates = set()
        for it in items:
            when = news._when(it)
            if when is None:
                continue
            head = str(it.get("headline") or "")
            body = str(it.get("story") or it.get("description") or "")
            about = it.get("type") in news.PLAYER_TYPES
            text = f"{head}. {body}" if about else news._plain(f"{head}. {body}")
            hit = False
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                named = re.search(rf"\b{re.escape(last)}\b", sentence)
                if not (named or (about and re.match(r"\s*(He|His|The injury)\b", sentence))):
                    continue
                if WORD.search(sentence) and not news._HEDGE.search(sentence) and not news._PAST.search(sentence):
                    hit = True
                    break
            d = when.date()
            if not hit or d in seen_dates:
                continue
            seen_dates.add(d)
            # The season and week: his team's next game on or after the report.
            for s in SEASONS:
                tg = games.get((s, r.team)) or []
                nxt = next((w for w, gd in tg if gd >= d), None)
                if nxt is not None and tg and d >= tg[0][1].replace(month=8, day=1):
                    found.append((s, str(r.gsis_id), r.position, nxt, missed(tg, played[(s, str(r.gsis_id))], nxt, done[s])))
                    break
    # One report per absence: the first "indefinitely" for a player in a season.
    first = {}
    for s, g, pos, w, res in sorted(found, key=lambda x: (x[0], x[1], x[3])):
        first.setdefault((s, g), (pos, w, res))
    print(f"news fetched for {fetched} players ({errors} failed); 'out indefinitely' reports: {len(first)} absences")
    summary("first Out designation (what the model draws from)", base)
    summary("reported out indefinitely", [v[2] for v in first.values()])
    for s in SEASONS:
        summary(f"  {s} only, indefinitely", [v[2] for (ss, _g), v in first.items() if ss == s])
    for pos in SKILL:
        summary(f"  {pos}, indefinitely", [v[2] for v in first.values() if v[0] == pos])
    print(f"(2026 absences still running are counted as not back; games counted through week {done[2026]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
