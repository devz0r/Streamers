"""Live scores on the page, read in the browser while the games are on.

The page cannot run the simulation, but it does not need to: it carries a
slice of the draws that priced this week's matchup (each starter's score, on
both sides, draw by draw) and the league's scoring rules. In the browser,
``static/live.js`` reads ESPN's public NFL scoreboard and box scores (they
allow it: ``Access-Control-Allow-Origin: *``), turns each starter's stat line
into fantasy points, and rescores the draws: a finished game counts what he
scored, one not started counts the draw, and one in progress counts what he
has so far plus the draw's share of the time left. P(win) is the share of
draws you finish ahead. No workflow runs; it updates every minute while the
page is open. DECISIONS.md, "Live scores in the browser".
"""

from __future__ import annotations

import base64
import json
import logging

import numpy as np

from ..config import Config
from ..league.model import LeagueSnapshot, PlayerRow
from ..teams import normalize_team

log = logging.getLogger(__name__)

#: Draws carried to the page for the live P(win): enough to read it to about a
#: point (the standard error at 50% is 1.1 points), small enough for a phone.
LIVE_SIMS = 2000

#: ESPN's abbreviations where they differ from ours.
ESPN_TEAMS = {"LAR": "LA", "WSH": "WAS"}


def espn_ids(snapshot: LeagueSnapshot, cfg: Config, players: list[PlayerRow]) -> dict[str, str]:
    """League player id -> ESPN athlete id. ESPN's league ids are ESPN's
    athlete ids; anyone else is matched through nflverse's roster table
    (gsis id -> espn id). Missing ids fall back to the name in the browser."""
    out: dict[str, str] = {}
    if snapshot.platform == "espn":
        for p in players:
            if p.player_id.isdigit():
                out[p.player_id] = p.player_id
        return out
    try:
        import pandas as pd

        path = cfg.raw_dir / f"rosters_weekly_{int(snapshot.season)}.parquet"
        if path.exists():
            r = pd.read_parquet(path, columns=["gsis_id", "espn_id"])
        else:
            import nflreadpy

            r = nflreadpy.load_rosters_weekly(seasons=[int(snapshot.season)]).to_pandas()[["gsis_id", "espn_id"]]
        r = r.dropna().drop_duplicates("gsis_id", keep="last")
        by_gsis = {str(g): str(e).split(".")[0] for g, e in zip(r["gsis_id"], r["espn_id"])}
    except Exception as exc:  # noqa: BLE001 - names still match in the browser
        log.warning("no ESPN ids for the live scores: %s", exc)
        return out
    for p in players:
        nid = getattr(p, "nfl_id", None)
        if nid and str(nid) in by_gsis:
            out[p.player_id] = by_gsis[str(nid)]
    return out


def _b64(values: np.ndarray) -> str:
    """Scores in tenths of a point as little-endian int16, base64 (as the
    lineup lab carries them)."""
    return base64.b64encode(np.round(values * 10).clip(-32000, 32000).astype("<i2").tobytes()).decode()


def _row(p: PlayerRow, slot: str, eid: str | None) -> dict:
    return {"n": p.name, "pos": p.position, "tm": normalize_team(p.team) or (p.team or ""),
            "eid": eid, "slot": slot, "m": round(float(p.week_value), 2)}


def payload(snapshot: LeagueSnapshot, opt, cfg: Config) -> dict | None:
    """What the live block needs: both sides' starters (your lineup as set,
    else the recommended one; your opponent's as the optimiser faced it), the
    league's scoring, and the draws, player by player."""
    if opt is None or opt.samples is None or opt.opp_player_samples is None or not opt.opp_ids:
        return None
    mine = opt.current or opt.best_win
    me = [(slot, p) for slot, ps in mine.starters.items() for p in ps]
    by_opp = {p.player_id: (slot, p) for slot, ps in opt.opponent.starters.items() for p in ps} \
        if opt.opponent is not None else {}
    them = [by_opp[i] for i in opt.opp_ids if i in by_opp]
    col = {pid: k for k, pid in enumerate(opt.sample_ids)}
    if not me or not them or any(p.player_id not in col for _s, p in me):
        return None
    n = min(LIVE_SIMS, int(opt.samples.shape[0]), int(opt.opp_player_samples.shape[0]))
    mine_s = np.stack([opt.samples[:n, col[p.player_id]] for _s, p in me])           # (players, draws)
    theirs_s = np.asarray(opt.opp_player_samples[:n, :len(them)]).T
    ids = espn_ids(snapshot, cfg, [p for _s, p in me + them])
    prof = cfg.profiles[cfg.profile]
    return {
        "season": int(snapshot.season), "week": int(snapshot.week), "n": n,
        "sc": {"off": prof.get("offense_scoring", {}), "k": prof.get("kicker_scoring", {}),
               "dst": prof.get("dst_scoring", {})},
        "teams": ESPN_TEAMS,
        "me": [_row(p, s, ids.get(p.player_id)) for s, p in me],
        "opp": [_row(p, s, ids.get(p.player_id)) for s, p in them],
        "S": _b64(mine_s), "O": _b64(theirs_s),
    }


def block(snapshot: LeagueSnapshot, opt, cfg: Config, opponent_name: str = "") -> str:
    """The live card: empty until the script has scores to show."""
    data = payload(snapshot, opt, cfg)
    if data is None:
        return ""
    data["opp_name"] = opponent_name
    uid = f"live-{snapshot.profile}"
    body = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return (f'<div class="card live-card" id="{uid}" hidden>'
            f'<script type="application/json" class="live-data">{body}</script>'
            '<div class="live-out"></div></div>'
            f"<script>window.StreamerLive&&window.StreamerLive.mount('{uid}')</script>")
