"""Do our fantasy points match the platform's? Checked every refresh.

Yahoo's quarterbacks were scored on the standard sheet (4 a touchdown pass,
-2 an interception) all season while the league pays 6 and -1, because the
sheet was assumed rather than checked. Where the platform reports each
player's season average (ESPN does; the Yahoo sync does not yet), it is
compared with ours on the league's sheet, position by position, and a
mismatch is logged as a warning naming the position and the typical gap.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..league.model import LeagueSnapshot

log = logging.getLogger(__name__)

#: A position whose typical gap is above this (points a game) is flagged.
TOLERANCE = 0.3
#: Players needed at a position before it is judged.
MIN_PLAYERS = 5


def flatten(profile: dict | None) -> dict[str, float]:
    """A profile's offensive, kicker and D/ST sheets in the keys the platform
    reader uses (:func:`streamer.league.yahoo_web.parse_scoring`): offense keys
    as they are, ``k.*`` per Yahoo's kicking distances, ``dst.*`` with each
    points-allowed band as ``dst.pa_<low>_<high|plus>``."""
    p = profile or {}
    out = {k: float(v) for k, v in (p.get("offense_scoring") or {}).items()}
    k = p.get("kicker_scoring") or {}
    if k:
        for yahoo, ours in (("fg_0_19", "fg_0_39"), ("fg_20_29", "fg_0_39"), ("fg_30_39", "fg_0_39"),
                            ("fg_40_49", "fg_40_49"), ("fg_50_plus", "fg_50_59"),
                            ("fg_missed_0_19", "fg_missed_0_39"), ("fg_missed_20_29", "fg_missed_0_39"),
                            ("fg_missed_30_39", "fg_missed_0_39"), ("fg_missed_40_49", "fg_missed_40_49"),
                            ("fg_missed_50_plus", "fg_missed_50_59"), ("pat_made", "pat_made"),
                            ("pat_missed", "pat_missed")):
            if ours in k:
                out[f"k.{yahoo}"] = float(k[ours])
    d = p.get("dst_scoring") or {}
    for key in ("sack", "interception", "fumble_recovery", "defensive_td", "safety", "blocked_kick", "return_td",
                "fourth_down_stop", "extra_point_returned"):
        if key in d:
            out[f"dst.{key}"] = float(d[key])
    for lo, hi, pts in d.get("points_allowed_tiers") or []:
        out[f"dst.pa_{lo}_{'plus' if hi is None else hi}"] = float(pts)
    return out


def check_sheet(snapshot: LeagueSnapshot, profile: dict | None) -> list[str]:
    """The platform's own scoring sheet (read from the league's settings at
    sync: ``rules["platform_scoring"]`` from Yahoo's settings page, or
    ``extra["platform_scoring"]``) against the profile's sheets in
    config.yaml (:func:`flatten`): every stat both have that differs, as a
    warning. Returns the differences (empty: they agree, or nothing to compare)."""
    theirs = ((getattr(snapshot, "rules", None) or {}).get("platform_scoring")
              or (getattr(snapshot, "extra", None) or {}).get("platform_scoring") or {})
    if not theirs:
        return []
    ours = flatten(profile) if profile and any(k in profile for k in ("offense_scoring", "kicker_scoring",
                                                                       "dst_scoring")) else dict(profile or {})
    diffs = [f"{k}: platform {v:g}, config {float(ours[k]):g}" for k, v in sorted(theirs.items())
             if k in ours and abs(float(ours[k]) - float(v)) > 1e-9]
    if diffs:
        log.warning("%s: the platform's scoring differs from offense_scoring in config.yaml -- %s",
                    snapshot.profile, "; ".join(diffs))
    else:
        log.info("%s: scoring sheet matches the platform's (%d stats checked)", snapshot.profile,
                 sum(1 for k in theirs if k in ours))
    return diffs


def check(snapshot: LeagueSnapshot, history: pd.DataFrame) -> dict[str, float]:
    """Position -> median |platform season average - ours| over players with
    three or more games; logs a warning for any above :data:`TOLERANCE`."""
    if history is None or history.empty:
        return {}
    h = history[history["season"] == int(snapshot.season)]
    g = h.groupby("player_id")["fantasy_points_ppr"].agg(["mean", "count"])
    gaps: dict[str, list[float]] = {}
    for p in snapshot.all_players():
        nid = getattr(p, "nfl_id", None)
        # ESPN reports 0.0 for many rostered players (Josh Allen, Jahmyr
        # Gibbs): not an average, a missing one.
        if p.season_avg is None or float(p.season_avg) == 0.0 or not nid or str(nid) not in g.index:
            continue
        mean, n = g.loc[str(nid)]
        if n >= 3:
            gaps.setdefault(p.position, []).append(abs(float(p.season_avg) - float(mean)))
    out = {pos: float(pd.Series(v).median()) for pos, v in gaps.items() if len(v) >= MIN_PLAYERS}
    bad = {pos: v for pos, v in out.items() if v > TOLERANCE}
    if bad:
        log.warning("%s: fantasy points differ from the platform's: %s -- check offense_scoring in config.yaml",
                    snapshot.profile, ", ".join(f"{k} typically {v:.1f} a game" for k, v in sorted(bad.items())))
    elif out:
        log.info("%s: fantasy points match the platform's (%s)", snapshot.profile,
                 ", ".join(f"{k} {v:.2f}" for k, v in sorted(out.items())))
    return out
