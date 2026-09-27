"""The "My team" panel for the published page.

Rendered per profile when a league snapshot exists; silently omitted when it
does not, so the streaming page never depends on a sync having run.
"""

from __future__ import annotations

import html

from ..config import Config
from ..league.model import LeagueSnapshot, short_status
from .matchup import MatchupReport
from .waivers import Move, drop_watch, stashes


def _e(v: object) -> str:
    return html.escape("" if v is None else str(v))


def _pct(v: float | None) -> str:
    return "--" if v is None else f"{v * 100:.0f}%"


def render_my_team(
    snapshot: LeagueSnapshot, report: MatchupReport, moves: list[Move], cfg: Config
) -> str:
    """HTML for the panel; empty string if there is nothing to show."""
    me = snapshot.my_team
    opt = report.optimisation
    parts: list[str] = [
        "<h2>My team</h2>",
        f'<p class="sub">{_e(snapshot.league_name or snapshot.platform.upper())} &middot; '
        f"{_e(me.name)} ({me.wins}-{me.losses}) &middot; synced "
        f"{_e(snapshot.synced_at[:16].replace('T', ' '))} UTC</p>",
    ]

    # -- matchup ---------------------------------------------------------
    cur = report.current_win_probability
    gain = "" if cur is None else f" (from {_pct(cur)} as set)"
    opp_line = ""
    if opt.opponent is not None:
        opp_line = (f"<span>you {opt.best_win.expected:.1f} &plusmn; {opt.best_win.sd:.0f}</span>"
                    f"<span>them {opt.opponent.expected:.1f} &plusmn; {opt.opponent.sd:.0f}</span>")
    parts.append(
        '<div class="card"><div class="row"><div class="rank">&#9878;</div>'
        f'<div><span class="name">vs {_e(report.opponent_name or "?")}</span> '
        f'<span class="opp">{_e(report.verdict())}</span></div>'
        f'<div class="pts">{_pct(report.win_probability)}</div></div>'
        f'<div class="meta"><span>P(win) with the lineup below{_e(gain)}</span>{opp_line}</div>'
        "</div>"
    )

    # -- lineup ----------------------------------------------------------
    changed = {p.player_id for _s, _b, p in opt.changes}
    has_vegas = any(p.vegas_points is not None for p in snapshot.my_team.roster)
    rows = []
    for slot, p in opt.best_win.flat():
        flag = ""
        if p.player_id in changed:
            flag = ' <span class="hold-tag">start</span>'
        elif p.is_questionable:
            flag = f' <span class="opp">{_e(short_status(p.status))}</span>'
        vegas = ""
        if has_vegas:
            vegas = (f"<td>{p.vegas_points:.1f}</td>" if p.vegas_points is not None
                     else '<td class="opp">--</td>')
        lo, hi = opt.ranges.get(p.player_id, (None, None))
        spread = (f"{lo:.0f}&ndash;{hi:.0f}" if lo is not None
                  else f"&plusmn;{(p.projection_sd or 0):.0f}")
        rows.append(
            f"<tr><td>{_e(slot)}</td><td class='unit'>{_e(p.name)}{flag}</td>"
            f"<td>{_e(p.position)}</td><td>{_e(p.team or '--')}</td>"
            f"<td>{(p.projection or 0):.1f}</td>{vegas}"
            f"<td>{spread}</td></tr>"
        )
    vegas_head = "<th>Vegas</th>" if has_vegas else ""
    parts.append(
        "<h3>Recommended lineup</h3>"
        '<div class="scroll"><table><thead><tr><th>Slot</th><th class="unit">Player</th>'
        f"<th>Pos</th><th>Tm</th><th>Proj</th>{vegas_head}<th>Range</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
    if opt.changes:
        items = []
        for slot, benched, started in opt.changes:
            if benched is not None:
                items.append(f"<li><strong>{_e(slot)}</strong>: start {_e(started.name)} over {_e(benched.name)}</li>")
            else:
                items.append(f"<li><strong>{_e(slot)}</strong>: start {_e(started.name)}</li>")
        parts.append(f'<p class="sub">Changes from your set lineup:</p><ul class="sub">{"".join(items)}</ul>')
    else:
        parts.append('<p class="sub">Your set lineup is already the recommended one.</p>')
    parts.append(
        '<p class="sub">Range is the middle 70% of simulated outcomes (15th to 85th '
        "percentile), with teammates and opponents correlated as they are on the field.</p>"
    )
    if opt.reasons:
        items = "".join(f"<li>{_e(r)}</li>" for r in opt.reasons)
        parts.append(f'<p class="sub">Not simply the highest projections, because:</p><ul class="sub">{items}</ul>')

    if has_vegas:
        parts.append(_vegas_section(snapshot, opt, cfg, getattr(snapshot, "_vegas_report", None)))

    for note in report.notes:
        parts.append(f'<p class="sub">Note: {_e(note)}.</p>')

    # -- waivers ---------------------------------------------------------
    parts.append("<h3>Waiver moves</h3>")
    if not moves:
        parts.append('<p class="sub">Nothing on the wire clears the bar this week.</p>')
    else:
        cards = []
        for m in moves[:6]:
            drop = f" &middot; drop {_e(m.drop.name)}" if m.drop else ""
            cards.append(
                '<div class="card"><div class="row">'
                f'<div class="rank">{_e(m.tag[:1].upper())}</div>'
                f'<div><span class="name">{_e(m.add.name)}</span> '
                f'<span class="opp">{_e(m.add.position)} {_e(m.add.team or "")}{drop}</span></div>'
                f'<div class="pts">+{m.score:.1f}</div></div>'
                f'<div class="why">{_e(m.reason)}</div></div>'
            )
        parts.append("".join(cards))

    tickets = stashes(snapshot, moves, n=3)
    if tickets:
        cards = []
        for t in tickets:
            p = t.player
            cards.append(
                '<div class="card"><div class="row">'
                '<div class="rank">&#8599;</div>'
                f'<div><span class="name">{_e(p.name)}</span> '
                f'<span class="opp">{_e(p.position)} {_e(p.team or "")}</span></div>'
                f'<div class="pts">{t.ceiling:.1f}</div></div>'
                f'<div class="why">{(p.ros_value or 0):.1f} a game now; {_e("; ".join(t.reasons))}</div></div>'
            )
        parts.append(
            "<h3>Upside stashes</h3>"
            '<p class="sub">Not enough projected value to clear the bar yet, but the most room '
            "to grow. The number is what he could be worth a game in a month if things break "
            "his way (85th percentile of how projections move). Worth a bench spot you would "
            "otherwise waste.</p>" + "".join(cards)
        )

    watch = drop_watch(snapshot, n=3)
    if watch:
        parts.append(
            '<p class="sub">Drop watch: '
            + ", ".join(f"{_e(p.name)} ({(p.projection or 0):.1f})" for p in watch)
            + "</p>"
        )
    return "".join(parts)


def _vegas_section(snapshot, opt, cfg: Config, report=None) -> str:
    """What the sportsbooks would start, and where they disagree with us."""
    from .vegas import disagreements, vegas_lineup

    market = vegas_lineup(snapshot, cfg)
    ours = opt.best_win.player_ids
    theirs = market.player_ids
    decisive = set(ours) | set(theirs)
    bits = []
    # Books post a game's props a day or two out, so early in the week most of
    # the slate has no prices yet. Saying so beats a column of dashes.
    if report is not None and report.eligible:
        covered = report.matched_startable
        if covered < report.eligible:
            bits.append(
                f"The books have priced {covered} of your {report.eligible} startable "
                "players so far; the rest fill in as their games approach."
            )
    swaps = [p for _s, p in market.flat() if p.player_id not in ours]
    benched = [p for _s, p in opt.best_win.flat() if p.player_id not in theirs]
    if swaps:
        pairs = ", ".join(
            f"{_e(a.name)} over {_e(b.name)}" for a, b in zip(swaps, benched)
        )
        bits.append(f"The sportsbook numbers would start {pairs}.")
    else:
        bits.append("The sportsbook numbers would start the same lineup.")
    gaps = disagreements(snapshot, 3, among=decisive)
    if gaps:
        detail = ", ".join(
            f"{_e(p.name)} {'+' if d > 0 else ''}{d:.1f}" for p, d in gaps
        )
        bits.append(f"Biggest disagreements (market minus ours): {detail}.")
    credits = getattr(snapshot, "_vegas_credits", None)
    if credits is not None:
        bits.append(f"Odds API credits left: {credits}.")
    return f'<p class="sub">{" ".join(bits)}</p>'


def render_sync_failure(status: dict, cfg: Config, stale_week: int | None = None) -> str:
    """A short note for a league whose sync did not succeed.

    Silence reads as "nothing to report"; a league that could not be reached
    has plenty to report, and the reason is what tells you how to fix it.
    """
    platform = str(status.get("platform") or cfg.profile).upper()
    when = str(status.get("at") or "")[:16].replace("T", " ")
    reason = str(status.get("error") or "no reason recorded")
    if stale_week is not None:
        lead = (f"Your {platform} league could not be synced for week "
                f"{status.get('week')}, so the panel below is week {stale_week}.")
    else:
        lead = f"Your {platform} league could not be synced, so there is no team panel."
    return (
        '<h2>My team</h2>'
        f'<div class="card"><div class="row"><div class="rank">!</div>'
        f'<div><span class="name">{_e(platform)} sync failed</span></div></div>'
        f'<div class="why">{_e(lead)} {_e(reason)}'
        f'{(" Last attempt " + _e(when) + " UTC.") if when else ""}</div></div>'
    )
