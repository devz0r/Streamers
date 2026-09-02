"""The "My team" panel for the published page.

Rendered per profile when a league snapshot exists; silently omitted when it
does not, so the streaming page never depends on a sync having run.
"""

from __future__ import annotations

import html

from ..config import Config
from ..league.model import LeagueSnapshot, short_status
from .matchup import MatchupReport
from .waivers import Move, drop_watch


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
    rows = []
    for slot, p in opt.best_win.flat():
        flag = ""
        if p.player_id in changed:
            flag = ' <span class="hold-tag">start</span>'
        elif p.is_questionable:
            flag = f' <span class="opp">{_e(short_status(p.status))}</span>'
        rows.append(
            f"<tr><td>{_e(slot)}</td><td class='unit'>{_e(p.name)}{flag}</td>"
            f"<td>{_e(p.position)}</td><td>{_e(p.team or '--')}</td>"
            f"<td>{(p.projection or 0):.1f}</td><td>&plusmn;{(p.projection_sd or 0):.0f}</td></tr>"
        )
    parts.append(
        "<h3>Recommended lineup</h3>"
        '<div class="scroll"><table><thead><tr><th>Slot</th><th class="unit">Player</th>'
        "<th>Pos</th><th>Tm</th><th>Proj</th><th>Range</th></tr></thead>"
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
    if opt.best_ev.player_ids != opt.best_win.player_ids:
        parts.append(
            f'<p class="sub">The max-points lineup would project {opt.best_ev.expected:.1f} '
            f"but win only {_pct(opt.best_ev.win_probability)}; the recommendation trades "
            f"{opt.best_ev.expected - opt.best_win.expected:.1f} points for "
            f"{(opt.best_win.win_probability - opt.best_ev.win_probability) * 100:+.0f} points of win probability.</p>"
        )

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

    watch = drop_watch(snapshot, n=3)
    if watch:
        parts.append(
            '<p class="sub">Drop watch: '
            + ", ".join(f"{_e(p.name)} ({(p.projection or 0):.1f})" for p in watch)
            + "</p>"
        )
    return "".join(parts)
