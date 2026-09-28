"""The "My team" panel for the published page.

Rendered per profile when a league snapshot exists; silently omitted when it
does not, so the streaming page never depends on a sync having run.
"""

from __future__ import annotations

import base64
import html
import json
import os

import numpy as np

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
        f"{_e(snapshot.synced_at[:16].replace('T', ' '))} UTC{_refresh_link()}</p>",
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

    parts.append(_season(report))

    # -- lineup ----------------------------------------------------------
    changed = {p.player_id for _s, _b, p in opt.changes}
    has_vegas = any(p.vegas_points is not None for p in snapshot.my_team.roster)
    rows = []
    for slot, p in opt.best_win.flat():
        flag = ""
        if p.actual_points is not None:
            flag = ' <span class="opp">final</span>'
        elif p.locked:
            flag = ' <span class="opp">locked</span>'
        elif p.player_id in changed:
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
        if p.actual_points is not None:
            spread = "scored"
        rows.append(
            f"<tr><td>{_e(slot)}</td><td class='unit'>{_e(p.name)}{flag}</td>"
            f"<td>{_e(p.position)}</td><td>{_e(p.team or '--')}</td>"
            f"<td>{_proj_cell(p)}</td>{vegas}"
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
    parts.append(_bench(snapshot, opt, has_vegas))
    parts.append(
        '<p class="sub">Range is the middle 70% of simulated outcomes (15th to 85th '
        "percentile), with teammates and opponents correlated as they are on the field.</p>"
    )
    parts.append(_compare_lineups(opt))
    parts.append(_editor(snapshot, opt))
    if opt.reasons:
        items = "".join(f"<li>{_e(r)}</li>" for r in opt.reasons)
        parts.append(f'<p class="sub">Not simply the highest projections, because:</p><ul class="sub">{items}</ul>')

    if has_vegas:
        parts.append(_vegas_section(snapshot, opt, cfg, getattr(snapshot, "_vegas_report", None)))

    for note in report.notes:
        parts.append(f'<p class="sub">Note: {_e(note)}.</p>')

    # -- waivers ---------------------------------------------------------
    if report.title_moves is not None:
        parts.append(_title_moves(report))
        parts.append(_plans(report))
        parts.append(_trades(report))
        moves = [m for m in moves if m.add.position in ("DST", "K")]
    parts.append("<h3>Waiver moves</h3>" if report.title_moves is None else "<h3>D/ST and K streams</h3>")
    if not moves:
        parts.append('<p class="sub">Nothing on the wire clears the bar this week.</p>')
    else:
        cards = []
        for m in moves[:6]:
            drop = f" &middot; drop {_e(m.drop.name)}" if m.drop else ""
            note = _stream_note(report, m.add)
            if note:
                m.reason = f"{m.reason}; {note}"
            cards.append(
                '<div class="card"><div class="row">'
                f'<div class="rank">{_e(m.tag[:1].upper())}</div>'
                f'<div><span class="name">{_e(m.add.name)}</span> '
                f'<span class="opp">{_e(m.add.position)} {_e(m.add.team or "")}{drop}</span></div>'
                f'<div class="pts">+{m.score:.1f}</div></div>'
                f'<div class="why">{_e(m.reason)}</div></div>'
            )
        parts.append("".join(cards))

    parts.append(_streams(report))

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

    parts.append(_lottery(snapshot, cfg))

    watch = drop_watch(snapshot, n=3)
    if watch:
        parts.append(
            '<p class="sub">Drop watch: '
            + ", ".join(f"{_e(p.name)} ({(p.projection or 0):.1f})" for p in watch)
            + "</p>"
        )
    return "".join(parts)


_VERDICT = {"claim": ("Claim now", "hold-tag"), "lean": ("Close call: lean claim", "opp"),
            "wait": ("Worth adding, but wait", "opp")}


def _title_moves(report: MatchupReport) -> str:
    """Pickups valued by what they do to your title odds."""
    moves = report.title_moves or []
    head = ("<h3>Waiver moves, by title odds</h3>"
            '<p class="sub">Each pickup is priced in P(win the title) on the same simulated seasons: '
            "claiming now, against waiting and claiming only if he breaks out (you win that claim when "
            "your priority beats the other managers who also want him), less what your waiver priority "
            "is worth on future claims. Each move is priced on its own; make one, then refresh. "
            "Assumes about a third of active managers chase any one breakout.</p>")
    if not moves:
        return head + '<p class="sub">No free agent raises your title odds enough to be worth a move.</p>'
    cards = []
    for m in moves:
        label, css = _VERDICT[m.verdict]
        cards.append(
            '<div class="card"><div class="row">'
            f'<div class="rank">{"&#10003;" if m.verdict == "claim" else "&middot;"}</div>'
            f'<div><span class="name">{_e(m.add.name)}</span> '
            f'<span class="opp">{_e(m.add.position)} {_e(m.add.team or "")} &middot; drop {_e(m.drop.name)}</span></div>'
            f'<div class="pts">{m.gain_now * 100:+.1f}</div></div>'
            f'<div class="meta"><span class="{css}">{label}</span>'
            f"<span>title {m.p_base:.1%} &rarr; {m.p_now:.1%}</span></div>"
            f'<div class="why">{_e("; ".join(m.reasons))}</div></div>')
    return head + "".join(cards)


def _plans(report: MatchupReport) -> str:
    """Two or three waiver moves made together, priced as one."""
    plans = report.plans
    if plans is None:
        return ""
    head = ("<h3>Waiver plans: several moves together</h3>"
            '<p class="sub">Moves interact -- two backs chasing one lineup spot are worth less together, '
            "two dead roster spots turned into a starter and his handcuff can be worth more -- so every "
            "combination of two or three pickups and drops is screened, and the best priced in title odds "
            "against standing pat and against the best single move. <b>Make the plan</b> when it beats that "
            "single move by more than the simulation noise; a <b>close call</b> when by less.</p>")
    if not plans:
        return head + '<p class="sub">No plan beats the best single move; make that one and refresh.</p>'
    cards = []
    for plan in plans:
        label = ('<span class="hold-tag">make the plan</span>' if plan.verdict == "plan"
                 else '<span class="opp">close call</span>')
        steps = "".join(
            f"<li>Add <b>{_e(a.name)}</b> ({_e(a.position)} {_e(a.team or '')}), drop {_e(d.name)}</li>"
            for a, d in plan.moves)
        cards.append(
            '<div class="card"><div class="row"><div class="rank">&#9776;</div>'
            f'<div><span class="name">{len(plan.moves)} moves</span> '
            f'<span class="opp">{_e(", ".join(a.name for a in plan.adds))}</span></div>'
            f'<div class="pts">{plan.gain * 100:+.1f}</div></div>'
            f'<div class="meta">{label}<span>title {plan.p_base:.1%} &rarr; {plan.p_plan:.1%}</span>'
            f"<span>best single move {plan.p_single:.1%}</span>"
            f"<span>&plusmn;{2 * plan.noise * 100:.1f} noise</span></div>"
            f'<div class="why"><ol class="steps">{steps}</ol>{_e("; ".join(plan.reasons))}.</div></div>')
    return head + "".join(cards)


def _names(players) -> str:
    return " + ".join(f"{p.name} ({p.position})" for p in players)


def _trade_card(t) -> str:
    from .trades import acceptance_label

    win = '<span class="hold-tag">win-win</span>' if t.win_win else ""
    return (
        '<div class="card"><div class="row"><div class="rank">&#8644;</div>'
        f'<div><span class="name">Get {_e(_names(t.get))}</span> '
        f'<span class="opp">from {_e(t.partner.name)} &middot; give {_e(_names(t.give))}</span></div>'
        f'<div class="pts">{t.gain * 100:+.1f}</div></div>'
        f'<div class="meta">{win}<span>title {t.p_base:.1%} &rarr; {t.p_new:.1%}</span>'
        f"<span>yes: {acceptance_label(t.p_accept)} (~{t.p_accept:.0%})</span>"
        f"<span>his title {t.their_base:.1%} &rarr; {t.their_new:.1%}</span></div>"
        f'<div class="why"><b>For you:</b> {_e("; ".join(t.why_you))}.<br>'
        f'<b>For him:</b> {_e("; ".join(t.why_them))}.</div></div>')


_TRADE_TABS = (
    ("top", "Top trades", "By expected gain: the chance he says yes times the title odds you gain."),
    ("best", "Best for you", "By title odds gained, among offers with at least a 15% chance of a yes."),
    ("likely", "Most likely yes", "By the chance he says yes, among offers that clearly raise your title odds."),
)


def _trades(report: MatchupReport) -> str:
    """Three views of the same trades: overall, best for you, most likely accepted."""
    board = report.trades
    if board is None:
        return ""
    head = ("<h3>Trades</h3>"
            '<p class="sub">Every 1-for-1, 2-for-1 and 1-for-2 with every team. <b>For you</b>: priced in '
            "title odds for both teams on the same simulated seasons as the waiver moves. <b>For him</b>: he "
            "cannot see this tool, so his side is judged the way the market values players -- the "
            "platform's projection, a player's name and last season, this season so far, injuries -- "
            "in the proportions measured from Yahoo's rostership, plus whether he gets the best player in "
            "the deal and how active he is. The yes-chance ranks offers; it is a rough estimate, not a "
            "measured rate. Each trade is priced on its own, as an alternative to the others.</p>")
    if not board:
        return head + '<p class="sub">No trade both raises your title odds and has a real chance of a yes.</p>'
    uid = _e(report.snapshot.profile)
    radios, labels, panes = [], [], []
    for i, (key, label, note) in enumerate(_TRADE_TABS):
        trades = getattr(board, key)
        tid = f"trades-{uid}-{key}"
        radios.append(f'<input class="tab-radio t{i}" type="radio" name="trades-{uid}" id="{tid}"'
                      f'{" checked" if i == 0 else ""}>')
        labels.append(f'<label for="{tid}">{label}</label>')
        body = "".join(_trade_card(t) for t in trades) or \
            '<p class="sub">Nothing clears this bar.</p>'
        panes.append(f'<div class="tab-pane p{i}"><p class="sub">{note}</p>{body}</div>')
    return (head + '<div class="tabs">' + "".join(radios)
            + '<div class="tab-labels">' + "".join(labels) + "</div>" + "".join(panes) + "</div>")


def _season(report: MatchupReport) -> str:
    """Playoff and title odds for every team, yours first."""
    o = report.season
    if o is None or o.mine is None:
        return ""
    m = o.mine
    rows = []
    for i in sorted(range(len(o.names)), key=lambda i: -o.p_title[i]):
        me = " style='font-weight:700'" if i == m else ""
        rows.append(f"<tr{me}><td class='unit'>{_e(o.names[i])}</td><td>{_e(o.records[i])}</td>"
                    f"<td>{o.p_title[i]:.1%}</td><td>{o.p_playoffs[i]:.0%}</td><td>{o.exp_wins[i]:.1f}</td></tr>")
    return (
        '<div class="card"><div class="row"><div class="rank">&#127942;</div>'
        f'<div><span class="name">Season outlook</span> '
        f'<span class="opp">{o.p_playoffs[m]:.0%} to make the playoffs, {o.p_bye[m]:.0%} for a bye</span></div>'
        f'<div class="pts">{o.p_title[m]:.1%}</div></div>'
        f'<div class="meta"><span>P(title)</span><span>{o.exp_wins[m]:.1f} expected wins</span>'
        f"<span>{o.n_sims:,} simulated seasons</span></div></div>"
        '<details><summary>Every team\'s odds</summary><div class="scroll"><table><thead><tr>'
        '<th class="unit">Team</th><th>Rec</th><th>Title</th><th>Playoffs</th><th>Exp W</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        '<p class="sub">Each player\'s future is simulated (role drift, injuries, byes, the next man up '
        "taking over -- validated to put 82% of real six-week outcomes inside its 80% range); every team "
        "starts its best lineup each week on what it could see then; the season is played on the real "
        "schedule and the playoffs on the real bracket. Rosters are held as they stand.</p></details>"
    )


def _stream_note(report: MatchupReport, player) -> str:
    """This week's P(win) with a D/ST or kicker pickup, against the current one."""
    opts = report.streams.get(player.position) or []
    mine = next((o for o in opts if o.current), None)
    this = next((o for o in opts if o.player.player_id == player.player_id), None)
    if mine is None or this is None:
        return ""
    return (f"P(win) this week {this.win_probability:.1%} against {mine.win_probability:.1%} "
            f"with {mine.player.name}")


def _streams(report: MatchupReport) -> str:
    """D/ST and kickers ranked by what they do to *this* week's P(win)."""
    blocks = []
    for pos, label in (("DST", "Defence"), ("K", "Kicker")):
        opts = report.streams.get(pos) or []
        if len(opts) < 2:
            continue
        mine = next((o for o in opts if o.current), None)
        shown = opts[:5] + ([mine] if mine is not None and mine not in opts[:5] else [])
        rows = []
        for o in shown:
            delta = (o.win_probability - mine.win_probability) * 100 if mine else 0.0
            vs = "yours" if o.current else ("&asymp; same" if abs(delta) < 0.5 else f"{delta:+.1f}")
            links = f"<tr><td class='why' colspan='5'>&#8627; {_e('; '.join(o.links[:2]))}</td></tr>" if o.links else ""
            rows.append(
                f"<tr><td class='unit'>{_e(o.player.name)}</td><td>{_e(o.player.nfl_opponent or '')}</td>"
                f"<td>{(o.player.projection or 0):.1f}</td><td>{o.win_probability:.1%}</td><td>{vs}</td></tr>{links}")
        blocks.append(
            f"<p class='sub'><strong>{label}</strong></p>"
            '<div class="scroll"><table><thead><tr><th class="unit">Unit</th><th>Opp</th><th>Proj</th>'
            f"<th>P(win)</th><th>vs yours</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    if not blocks:
        return ""
    return ("<h3>Best D/ST and K for this matchup</h3>"
            '<p class="sub">Free agents and your own, each swapped into your recommended lineup and '
            "scored against your opponent on the same correlated simulations. A defence facing your own "
            "quarterback hedges him: that steadies your score, which helps when you are favoured and "
            "hurts when you are not. Gaps under half a point are noise.</p>"
            + "".join(blocks))


def _lottery(snapshot: LeagueSnapshot, cfg: Config) -> str:
    """Tickets for each kickoff window still to come, and what to drop."""
    from .lottery import enabled, lottery_tickets

    if not enabled(snapshot, cfg):
        return ""
    try:
        windows = lottery_tickets(snapshot, cfg)
    except Exception:  # noqa: BLE001 - a bonus section never blocks the page
        return ""
    if not windows:
        return ""
    blocks = []
    for w in windows:
        rows = "".join(
            f"<tr><td class='unit'>{_e(t.player.name)}</td><td>{_e(t.player.position)}</td>"
            f"<td>{_e(t.player.team or '')}</td><td>{(t.player.projection or 0):.1f}</td>"
            f"<td>{t.p_keep:.0%}</td></tr>"
            f"<tr><td class='why' colspan='5'>&#8627; {_e('; '.join(t.reasons))}</td></tr>"
            for t in w.tickets)
        blocks.append(
            f"<p class='sub'><strong>{_e(w.label)}</strong></p>"
            '<div class="scroll"><table><thead><tr><th class="unit">Player</th><th>Pos</th><th>Tm</th>'
            f"<th>Proj</th><th>Keep?</th></tr></thead><tbody>{rows}</tbody></table></div>")
    watch = drop_watch(snapshot, n=3)
    drops = ""
    if watch:
        drops = ('<p class="sub">Cheapest spots to open for the rotation: '
                 + ", ".join(f"{_e(p.name)} ({(p.ros_value or 0):.1f} a game rest of season)" for p in watch)
                 + ".</p>")
    return (
        "<h3>Game-time lottery tickets</h3>"
        '<p class="sub">Pick one up before each kickoff; keep him if the game changes his value, '
        "otherwise drop him once his game has started and grab the next window's ticket. "
        f"<em>Keep?</em> is the chance he gives you a reason to hold him: a top-{len(snapshot.teams)} week "
        "at his position (the bar measured from 2021-2025), "
        "or the lead back ahead of him going down (they miss the next game 8.5% of the time). "
        "A big game without more opportunity behind it often fades, so treat it as a look, not a lock.</p>"
        + "".join(blocks) + drops)


def _proj_cell(p) -> str:
    """The projection, and for an injury tag what he projects if he plays."""
    cell = f"{p.week_value:.1f}"
    q = p.play_probability
    if p.actual_points is None and 0 < q < 1 and p.projection:
        cell += (f' <span class="opp">({p.projection / q:.1f} if he plays, '
                 f"{q:.0%})</span>")
    return cell


def _bench_reason(p, opt) -> str:
    """Why a benched player sits when he projects more than a starter whose
    slot he could fill; empty when he simply projects less."""
    from .lineup import _eligible

    if p.in_ir_slot:
        return "in an IR slot"
    lower = [q for slot, q in opt.best_win.flat() if _eligible(p, slot) and q.week_value < p.week_value]
    if not lower:
        return ""
    if p.locked:
        return "his game has started, so he cannot be moved in"
    if p.on_bye:
        return "on bye"
    if p.is_out:
        return f"listed {p.status.lower() or 'out'}"
    if p.player_id in opt.bench_reasons:
        return opt.bench_reasons[p.player_id]
    q = min(lower, key=lambda x: x.week_value)
    if q.locked:
        return f"{q.name}'s game has started, so his spot is frozen"
    return ""


def _bench(snapshot: LeagueSnapshot, opt, has_vegas: bool) -> str:
    """Everyone not in the recommended lineup, with the same numbers."""
    starting = opt.best_win.player_ids
    bench = [p for p in snapshot.my_team.roster if p.player_id not in starting]
    if not bench:
        return ""
    bench.sort(key=lambda p: (p.in_ir_slot, -p.week_value))
    rows = []
    for p in bench:
        tag = ""
        if p.actual_points is not None:
            tag = ' <span class="opp">final</span>'
        elif p.status:
            tag = f' <span class="opp">{_e(short_status(p.status))}</span>'
        vegas = ""
        if has_vegas:
            vegas = (f"<td>{p.vegas_points:.1f}</td>" if p.vegas_points is not None
                     else '<td class="opp">--</td>')
        lo, hi = opt.ranges.get(p.player_id, (None, None))
        spread = ("scored" if p.actual_points is not None
                  else f"{lo:.0f}&ndash;{hi:.0f}" if lo is not None else "--")
        why = _bench_reason(p, opt)
        rows.append(
            f"<tr><td class='unit'>{_e(p.name)}{tag}</td><td>{_e(p.position)}</td>"
            f"<td>{_e(p.team or '--')}</td><td>{_proj_cell(p)}</td>{vegas}<td>{spread}</td></tr>"
            + (f"<tr><td class='why' colspan='{6 if has_vegas else 5}'>&#8627; {_e(why)}</td></tr>" if why else "")
        )
    vegas_head = "<th>Vegas</th>" if has_vegas else ""
    return ("<h3>Bench</h3>"
            '<div class="scroll"><table><thead><tr><th class="unit">Player</th><th>Pos</th><th>Tm</th>'
            f"<th>Proj</th>{vegas_head}<th>Range</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def _refresh_link() -> str:
    """A link to the workflow's Run button. A static page cannot start a
    workflow itself without shipping a token, so it opens GitHub, where the
    default job, ``refresh``, re-syncs both leagues without spending Odds API
    credits."""
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if not repo:
        return ""
    url = f"https://github.com/{repo}/actions/workflows/weekly.yml"
    return (f' &middot; <a href="{_e(url)}" title="Opens GitHub: tap Run workflow, job refresh '
            '(no Odds API credits). Takes about two minutes.">&#8635; refresh</a>')


def _compare_lineups(opt) -> str:
    """As set, best P(win) and most points, side by side (no script needed)."""
    rows = []
    options = [("As set", opt.current), ("Best P(win)", opt.best_win), ("Most points", opt.best_ev)]
    for label, lu in options:
        if lu is None:
            continue
        rows.append(f"<tr><td class='unit'>{label}</td><td>{_pct(lu.win_probability)}</td>"
                    f"<td>{lu.expected:.1f}</td></tr>")
    return ('<div class="scroll"><table><thead><tr><th class="unit">Lineup</th><th>P(win)</th>'
            f"<th>Projected</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def _b64(values: np.ndarray) -> str:
    """Scores in tenths of a point as little-endian int16, base64."""
    return base64.b64encode(np.round(values * 10).clip(-32000, 32000).astype("<i2").tobytes()).decode()


def _editor(snapshot: LeagueSnapshot, opt) -> str:
    """Try any lineup by hand and see P(win) and the projection move.

    The page carries a slice of the same simulation that chose the lineup
    (5,000 draws of your players and your opponent's total), so a lineup
    scored here is scored exactly as the optimiser scored it, just with
    fewer draws. Players whose game has kicked off cannot be moved.
    """
    from .lineup import _eligible

    if opt.samples is None or not opt.sample_ids or opt.opp_samples is None:
        return ""
    by_id = {p.player_id: p for p in snapshot.my_team.roster}
    players = [by_id[i] for i in opt.sample_ids if i in by_id]
    if len(players) != len(opt.sample_ids):
        return ""
    idx = {p.player_id: k for k, p in enumerate(players)}
    slots = [slot for slot, n in snapshot.starting_slots.items() for _ in range(n)]

    def preset(lu) -> list[int]:
        if lu is None:
            return []
        pools = {slot: [idx[p.player_id] for p in ps if p.player_id in idx] for slot, ps in lu.starters.items()}
        return [pools.get(slot, []).pop(0) if pools.get(slot) else -1 for slot in slots]

    data = {
        "slots": slots,
        "players": [{
            "n": p.name, "pos": p.position, "tm": p.team or "", "p": round(p.week_value, 1),
            "el": [s for s in dict.fromkeys(slots) if _eligible(p, s)],
            "l": bool(p.locked), "f": p.actual_points is not None,
            "st": p.slot if p.starting else "",
        } for p in players],
        "presets": {"set": preset(opt.current), "win": preset(opt.best_win), "pts": preset(opt.best_ev)},
        "m": len(players), "n": int(opt.samples.shape[0]),
        "s": _b64(np.asarray(opt.samples, dtype=float)), "o": _b64(np.asarray(opt.opp_samples, dtype=float)),
    }
    uid = f"ed-{_e(snapshot.profile)}"
    # Inside a script element "</" would end it early.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return (
        f'<details class="card" id="{uid}"><summary><strong>Try a lineup</strong> '
        '<span class="opp">pick any starters and see P(win) and points change</span></summary>'
        '<div class="ed-buttons">'
        '<button type="button" data-p="set">As set</button>'
        '<button type="button" data-p="win">Best P(win)</button>'
        '<button type="button" data-p="pts">Most points</button></div>'
        '<div class="scroll"><table><tbody class="ed-rows"></tbody></table></div>'
        '<p class="ed-out sub"></p>'
        f'<script type="application/json" class="ed-data">{payload}</script>'
        f"<script>{_EDITOR_JS.replace('__UID__', uid)}</script>"
        "</details>"
    )


_EDITOR_JS = r"""
(function(){
var root=document.getElementById('__UID__');if(!root)return;
var D=JSON.parse(root.querySelector('.ed-data').textContent);
function dec(b){var s=atob(b),u=new Uint8Array(s.length);for(var i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new Int16Array(u.buffer);}
var S=dec(D.s),O=dec(D.o),m=D.m,n=D.n,body=root.querySelector('.ed-rows'),out=root.querySelector('.ed-out'),sel=[];
var base=null;
D.slots.forEach(function(slot,k){
  var tr=document.createElement('tr'),td=document.createElement('td'),td2=document.createElement('td'),s=document.createElement('select');
  td.textContent=slot;
  var fixed=-1;D.players.forEach(function(p,i){if(p.l&&p.st===slot&&D.presets.set.indexOf(i)===k)fixed=i;});
  var o=document.createElement('option');o.value=-1;o.textContent='(empty)';s.appendChild(o);
  D.players.forEach(function(p,i){
    if(p.el.indexOf(slot)<0)return;
    if(p.l&&i!==fixed)return;
    var op=document.createElement('option');op.value=i;
    op.textContent=p.n+' '+p.pos+' '+p.tm+' ('+p.p.toFixed(1)+(p.f?' final':'')+')';s.appendChild(op);});
  if(fixed>=0){s.value=fixed;s.disabled=true;}
  s.addEventListener('change',calc);sel.push(s);td2.appendChild(s);tr.appendChild(td);tr.appendChild(td2);body.appendChild(tr);});
function score(ch){var w=0;for(var r=0;r<n;r++){var t=0,o=r*m;for(var j=0;j<ch.length;j++)t+=S[o+ch[j]];w+=t>O[r]?1:(t===O[r]?0.5:0);}return w/n;}
function calc(){
  var ch=[],seen={},dup=false,pts=0;
  sel.forEach(function(s){var v=+s.value;if(v<0)return;if(seen[v])dup=true;seen[v]=1;ch.push(v);pts+=D.players[v].p;});
  if(dup){out.textContent='A player is picked twice.';return;}
  var p=score(ch),txt='P(win) '+(p*100).toFixed(1)+'%  ·  projected '+pts.toFixed(1);
  if(base!==null)txt+='  ('+((p-base)*100>=0?'+':'')+((p-base)*100).toFixed(1)+' vs best P(win))';
  out.textContent=txt+'  —  from '+n.toLocaleString()+' of the simulated weeks, so ±0.7.';}
function load(k){var pr=D.presets[k]||[];sel.forEach(function(s,i){if(!s.disabled)s.value=(pr[i]===undefined?-1:pr[i]);});calc();}
root.querySelectorAll('button[data-p]').forEach(function(b){b.addEventListener('click',function(){load(b.getAttribute('data-p'));});});
var bw=D.presets.win.filter(function(v){return v>=0;});base=score(bw);
load('set');
})();
"""


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
    weights = [p.market_weight for p in snapshot.my_team.roster if p.market_weight > 0]
    if weights:
        bits.append(
            f"Where a book priced a player, his projection blends in the market at up to "
            f"{max(weights):.0%} (less when fewer books posted him); the weight is refit as "
            "this season's results come in.")
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
