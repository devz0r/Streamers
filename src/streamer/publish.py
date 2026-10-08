"""Rendering the rankings to a static, mobile-first page.

``streamer publish --week N`` writes ``docs/index.html`` (plus an archived
``docs/week_N.html``) so GitHub Pages can serve it at a stable URL. The page is
deliberately plain HTML and CSS with no build step, no framework and no
JavaScript dependencies -- it has to render instantly on a phone and keep
working years from now.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import load_benchmark
from .calibrate import load_history
from .config import Config, get_config
from .models.base import spearman
from .models.ledger import load_ledger
from .rankings import Rankings, two_week_candidates

STYLE = """
:root {
  color-scheme: dark light;
  --bg: #0f1216;
  --panel: #171b21;
  --panel-2: #1e242c;
  --line: #2a323d;
  --text: #e7ecf3;
  --muted: #97a3b4;
  --accent: #4ea1ff;
  --good: #4ade80;
  --warn: #fbbf24;
  --bad: #f87171;
  --radius: 12px;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --panel-2: #f0f2f5; --line: #dfe3e8;
    --text: #14181d; --muted: #5b6673; --accent: #0b63c5;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 0 3rem;
  background: var(--bg); color: var(--text);
  font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-text-size-adjust: 100%;
}
.wrap { max-width: 900px; margin: 0 auto; padding: 1rem; }
header { padding: .6rem 0 .4rem; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; letter-spacing: -0.01em; }
h2 { font-size: 1.15rem; margin: 2rem 0 .75rem; letter-spacing: -0.01em; }
h3 { font-size: 1rem; margin: 1.25rem 0 .5rem; color: var(--muted); font-weight: 600; }
p { margin: .5rem 0; }
.sub { color: var(--muted); font-size: .875rem; margin: 0; }
.badges { display: flex; flex-wrap: wrap; gap: .4rem; margin: .75rem 0 0; }
.badge {
  display: inline-block; padding: .2rem .55rem; border-radius: 999px;
  font-size: .75rem; background: var(--panel-2); color: var(--muted);
  border: 1px solid var(--line);
}
.badge.ok { color: var(--good); border-color: color-mix(in srgb, var(--good) 40%, var(--line)); }
.badge.warn { color: var(--warn); border-color: color-mix(in srgb, var(--warn) 40%, var(--line)); }
.badge.bad { color: var(--bad); border-color: color-mix(in srgb, var(--bad) 40%, var(--line)); }
.notice {
  background: color-mix(in srgb, var(--warn) 12%, var(--panel));
  border: 1px solid color-mix(in srgb, var(--warn) 35%, var(--line));
  border-radius: var(--radius); padding: .75rem 1rem; margin: 1rem 0;
  font-size: .875rem;
}
.card {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: var(--radius); margin: 0 0 .6rem; overflow: hidden;
}
.row {
  display: grid; grid-template-columns: 2.2rem 1fr auto;
  gap: .6rem; align-items: baseline; padding: .7rem .85rem .3rem;
}
.rank { font-variant-numeric: tabular-nums; font-weight: 700; color: var(--accent); font-size: 1.05rem; }
.name { font-weight: 600; }
.opp { color: var(--muted); font-weight: 400; font-size: .875rem; }
.pts { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; font-size: 1.05rem; }
.meta {
  display: flex; flex-wrap: wrap; gap: .25rem .5rem; align-items: center;
  padding: 0 .85rem .5rem 3.65rem; color: var(--muted); font-size: .8rem;
  font-variant-numeric: tabular-nums;
}
.meta > span { background: var(--panel-2); border-radius: 6px; padding: .1rem .4rem; }
.why { padding: 0 .85rem .75rem 3.65rem; color: var(--muted); font-size: .82rem; }
.hold { border-left: 3px solid var(--good); }
.meta > span.avail-open { background: color-mix(in srgb, var(--good) 22%, transparent); color: var(--good); font-weight: 600; }
.meta > span.avail-yours { background: color-mix(in srgb, var(--accent) 22%, transparent); color: var(--accent); font-weight: 600; }
.meta > span.avail-taken { color: var(--muted); }
.card.taken { opacity: .55; }
.hold-tag { color: var(--good); font-size: .72rem; font-weight: 600; text-transform: uppercase; letter-spacing: .03em; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; font-size: .8rem; }
th, td { padding: .45rem .45rem; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
td.unit, th.unit { text-align: left; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; font-size: .78rem; text-transform: uppercase; letter-spacing: .03em; }
td { font-variant-numeric: tabular-nums; }
.pos { color: var(--good); } .neg { color: var(--bad); }
footer { color: var(--muted); font-size: .8rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--line); }
a { color: var(--accent); }
.ed-buttons { display: flex; gap: .4rem; flex-wrap: wrap; margin: .6rem 0; }
.ed-buttons button, .ed-rows select { font: inherit; color: var(--text); background: var(--bg);
  border: 1px solid var(--line); border-radius: 8px; padding: .35rem .6rem; max-width: 100%; }
.ed-rows select { width: 100%; }
.ed-out { font-weight: 600; color: var(--text); }
.te-cols { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem; margin: .6rem 0; }
@media (max-width: 600px) { .te-cols { grid-template-columns: 1fr; } }
.te-cols fieldset { border: 1px solid var(--line); border-radius: 8px; padding: .4rem .6rem; margin: 0; min-width: 0; }
.te-cols legend { font-weight: 600; font-size: .85rem; padding: 0 .3rem; }
.te-sum { position: sticky; bottom: .5rem; z-index: 3; margin: .5rem 0; padding: .45rem .7rem;
  background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px; font-weight: 600;
  font-size: .85rem; font-variant-numeric: tabular-nums; }
.te-pick { display: block; font-size: .82rem; padding: .2rem 0; overflow-wrap: anywhere; }
.te-partner, .te-dropline select { font: inherit; color: var(--text); background: var(--bg);
  border: 1px solid var(--line); border-radius: 8px; padding: .35rem .6rem; max-width: 100%; }
details { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); padding: .75rem .9rem; margin: .6rem 0; }
summary { cursor: pointer; font-weight: 600; font-size: .9rem; }
.archive { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .5rem; }
.archive a { font-size: .8rem; padding: .25rem .6rem; background: var(--panel-2); border: 1px solid var(--line); border-radius: 999px; text-decoration: none; }

/* Profile switch.
   Radios live at the top of <body>; the labels are styled as a segmented
   control and the :checked state drives which panel is visible. No JavaScript,
   so the switch works instantly on a phone and still works with JS disabled. */
.profile-radio { position: absolute; opacity: 0; pointer-events: none; }
:root { --switch-h: 3.55rem; }
.switch-bar { position: sticky; top: 0; z-index: 5; background: var(--bg);
  margin: 0 -1rem; padding: .5rem 1rem .3rem; }
.switch {
  display: flex; gap: .25rem; padding: .25rem;
  background: var(--panel-2); border: 1px solid var(--line); border-radius: 999px;
}
.switch label {
  flex: 1; text-align: center; padding: .45rem .5rem; border-radius: 999px;
  font-size: .85rem; font-weight: 600; color: var(--muted); cursor: pointer;
  user-select: none; -webkit-tap-highlight-color: transparent;
  transition: background .12s ease, color .12s ease;
}
.switch label:hover { color: var(--text); }
.profile-panel { display: none; }

ol.steps { margin: 0 0 .4rem; padding-left: 1.2rem; color: var(--text); }
.card.hub { border-left: 3px solid var(--accent); margin-top: .8rem; }
.hub-table td { white-space: normal; vertical-align: top; }
.hub-table td .why { padding: .15rem 0 0; }
.kind { display: inline-block; font-size: .68rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .03em; padding: .05rem .35rem; border-radius: 5px; background: var(--panel-2);
  color: var(--muted); margin-right: .25rem; }
.kind-trade { color: var(--warn); } .kind-block { color: var(--bad); }
.kind-lineup, .kind-stream, .kind-pickup { color: var(--accent); } .kind-waiver, .kind-plan { color: var(--good); }
details > summary + h3 { margin-top: .6rem; }
ol.steps li { margin: .1rem 0; }

/* A league's sections, as tabs: the same radio + :checked pattern, with the
   strip sticking under the league switch. Rules per tab are generated. */
.sec-radio { position: absolute; opacity: 0; pointer-events: none; }
.sec-labels { display: flex; overflow-x: auto; scrollbar-width: none; position: sticky; top: 0; z-index: 4;
  margin: 0 -1rem .5rem; padding: 0 .6rem; background: var(--bg); border-bottom: 1px solid var(--line); }
.sec-labels::-webkit-scrollbar { display: none; }
.sec-labels label { flex: 0 0 auto; padding: .55rem .6rem .5rem; font-size: .85rem; font-weight: 600;
  color: var(--muted); cursor: pointer; user-select: none; white-space: nowrap;
  -webkit-tap-highlight-color: transparent; }
.sec-labels label:hover { color: var(--text); }
.sec-pane { display: none; }
.sec-pane > h2:first-child, .sec-pane > h3:first-child, .sec-pane > .card:first-child { margin-top: .4rem; }
.roster-table td { vertical-align: top; }
.roster-table td.unit { white-space: normal; min-width: 9rem; }
.roster-table .tags { color: var(--muted); font-size: .72rem; font-weight: 400; }
.roster-table tr.has-note td { border-bottom: 0; padding-bottom: .15rem; }
.roster-table tr.note td { white-space: normal; text-align: left; padding-top: 0; color: var(--muted);
  font-size: .75rem; }
.tab-link { color: var(--accent); cursor: pointer; text-decoration: underline; text-underline-offset: 2px; }
.news { list-style: none; padding: 0; margin: .4rem 0; }
.news li { padding: .45rem 0; border-bottom: 1px solid var(--line); line-height: 1.4; }
.news a { color: var(--text); }
.news .tag { display: inline-block; font-size: .72rem; padding: 0 .4rem; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); margin-right: .25rem; }
.news .t-out, .news .t-legal { color: var(--bad); border-color: var(--bad); }
.news .t-healthy, .news .t-role-up { color: var(--good); border-color: var(--good); }
.news .t-injury, .news .t-role-down { color: var(--warn); border-color: var(--warn); }

/* Tabs inside a panel (trade views): the same radio + :checked pattern. */
.tab-radio { position: absolute; opacity: 0; pointer-events: none; }
.tab-labels { display: flex; gap: .25rem; padding: .2rem; margin: .4rem 0 .6rem;
  background: var(--panel-2); border: 1px solid var(--line); border-radius: 999px; }
.tab-labels label { flex: 1; text-align: center; padding: .35rem .4rem; border-radius: 999px;
  font-size: .8rem; font-weight: 600; color: var(--muted); cursor: pointer; user-select: none;
  -webkit-tap-highlight-color: transparent; }
.tab-pane { display: none; }
.t0:checked ~ .p0, .t1:checked ~ .p1, .t2:checked ~ .p2 { display: block; }
.t0:checked ~ .tab-labels label:nth-child(1), .t1:checked ~ .tab-labels label:nth-child(2),
.t2:checked ~ .tab-labels label:nth-child(3) { background: var(--accent); color: #fff; }
"""


def _switch_rules(profiles: list[str]) -> str:
    """Per-profile :checked rules, generated so any number of profiles works."""
    rules = []
    for name in profiles:
        rules.append(
            f"#profile-{name}:checked ~ .wrap .switch label[for=profile-{name}]"
            "{background:var(--accent);color:#fff;}"
        )
        rules.append(
            f"#profile-{name}:checked ~ .wrap #panel-{name}{{display:block;}}"
        )
    return "\n".join(rules)


def _section_tabs(profile: str, sections: dict[str, str]) -> str:
    """One league's sections as tabs: a hidden radio per tab, a sticky strip of
    labels, and a pane each, switched by ``:checked`` sibling rules like the
    league switch -- no script. Empty sections get no tab; the first one
    present opens."""
    from .roster.page import TABS

    tabs = [(i, key, label) for i, (key, label) in enumerate(TABS) if sections.get(key)]
    if not tabs:
        return ""
    name = _e(profile)
    radios = "".join(
        f'<input class="sec-radio s{i}" type="radio" name="sec-{name}" id="sec-{name}-{key}"'
        f'{" checked" if n == 0 else ""}>' for n, (i, key, _label) in enumerate(tabs))
    labels = "".join(f'<label class="l{i}" for="sec-{name}-{key}">{_e(label)}</label>'
                     for i, key, label in tabs)
    panes = "".join(f'<section class="sec-pane q{i}" id="{name}-{key}">{sections[key]}</section>'
                    for i, key, _label in tabs)
    return f'{radios}<nav class="sec-labels" aria-label="Sections">{labels}</nav>{panes}'


def _section_rules(profiles: list[str]) -> str:
    """The tab rules, one per tab position, and where the strip sticks: below
    the league switch when there is one."""
    from .roster.page import TABS

    rules = []
    for i in range(len(TABS)):
        rules.append(f".s{i}:checked ~ .q{i}{{display:block;}}")
        rules.append(f".s{i}:checked ~ .sec-labels .l{i}{{color:var(--accent);box-shadow:inset 0 -2px 0 var(--accent);}}")
    if len(profiles) > 1:
        rules.append(".sec-labels{top:var(--switch-h);}")
    return "\n".join(rules)


def _e(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _num(value, places: int = 1, dash: str = "--") -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return dash
    return dash if not np.isfinite(f) else f"{f:.{places}f}"


def _pct(value, dash: str = "--") -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return dash
    return dash if not np.isfinite(f) else f"{f * 100:.0f}%"


@dataclass
class PublishResult:
    index_path: Path
    archive_path: Path


def render_page(
    ranked: dict[str, Rankings] | Rankings,
    cfg: Config | None = None,
    team_panels: dict[str, dict[str, str] | str] | None = None,
) -> str:
    """Render the whole page to an HTML string.

    ``ranked`` maps profile name to that profile's rankings. When more than one
    is supplied the page carries a segmented switch between them, implemented
    with a hidden radio per profile and ``:checked`` sibling rules -- no
    JavaScript, so switching is instant on a phone and survives a stale cache.
    """
    cfg = cfg or get_config()
    if isinstance(ranked, Rankings):
        ranked = {cfg.profile: ranked}
    profiles = [name for name in cfg.profile_names if name in ranked] or list(ranked)
    if not profiles:
        raise ValueError("nothing to publish")

    conf = cfg.publish
    generated = datetime.now(UTC).strftime("%a %d %b %Y, %H:%M UTC")
    first = ranked[profiles[0]]
    default = cfg.default_profile if cfg.default_profile in profiles else profiles[0]

    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head>',
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        '<meta name="color-scheme" content="dark light">',
        f"<title>{_e(conf['site_title'])} - Week {first.week}</title>",
        f"<style>{STYLE}\n{_switch_rules(profiles)}\n{_section_rules(profiles)}</style>",
        "</head><body>",
    ]

    # The radios must precede .wrap so the :checked sibling selectors reach it.
    for name in profiles:
        checked = " checked" if name == default else ""
        parts.append(
            f'<input class="profile-radio" type="radio" name="profile" '
            f'id="profile-{_e(name)}"{checked}>'
        )

    parts += [
        '<div class="wrap">',
        "<header>",
        f"<h1>Week {first.week}</h1>",
        f'<p class="sub">{_e(conf["site_title"])} &middot; {first.season} season &middot; '
        f"updated {generated}</p>",
        "</header>",
    ]

    if len(profiles) > 1:
        labels = "".join(
            f'<label for="profile-{_e(n)}">'
            f'{_e(cfg.for_profile(n).profile_description)}</label>'
            for n in profiles
        )
        parts.append(f'<div class="switch-bar"><div class="switch">{labels}</div></div>')

    for name in profiles:
        bound = cfg.for_profile(name)
        rankings = ranked[name]
        team = (team_panels or {}).get(name) or {}
        if isinstance(team, str):          # a whole panel, unsplit
            team = {"hub": team}
        units = {pos: [str(t) for t in frame["team"]] for pos, frame in (("DST", rankings.dst), ("K", rankings.kicker))
                 if frame is not None and not frame.empty and "team" in frame}
        avail = unit_availability(bound, rankings.week, units) if team_panels else {}
        sections = dict(team)
        sections["streams"] = "".join([
            _badges(rankings, bound), _notices(rankings), team.get("streams", ""),
            _two_week_section(rankings, avail),
            _ranking_section("Defense / Special Teams", rankings.dst, "DST", int(conf["top_n"]), bound, avail),
            _ranking_section("Kickers", rankings.kicker, "K", int(conf["top_n"]), bound, avail),
        ])
        sections["model"] = "".join([team.get("model", ""), _benchmark_section(bound),
                                     _calibration_section(bound, rankings), _ledger_section(bound)])
        parts.append(f'<div class="profile-panel" id="panel-{_e(name)}">')
        parts.append(_section_tabs(name, sections))
        parts.append("</div>")

    parts.append(_archive_section(cfg))
    parts.append(_footer())
    parts.append("</div></body></html>")
    return "\n".join(p for p in parts if p)


#: Badge styling and wording per line status. Only an unpriced game is amber:
#: nflverse ships the closing spread and total, so a slate priced from the
#: schedule is properly priced, just not freshly pulled.
_LINE_BADGE = {
    "live": ("ok", "live odds"),
    "fallback": ("", None),        # the source phrase is the label
    "incomplete": ("warn", "incomplete lines"),
}


def _badges(rankings: Rankings, cfg: Config) -> str:
    badges = []
    if rankings.context and rankings.context.lines:
        lines = rankings.context.lines
        cls, label = _LINE_BADGE.get(lines.status, ("", None))
        # Only the dominant source; describe() names the rest, so the full
        # phrase here would repeat itself on a mixed slate.
        label = label or lines.primary_label()
        badges.append(
            f'<span class="badge {cls}">{_e(label)}: {_e(lines.describe())}</span>'
        )
    est = rankings.dst["estimator"].iloc[0] if not rankings.dst.empty and "estimator" in rankings.dst else None
    if est:
        badges.append(f'<span class="badge">model: {_e(est)}</span>')
    badges.append(
        f'<span class="badge">{_e(cfg.profile_label)} scoring &middot; '
        f'{cfg.league["teams"]}-team</span>'
    )
    return f'<div class="badges">{"".join(badges)}</div>'


def _notices(rankings: Rankings) -> str:
    """Warn only when something is actually wrong with the numbers.

    A slate priced from the nflverse schedule instead of a live API pull is not
    a problem -- those are the closing spread and total. Shouting about it every
    week trains the reader to ignore the banner for the week it matters, which
    is when a game has no line at all.
    """
    lines = rankings.context.lines if rankings.context else None
    status = lines.status if lines else "incomplete"

    if status == "incomplete":
        items = "".join(f"<div>{_e(w)}</div>" for w in rankings.warnings)
        detail = (
            f"{lines.missing} game(s) on this slate have no spread or total, so "
            "those units are projected without a market anchor."
            if lines and lines.missing
            else "No betting lines could be resolved for this slate."
        )
        return f'<div class="notice"><strong>Heads up.</strong> {detail}{items}</div>'

    if status == "fallback":
        # Quiet and factual. This is a normal, usable state, so it gets a line
        # of prose rather than a banner -- but it still says why.
        source = lines.source_phrase() if lines else "a fallback source"
        why = "; ".join(w for w in rankings.warnings if w)
        note = f" &mdash; {_e(why)}" if why else ""
        return (
            f'<p class="sub">Priced from {_e(source)} rather than a live pull'
            f"{note}. Those are real closing numbers, not estimates.</p>"
        )
    return ""


#: Yahoo's free-agent pages hold 25 players; a shorter list is the whole pool.
YAHOO_PAGE = 25


def unit_availability(cfg: Config, week: int | None = None,
                      ranked_units: dict[str, list[str]] | None = None) -> dict[tuple[str, str], tuple[str, str]]:
    """(position, NFL team) -> (status, whose) for every D/ST and kicker in the
    league's snapshot: "yours", "available", or "taken" with the fantasy team
    that has him. Empty when there is no snapshot, so the rankings still
    render without a league."""
    from .league.store import load_snapshot

    try:
        snap = load_snapshot(cfg, week)
    except FileNotFoundError:
        try:
            snap = load_snapshot(cfg)
        except FileNotFoundError:
            return {}
    except Exception:  # noqa: BLE001 - a label must never block the page
        return {}
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for p in snap.free_agents:
        if p.position in ("DST", "K") and p.team:
            out.setdefault((p.position, p.team), ("available", ""))
    for team in snap.teams:
        for p in team.roster:
            if p.position in ("DST", "K") and p.team:
                out[(p.position, p.team)] = ("yours", team.name) if team.is_mine else ("taken", team.name)
    # Both platforms now sync every roster, so anything unlisted is simply
    # unknown. Older Yahoo snapshots held only yours and your opponent's; for
    # those, a free-agent list shorter than a page is the complete pool, so a
    # unit not in it is on somebody's roster.
    for pos, teams in (ranked_units or {}).items():
        n_free = sum(1 for p in snap.free_agents if p.position == pos)
        if snap.platform == "yahoo" and 0 < n_free < YAHOO_PAGE:
            for t in teams:
                out.setdefault((pos, t), ("taken", ""))
    return out


_AVAIL_CHIP = {"yours": ("avail-yours", "yours"), "available": ("avail-open", "available"),
               "taken": ("avail-taken", "taken")}


def _ranking_section(
    title: str, frame: pd.DataFrame, position: str, top_n: int, cfg: Config,
    availability: dict | None = None,
) -> str:
    if frame is None or frame.empty:
        return f"<h2>{_e(title)}</h2><p class='sub'>No projections available.</p>"
    out = [f"<h2>{_e(title)}</h2>"]
    for row in frame.head(top_n).itertuples():
        hold = bool(getattr(row, "two_week_hold", False))
        venue = "vs" if getattr(row, "is_home", 1) == 1 else "at"
        meta = [
            f"floor {_num(getattr(row, 'floor', np.nan))}-{_num(getattr(row, 'ceiling', np.nan))}",
            f"P(top-12) {_pct(getattr(row, 'p_top12', np.nan))}",
        ]
        if position == "DST":
            meta.append(f"proj allowed {_num(getattr(row, 'expected_points_allowed', np.nan), 0)}")
        chips = "".join(f"<span>{m}</span>" for m in meta)
        status, whose = (availability or {}).get((position, str(getattr(row, "team", ""))), ("", ""))
        card_class = ""
        if status in _AVAIL_CHIP:
            css, label = _AVAIL_CHIP[status]
            title_attr = f' title="{_e(whose)}"' if whose else ""
            chips = f'<span class="{css}"{title_attr}>{label}</span>' + chips
            if status == "taken":
                card_class = " taken"
        if hold:
            chips += (
                f'<span class="hold-tag">hold thru wk {int(getattr(row, "week", 0)) + 1}</span>'
            )
        out.append(
            f'<div class="card{" hold" if hold else ""}{card_class}">'
            f'<div class="row"><div class="rank">{int(row.rank)}</div>'
            f'<div><span class="name">{_e(row.display_name)}</span> '
            f'<span class="opp">{venue} {_e(getattr(row, "opponent", ""))}</span></div>'
            f'<div class="pts">{_num(row.expected_points)}</div></div>'
            f'<div class="meta">{chips}</div>'
            f'<div class="why">{_e(getattr(row, "rationale", ""))}</div>'
            "</div>"
        )
    return "".join(out)


def _two_week_section(rankings: Rankings, availability: dict | None = None) -> str:
    candidates = two_week_candidates(rankings)
    rows = []
    for position, frame in candidates.items():
        if frame is None or frame.empty:
            continue
        for row in frame.head(6).itertuples():
            status, _whose = (availability or {}).get((position, str(getattr(row, "team", ""))), ("", ""))
            tag = f' <span class="opp">{status}</span>' if status else ""
            rows.append(
                f"<tr><td>{_e(position)}</td><td class='unit'>{_e(row.display_name)}{tag}</td>"
                f"<td>{int(row.rank)}</td><td>{_e(getattr(row, 'opponent', ''))}</td>"
                f"<td>{int(row.next_rank) if np.isfinite(row.next_rank) else '--'}</td>"
                f"<td>{_e(getattr(row, 'next_opponent', '') or '')}</td></tr>"
            )
    if not rows:
        return (
            "<h2>Two-week stream candidates</h2>"
            "<p class='sub'>Nothing ranks inside the cutoff in both weeks -- "
            "stream week by week.</p>"
        )
    return (
        "<h2>Two-week stream candidates</h2>"
        "<p class='sub'>Favourable this week <em>and</em> next, so a waiver add "
        "can be held rather than churned.</p>"
        '<div class="scroll"><table><thead><tr><th>Pos</th><th class="unit">Unit</th>'
        "<th>Rank</th><th>Opp</th><th>Next</th><th>Next opp</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _benchmark_section(cfg: Config) -> str:
    frame = load_benchmark(cfg)
    if frame.empty:
        return (
            "<h2>D/ST and K vs Subvertadown</h2>"
            "<p class='sub'>No weeks benchmarked yet. Paste his rankings into "
            "<code>data/subvertadown_week_N.csv</code> and run "
            "<code>streamer benchmark --week N</code>.</p>"
        )
    rows = []
    for position, grp in frame.groupby("position"):
        edge = float(grp["rank_corr_edge"].mean())
        cls = "pos" if edge >= 0 else "neg"
        won = int((grp["rank_corr_edge"] > 0).sum())
        rows.append(
            f"<tr><td>{_e(position)}</td><td>{len(grp)}</td>"
            f"<td>{_num(grp['streamer_rank_corr'].mean(), 3)}</td>"
            f"<td>{_num(grp['subvertadown_rank_corr'].mean(), 3)}</td>"
            f"<td class='{cls}'>{edge:+.3f}</td>"
            f"<td>{_pct(grp['streamer_top5_hit_rate'].mean())}</td>"
            f"<td>{_pct(grp['subvertadown_top5_hit_rate'].mean())}</td>"
            f"<td>{won}/{len(grp)}</td></tr>"
        )
    return (
        "<h2>D/ST and K vs Subvertadown</h2>"
        "<p class='sub'>Rank correlation against actual finishes, over the units "
        "both systems ranked.</p>"
        '<div class="scroll"><table><thead><tr><th>Pos</th><th>Wks</th><th>streamer</th>'
        "<th>Subvertadown</th><th>Edge</th><th>Top-5 (us)</th><th>Top-5 (him)</th>"
        f"<th>Weeks won</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _calibration_section(cfg: Config, rankings: Rankings) -> str:
    history = load_history(cfg)
    if history.empty:
        return ""
    season = history[history["season"] == rankings.season]
    if season.empty:
        season = history
    rows = []
    for position, grp in season.groupby("position"):
        per_week = [spearman(g["expected_points"], g["actual_points"]) for _w, g in grp.groupby("week")]
        top5 = [
            float((g.loc[g["model_rank"] <= 5, "actual_rank"] <= cfg.startable_rank).mean())
            for _w, g in grp.groupby("week")
        ]
        rows.append(
            f"<tr><td>{_e(position)}</td><td>{grp['week'].nunique()}</td>"
            f"<td>{_num(grp['abs_error'].mean(), 2)}</td>"
            f"<td>{_num(float(np.nanmean(per_week)), 3)}</td>"
            f"<td>{_pct(float(np.nanmean(top5)))}</td></tr>"
        )
    return (
        "<h2>D/ST and K rankings, graded</h2>"
        '<div class="scroll"><table><thead><tr><th>Pos</th><th>Weeks</th><th>MAE</th>'
        f"<th>Rank corr</th><th>Top-5 hit rate</th></tr></thead><tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _ledger_section(cfg: Config) -> str:
    ledger = load_ledger(cfg)
    if ledger.empty:
        return ""
    latest = ledger.sort_values(["asof_season", "asof_week"]).groupby(
        ["position", "factor"]
    ).tail(1)
    blocks = []
    for position, grp in latest.groupby("position"):
        grp = grp.reindex(grp["r_hist"].abs().sort_values(ascending=False).index).head(10)
        rows = []
        for row in grp.itertuples():
            delta = row.weight_delta
            cls = "" if (delta is None or not np.isfinite(delta)) else ("pos" if delta > 0 else "neg")
            delta_txt = "--" if (delta is None or not np.isfinite(delta)) else f"{delta:+.3f}"
            rows.append(
                f"<tr><td>{_e(row.label)}</td><td>{_num(row.r_hist, 3)}</td>"
                f"<td>{_num(row.r_current, 3)}</td><td>{_num(row.r_trailing, 3)}</td>"
                f"<td>{_num(row.model_weight, 3)}</td>"
                f"<td class='{cls}'>{delta_txt}</td></tr>"
            )
        blocks.append(
            f"<h3>{_e(position)}</h3>"
            '<div class="scroll"><table><thead><tr><th>Factor</th><th>Hist r</th>'
            "<th>Season r</th><th>Last 4wk</th><th>Weight</th><th>Moved</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )
    return (
        "<details><summary>Factor ledger &mdash; what the model is leaning on</summary>"
        "<p class='sub'>Each factor's correlation with actual fantasy points, "
        "historically vs this season. Weights are re-derived every week from a "
        "shrinkage blend of the two.</p>"
        f"{''.join(blocks)}</details>"
    )


def _archive_section(cfg: Config) -> str:
    docs = cfg.docs_dir
    weeks = sorted(
        (int(p.stem.split("_")[1]) for p in docs.glob("week_*.html") if p.stem.split("_")[1].isdigit()),
        reverse=True,
    )
    if not weeks:
        return ""
    links = "".join(f'<a href="week_{w}.html">Week {w}</a>' for w in weeks)
    return f"<h2>Archive</h2><div class='archive'>{links}</div>"


def _footer() -> str:
    return (
        "<footer>"
        "Projections are Vegas-anchored regressions fit on 2021-2025 nflverse data, "
        "re-calibrated weekly against actual results. Floor and ceiling are the 15th "
        "and 85th percentiles of the model's own historical error at that projection "
        "level. Not betting advice."
        "</footer>"
    )


def publish_profiles(
    ranked: dict[str, Rankings],
    cfg: Config | None = None,
    team_panels: dict[str, dict[str, str] | str] | None = None,
) -> PublishResult:
    """Write ``docs/index.html`` and the week's archive copy.

    One page carries every profile, switched client-side, so a single URL on a
    phone home screen covers both leagues.
    """
    cfg = cfg or get_config()
    if not ranked:
        raise ValueError("nothing to publish")
    docs = cfg.docs_dir
    week = next(iter(ranked.values())).week
    html = render_page(ranked, cfg, team_panels)

    archive = docs / f"week_{week}.html"
    archive.write_text(html, encoding="utf-8")
    # Re-rendered so the index's archive list includes the week just written.
    index = docs / "index.html"
    index.write_text(render_page(ranked, cfg, team_panels), encoding="utf-8")

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "week": week,
        "profiles": {
            name: {
                "label": cfg.for_profile(name).profile_description,
                "season": r.season,
                "line_source": r.line_source,
                "dst": _json_rows(r.dst),
                "kicker": _json_rows(r.kicker),
            }
            for name, r in ranked.items()
        },
    }
    (docs / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    nojekyll = docs / ".nojekyll"
    if not nojekyll.exists():
        nojekyll.write_text("")
    return PublishResult(index_path=index, archive_path=archive)


def team_panels_for(
    ranked: dict[str, Rankings], cfg: Config, allow_network: bool = True
) -> dict[str, dict[str, str]]:
    """Render the My-team sections (tab key -> HTML, see
    :data:`streamer.roster.page.TABS`) for each profile with a league snapshot.

    Anything going wrong here -- no snapshot, a projection failure -- is logged
    and skipped, so the streaming page always publishes.
    """
    import logging

    from .league.store import load_snapshot, read_status
    from .roster.matchup import build_report
    from .roster.page import render_sync_failure, team_sections
    from .roster.projections import project_snapshot
    from .roster.vegas import attach as attach_vegas
    from .roster.waivers import recommend

    log = logging.getLogger(__name__)
    panels: dict[str, dict[str, str]] = {}

    # Load every league first, so the sportsbook props can be pulled once for
    # all of them: pulling per league bought the same games twice.
    loaded: list[tuple[str, Config, object, dict | None, Rankings]] = []
    for name, rankings in ranked.items():
        bound = cfg.for_profile(name)
        status = read_status(bound)
        try:
            snap = load_snapshot(bound, rankings.week)
        except FileNotFoundError:
            try:
                snap = load_snapshot(bound)   # newest, if this week's is absent
            except FileNotFoundError:
                # No snapshot at all. If a sync was attempted and failed, say
                # so on the page rather than leaving a silent gap.
                if status and not status.get("ok"):
                    panels[name] = {"hub": render_sync_failure(status, bound)}
                continue
        loaded.append((name, bound, snap, status, rankings))

    shared = None
    if loaded and allow_network:
        try:
            from .data.props import fetch_props
            from .roster.vegas import teams_for_all

            shared = fetch_props(loaded[0][1], teams=teams_for_all([s for _n, _b, s, _st, _r in loaded]))
            log.info("player props: %d games (%d fetched, %d reused from cache)",
                     shared.events, shared.requested, shared.reused)
        except Exception as exc:  # noqa: BLE001 - props are a bonus column
            log.warning("player props skipped: %s", exc)
        # A second, per-game-priced feed: more books, any day of the week.
        try:
            from datetime import UTC as _UTC
            from datetime import datetime as _dt

            from .data import sgo
            from .data.props import PropsResult

            extra, note = sgo.props_frame(loaded[0][1])
            log.info("SportsGameOdds props: %s", note)
            if not extra.empty:
                if shared is None:
                    shared = PropsResult(extra.iloc[:0], _dt.now(_UTC))
                shared.frame = sgo.merge(shared.frame, extra, loaded[0][1])
                shared.events = max(shared.events, int(shared.frame["event_id"].nunique()))
        except Exception as exc:  # noqa: BLE001 - a second source never blocks the page
            log.warning("SportsGameOdds props skipped: %s", exc)

    week_w = None
    if loaded:
        try:
            from .roster.vegas import fit_week_weights

            week_w = fit_week_weights(loaded[0][1])
            log.info("this week's blend: market %.2f (%d graded games), FantasyPros projection %.2f (%d)",
                     week_w.market, week_w.market_games, week_w.consensus, week_w.consensus_games)
        except Exception as exc:  # noqa: BLE001
            log.warning("blend weight fit failed, using the priors: %s", exc)
    # Free agents the news has close to signing: ESPN's news is asked for by
    # ESPN id, so a Yahoo player is found by name in the ESPN league.
    from .data import news
    from .roster.players import normalize_name

    espn_ids = {normalize_name(p.name): p.player_id for _n, _b, lg, _s, _r in loaded
                if lg.platform == "espn" for p in lg.all_players()}
    fp_names: dict[str, str] = {}
    if loaded:
        try:
            from .data import fantasypros as fp

            first = loaded[0]
            ros = fp.rankings(first[1], first[2].season, first[2].week, "ros")
            fp_names = {str(i): str(n) for i, n in zip(ros["fp_id"], ros["name"]) if i is not None and n}
        except Exception as exc:  # noqa: BLE001
            log.warning("FantasyPros ids for news unavailable: %s", exc)
    for name, bound, snap, _status, _rankings in loaded:
        try:
            for line in news.attach(snap, bound, espn_ids, allow_network=allow_network):
                log.info("%s news: %s", name, line)
            snap._news = news.breaking(snap, bound, espn_ids, fp_names, allow_network=allow_network)
            acted = [f"{n.name}: {n.acted}" for n in snap._news if n.acted]
            log.info("%s breaking news: %d items%s", name, len(snap._news),
                     f"; {'; '.join(acted)}" if acted else "")
        except Exception as exc:  # noqa: BLE001 - news is a bonus; never block the page
            log.warning("player news for %s skipped: %s", name, exc)
    for name, bound, snap, status, rankings in loaded:
        try:
            projected = project_snapshot(snap, bound, rankings, allow_network=allow_network)
            try:
                attach_vegas(snap, bound, allow_network=allow_network and shared is not None,
                             prefetched=shared)
            except Exception as exc:  # noqa: BLE001 - props are a bonus column
                log.warning("player props for %s skipped: %s", name, exc)
            # The FantasyPros consensus first: its projection is one of this
            # week's second forecasts, its rankings one of the season's.
            has_fp = False
            try:
                from .roster import consensus

                has_fp = consensus.attach(snap, bound) > 0
                if has_fp:
                    consensus.log_week(snap, bound)       # our own numbers, before any blend
            except Exception as exc:  # noqa: BLE001 - the consensus is a bonus input
                log.warning("FantasyPros consensus for %s skipped: %s", name, exc)
            try:
                from .roster.vegas import blend_market

                blend_market(snap, bound, weight=week_w.market if week_w else None,
                             consensus_weight=week_w.consensus if week_w else 0.0)
                snap._week_weights = week_w
            except Exception as exc:  # noqa: BLE001 - the second forecasts are bonus inputs
                log.warning("this week's blend for %s skipped: %s", name, exc)
            try:
                from .roster.vegas import log_projections

                log_projections(snap, bound)
            except Exception as exc:  # noqa: BLE001 - a log must never block the page
                log.warning("projection log for %s skipped: %s", name, exc)
            season_w = None
            if has_fp:
                try:
                    from .roster import consensus
                    from .roster.projections import load_history

                    season_w = consensus.season_weight(bound, load_history(bound))
                    moved = consensus.blend_season(snap, season_w[0])
                    snap._season_weight = season_w
                    log.info("consensus weight in season values %.2f (%d graded games); %d players moved",
                             season_w[0], season_w[1], moved)
                except Exception as exc:  # noqa: BLE001 - the consensus is a bonus input
                    log.warning("FantasyPros season blend for %s skipped: %s", name, exc)
            report = build_report(snap, bound)
            try:
                from .roster import scorecard
                from .roster.projections import load_history

                report.scorecard = scorecard.build(bound, load_history(bound), snap.platform.upper()
                                                   if snap.platform != "yahoo" else "Yahoo")
                if season_w is not None:
                    from .data.fantasypros import conf as fp_conf

                    report.scorecard.season_weight = {
                        "weight": round(season_w[0], 2), "games": season_w[1],
                        "measured": season_w[1] >= int(fp_conf(bound).get("season_min_games", 150))}
                scorecard.save(bound, report.scorecard)
            except Exception as exc:  # noqa: BLE001 - grading is a bonus section
                log.warning("scorecard for %s skipped: %s", name, exc)
            if (snap.rules or {}).get("schedule"):
                seen = None
                try:
                    from .roster import perception
                    from .roster.projections import load_history

                    seen = perception.for_snapshot(snap, load_history(bound))
                except Exception as exc:  # noqa: BLE001
                    log.warning("market view for %s skipped: %s", name, exc)
                try:
                    from .roster.title_moves import TitleEngine

                    engine = TitleEngine(snap, bound, extra=[o.player for o in report.pickups])
                    engine.market = {pid: s.value for pid, s in (seen or {}).items()}
                    report.season = engine.model.odds()
                    report.title_moves = engine.moves(5)
                    report.passed = engine.passed
                    report.waiver_rank = engine.rank
                    report.stakes = engine.model.stakes(snap.my_team.team_id)
                    try:
                        from .roster.matchup import price_pickups

                        price_pickups(report, engine)
                    except Exception as exc:  # noqa: BLE001 - pickups keep their P(win) without it
                        log.warning("pricing this week's pickups for %s skipped: %s", name, exc)
                    try:
                        from .roster.roster_view import roster_values

                        report.roster_values = roster_values(engine, report, seen)
                    except Exception as exc:  # noqa: BLE001 - a bonus section
                        log.warning("roster values for %s skipped: %s", name, exc)
                except Exception as exc:  # noqa: BLE001 - odds are a bonus section
                    log.warning("title engine for %s skipped: %s", name, exc)
                    engine = None
                if engine is not None:
                    try:
                        from .roster.waiver_plans import PlanFinder

                        report.plans = PlanFinder(engine, report.title_moves, seen=seen).find(3)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("waiver plans for %s skipped: %s", name, exc)
                    finder = None
                    try:
                        from .roster.trades import TradeFinder

                        finder = TradeFinder(snap, engine.model, seen=seen)
                        report.trades = finder.find(5)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("trade finder for %s skipped: %s", name, exc)
                    if finder is not None:
                        try:
                            from .roster import trade_eval

                            report.trade_eval = trade_eval.export(snap, engine.model, finder,
                                                                  getattr(finder, "priced", []))
                            log.info("trade evaluator for %s: %s", name, report.trade_eval.get("check"))
                        except Exception as exc:  # noqa: BLE001 - a bonus section
                            log.warning("trade evaluator for %s skipped: %s", name, exc)
            report.notes.extend(n for n in projected.notes if "not playing" in n)
            report.notes.extend(projected.lock_notes)
            moves = recommend(snap, min_gain=float(bound.raw["roster"]["waiver_min_gain"]))
            panel = team_sections(snap, report, moves, bound)
            if status and not status.get("ok") and snap.week != rankings.week:
                panel["hub"] = render_sync_failure(status, bound, stale_week=snap.week) + panel.get("hub", "")
            panels[name] = panel
        except Exception as exc:  # noqa: BLE001
            log.warning("my-team panel for %s skipped: %s", name, exc)
    return panels


def publish(rankings: Rankings, cfg: Config | None = None) -> PublishResult:
    """Publish a single profile's rankings."""
    cfg = cfg or get_config()
    return publish_profiles({cfg.profile: rankings}, cfg)


def _json_rows(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    cols = [
        c for c in ("rank", "display_name", "team", "opponent", "is_home",
                    "expected_points", "floor", "ceiling", "p_top12",
                    "expected_points_allowed", "two_week_hold", "next_rank", "rationale")
        if c in frame.columns
    ]
    out = frame[cols].copy()
    return json.loads(out.to_json(orient="records"))
