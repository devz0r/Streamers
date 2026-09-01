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
header { padding: 1.5rem 1rem 0.5rem; }
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
details { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); padding: .75rem .9rem; margin: .6rem 0; }
summary { cursor: pointer; font-weight: 600; font-size: .9rem; }
.archive { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .5rem; }
.archive a { font-size: .8rem; padding: .25rem .6rem; background: var(--panel-2); border: 1px solid var(--line); border-radius: 999px; text-decoration: none; }

/* Profile switch.
   Radios live at the top of <body>; the labels are styled as a segmented
   control and the :checked state drives which panel is visible. No JavaScript,
   so the switch works instantly on a phone and still works with JS disabled. */
.profile-radio { position: absolute; opacity: 0; pointer-events: none; }
.switch {
  display: flex; gap: .25rem; padding: .25rem; margin: .9rem 0 .25rem;
  background: var(--panel-2); border: 1px solid var(--line);
  border-radius: 999px; position: sticky; top: .5rem; z-index: 5;
  backdrop-filter: blur(8px);
}
.switch label {
  flex: 1; text-align: center; padding: .45rem .5rem; border-radius: 999px;
  font-size: .85rem; font-weight: 600; color: var(--muted); cursor: pointer;
  user-select: none; -webkit-tap-highlight-color: transparent;
  transition: background .12s ease, color .12s ease;
}
.switch label:hover { color: var(--text); }
.profile-panel { display: none; }
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
    ranked: dict[str, Rankings] | Rankings, cfg: Config | None = None
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
        f"<style>{STYLE}\n{_switch_rules(profiles)}</style>",
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
        f"<h1>Week {first.week} streaming rankings</h1>",
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
        parts.append(f'<div class="switch">{labels}</div>')

    for name in profiles:
        bound = cfg.for_profile(name)
        rankings = ranked[name]
        parts.append(f'<div class="profile-panel" id="panel-{_e(name)}">')
        parts.append(_badges(rankings, bound))
        parts.append(_notices(rankings))
        parts.append(_two_week_section(rankings))
        parts.append(
            _ranking_section("Defense / Special Teams", rankings.dst, "DST",
                             int(conf["top_n"]), bound)
        )
        parts.append(
            _ranking_section("Kickers", rankings.kicker, "K", int(conf["top_n"]), bound)
        )
        parts.append(_benchmark_section(bound))
        parts.append(_calibration_section(bound, rankings))
        parts.append(_ledger_section(bound))
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
    "fallback": ("", "lines"),
    "incomplete": ("warn", "incomplete lines"),
}


def _badges(rankings: Rankings, cfg: Config) -> str:
    badges = []
    if rankings.context and rankings.context.lines:
        lines = rankings.context.lines
        cls, label = _LINE_BADGE.get(lines.status, ("", "lines"))
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


def _ranking_section(
    title: str, frame: pd.DataFrame, position: str, top_n: int, cfg: Config
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
        if hold:
            chips += (
                f'<span class="hold-tag">hold thru wk {int(getattr(row, "week", 0)) + 1}</span>'
            )
        out.append(
            f'<div class="card{" hold" if hold else ""}">'
            f'<div class="row"><div class="rank">{int(row.rank)}</div>'
            f'<div><span class="name">{_e(row.display_name)}</span> '
            f'<span class="opp">{venue} {_e(getattr(row, "opponent", ""))}</span></div>'
            f'<div class="pts">{_num(row.expected_points)}</div></div>'
            f'<div class="meta">{chips}</div>'
            f'<div class="why">{_e(getattr(row, "rationale", ""))}</div>'
            "</div>"
        )
    return "".join(out)


def _two_week_section(rankings: Rankings) -> str:
    candidates = two_week_candidates(rankings)
    rows = []
    for position, frame in candidates.items():
        if frame is None or frame.empty:
            continue
        for row in frame.head(6).itertuples():
            rows.append(
                f"<tr><td>{_e(position)}</td><td class='unit'>{_e(row.display_name)}</td>"
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
            "<h2>vs Subvertadown</h2>"
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
        "<h2>vs Subvertadown</h2>"
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
        "<h2>How the model has been doing</h2>"
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
    ranked: dict[str, Rankings], cfg: Config | None = None
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
    html = render_page(ranked, cfg)

    archive = docs / f"week_{week}.html"
    archive.write_text(html, encoding="utf-8")
    # Re-rendered so the index's archive list includes the week just written.
    index = docs / "index.html"
    index.write_text(render_page(ranked, cfg), encoding="utf-8")

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
