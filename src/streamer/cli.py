"""Command-line interface.

    streamer rank      --week N      ranked D/ST and K tables for the week
    streamer update    --week N      score last week, refit, rewrite the ledger
    streamer benchmark --week N      head-to-head vs Subvertadown
    streamer backtest  --seasons A-B walk-forward validation vs a Vegas baseline
    streamer publish   --week N      render docs/index.html for phone access

Every command runs against one or more **scoring profiles** (``espn``,
``yahoo``). ``--profile`` selects one; the default is ``all``, because the two
leagues score the same games differently and you almost always want both.
Each profile keeps its own trained state, history and ledger.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import Config, get_config, load_config

log = logging.getLogger("streamer")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _format_cell(value, col: str, fmt: str | None) -> str:
    """Format one cell, choosing sensible precision from the column's meaning."""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else ""
    if value is None:
        return "--"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(value)
    if not isinstance(value, float):
        return str(value)
    if not np.isfinite(value):
        return "--"
    kind = fmt or _infer_format(col)
    if kind == "pct":
        return f"{value * 100:.0f}%"
    if kind == "corr":
        # Correlations and edges live in the third decimal; rounding to one
        # turns a real result into a row of zeroes.
        return f"{value:+.3f}"
    if kind == "mae":
        return f"{value:.2f}"
    return f"{value:.1f}"


def _infer_format(col: str) -> str:
    if col.startswith("p_") or col.endswith("hit_rate"):
        return "pct"
    if "corr" in col or col.endswith("edge"):
        return "corr"
    if "mae" in col:
        return "mae"
    return "num"


def _table(
    frame: pd.DataFrame,
    columns: list[tuple[str, ...]],
    limit: int | None = None,
) -> str:
    """Render a frame as a fixed-width table without pulling in a dependency.

    Each column is ``(name, header)`` or ``(name, header, format)`` where format
    is one of ``pct``, ``corr``, ``mae``, ``num``.
    """
    if frame is None or frame.empty:
        return "  (nothing to show)"
    view = frame.head(limit) if limit else frame
    headers = [spec[1] for spec in columns]
    rows: list[list[str]] = []
    for row in view.itertuples():
        cells = []
        for spec in columns:
            col = spec[0]
            fmt = spec[2] if len(spec) > 2 else None
            cells.append(_format_cell(getattr(row, col, None), col, fmt))
        rows.append(cells)
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(headers))
    ]
    # The final column is free-text; let it run to the edge rather than padding.
    def line(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            out.append(cell if i == len(cells) - 1 else cell.ljust(widths[i]))
        return "  " + "  ".join(out).rstrip()

    sep = "  " + "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def _print_rankings(rankings, cfg: Config, limit: int) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {cfg.profile_description}")
    print(f"\n=== Week {rankings.week} ({rankings.season}) — D/ST ===")
    print(f"    lines: {rankings.line_source}")
    print(_table(rankings.dst, [
        ("rank", "#"), ("display_name", "Defense"), ("opponent", "Opp"),
        ("expected_points", "Proj"), ("floor", "Floor"), ("ceiling", "Ceil"),
        ("p_top12", "Top12"), ("two_week_hold", "Hold"), ("rationale", "Why"),
    ], limit))

    print(f"\n=== Week {rankings.week} ({rankings.season}) — Kickers ===")
    print(_table(rankings.kicker, [
        ("rank", "#"), ("display_name", "Kicker"), ("team", "Tm"), ("opponent", "Opp"),
        ("expected_points", "Proj"), ("floor", "Floor"), ("ceiling", "Ceil"),
        ("p_top12", "Top12"), ("two_week_hold", "Hold"), ("rationale", "Why"),
    ], limit))

    from .rankings import two_week_candidates

    candidates = two_week_candidates(rankings)
    held = {p: f for p, f in candidates.items() if f is not None and not f.empty}
    print(f"\n=== Two-week stream candidates (weeks {rankings.week}-{rankings.week + 1}) ===")
    if not held:
        print("  none — nothing ranks inside the cutoff in both weeks")
    for position, frame in held.items():
        for row in frame.head(6).itertuples():
            print(
                f"  {position:3s} {row.display_name:<24s} "
                f"wk{rankings.week} #{int(row.rank):<3d} vs {getattr(row, 'opponent', '?'):<4s} "
                f"| wk{rankings.week + 1} #{int(row.next_rank):<3d} vs "
                f"{getattr(row, 'next_opponent', '?') or '?'}"
            )
    if rankings.warnings:
        print("\n  warnings:")
        for w in rankings.warnings:
            print(f"    - {w}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_rank(args: argparse.Namespace, cfg: Config) -> int:
    from .publish import publish_profiles

    ranked = {}
    for bound in _selected(args, cfg):
        ranked[bound.profile] = _rank_one(args, bound)
    if args.publish:
        result = publish_profiles(ranked, cfg)
        print(f"\n  published -> {result.index_path}")
    return 0


def _rank_one(args: argparse.Namespace, cfg: Config):
    from .calibrate import store_predictions
    from .rankings import rank_week

    rankings = rank_week(
        args.week, args.season, cfg, allow_network=not args.offline
    )
    _print_rankings(rankings, cfg, args.limit)

    if not args.no_store:
        frames = []
        for frame in (rankings.dst, rankings.kicker):
            if frame is None or frame.empty:
                continue
            keep = [c for c in (
                "season", "week", "position", "team", "opponent", "player_id",
                "player_name", "display_name", "expected_points", "floor", "ceiling",
                "p_top12", "rank", "baseline_points", "rationale", "two_week_hold",
            ) if c in frame.columns]
            frames.append(frame[keep])
        if frames:
            store_predictions(pd.concat(frames, ignore_index=True), cfg)
            print(f"\n  stored predictions -> {cfg.results_dir / 'predictions.parquet'}")
    return rankings


def cmd_update(args: argparse.Namespace, cfg: Config) -> int:
    for bound in _selected(args, cfg):
        print(f"\n{'=' * 72}\n  {bound.profile_description}")
        _update_one(args, bound)
    return 0


def _update_one(args: argparse.Namespace, cfg: Config) -> int:
    from .calibrate import run_update

    result = run_update(args.week, args.season, cfg, allow_network=not args.offline)
    if not result.scores:
        print(f"No completed results to score for week {args.week} yet.")
        print(f"  review written -> {result.report_path}")
        return 0

    print(f"\n=== Week {result.week} calibration ({result.season}) ===")
    for position, score in result.scores.items():
        print(
            f"  {position:4s} n={score.n:3d}  MAE {score.mae:5.2f} "
            f"(vegas {score.baseline_mae:5.2f})  rank corr {score.rank_corr:+.3f} "
            f"(vegas {score.baseline_rank_corr:+.3f})  top-5 hit {score.top5_hit_rate:.0%}"
        )
    for position, ledger in result.ledgers.items():
        movers = ledger.top_movers(3)
        if movers.empty:
            continue
        print(f"\n  {position} weight movers:")
        for row in movers.itertuples():
            print(f"    {row.label:<38s} {row.weight_delta:+.3f}  (r {row.r_hist:+.3f} -> {row.r_blend:+.3f})")
    print(f"\n  history  -> {cfg.results_dir / 'history.parquet'}")
    print(f"  ledger   -> {cfg.results_dir / 'factor_ledger.parquet'}")
    print(f"  review   -> {result.report_path}")
    return 0


def cmd_benchmark(args: argparse.Namespace, cfg: Config) -> int:
    codes = []
    for bound in _selected(args, cfg):
        print(f"\n{'=' * 72}\n  {bound.profile_description}")
        codes.append(_benchmark_one(args, bound))
    return 0 if any(c == 0 for c in codes) else 1


def _benchmark_one(args: argparse.Namespace, cfg: Config) -> int:
    from .benchmark import append_benchmark, compare_week, write_benchmark_report
    from .calibrate import load_stored_predictions
    from .pipeline import build_slate

    season = args.season or cfg.current_season
    context = build_slate(season, args.week, cfg, weeks_ahead=1, allow_network=not args.offline)
    actuals = {
        "DST": context.dst_train[
            (context.dst_train["season"] == season) & (context.dst_train["week"] == args.week)
        ],
        "K": context.kicker_train[
            (context.kicker_train["season"] == season) & (context.kicker_train["week"] == args.week)
        ],
    }
    if all(f.empty for f in actuals.values()):
        print(f"Week {args.week} results are not published yet; nothing to benchmark.")
        return 1

    stored = load_stored_predictions(cfg)
    predictions = {}
    for position in ("DST", "K"):
        sub = stored[
            (stored["season"] == season)
            & (stored["week"] == args.week)
            & (stored["position"] == position)
        ] if not stored.empty else pd.DataFrame()
        if sub.empty:
            from .calibrate import _reproject

            slate = context.dst_slate if position == "DST" else context.kicker_slate
            train = context.dst_train if position == "DST" else context.kicker_train
            sub = _reproject(position, slate, train, season, args.week, cfg)
        if sub is not None and not sub.empty:
            predictions[position] = sub

    try:
        result = compare_week(args.week, predictions, actuals, season, cfg)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if result.rows.empty:
        print("Nothing could be compared — check the CSV's names against the slate.")
        for position, names in result.unmatched.items():
            print(f"  unmatched {position}: {', '.join(names)}")
        return 1

    print(f"\n=== Week {args.week} vs Subvertadown ===")
    for row in result.rows.itertuples():
        verdict = "ahead" if row.rank_corr_edge > 0 else ("level" if row.rank_corr_edge == 0 else "behind")
        print(
            f"  {row.position:4s} n={row.n_matched:3d}  "
            f"streamer {row.streamer_rank_corr:+.3f} vs "
            f"Subvertadown {row.subvertadown_rank_corr:+.3f}  "
            f"({row.rank_corr_edge:+.3f}, {verdict})   "
            f"top-5 pts {row.streamer_top5_points:.1f} vs {row.subvertadown_top5_points:.1f}"
        )
    for position, names in result.unmatched.items():
        print(f"  unmatched {position} entries: {', '.join(names)}")

    append_benchmark(result.rows, cfg)
    path = write_benchmark_report(cfg)
    print(f"\n  running record -> {path}")
    return 0


def cmd_backtest(args: argparse.Namespace, cfg: Config) -> int:
    codes = [_backtest_one(args, bound) for bound in _selected(args, cfg)]
    return max(codes) if codes else 1


def _backtest_one(args: argparse.Namespace, cfg: Config) -> int:
    from .backtest import run_backtest, select_best
    from .data.nflverse import games_frame, load_pbp
    from .features.build import build_dst_features, build_kicker_features
    from .pipeline import save_selection, training_seasons

    seasons = _parse_seasons(args.seasons)
    print(f"\n{'=' * 72}\n  {cfg.profile_description}")
    print("Building features (this pulls and caches play-by-play)...")
    pbp = load_pbp(training_seasons(cfg), cfg)
    games = games_frame(cfg)
    kicker = build_kicker_features(pbp, games, cfg)
    dst = build_dst_features(pbp, games, cfg)
    print(f"  kicker rows: {len(kicker)}   dst rows: {len(dst)}")

    kinds = [args.estimator] if args.estimator else None
    print(f"Walk-forward backtesting {seasons}...")
    results = run_backtest(kicker, dst, seasons, cfg, kinds=kinds, tune=args.tune)
    if not results:
        print("No backtest results — not enough training data.", file=sys.stderr)
        return 1

    for key in sorted(results):
        result = results[key]
        summary = result.summary
        print(f"\n=== {key} ===")
        print(_table(summary.assign(season=summary["season"].astype(str)), [
            ("season", "Season"), ("weeks", "Wks"),
            ("mae", "MAE"), ("baseline_mae", "VegasMAE"),
            ("rank_corr", "RankR"), ("baseline_rank_corr", "VegasR"),
            ("rank_corr_edge", "Edge"),
            ("top5_hit_rate", "Top5"), ("baseline_top5_hit_rate", "VegasTop5"),
        ]))
        beats = result.beats_baseline()
        for position, ok in beats.items():
            mean_edge = float(summary["rank_corr_edge"].mean())
            print(
                f"  {position}: rank correlation "
                f"{'BEATS' if ok else 'does NOT beat'} the Vegas-only baseline "
                f"(mean edge {mean_edge:+.4f})"
            )

    best = select_best(results)
    print("\n=== Selected configuration ===")
    for position, choice in sorted(best.items()):
        print(f"  {position}: {json.dumps(choice, sort_keys=True)}")
    if args.save:
        save_selection(best, cfg)
        print(f"  saved -> {cfg.results_dir / 'model_selection.json'}")

    out = cfg.results_dir / "backtest_summary.csv"
    pd.concat(
        [r.summary.assign(config=k) for k, r in results.items()], ignore_index=True
    ).to_csv(out, index=False)
    print(f"  summary -> {out}")

    failures = [
        f"{k}:{p}" for k, r in results.items() for p, ok in r.beats_baseline().items() if not ok
    ]
    if args.strict and failures:
        print(f"\nFAILED the baseline gate: {', '.join(failures)}", file=sys.stderr)
        return 2
    return 0


def cmd_publish(args: argparse.Namespace, cfg: Config) -> int:
    from .publish import publish_profiles
    from .rankings import rank_week

    ranked = {}
    for bound in _selected(args, cfg):
        print(f"  ranking {bound.profile_description}...")
        ranked[bound.profile] = rank_week(
            args.week, args.season, bound, allow_network=not args.offline
        )
    result = publish_profiles(ranked, cfg)
    print(f"  wrote {result.index_path}")
    print(f"  wrote {result.archive_path}")
    for rankings in ranked.values():
        for w in rankings.warnings:
            print(f"    - {w}")
    return 0


def _selected(args: argparse.Namespace, cfg: Config) -> list[Config]:
    """Configs for the profiles this invocation should act on."""
    choice = getattr(args, "profile", None) or "all"
    names = cfg.profile_names if choice == "all" else [choice]
    return [cfg.for_profile(name) for name in names]


def _parse_seasons(text: str) -> list[int]:
    """Parse ``2023-2025`` or ``2023,2024,2025``."""
    text = str(text).strip()
    if "-" in text and "," not in text:
        start, end = text.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in text.split(",") if part.strip()]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streamer",
        description="Weekly DST and Kicker streaming rankings, Vegas-anchored and self-calibrating.",
    )
    parser.add_argument("--version", action="version", version=f"streamer {__version__}")
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument(
        "--offline", action="store_true",
        help="skip network calls; use cached data and manual CSVs only",
    )
    parser.add_argument(
        "--profile", default="all",
        help="scoring profile to run: a profile name from config.yaml, or "
             "'all' (the default) to run every configured league",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_week(p: argparse.ArgumentParser) -> None:
        p.add_argument("--week", type=int, required=True, help="NFL week number")
        p.add_argument("--season", type=int, default=None, help="override the configured season")

    rank = sub.add_parser("rank", help="ranked D/ST and K tables for a week")
    add_week(rank)
    rank.add_argument(
        "--limit", type=int, default=16,
        help="rows to print per table (0 for the whole slate)",
    )
    rank.add_argument("--publish", action="store_true", help="also write docs/index.html")
    rank.add_argument("--no-store", action="store_true", help="do not persist predictions")
    rank.set_defaults(func=cmd_rank)

    update = sub.add_parser("update", help="score a completed week and recalibrate")
    add_week(update)
    update.set_defaults(func=cmd_update)

    bench = sub.add_parser("benchmark", help="compare against Subvertadown's rankings")
    add_week(bench)
    bench.set_defaults(func=cmd_benchmark)

    back = sub.add_parser("backtest", help="walk-forward validation against a Vegas baseline")
    back.add_argument("--seasons", default="2023-2025", help="e.g. 2023-2025 or 2023,2024")
    back.add_argument("--estimator", choices=["ridge", "gbm"], default=None)
    back.add_argument("--tune", action="store_true", help="also sweep the ridge penalty")
    back.add_argument("--save", action="store_true", help="persist the winning configuration")
    back.add_argument(
        "--strict", action="store_true",
        help="exit non-zero if any position fails to beat the Vegas baseline",
    )
    back.set_defaults(func=cmd_backtest)

    pub = sub.add_parser("publish", help="render the static page into docs/")
    add_week(pub)
    pub.set_defaults(func=cmd_publish)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        logging.getLogger("streamer").setLevel(logging.INFO)
    cfg = get_config() if args.config is None else load_config(args.config)
    if args.profile != "all" and args.profile not in cfg.profile_names:
        print(
            f"error: unknown profile {args.profile!r}; available: "
            f"{', '.join(cfg.profile_names)}, or 'all'",
            file=sys.stderr,
        )
        return 2
    try:
        return int(args.func(args, cfg))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - surface a clean message, not a traceback
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
