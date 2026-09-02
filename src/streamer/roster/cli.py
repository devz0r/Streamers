"""In-season commands: sync, lineup, waivers, matchup, yahoo-auth.

Kept out of the main CLI module so the streaming commands stay readable; the
main parser imports :func:`register` to attach these.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

from ..config import Config
from ..league.model import short_status

log = logging.getLogger(__name__)


@dataclass
class Prepared:
    snapshot: object
    rankings: object
    report: object
    projection_report: object


def _prepare(args: argparse.Namespace, cfg: Config) -> Prepared:
    """Load the snapshot, project every player, optimise the lineup."""
    from ..league.store import load_snapshot
    from ..rankings import rank_week
    from .matchup import build_report
    from .projections import project_snapshot

    week = getattr(args, "week", None)
    snapshot = load_snapshot(cfg, week)
    rankings = rank_week(snapshot.week, snapshot.season, cfg, allow_network=not args.offline)
    projection_report = project_snapshot(snapshot, cfg, rankings, allow_network=not args.offline)
    report = build_report(snapshot, cfg)
    return Prepared(snapshot, rankings, report, projection_report)


def _fmt_player(p, width: int = 26) -> str:
    tag = ""
    if p.is_out:
        tag = " (OUT)" if not p.on_bye else " (BYE)"
    elif p.is_questionable:
        tag = f" ({short_status(p.status)})"
    name = f"{p.name}{tag}"
    return f"{name:<{width}}"[:width]


def _print_lineup_block(title: str, lineup, cfg: Config, changes=None) -> None:
    print(f"\n{title}")
    print(f"  {'Slot':<6} {'Player':<24} {'Pos':<4} {'Tm':<4} {'Proj':>6} {'±':>5}")
    changed = {p.player_id for _s, _b, p in (changes or [])}
    for slot, p in lineup.flat():
        mark = " *" if p.player_id in changed else ""
        print(f"  {slot:<6} {_fmt_player(p)} {p.position:<4} {p.team or '--':<4} "
              f"{(p.projection or 0):>6.1f} {(p.projection_sd or 0):>5.1f}{mark}")
    print(f"  {'':<6} {'expected':<24} {'':<4} {'':<4} {lineup.expected:>6.1f} {lineup.sd:>5.1f}")


def cmd_sync(args: argparse.Namespace, cfg: Config) -> int:
    from ..league.store import platform_for, sync

    codes = []
    for bound in _selected(args, cfg):
        try:
            platform = platform_for(bound)
        except ValueError as exc:
            print(f"  {bound.profile_description}: {exc}")
            continue
        try:
            path = sync(bound, week=args.week, season=args.season)
            print(f"  {bound.profile_description}: synced {platform} league -> {path}")
            codes.append(0)
        except RuntimeError as exc:
            msg = str(exc)
            if args.skip_missing and ("not set" in msg or "missing" in msg.lower()):
                print(f"  {bound.profile_description}: skipped ({msg})")
                continue
            print(f"  {bound.profile_description}: sync failed: {msg}", file=sys.stderr)
            codes.append(1)
        except Exception as exc:  # noqa: BLE001
            print(f"  {bound.profile_description}: sync failed: {exc}", file=sys.stderr)
            codes.append(1)
    return max(codes) if codes else 0


def cmd_lineup(args: argparse.Namespace, cfg: Config) -> int:
    for bound in _selected(args, cfg):
        print(f"\n{'=' * 72}\n  {bound.profile_description}")
        try:
            prep = _prepare(args, bound)
        except FileNotFoundError as exc:
            print(f"  {exc}")
            continue
        rep = prep.report
        opt = rep.optimisation
        print(f"  Week {prep.snapshot.week} vs {rep.opponent_name or '?'}: "
              f"P(win) {rep.win_probability:.0%} ({rep.verdict()})")
        if rep.current_win_probability is not None:
            print(f"  current lineup P(win) {rep.current_win_probability:.0%} -> "
                  f"recommended {opt.best_win.win_probability:.0%} "
                  f"({opt.win_gain:+.1%}), {opt.n_lineups} lineups evaluated")
        _print_lineup_block("  Recommended (max P(win)):", opt.best_win, bound, opt.changes)
        if opt.best_ev.player_ids != opt.best_win.player_ids:
            print(f"\n  Note: the max-expected-points lineup differs "
                  f"(EV {opt.best_ev.expected:.1f}, P(win) {opt.best_ev.win_probability:.0%}); "
                  f"the recommendation trades {opt.best_ev.expected - opt.best_win.expected:.1f} "
                  f"expected points for {opt.best_win.win_probability - opt.best_ev.win_probability:+.1%} win probability.")
        if opt.changes:
            print("\n  Changes from your current lineup:")
            for slot, benched, started in opt.changes:
                if benched is not None:
                    print(f"    {slot:<6} start {started.name} over {benched.name}")
                else:
                    print(f"    {slot:<6} start {started.name}")
        else:
            print("\n  Your current lineup is already optimal.")
        if opt.opponent is not None:
            print(f"\n  Opponent projected {opt.opponent.expected:.1f} ± {opt.opponent.sd:.1f}")
        for n in rep.notes + prep.projection_report.notes:
            print(f"  note: {n}")
    return 0


def cmd_waivers(args: argparse.Namespace, cfg: Config) -> int:
    from .waivers import drop_watch, recommend

    for bound in _selected(args, cfg):
        print(f"\n{'=' * 72}\n  {bound.profile_description}")
        try:
            prep = _prepare(args, bound)
        except FileNotFoundError as exc:
            print(f"  {exc}")
            continue
        moves = recommend(prep.snapshot, min_gain=float(bound.raw["roster"]["waiver_min_gain"]))
        print(f"  Week {prep.snapshot.week} waiver moves ({len(moves)}):")
        if not moves:
            print("    nothing on the wire clears the bar")
        for i, m in enumerate(moves, 1):
            drop = f" drop {m.drop.name} ({m.drop.position})" if m.drop else ""
            print(f"  {i:>2}. [{m.tag:<7}] ADD {m.add.name} ({m.add.position}, {m.add.team or '--'})"
                  f"{drop}   +{m.score:.1f}")
            print(f"       {m.reason}")
        watch = drop_watch(prep.snapshot)
        print("\n  Drop watch (lowest value on your roster):")
        for p in watch:
            print(f"    {_fmt_player(p)} {p.position:<4} proj {(p.projection or 0):.1f}  ros {(p.ros_value or 0):.1f}")
        for n in prep.projection_report.notes:
            print(f"  note: {n}")
    return 0


def cmd_matchup(args: argparse.Namespace, cfg: Config) -> int:
    for bound in _selected(args, cfg):
        print(f"\n{'=' * 72}\n  {bound.profile_description}")
        try:
            prep = _prepare(args, bound)
        except FileNotFoundError as exc:
            print(f"  {exc}")
            continue
        rep, opt = prep.report, prep.report.optimisation
        me = prep.snapshot.my_team
        print(f"  Week {prep.snapshot.week}: {me.name} ({me.wins}-{me.losses}) vs {rep.opponent_name}")
        print(f"  P(win) {rep.win_probability:.0%} -- {rep.verdict()}")
        print(f"  you   {opt.best_win.expected:6.1f} ± {opt.best_win.sd:4.1f}")
        if opt.opponent:
            print(f"  them  {opt.opponent.expected:6.1f} ± {opt.opponent.sd:4.1f}")
        swing = rep.swing_players(3)
        if swing:
            print("  swing players: " + ", ".join(f"{p.name} (±{p.projection_sd:.0f})" for p in swing))
        if opt.changes:
            print(f"  {len(opt.changes)} lineup change(s) would take you from "
                  f"{rep.current_win_probability:.0%} to {rep.win_probability:.0%} -- see `streamer lineup`")
    return 0


def cmd_yahoo_auth(args: argparse.Namespace, cfg: Config) -> int:
    """One-time browser authorisation; prints the refresh token to store."""
    import json
    import os

    from ..data.odds import _load_dotenv

    _load_dotenv(cfg.root)
    cid = os.environ.get("YAHOO_CLIENT_ID", "").strip()
    sec = os.environ.get("YAHOO_CLIENT_SECRET", "").strip()
    if not cid or not sec:
        print("Set YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET first (from developer.yahoo.com).",
              file=sys.stderr)
        return 2
    try:
        from yahoo_oauth import OAuth2
    except ImportError:
        print("pip install yahoo-fantasy-api yahoo-oauth", file=sys.stderr)
        return 2
    path = cfg.data_dir / ".yahoo_oauth.json"
    path.write_text(json.dumps({"consumer_key": cid, "consumer_secret": sec}))
    print("A browser window (or a URL to open) will ask you to approve the app.")
    print("Paste the verification code back here when prompted.\n")
    session = OAuth2(None, None, from_file=str(path))
    if not session.token_is_valid():
        session.refresh_access_token()
    data = json.loads(path.read_text())
    token = data.get("refresh_token")
    if not token:
        print("Authorisation did not produce a refresh token.", file=sys.stderr)
        return 1
    print("\nAdd this as the YAHOO_REFRESH_TOKEN secret (and to .env for local runs):\n")
    print(f"  {token}\n")
    print(f"The token file at {path} is git-ignored; it will be refreshed automatically.")
    return 0


def _selected(args: argparse.Namespace, cfg: Config) -> list[Config]:
    choice = getattr(args, "profile", None) or "all"
    names = cfg.profile_names if choice == "all" else [choice]
    return [cfg.for_profile(name) for name in names]


def register(sub: argparse._SubParsersAction, add_week) -> None:
    """Attach the in-season commands to the main parser."""
    p = sub.add_parser("sync", help="pull your league (rosters, free agents, matchup) into a snapshot")
    add_week(p)
    p.add_argument("--skip-missing", action="store_true",
                   help="skip a league whose credentials are not set instead of failing")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("lineup", help="the lineup that maximises your chance of winning this week")
    p.add_argument("--week", type=int, default=None, help="snapshot week (default: newest)")
    p.add_argument("--season", type=int, default=None)
    p.set_defaults(func=cmd_lineup)

    p = sub.add_parser("waivers", help="ranked add/drop moves from the free-agent pool")
    p.add_argument("--week", type=int, default=None, help="snapshot week (default: newest)")
    p.add_argument("--season", type=int, default=None)
    p.set_defaults(func=cmd_waivers)

    p = sub.add_parser("matchup", help="win probability and what would move it")
    p.add_argument("--week", type=int, default=None, help="snapshot week (default: newest)")
    p.add_argument("--season", type=int, default=None)
    p.set_defaults(func=cmd_matchup)

    p = sub.add_parser("yahoo-auth", help="one-time Yahoo OAuth; prints the refresh token")
    p.set_defaults(func=cmd_yahoo_auth)
