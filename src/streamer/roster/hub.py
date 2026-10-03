"""The championship hub: every move you can make, in one unit.

Lineups, D/ST and K streams, waiver claims (blocks included), multi-move
waiver plans and trades are each priced elsewhere; this puts them on one
list ranked by what they do to P(win the title).

Season moves are already priced in title odds. A move that only changes
this week -- a lineup swap, a streamed defence or kicker -- moves P(win the
week), and a win this week is worth a measured amount of title odds (the
must-win calculation: this week's game forced to a win and to a loss in
every simulated season). Their product is the move's title-odds value.

A trade adds its gain only if the other manager says yes, so it is ranked
by the expected gain (gain x chance of a yes), with the gain if accepted
shown beside it; offers of the same players are one decision, listed once.

Each move is priced on its own, against the roster as it stands; they are
not additive. Make the top one and refresh.
"""

from __future__ import annotations

from dataclasses import dataclass

from .title_moves import BLOCK_NOTE, drop_choice

KINDS = ("Lineup", "Stream", "Pickup", "Waiver", "Block", "Plan", "Trade")


@dataclass
class HubAction:
    kind: str           # one of KINDS
    headline: str
    detail: str
    gain: float         # change in P(title)
    note: str           # verdict, yes-chance, timing
    section: str        # which detail section explains it
    firm: bool = True   # False when the outcome is not yours alone (a trade offer)


def week_title_value(report) -> float | None:
    """Title odds a win this week is worth, from the must-win calculation."""
    st = getattr(report, "stakes", None)
    if st is None:
        return None
    week = int(report.snapshot.week)
    game = next((g for g in st.games if g.week == week), None)
    return None if game is None else max(game.title_win - game.title_loss, 0.0)


def _second_opinion(report, cfg):
    """"; <what the FantasyPros consensus thinks>" where it is well off our view."""
    if cfg is None:
        return lambda p: ""
    from .consensus import note, our_ranks

    ours = our_ranks(report.snapshot)

    def say(p) -> str:
        n = note(p, ours, cfg)
        return f"; {p.name}: {n}" if n else ""
    return say


def actions(report, cfg=None) -> list[HubAction]:
    """Every priced move, best first."""
    out: list[HubAction] = []
    second = _second_opinion(report, cfg)
    per_win = week_title_value(report)
    opt = report.optimisation

    # -- this week ------------------------------------------------------
    cur = report.current_win_probability
    if per_win is not None and cur is not None and opt.changes:
        dp = report.win_probability - cur
        if dp > 0:
            swaps = "; ".join(f"start {s.name}" + (f" over {b.name}" if b is not None else "")
                              for _slot, b, s in opt.changes)
            out.append(HubAction("Lineup", "Set the recommended lineup", swaps, dp * per_win,
                                 f"P(win this week) {cur:.0%} to {report.win_probability:.0%}", "lineup"))
    if per_win is not None:
        for pos, label in (("DST", "defence"), ("K", "kicker")):
            opts = report.streams.get(pos) or []
            mine = next((o for o in opts if o.current), None)
            best = next((o for o in opts if not o.current), None)
            if mine is None or best is None or best.win_probability <= mine.win_probability:
                continue
            dp = best.win_probability - mine.win_probability
            out.append(HubAction("Stream", f"Stream {best.player.name}",
                                 f"{label} for this week, over {mine.player.name}", dp * per_win,
                                 f"P(win this week) +{dp * 100:.1f}", "streams"))

    # A free agent for this week: priced as the whole move where the season
    # engine could (this week and the rest of the season, the drop
    # included), else as this week's win alone. One already on the waiver
    # list is left to that entry.
    on_list = {m.add.player_id for m in report.title_moves or []}
    for o in getattr(report, "pickups", None) or []:
        if o.player.player_id in on_list or not o.clear:
            continue
        gain = o.title_gain if o.title_gain is not None else (o.gain * per_win if per_win is not None else None)
        if gain is None:
            continue
        detail = "; ".join(filter(None, [
            o.moves_text,
            f"drop {o.drop.name}" if o.drop is not None else ""])) + second(o.player)
        out.append(HubAction("Pickup", f"Add {o.player.name} for this week", detail, gain,
                             f"P(win this week) {o.base:.0%} to {o.win_probability:.0%}"
                             + ("" if o.title_gain is not None else "; this week only"), "waivers"))

    # -- the season ------------------------------------------------------
    for m in report.title_moves or []:
        gain = m.gain_now - m.priority_cost
        blocking = m.block_value >= BLOCK_NOTE and bool(m.rival_name)
        note = {"claim": "claim now", "lean": "close call", "wait": "worth adding, but could wait"}[m.verdict]
        if m.priority_cost > 0:
            note += f"; after your waiver priority ({m.priority_cost * 100:.1f})"
        options = getattr(m, "drop_options", None)
        detail = f"drop {drop_choice(m.drop, options)}"
        if options:
            detail += " -- a toss-up, your call"
        if blocking:
            detail += f"; keeps him from {m.rival_name}"
        detail += second(m.add)
        out.append(HubAction("Block" if blocking else "Waiver", f"Add {m.add.name}", detail, gain,
                             note, "waivers"))
    for plan in getattr(report, "plans", None) or []:
        from .waiver_plans import claims_text, drops_text

        steps = f"claim {claims_text(plan)}, in that order; drop {drops_text(plan)}"
        if getattr(plan, "drop_options", None):
            steps += " (the one-of drop is a toss-up: your call; odds are for the whole plan)"
        out.append(HubAction("Plan", f"{len(plan.moves)} moves together", steps, plan.gain,
                             "make the plan" if plan.verdict == "plan" else "close call vs the best single move",
                             "waivers"))
    board = getattr(report, "trades", None)
    if board:
        # A trade adds its gain only if he says yes, so it is ranked by the
        # expected gain; and alternatives offering the same players are one
        # decision, so only the best of them is listed.
        best_by_give: dict[tuple, object] = {}
        for t in board.top + board.best + board.likely:
            key = tuple(sorted(p.player_id for p in t.give))
            if key not in best_by_give or t.p_accept * t.gain > best_by_give[key].p_accept * best_by_give[key].gain:
                best_by_give[key] = t
        for t in best_by_give.values():
            give = " + ".join(p.name for p in t.give)
            get = " + ".join(p.name for p in t.get)
            out.append(HubAction("Trade", f"Trade {give} for {get}",
                                 f"with {t.partner.name}" + "".join(second(p) for p in t.get + t.give),
                                 t.p_accept * t.gain,
                                 f"{t.gain * 100:+.1f} if he accepts, chance of a yes ~{t.p_accept:.0%}"
                                 + ("; win-win" if t.win_win else ""), "trades", firm=False))
    out.sort(key=lambda a: -a.gain)
    return [a for a in out if a.gain > 0]
