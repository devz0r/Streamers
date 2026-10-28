"""Free agents who raise this week's P(win), and what the move costs the season."""

from __future__ import annotations

from types import SimpleNamespace as NS

from fixtures_league import SLOTS, _player, my_roster, opponent_roster, snapshot
from streamer.roster import lineup
from streamer.roster.hub import actions
from streamer.roster.matchup import this_week_pickups


def _opt(roster=None):
    return lineup.optimise(roster or my_roster(), SLOTS, opponent_roster(1.1), n_sims=6000)


def test_a_clear_upgrade_starts_and_raises_p_win_on_the_same_games():
    opt = _opt()
    star = _player(301, "WR", 19.0, 7.0, "SEA", slot="FA")
    scrub = _player(302, "WR", 2.0, 2.0, "ATL", slot="FA")
    opts = {o.player.player_id: o for o in lineup.pickup_options(opt, my_roster(), SLOTS, [star, scrub],
                                                                 n_sims=6000)}
    good = opts["301"]
    assert good.gain > 0 and good.clear
    assert any(s.player_id == "301" for _slot, _b, s in good.changes)
    assert good.base == opts["302"].base                      # one baseline, shared draws
    assert "start WR Player 301" in good.moves_text
    assert opts["302"].gain < good.gain


def test_no_pickup_whose_game_has_started_or_who_is_out():
    opt = _opt()
    locked = _player(303, "WR", 25.0, 7.0, "SEA", slot="FA")
    locked.locked = True
    out = _player(304, "WR", 25.0, 7.0, "SEA", slot="FA", status="OUT")
    assert lineup.pickup_options(opt, my_roster(), SLOTS, [locked, out], n_sims=2000) == []


def test_the_drop_never_starts_this_week_nor_has_kicked_off():
    snap = snapshot(week=3)
    snap.free_agents.append(_player(305, "WR", 19.0, 7.0, "SEA", slot="FA"))
    bench_rb = next(p for p in snap.my_team.roster if p.player_id == "4")
    bench_rb.locked = True                                    # his game has started: cannot be dropped
    opt = lineup.optimise(snap.my_team.roster, SLOTS, snap.opponent.roster, n_sims=6000)
    picks = this_week_pickups(snap, opt, n_sims=6000)
    top = next(o for o in picks if o.player.player_id == "305")
    starting = {p.player_id for _s, p in top.lineup.flat()}
    assert top.drop is not None and top.drop.player_id not in starting and not top.drop.locked
    assert all(o.gain > 0 for o in picks)


def _pick(name, gain, title=None, clear=True):
    p = NS(player_id=name, name=name, position="WR")
    return NS(player=p, base=0.40, win_probability=0.40 + gain, gain=gain, clear=clear, title_gain=title,
              drop=NS(name="Bench Guy"), moves_text=f"start {name} over Starter", changes=[1])


def test_the_hub_prices_a_pickup_as_the_whole_move():
    stakes = NS(games=[NS(week=4, title_win=0.06, title_loss=0.02)])
    report = NS(snapshot=NS(week=4), stakes=stakes, optimisation=NS(changes=[]), current_win_probability=0.4,
                win_probability=0.4, streams={}, plans=[], trades=None,
                title_moves=[NS(add=NS(player_id="Listed", name="Listed"), drop=NS(name="D"), gain_now=0.01,
                                priority_cost=0.0, block_value=0.0, rival_name="", verdict="claim")],
                pickups=[_pick("Season", 0.03, title=0.004), _pick("Week", 0.05), _pick("Costly", 0.04, -0.002),
                         _pick("Noise", 0.004, clear=False), _pick("Listed", 0.05, 0.02)])
    acts = {a.headline: a for a in actions(report) if a.kind == "Pickup"}
    assert abs(acts["Add Season for this week"].gain - 0.004) < 1e-12          # the season engine's price
    assert abs(acts["Add Week for this week"].gain - 0.05 * 0.04) < 1e-12       # this week's win alone
    assert "this week only" in acts["Add Week for this week"].note
    assert "Add Costly for this week" not in acts                               # wins the week, costs the season
    assert "Add Noise for this week" not in acts and "Add Listed for this week" not in acts


def test_when_no_pickup_helps_the_closest_are_still_shown(cfg):
    """A section that only says "none" reads as missing: the closest free
    agents tried are listed, on the waivers tab and in a line on the lineup."""
    from streamer.roster.matchup import build_report
    from streamer.roster.page import team_sections

    snap = snapshot(week=3)
    snap.free_agents = [_player(306, "WR", 9.0, 6.0, "SEA", slot="FA"), _player(307, "RB", 3.0, 3.0, "ATL", slot="FA")]
    report = build_report(snap, cfg)
    assert report.pickups == [] and report.pickup_near
    assert all(o.gain <= 0 for o in report.pickup_near)
    parts = team_sections(snap, report, [], cfg)
    assert "Closest:" in parts["waivers"] and "WR Player 306" in parts["waivers"]
    assert "No free agent clearly raises P(win) this week (closest: WR Player 306" in parts["lineup"]
    assert 'for="sec-espn-waivers"' in parts["lineup"]


def test_a_pickup_held_as_cover_says_whom_he_covers():
    """Darius Cooper's row showed only the drop: a pickup held on the bench
    for a questionable starter has no lineup change, and the line naming
    the starter he would replace was skipped."""
    from streamer.league.model import PlayerRow
    from streamer.roster.lineup import PickupOption

    q = PlayerRow(player_id="q", name="Q Starter", position="WR", team="KC")
    o = PickupOption(player=PlayerRow(player_id="c", name="Darius Cooper", position="WR", team="PHI"),
                     win_probability=0.6, base=0.56, noise=0.005, lineup=None, cover=q)
    assert "replace Q Starter" in o.moves_text
    o2 = PickupOption(player=o.player, win_probability=0.6, base=0.56, noise=0.005, lineup=None,
                      changes=[("FLEX", q, o.player)])
    assert o2.moves_text == "start Darius Cooper at FLEX over Q Starter"


def test_a_pickup_splits_its_title_odds_into_this_week_and_the_rest():
    """Net is the whole move (him in, the drop out, every week), less the
    waiver priority; this week's part is the P(win) gain times the game's
    title swing, and the rest is what is left."""
    from streamer.league.model import PlayerRow
    from streamer.roster.lineup import PickupOption
    from streamer.roster.page import _pickup_verdict

    p = PlayerRow(player_id="c", name="Caleb Douglas", position="WR", team="MIA", on_waivers=True)
    o = PickupOption(player=p, win_probability=0.338, base=0.330, noise=0.001, lineup=None,
                     title_gain=-0.002, title_noise=0.0005, week_title=0.008 * 0.09, claim_cost=0.0005)
    assert abs(o.rest_title - (-0.002 - 0.00072)) < 1e-12
    assert abs(o.title_net - (-0.0025)) < 1e-12
    assert _pickup_verdict(o).startswith("<b>pass</b>") and "helps this week, costs more later" in _pickup_verdict(o)
    o.title_gain = 0.004
    assert _pickup_verdict(o).startswith("<b>claim him</b>") and "waiver priority" in _pickup_verdict(o)
    p.on_waivers = False
    o.claim_cost = 0.0
    assert _pickup_verdict(o).startswith("<b>add him</b>")


def test_the_lineup_tab_lists_starters_an_available_player_beats(cfg):
    from streamer.roster.matchup import build_report
    from streamer.roster.page import _weak_starters

    snap = snapshot(week=3)
    report = build_report(snap, cfg)
    flex = [p for s, p in report.optimisation.best_win.flat() if p.position == "WR"]
    worst = min(flex, key=lambda p: p.week_value)
    snap.free_agents = [_player(390, "WR", worst.week_value + 2.0, 6.0, "SEA", slot="FA")]
    html = _weak_starters(report)
    assert "of your" in html and worst.name in html and "WR Player 390" in html
    snap.free_agents = []
    assert "None of your" in _weak_starters(report)


def test_a_pickup_that_loses_title_odds_now_and_gains_none_later_is_a_watch():
    from streamer.roster.title_moves import TitleMove

    base = dict(add=None, drop=None, p_base=0.105, priority_cost=0.0)
    wicks = TitleMove(p_now=0.101, p_wait=0.105, verdict="wait", **base)
    assert wicks.watch                                   # -0.4 now, +0.0 waiting
    breakout = TitleMove(p_now=0.101, p_wait=0.110, verdict="wait", **base)
    assert not breakout.watch                            # waiting for a breakout still gains
    assert not TitleMove(p_now=0.108, p_wait=0.105, verdict="lean", **base).watch


def test_the_waivers_tab_puts_real_moves_before_watch_only_ones(cfg):
    from types import SimpleNamespace as NS
    from streamer.roster.page import _title_moves

    p = lambda n: NS(name=n, position="WR", team="PHI", on_waivers=False, photo="", signals=[], ros_value=7.0, ros_prev=None)  # noqa: E731
    mk = lambda n, now, wait, v: NS(add=p(n), drop=p("D"), p_now=now, p_wait=wait, p_base=0.105, priority_cost=0.0, verdict=v,  # noqa: E731
                                    reasons=[], gain_now=now - 0.105, watch=(v == "wait" and now <= 0.105 and wait - 0.105 < 0.002),
                                    po_base=0.68, po_now=0.68, how="", later=None, drop_options=[])
    html = _title_moves(NS(title_moves=[mk("Wicks", 0.101, 0.105, "wait"), mk("Good", 0.12, 0.11, "claim")], passed=None,
                           snapshot=NS(profile="yahoo", teams=[]), plans=None))
    assert html.index("Good") < html.index("Wicks") and "Watch: not worth a move yet" in html
