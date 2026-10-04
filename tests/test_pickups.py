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
