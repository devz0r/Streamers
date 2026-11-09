"""The Betting tab: this refresh's best prices, and the record of past flags."""

from __future__ import annotations

import html
import json

import pandas as pd

from ..config import Config

STAT = {"passing_yards": "passing yards", "passing_tds": "passing TDs", "passing_interceptions": "interceptions",
        "rushing_yards": "rushing yards", "receiving_yards": "receiving yards", "receptions": "receptions"}
BOOK = {"fanduel": "FanDuel", "draftkings": "DraftKings", "betmgm": "BetMGM", "caesars": "Caesars",
        "espnbet": "ESPN Bet", "bovada": "Bovada", "pinnacle": "Pinnacle", "williamhill_us": "Caesars"}


def _e(v: object) -> str:
    return html.escape(str(v))


def _price(p: float) -> str:
    return f"{p:+.0f}"


def _games_html(cfg: Config) -> str:
    """Spreads, moneylines and totals: the prices the other books say are wrong, sure bets, middles."""
    from .game_lines import side_text

    path = cfg.path("results") / "game_board.json"
    if not path.exists():
        return ""
    g = json.loads(path.read_text(encoding="utf-8")) or {}
    if not g:
        return ""

    def game(r):
        return f"{_e(r['away'])} @ {_e(r['home'])}"

    def row(r) -> str:
        tag = " <span class='hold-tag'>way off</span>" if r["kind"] == "way off" else ""
        cons = r.get("consensus")
        if cons is None:
            books = ""
        elif r["market"] == "spread":
            books = f" <span class='opp'>(books {cons if r['side'] == 'a' else -cons:+g})</span>"
        else:
            books = f" <span class='opp'>(books {cons:g})</span>"
        push = f" (push {r['push']:.0%})" if r.get("push") else ""
        return (f"<tr><td class='unit'>{game(r)}{tag}</td><td><b>Bet {_e(side_text(r))}</b>{books}</td>"
                f"<td>{_price(r['price'])} {_e(BOOK.get(r['book'], r['book']))}</td><td>{r['win']:.0%}{push}</td>"
                f"<td><b>{r['ev']:+.1%}</b></td><td>{r['books']}</td></tr>")

    rows = "".join(row(r) for r in g.get("rows", []))
    table = ("<div class='scroll'><table><thead><tr><th class='unit'>Game</th><th>Bet</th><th>Price</th><th>Wins</th>"
             "<th>EV</th><th>Books</th></tr></thead><tbody>" + rows + "</tbody></table></div>") if rows else (
        f"<p class='sub'>None right now: no spread, moneyline or total beats what the other books say by 2%+ "
        f"({g.get('quotes', 0)} quotes on {g.get('games', 0)} games checked).</p>")
    arbs = "".join(
        f"<li>{game(r)} {_e(r['market'])}{'' if r.get('line') is None else ' ' + format(r['line'], 'g')}: "
        f"{_price(r['price_a'])} {_e(BOOK.get(r['book_a'], r['book_a']))} and {_price(r['price_b'])} "
        f"{_e(BOOK.get(r['book_b'], r['book_b']))}, stake {r['share_a']:.0%} / {1 - r['share_a']:.0%}: "
        f"<b>{r['profit']:+.1%} guaranteed</b></li>" for r in g.get("arbs", []))
    mids = "".join(
        f"<li>{game(r)} {_e(r['market'])}: {r['line_a']:g} at {_price(r['price_a'])} {_e(BOOK.get(r['book_a'], r['book_a']))}"
        f" and the other side of {r['line_b']:g} at {_price(r['price_b'])} {_e(BOOK.get(r['book_b'], r['book_b']))}: "
        f"{r['between']:.0%} to land between, <b>{r['ev']:+.1%}</b> expected</li>" for r in g.get("middles", []))
    asof = ""
    if g.get("quoted_at"):
        hours = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(g["quoted_at"])).total_seconds() / 3600
        asof = (f"<p class='sub'>These prices were read {pd.Timestamp(g['quoted_at']).tz_convert('America/New_York'):%a %-I:%M %p} ET"
                f" ({hours:.0f}h before this page was built); the feed is re-read every few hours, not every refresh.</p>")
    rec = g.get("record") or {}
    record = (f"<p class='sub'><b>Record:</b> {rec['flagged']} flagged, {rec.get('wins', 0)}-{rec.get('losses', 0)}, "
              f"{rec.get('profit', 0):+.1f} units at one a bet.</p>" if rec.get("flagged") else "")
    return ("<h3>Game lines: spreads, moneylines, totals</h3><p class='sub'>The same check on the main markets. "
            "Each book's price against the other books' read; a spread or total at a different number is priced on "
            "how often NFL games end on each margin and total (since 1999) -- half a point around 3 or 7 is worth "
            "far more than elsewhere. <b>Wins</b> is the chance the bet wins at that number (a push returns the "
            "stake). No opinion of ours on who wins: main lines are the sharpest market there is, so expect few. "
            "Long-shot moneylines (under a 15% chance) are left out: the books' read overstates them.</p>"
            + asof + table
            + ("<p class='sub'><b>Sure bets:</b></p><ul class='sub'>" + arbs + "</ul>" if arbs else "")
            + ("<p class='sub'><b>Middles</b> (not guaranteed; both can win):</p><ul class='sub'>" + mids + "</ul>" if mids else "")
            + record)


def render(cfg: Config) -> str:
    path = cfg.path("results") / "betting_board.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows, rec = data.get("rows") or [], data.get("record") or {}
    built_utc = pd.Timestamp(data["built_at"])
    built = built_utc.tz_convert("America/New_York")
    def table(sub: list[dict], empty: str, verb: str = "Bet") -> str:
        body = []
        for r in sub:
            age = max((built_utc - pd.Timestamp(r["logged_at"])).total_seconds() / 3600, 0.0)
            ours = "&ndash;" if r.get("p_ours") is None else f"{r['p_ours']:.0%}"
            mean = "" if r.get("ours_mean") is None else f" <span class='opp'>(ours {r['ours_mean']:.1f})</span>"
            tag = " <span class='hold-tag'>way off</span>" if r.get("way_off") else ""
            body.append(
                f"<tr><td class='unit'>{_e(r['player'])}{tag}</td><td>{_e(STAT.get(r['stat'], r['stat']))}</td>"
                f"<td><b>{verb} {_e(r['side']).upper()} {r['line']:g}</b> <span class='opp'>(books {r['consensus']:.1f})</span>{mean}</td>"
                f"<td>{_price(r['price'])} {_e(BOOK.get(r['book'], r['book']))}</td>"
                f"<td>{r['p_market']:.0%}</td><td><b>{r['ev_market']:+.1%}</b></td><td>{ours}</td>"
                f"<td>{r['ev']:+.1%}</td><td>{r['books']}</td><td>{age:.1f}h</td></tr>")
        if not body:
            return f"<p class='sub'>{empty}</p>"
        return ("<div class='scroll'><table><thead><tr><th class='unit'>Player</th><th>Stat</th><th>Bet</th><th>Price</th>"
                "<th>Books' chance</th><th>EV (books)</th><th>Ours</th><th>EV (combined)</th><th>Books</th><th>Age</th>"
                "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")

    # A recommendation only where the price is expected to earn over time: on the books' own read, and on
    # the combined one. Where only our projection says so, it is a lean, not a recommendation, until proven.
    market = [r for r in rows if r.get("kind") in ("way off", "market") and r["ev_market"] > 0 and r["ev"] > 0]
    model = [r for r in rows if r.get("kind") == "model"]
    tables = ("<h3>Recommended: prices the other books say are wrong</h3><p class='sub'>Each one pays more than "
              "the other books' read says it should, and more than the combined read does: positive expected "
              "profit <i>if</i> the other books are right and the price is still there. Whether that has held is "
              "the record below (wins and losses, and whether the market then moved toward these bets or away "
              "from them); until it does, keep stakes small. Even a real edge loses most single bets.</p>"
              + table(market, f"None right now: no price beats what the other books say by 2%+ "
                              f"({data.get('sides', 0)} sides priced). Lines that are way off are rare and get "
                              "fixed fast; the board checks every refresh.")
              + "<h3>Where only our projection disagrees (not recommended)</h3><p class='sub'>The books agree with each "
                "other here and our per-stat projection does not; on the books' read these lose money, so they are leans, "
                "not bets. Unproven: our edge over the market has three weeks behind it, so "
                "these are graded separately below before they mean anything.</p>"
              + table(model, "None.", verb="Lean"))
    def close(k: dict) -> str:
        bits = []
        if k.get("clv_hit") is not None:
            bits.append(f"{k['clv_hit']:.0%} of {k['clv_n']} beat the closing price")
        if k.get("move_toward") is not None:
            bits.append(f"the market moved toward {k['move_toward']:.0%} and away from {k['move_away']:.0%} "
                        f"(average {k['move_mean'] * 100:+.1f} points)")
        return "; ".join(bits)

    record = ""
    if rec.get("flagged"):
        record = (f"<p class='sub'><b>Record of past flags:</b> {rec['flagged']} bets flagged (a player, stat and side "
                  f"counts once, at the first price flagged; {rec.get('prices', rec['flagged'])} prices in all), "
                  f"{rec.get('settled', 0)} settled ({rec.get('wins', 0)}-{rec.get('losses', 0)}), "
                  f"{rec.get('profit', 0):+.1f} units at one unit a bet, average expected {rec['mean_ev']:+.1%}. "
                  "<b>The close</b> is the early test of an edge: a real one beats the closing price more often than "
                  "not, and the market moves toward it."
                  f"{' Overall: ' + close(rec) + '.' if close(rec) else ''}</p>")
    by_kind = ""
    for kind, label in (("way off", "Way off"), ("market", "Books' read"), ("model", "Our projection only")):
        k = (data.get("by_kind") or {}).get(kind)
        if k and k.get("flagged"):
            by_kind += (f"<li><b>{label}</b>: {k['flagged']} flagged, {k.get('wins', 0)}-{k.get('losses', 0)}, "
                        f"{k.get('profit', 0):+.1f} units{'; ' + close(k) if close(k) else ''}</li>")
    by_kind = f"<ul class='sub'>{by_kind}</ul>" if by_kind else ""
    par = data.get("parlays") or []
    prows = "".join(
        "<tr><td class='unit'>" + "<br>".join(
            f"{_e(leg['side']).upper()} {leg['line']:g} {_e(leg['player'])} {_e(STAT.get(leg['stat'], leg['stat']))} "
            f"<span class='opp'>({_price(leg['price'])})</span>" for leg in r["legs"])
        + f"</td><td>{_e(BOOK.get(r['book'], r['book']))}</td><td>{r['win']:.0%}</td><td>{r['payout']:.1f}x</td>"
        f"<td><b>{r['ev']:+.1%}</b></td><td>{r['ev_legs']:+.1%}</td></tr>" for r in par)
    parlay = (
        "<h3>Parlays</h3><p class='sub'>Two or three of the recommended bets above, at the same book and from "
        "different games (so they don't move together). A parlay pays the legs' odds multiplied and wins only if "
        "all do, so a parlay of good prices multiplies the edge &mdash; <i>and the error in the chances</i>, and the "
        "book's margin on any leg that isn't an edge. It wins far less often, so keep the stake smaller than a "
        "single bet's. <b>Legs avg</b> is what the same legs earn as separate bets.</p>"
        + (f"<div class='scroll'><table><thead><tr><th class='unit'>Legs</th><th>Book</th><th>Wins</th><th>Pays</th>"
           f"<th>EV</th><th>Legs avg</th></tr></thead><tbody>{prows}</tbody></table></div>" if prows else
           "<p class='sub'>None right now: it needs two recommended bets at one book from different games.</p>"))
    h = data.get("hedges") or {}
    arbs = "".join(
        f"<tr><td class='unit'>{_e(r['player'])}</td><td>{_e(STAT.get(r['stat'], r['stat']))}</td>"
        f"<td>over {r['line']:g} {_price(r['over_price'])} {_e(BOOK.get(r['over_book'], r['over_book']))}<br>"
        f"under {r['line']:g} {_price(r['under_price'])} {_e(BOOK.get(r['under_book'], r['under_book']))}</td>"
        f"<td>{r['over_share']:.0%} / {1 - r['over_share']:.0%}</td><td><b>{r['profit']:+.1%}</b></td></tr>"
        for r in h.get("arbs", []))
    mids = "".join(
        f"<tr><td class='unit'>{_e(r['player'])}</td><td>{_e(STAT.get(r['stat'], r['stat']))}</td>"
        f"<td>over {r['over_line']:g} {_price(r['over_price'])} {_e(BOOK.get(r['over_book'], r['over_book']))}<br>"
        f"under {r['under_line']:g} {_price(r['under_price'])} {_e(BOOK.get(r['under_book'], r['under_book']))}</td>"
        f"<td>{r['gap']:.0%}</td><td><b>{r['ev']:+.1%}</b></td></tr>"
        for r in h.get("middles", []))
    hedge = (
        "<h3>Hedging: sure bets</h3><p class='sub'>The same line at two books, over at one and under at the other, "
        "where the two prices together pay more than the stakes whichever way it lands. Stake the split shown. "
        "They are rare and corrected in minutes; by the time a refresh shows one it may be gone, and both bets must "
        "be placed before either price moves.</p>"
        + (f"<div class='scroll'><table><thead><tr><th class='unit'>Player</th><th>Stat</th><th>Bets</th><th>Stake split</th>"
           f"<th>Guaranteed</th></tr></thead><tbody>{arbs}</tbody></table></div>" if arbs else
           f"<p class='sub'>None right now ({h.get('checked', 0)} quotes checked).</p>")
        + "<h3>Hedging: middles</h3><p class='sub'>Over at a lower line at one book, under at a higher line at another. "
        "If the result lands in between, both win; if not, one wins and one loses (about the books' margin). "
        "<b>Between</b> is the chance it lands there, from the books' consensus curve; <b>EV</b> is the expected "
        "profit per unit staked. Not guaranteed. Only positive ones are shown.</p>"
        + (f"<div class='scroll'><table><thead><tr><th class='unit'>Player</th><th>Stat</th><th>Bets</th><th>Between</th>"
           f"<th>EV</th></tr></thead><tbody>{mids}</tbody></table></div>" if mids else
           "<p class='sub'>None with positive expected profit right now.</p>"))
    return (
        "<h2>Betting</h2>"
        "<p class='sub'>Prop lines that look wrong. For each player and stat, the other books' quotes (their margin "
        "taken out; Pinnacle, the sharpest, counts double) are fitted to one curve of his outcomes, so <i>any</i> line "
        "can be priced: a book at 30.5 rushing yards when the rest sit near 72 shows as a near-certain over. "
        "<b>Ours</b> is the chance from our own projection of that stat and how much such projections miss by "
        "(2021-2025). <b>Combined</b> leans toward the books on yards (they know the matchup) and toward ours where "
        "we have been as good. <b>EV</b> is the expected profit per unit staked at that price: on the books' chance, and on the combined one. "
        "<b>Way off</b>: 10%+ on both the books' read and ours. One-sided markets (anytime touchdown) are left out.</p>"
        f"<p class='sub'>Built {built:%a %-I:%M %p} ET. <b>Age</b> is how long before that each price was last read "
        "from its feed: the feeds are re-read every few hours (one of them mostly on Sunday, Tuesday and Wednesday "
        "mornings), not every refresh, and a price more than 90 minutes behind the newest for the same player, or "
        "more than 8 hours old, is left out as stale. Props move fast, so check the book before betting. Books limit accounts that keep "
        "taking prices well off the market.</p>"
        f"{tables}{record}{by_kind}{parlay}{hedge}{_games_html(cfg)}"
        "<p class='sub'>Not advice: an edge on paper is a fraction of a percent to a few percent, and a long losing "
        "stretch is normal. The record above is the test of whether these flags are worth acting on.</p>")
