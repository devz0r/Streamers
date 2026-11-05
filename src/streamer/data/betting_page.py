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


def render(cfg: Config) -> str:
    path = cfg.path("results") / "betting_board.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows, rec = data.get("rows") or [], data.get("record") or {}
    built = pd.Timestamp(data["built_at"]).tz_convert("America/New_York")
    def table(sub: list[dict], empty: str, verb: str = "Bet") -> str:
        body = []
        for r in sub:
            age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(r["logged_at"])).total_seconds() / 3600
            ours = "&ndash;" if r.get("p_ours") is None else f"{r['p_ours']:.0%}"
            mean = "" if r.get("ours_mean") is None else f" <span class='opp'>(ours {r['ours_mean']:.1f})</span>"
            tag = " <span class='hold-tag'>way off</span>" if r.get("way_off") else ""
            body.append(
                f"<tr><td class='unit'>{_e(r['player'])}{tag}</td><td>{_e(STAT.get(r['stat'], r['stat']))}</td>"
                f"<td><b>{verb} {_e(r['side']).upper()} {r['line']:g}</b> <span class='opp'>(books {r['consensus']:.1f})</span>{mean}</td>"
                f"<td>{_price(r['price'])} {_e(BOOK.get(r['book'], r['book']))}</td>"
                f"<td>{r['p_market']:.0%}</td><td><b>{r['ev_market']:+.1%}</b></td><td>{ours}</td>"
                f"<td>{r['ev']:+.1%}</td><td>{r['books']}</td><td>{age:.0f}h</td></tr>")
        if not body:
            return f"<p class='sub'>{empty}</p>"
        return ("<div class='scroll'><table><thead><tr><th class='unit'>Player</th><th>Stat</th><th>Bet</th><th>Price</th>"
                "<th>Books' chance</th><th>EV (books)</th><th>Ours</th><th>EV (combined)</th><th>Books</th><th>Age</th>"
                "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")

    # A recommendation only where the price is expected to earn over time: on the books' own read, and on
    # the combined one. Where only our projection says so, it is a lean, not a recommendation, until proven.
    market = [r for r in rows if r.get("kind") in ("way off", "market") and r["ev_market"] > 0 and r["ev"] > 0]
    model = [r for r in rows if r.get("kind") == "model"]
    tables = ("<h3>Recommended: prices the other books say are wrong</h3><p class='sub'>Each one is expected to "
              "earn over time, on the other books' read and on the combined one. Even so most single bets here "
              "lose; the edge shows over many.</p>"
              + table(market, f"None right now: no price beats what the other books say by 2%+ "
                              f"({data.get('sides', 0)} sides priced). Lines that are way off are rare and get "
                              "fixed fast; the board checks every refresh.")
              + "<h3>Where only our projection disagrees (not recommended)</h3><p class='sub'>The books agree with each "
                "other here and our per-stat projection does not; on the books' read these lose money, so they are leans, "
                "not bets. Unproven: our edge over the market has three weeks behind it, so "
                "these are graded separately below before they mean anything.</p>"
              + table(model, "None.", verb="Lean"))
    record = ""
    if rec.get("flagged"):
        clv = ("" if rec.get("clv_hit") is None else
               f" The market moved toward {rec['clv_hit']:.0%} of the {rec['clv_n']} it could be checked on "
               f"(average {rec['clv_mean'] * 100:+.1f} points).")
        record = (f"<p class='sub'><b>Record of past flags:</b> {rec['flagged']} flagged at 2%+ combined, {rec.get('settled', 0)} "
                  f"settled ({rec.get('wins', 0)}-{rec.get('losses', 0)}), {rec.get('profit', 0):+.1f} units at one "
                  f"unit a bet, average expected {rec['mean_ev']:+.1%}.{clv}</p>")
    by_kind = ""
    for kind, label in (("way off", "Way off"), ("market", "Books' read"), ("model", "Our projection only")):
        k = (data.get("by_kind") or {}).get(kind)
        if k and k.get("flagged"):
            clv = "" if k.get("clv_hit") is None else f", market moved its way {k['clv_hit']:.0%} of {k['clv_n']}"
            by_kind += (f"<li><b>{label}</b>: {k['flagged']} flagged, {k.get('wins', 0)}-{k.get('losses', 0)}, "
                        f"{k.get('profit', 0):+.1f} units{clv}</li>")
    by_kind = f"<ul class='sub'>{by_kind}</ul>" if by_kind else ""
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
        f"<p class='sub'>Prices pulled with the last refresh ({built:%a %-I:%M %p} ET); props move fast, so check the "
        "book before betting. Books limit accounts that keep taking prices well off the market.</p>"
        f"{tables}{record}{by_kind}{hedge}"
        "<p class='sub'>Not advice: an edge on paper is a fraction of a percent to a few percent, and a long losing "
        "stretch is normal. The record above is the test of whether these flags are worth acting on.</p>")
