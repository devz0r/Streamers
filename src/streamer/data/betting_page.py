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
    body = []
    for r in rows:
        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(r["logged_at"])).total_seconds() / 3600
        body.append(
            f"<tr><td class='unit'>{_e(r['player'])}</td><td>{_e(STAT.get(r['stat'], r['stat']))}</td>"
            f"<td><b>{_e(r['side'])} {r['line']:g}</b></td><td>{_price(r['price'])}</td>"
            f"<td>{_e(BOOK.get(r['book'], r['book']))}</td><td>{r['fair']:.1%}</td>"
            f"<td><b>{r['ev']:+.1%}</b></td><td>{r['books']}</td><td>{age:.0f}h</td></tr>")
    table = ("<div class='scroll'><table><thead><tr><th class='unit'>Player</th><th>Stat</th><th>Bet</th><th>Price</th>"
             "<th>Book</th><th>Fair</th><th>EV</th><th>Books</th><th>Age</th></tr></thead><tbody>"
             + "".join(body) + "</tbody></table></div>") if body else (
        "<p class='sub'>No price beats the other books' average by anything right now"
        f" ({data.get('sides', 0)} sides priced).</p>")
    record = ""
    if rec.get("flagged"):
        clv = ("" if rec.get("clv_hit") is None else
               f" The market moved toward {rec['clv_hit']:.0%} of the {rec['clv_n']} it could be checked on "
               f"(average {rec['clv_mean'] * 100:+.1f} points).")
        record = (f"<p class='sub'><b>Record of past flags:</b> {rec['flagged']} flagged at 2%+, {rec.get('settled', 0)} "
                  f"settled ({rec.get('wins', 0)}-{rec.get('losses', 0)}), {rec.get('profit', 0):+.1f} units at one "
                  f"unit a bet, average expected {rec['mean_ev']:+.1%}.{clv}</p>")
    return (
        "<h2>Betting</h2>"
        "<p class='sub'>The best price on each side of each prop line, against what the <i>other</i> books say the "
        "chance is. Each book's two prices carry its margin; scaled to sum to one they give that book's own read, "
        "and Pinnacle, the sharpest, counts double. <b>EV</b> is the expected profit per unit staked if the other books "
        "are right. No projection of ours is in it. Needs three books quoting the same line; one-sided markets "
        "(anytime touchdown) are left out.</p>"
        f"<p class='sub'>Prices pulled with the last refresh ({built:%a %-I:%M %p} ET); props move fast, so check the "
        "book before betting. Books limit accounts that keep taking prices well off the market.</p>"
        f"{table}{record}"
        "<p class='sub'>Not advice: an edge on paper is a fraction of a percent to a few percent, and a long losing "
        "stretch is normal. The record above is the test of whether these flags are worth acting on.</p>")
