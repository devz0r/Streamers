"""Player photos and team logos for the page, from ESPN's public image CDN.

An ESPN player id gives his headshot; a Yahoo league's players are matched to
ESPN ids by name (the ESPN league's players). A D/ST shows its team's logo.
The page loads them lazily and drops any that fail, so a missing photo costs
nothing but the picture.
"""

from __future__ import annotations

from ..league.model import LeagueSnapshot

HEADSHOT = "https://a.espncdn.com/i/headshots/nfl/players/full/{id}.png"
LOGO = "https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png"
#: nflverse abbreviations whose ESPN logo slug is not simply lowercase.
LOGO_SLUG = {"LA": "lar", "LAR": "lar", "WAS": "wsh", "WSH": "wsh", "JAC": "jax"}


#: A headshot that has been on the CDN for years: the run checks it loads
#: before putting any photo on the page.
PROBE = HEADSHOT.format(id="3054211")
#: Set by the publish step when the CDN answered: pictures are used only then.
ON = False


def reachable(timeout: float = 6.0) -> bool:
    """Whether ESPN's image CDN answers with an image right now."""
    import requests

    try:
        r = requests.get(PROBE, timeout=timeout, headers={"User-Agent": "streamer-fantasy-tool/1.0"})
        return r.ok and r.headers.get("content-type", "").startswith("image/")
    except requests.RequestException:
        return False


def logo(team: str | None) -> str:
    if not team:
        return ""
    code = str(team).upper()
    return LOGO.format(slug=LOGO_SLUG.get(code, code.lower()))


def attach(snapshot: LeagueSnapshot, espn_ids: dict[str, str]) -> int:
    """Set ``photo`` on every player in the league; returns how many got one.
    ``espn_ids``: normalized name -> ESPN player id."""
    from .players import normalize_name

    n = 0
    for p in snapshot.all_players():
        if p.position in ("DST", "D/ST", "DEF"):
            p.photo = logo(p.team)
        else:
            pid = p.player_id if snapshot.platform == "espn" else espn_ids.get(normalize_name(p.name))
            p.photo = HEADSHOT.format(id=pid) if pid and str(pid).isdigit() else ""
        n += bool(p.photo)
    return n
