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


#: Long-standing images the run checks before putting any picture on the
#: page: a team logo and Patrick Mahomes's headshot.
PROBES = ("https://a.espncdn.com/i/teamlogos/nfl/500/kc.png", HEADSHOT.format(id="3139477"))
#: Set by the publish step when the CDN answered: pictures are used only then.
ON = False
#: What the check saw, for the run log.
SEEN = ""


def reachable(timeout: float = 6.0) -> bool:
    """Whether ESPN's image CDN answers with an image right now."""
    global SEEN
    import requests

    seen = []
    for url in PROBES:
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (streamer-fantasy-tool)"})
            seen.append(f"{r.status_code} {r.headers.get('content-type', '?')}")
            if r.ok and r.headers.get("content-type", "").startswith("image/"):
                SEEN = "; ".join(seen)
                return True
        except requests.RequestException as exc:
            seen.append(type(exc).__name__)
    SEEN = "; ".join(seen)
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
