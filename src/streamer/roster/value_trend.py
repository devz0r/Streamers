"""How each player's rest-of-season value has moved since last week.

A breakout rarely shows up first as a big season value; it shows up as the
value climbing -- a bigger snap share, a teammate hurt, the market moving.
So every player with a season value (``ros_value``: points a game when he
plays, after the consensus blend) is compared with where he stood at the end
of last week, and the change is shown beside him: green up, red down,
yellow flat, deeper for a bigger move.

The change is **absolute** (points a game), not a percentage: percentages
explode at the bottom of the pool (0.5 -> 1.5 is +200%) and would paint the
waiver wire with noise, while lineups and titles are won on points a game,
so +4 for a 4-point player outweighs +1 for a 20-point star, as it should.

History: ``results/<profile>/value_history.parquet``, one row per player per
week, overwritten through the week, so last week's row is his value at the
last refresh before the week turned over. Before that file had a week,
the projection log's ``ros_value`` (``skill_log.parquet``) stands in.

When the model itself changes, every value moves at once; bump
:data:`MODEL_EPOCH` with such a change and the badges for that week say part
of the move is the model.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from ..config import Config
from ..league.model import LeagueSnapshot, PlayerRow

log = logging.getLogger(__name__)

#: Bump when a change to the model moves season values across the board.
MODEL_EPOCH = 1

#: Bands in points a game: under FLAT is no change (week-to-week noise in
#: the season value is about that size); then light, medium, strong.
FLAT = 0.3
BANDS = (1.0, 2.0)

HISTORY = "value_history.parquet"


def level(delta: float) -> int:
    """-3..3: the sign of the move and its band (0: flat)."""
    a = abs(delta)
    if a < FLAT:
        return 0
    n = 1 + sum(a >= b for b in BANDS)
    return n if delta > 0 else -n


def change(p: PlayerRow) -> float | None:
    if p.ros_value is None or p.ros_prev is None:
        return None
    return float(p.ros_value) - float(p.ros_prev)


def record(snapshot: LeagueSnapshot, cfg: Config) -> int:
    """Write this week's season values (every player who has one)."""
    import pandas as pd

    rows = {p.player_id: float(p.ros_value) for p in snapshot.all_players() if p.ros_value is not None}
    if not rows:
        return 0
    path = cfg.results_dir / HISTORY
    now = datetime.now(UTC).isoformat(timespec="seconds")
    new = pd.DataFrame({"season": int(snapshot.season), "week": int(snapshot.week),
                        "player_id": list(rows), "ros_value": list(rows.values()),
                        "epoch": MODEL_EPOCH, "logged_at": now})
    if path.exists():
        old = pd.read_parquet(path)
        keep = ~((old["season"] == snapshot.season) & (old["week"] == snapshot.week))
        new = pd.concat([old[keep], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return len(rows)


def baseline(cfg: Config, season: int, week: int) -> dict[str, tuple[float, int]]:
    """player id -> (season value, model epoch) at the end of last week."""
    import pandas as pd

    path = cfg.results_dir / HISTORY
    if path.exists():
        h = pd.read_parquet(path)
        h = h[(h["season"] == season) & (h["week"] == week - 1)]
        if len(h):
            return {str(r.player_id): (float(r.ros_value), int(r.epoch)) for r in h.itertuples()}
    path = cfg.results_dir / "skill_log.parquet"
    if path.exists():
        s = pd.read_parquet(path)
        if "ros_value" in s:
            s = s[(s["season"] == season) & (s["week"] == week - 1) & s["ros_value"].notna()]
            return {str(r.player_id): (float(r.ros_value), MODEL_EPOCH) for r in s.itertuples()}
    return {}


def attach(snapshot: LeagueSnapshot, cfg: Config) -> int:
    """Set ``ros_prev`` on every player last week's values cover."""
    base = baseline(cfg, int(snapshot.season), int(snapshot.week))
    n = 0
    for p in snapshot.all_players():
        b = base.get(str(p.player_id))
        if b is None or p.ros_value is None:
            continue
        p.ros_prev, p.ros_prev_model_changed = b[0], b[1] != MODEL_EPOCH
        n += 1
    return n


def payload(p: PlayerRow) -> list | None:
    """``[change, level, last week]`` for a page's JSON table (None: no history)."""
    d = change(p)
    return None if d is None else [round(d, 1), level(d), round(float(p.ros_prev), 1)]
