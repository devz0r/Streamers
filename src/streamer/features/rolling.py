"""Leak-free rolling priors with Bayesian shrinkage.

Every rate a model sees for game *g* is built **only** from games strictly
before *g*. The recursion below keeps an exponentially-decayed numerator and
denominator per team; the value handed to game *g* is the accumulator state
*before* game *g* is folded in, so there is no way for a label to leak into its
own feature.

Rates are then shrunk toward a (also leak-free) league mean:

``rate = (num + m * league_rate) / (den + m)``

where ``m`` is the configured prior strength converted into denominator units.
A team with three games of data therefore sits close to league average; a team
with two seasons of data sits close to its own number. This is the same
mechanism used for the per-team in-season adjustment factors required by the
self-calibration loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Per-game decay applied to the history accumulators. 0.94 gives a ~11-game
#: half-life: roughly "this season plus a fading memory of last season".
GAME_DECAY = 0.94

#: Extra decay applied when crossing a season boundary, for roster/scheme churn.
SEASON_DECAY = 0.55


@dataclass(frozen=True)
class RateSpec:
    """A shrunk rate: ``name`` = ``numerator`` / ``denominator``."""

    name: str
    numerator: str
    denominator: str
    #: Prior strength in *denominator units per game*. Resolved at build time
    #: from the league's average denominator per game times the configured
    #: prior-games count.
    prior_games: float = 6.0


#: Column marking rows for games that have not been played yet. Such rows
#: receive the prior state like any other row, but contribute nothing back to
#: the accumulator -- otherwise projecting week N would decay the priors used
#: for week N+1.
FUTURE_FLAG = "is_future"


def _decayed_prior(
    frame: pd.DataFrame,
    group_cols: list[str],
    num_col: str,
    den_col: str,
    game_decay: float = GAME_DECAY,
    season_decay: float = SEASON_DECAY,
) -> tuple[np.ndarray, np.ndarray]:
    """Exponentially-decayed sums of ``num_col``/``den_col`` over *prior* rows.

    ``frame`` must already be sorted chronologically. Returns two arrays aligned
    to ``frame``'s rows.
    """
    num = pd.to_numeric(frame[num_col], errors="coerce").fillna(0.0).to_numpy(float)
    den = pd.to_numeric(frame[den_col], errors="coerce").fillna(0.0).to_numpy(float)
    seasons = frame["season"].to_numpy()
    group_key = (
        frame[group_cols].astype(str).agg("|".join, axis=1).to_numpy()
        if group_cols
        else np.zeros(len(frame), dtype=object)
    )

    if FUTURE_FLAG in frame.columns:
        future = pd.to_numeric(frame[FUTURE_FLAG], errors="coerce").fillna(0).to_numpy() == 1
    else:
        future = np.zeros(len(frame), dtype=bool)

    out_num = np.zeros(len(frame))
    out_den = np.zeros(len(frame))
    state: dict[object, tuple[float, float, int]] = {}
    for i in range(len(frame)):
        key = group_key[i]
        acc_n, acc_d, last_season = state.get(key, (0.0, 0.0, seasons[i]))
        if seasons[i] != last_season:
            steps = int(seasons[i]) - int(last_season)
            factor = season_decay ** max(1, steps)
            acc_n *= factor
            acc_d *= factor
        out_num[i] = acc_n
        out_den[i] = acc_d
        if future[i]:
            # An unplayed game observes the prior but does not update it.
            state[key] = (acc_n, acc_d, seasons[i])
        else:
            state[key] = (acc_n * game_decay + num[i], acc_d * game_decay + den[i], seasons[i])
    return out_num, out_den


def add_shrunk_rate(
    frame: pd.DataFrame,
    spec: RateSpec,
    group_cols: list[str],
    out_col: str | None = None,
    league_group: list[str] | None = None,
) -> pd.DataFrame:
    """Add a leak-free, league-shrunk rate column to ``frame``.

    ``frame`` is expected to be sorted by ``(season, week, game_id)``.
    """
    out_col = out_col or spec.name
    frame = frame.copy()

    team_num, team_den = _decayed_prior(frame, group_cols, spec.numerator, spec.denominator)
    # The league mean is itself built only from prior games, pooled across all
    # teams, so early-season rows shrink toward last season's league rate.
    league_num, league_den = _decayed_prior(
        frame, league_group or [], spec.numerator, spec.denominator
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        league_rate = np.where(league_den > 0, league_num / league_den, np.nan)

    # Fall back to the whole-sample rate for the very first rows, where no prior
    # league history exists at all.
    global_num = pd.to_numeric(frame[spec.numerator], errors="coerce").fillna(0.0).sum()
    global_den = pd.to_numeric(frame[spec.denominator], errors="coerce").fillna(0.0).sum()
    global_rate = float(global_num / global_den) if global_den > 0 else 0.0
    league_rate = np.where(np.isnan(league_rate), global_rate, league_rate)

    den_per_game = float(
        pd.to_numeric(frame[spec.denominator], errors="coerce").fillna(0.0).mean()
    )
    m = max(1e-9, spec.prior_games * max(den_per_game, 1e-9))

    frame[out_col] = (team_num + m * league_rate) / (team_den + m)
    frame[f"{out_col}__n"] = team_den
    return frame


def add_shrunk_rates(
    frame: pd.DataFrame,
    specs: list[RateSpec],
    group_cols: list[str],
    prefix: str = "",
) -> pd.DataFrame:
    """Vectorised :func:`add_shrunk_rate` over several specs."""
    frame = frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    for spec in specs:
        frame = add_shrunk_rate(frame, spec, group_cols, out_col=f"{prefix}{spec.name}")
    return frame


def decayed_per_game(
    frame: pd.DataFrame, col: str, group_cols: list[str], out_col: str
) -> pd.DataFrame:
    """Leak-free exponentially-weighted per-game average of ``col``."""
    frame = frame.copy()
    frame["__one"] = 1.0
    num, den = _decayed_prior(frame, group_cols, col, "__one")
    league_num, league_den = _decayed_prior(frame, [], col, "__one")
    with np.errstate(divide="ignore", invalid="ignore"):
        league = np.where(league_den > 0, league_num / league_den, np.nan)
    fallback = float(pd.to_numeric(frame[col], errors="coerce").fillna(0.0).mean())
    league = np.where(np.isnan(league), fallback, league)
    m = 4.0  # four league-average pseudo-games
    frame[out_col] = (num + m * league) / (den + m)
    return frame.drop(columns="__one")


def bayesian_update(
    prior_mean: float, prior_strength: float, observed_sum: float, observed_n: float
) -> float:
    """Posterior mean of a rate under a conjugate prior.

    Used for the per-team in-season adjustment factors: a team whose sack rate
    runs hot gets credited gradually, in proportion to how much evidence it has
    actually produced, rather than instantly.
    """
    if observed_n <= 0:
        return float(prior_mean)
    return float(
        (prior_mean * prior_strength + observed_sum) / (prior_strength + observed_n)
    )
