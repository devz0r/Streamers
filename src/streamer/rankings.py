"""Producing the weekly ranked tables -- ``streamer rank --week N``.

Adds the things a ranking needs to be *useful* rather than merely ordered: a
one-line reason per recommendation, and a flag for units worth holding two
weeks because the following matchup is also favourable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, get_config
from .models.ledger import multipliers_for
from .models.positions import BASELINE_COLUMN, DstModel, KickerModel, VegasBaseline
from .pipeline import SlateContext, build_slate, estimator_for
from .teams import TEAM_NAMES

log = logging.getLogger(__name__)


@dataclass
class Rankings:
    """One week's ranked D/ST and K tables, plus the context that produced them."""

    season: int
    week: int
    dst: pd.DataFrame
    kicker: pd.DataFrame
    dst_next: pd.DataFrame = field(default_factory=pd.DataFrame)
    kicker_next: pd.DataFrame = field(default_factory=pd.DataFrame)
    context: SlateContext | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def line_source(self) -> str:
        return self.context.lines.describe() if (self.context and self.context.lines) else "unknown"

    @property
    def lines_degraded(self) -> bool:
        return bool(self.context and self.context.lines and self.context.lines.is_degraded)


def rank_week(
    week: int,
    season: int | None = None,
    cfg: Config | None = None,
    allow_network: bool = True,
    context: SlateContext | None = None,
) -> Rankings:
    """Project and rank every D/ST and kicker for ``week`` (and ``week + 1``)."""
    cfg = cfg or get_config()
    season = season or cfg.current_season
    context = context or build_slate(season, week, cfg, weeks_ahead=2, allow_network=allow_network)

    warnings = list(context.lines.warnings) if context.lines else []

    dst_all = _project(
        "DST", context.dst_slate, context.dst_train, season, week, cfg
    )
    kicker_all = _project(
        "K", context.kicker_slate, context.kicker_train, season, week, cfg
    )

    dst = _finalise(dst_all[dst_all["week"] == week], "DST", cfg)
    kicker = _finalise(kicker_all[kicker_all["week"] == week], "K", cfg)
    dst_next = _finalise(dst_all[dst_all["week"] == week + 1], "DST", cfg)
    kicker_next = _finalise(kicker_all[kicker_all["week"] == week + 1], "K", cfg)

    dst = flag_two_week(dst, dst_next, "team", cfg)
    kicker = flag_two_week(kicker, kicker_next, "team", cfg)

    return Rankings(
        season=season, week=week, dst=dst, kicker=kicker,
        dst_next=dst_next, kicker_next=kicker_next,
        context=context, warnings=warnings,
    )


def _project(
    position: str, slate: pd.DataFrame, train: pd.DataFrame,
    season: int, week: int, cfg: Config
) -> pd.DataFrame:
    """Fit on everything completed and project the slate."""
    if slate.empty:
        return pd.DataFrame()
    if len(train) < int(cfg.model["min_train_rows"]):
        raise RuntimeError(
            f"only {len(train)} completed {position} games available; "
            f"need {cfg.model['min_train_rows']} to fit"
        )
    model_cls = KickerModel if position == "K" else DstModel
    kind = estimator_for(position, cfg)
    fit_target = (
        DstModel.BIG_PLAY_TARGET
        if (position == "DST" and _two_stage(cfg)) else "fantasy_points"
    )
    multipliers = multipliers_for(
        train, list(model_cls.FEATURES), fit_target, position, season, week, cfg
    )
    kwargs = {"multipliers": multipliers, "current_season": season, "kind": kind}
    if position == "DST":
        kwargs["two_stage"] = _two_stage(cfg)
    model = model_cls.fit(train, cfg, **kwargs)
    projection = model.predict(slate)
    frame = projection.frame.copy()
    baseline = VegasBaseline.fit(train, BASELINE_COLUMN[position])
    frame["baseline_points"] = baseline.predict(slate)
    frame["position"] = position
    frame["estimator"] = projection.estimator_kind
    return frame


def _two_stage(cfg: Config) -> bool:
    """Whether the D/ST point projection uses the two-stage structure.

    Set by the backtest (``results/model_selection.json``), not by hand.
    """
    from .pipeline import load_selection

    selection = load_selection(cfg).get("DST")
    if isinstance(selection, dict) and "two_stage" in selection:
        return bool(selection["two_stage"])
    return False


def _finalise(frame: pd.DataFrame, position: str, cfg: Config) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.sort_values("expected_points", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["position"] = position
    # The matchup is shown separately (its own column in the CLI, the card
    # header on the page), so the rationale carries only the reasons.
    out["matchup"] = [
        f"{'vs' if getattr(row, 'is_home', 1) == 1 else 'at'} {getattr(row, 'opponent', '?')}"
        for row in out.itertuples()
    ]
    out["rationale"] = [
        _rationale(row, position, cfg) for row in out.itertuples()
    ]
    if position == "DST":
        out["display_name"] = out["team"].map(TEAM_NAMES).fillna(out["team"])
    else:
        out["display_name"] = out.get("player_name", out["team"])
    return out


def flag_two_week(
    current: pd.DataFrame, following: pd.DataFrame, key: str, cfg: Config | None = None
) -> pd.DataFrame:
    """Mark units that rank inside the cutoff this week **and** next.

    Those are the waiver adds worth holding rather than churning.
    """
    cfg = cfg or get_config()
    cutoff = int(cfg.publish["two_week_rank_cutoff"])
    out = current.copy()
    out["next_rank"] = np.nan
    out["next_points"] = np.nan
    out["next_opponent"] = None
    out["two_week_hold"] = False
    if following is None or following.empty or out.empty:
        return out
    lookup = following.set_index(key)
    for idx, row in out.iterrows():
        if row[key] not in lookup.index:
            continue
        entry = lookup.loc[row[key]]
        if isinstance(entry, pd.DataFrame):
            entry = entry.iloc[0]
        out.at[idx, "next_rank"] = float(entry["rank"])
        out.at[idx, "next_points"] = float(entry["expected_points"])
        out.at[idx, "next_opponent"] = entry.get("opponent")
        out.at[idx, "two_week_hold"] = bool(
            row["rank"] <= cutoff and float(entry["rank"]) <= cutoff
        )
    return out


# ---------------------------------------------------------------------------
# Rationales
# ---------------------------------------------------------------------------
def _rationale(row, position: str, cfg: Config) -> str:
    """One line explaining why this unit is where it is.

    The opponent is deliberately left out: every surface that shows a rationale
    already shows the matchup, and repeating it wastes the line.
    """
    parts: list[str] = []

    if position == "DST":
        implied = getattr(row, "opp_implied_total", np.nan)
        if np.isfinite(implied):
            parts.append(f"opp implied {implied:.1f}")
        pa = getattr(row, "expected_points_allowed", np.nan)
        shutout = getattr(row, "p_shutout", np.nan)
        under14 = getattr(row, "p_under_14", np.nan)
        if np.isfinite(under14):
            parts.append(f"{under14:.0%} to hold them under 14")
        parts.extend(_dst_edges(row))
        if np.isfinite(pa) and not np.isfinite(under14):
            parts.append(f"projected {pa:.0f} allowed")
        if np.isfinite(shutout) and shutout >= 0.06:
            parts.append(f"{shutout:.0%} shutout")
    else:
        implied = getattr(row, "team_implied_total", np.nan)
        if np.isfinite(implied):
            parts.append(f"implied {implied:.1f}")
        parts.extend(_kicker_edges(row, cfg))

    return ", ".join(parts) if parts else "no standout matchup edge"


def _dst_edges(row) -> list[str]:
    """The two or three matchup facts that most justify a D/ST ranking."""
    notes: list[str] = []
    sack_allowed = getattr(row, "opp_sack_rate_allowed", np.nan)
    if np.isfinite(sack_allowed) and sack_allowed >= 0.075:
        notes.append(f"opp sacked on {sack_allowed:.1%} of dropbacks")
    pressure = getattr(row, "opp_pressure_rate_allowed", np.nan)
    if np.isfinite(pressure) and pressure >= 0.22:
        notes.append(f"leaky pass protection ({pressure:.0%} pressure allowed)")
    int_rate = getattr(row, "opp_int_rate", np.nan)
    if np.isfinite(int_rate) and int_rate >= 0.028:
        notes.append(f"opp INT rate {int_rate:.1%}")
    spread = getattr(row, "team_spread", np.nan)
    if np.isfinite(spread) and spread >= 5:
        notes.append(f"{abs(spread):.1f}-point favourite")
    elif np.isfinite(spread) and spread <= -7:
        notes.append(f"{abs(spread):.1f}-point underdog")
    return notes[:3]


def _kicker_edges(row, cfg: Config) -> list[str]:
    notes: list[str] = []
    fga = getattr(row, "off_fga_per_drive", np.nan)
    if np.isfinite(fga) and fga >= 0.34:
        notes.append(f"{fga:.2f} FG attempts per drive")
    rz = getattr(row, "off_rz_td_rate", np.nan)
    if np.isfinite(rz) and rz <= 0.52:
        notes.append(f"stalls in the red zone ({rz:.0%} TD rate)")
    dome = getattr(row, "is_dome", 0)
    wind = getattr(row, "wind", np.nan)
    threshold = float(cfg.weather["high_wind_threshold"])
    if dome == 1:
        notes.append("indoors")
    elif np.isfinite(wind) and wind >= threshold:
        notes.append(f"{wind:.0f} mph wind")
    long_share = getattr(row, "k_long_att_share", np.nan)
    if np.isfinite(long_share) and long_share >= 0.20:
        notes.append(f"{long_share:.0%} of attempts from 50+")
    return notes[:3]


def two_week_candidates(rankings: Rankings) -> dict[str, pd.DataFrame]:
    """The units flagged as worth holding across both weeks."""
    out = {}
    for position, frame in (("DST", rankings.dst), ("K", rankings.kicker)):
        if frame.empty or "two_week_hold" not in frame.columns:
            out[position] = pd.DataFrame()
            continue
        held = frame[frame["two_week_hold"]].copy()
        out[position] = held.sort_values(["rank", "next_rank"])
    return out
