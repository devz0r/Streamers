"""Fantasy scoring for Kickers and D/ST, for any configured league profile.

Pure arithmetic over stat lines -- no data access, no pandas in the hot path --
so every tier boundary can be tested exactly. All values are read from the
active profile in ``config.yaml`` (see :mod:`streamer.config`); nothing here is
hard-coded.

The two shipped profiles differ in more than magnitudes. ESPN scores a sack at
2.5 and adds a **yards-allowed** ladder on top of points allowed; Yahoo scores
a sack at 1, has no yards component, awards a point per **fourth-down stop**,
and applies no penalty for a missed field goal. Both ladders are expressed
here as generic step functions so a third profile needs only config.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .config import Config, get_config

#: Distance buckets used for *scoring*, as ``(name, low, high)`` with inclusive
#: bounds in yards. ``high`` of ``None`` means "and beyond". This is the finest
#: partition any configured profile needs: ESPN pays 6 for a 60-yarder, Yahoo
#: pays the same 5 it pays for a 50-yarder.
FG_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("0_39", 0, 39),
    ("40_49", 40, 49),
    ("50_59", 50, 59),
    ("60_plus", 60, None),
)

#: Distance buckets used for *modelling* a kicker's accuracy. Coarser than the
#: scoring buckets on purpose: 60-yard attempts happen a handful of times a
#: season league-wide, so a per-kicker 60+ accuracy rate would be noise. Each
#: entry maps a feature bucket to the scoring buckets it absorbs.
FG_FEATURE_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("0_39", ("0_39",)),
    ("40_49", ("40_49",)),
    ("50_plus", ("50_59", "60_plus")),
)


def fg_bucket(distance: float) -> str:
    """Return the scoring bucket name for a kick of ``distance`` yards."""
    for name, low, high in FG_BUCKETS:
        if distance >= low and (high is None or distance <= high):
            return name
    raise ValueError(f"field goal distance out of range: {distance}")


# ---------------------------------------------------------------------------
# Kicker
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KickerStatLine:
    """A kicker's counting stats for one game."""

    fg_made_0_39: int = 0
    fg_made_40_49: int = 0
    fg_made_50_59: int = 0
    fg_made_60_plus: int = 0
    fg_missed_0_39: int = 0
    fg_missed_40_49: int = 0
    fg_missed_50_59: int = 0
    fg_missed_60_plus: int = 0
    pat_made: int = 0
    pat_missed: int = 0

    @property
    def fg_made(self) -> int:
        return sum(getattr(self, f"fg_made_{n}") for n, _l, _h in FG_BUCKETS)

    @property
    def fg_missed(self) -> int:
        return sum(getattr(self, f"fg_missed_{n}") for n, _l, _h in FG_BUCKETS)

    @property
    def fg_attempted(self) -> int:
        return self.fg_made + self.fg_missed


@dataclass(frozen=True)
class KickerScoring:
    """Kicker scoring rules for one league profile."""

    fg_0_39: float
    fg_40_49: float
    fg_50_59: float
    fg_60_plus: float
    fg_missed_0_39: float
    fg_missed_40_49: float
    fg_missed_50_59: float
    fg_missed_60_plus: float
    pat_made: float
    pat_missed: float

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> KickerScoring:
        raw = (cfg or get_config()).kicker_scoring
        return cls(**{f: float(raw[f]) for f in cls.__dataclass_fields__})

    def made_value(self, bucket: str) -> float:
        return float(getattr(self, f"fg_{bucket}"))

    def missed_value(self, bucket: str) -> float:
        return float(getattr(self, f"fg_missed_{bucket}"))

    def score(self, line: KickerStatLine) -> float:
        """Fantasy points for a kicker stat line."""
        total = 0.0
        for bucket, _low, _high in FG_BUCKETS:
            total += getattr(line, f"fg_made_{bucket}") * self.made_value(bucket)
            total += getattr(line, f"fg_missed_{bucket}") * self.missed_value(bucket)
        total += line.pat_made * self.pat_made
        total += line.pat_missed * self.pat_missed
        return total

    def expected_points(
        self,
        expected_fg_attempts: Mapping[str, float],
        make_prob: Mapping[str, float],
        expected_pat_attempts: float,
        pat_make_prob: float,
    ) -> float:
        """Expected points from expected attempt volume and make rates.

        ``expected_fg_attempts`` and ``make_prob`` are keyed by bucket name.
        This is the structural (component) kicker projection; the regression
        models in :mod:`streamer.models` are validated against it.
        """
        total = 0.0
        for bucket, _low, _high in FG_BUCKETS:
            attempts = float(expected_fg_attempts.get(bucket, 0.0))
            p = float(make_prob.get(bucket, 0.0))
            total += attempts * (p * self.made_value(bucket) + (1.0 - p) * self.missed_value(bucket))
        total += expected_pat_attempts * (
            pat_make_prob * self.pat_made + (1.0 - pat_make_prob) * self.pat_missed
        )
        return total


# ---------------------------------------------------------------------------
# D/ST
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DstStatLine:
    """A defense/special-teams unit's counting stats for one game."""

    points_allowed: int = 0
    #: Total yards allowed. Scored by ESPN, ignored by Yahoo.
    yards_allowed: int = 0
    sacks: float = 0.0
    interceptions: int = 0
    fumble_recoveries: int = 0
    safeties: int = 0
    one_point_safeties: int = 0
    defensive_tds: int = 0
    return_tds: int = 0
    blocked_kicks: int = 0
    blocked_kick_tds: int = 0
    extra_points_returned: int = 0
    #: Offence stopped on a fourth-down conversion attempt. Scored by Yahoo.
    fourth_down_stops: int = 0


#: Scalar (per-event) D/ST scoring fields, as opposed to the tier ladders.
DST_EVENT_FIELDS: tuple[str, ...] = (
    "sack", "interception", "fumble_recovery", "safety", "one_point_safety",
    "defensive_td", "return_td", "blocked_kick", "blocked_kick_td",
    "extra_point_returned", "fourth_down_stop",
)


@dataclass(frozen=True)
class DstScoring:
    """D/ST scoring rules for one league profile.

    Carries up to two step-function ladders: points allowed (every profile) and
    yards allowed (ESPN only -- ``None`` where the league does not score it).
    """

    sack: float
    interception: float
    fumble_recovery: float
    safety: float
    one_point_safety: float
    defensive_td: float
    return_td: float
    blocked_kick: float
    blocked_kick_td: float
    extra_point_returned: float
    fourth_down_stop: float
    #: ``(low, high_or_None, points)`` with inclusive bounds, ascending.
    points_allowed_tiers: tuple[tuple[int, int | None, float], ...]
    #: Same shape, or ``None`` when the profile does not score yards allowed.
    yards_allowed_tiers: tuple[tuple[int, int | None, float], ...] | None = None
    points_allowed_excludes_opponent_dst_st: bool = False

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> DstScoring:
        raw = (cfg or get_config()).dst_scoring
        tiers = _parse_tiers(raw["points_allowed_tiers"])
        _validate_tiers(tiers)
        yards = _parse_tiers(raw.get("yards_allowed_tiers"))
        if yards is not None:
            _validate_tiers(yards)
        return cls(
            points_allowed_tiers=tiers,
            yards_allowed_tiers=yards,
            points_allowed_excludes_opponent_dst_st=bool(
                raw.get("points_allowed_excludes_opponent_dst_st", False)
            ),
            # A profile that does not score an event may simply omit it.
            **{f: float(raw.get(f, 0.0)) for f in DST_EVENT_FIELDS},
        )

    @property
    def scores_yards_allowed(self) -> bool:
        return self.yards_allowed_tiers is not None

    # -- tier ladders ------------------------------------------------------
    def points_allowed_points(self, points_allowed: float) -> float:
        """Tier points for a given number of points allowed."""
        return _ladder_value(self.points_allowed_tiers, points_allowed, "points allowed")

    def yards_allowed_points(self, yards_allowed: float) -> float:
        """Tier points for total yards allowed (0 where the league ignores it)."""
        if self.yards_allowed_tiers is None:
            return 0.0
        return _ladder_value(self.yards_allowed_tiers, yards_allowed, "yards allowed")

    def tier_index(self, points_allowed: float) -> int:
        """Index into :attr:`points_allowed_tiers` for ``points_allowed``."""
        return _ladder_index(self.points_allowed_tiers, points_allowed, "points allowed")

    def yards_tier_index(self, yards_allowed: float) -> int:
        if self.yards_allowed_tiers is None:
            raise ValueError("this profile does not score yards allowed")
        return _ladder_index(self.yards_allowed_tiers, yards_allowed, "yards allowed")

    @property
    def tier_values(self) -> tuple[float, ...]:
        return tuple(pts for _low, _high, pts in self.points_allowed_tiers)

    @property
    def tier_labels(self) -> tuple[str, ...]:
        return _ladder_labels(self.points_allowed_tiers)

    @property
    def yards_tier_values(self) -> tuple[float, ...]:
        if self.yards_allowed_tiers is None:
            return ()
        return tuple(pts for _low, _high, pts in self.yards_allowed_tiers)

    @property
    def yards_tier_labels(self) -> tuple[str, ...]:
        if self.yards_allowed_tiers is None:
            return ()
        return _ladder_labels(self.yards_allowed_tiers)

    def expected_yards_tier_points(self, tier_probs: Sequence[float]) -> float:
        """Expected yards-allowed points given a probability per tier."""
        values = self.yards_tier_values
        if not values:
            return 0.0
        if len(tier_probs) != len(values):
            raise ValueError(
                f"expected {len(values)} yards tier probabilities, got {len(tier_probs)}"
            )
        return float(sum(p * v for p, v in zip(tier_probs, values)))

    def expected_tier_points(self, tier_probs: Sequence[float]) -> float:
        """Expected tier points given a probability per tier.

        This is the required "tiers as probabilities, not a point estimate"
        path: the DST model produces a distribution over opponent points and
        this collapses it to expected fantasy points.
        """
        values = self.tier_values
        if len(tier_probs) != len(values):
            raise ValueError(
                f"expected {len(values)} tier probabilities, got {len(tier_probs)}"
            )
        return float(sum(p * v for p, v in zip(tier_probs, values)))

    # -- full line ---------------------------------------------------------
    def score(self, line: DstStatLine) -> float:
        """Fantasy points for a D/ST stat line."""
        return (
            self.points_allowed_points(line.points_allowed)
            + self.yards_allowed_points(line.yards_allowed)
            + self.score_big_plays(line)
        )

    def score_big_plays(self, line: DstStatLine) -> float:
        """Everything except the points- and yards-allowed ladders.

        The D/ST model projects this component with a regression and each
        ladder with its own probability distribution, then adds them.
        """
        return (
            line.sacks * self.sack
            + line.interceptions * self.interception
            + line.fumble_recoveries * self.fumble_recovery
            + line.safeties * self.safety
            + line.one_point_safeties * self.one_point_safety
            + line.defensive_tds * self.defensive_td
            + line.return_tds * self.return_td
            + line.blocked_kicks * self.blocked_kick
            + line.blocked_kick_tds * self.blocked_kick_td
            + line.extra_points_returned * self.extra_point_returned
            + line.fourth_down_stops * self.fourth_down_stop
        )

    #: Retained under its old name for callers that predate the yards ladder.
    score_without_tier = score_big_plays


def _parse_tiers(raw) -> tuple[tuple[int, int | None, float], ...] | None:
    """Parse a ladder from config; ``None``/empty means the league ignores it."""
    if not raw:
        return None
    return tuple(
        (int(low), None if high is None else int(high), float(pts))
        for low, high, pts in raw
    )


def _ladder_value(
    tiers: Sequence[tuple[int, int | None, float]], value: float, what: str
) -> float:
    return tiers[_ladder_index(tiers, value, what)][2]


def _ladder_index(
    tiers: Sequence[tuple[int, int | None, float]], value: float, what: str
) -> int:
    v = int(value)
    if v < 0:
        raise ValueError(f"{what} cannot be negative: {value}")
    for i, (low, high, _pts) in enumerate(tiers):
        if v >= low and (high is None or v <= high):
            return i
    raise ValueError(f"no {what} tier matched {v}")


def _ladder_labels(tiers: Sequence[tuple[int, int | None, float]]) -> tuple[str, ...]:
    out = []
    for low, high, _pts in tiers:
        out.append(f"{low}+" if high is None else (f"{low}" if low == high else f"{low}-{high}"))
    return tuple(out)


def _validate_tiers(tiers: Sequence[tuple[int, int | None, float]]) -> None:
    """Reject a ladder with gaps, overlaps or a missing open top bucket."""
    if not tiers:
        raise ValueError("points_allowed_tiers must not be empty")
    if tiers[0][0] != 0:
        raise ValueError("points_allowed_tiers must start at 0")
    for i, (low, high, _pts) in enumerate(tiers):
        if high is not None and high < low:
            raise ValueError(f"tier {i} has high < low: {low}..{high}")
        if i + 1 < len(tiers):
            if high is None:
                raise ValueError("only the final tier may be open-ended")
            if tiers[i + 1][0] != high + 1:
                raise ValueError(
                    f"tier ladder is not contiguous between {tiers[i]} and {tiers[i + 1]}"
                )
    if tiers[-1][1] is not None:
        raise ValueError("the final points-allowed tier must be open-ended (null high)")


def tier_probabilities_from_samples(
    samples: Iterable[float],
    scoring: DstScoring,
    weights: Iterable[float] | None = None,
    ladder: str = "points",
) -> list[float]:
    """Convert a sample of realised totals into per-tier probabilities.

    ``ladder`` selects ``"points"`` or ``"yards"``.
    """
    tiers = scoring.points_allowed_tiers if ladder == "points" else scoring.yards_allowed_tiers
    if tiers is None:
        raise ValueError("this profile does not score yards allowed")
    index = scoring.tier_index if ladder == "points" else scoring.yards_tier_index
    acc = [0.0] * len(tiers)
    total_w = 0.0
    weight_iter = iter(weights) if weights is not None else None
    for value in samples:
        w = 1.0 if weight_iter is None else float(next(weight_iter))
        acc[index(max(0.0, value))] += w
        total_w += w
    if total_w <= 0:
        raise ValueError("cannot build tier probabilities from an empty sample")
    return [a / total_w for a in acc]
