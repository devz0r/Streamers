"""ESPN fantasy scoring for Kickers and D/ST.

Pure arithmetic over stat lines -- no data access, no pandas in the hot path --
so the tier boundaries can be tested exactly. All values are read from
``config.yaml`` (see :mod:`streamer.config`); nothing here is hard-coded.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .config import Config, get_config

#: Distance buckets used throughout the project, as ``(name, low, high)`` with
#: inclusive bounds in yards. ``high`` of ``None`` means "and beyond".
FG_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("0_39", 0, 39),
    ("40_49", 40, 49),
    ("50_plus", 50, None),
)


def fg_bucket(distance: float) -> str:
    """Return the bucket name (``0_39`` / ``40_49`` / ``50_plus``) for a kick."""
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
    fg_made_50_plus: int = 0
    fg_missed_0_39: int = 0
    fg_missed_40_49: int = 0
    fg_missed_50_plus: int = 0
    pat_made: int = 0
    pat_missed: int = 0

    @property
    def fg_made(self) -> int:
        return self.fg_made_0_39 + self.fg_made_40_49 + self.fg_made_50_plus

    @property
    def fg_attempted(self) -> int:
        return (
            self.fg_made
            + self.fg_missed_0_39
            + self.fg_missed_40_49
            + self.fg_missed_50_plus
        )


@dataclass(frozen=True)
class KickerScoring:
    """ESPN kicker scoring rules."""

    fg_0_39: float
    fg_40_49: float
    fg_50_plus: float
    fg_missed_0_39: float
    fg_missed_40_49: float
    fg_missed_50_plus: float
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
    sacks: float = 0.0
    interceptions: int = 0
    fumble_recoveries: int = 0
    safeties: int = 0
    defensive_tds: int = 0
    return_tds: int = 0
    blocked_kicks: int = 0
    blocked_kick_tds: int = 0
    extra_points_returned: int = 0


@dataclass(frozen=True)
class DstScoring:
    """ESPN D/ST scoring rules, including the points-allowed tier ladder."""

    sack: float
    interception: float
    fumble_recovery: float
    safety: float
    defensive_td: float
    return_td: float
    blocked_kick: float
    blocked_kick_td: float
    extra_point_returned: float
    #: ``(low, high_or_None, points)`` with inclusive bounds, ascending.
    points_allowed_tiers: tuple[tuple[int, int | None, float], ...]
    points_allowed_excludes_opponent_dst_st: bool = False

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> DstScoring:
        raw = (cfg or get_config()).dst_scoring
        tiers = tuple(
            (int(low), None if high is None else int(high), float(pts))
            for low, high, pts in raw["points_allowed_tiers"]
        )
        _validate_tiers(tiers)
        scalar_fields = [
            f for f in cls.__dataclass_fields__
            if f not in ("points_allowed_tiers", "points_allowed_excludes_opponent_dst_st")
        ]
        return cls(
            points_allowed_tiers=tiers,
            points_allowed_excludes_opponent_dst_st=bool(
                raw.get("points_allowed_excludes_opponent_dst_st", False)
            ),
            **{f: float(raw[f]) for f in scalar_fields},
        )

    # -- tier ladder -------------------------------------------------------
    def points_allowed_points(self, points_allowed: float) -> float:
        """Tier points for a given number of points allowed."""
        pa = int(points_allowed)
        if pa < 0:
            raise ValueError(f"points allowed cannot be negative: {points_allowed}")
        for low, high, pts in self.points_allowed_tiers:
            if pa >= low and (high is None or pa <= high):
                return pts
        raise ValueError(f"no points-allowed tier matched {pa}")

    def tier_index(self, points_allowed: float) -> int:
        """Index into :attr:`points_allowed_tiers` for ``points_allowed``."""
        pa = int(points_allowed)
        for i, (low, high, _pts) in enumerate(self.points_allowed_tiers):
            if pa >= low and (high is None or pa <= high):
                return i
        raise ValueError(f"no points-allowed tier matched {pa}")

    @property
    def tier_values(self) -> tuple[float, ...]:
        return tuple(pts for _low, _high, pts in self.points_allowed_tiers)

    @property
    def tier_labels(self) -> tuple[str, ...]:
        out = []
        for low, high, _pts in self.points_allowed_tiers:
            out.append(f"{low}+" if high is None else (f"{low}" if low == high else f"{low}-{high}"))
        return tuple(out)

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
        total = self.points_allowed_points(line.points_allowed)
        total += line.sacks * self.sack
        total += line.interceptions * self.interception
        total += line.fumble_recoveries * self.fumble_recovery
        total += line.safeties * self.safety
        total += line.defensive_tds * self.defensive_td
        total += line.return_tds * self.return_td
        total += line.blocked_kicks * self.blocked_kick
        total += line.blocked_kick_tds * self.blocked_kick_td
        total += line.extra_points_returned * self.extra_point_returned
        return total

    def score_without_tier(self, line: DstStatLine) -> float:
        """Everything except the points-allowed tier.

        The DST model projects this "big-play" component with a regression and
        the tier component with a probability distribution, then adds them.
        """
        return self.score(line) - self.points_allowed_points(line.points_allowed)


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
    samples: Iterable[float], scoring: DstScoring, weights: Iterable[float] | None = None
) -> list[float]:
    """Convert a sample of opponent point totals into per-tier probabilities."""
    n_tiers = len(scoring.points_allowed_tiers)
    acc = [0.0] * n_tiers
    total_w = 0.0
    weight_iter = iter(weights) if weights is not None else None
    for value in samples:
        w = 1.0 if weight_iter is None else float(next(weight_iter))
        acc[scoring.tier_index(max(0.0, value))] += w
        total_w += w
    if total_w <= 0:
        raise ValueError("cannot build tier probabilities from an empty sample")
    return [a / total_w for a in acc]
