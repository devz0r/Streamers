"""Factor-correlation ledger.

For every input factor this tracks its rolling correlation with actual fantasy
outcomes across three windows -- full historical, current season to date, and
trailing N weeks -- and turns that into the feature weighting the next refit
uses.

The blend is a standard conjugate shrinkage of the current-season correlation
toward the historical one:

    r_blend = (n_cur * r_cur + k * r_hist) / (n_cur + k)

``k`` is a fixed pseudo-count (``ledger.prior_strength``), so the prior's
influence decays automatically as the season's sample grows: in Week 2 the
historical value dominates; by Week 12 the current season does. A factor whose
current-season correlation has genuinely diverged -- a rules change making
sacks scarcer, say -- therefore pulls its own weight down without anyone
touching the code.

The multiplier handed to the models has two parts, both derived from data:

``signal_prior``
    How strong this factor's correlation is *relative to the other factors*,
    ``clip(|r_full| / median|r_full|, 0.2, 2.0)``. This is what stops twenty
    weak factors from collectively drowning out the implied total. It is
    measured over **every** completed game available to the refit, not just
    prior seasons: it is a statement about a factor's absolute reliability, and
    starving it of the current season's games only makes it noisier. The
    historical-vs-current split belongs to the divergence term below.

``divergence``
    ``clip(|r_blend| / |r_hist|, lo, hi) ** gamma`` -- how far the current
    season has moved this factor away from its historical value.

    multiplier = signal_prior * divergence

For ridge the multiplier scales the standardised column, so it lands directly
on the fitted coefficient. Tree models are invariant to monotone rescaling, so
there the ledger acts by pruning factors whose multiplier collapses (see
:data:`streamer.models.positions.LEDGER_DROP_THRESHOLD`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, get_config
from ..features.build import FACTOR_LABELS

LEDGER_COLUMNS = (
    "asof_season", "asof_week", "position", "factor", "label",
    "r_hist", "n_hist", "r_current", "n_current", "r_trailing", "n_trailing",
    "r_blend", "r_full", "n_full", "signal_prior", "divergence_multiplier",
    "multiplier", "model_weight", "prev_model_weight", "weight_delta", "divergence",
)

#: Bounds on the absolute-signal component of the multiplier.
SIGNAL_PRIOR_CLIP = (0.2, 2.0)


def _corr(frame: pd.DataFrame, factor: str, target: str) -> tuple[float, int]:
    if factor not in frame.columns or target not in frame.columns or frame.empty:
        return float("nan"), 0
    x = pd.to_numeric(frame[factor], errors="coerce")
    y = pd.to_numeric(frame[target], errors="coerce")
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 10 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan"), n
    return float(np.corrcoef(x[mask], y[mask])[0, 1]), n


@dataclass
class FactorLedger:
    """One refit's worth of factor diagnostics."""

    frame: pd.DataFrame

    @classmethod
    def compute(
        cls,
        history: pd.DataFrame,
        factors: list[str],
        target: str,
        position: str,
        asof_season: int,
        asof_week: int,
        cfg: Config | None = None,
        previous: pd.DataFrame | None = None,
    ) -> "FactorLedger":
        """Build the ledger from every completed game available to this refit."""
        cfg = cfg or get_config()
        conf = cfg.ledger
        k = float(conf["prior_strength"])
        lo, hi = (float(v) for v in conf["weight_clip"])
        gamma = float(conf["weight_gamma"])
        trailing_weeks = int(conf["trailing_weeks"])

        played = history[history[target].notna()].copy()
        hist_window = played[played["season"] < asof_season]
        if hist_window.empty:
            # Season 1 of a fresh install: use everything as the "historical"
            # prior rather than leaving every factor undefined.
            hist_window = played
        current = played[(played["season"] == asof_season) & (played["week"] < asof_week)]
        trailing = current[current["week"] >= max(1, asof_week - trailing_weeks)]

        prev_lookup: dict[str, float] = {}
        if previous is not None and not previous.empty:
            prev = previous[previous["position"] == position]
            if not prev.empty:
                latest = prev.sort_values(["asof_season", "asof_week"]).groupby("factor").tail(1)
                prev_lookup = dict(zip(latest["factor"], latest["model_weight"]))

        # Everything the refit is allowed to see, for the absolute-signal term.
        full_window = played[
            (played["season"] < asof_season)
            | ((played["season"] == asof_season) & (played["week"] < asof_week))
        ]
        if full_window.empty:
            full_window = played

        rows = []
        for factor in factors:
            r_hist, n_hist = _corr(hist_window, factor, target)
            r_full, n_full = _corr(full_window, factor, target)
            r_cur, n_cur = _corr(current, factor, target)
            r_trail, n_trail = _corr(trailing, factor, target)

            if np.isnan(r_hist):
                r_blend, divergence_mult = r_cur, 1.0
            elif np.isnan(r_cur) or n_cur == 0:
                r_blend, divergence_mult = r_hist, 1.0
            else:
                r_blend = (n_cur * r_cur + k * r_hist) / (n_cur + k)
                denom = max(abs(r_hist), 0.02)  # guard: a ~0 prior is not a ratio
                divergence_mult = float(np.clip(abs(r_blend) / denom, lo, hi) ** gamma)

            divergence = (
                float("nan") if (np.isnan(r_cur) or np.isnan(r_hist)) else float(r_cur - r_hist)
            )
            rows.append(
                {
                    "_signal_abs": abs(r_full) if not np.isnan(r_full) else (
                        abs(r_hist) if not np.isnan(r_hist) else np.nan
                    ),
                    "_divergence_mult": divergence_mult,
                    "asof_season": asof_season,
                    "asof_week": asof_week,
                    "position": position,
                    "factor": factor,
                    "label": FACTOR_LABELS.get(factor, factor),
                    "r_hist": r_hist,
                    "n_hist": n_hist,
                    "r_current": r_cur,
                    "n_current": n_cur,
                    "r_trailing": r_trail,
                    "n_trailing": n_trail,
                    "r_blend": r_blend,
                    "r_full": r_full,
                    "n_full": n_full,
                    "signal_prior": np.nan,
                    "divergence_multiplier": divergence_mult,
                    "multiplier": np.nan,
                    "model_weight": np.nan,
                    "prev_model_weight": prev_lookup.get(factor, np.nan),
                    "weight_delta": np.nan,
                    "divergence": divergence,
                }
            )
        frame = pd.DataFrame(rows)
        # The absolute-signal component is relative to the other factors in this
        # position's set, so it can only be computed once all of them are known.
        strengths = frame["_signal_abs"]
        median = float(strengths.median()) if strengths.notna().any() else 0.0
        if median > 0:
            frame["signal_prior"] = np.clip(
                (strengths / median).fillna(1.0), *SIGNAL_PRIOR_CLIP
            )
        else:
            frame["signal_prior"] = 1.0
        frame["multiplier"] = frame["signal_prior"] * frame["_divergence_mult"]
        frame = frame.drop(columns=["_signal_abs", "_divergence_mult"])
        return cls(frame=frame[list(LEDGER_COLUMNS)])

    # -- accessors ---------------------------------------------------------
    @property
    def multipliers(self) -> dict[str, float]:
        return dict(zip(self.frame["factor"], self.frame["multiplier"]))

    def attach_model_weights(self, weights: dict[str, float]) -> None:
        """Record the weight each factor actually ended up with, and its move."""
        self.frame["model_weight"] = self.frame["factor"].map(weights)
        self.frame["weight_delta"] = (
            self.frame["model_weight"] - self.frame["prev_model_weight"]
        )

    def top_movers(self, n: int = 6) -> pd.DataFrame:
        moved = self.frame.dropna(subset=["weight_delta"])
        return moved.reindex(moved["weight_delta"].abs().sort_values(ascending=False).index).head(n)

    def most_divergent(self, n: int = 6) -> pd.DataFrame:
        div = self.frame.dropna(subset=["divergence"])
        return div.reindex(div["divergence"].abs().sort_values(ascending=False).index).head(n)


def ledger_path(cfg: Config | None = None) -> Path:
    cfg = cfg or get_config()
    return cfg.results_dir / "factor_ledger.parquet"


def load_ledger(cfg: Config | None = None) -> pd.DataFrame:
    path = ledger_path(cfg)
    if not path.exists():
        return pd.DataFrame(columns=list(LEDGER_COLUMNS))
    return pd.read_parquet(path)


def append_ledger(frames: list[pd.DataFrame], cfg: Config | None = None) -> pd.DataFrame:
    """Append this refit's ledger rows, replacing any prior run for the same week."""
    cfg = cfg or get_config()
    path = ledger_path(cfg)
    new = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True)
    existing = load_ledger(cfg)
    if not existing.empty:
        keys = set(zip(new["asof_season"], new["asof_week"], new["position"]))
        mask = [
            (s, w, p) not in keys
            for s, w, p in zip(existing["asof_season"], existing["asof_week"], existing["position"])
        ]
        existing = existing[mask]
    out = pd.concat([existing, new], ignore_index=True)
    out.to_parquet(path, index=False)
    return out


def multipliers_for(
    history: pd.DataFrame,
    factors: list[str],
    target: str,
    position: str,
    asof_season: int,
    asof_week: int,
    cfg: Config | None = None,
) -> dict[str, float]:
    """Ledger multipliers only, for callers that do not need the full frame.

    The walk-forward backtest uses this so it measures exactly the model that
    production fits -- ledger weighting included -- rather than a plain ridge.
    """
    try:
        ledger = FactorLedger.compute(
            history, factors, target, position, asof_season, asof_week, cfg
        )
    except Exception:  # noqa: BLE001 - weighting is an improvement, not a hard requirement
        return {f: 1.0 for f in factors}
    return {
        str(f): (1.0 if not np.isfinite(m) else float(m))
        for f, m in zip(ledger.frame["factor"], ledger.frame["multiplier"])
    }
