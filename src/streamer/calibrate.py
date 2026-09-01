"""The weekly self-calibration loop -- ``streamer update --week N``.

This is what separates the tool from a static projection sheet. Every week it:

1. pulls the completed week's actual results and scores the predictions it made;
2. appends the scored rows to ``results/history.parquet``;
3. refits both models on everything now known, with current-season games
   weighted 2x recent historical ones;
4. updates per-team in-season adjustment factors by Bayesian updating against
   each team's own multi-season prior, so a hot streak is credited gradually;
5. rewrites the factor-correlation ledger and derives the next refit's feature
   weights from it;
6. writes a plain-English ``reports/week_N_review.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, get_config
from .features.build import DST_FEATURES, KICKER_FEATURES
from .features.rolling import bayesian_update
from .models.base import spearman
from .models.ledger import FactorLedger, append_ledger, load_ledger
from .models.positions import BASELINE_COLUMN, DstModel, KickerModel, VegasBaseline
from .pipeline import build_slate, estimator_for

log = logging.getLogger(__name__)

HISTORY_COLUMNS = (
    "season", "week", "position", "team", "opponent", "player_id", "player_name",
    "expected_points", "floor", "ceiling", "p_top12", "model_rank",
    "actual_points", "actual_rank", "baseline_points", "error", "abs_error",
    "scored_at",
)


def history_path(cfg: Config | None = None) -> Path:
    cfg = cfg or get_config()
    return cfg.results_dir / "history.parquet"


def predictions_path(cfg: Config | None = None) -> Path:
    cfg = cfg or get_config()
    return cfg.results_dir / "predictions.parquet"


def load_history(cfg: Config | None = None) -> pd.DataFrame:
    path = history_path(cfg)
    if not path.exists():
        return pd.DataFrame(columns=list(HISTORY_COLUMNS))
    return pd.read_parquet(path)


def load_stored_predictions(cfg: Config | None = None) -> pd.DataFrame:
    path = predictions_path(cfg)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def store_predictions(frame: pd.DataFrame, cfg: Config | None = None) -> None:
    """Persist a week's rankings so ``update`` can score exactly what was published."""
    cfg = cfg or get_config()
    path = predictions_path(cfg)
    existing = load_stored_predictions(cfg)
    if not existing.empty:
        keys = set(zip(frame["season"], frame["week"], frame["position"]))
        mask = [
            (s, w, p) not in keys
            for s, w, p in zip(existing["season"], existing["week"], existing["position"])
        ]
        existing = existing[mask]
    pd.concat([existing, frame], ignore_index=True).to_parquet(path, index=False)


@dataclass
class WeekScore:
    """How one position's predictions actually did in one week."""

    position: str
    n: int
    mae: float
    rank_corr: float
    top5_hit_rate: float
    baseline_mae: float
    baseline_rank_corr: float
    baseline_top5_hit_rate: float
    top5_mean_points: float
    slate_mean_points: float
    misses: pd.DataFrame = field(default_factory=pd.DataFrame)
    hits: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class UpdateResult:
    season: int
    week: int
    scores: dict[str, WeekScore]
    ledgers: dict[str, FactorLedger]
    history: pd.DataFrame
    team_adjustments: pd.DataFrame
    report_path: Path | None = None
    scored_from_stored: bool = True


# ---------------------------------------------------------------------------
# Scoring last week
# ---------------------------------------------------------------------------
def score_week(
    predictions: pd.DataFrame, actuals: pd.DataFrame, position: str, cfg: Config | None = None
) -> tuple[WeekScore, pd.DataFrame]:
    """Join predictions to results and compute the week's calibration metrics."""
    cfg = cfg or get_config()
    join_cols = ["team"] if position == "DST" else ["team", "player_id"]
    actual_cols = join_cols + ["fantasy_points"]
    merged = predictions.merge(
        actuals[actual_cols].rename(columns={"fantasy_points": "actual_points"}),
        on=join_cols,
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"no {position} predictions could be matched to results")

    merged["actual_rank"] = merged["actual_points"].rank(ascending=False, method="min")
    merged["model_rank"] = merged["expected_points"].rank(ascending=False, method="first")
    merged["error"] = merged["expected_points"] - merged["actual_points"]
    merged["abs_error"] = merged["error"].abs()

    cutoff = cfg.startable_rank
    top5 = merged["model_rank"] <= 5
    if "baseline_points" in merged.columns and merged["baseline_points"].notna().any():
        base_rank = merged["baseline_points"].rank(ascending=False, method="first")
        baseline_mae = float((merged["baseline_points"] - merged["actual_points"]).abs().mean())
        baseline_rc = spearman(merged["baseline_points"], merged["actual_points"])
        baseline_top5 = float((merged.loc[base_rank <= 5, "actual_rank"] <= cutoff).mean())
    else:
        baseline_mae = baseline_rc = baseline_top5 = float("nan")

    score = WeekScore(
        position=position,
        n=len(merged),
        mae=float(merged["abs_error"].mean()),
        rank_corr=spearman(merged["expected_points"], merged["actual_points"]),
        top5_hit_rate=float((merged.loc[top5, "actual_rank"] <= cutoff).mean()),
        baseline_mae=baseline_mae,
        baseline_rank_corr=baseline_rc,
        baseline_top5_hit_rate=baseline_top5,
        top5_mean_points=float(merged.loc[top5, "actual_points"].mean()),
        slate_mean_points=float(merged["actual_points"].mean()),
        misses=merged.nlargest(5, "error"),
        hits=merged.nsmallest(5, "error"),
    )
    return score, merged


def append_history(rows: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Append scored rows to ``results/history.parquet``, replacing re-runs."""
    cfg = cfg or get_config()
    rows = rows.copy()
    rows["scored_at"] = pd.Timestamp.utcnow().isoformat()
    for col in HISTORY_COLUMNS:
        if col not in rows.columns:
            rows[col] = np.nan
    rows = rows[list(HISTORY_COLUMNS)]

    existing = load_history(cfg)
    if not existing.empty:
        keys = set(zip(rows["season"], rows["week"], rows["position"]))
        mask = [
            (s, w, p) not in keys
            for s, w, p in zip(existing["season"], existing["week"], existing["position"])
        ]
        existing = existing[mask]
    out = pd.concat([existing, rows], ignore_index=True)
    out.to_parquet(history_path(cfg), index=False)
    return out


# ---------------------------------------------------------------------------
# Per-team in-season adjustments
# ---------------------------------------------------------------------------
#: Rates tracked per team, as ``(name, numerator, denominator)`` on the DST
#: feature frame. These are the "is this team's identity actually different
#: this year?" signals.
TEAM_ADJUSTMENT_RATES = (
    ("sack_rate", "sacks", "opp_dropbacks_per_game"),
    ("big_play_points", "big_play_points", None),
    ("points_allowed", "points_allowed", None),
)


def update_team_adjustments(
    dst_features: pd.DataFrame, season: int, week: int, cfg: Config | None = None
) -> pd.DataFrame:
    """Bayesian per-team adjustment factors for the current season.

    Each team's season-to-date rate is blended with its own multi-season prior
    using :func:`~streamer.features.rolling.bayesian_update`, so a defense
    running hot gets credited in proportion to how many games back it up --
    never instantly.
    """
    cfg = cfg or get_config()
    prior_strength = float(cfg.shrinkage["team_adjustment_prior_games"])
    played = dst_features[dst_features["fantasy_points"].notna()]
    current = played[(played["season"] == season) & (played["week"] <= week)]
    prior = played[played["season"] < season]
    if current.empty:
        return pd.DataFrame(columns=["season", "week", "team", "metric", "prior",
                                     "observed", "posterior", "adjustment", "n"])

    rows = []
    for metric, num_col, _den in TEAM_ADJUSTMENT_RATES:
        if num_col not in played.columns:
            continue
        league_mean = float(pd.to_numeric(prior[num_col], errors="coerce").mean()) if not prior.empty \
            else float(pd.to_numeric(current[num_col], errors="coerce").mean())
        prior_by_team = (
            pd.to_numeric(prior[num_col], errors="coerce").groupby(prior["team"]).mean()
            if not prior.empty else pd.Series(dtype=float)
        )
        for team, grp in current.groupby("team"):
            values = pd.to_numeric(grp[num_col], errors="coerce").dropna()
            if values.empty:
                continue
            team_prior = float(prior_by_team.get(team, league_mean))
            posterior = bayesian_update(
                prior_mean=team_prior,
                prior_strength=prior_strength,
                observed_sum=float(values.sum()),
                observed_n=float(len(values)),
            )
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "team": team,
                    "metric": metric,
                    "prior": team_prior,
                    "observed": float(values.mean()),
                    "posterior": posterior,
                    "adjustment": posterior - team_prior,
                    "n": int(len(values)),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_parquet(cfg.results_dir / "team_adjustments.parquet", index=False)
    return out


# ---------------------------------------------------------------------------
# The update command
# ---------------------------------------------------------------------------
def run_update(
    week: int,
    season: int | None = None,
    cfg: Config | None = None,
    allow_network: bool = True,
) -> UpdateResult:
    """Score week ``week``, refit, update the ledger, and write the review."""
    cfg = cfg or get_config()
    season = season or cfg.current_season

    context = build_slate(season, week, cfg, weeks_ahead=1, allow_network=allow_network)
    stored = load_stored_predictions(cfg)

    scores: dict[str, WeekScore] = {}
    ledgers: dict[str, FactorLedger] = {}
    history_rows: list[pd.DataFrame] = []
    scored_from_stored = True

    previous_ledger = load_ledger(cfg)

    for position, slate, train, features in (
        ("DST", context.dst_slate, context.dst_train, list(DST_FEATURES)),
        ("K", context.kicker_slate, context.kicker_train, list(KICKER_FEATURES)),
    ):
        actuals = train[(train["season"] == season) & (train["week"] == week)]
        if actuals.empty:
            log.warning("week %s %s results are not published yet; nothing to score", week, position)
            continue

        predictions = _predictions_for(stored, season, week, position)
        if predictions is None:
            scored_from_stored = False
            predictions = _reproject(position, slate, train, season, week, cfg)
        if predictions is None or predictions.empty:
            continue

        try:
            score, merged = score_week(predictions, actuals, position, cfg)
        except ValueError as exc:
            log.warning("%s", exc)
            continue
        scores[position] = score
        merged["position"] = position
        merged["season"] = season
        merged["week"] = week
        history_rows.append(merged)

        # Refit inputs now include this week, so the ledger is computed as of
        # week+1 -- the state the next projection will use.
        ledger = FactorLedger.compute(
            train, features, "fantasy_points", position, season, week + 1, cfg,
            previous=previous_ledger,
        )
        model_cls = KickerModel if position == "K" else DstModel
        model = model_cls.fit(
            train, cfg, kind=estimator_for(position, cfg), current_season=season,
            multipliers=ledger.multipliers,
        )
        ledger.attach_model_weights(model.core.weights)
        ledgers[position] = ledger

    history = append_history(pd.concat(history_rows, ignore_index=True), cfg) if history_rows \
        else load_history(cfg)
    if ledgers:
        append_ledger([entry.frame for entry in ledgers.values()], cfg)
    adjustments = update_team_adjustments(context.dst_train, season, week, cfg)

    result = UpdateResult(
        season=season,
        week=week,
        scores=scores,
        ledgers=ledgers,
        history=history,
        team_adjustments=adjustments,
        scored_from_stored=scored_from_stored,
    )
    result.report_path = write_week_review(result, cfg)
    return result


def _predictions_for(
    stored: pd.DataFrame, season: int, week: int, position: str
) -> pd.DataFrame | None:
    if stored.empty:
        return None
    sub = stored[
        (stored["season"] == season)
        & (stored["week"] == week)
        & (stored["position"] == position)
    ]
    return sub.copy() if not sub.empty else None


def _reproject(
    position: str, slate: pd.DataFrame, train: pd.DataFrame,
    season: int, week: int, cfg: Config
) -> pd.DataFrame | None:
    """Rebuild what the model would have said, for weeks never published.

    Training is restricted to games before the week in question, so a
    retro-scored week is still an honest out-of-sample test.
    """
    target = slate[(slate["season"] == season) & (slate["week"] == week)]
    if target.empty:
        return None
    history = train[
        (train["season"] < season) | ((train["season"] == season) & (train["week"] < week))
    ]
    if len(history) < int(cfg.model["min_train_rows"]):
        return None
    model_cls = KickerModel if position == "K" else DstModel
    try:
        model = model_cls.fit(history, cfg, kind=estimator_for(position, cfg), current_season=season)
        projection = model.predict(target)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not reproject %s week %s: %s", position, week, exc)
        return None
    frame = projection.frame
    baseline = VegasBaseline.fit(history, BASELINE_COLUMN[position])
    frame["baseline_points"] = baseline.predict(target)
    frame["position"] = position
    return frame


# ---------------------------------------------------------------------------
# The weekly review
# ---------------------------------------------------------------------------
def write_week_review(result: UpdateResult, cfg: Config | None = None) -> Path:
    """Write ``reports/week_N_review.md`` in plain English."""
    cfg = cfg or get_config()
    path = cfg.reports_dir / f"week_{result.week}_review.md"
    lines: list[str] = [
        f"# Week {result.week} review ({result.season})",
        "",
        f"_Generated {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}._",
        "",
    ]
    if not result.scores:
        lines += ["No results were available to score for this week yet.", ""]
        path.write_text("\n".join(lines))
        return path

    if not result.scored_from_stored:
        lines += [
            "> These rankings were not published before kickoff, so the model was "
            "re-fit on games prior to this week and re-projected. It is still an "
            "out-of-sample score, but it is a reconstruction, not a live record.",
            "",
        ]

    lines += ["## How the model did", "",
              "| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | "
              "Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for position in ("DST", "K"):
        s = result.scores.get(position)
        if s is None:
            continue
        lines.append(
            f"| {position} | {s.n} | {s.mae:.2f} | {_fmt(s.baseline_mae)} | "
            f"{_fmt(s.rank_corr, 3)} | {_fmt(s.baseline_rank_corr, 3)} | "
            f"{s.top5_hit_rate:.0%} | {_pct(s.baseline_top5_hit_rate)} | "
            f"{s.top5_mean_points:.1f} | {s.slate_mean_points:.1f} |"
        )
    lines.append("")

    for position in ("DST", "K"):
        score = result.scores.get(position)
        if score is None:
            continue
        label = "D/ST" if position == "DST" else "Kicker"
        lines += [f"## {label}: what it got wrong", ""]
        lines.append(_verdict_line(score))
        lines.append("")
        lines.append("**Biggest overshoots** (projected high, scored low):")
        lines.append("")
        lines += _miss_table(score.misses, position)
        lines.append("")
        lines.append("**Biggest undershoots** (projected low, scored high):")
        lines.append("")
        lines += _miss_table(score.hits, position)
        lines.append("")

    lines += _ledger_section(result)
    lines += _adjustment_section(result)
    lines += _trend_section(result, cfg)

    path.write_text("\n".join(lines) + "\n")
    return path


def _pct(value: float) -> str:
    return "n/a" if value is None or np.isnan(value) else f"{value:.0%}"


def _fmt(value: float, places: int = 2) -> str:
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) \
        else f"{value:.{places}f}"


def _verdict_line(score: WeekScore) -> str:
    if np.isnan(score.baseline_rank_corr):
        return (
            f"Rank correlation {_fmt(score.rank_corr, 3)}, MAE {score.mae:.2f}. "
            "No Vegas-only comparison was stored for this week."
        )
    edge = score.rank_corr - score.baseline_rank_corr
    if edge > 0.02:
        verdict = "clearly better than ranking on the Vegas number alone"
    elif edge > 0:
        verdict = "slightly ahead of ranking on the Vegas number alone"
    elif edge > -0.02:
        verdict = "essentially level with the Vegas-only ranking"
    else:
        verdict = "behind the Vegas-only ranking, which is a bad week"
    return (
        f"Rank correlation {_fmt(score.rank_corr, 3)} against a Vegas-only "
        f"{_fmt(score.baseline_rank_corr, 3)} -- {verdict}. "
        f"The five recommended units averaged {score.top5_mean_points:.1f} points "
        f"against a slate average of {score.slate_mean_points:.1f}, and "
        f"{score.top5_hit_rate:.0%} of them finished startable "
        f"(Vegas-only: {_pct(score.baseline_top5_hit_rate)})."
    )


def _miss_table(frame: pd.DataFrame, position: str) -> list[str]:
    if frame.empty:
        return ["_none_"]
    name_col = "team" if position == "DST" else "player_name"
    header = ["| Unit | Opp | Projected | Actual | Miss |", "|---|---|---|---|---|"]
    rows = []
    for row in frame.itertuples():
        name = getattr(row, name_col, getattr(row, "team", "?"))
        opp = getattr(row, "opponent", "")
        rows.append(
            f"| {name} | {opp} | {row.expected_points:.1f} | "
            f"{row.actual_points:.1f} | {-row.error:+.1f} |"
        )
    return header + rows


def _ledger_section(result: UpdateResult) -> list[str]:
    if not result.ledgers:
        return []
    lines = [
        "## Factor ledger: which weights moved and why",
        "",
        "Each factor's correlation with actual fantasy points, over the full "
        "historical window, the current season to date, and the trailing four "
        "weeks. The model weight is derived from a shrinkage blend of the "
        "historical and current-season numbers -- the historical prior's pull "
        "decays automatically as the season's sample grows.",
        "",
    ]
    for position, ledger in result.ledgers.items():
        label = "D/ST" if position == "DST" else "Kicker"
        lines += [f"### {label}", "",
                  "| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |",
                  "|---|---|---|---|---|---|---|"]
        frame = ledger.frame.copy()
        frame["_abs"] = frame["r_hist"].abs()
        for row in frame.sort_values("_abs", ascending=False).head(12).itertuples():
            delta = "-" if (row.weight_delta is None or np.isnan(row.weight_delta)) \
                else f"{row.weight_delta:+.3f}"
            lines.append(
                f"| {row.label} | {_fmt(row.r_hist, 3)} | {_fmt(row.r_current, 3)} | "
                f"{_fmt(row.r_trailing, 3)} | {_fmt(row.r_blend, 3)} | "
                f"{_fmt(row.model_weight, 3)} | {delta} |"
            )
        lines.append("")

        divergent = ledger.most_divergent(3)
        if not divergent.empty:
            lines.append("**Biggest divergences from the historical prior:**")
            lines.append("")
            for row in divergent.itertuples():
                if np.isnan(row.divergence):
                    continue
                direction = "stronger" if row.divergence > 0 else "weaker"
                lines.append(
                    f"- **{row.label}** is running {direction} this season "
                    f"({_fmt(row.r_current, 3)} vs {_fmt(row.r_hist, 3)} historically). "
                    f"Blended to {_fmt(row.r_blend, 3)}, so its weight multiplier is "
                    f"{_fmt(row.multiplier, 2)}x."
                )
            lines.append("")
    return lines


def _adjustment_section(result: UpdateResult) -> list[str]:
    adj = result.team_adjustments
    if adj is None or adj.empty:
        return []
    lines = ["## Per-team in-season adjustments", "",
             "Season-to-date form blended against each team's own multi-season "
             "prior. A team only earns a large adjustment once it has the games "
             "to back it up.", ""]
    for metric, grp in adj.groupby("metric"):
        movers = grp.reindex(grp["adjustment"].abs().sort_values(ascending=False).index).head(5)
        lines.append(f"**{metric.replace('_', ' ').title()}**")
        lines.append("")
        lines.append("| Team | Prior | Season | Posterior | Move | Games |")
        lines.append("|---|---|---|---|---|---|")
        for row in movers.itertuples():
            lines.append(
                f"| {row.team} | {row.prior:.2f} | {row.observed:.2f} | "
                f"{row.posterior:.2f} | {row.adjustment:+.2f} | {row.n} |"
            )
        lines.append("")
    return lines


def _trend_section(result: UpdateResult, cfg: Config) -> list[str]:
    history = result.history
    if history is None or history.empty:
        return []
    season = history[history["season"] == result.season]
    if season.empty:
        return []
    lines = ["## Season-to-date calibration", "",
             "| Position | Weeks | MAE | Rank corr | Top-5 hit rate |", "|---|---|---|---|---|"]
    for position, grp in season.groupby("position"):
        weeks = grp["week"].nunique()
        mae = float(grp["abs_error"].mean())
        per_week = [
            spearman(g["expected_points"], g["actual_points"]) for _w, g in grp.groupby("week")
        ]
        top5 = [
            float((g.loc[g["model_rank"] <= 5, "actual_rank"] <= cfg.startable_rank).mean())
            for _w, g in grp.groupby("week")
        ]
        lines.append(
            f"| {position} | {weeks} | {mae:.2f} | {_fmt(float(np.nanmean(per_week)), 3)} | "
            f"{float(np.nanmean(top5)):.0%} |"
        )
    lines.append("")
    return lines
