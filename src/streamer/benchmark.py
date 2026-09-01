"""Head-to-head benchmarking against Subvertadown.

Subvertadown publishes ranked DST and K lists each week but no projections, so
the only fair comparison is on *ordering*: rank correlation against actual
finishes, and how often each system's top five finished startable.

Input is a hand-entered CSV at ``data/subvertadown_week_N.csv`` with columns
``rank`` and ``team`` (D/ST) or ``player`` (K); a ``position`` column lets one
file carry both lists. The running record lands in ``reports/benchmark.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, get_config
from .models.base import spearman
from .teams import normalize_team

log = logging.getLogger(__name__)

BENCHMARK_COLUMNS = (
    "season", "week", "position", "n_matched",
    "streamer_rank_corr", "subvertadown_rank_corr", "rank_corr_edge",
    "streamer_top5_hit_rate", "subvertadown_top5_hit_rate",
    "streamer_top5_points", "subvertadown_top5_points", "slate_mean_points",
)


def subvertadown_path(week: int, cfg: Config | None = None) -> Path:
    cfg = cfg or get_config()
    return cfg.data_dir / f"subvertadown_week_{week}.csv"


def read_subvertadown(week: int, cfg: Config | None = None) -> pd.DataFrame:
    """Read a hand-entered Subvertadown ranking file.

    Accepted columns: ``rank`` plus one of ``team`` / ``player`` / ``name``, and
    an optional ``position`` (``DST`` / ``K``) when both lists share a file.
    """
    cfg = cfg or get_config()
    path = subvertadown_path(week, cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- paste Subvertadown's week {week} rankings there "
            "(columns: rank,team for D/ST; rank,player for K)"
        )
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "rank" not in df.columns:
        raise ValueError(f"{path} must have a 'rank' column")

    name_col = next((c for c in ("team", "player", "name", "unit") if c in df.columns), None)
    if name_col is None:
        raise ValueError(f"{path} must have a 'team' or 'player' column")

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df[df["rank"].notna()].copy()
    df["entry"] = df[name_col].astype(str).str.strip()

    if "position" in df.columns:
        df["position"] = (
            df["position"].astype(str).str.upper().str.strip()
            .replace({"D/ST": "DST", "DEF": "DST", "D": "DST", "PK": "K"})
        )
    else:
        # No position column: infer from which name column was supplied.
        df["position"] = "DST" if name_col == "team" else "K"
    return df[["position", "rank", "entry"]]


@dataclass
class BenchmarkResult:
    season: int
    week: int
    rows: pd.DataFrame
    unmatched: dict[str, list[str]]


def _match_dst(entries: pd.DataFrame, actuals: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    entries = entries.copy()
    entries["team"] = entries["entry"].map(normalize_team)
    unmatched = entries.loc[entries["team"].isna(), "entry"].tolist()
    merged = entries.dropna(subset=["team"]).merge(
        actuals[["team", "fantasy_points"]], on="team", how="inner"
    )
    return merged, unmatched


def _match_kicker(entries: pd.DataFrame, actuals: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Match kicker names, tolerating ``C.Boswell`` vs ``Chris Boswell``."""
    actuals = actuals.copy()
    actuals["_last"] = (
        actuals["player_name"].astype(str).str.split(".").str[-1].str.strip().str.lower()
    )
    actuals["_key"] = actuals["player_name"].astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)

    rows, unmatched = [], []
    for entry in entries.itertuples():
        text = str(entry.entry).strip()
        key = text.lower().replace(".", "").replace(" ", "")
        last = text.split()[-1].lower() if text.split() else ""
        hit = actuals[actuals["_key"] == key]
        if hit.empty and last:
            hit = actuals[actuals["_last"] == last]
        if hit.empty:
            unmatched.append(text)
            continue
        if len(hit) > 1:
            hit = hit.nlargest(1, "fantasy_points")
        rows.append(
            {"rank": entry.rank, "entry": text, "team": hit.iloc[0]["team"],
             "fantasy_points": float(hit.iloc[0]["fantasy_points"])}
        )
    return pd.DataFrame(rows), unmatched


def compare_week(
    week: int,
    streamer_predictions: dict[str, pd.DataFrame],
    actuals: dict[str, pd.DataFrame],
    season: int | None = None,
    cfg: Config | None = None,
    top_k: int = 5,
) -> BenchmarkResult:
    """Compare both systems' orderings against the week's actual finishes."""
    cfg = cfg or get_config()
    season = season or cfg.current_season
    cutoff = cfg.startable_rank
    sub = read_subvertadown(week, cfg)

    rows, unmatched = [], {}
    for position in ("DST", "K"):
        entries = sub[sub["position"] == position]
        actual = actuals.get(position)
        mine = streamer_predictions.get(position)
        if entries.empty or actual is None or actual.empty or mine is None or mine.empty:
            continue

        matched, missing = (
            _match_dst(entries, actual) if position == "DST" else _match_kicker(entries, actual)
        )
        if missing:
            unmatched[position] = missing
        if len(matched) < 5:
            log.warning("only %s %s entries matched; skipping", len(matched), position)
            continue

        # Compare on the same set of units both systems ranked.
        key = "team" if position == "DST" else "team"
        mine_ranked = mine.copy()
        mine_ranked["model_rank"] = mine_ranked["expected_points"].rank(
            ascending=False, method="first"
        )
        pair = matched.merge(
            mine_ranked[[key, "model_rank", "expected_points"]], on=key, how="inner"
        )
        if len(pair) < 5:
            continue

        actual_rank = pair["fantasy_points"].rank(ascending=False, method="min")
        # Both systems' ranks are re-densified over the shared set so neither is
        # penalised for units the other did not list.
        mine_order = pair["model_rank"].rank(method="first")
        sub_order = pair["rank"].rank(method="first")
        slate_actual_rank = actual["fantasy_points"].rank(ascending=False, method="min")
        startable = set(actual.loc[slate_actual_rank <= cutoff, key])

        rows.append(
            {
                "season": season,
                "week": week,
                "position": position,
                "n_matched": len(pair),
                "streamer_rank_corr": spearman(-mine_order, pair["fantasy_points"]),
                "subvertadown_rank_corr": spearman(-sub_order, pair["fantasy_points"]),
                "streamer_top5_hit_rate": float(
                    pair.loc[mine_order <= top_k, key].isin(startable).mean()
                ),
                "subvertadown_top5_hit_rate": float(
                    pair.loc[sub_order <= top_k, key].isin(startable).mean()
                ),
                "streamer_top5_points": float(pair.loc[mine_order <= top_k, "fantasy_points"].mean()),
                "subvertadown_top5_points": float(pair.loc[sub_order <= top_k, "fantasy_points"].mean()),
                "slate_mean_points": float(actual["fantasy_points"].mean()),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["rank_corr_edge"] = (
            frame["streamer_rank_corr"] - frame["subvertadown_rank_corr"]
        )
        frame = frame[list(BENCHMARK_COLUMNS)]
    return BenchmarkResult(season=season, week=week, rows=frame, unmatched=unmatched)


def benchmark_path(cfg: Config | None = None) -> Path:
    cfg = cfg or get_config()
    return cfg.results_dir / "benchmark.parquet"


def load_benchmark(cfg: Config | None = None) -> pd.DataFrame:
    path = benchmark_path(cfg)
    if not path.exists():
        return pd.DataFrame(columns=list(BENCHMARK_COLUMNS))
    return pd.read_parquet(path)


def append_benchmark(rows: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or get_config()
    if rows.empty:
        return load_benchmark(cfg)
    existing = load_benchmark(cfg)
    if not existing.empty:
        keys = set(zip(rows["season"], rows["week"], rows["position"]))
        mask = [
            (s, w, p) not in keys
            for s, w, p in zip(existing["season"], existing["week"], existing["position"])
        ]
        existing = existing[mask]
    out = pd.concat([existing, rows], ignore_index=True)
    out = out.sort_values(["season", "week", "position"]).reset_index(drop=True)
    out.to_parquet(benchmark_path(cfg), index=False)
    return out


def write_benchmark_report(cfg: Config | None = None) -> Path:
    """Rewrite ``reports/benchmark.md`` from the accumulated record."""
    cfg = cfg or get_config()
    frame = load_benchmark(cfg)
    path = cfg.reports_dir / "benchmark.md"
    lines = [
        "# Head-to-head vs Subvertadown",
        "",
        "Rank correlation against actual weekly finishes, computed over the units "
        "both systems ranked. The season goal is to meet or beat his rank "
        "correlation by Week 8.",
        "",
    ]
    if frame.empty:
        lines += [
            "No weeks benchmarked yet. Paste his rankings into "
            "`data/subvertadown_week_N.csv` and run `streamer benchmark --week N`.",
            "",
        ]
        path.write_text("\n".join(lines) + "\n")
        return path

    lines += ["## Running totals", "",
              "| Position | Weeks | streamer rank corr | Subvertadown rank corr | Edge | "
              "streamer top-5 | Subvertadown top-5 | Weeks won |", "|---|---|---|---|---|---|---|---|"]
    for position, grp in frame.groupby("position"):
        won = int((grp["rank_corr_edge"] > 0).sum())
        lines.append(
            f"| {position} | {len(grp)} | {grp['streamer_rank_corr'].mean():+.3f} | "
            f"{grp['subvertadown_rank_corr'].mean():+.3f} | "
            f"{grp['rank_corr_edge'].mean():+.3f} | "
            f"{grp['streamer_top5_hit_rate'].mean():.0%} | "
            f"{grp['subvertadown_top5_hit_rate'].mean():.0%} | {won}/{len(grp)} |"
        )
    lines.append("")

    lines += ["## Week by week", "",
              "| Season | Week | Position | n | streamer | Subvertadown | Edge | "
              "streamer top-5 pts | Subvertadown top-5 pts |", "|---|---|---|---|---|---|---|---|---|"]
    for row in frame.itertuples():
        lines.append(
            f"| {row.season} | {row.week} | {row.position} | {row.n_matched} | "
            f"{row.streamer_rank_corr:+.3f} | {row.subvertadown_rank_corr:+.3f} | "
            f"{row.rank_corr_edge:+.3f} | {row.streamer_top5_points:.1f} | "
            f"{row.subvertadown_top5_points:.1f} |"
        )
    lines.append("")

    goal = frame[frame["week"] <= 8]
    if not goal.empty:
        lines += ["## Week-8 goal check", ""]
        for position, grp in goal.groupby("position"):
            edge = float(grp["rank_corr_edge"].mean())
            verdict = "met" if edge >= 0 else "not met"
            lines.append(
                f"- **{position}**: mean edge {edge:+.3f} over {len(grp)} weeks -- goal {verdict}."
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")
    return path
