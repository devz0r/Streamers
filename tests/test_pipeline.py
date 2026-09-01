"""End-to-end plumbing: benchmark matching, publishing, the CLI and calibration.

These run without network access by driving the modules with synthetic frames,
so the suite stays fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from streamer.benchmark import (
    append_benchmark,
    compare_week,
    read_subvertadown,
    write_benchmark_report,
)
from streamer.calibrate import append_history, score_week, store_predictions


@pytest.fixture
def slate_predictions():
    teams = ["KC", "BUF", "SF", "DAL", "NYJ", "PHI", "BAL", "DET",
             "GB", "MIN", "SEA", "LA", "MIA", "CIN", "PIT", "CLE"]
    rng = np.random.default_rng(4)
    return pd.DataFrame({
        "season": 2026, "week": 5, "position": "DST", "team": teams,
        "opponent": teams[::-1], "player_id": None, "player_name": None,
        "display_name": teams,
        "expected_points": np.linspace(11, 3, len(teams)),
        "floor": np.linspace(4, -1, len(teams)),
        "ceiling": np.linspace(19, 8, len(teams)),
        "p_top12": np.linspace(0.9, 0.2, len(teams)),
        "rank": range(1, len(teams) + 1),
        "baseline_points": np.linspace(10, 4, len(teams)),
        "rationale": ["because" for _ in teams],
        "two_week_hold": [i < 3 for i in range(len(teams))],
    })


@pytest.fixture
def slate_actuals(slate_predictions):
    rng = np.random.default_rng(9)
    frame = slate_predictions[["team"]].copy()
    frame["fantasy_points"] = rng.normal(7, 5, len(frame)).round(1)
    frame["season"], frame["week"] = 2026, 5
    return frame


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_score_week_produces_calibration_metrics(cfg, slate_predictions, slate_actuals):
    score, merged = score_week(slate_predictions, slate_actuals, "DST", cfg)
    assert score.n == len(slate_predictions)
    assert score.mae > 0
    assert -1 <= score.rank_corr <= 1
    assert 0 <= score.top5_hit_rate <= 1
    assert len(merged) == len(slate_predictions)
    assert "abs_error" in merged.columns
    assert not score.misses.empty and not score.hits.empty


def test_score_week_needs_a_match(cfg, slate_predictions, slate_actuals):
    orphaned = slate_actuals.copy()
    orphaned["team"] = "ZZZ"
    with pytest.raises(ValueError, match="no DST predictions"):
        score_week(slate_predictions, orphaned, "DST", cfg)


def test_history_round_trips_and_replaces_reruns(tmp_cfg, slate_predictions, slate_actuals):
    _score, merged = score_week(slate_predictions, slate_actuals, "DST", tmp_cfg)
    merged["position"], merged["season"], merged["week"] = "DST", 2026, 5
    first = append_history(merged, tmp_cfg)
    assert len(first) == len(merged)
    second = append_history(merged, tmp_cfg)
    # Re-running the same week must overwrite, not duplicate.
    assert len(second) == len(merged)


def test_predictions_store_replaces_the_same_week(tmp_cfg, slate_predictions):
    store_predictions(slate_predictions, tmp_cfg)
    store_predictions(slate_predictions, tmp_cfg)
    from streamer.calibrate import load_stored_predictions

    assert len(load_stored_predictions(tmp_cfg)) == len(slate_predictions)


def test_team_adjustments_move_gradually(tmp_cfg):
    rng = np.random.default_rng(2)
    rows = []
    for season in (2024, 2025, 2026):
        for week in range(1, 10):
            for team in ("KC", "BUF"):
                rows.append({
                    "season": season, "week": week, "team": team,
                    "fantasy_points": 7.0, "opp_dropbacks_per_game": 35.0,
                    "sacks": 6.0 if (team == "KC" and season == 2026) else 2.0,
                    "big_play_points": 6.0, "points_allowed": 20.0,
                })
    from streamer.calibrate import update_team_adjustments

    adj = update_team_adjustments(pd.DataFrame(rows), 2026, 9, tmp_cfg)
    kc = adj[(adj["team"] == "KC") & (adj["metric"] == "sack_rate")].iloc[0]
    # KC's observed rate is 6.0 against a prior of 2.0; the posterior must sit
    # between the two rather than jumping to the observation.
    assert kc["prior"] == pytest.approx(2.0)
    assert kc["observed"] == pytest.approx(6.0)
    assert 2.0 < kc["posterior"] < 6.0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def test_subvertadown_csv_is_parsed(tmp_cfg):
    (tmp_cfg.data_dir / "subvertadown_week_5.csv").write_text(
        "rank,team\n1,Chiefs\n2,BUF\n3,San Francisco 49ers\n"
    )
    frame = read_subvertadown(5, tmp_cfg)
    assert list(frame["position"]) == ["DST"] * 3
    assert list(frame["rank"]) == [1, 2, 3]


def test_subvertadown_csv_can_carry_both_positions(tmp_cfg):
    (tmp_cfg.data_dir / "subvertadown_week_6.csv").write_text(
        "position,rank,name\nDST,1,KC\nD/ST,2,BUF\nK,1,Harrison Butker\n"
    )
    frame = read_subvertadown(6, tmp_cfg)
    assert set(frame["position"]) == {"DST", "K"}


def test_missing_subvertadown_csv_gives_a_helpful_error(tmp_cfg):
    with pytest.raises(FileNotFoundError, match="paste Subvertadown"):
        read_subvertadown(11, tmp_cfg)


def test_benchmark_compares_both_systems(tmp_cfg, slate_predictions, slate_actuals):
    teams = list(slate_predictions["team"])
    lines = "rank,team\n" + "".join(f"{i + 1},{t}\n" for i, t in enumerate(teams))
    (tmp_cfg.data_dir / "subvertadown_week_5.csv").write_text(lines)

    result = compare_week(
        5, {"DST": slate_predictions}, {"DST": slate_actuals}, 2026, tmp_cfg
    )
    assert len(result.rows) == 1
    row = result.rows.iloc[0]
    assert row["n_matched"] == len(teams)
    assert np.isfinite(row["streamer_rank_corr"])
    assert np.isfinite(row["subvertadown_rank_corr"])
    assert row["rank_corr_edge"] == pytest.approx(
        row["streamer_rank_corr"] - row["subvertadown_rank_corr"]
    )


def test_benchmark_report_is_written(tmp_cfg, slate_predictions, slate_actuals):
    teams = list(slate_predictions["team"])
    (tmp_cfg.data_dir / "subvertadown_week_5.csv").write_text(
        "rank,team\n" + "".join(f"{i + 1},{t}\n" for i, t in enumerate(teams))
    )
    result = compare_week(5, {"DST": slate_predictions}, {"DST": slate_actuals}, 2026, tmp_cfg)
    append_benchmark(result.rows, tmp_cfg)
    path = write_benchmark_report(tmp_cfg)
    text = path.read_text()
    assert "Head-to-head vs Subvertadown" in text
    assert "Running totals" in text


def test_kicker_names_match_across_formats(tmp_cfg):
    actuals = pd.DataFrame({
        "team": ["PIT", "DAL"],
        "player_name": ["C.Boswell", "B.Aubrey"],
        "fantasy_points": [14.0, 9.0],
    })
    predictions = pd.DataFrame({
        "team": ["PIT", "DAL"], "expected_points": [11.0, 10.0],
    })
    (tmp_cfg.data_dir / "subvertadown_week_7.csv").write_text(
        "rank,player\n1,Chris Boswell\n2,Brandon Aubrey\n"
    )
    result = compare_week(7, {"K": predictions}, {"K": actuals}, 2026, tmp_cfg)
    # Only two units, below the five needed to score a week -- but the names
    # must still have resolved, which is what this is checking.
    assert result.unmatched.get("K", []) == []


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------
def _rankings(tmp_cfg, slate_predictions):
    from streamer.rankings import Rankings

    dst = slate_predictions.copy()
    dst["is_home"] = 1
    dst["next_rank"] = np.arange(1, len(dst) + 1)
    dst["next_opponent"] = "XXX"
    dst["expected_points_allowed"] = 20.0
    dst["estimator"] = "ridge"
    kicker = dst.copy()
    kicker["position"] = "K"
    return Rankings(season=2026, week=5, dst=dst, kicker=kicker, context=None)


def test_published_page_contains_the_essentials(tmp_cfg, slate_predictions):
    from streamer.publish import publish, render_page

    rankings = _rankings(tmp_cfg, slate_predictions)
    html = render_page(rankings, tmp_cfg)
    assert "<!doctype html>" in html
    assert "viewport" in html                      # mobile-first
    assert "prefers-color-scheme" in html          # dark-mode friendly
    assert "Week 5" in html
    assert "Two-week stream candidates" in html
    assert "Defense / Special Teams" in html
    assert "Kickers" in html
    assert "<script" not in html                   # no JS dependency

    result = publish(rankings, tmp_cfg)
    assert result.index_path.exists()
    assert result.archive_path.name == "week_5.html"
    assert (tmp_cfg.docs_dir / ".nojekyll").exists()
    assert (tmp_cfg.docs_dir / "latest.json").exists()


def test_page_escapes_untrusted_text(tmp_cfg, slate_predictions):
    from streamer.publish import render_page

    rankings = _rankings(tmp_cfg, slate_predictions)
    rankings.dst.loc[rankings.dst.index[0], "display_name"] = "<img src=x onerror=alert(1)>"
    html = render_page(rankings, tmp_cfg)
    assert "<img src=x" not in html
    assert "&lt;img" in html


def test_degraded_lines_are_flagged_on_the_page(tmp_cfg, slate_predictions):
    from streamer.publish import render_page

    rankings = _rankings(tmp_cfg, slate_predictions)
    rankings.warnings = ["The Odds API returned no usable lines"]
    html = render_page(rankings, tmp_cfg)
    assert "Heads up" in html
    assert "Odds API" in html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [("2023-2025", [2023, 2024, 2025]), ("2024", [2024]), ("2021,2023", [2021, 2023])],
)
def test_season_parsing(text, expected):
    from streamer.cli import _parse_seasons

    assert _parse_seasons(text) == expected


def test_cli_exposes_every_documented_command():
    from streamer.cli import build_parser

    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    commands = set()
    for action in actions:
        commands |= set(action.choices or {})
    for expected in ("rank", "update", "benchmark", "backtest", "publish"):
        assert expected in commands


def test_cli_help_does_not_crash(capsys):
    from streamer.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Model selection round-trip
# ---------------------------------------------------------------------------
def test_selection_round_trips(tmp_cfg):
    from streamer.pipeline import (
        estimator_for,
        load_selection,
        save_selection,
        tuned_alpha_for,
    )

    assert estimator_for("DST", tmp_cfg) == "ridge"      # default with no file
    assert tuned_alpha_for("DST", tmp_cfg) is None

    save_selection({"DST": {"estimator": "gbm", "ridge_alpha": 300.0}}, tmp_cfg)
    assert estimator_for("DST", tmp_cfg) == "gbm"
    assert tuned_alpha_for("DST", tmp_cfg) == pytest.approx(300.0)
    assert load_selection(tmp_cfg)["DST"]["estimator"] == "gbm"


def test_selection_accepts_a_bare_string(tmp_cfg):
    """A hand-written {"DST": "ridge"} must still work."""
    from streamer.pipeline import estimator_for, selection_path

    selection_path(tmp_cfg).write_text('{"DST": "gbm"}')
    assert estimator_for("DST", tmp_cfg) == "gbm"


def test_a_corrupt_selection_file_falls_back(tmp_cfg):
    from streamer.pipeline import estimator_for, selection_path

    selection_path(tmp_cfg).write_text("{not json")
    assert estimator_for("DST", tmp_cfg) == "ridge"


def test_tuned_alpha_overrides_the_config_seed(tmp_cfg):
    from streamer.models.base import ridge_alpha
    from streamer.pipeline import save_selection

    seed = ridge_alpha(tmp_cfg, "DST")
    save_selection({"DST": {"estimator": "ridge", "ridge_alpha": seed + 111.0}}, tmp_cfg)
    assert ridge_alpha(tmp_cfg, "DST") == pytest.approx(seed + 111.0)


# ---------------------------------------------------------------------------
# Terminal table formatting
# ---------------------------------------------------------------------------
def test_table_keeps_correlation_precision():
    """Rounding a rank correlation to one decimal turns a result into zeroes."""
    from streamer.cli import _table

    frame = pd.DataFrame({
        "season": ["2024"], "mae": [4.3172], "rank_corr": [0.3296],
        "rank_corr_edge": [0.0119], "top5_hit_rate": [0.663],
    })
    out = _table(frame, [
        ("season", "Season"), ("mae", "MAE"),
        ("rank_corr", "RankR"), ("rank_corr_edge", "Edge"),
        ("top5_hit_rate", "Top5"),
    ])
    assert "4.32" in out       # MAE to two places
    assert "+0.330" in out     # correlation to three, with a sign
    assert "+0.012" in out
    assert "66%" in out        # rates as percentages


def test_table_handles_missing_and_empty():
    from streamer.cli import _table

    assert "(nothing to show)" in _table(pd.DataFrame(), [("a", "A")])
    frame = pd.DataFrame({"a": [float("nan")], "b": [True], "c": [False]})
    out = _table(frame, [("a", "A"), ("b", "B"), ("c", "C")])
    assert "--" in out
    assert "yes" in out


def test_table_limit_zero_shows_everything():
    from streamer.cli import _table

    frame = pd.DataFrame({"a": list(range(20))})
    assert _table(frame, [("a", "A")], limit=0).count("\n") == 21  # header + rule + 20
