"""Profile isolation.

Two leagues score the same games differently, so the failure mode that matters
is *bleed*: one profile reading the other's scoring, or overwriting its trained
state. These tests assert the separation directly, because a bleed would not
raise -- it would just quietly publish the wrong numbers.
"""

from __future__ import annotations

import pytest

from streamer.actuals import dst_game_lines, kicker_game_lines
from streamer.config import load_config
from streamer.scoring import DstScoring, KickerScoring


def test_binding_a_profile_does_not_mutate_the_original(cfg):
    espn = cfg.for_profile("espn")
    yahoo = cfg.for_profile("yahoo")
    assert espn.profile == "espn"
    assert yahoo.profile == "yahoo"
    # Re-reading after binding the other must be unchanged.
    assert DstScoring.from_config(espn).sack == 2.5
    assert DstScoring.from_config(yahoo).sack == 1.0
    assert DstScoring.from_config(espn).sack == 2.5


def test_for_profile_is_a_no_op_for_the_same_profile(cfg):
    espn = cfg.for_profile("espn")
    assert espn.for_profile("espn") is espn


def test_results_and_reports_are_namespaced(cfg):
    espn = cfg.for_profile("espn")
    yahoo = cfg.for_profile("yahoo")
    assert espn.results_dir != yahoo.results_dir
    assert espn.results_dir.name == "espn"
    assert yahoo.results_dir.name == "yahoo"
    assert espn.reports_dir != yahoo.reports_dir
    # The published page is deliberately shared: one URL carries both.
    assert espn.docs_dir == yahoo.docs_dir


def test_raw_data_cache_is_shared(cfg):
    """Play-by-play does not depend on scoring, so it must not be duplicated."""
    assert cfg.for_profile("espn").raw_dir == cfg.for_profile("yahoo").raw_dir


def test_league_size_follows_the_profile(cfg):
    assert cfg.for_profile("espn").league["teams"] == 10
    assert cfg.for_profile("yahoo").league["teams"] == 14
    assert cfg.for_profile("espn").startable_rank == 12
    assert cfg.for_profile("yahoo").startable_rank == 14


def test_the_same_game_scores_differently_in_each_profile(cfg, toy_pbp, toy_games):
    totals = {}
    for name in cfg.profile_names:
        bound = cfg.for_profile(name)
        dst = dst_game_lines(toy_pbp, toy_games, bound)
        kick = kicker_game_lines(toy_pbp, bound)
        totals[name] = (
            float(dst["fantasy_points"].sum()), float(kick["fantasy_points"].sum())
        )
    assert totals["espn"] != totals["yahoo"]


def test_counting_stats_are_identical_across_profiles(cfg, toy_pbp, toy_games):
    """Only the scoring sheet differs; the events themselves are the same."""
    espn = dst_game_lines(toy_pbp, toy_games, cfg.for_profile("espn"))
    yahoo = dst_game_lines(toy_pbp, toy_games, cfg.for_profile("yahoo"))
    shared = ["sacks", "interceptions", "fumble_recoveries", "points_allowed",
              "yards_allowed", "fourth_down_stops"]
    for column in shared:
        assert espn[column].tolist() == yahoo[column].tolist(), column


def test_default_profile_is_configured_and_valid():
    cfg = load_config()
    assert cfg.default_profile in cfg.profile_names
    assert load_config().profile == cfg.default_profile


@pytest.mark.parametrize("profile", ["espn", "yahoo"])
def test_every_profile_produces_a_complete_scoring_sheet(cfg, profile):
    bound = cfg.for_profile(profile)
    dst = DstScoring.from_config(bound)
    kicker = KickerScoring.from_config(bound)
    # No scoring value may be left as None; a missing event scores zero.
    for field in dst.__dataclass_fields__:
        value = getattr(dst, field)
        if field.endswith("_tiers"):
            continue
        assert value is not None, field
    for field in kicker.__dataclass_fields__:
        assert getattr(kicker, field) is not None, field


def test_cli_selects_profiles(cfg):
    import argparse

    from streamer.cli import _selected

    args = argparse.Namespace(profile="all")
    assert [c.profile for c in _selected(args, cfg)] == cfg.profile_names

    args = argparse.Namespace(profile="yahoo")
    assert [c.profile for c in _selected(args, cfg)] == ["yahoo"]


def test_page_carries_both_profiles_and_no_javascript(tmp_cfg):
    """The switch has to be CSS-only: it must work on a phone with a stale cache."""
    import numpy as np
    import pandas as pd

    from streamer.publish import publish_profiles, render_page
    from streamer.rankings import Rankings

    def _rankings(label):
        frame = pd.DataFrame({
            "season": 2026, "week": 5, "position": "DST",
            "team": ["KC", "BUF", "SF", "DAL", "NYJ"],
            "opponent": ["DEN", "MIA", "LA", "PHI", "NE"],
            "display_name": [f"{label} {t}" for t in ["KC", "BUF", "SF", "DAL", "NYJ"]],
            "is_home": 1, "rank": range(1, 6),
            "expected_points": np.linspace(11, 4, 5),
            "floor": np.linspace(4, 0, 5), "ceiling": np.linspace(18, 9, 5),
            "p_top12": np.linspace(0.9, 0.3, 5),
            "expected_points_allowed": 20.0, "estimator": "ridge",
            "rationale": "because", "two_week_hold": False,
            "next_rank": np.nan, "next_opponent": None,
        })
        return Rankings(season=2026, week=5, dst=frame, kicker=frame.copy(), context=None)

    ranked = {"espn": _rankings("E"), "yahoo": _rankings("Y")}
    html = render_page(ranked, tmp_cfg)
    assert "<script" not in html
    assert 'id="panel-espn"' in html and 'id="panel-yahoo"' in html
    assert 'id="profile-espn"' in html and 'id="profile-yahoo"' in html
    assert "ESPN, 10-team" in html and "Yahoo, 14-team" in html
    # One radio per profile, exactly one of them pre-selected.
    assert html.count('class="profile-radio"') == 2
    assert html.count(" checked>") == 1
    # The switch is driven purely by :checked sibling rules.
    assert "#profile-yahoo:checked ~ .wrap #panel-yahoo" in html

    result = publish_profiles(ranked, tmp_cfg)
    assert result.index_path.exists()
    assert (tmp_cfg.docs_dir / "latest.json").exists()


def test_single_profile_page_has_no_switch(tmp_cfg):
    import numpy as np
    import pandas as pd

    from streamer.publish import render_page
    from streamer.rankings import Rankings

    frame = pd.DataFrame({
        "season": 2026, "week": 5, "position": "DST", "team": ["KC"],
        "opponent": ["DEN"], "display_name": ["KC"], "is_home": [1], "rank": [1],
        "expected_points": [9.0], "floor": [3.0], "ceiling": [15.0],
        "p_top12": [0.5], "expected_points_allowed": [20.0], "estimator": ["ridge"],
        "rationale": ["because"], "two_week_hold": [False],
        "next_rank": [np.nan], "next_opponent": [None],
    })
    rankings = Rankings(season=2026, week=5, dst=frame, kicker=frame.copy(), context=None)
    html = render_page({"espn": rankings}, tmp_cfg)
    assert '<div class="switch">' not in html
    assert 'id="panel-espn"' in html


# ---------------------------------------------------------------------------
# Subvertadown files per profile
# ---------------------------------------------------------------------------
def test_profile_specific_subvertadown_file_wins(tmp_cfg):
    from streamer.benchmark import read_subvertadown, subvertadown_path

    espn = tmp_cfg.for_profile("espn")
    yahoo = tmp_cfg.for_profile("yahoo")

    shared = tmp_cfg.data_dir / "subvertadown_week_5.csv"
    shared.write_text("rank,team\n1,KC\n2,BUF\n")
    # With only a shared file, both profiles read it.
    assert subvertadown_path(5, espn) == shared
    assert subvertadown_path(5, yahoo) == shared

    specific = tmp_cfg.data_dir / "subvertadown_week_5_yahoo.csv"
    specific.write_text("rank,team\n1,SF\n2,DAL\n")
    assert subvertadown_path(5, yahoo) == specific
    assert subvertadown_path(5, espn) == shared          # ESPN still on the shared one
    assert list(read_subvertadown(5, yahoo)["entry"]) == ["SF", "DAL"]
    assert list(read_subvertadown(5, espn)["entry"]) == ["KC", "BUF"]


def test_missing_file_names_the_profile(tmp_cfg):
    from streamer.benchmark import read_subvertadown

    with pytest.raises(FileNotFoundError, match="Yahoo"):
        read_subvertadown(9, tmp_cfg.for_profile("yahoo"))


# ---------------------------------------------------------------------------
# Yards ladder wiring
# ---------------------------------------------------------------------------
def test_yards_model_only_fits_where_the_league_scores_it(cfg):
    import numpy as np
    import pandas as pd

    from streamer.models.positions import DstModel

    rng = np.random.default_rng(3)
    n = 700
    frame = pd.DataFrame({
        "season": rng.choice([2023, 2024], n),
        "week": rng.integers(1, 18, n),
        "game_id": [f"g{i}" for i in range(n)],
        "team": rng.choice(list("ABCDEFGH"), n),
        "opponent": rng.choice(list("IJKLMNOP"), n),
        "opp_implied_total": rng.normal(22, 4, n),
        "total_line": rng.normal(44, 4, n),
        "team_spread": rng.normal(0, 5, n),
        "is_home": rng.integers(0, 2, n),
        "def_points_allowed_per_drive": rng.normal(2.0, 0.4, n),
        "opp_points_per_drive": rng.normal(2.0, 0.4, n),
        "opp_drives_per_game": rng.normal(11, 1, n),
        "opp_plays_per_game": rng.normal(63, 5, n),
        "opp_dropbacks_per_game": rng.normal(36, 4, n),
        "opp_pass_rate": rng.normal(0.58, 0.05, n),
        "opp_neutral_plays_per_game": rng.normal(30, 4, n),
        "def_yards_allowed_per_game": rng.normal(338, 30, n),
        "points_allowed": rng.integers(0, 40, n),
        "yards_allowed": rng.integers(150, 500, n),
    })
    frame["fantasy_points"] = rng.normal(8, 5, n)
    frame["big_play_points"] = frame["fantasy_points"] - 3

    espn = DstModel.fit(frame, cfg.for_profile("espn"), current_season=2024)
    yahoo = DstModel.fit(frame, cfg.for_profile("yahoo"), current_season=2024)
    assert espn.yards_model is not None
    assert yahoo.yards_model is None

    espn_out = espn.predict(frame.head(20)).frame
    yahoo_out = yahoo.predict(frame.head(20)).frame
    assert espn_out["expected_yards_allowed"].notna().all()
    assert (espn_out["expected_yards_tier_points"] != 0).any()
    # Yahoo reports the column for a uniform schema, but contributes nothing.
    assert (yahoo_out["expected_yards_tier_points"] == 0).all()
    assert yahoo_out["expected_yards_allowed"].isna().all()


def test_yards_ladder_probabilities_are_a_distribution(cfg):
    import numpy as np
    import pandas as pd

    from streamer.models.tiers import LadderModel

    rng = np.random.default_rng(8)
    n = 500
    frame = pd.DataFrame({
        "season": 2024, "week": rng.integers(1, 18, n),
        "opp_plays_per_game": rng.normal(63, 5, n),
        "opp_drives_per_game": rng.normal(11, 1, n),
        "opp_dropbacks_per_game": rng.normal(36, 4, n),
        "opp_points_per_drive": rng.normal(2.0, 0.4, n),
        "opp_pass_rate": rng.normal(0.58, 0.05, n),
        "opp_neutral_plays_per_game": rng.normal(30, 4, n),
        "opp_implied_total": rng.normal(22, 4, n),
        "total_line": rng.normal(44, 4, n),
        "team_spread": rng.normal(0, 5, n),
        "is_home": rng.integers(0, 2, n),
        "def_points_allowed_per_drive": rng.normal(2.0, 0.4, n),
        "def_yards_allowed_per_game": rng.normal(338, 30, n),
        "yards_allowed": rng.normal(338, 80, n).clip(50),
    })
    model = LadderModel.fit(frame, cfg.for_profile("espn"), ladder="yards")
    probs = model.tier_probabilities(frame.head(25))
    assert probs.shape == (25, 9)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()


def test_yards_ladder_is_unavailable_for_yahoo(cfg):
    import numpy as np
    import pandas as pd

    from streamer.models.tiers import LadderModel

    frame = pd.DataFrame({
        "season": [2024] * 50, "week": range(1, 51),
        "opp_plays_per_game": np.full(50, 63.0),
        "yards_allowed": np.full(50, 330.0),
    })
    model = LadderModel.fit(frame, cfg.for_profile("yahoo"), ladder="yards")
    with pytest.raises(ValueError, match="does not score yards"):
        _ = model.tiers
