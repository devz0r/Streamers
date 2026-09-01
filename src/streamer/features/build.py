"""Model-ready feature matrices for the Kicker and D/ST models.

Two public entry points, :func:`build_kicker_features` and
:func:`build_dst_features`, each returning one row per unit-game with:

* join keys (``season``, ``week``, ``game_id``, ``team``, ``opponent``)
* the feature columns named in :data:`KICKER_FEATURES` / :data:`DST_FEATURES`
* the training targets, where the game has already been played

Every feature is either a Vegas number known before kickoff or a leak-free
prior built from strictly earlier games (see :mod:`streamer.features.rolling`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..actuals import dst_game_lines, kicker_game_lines
from ..config import Config, get_config
from ..data.weather import resolve_weather
from ..scoring import FG_BUCKETS
from .rolling import RateSpec, add_shrunk_rate, decayed_per_game
from .team_week import build_team_week

# ---------------------------------------------------------------------------
# Feature registries. The factor ledger and every model read these, so a
# feature added here automatically appears in the weekly correlation report.
# ---------------------------------------------------------------------------
KICKER_FEATURES: tuple[str, ...] = (
    "team_implied_total",
    "total_line",
    "team_spread",
    "opp_implied_total",
    "is_home",
    "is_dome",
    "wind",
    "off_fga_per_drive",
    "off_rz_td_rate",
    "off_rz_trips_per_drive",
    "off_drives_per_game",
    "off_plays_per_game",
    "off_td_per_drive",
    "off_fg_share_of_scores",
    "k_acc_0_39",
    "k_acc_40_49",
    "k_acc_50_plus",
    "k_long_att_share",
    "k_pat_acc",
    "def_rz_td_rate_allowed",
    "def_points_per_drive_allowed",
    "def_drives_per_game_allowed",
    "implied_x_fga_rate",
    "implied_x_rz_stall",
    "wind_x_long_share",
)

DST_FEATURES: tuple[str, ...] = (
    "opp_implied_total",
    "total_line",
    "team_spread",
    "is_home",
    "opp_sack_rate_allowed",
    "opp_pressure_rate_allowed",
    "opp_pressure_to_sack",
    "opp_int_rate",
    "opp_twp_rate",
    "opp_fumble_rate",
    "opp_dropbacks_per_game",
    "opp_plays_per_game",
    "opp_neutral_plays_per_game",
    "opp_pass_rate",
    "opp_points_per_drive",
    "opp_drives_per_game",
    "def_sack_rate",
    "def_pressure_rate",
    "def_int_rate",
    "def_fumble_rec_per_game",
    "def_takeaway_td_per_game",
    "def_points_allowed_per_drive",
    "def_big_play_points_per_game",
    "sack_opportunity",
    "takeaway_opportunity",
)

#: The Vegas block each position's projection is *anchored* on. The model fits
#: this first, lightly regularised, then fits every other factor against what it
#: leaves behind -- "start from a Vegas-derived baseline, then apply
#: adjustments", implemented literally.
VEGAS_ANCHOR: dict[str, tuple[str, ...]] = {
    "K": ("team_implied_total", "total_line", "team_spread", "is_home", "is_dome", "wind"),
    "DST": ("opp_implied_total", "total_line", "team_spread", "is_home"),
}

#: Human-readable factor names for the weekly review and the published page.
FACTOR_LABELS: dict[str, str] = {
    "team_implied_total": "Team implied total",
    "opp_implied_total": "Opponent implied total",
    "total_line": "Game total",
    "team_spread": "Point spread",
    "is_home": "Home field",
    "is_dome": "Dome / indoors",
    "wind": "Wind (mph)",
    "off_fga_per_drive": "FG attempts per drive",
    "off_rz_td_rate": "Red-zone TD rate",
    "off_rz_trips_per_drive": "Red-zone trips per drive",
    "off_drives_per_game": "Drives per game",
    "off_plays_per_game": "Plays per game (pace)",
    "off_td_per_drive": "TDs per drive",
    "off_fg_share_of_scores": "FG share of scores (stall rate)",
    "k_acc_0_39": "Kicker accuracy 0-39",
    "k_acc_40_49": "Kicker accuracy 40-49",
    "k_acc_50_plus": "Kicker accuracy 50+",
    "k_long_att_share": "Kicker 50+ attempt share",
    "k_pat_acc": "Kicker PAT accuracy",
    "def_rz_td_rate_allowed": "Opp red-zone TD rate allowed",
    "def_points_per_drive_allowed": "Opp points per drive allowed",
    "def_drives_per_game_allowed": "Opp drives per game allowed",
    "implied_x_fga_rate": "Implied total x FG rate",
    "implied_x_rz_stall": "Implied total x red-zone stall",
    "wind_x_long_share": "Wind x 50+ attempt share",
    "opp_sack_rate_allowed": "Opp sack rate allowed (O-line)",
    "opp_pressure_rate_allowed": "Opp pressure rate allowed",
    "opp_pressure_to_sack": "Opp pressure-to-sack rate",
    "opp_int_rate": "Opp INT rate",
    "opp_twp_rate": "Opp turnover-worthy play rate",
    "opp_fumble_rate": "Opp fumble-lost rate",
    "opp_dropbacks_per_game": "Opp dropbacks per game",
    "opp_plays_per_game": "Opp plays per game (pace)",
    "opp_neutral_plays_per_game": "Opp neutral-script pace",
    "opp_pass_rate": "Opp pass rate",
    "opp_points_per_drive": "Opp points per drive",
    "opp_drives_per_game": "Opp drives per game",
    "def_sack_rate": "Defense sack rate",
    "def_pressure_rate": "Defense pressure rate",
    "def_int_rate": "Defense INT rate",
    "def_fumble_rec_per_game": "Defense fumble recoveries/game",
    "def_takeaway_td_per_game": "Defense TDs per game",
    "def_points_allowed_per_drive": "Defense points allowed per drive",
    "def_big_play_points_per_game": "Defense big-play points/game",
    "sack_opportunity": "Sack opportunity (def x O-line x volume)",
    "takeaway_opportunity": "Takeaway opportunity",
}


def _safe_div(num, den, default=0.0):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = np.where(den > 0, num / den.replace(0, np.nan), default)
    return pd.Series(out, index=getattr(num, "index", None)).fillna(default)


# ---------------------------------------------------------------------------
# Shared team-level priors
# ---------------------------------------------------------------------------
def team_priors(
    pbp: pd.DataFrame,
    games: pd.DataFrame,
    cfg: Config | None = None,
    future_games: pd.DataFrame | None = None,
    position: str = "DST",
) -> pd.DataFrame:
    """Leak-free offensive and defensive priors for every team-game.

    ``position`` selects the shrinkage strength; see
    :meth:`streamer.config.Config.team_prior_games`.
    """
    cfg = cfg or get_config()
    k = cfg.team_prior_games(position)
    tw = build_team_week(pbp, games, future_games)
    if tw.empty:
        return tw
    tw = tw.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)

    tw["scores"] = tw["drive_tds"] + tw["drive_fgs"]
    tw["faced_scores"] = tw["faced_drive_tds"] + tw["faced_drive_fgs"]
    tw["one"] = 1.0

    offense_specs = [
        RateSpec("off_fga_per_drive", "fga", "drives", k),
        RateSpec("off_rz_td_rate", "rz_tds", "rz_trips", k),
        RateSpec("off_rz_trips_per_drive", "rz_trips", "drives", k),
        RateSpec("off_td_per_drive", "drive_tds", "drives", k),
        RateSpec("off_fg_share_of_scores", "drive_fgs", "scores", k),
        RateSpec("off_drives_per_game", "drives", "one", k),
        RateSpec("off_plays_per_game", "plays", "one", k),
        RateSpec("off_neutral_plays_per_game", "neutral_plays", "one", k),
        RateSpec("off_dropbacks_per_game", "dropbacks", "one", k),
        RateSpec("off_sack_rate_allowed", "sacks_allowed", "dropbacks", k),
        RateSpec("off_pressure_rate_allowed", "qb_hits_allowed", "dropbacks", k),
        RateSpec("off_pressure_to_sack", "sacks_allowed", "qb_hits_allowed", k),
        RateSpec("off_int_rate", "ints_thrown", "dropbacks", k),
        RateSpec("off_fumble_rate", "fumbles_lost", "plays", k),
        RateSpec("off_pass_rate", "pass_att", "plays", k),
        RateSpec("off_points_per_drive", "team_score", "drives", k),
        RateSpec("off_long_fga_share", "fga_50_plus", "fga", k),
    ]
    defense_specs = [
        RateSpec("def_rz_td_rate_allowed", "faced_rz_tds", "faced_rz_trips", k),
        RateSpec("def_points_per_drive_allowed", "opp_score", "faced_drives", k),
        RateSpec("def_drives_per_game_allowed", "faced_drives", "one", k),
        RateSpec("def_sack_rate", "faced_sacks_allowed", "faced_dropbacks", k),
        RateSpec("def_pressure_rate", "faced_qb_hits_allowed", "faced_dropbacks", k),
        RateSpec("def_int_rate", "faced_ints_thrown", "faced_dropbacks", k),
        RateSpec("def_fga_per_drive_allowed", "faced_fga", "faced_drives", k),
    ]
    for spec in offense_specs + defense_specs:
        if spec.numerator in tw.columns and spec.denominator in tw.columns:
            tw = add_shrunk_rate(tw, spec, ["team"])
    return tw


# ---------------------------------------------------------------------------
# Kicker
# ---------------------------------------------------------------------------
def kicker_priors(
    pbp: pd.DataFrame,
    cfg: Config | None = None,
    future_kickers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per kicker-game accuracy priors, shrunk to league mean by attempts.

    ``future_kickers`` adds placeholder rows (``season``, ``week``, ``game_id``,
    ``team``, ``player_id``, ``player_name``) for an upcoming slate.
    """
    cfg = cfg or get_config()
    k = float(cfg.shrinkage["kicker_accuracy_prior_attempts"])
    lines = kicker_game_lines(pbp, cfg)
    if lines.empty:
        return lines
    lines["is_future"] = 0
    if future_kickers is not None and not future_kickers.empty:
        extra = future_kickers.copy()
        for col in lines.columns:
            if col not in extra.columns:
                extra[col] = 0.0 if pd.api.types.is_numeric_dtype(lines[col]) else None
        extra["is_future"] = 1
        lines = pd.concat([lines, extra[lines.columns]], ignore_index=True)
    lines = lines.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    for bucket, _low, _high in FG_BUCKETS:
        lines[f"att_{bucket}"] = lines[f"fg_made_{bucket}"] + lines[f"fg_missed_{bucket}"]
    lines["pat_att"] = lines["pat_made"] + lines["pat_missed"]
    lines["fga_50_plus"] = lines["att_50_plus"]

    # Prior strength is expressed in attempts, and `add_shrunk_rate` multiplies
    # prior_games by the mean denominator per game, so convert here.
    specs = []
    for bucket, _low, _high in FG_BUCKETS:
        mean_att = max(lines[f"att_{bucket}"].mean(), 1e-6)
        specs.append(
            RateSpec(f"k_acc_{bucket}", f"fg_made_{bucket}", f"att_{bucket}", k / mean_att)
        )
    mean_pat = max(lines["pat_att"].mean(), 1e-6)
    specs.append(RateSpec("k_pat_acc", "pat_made", "pat_att", k / mean_pat))
    mean_fga = max(lines["fg_att"].mean(), 1e-6)
    specs.append(RateSpec("k_long_att_share", "fga_50_plus", "fg_att", k / mean_fga))

    for spec in specs:
        lines = add_shrunk_rate(lines, spec, ["player_id"])
    return lines


def build_kicker_features(
    pbp: pd.DataFrame,
    games: pd.DataFrame,
    cfg: Config | None = None,
    future_games: pd.DataFrame | None = None,
    future_kickers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per kicker-game with :data:`KICKER_FEATURES` and the target."""
    cfg = cfg or get_config()
    priors = team_priors(pbp, games, cfg, future_games, position="K")
    kick = kicker_priors(pbp, cfg, future_kickers)
    if priors.empty or kick.empty:
        return pd.DataFrame(columns=["season", "week", "game_id", "team", *KICKER_FEATURES])

    keep_k = [
        "season", "week", "game_id", "team", "player_id", "player_name",
        "fantasy_points", "fg_att", "fg_made", "pat_made",
        "k_acc_0_39", "k_acc_40_49", "k_acc_50_plus", "k_pat_acc", "k_long_att_share",
        "k_acc_0_39__n", "k_long_att_share__n",
    ]
    kick = kick[[c for c in keep_k if c in kick.columns]]

    keep_t = [
        "season", "week", "game_id", "team", "opponent", "is_home", "is_dome",
        "roof", "wind", "temp", "team_spread", "total_line",
        "team_implied_total", "opp_implied_total",
        "off_fga_per_drive", "off_rz_td_rate", "off_rz_trips_per_drive",
        "off_drives_per_game", "off_plays_per_game", "off_td_per_drive",
        "off_fg_share_of_scores", "off_long_fga_share",
    ]
    team = priors[[c for c in keep_t if c in priors.columns]].copy()

    # Opponent defensive priors come from the opponent's own row.
    opp_cols = ["def_rz_td_rate_allowed", "def_points_per_drive_allowed",
                "def_drives_per_game_allowed", "def_fga_per_drive_allowed"]
    opp = priors[["season", "week", "game_id", "team", *[c for c in opp_cols if c in priors.columns]]]
    opp = opp.rename(columns={"team": "opponent"})
    team = team.merge(opp, on=["season", "week", "game_id", "opponent"], how="left")

    df = kick.merge(team, on=["season", "week", "game_id", "team"], how="inner")
    return _finalize_kicker(df, cfg)


def _finalize_kicker(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = resolve_weather(df.assign(home_team=np.where(df["is_home"] == 1, df["team"], df["opponent"])),
                         week=int(df["week"].iloc[0]) if len(df) else 1, cfg=cfg)
    df["implied_x_fga_rate"] = df["team_implied_total"] * df["off_fga_per_drive"]
    df["implied_x_rz_stall"] = df["team_implied_total"] * (1.0 - df["off_rz_td_rate"])
    df["wind_x_long_share"] = df["wind"] * df.get("k_long_att_share", 0.0)
    for col in KICKER_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# D/ST
# ---------------------------------------------------------------------------
def build_dst_features(
    pbp: pd.DataFrame,
    games: pd.DataFrame,
    cfg: Config | None = None,
    future_games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per defense-game with :data:`DST_FEATURES` and the targets."""
    cfg = cfg or get_config()
    priors = team_priors(pbp, games, cfg, future_games, position="DST")
    if priors.empty:
        return pd.DataFrame(columns=["season", "week", "game_id", "team", *DST_FEATURES])

    lines = dst_game_lines(pbp, games, cfg)

    # A defense's own prior on the non-tier component: how many big-play points
    # it has been generating, decayed and shrunk like every other prior.
    if not lines.empty:
        bp = lines[["season", "week", "game_id", "team", "big_play_points",
                    "defensive_tds", "return_tds", "fumble_recoveries"]].copy()
        bp["takeaway_tds"] = bp["defensive_tds"] + bp["return_tds"]
        priors = priors.merge(bp, on=["season", "week", "game_id", "team"], how="left")
        for col in ("big_play_points", "fumble_recoveries", "takeaway_tds"):
            priors[col] = priors[col].fillna(0.0)
        priors = priors.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
        for col, out in (
            ("big_play_points", "def_big_play_points_per_game"),
            ("fumble_recoveries", "def_fumble_rec_per_game"),
            ("takeaway_tds", "def_takeaway_td_per_game"),
        ):
            priors = decayed_per_game(priors, col, ["team"], out)

    self_cols = [
        "season", "week", "game_id", "team", "opponent", "is_home",
        "team_spread", "total_line", "team_implied_total", "opp_implied_total",
        "is_dome", "wind", "temp",
        "def_sack_rate", "def_pressure_rate", "def_int_rate",
        "def_points_per_drive_allowed", "def_big_play_points_per_game",
        "def_fumble_rec_per_game", "def_takeaway_td_per_game",
    ]
    df = priors[[c for c in self_cols if c in priors.columns]].copy()
    df = df.rename(columns={"def_points_per_drive_allowed": "def_points_allowed_per_drive"})

    # What the opponent's offense brings, taken from the opponent's own row.
    opp_map = {
        "off_sack_rate_allowed": "opp_sack_rate_allowed",
        "off_pressure_rate_allowed": "opp_pressure_rate_allowed",
        "off_pressure_to_sack": "opp_pressure_to_sack",
        "off_int_rate": "opp_int_rate",
        "off_fumble_rate": "opp_fumble_rate",
        "off_dropbacks_per_game": "opp_dropbacks_per_game",
        "off_plays_per_game": "opp_plays_per_game",
        "off_neutral_plays_per_game": "opp_neutral_plays_per_game",
        "off_pass_rate": "opp_pass_rate",
        "off_points_per_drive": "opp_points_per_drive",
        "off_drives_per_game": "opp_drives_per_game",
    }
    have = {k: v for k, v in opp_map.items() if k in priors.columns}
    opp = priors[["season", "week", "game_id", "team", *have.keys()]].rename(
        columns={"team": "opponent", **have}
    )
    df = df.merge(opp, on=["season", "week", "game_id", "opponent"], how="left")

    if "opp_int_rate" in df.columns and "opp_fumble_rate" in df.columns:
        df["opp_twp_rate"] = df["opp_int_rate"] + df["opp_fumble_rate"]

    # Opportunity terms: a defense only converts what the matchup offers, and
    # volume multiplies both sides. These are the interactions a linear model
    # cannot discover on its own.
    df["sack_opportunity"] = (
        df.get("def_sack_rate", 0.0)
        * df.get("opp_sack_rate_allowed", 0.0)
        * df.get("opp_dropbacks_per_game", 0.0)
        * 100.0
    )
    df["takeaway_opportunity"] = (
        df.get("def_int_rate", 0.0)
        * df.get("opp_twp_rate", 0.0)
        * df.get("opp_dropbacks_per_game", 0.0)
        * 100.0
    )

    if not lines.empty:
        target = lines[["season", "week", "game_id", "team", "fantasy_points",
                        "big_play_points", "tier_points", "points_allowed",
                        "sacks", "interceptions"]]
        df = df.merge(target, on=["season", "week", "game_id", "team"], how="left")

    for col in DST_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
