"""How the projections are doing: graded every week against what happened.

Two tests, on the players each source covered who actually played -- and, in
two further views, counting the players who did not play (a projection for
someone who ends up inactive is a miss, scored against zero) and only the
players likely to be started (the top of each position by our final number):

* **Start/sit** -- of any two players at the same position that week, did
  the source rank the one who scored more higher? Every source can take this
  test, rankings included.
* **Average miss** -- points off, for sources that give points.

Every source is graded on its own players with our final number graded on
the same players beside it, so a source that only covers the easy half of
the league cannot look better for it.

**Rest of season**: from each week's log, our season values and the
FantasyPros rest-of-season ranks are graded against points per game scored
since (players with 3+ games), by rank correlation within each position.

Inputs: ``results/<profile>/skill_log.parquet`` (ours, the platform's, the
betting market's; committed) and, when FantasyPros is set up, the runner-only
consensus log. Output: ``results/<profile>/scorecard.json`` -- aggregate
scores only, safe to publish.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config

#: (column, label) of every point source in the projection log.
POINT_SOURCES = [
    ("blended_projection", "Our final number"),
    ("model_projection", "Our model alone"),
    ("platform_projection", "{platform}'s projection"),
    ("vegas_points", "Betting props"),
    ("fp_projection", "FantasyPros projections"),
    ("sleeper_projection", "Sleeper's projection"),
]
RANK_SOURCES = [("ecr_week_pos_rank", "FantasyPros rankings")]
MIN_ROS_GAMES = 3
#: Likely starters: the top of each position by our final number that week
#: (about a 12-team league's starters plus flex).
STARTERS = {"QB": 12, "RB": 24, "WR": 30, "TE": 12}
SCOPES = (("played", "Players who played"),
          ("all", "Counting players who did not play, as zero"),
          ("starters", "Likely starters only, counting those who did not play"))


@dataclass
class Grade:
    source: str
    players: int
    weeks: int
    pairs: float | None        # start/sit accuracy
    miss: float | None         # average miss, points
    ours_pairs: float | None   # our final number on the same players
    ours_miss: float | None
    scope: str = "played"      # "played", "all" (inactive counted as zero) or "starters"


@dataclass
class Scorecard:
    grades: list[Grade] = field(default_factory=list)
    per_week: list[dict] = field(default_factory=list)
    ros: list[dict] = field(default_factory=list)
    #: The consensus's share of season values, and the graded games behind it.
    season_weight: dict | None = None
    #: Projected players with no box score in a finished week.
    inactive: int = 0

    def to_json(self) -> str:
        return json.dumps({"grades": [g.__dict__ for g in self.grades], "per_week": self.per_week,
                           "ros": self.ros, "season_weight": self.season_weight,
                           "inactive": self.inactive}, indent=2)


def pair_accuracy(pred: np.ndarray, actual: np.ndarray, groups: np.ndarray) -> tuple[int, int]:
    """(right, total) over every pair in the same group, ties skipped."""
    right = total = 0
    for g in np.unique(groups):
        m = groups == g
        p, a = pred[m], actual[m]
        if len(p) < 2:
            continue
        dp = np.sign(p[:, None] - p[None, :])
        da = np.sign(a[:, None] - a[None, :])
        upper = np.triu(np.ones_like(dp, bool), 1)
        ok = upper & (dp != 0) & (da != 0)
        right += int((dp == da)[ok].sum())
        total += int(ok.sum())
    return right, total


def _actuals(history: pd.DataFrame) -> pd.DataFrame:
    return (history.groupby(["player_id", "season", "week"])["fantasy_points_ppr"].sum()
            .rename("actual").reset_index().rename(columns={"player_id": "nfl_id"}))


def graded_log(cfg: Config, history: pd.DataFrame, include_inactive: bool = False) -> pd.DataFrame:
    """The projection log with the consensus and actual points joined.

    ``include_inactive``: also the projected players with no box score in a
    week the history has moved past (one that is over), scored 0 -- ``played``
    says which is which. A week still being played is never counted this way:
    a late game's players have not had their chance yet.
    """
    path = cfg.results_dir / "skill_log.parquet"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_parquet(path).dropna(subset=["nfl_id"])
    d = d.sort_values("logged_at").drop_duplicates(["season", "week", "nfl_id"], keep="last")
    from ..data import fantasypros as fp

    fplog = fp.cache_dir(cfg) / "log.parquet"
    if fplog.exists():
        c = pd.read_parquet(fplog).drop_duplicates(["season", "week", "nfl_id"], keep="last")
        d = d.merge(c[["season", "week", "nfl_id", "fp_projection", "ecr_week_pos_rank"]],
                    on=["season", "week", "nfl_id"], how="left")
    d = d.merge(_actuals(history), on=["nfl_id", "season", "week"], how="left")
    d["played"] = d["actual"].notna()
    if not include_inactive:
        return d[d["played"]].reset_index(drop=True)
    newest = history.groupby("season")["week"].max()
    over = d["week"] < d["season"].map(newest).fillna(0)
    d = d[d["played"] | (over & (d["blended_projection"] > 0))]
    return d.assign(actual=d["actual"].fillna(0.0)).reset_index(drop=True)


def starters_only(d: pd.DataFrame) -> pd.DataFrame:
    """The likely starters: each week's top of every position by our final number."""
    if d.empty:
        return d
    rank = d.groupby(["season", "week", "position"])["blended_projection"].rank(ascending=False, method="first")
    return d[rank <= d["position"].map(STARTERS).fillna(0)]


def _grades(d: pd.DataFrame, sources: list, platform: str, scope: str) -> list[Grade]:
    out = []
    for col, label, is_rank in sources:
        if col not in d:
            continue
        s = d[d[col].notna() & d["blended_projection"].notna()]
        if not is_rank:
            s = s[s[col] > 0]
        if len(s) < 10:
            continue
        right = total = o_right = o_total = 0
        for (_season, _week), w in s.groupby(["season", "week"]):
            pred = -w[col].to_numpy(float) if is_rank else w[col].to_numpy(float)
            r, t = pair_accuracy(pred, w.actual.to_numpy(float), w.position.to_numpy())
            orr, ot = pair_accuracy(w.blended_projection.to_numpy(float), w.actual.to_numpy(float),
                                    w.position.to_numpy())
            right, total, o_right, o_total = right + r, total + t, o_right + orr, o_total + ot
        out.append(Grade(
            source=label, players=len(s), weeks=int(s.groupby(["season", "week"]).ngroups),
            pairs=right / total if total else None,
            miss=None if is_rank else float(np.abs(s[col] - s.actual).mean()),
            ours_pairs=o_right / o_total if o_total else None,
            ours_miss=float(np.abs(s.blended_projection - s.actual).mean()), scope=scope))
    return out


def build(cfg: Config, history: pd.DataFrame, platform: str) -> Scorecard:
    d = graded_log(cfg, history)
    card = Scorecard()
    if d.empty:
        return card
    sources = [(c, label.format(platform=platform), False) for c, label in POINT_SOURCES] + \
              [(c, label, True) for c, label in RANK_SOURCES]
    card.grades = _grades(d, sources, platform, "played")
    everyone = graded_log(cfg, history, include_inactive=True)
    card.grades += _grades(everyone, sources, platform, "all")
    card.grades += _grades(starters_only(everyone), sources, platform, "starters")
    card.inactive = int((~everyone["played"]).sum()) if "played" in everyone else 0
    for (season, week), w in d.groupby(["season", "week"]):
        row = {"season": int(season), "week": int(week), "players": int(len(w))}
        for col, label, _r in sources:
            if col in w and not label.startswith("FantasyPros rank"):
                v = w[w[col].notna() & (w[col] > 0)] if col in w else w.iloc[0:0]
                if len(v) >= 10:
                    row[label] = round(float(np.abs(v[col] - v.actual).mean()), 2)
        card.per_week.append(row)
    card.ros = ros_benchmark(cfg, history)
    return card


def ros_benchmark(cfg: Config, history: pd.DataFrame) -> list[dict]:
    """Our season values and the consensus ranks against points per game
    since, by rank correlation within position (players with 3+ games)."""
    from ..data import fantasypros as fp

    path = fp.cache_dir(cfg) / "log.parquet"
    if not path.exists():
        return []
    c = pd.read_parquet(path).dropna(subset=["ecr_ros_pos_rank", "our_ros"])
    out = []
    for (season, week), w in c.groupby(["season", "week"]):
        later = history[(history.season == season) & (history.week >= week)]
        ppg = later.groupby("player_id")["fantasy_points_ppr"].agg(["mean", "count"])
        ppg = ppg[ppg["count"] >= MIN_ROS_GAMES]
        weeks_since = int(later.week.nunique())
        if weeks_since < MIN_ROS_GAMES:
            continue
        j = w.merge(ppg, left_on="nfl_id", right_index=True)
        ours, theirs = [], []
        for _pos, g in j.groupby("position"):
            if len(g) < 8:
                continue
            ours.append(g.our_ros.corr(g["mean"], method="spearman"))
            theirs.append((-g.ecr_ros_pos_rank).corr(g["mean"], method="spearman"))
        if ours:
            out.append({"season": int(season), "week": int(week), "weeks_since": weeks_since,
                        "players": int(len(j)), "ours": round(float(np.mean(ours)), 3),
                        "consensus": round(float(np.mean(theirs)), 3)})
    return out


def save(cfg: Config, card: Scorecard) -> None:
    (cfg.results_dir / "scorecard.json").write_text(card.to_json() + "\n", encoding="utf-8")
