# streamer

Weekly **D/ST and Kicker streaming rankings** for fantasy football, in the style
of Subvertadown: Vegas-anchored projections that re-calibrate themselves against
their own results every week, published to a phone-friendly page you can open
from anywhere.

Defaults are **ESPN standard D/ST and Kicker scoring** in a 10-team league
context. Everything lives in [`config.yaml`](config.yaml).

```
$ streamer rank --week 1

=== Week 1 (2026) — D/ST ===
  #  Defense               Opp  Proj  Floor  Ceil  Top12  Hold  Why
  1  Jacksonville Jaguars  CLE   9.1    3.3  14.8    39%        opp implied 16.5, 39% to hold them under 14,
                                                                opp sacked on 8.2% of dropbacks, 7.5-point favourite
  2  Los Angeles Chargers  ARI   8.1    2.3  13.8    32%   yes  opp implied 18.0, 34% to hold them under 14,
                                                                10.5-point favourite
  3  Seattle Seahawks       NE   6.9    1.1  12.6    26%   yes  opp implied 20.5, opp sacked on 8.8% of dropbacks
```

---

## How it works

**Vegas is the backbone.** Every projection starts from a lightly-regularised
regression on the betting market — implied team total, game total, spread, home
field — and then applies *adjustments* from everything else, heavily regularised
so they can only move the number as far as their measured signal justifies. This
is why the model beats a Vegas-only ranking instead of drowning in its own
features. See [DECISIONS.md](DECISIONS.md) for the evidence.

**Kickers** are projected from team implied total, the team's field-goal
*generation* profile (attempts per drive, red-zone stall rate), the kicker's own
distance-bucket accuracy regressed toward league mean by attempt volume, dome
and wind, and the opponent's red-zone defence.

**D/ST** splits into two pieces with different statistical character. The
points-allowed component is modelled as a **probability distribution over the
ESPN tier ladder**, not a point estimate — a defence projected to allow 20.5
points is a mixture that is mostly 0 and -1 with real mass on +3 and +4. The
big-play component (sacks, takeaways, scores) is driven by opponent sack rate
allowed, pressure-to-sack, interception rate, pace and home field. Output is
expected points plus **P(top-12)**.

**It learns every week.** `streamer update` scores what it predicted, appends to
`results/history.parquet`, re-fits with current-season games weighted 2x, and
rewrites a **factor-correlation ledger** tracking every input's correlation with
actual outcomes across three windows (full historical / season-to-date /
trailing four weeks). Next week's feature weights come from a shrinkage blend of
those, so a factor whose signal shifts loses weight automatically. Each week
gets a plain-English review in `reports/`.

### Does it actually beat the market?

Walk-forward on 2023-2025, training only on strictly earlier games:

| Position | Rank corr | Vegas-only | Edge | MAE | Vegas-only MAE | Top-5 hit rate | Vegas-only |
|---|---|---|---|---|---|---|---|
| D/ST | **+0.330** | +0.318 | **+0.012** | **4.32** | 4.35 | **66%** | 62% |
| Kicker | **+0.170** | +0.120 | **+0.050** | **3.72** | 3.73 | **57%** | 50% |

Reproduce with `streamer backtest --seasons 2023-2025`. Ridge is the selected
estimator for both positions; gradient boosting was evaluated on the same
walk-forward split and fails the baseline gate (D/ST +0.262, Kicker +0.041).

Two honest caveats. The D/ST rank-correlation edge is real but small — opponent
implied total is genuinely most of the available signal, and the practical gain
shows up more clearly in the top-5 hit rate (66% vs 62%). And weekly *kicker*
scoring is close to irreducible noise; the relative edge is large, but nobody
should expect precision. Read the floor/ceiling band, not the point estimate.

---

## Setup

Requires **Python 3.11+**.

```bash
git clone <your-repo-url> streamer && cd streamer

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Optional: live odds. Without a key the tool falls back automatically.
cp .env.example .env      # then paste your key from https://the-odds-api.com/
```

Verify:

```bash
pytest                                             # test suite (~2s)
streamer backtest --seasons 2023-2025 --estimator ridge   # ~1 min
streamer rank --week 1
```

The first data pull downloads five seasons of play-by-play (~65 MB) into
`data/raw/`. It is cached, so everything after the first run is fast, and
backtests are reproducible. `STREAMER_REFRESH=1` forces a re-fetch.

Dropping `--estimator ridge` also backtests the gradient-boosted candidate,
which takes about seven minutes — worth doing once, or after changing the
feature set, to confirm the choice still holds.

---

## Weekly workflow

**Tuesday — score last week.**

```bash
streamer update --week 5          # score, re-fit, rewrite the ledger
streamer benchmark --week 5       # optional: head-to-head vs Subvertadown
```

Writes `reports/week_5_review.md` (what it got wrong, which weights moved and
why) and updates `results/history.parquet` and `results/factor_ledger.parquet`.

**Wednesday — rank the coming week.**

```bash
streamer rank --week 6 --publish  # rankings + docs/index.html
```

Wednesday is deliberate: waiver claims process Wednesday morning and lines have
settled by then.

### Where to paste CSVs

All optional — each one only overrides a fallback the tool already handles.

| File | When you'd use it | Columns |
|---|---|---|
| `data/lines_week_N.csv` | No API key, or you want your own book's numbers | `home_team,away_team,spread,total` |
| `data/weather_week_N.csv` | Wind matters for an outdoor game | `home_team,wind,temp` |
| `data/subvertadown_week_N.csv` | Benchmarking | `rank,team` (D/ST) or `rank,player` (K) |

`spread` is the **home team's sportsbook line** — type it the way the book
prints it (`-3.5` means the home team is favoured by 3.5). Team names accept
abbreviations, nicknames or full names: `KC`, `Chiefs` and
`Kansas City Chiefs` all work.

A one-file version carrying both positions also works:

```csv
position,rank,name
DST,1,Jaguars
DST,2,Chargers
K,1,Cameron Dicker
```

### Lines fallback chain

Tried in order, first success wins, recorded per game and flagged on the page:

1. **The Odds API** — needs `ODDS_API_KEY`
2. **`data/lines_week_N.csv`** — your manual entry
3. **nflverse schedule closing lines** — present for upcoming games, so even a
   key-less install gets real numbers
4. **Last cached pull** for that week

The tool never hard-fails on a missing key.

---

## Reading it on your phone

`streamer publish` renders a single self-contained page — plain HTML and CSS,
no framework, no JavaScript, dark-mode aware — to `docs/index.html`, with each
week archived at `docs/week_N.html`.

### One-time GitHub Pages setup

1. **Create the repo** on GitHub and push to it:
   ```bash
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin HEAD
   ```

2. **Enable Pages:** repo → **Settings** → **Pages** → under *Build and
   deployment* set **Source: Deploy from a branch**, then pick **the branch you
   pushed** and **Folder: `/docs`** → **Save**.

   The branch dropdown lists what actually exists in your repo, which may not be
   `main` — if you pushed a feature branch, select that one.

3. **Add the odds key** (optional): repo → **Settings** → **Secrets and
   variables** → **Actions** → **New repository secret** → name `ODDS_API_KEY`,
   value your key from [The Odds API](https://the-odds-api.com/). The free tier
   allows 500 requests a month; the weekly job uses about four.

4. Your page is live at `https://<you>.github.io/<repo>/`. Add it to your
   iPhone home screen (Share → Add to Home Screen) and it opens like an app.

**Private repos** need a paid plan for Pages. On a free plan, make the repo
public (**Settings → General → Danger Zone → Change visibility**) or the URL
will 404.

**The branch you serve from matters more than it looks.** GitHub runs
`on: schedule` workflows *only from the repository's default branch*, and it
does so silently — if the workflow lives on a non-default branch it simply
never fires, with no error anywhere. So make the branch carrying this project
your default (**Settings → Branches**, or rename it under the pencil icon),
and if you later switch defaults, move the workflow with it. The workflow
itself pushes to `${GITHUB_REF_NAME}`, so it works under any branch name.

### Automation

[`.github/workflows/weekly.yml`](.github/workflows/weekly.yml) runs at 11:00 UTC
(07:00 ET during EDT):

- **Tuesday** — `update` on the completed week, plus `benchmark` if you
  committed a Subvertadown CSV
- **Wednesday** — `rank` and `publish` the upcoming week, then commit the
  refreshed page

If the odds pull fails, it publishes anyway using the fallback chain and marks
the page as running on fallback lines. Run it by hand any time from the
**Actions** tab (*Run workflow*), optionally forcing a week.

Your Mac stays the manual-override path: everything runs locally through the
same CLI, and committing a lines/weather/Subvertadown CSV changes what the next
automated run produces.

---

## Commands

| Command | What it does |
|---|---|
| `streamer rank --week N` | Ranked D/ST and K tables with expected points, floor/ceiling, P(top-12), a one-line rationale each, and two-week hold flags. `--publish` also writes the page. |
| `streamer update --week N` | Scores that week's predictions, appends to history, re-fits, rewrites the ledger, writes `reports/week_N_review.md`. |
| `streamer benchmark --week N` | Head-to-head vs Subvertadown on rank correlation and top-5 hit rate; updates `reports/benchmark.md`. |
| `streamer backtest --seasons 2023-2025` | Walk-forward validation against the Vegas-only baseline. `--estimator ridge` skips the slower tree candidate, `--tune` sweeps the penalty, `--save` persists the winner to `results/model_selection.json`, `--strict` exits non-zero if a position fails to beat the baseline. |
| `streamer publish --week N` | Renders `docs/index.html` and `docs/week_N.html`. |

Global flags: `--offline` (cached data and manual CSVs only), `--season`,
`--verbose`, `--config`.

---

## Outputs

| Path | Contents |
|---|---|
| `docs/index.html` | The published page — current week |
| `docs/week_N.html` | Weekly archive |
| `reports/week_N_review.md` | Plain-English weekly review |
| `reports/benchmark.md` | Running head-to-head vs Subvertadown |
| `results/history.parquet` | Every scored prediction |
| `results/factor_ledger.parquet` | Per-factor correlations and weights, by week |
| `results/team_adjustments.parquet` | Bayesian per-team in-season adjustments |
| `results/predictions.parquet` | What was published, so `update` scores the real thing |

---

## Project layout

```
src/streamer/
  scoring.py          ESPN D/ST and K scoring; the tier ladder
  actuals.py          realised stat lines from play-by-play
  data/               nflverse loaders, odds fallback chain, weather, cache
  features/           team-week aggregates, leak-free rolling priors, builders
  models/             Vegas-anchored regressions, tier distribution, ledger
  backtest.py         walk-forward validation
  calibrate.py        the weekly self-calibration loop
  benchmark.py        Subvertadown head-to-head
  rankings.py         ranking, rationales, two-week candidates
  publish.py          static page rendering
  cli.py              command line
```

Design rationale, rejected approaches and the evidence behind every tuned
number are in **[DECISIONS.md](DECISIONS.md)**.

## Data & licence

Play-by-play, schedules and closing lines from
[nflverse](https://github.com/nflverse) via
[`nflreadpy`](https://pypi.org/project/nflreadpy/). Live odds optionally from
[The Odds API](https://the-odds-api.com/). MIT licensed. Not betting advice.
