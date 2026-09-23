# streamer

Weekly **D/ST and Kicker streaming rankings** for fantasy football, in the style
of Subvertadown: Vegas-anchored projections that re-calibrate themselves against
their own results every week, published to a phone-friendly page you can open
from anywhere.

Two leagues are configured out of the box — **ESPN (10-team)** and **Yahoo
(14-team)** — and they are ranked separately, because the same defense is worth
genuinely different amounts in each. The published page carries a switch
between them. Everything lives in [`config.yaml`](config.yaml).

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

### The two profiles

| | ESPN (10-team) | Yahoo (14-team) |
|---|---|---|
| Sack | 2.5 | 1 |
| Interception | 2.5 | 2 |
| Shutout | 6 | 10 |
| Points-allowed bands | 8 tiers, 46+ floor at -4 | 7 tiers, 35+ floor at -4 |
| Yards allowed | scored, 9 tiers | not scored |
| Fourth-down stop | — | 1 |
| Fumble lost by the unit | -2 | — |
| Missed FG | -1 | no penalty |
| Missed PAT | -0.5 | no penalty |
| 60+ yard FG | 6 | 5 |

Those differences are not cosmetic. A sack is worth 2.5x more in ESPN, which
pulls pass-rush matchups up its rankings; Yahoo's shutout pays 10 against
ESPN's 6, which rewards chasing the lowest implied totals harder. Kickers
reorder too — Yahoo penalises nothing, so a high-volume, lower-accuracy leg
ranks better there than in ESPN.

Each profile keeps its own trained models, history, factor ledger and benchmark
record under `results/<profile>/` and `reports/<profile>/`. They share only the
raw nflverse cache, which does not depend on scoring.

### Does it actually beat the market?

Walk-forward on 2023-2025, training only on strictly earlier games:

| Profile | Position | Rank corr | Vegas-only | Edge | MAE | Vegas MAE | Top-5 hit | Vegas |
|---|---|---|---|---|---|---|---|---|
| ESPN | D/ST | **+0.359** | +0.344 | **+0.015** | **6.19** | 6.23 | **64%** | 63% |
| ESPN | Kicker | **+0.171** | +0.122 | **+0.049** | **3.72** | 3.73 | **56%** | 49% |
| Yahoo | D/ST | **+0.333** | +0.318 | **+0.015** | **4.58** | 4.61 | **72%** | 71% |
| Yahoo | Kicker | **+0.178** | +0.128 | **+0.051** | **3.61** | 3.62 | **63%** | 58% |

Reproduce with `streamer backtest --seasons 2023-2025`. Ridge is the selected
estimator throughout; gradient boosting was evaluated on the same walk-forward
split and fails the baseline gate (D/ST +0.262, Kicker +0.041).

The absolute numbers are not comparable across profiles — ESPN pays 2.5 a sack
and adds a yards-allowed ladder, so its D/ST scores are simply bigger, and
Yahoo's higher top-5 rate reflects a 14-team startable bar against ESPN's 12.
The **edge over the Vegas-only baseline** is the comparable figure.

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
streamer update --week 5          # both leagues: score, re-fit, rewrite the ledger
streamer benchmark --week 5       # optional: head-to-head vs Subvertadown
```

Writes `reports/<profile>/week_5_review.md` (what it got wrong, which weights
moved and why) and updates each profile's `history.parquet` and
`factor_ledger.parquet`.

**Wednesday — rank the coming week.**

```bash
streamer rank --week 6 --publish  # both leagues + docs/index.html
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

Subvertadown publishes different rankings per scoring system. Name the file
`subvertadown_week_N_espn.csv` or `..._yahoo.csv` to give each profile its own,
or use the plain `subvertadown_week_N.csv` for both.

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

- **Tuesday** — `update` on the completed week for both leagues, plus
  `benchmark` if you committed a Subvertadown CSV
- **Wednesday** — `sync` both leagues (skipped for any league without
  secrets), then `rank` and `publish` the upcoming week and commit the
  refreshed page
- **Sunday, 07:00 ET** — publish once more before the early kickoffs, when
  the sportsbook player props are posted and the inactives are known

Both profiles run in every step, so each league keeps its own calibration
history and both appear on the one published page.

If the odds pull fails, it publishes anyway using the fallback chain and marks
the page as running on fallback lines. Run it by hand any time from the
**Actions** tab (*Run workflow*), optionally forcing a week.

Your Mac stays the manual-override path: everything runs locally through the
same CLI, and committing a lines/weather/Subvertadown CSV changes what the next
automated run produces.

---

## In-season: lineups, waivers, matchups

Once the season starts the page grows a **My team** panel per league, and four
commands work from a synced snapshot of your roster, your opponent's roster
and the free-agent pool:

| Command | What it does |
|---|---|
| `streamer sync --week N` | Pulls both leagues (`--profile` narrows) into `data/leagues/<profile>/week_N.json`. `--skip-missing` quietly skips a league whose credentials are absent. |
| `streamer lineup --week N` | The lineup that maximises **P(win) against this week's opponent**, with the swaps from what is currently set. |
| `streamer waivers --week N` | Ranked add/drop pairs, each scored by how much it raises your best lineup plus depth over what is left on the wire, with a one-line reason. |
| `streamer matchup --week N` | The head-to-head: your P(win), both sides' expected score and spread, and the players that swing it most. |
| `streamer yahoo-auth` | One-time browser authorisation that mints the Yahoo refresh token. |

All four read the snapshot, so `lineup`/`waivers`/`matchup` run offline and
instantly once `sync` has been done. The Wednesday workflow syncs both leagues
before it ranks, so the published page carries the panel automatically.

### The Vegas column

Where the sportsbooks post player props, the lineup table carries a second
number: the fantasy points those props imply. It is built from Pinnacle,
FanDuel and Caesars, weighted toward Pinnacle because its margin is thinnest.
A blank means no book posted that player, which is not the same as zero --
kickers, defences and deep bench players are rarely priced.

The panel also says what lineup the market's numbers would start and which
players the two projections disagree about most. Disagreement is the useful
part: the market sees beat-reporter news about a snap count days before it
reaches a box score, so a large gap is worth a look before you set the lineup.

Props are billed per market per event, so the pull is deliberately frugal:
only games with a startable player of yours, only games kicking off within 72
hours (books do not post Sunday's props on a Tuesday, so asking buys nothing),
and at most eight games. In practice the midweek runs cost a handful of
credits and the Sunday-morning run pays for real coverage -- roughly 330 a
month against the free tier's 500. The panel prints the balance the API
reports. Lower `odds.props.max_events` or `window_hours`, drop a market, or
set `odds.props.enabled` to `false` in config.yaml to spend less.

**Read the lineup on Sunday morning.** The page is published Tuesday and
Wednesday for waivers, and again at 07:00 ET Sunday. Only that last one has
the props posted and the inactives known, so it is the one to set a lineup
from.

### How players are projected

D/ST and K use the streaming rankings above. Every other position gets a
skill projection: a shrunk blend of trailing production and trailing
**opportunity-expected** points (nflverse `ff_opportunity`), scaled by this
week's Vegas implied total relative to the team's recent average. When the
platform publishes its own projection the two are averaged. Injuries scale the
mean and widen the spread; a bye or an OUT tag zeroes the week. Walk-forward
on 2021-2025 it ranks starters better than trailing average alone at every
position (DECISIONS.md has the numbers).

The lineup optimiser does not chase expected points. It draws 20,000 joint
samples of every player on both rosters, enumerates every valid lineup and
picks the one that wins the most draws — which is why it will start a
high-variance receiver when you are the underdog and the steady one when you
are not.

### Syncing your leagues

Credentials live in `.env` locally and in repository **Secrets** for the
workflow. Never commit them. Each league is independent; set up whichever you
have.

**ESPN** (cookie based, no app needed):

1. `ESPN_LEAGUE_ID` — the `leagueId=` number in any league URL.
2. `ESPN_S2` and `ESPN_SWID` — while logged in to fantasy.espn.com, open the
   browser's developer tools and copy the values of the `espn_s2` and `SWID`
   cookies (SWID includes the braces; `espn_s2` is long and contains `%`).
   - **Chrome**: DevTools → *Application* → *Cookies* → `espn.com`.
   - **Safari**: Settings → Advanced → *Show features for web developers*,
     then Cmd+Option+I → *Storage* → *Cookies* → `espn.com`.

   They last about a year; when a sync starts failing with a 401, refresh
   them.
3. `ESPN_TEAM_ID` — optional. The SWID normally identifies your team; set this
   (the `teamId=` in your team URL) only if the sync picks the wrong one.

**Yahoo, the working route: your browser session.** Yahoo no longer gives
out API access freely, so the league is read from the fantasy website with
your own logged-in session -- the same way ESPN works. In Safari (or any
browser), logged in to your league:

1. Open DevTools (Cmd+Option+I) -> **Storage** -> **Cookies** -> the
   `yahoo.com` entry, click a row, **Cmd+A**, **Cmd+C**. (Or: **Network** ->
   reload -> filter **Document** -> the page request -> Request Headers ->
   **Cookie**.)
2. Paste the whole thing into a `YAHOO_COOKIE` repository secret. Any of
   those formats works; it is normalised on read.
3. `YAHOO_LEAGUE_ID` as below. Your team and opponent are found
   automatically.

Know what you are storing: those cookies are your whole Yahoo sign-in, not
just fantasy -- the secret is encrypted and only your own workflows can read
it, and signing out of Yahoo invalidates it. They expire every few months; the
page will say so when a sync fails, and you re-copy them.

`streamer yahoo-probe --detail` describes what Yahoo's pages look like
(structure only, all text scrubbed) -- run it from the Actions tab with the
`yahoo-probe` job if Yahoo changes their layout and the sync starts failing.

**Yahoo, the official route** (only if Yahoo grants your app access):

> **Read this before spending time on it.** Since mid-2026 Yahoo no longer
> self-serve provisions the Fantasy Sports API. The "Fantasy Sports"
> permission was removed from the app creation form *for everyone* -- if you
> cannot find it, nothing is wrong with your account. Entitlement is now
> granted server-side against a specific Client ID after a manual review, and
> existing apps that used to work were de-provisioned too. OAuth still works
> throughout: tokens mint and refresh normally and then every fantasy call
> returns `additional_authorization_required` or a 403, which makes it look
> like a credential problem when it is an entitlement one. There is no
> published turnaround time.

1. Apply at <https://sports.yahoo.com/developer/access/>, **quoting the Client
   ID of the app you intend to use**, so the grant attaches to that app rather
   than to nothing. The review wants the product (a personal lineup/waiver
   tool), the data (your own leagues' rosters, matchups and free agents) and
   the user base (personal, single-league). Incomplete applications are closed
   without reply. Read access is all that is offered, and all this tool uses.
2. Complete the confirmation step at
   <https://sports.yahoo.com/developer/application-confirmation/> with the
   email address on your Yahoo developer account.
3. Create the app at <https://developer.yahoo.com/apps/> if you have not
   already: any name, **Redirect URI** `https://localhost:8080` (Yahoo insists
   on an https URI, but the tool authorises out-of-band so it is never
   visited), and **Confidential Client** so you are issued a client secret.
   Put its Client ID and Secret in `.env` as `YAHOO_CLIENT_ID` /
   `YAHOO_CLIENT_SECRET`. Do not go looking for API Permissions to tick.
4. `YAHOO_LEAGUE_ID` -- the number at the end of your league URL
   (`.../f1/123456` -> `123456`).
5. Once the grant lands, run `streamer yahoo-auth`. It authorises, then makes
   one real fantasy call and tells you whether it worked, so you find out
   immediately rather than at the next scheduled run. Put the refresh token it
   prints in `.env` and in the `YAHOO_REFRESH_TOKEN` secret.

The Yahoo *scoring profile* does not depend on any of this: its D/ST and K
rankings come from nflverse and the betting markets. Only the My-team panel
for that league needs API access.

Then:

```bash
streamer sync --week 6                 # both leagues
streamer lineup --week 6 --profile espn
streamer waivers --week 6 --profile yahoo
streamer matchup --week 6
```

---

## Commands

| Command | What it does |
|---|---|
| `streamer rank --week N` | Ranked D/ST and K tables with expected points, floor/ceiling, P(top-12), a one-line rationale each, and two-week hold flags. `--publish` also writes the page. |
| `streamer update --week N` | Scores that week's predictions, appends to history, re-fits, rewrites the ledger, writes `reports/week_N_review.md`. |
| `streamer benchmark --week N` | Head-to-head vs Subvertadown on rank correlation and top-5 hit rate; updates `reports/benchmark.md`. |
| `streamer backtest --seasons 2023-2025` | Walk-forward validation against the Vegas-only baseline. `--estimator ridge` skips the slower tree candidate, `--tune` sweeps the penalty, `--save` persists the winner to `results/model_selection.json`, `--strict` exits non-zero if a position fails to beat the baseline. |
| `streamer publish --week N` | Renders `docs/index.html` and `docs/week_N.html`, including the My-team panel when a league snapshot exists. |
| `streamer sync` / `lineup` / `waivers` / `matchup` / `yahoo-auth` | The in-season suite, above. |

Global flags: `--profile` (`espn`, `yahoo`, or `all` — the default),
`--offline` (cached data and manual CSVs only), `--season`, `--verbose`,
`--config`.

```bash
streamer rank --week 6                  # both leagues
streamer rank --week 6 --profile yahoo  # just the Yahoo one
```

---

## Outputs

| Path | Contents |
|---|---|
| `docs/index.html` | The published page — current week, both leagues |
| `docs/week_N.html` | Weekly archive |
| `reports/<profile>/week_N_review.md` | Plain-English weekly review |
| `reports/<profile>/benchmark.md` | Running head-to-head vs Subvertadown |
| `results/<profile>/history.parquet` | Every scored prediction |
| `results/<profile>/factor_ledger.parquet` | Per-factor correlations and weights, by week |
| `results/<profile>/team_adjustments.parquet` | Bayesian per-team in-season adjustments |
| `results/<profile>/predictions.parquet` | What was published, so `update` scores the real thing |
| `data/leagues/<profile>/week_N.json` | Synced league snapshot: rosters, matchup, free agents, projections |

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
  league/             ESPN and Yahoo adapters -> one normalised LeagueSnapshot
  roster/             skill projections, P(win) lineup optimiser, waivers, panel
```

Design rationale, rejected approaches and the evidence behind every tuned
number are in **[DECISIONS.md](DECISIONS.md)**.

## Data & licence

Play-by-play, schedules and closing lines from
[nflverse](https://github.com/nflverse) via
[`nflreadpy`](https://pypi.org/project/nflreadpy/). Live odds optionally from
[The Odds API](https://the-odds-api.com/). MIT licensed. Not betting advice.
