# streamer

A fantasy football manager's tool for two leagues (ESPN 10-team, Yahoo 14-team),
published to a phone-friendly page that rebuilds itself through the week. Every
number is simulated against your real roster, your opponent and the rest of the
league, and the question it answers is always the same: **what move raises your
chance of winning the week, making the playoffs and winning the title?**

It began as a D/ST and kicker streaming ranker in the style of Subvertadown
(Vegas-anchored, self-recalibrating) and that core is still here. Around it now
sit skill-player projections, a P(win) lineup optimiser, waiver and trade
pricing in title odds, and a rest-of-season simulator. Everything is configured
in [`config.yaml`](config.yaml); every non-obvious choice, with the evidence for
it, is in [DECISIONS.md](DECISIONS.md).

Two rules hold throughout. A change ships only if it beats the current model on
seasons it was not fitted on: better overall, and worse in at most one season.
And no credential, cookie or key is ever pasted into a chat, an issue or a log:
they live in GitHub Actions secrets or a gitignored `.env`.

---

## What you get each week

A team picker at the top of the page chooses whose view to read (any team in the
league); a switch picks the league. Under it, tabs:

| Tab | The decision it answers |
|---|---|
| **Hub** | Every move you can make, ranked by points of title odds: lineup changes, streams, waiver claims, trades, multi-move plans. Breaking news for everyone in the league sits at the bottom. |
| **Lineup** | Who to start. During games, a **live** card: both sides' points so far from ESPN's box scores, and P(win) updated every minute while the page is open. The lineup that maximises P(win) against this week's opponent, with the changes from what you have set, the range of each player, where each projection came from, and **"If a questionable starter is ruled out"**: when his inactives come out and who replaces him. A lab lets you try any lineup. |
| **Roster** | What each of your players is worth for the rest of the season, the title odds you would lose without him, and what other managers think he is worth. |
| **Waivers** | Free agents who raise this week's P(win) (including ones held on the bench as cover), the best add/drop pairs for the rest of the season, upside stashes, drop watch, and a browser of the free agents at each position. On Yahoo, which lets you drop a bench player after his game starts, a free agent to grab before each kickoff window still to come. |
| **Trades** | Offers ranked by chance of a yes x title odds gained, plus an evaluator for any trade you are offered or considering. |
| **D/ST & K** | The streaming rankings (below). |
| **Season** | Playoff and title odds for every team, the must-win weeks, and bye-week and schedule effects. |
| **Model** | How the projections have been graded against the platforms, the betting market and FantasyPros. |

The full detail (trade search, lineup lab) is built for your team and for
`roster.views.full` in the config; other teams get a lighter view with fewer
simulations, so the whole league fits in one refresh.

The page is plain HTML and CSS with a little script on top (remembered league
and tab, search, filters, "12 min ago" times with a warning when stale).
Add it to your home screen and it opens full-screen like an app.

---

## How it decides

Short versions; each links to the longer entry in DECISIONS.md.

**Projections.** Skill players: opportunity (nflverse expected points from
targets, carries and field position) x efficiency, scaled by this week's Vegas
implied total, then blended with the platform's projection, the betting
market's player props, the FantasyPros consensus and Sleeper, with weights refit
as the season goes. Games a player was on the field for but had no stat line
count as zero. A starter ruled out passes part of his work to the next man up;
a teammate returning takes it back. See "How a player's outlook can move" and
"Games on the field with no stat line".

**Lineups** maximise P(win), not points. Twenty thousand joint outcomes per
player on both rosters, correlated as measured (quarterback with his
receivers, against the defence facing him), skewed rather than bell-shaped;
injury tags are a coin flip. A questionable starter whose replacement can still
come in (a bench player or a free agent in a later game, with room on your
roster) costs little when he sits. See "The lineup objective is P(win)" and
"Late swaps".

**Season value** comes from simulating every player's rest of season (level,
spread, absences, byes, role changes), then playing out the schedule and the
playoffs in each of thousands of seasons. A move is priced by re-running only
the changed team on the same simulated futures, so the difference is the move.
Replacement level comes from the week-by-week wire: free agents on bye fill
nothing, and rivals ahead of you in waiver order take the best fill-ins.
Absence odds rise with games already missed this season or last. See "Winning
the championship", "Bye weeks thin the wire", "A record of missed games".

**Trades** are priced in title odds for you and judged for the other side the
way the market values players. **Waiver** drops are tried cheapest to your title
first. See "Trades: what helps you, and what he will accept".

**D/ST and K** use the model below.

---

## D/ST and kicker streaming

**Vegas is the backbone.** Every projection starts from a lightly regularised
regression on the betting market (implied team total, game total, spread, home
field) and then applies adjustments from everything else, heavily regularised
so they can only move the number as far as their measured signal justifies.

**Kickers** are projected from team implied total, the team's field-goal
generation profile, the kicker's own distance-bucket accuracy regressed toward
the league mean, dome and wind, and the opponent's red-zone defence. **D/ST**
splits into points allowed (a probability distribution over the scoring tiers,
not a point estimate) and the big-play component (sacks, takeaways, scores).
Output is expected points plus P(top-12).

Kicker lines and the opposing quarterback's interception line are logged beside
our projection every week and graded once there are enough games; neither moves
a projection until the grade says it helps.

**It learns every week.** `streamer update` scores what it predicted, re-fits
with the current season weighted 2x and rewrites a factor-correlation ledger
across three windows (full history, season to date, last four weeks). Next
week's weights come from a shrinkage blend, so a factor whose signal fades
loses weight on its own. Each week gets a plain-English review in `reports/`.

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

The differences are not cosmetic: a sack is worth 2.5x more in ESPN, which
lifts pass-rush matchups, and Yahoo's shutout pays 10, which rewards chasing the
lowest implied totals. Each profile keeps its own models, history, ledger and
benchmark under `results/<profile>/` and `reports/<profile>/`; they share only
the raw nflverse cache.

### Does it beat the market?

Walk-forward on 2023-2025, training only on strictly earlier games:

| Profile | Position | Rank corr | Vegas-only | Edge | MAE | Vegas MAE | Top-5 hit | Vegas |
|---|---|---|---|---|---|---|---|---|
| ESPN | D/ST | **+0.359** | +0.344 | **+0.015** | **6.19** | 6.23 | **64%** | 63% |
| ESPN | Kicker | **+0.171** | +0.122 | **+0.049** | **3.72** | 3.73 | **56%** | 49% |
| Yahoo | D/ST | **+0.333** | +0.318 | **+0.015** | **4.58** | 4.61 | **72%** | 71% |
| Yahoo | Kicker | **+0.178** | +0.128 | **+0.051** | **3.61** | 3.62 | **63%** | 58% |

Reproduce with `streamer backtest --seasons 2023-2025`. The edge over the
Vegas-only baseline is the comparable figure across profiles. Two honest
caveats: the D/ST edge is real but small (the opponent's implied total is most
of the available signal), and weekly kicker scoring is close to irreducible
noise, so read the floor/ceiling band, not the point estimate.

---

## Setup

Requires **Python 3.11+**.

```bash
git clone <your-repo-url> streamer && cd streamer

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

cp .env.example .env      # optional keys, never committed
```

Verify:

```bash
pytest                                             # the test suite, a few minutes
streamer backtest --seasons 2023-2025 --estimator ridge   # ~1 min
streamer rank --week 1
```

The first data pull downloads five seasons of play-by-play (~65 MB) into
`data/raw/` (gitignored; `STREAMER_REFRESH=1` re-fetches).

### Secrets

Names only; values go in repository **Secrets** (Settings -> Secrets and
variables -> Actions) and, locally, in `.env`. Everything is optional: a league
or feed without its secret is skipped.

| Secret | Used for |
|---|---|
| `ODDS_API_KEY` | Game lines and player props ([The Odds API](https://the-odds-api.com/), 500 free credits a month) |
| `SPORTSGAMEODDS_API_KEY` | A second props feed (2,500 games a month free) |
| `FANTASYPROS_API_KEY` | Consensus projections and rankings. Their data stays on the runner and is never committed or shown raw: only analysis, credited. |
| `ESPN_LEAGUE_ID`, `ESPN_S2`, `ESPN_SWID`, `ESPN_TEAM_ID` | ESPN league sync |
| `YAHOO_LEAGUE_ID`, `YAHOO_COOKIE`, `YAHOO_TEAM_ID` | Yahoo league sync through your browser session |
| `YAHOO_CLIENT_ID`, `YAHOO_CLIENT_SECRET`, `YAHOO_REFRESH_TOKEN` | Yahoo's official API, if it is ever granted |

Workflow logs are public: probes print aggregates only, never a name, price or
key.

---

## Publishing and automation

### One-time GitHub Pages setup

1. **Create the repo** on GitHub and push to it.
2. **Enable Pages:** Settings -> Pages -> *Build and deployment* -> **Source:
   GitHub Actions**. The workflow deploys the page itself and keeps the files
   that change on every run out of git, so the repository stops growing with
   each refresh. (*Deploy from a branch*, `/docs`, still works: the workflow
   reads the setting and, in that mode, commits the page as it always did.)
3. Your page is live at `https://<you>.github.io/<repo>/`.

Private repos need a paid plan for Pages; on a free plan make the repo public
or the URL 404s.

**The default branch matters.** GitHub runs `on: schedule` workflows only from
the repository's default branch, and silently: on any other branch the cron
never fires and nothing reports an error. Make the branch carrying this project
the default.

### The schedule

[`.github/workflows/weekly.yml`](.github/workflows/weekly.yml), in UTC:

| When | Job | Spends Odds API credits? |
|---|---|---|
| Tuesday 07:17 | `both`: score last week, recalibrate, rank and publish the next | yes |
| Wednesday 07:17 | `publish`: after waivers process | yes |
| Sunday 08:17 | `publish`: props posted, injury designations in | yes |
| Daily 00:41, 02:41, 04:41, 12:41-22:41 every two hours | `refresh` | no |
| Sunday 13:11-21:11 hourly | `refresh`, as inactives come out | no |
| Thursday, Sunday, Monday 22:55 and 23:55 | `refresh`, right after night-game inactives (6:55 PM ET either side of the clock change) | no |

GitHub starts scheduled runs when it has capacity, so they often land hours
after the scheduled time; the "synced" time on each panel says exactly when the
page was last built.

A `refresh` re-syncs both leagues and rebuilds the page from the lines and props
earlier runs bought, in a few minutes. On game days it reuses the trade and
add/drop work from the last full build. You can run one yourself any time: each panel has a "refresh" link
that opens the workflow (*Run workflow*, job `refresh`). Other manual jobs:
`both`, `update`, `publish`, and the probes `yahoo-probe`, `fantasypros-probe`,
`sgo-probe`, `sleeper-probe`, `news-probe`, plus `dstk-blend`, `wind-fit` and
`sleeper-fit` for grading experiments.

Props are billed per market per event; a Sunday `publish` buys as many games as
the month can afford, games with your players first. If the odds pull fails, the
page is published anyway on the lines fallback chain and says so.

What is committed after a run: `results/`, `reports/`, `data/leagues/`, and the
week's archive page on publish runs. The page itself, its data and the team views are
deployed, not committed.

---

## Running it locally

```bash
streamer rank --week 6 --publish   # D/ST and K rankings for both leagues + the page
streamer sync --week 6             # pull both leagues (--profile narrows)
streamer lineup --week 6 --profile espn
streamer waivers --week 6 --profile yahoo
streamer matchup --week 6
```

| Command | What it does |
|---|---|
| `streamer rank --week N` | Ranked D/ST and K tables with expected points, floor/ceiling, P(top-12) and a one-line rationale each. `--publish` also writes the page. |
| `streamer update --week N` | Scores that week's predictions, appends to history, re-fits, rewrites the ledger, writes `reports/<profile>/week_N_review.md`. |
| `streamer benchmark --week N` | Head-to-head vs Subvertadown on rank correlation and top-5 hit rate. |
| `streamer backtest --seasons 2023-2025` | Walk-forward validation against the Vegas-only baseline (`--estimator ridge`, `--tune`, `--save`, `--strict`). |
| `streamer publish --week N` | Renders the page into `docs/`. |
| `streamer sync --week N` | Pulls both leagues into `data/leagues/<profile>/week_N.json` (`--skip-missing` skips a league without credentials). |
| `streamer lineup` / `waivers` / `matchup` | The in-season suite from a synced snapshot; works offline. |
| `streamer yahoo-auth`, `yahoo-probe`, `fantasypros-probe`, `sgo-probe`, `sleeper-probe`, `news-probe`, `dstk-blend` | Setup and diagnostics. |

Global flags: `--profile` (`espn`, `yahoo`, or `all`, the default), `--offline`
(cached data and manual CSVs only), `--season`, `--verbose`, `--config`.

Everything also runs in the workflow; committing a CSV changes what the next
automated run produces.

### Optional CSVs

Each overrides a fallback the tool already handles:

| File | When you'd use it | Columns |
|---|---|---|
| `data/lines_week_N.csv` | No API key, or your own book's numbers | `home_team,away_team,spread,total` |
| `data/weather_week_N.csv` | Wind matters for an outdoor game | `home_team,wind,temp` |
| `data/subvertadown_week_N.csv` | Benchmarking (`_espn` / `_yahoo` suffix for a profile of its own) | `rank,team` (D/ST) or `rank,player` (K) |

`spread` is the home team's sportsbook line as the book prints it. Team names
accept abbreviations, nicknames or full names.

**Lines fallback chain**, first success wins and the source is flagged on the
page: The Odds API, then `data/lines_week_N.csv`, then nflverse schedule closing
lines (present for upcoming games, so a key-less install still gets real
numbers), then the last cached pull. The tool never hard-fails on a missing key.

---

## Syncing your leagues

Credentials live in `.env` locally and in repository Secrets for the workflow.
Never commit them. Each league is independent.

**ESPN** (cookie based, no app needed). `ESPN_LEAGUE_ID` is the `leagueId=`
number in any league URL. `ESPN_S2` and `ESPN_SWID` are the values of the
`espn_s2` and `SWID` cookies while logged in to fantasy.espn.com (SWID includes
the braces; `espn_s2` is long and contains `%`): DevTools -> Application ->
Cookies -> `espn.com` in Chrome, or Settings -> Advanced -> *Show features for
web developers*, then Cmd+Option+I -> Storage -> Cookies in Safari. They last
about a year; refresh them when a sync fails with a 401. `ESPN_TEAM_ID` is
optional, only if the sync picks the wrong team.

**Yahoo, the working route: your browser session.** Yahoo no longer gives out
API access freely, so the league is read from the fantasy website with your own
logged-in session:

1. In DevTools -> Storage -> Cookies -> the `yahoo.com` entry, select all and
   copy (or Network -> reload -> a page request -> Request Headers -> Cookie).
2. Paste the whole thing into a `YAHOO_COOKIE` repository secret; any of those
   formats works.
3. Set `YAHOO_LEAGUE_ID` (the number at the end of your league URL). Your team
   and opponent are found automatically.

Each sync reads the league and settings pages, every team's roster, your
matchup and up to six pages (the top 150) of offensive free agents, plus
kickers and defences: roughly twenty-five page loads, a short pause apart. The
first sync of a season also reads the rest of the schedule, once.

Know what you are storing: those cookies are your whole Yahoo sign-in, not just
fantasy. The secret is encrypted and only your own workflows can read it, and
signing out of Yahoo invalidates it. They expire every few months; the page says
so when a sync fails, and you re-copy them. If Yahoo changes its layout, run the
`yahoo-probe` job: it describes the pages (structure only, text scrubbed).

**Yahoo, the official route** (only if Yahoo grants your app access). Since
mid-2026 Yahoo no longer self-provisions the Fantasy Sports API: entitlement is
granted server-side against a specific Client ID after a manual review, and
tokens that mint and refresh normally still get `additional_authorization_required`
or a 403 on fantasy calls, which looks like a credential problem and is not.
Apply at <https://sports.yahoo.com/developer/access/> quoting the Client ID,
confirm at <https://sports.yahoo.com/developer/application-confirmation/>, create
the app at <https://developer.yahoo.com/apps/> (Redirect URI
`https://localhost:8080`, Confidential Client), set `YAHOO_CLIENT_ID`,
`YAHOO_CLIENT_SECRET` and `YAHOO_LEAGUE_ID`, and once the grant lands run
`streamer yahoo-auth`, which authorises, makes one real fantasy call to confirm,
and prints the refresh token for `YAHOO_REFRESH_TOKEN`.

The Yahoo *scoring profile* does not depend on any of this: its D/ST and K
rankings come from nflverse and the betting markets. Only the league panel needs
a sync.

---

## Outputs

| Path | Contents |
|---|---|
| `docs/index.html`, `docs/latest.json`, `docs/views/` | The published page, its data and every team's view (deployed by the workflow, not committed) |
| `docs/week_N.html` | Weekly archive |
| `reports/<profile>/week_N_review.md` | Plain-English weekly review |
| `reports/<profile>/benchmark.md` | Running head-to-head vs Subvertadown |
| `results/<profile>/history.parquet` | Every scored prediction |
| `results/<profile>/factor_ledger.parquet` | Per-factor correlations and weights, by week |
| `results/<profile>/predictions.parquet` | What was published, so `update` scores the real thing |
| `results/<profile>/skill_log.parquet`, `odds_log.parquet`, `unit_log.parquet` | Skill-player projections, title odds and kicker/D/ST lines as published, for grading |
| `data/leagues/<profile>/week_N.json` | Synced league snapshot: rosters, matchup, free agents, projections |

---

## Project layout

```
src/streamer/
  scoring.py, actuals.py   ESPN/Yahoo D/ST and K scoring; realised stat lines
  data/                    nflverse loaders, odds fallback chain, weather, cache
  features/, models/       team-week features; Vegas-anchored regressions, tier distribution, ledger
  backtest.py, calibrate.py, benchmark.py   validation, the weekly loop, Subvertadown
  rankings.py              D/ST and K ranking and rationales
  publish.py               page rendering and per-team views
  league/                  ESPN and Yahoo adapters -> one normalised LeagueSnapshot
  roster/                  skill projections (projections, idle, depth, snaps), the P(win)
                           lineup optimiser (lineup, matchup), waivers and pickups
                           (waivers, add_drop, waiver_plans), trades (trades, trade_eval),
                           the season simulator (futures, season, title_moves), page
scripts/                   the fitters: each refits one part of the model and grades it held out
tests/
.github/workflows/weekly.yml   the schedule above
```

Design rationale, rejected approaches and the evidence behind every tuned
number are in **[DECISIONS.md](DECISIONS.md)**.

## Data & licence

Play-by-play, schedules and closing lines from
[nflverse](https://github.com/nflverse) via
[`nflreadpy`](https://pypi.org/project/nflreadpy/). Live odds optionally from
[The Odds API](https://the-odds-api.com/). MIT licensed. Not betting advice.
