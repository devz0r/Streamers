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
no framework, dark-mode aware — to `docs/index.html`, with each week archived
at `docs/week_N.html`. A switch at the top picks the league; under it, a strip
of tabs picks the section: **Hub** (every move ranked by title odds),
**Lineup**, **Roster** (every player's rest-of-season value), **Waivers**,
**Trades** (with the trade evaluator), **D/ST & K** (the streaming
rankings), **Season** (playoff and title odds) and **Model** (how the
projections have graded). Above the tabs, four tiles: this week's chance to
win, the title and playoff odds, and the record -- each opens the tab that
explains it. On a phone the tabs are a bottom bar (Hub, Lineup, Waivers,
Trades, and More for the rest).

Both switches are CSS-only, so the page works with JavaScript off. A small
script adds conveniences on top: it remembers the league and tab you were
on, and an address like `index.html#espn-trades` opens that tab; a search box
filters the open tab (press `/`), and position chips filter Roster, Waivers
and Lineup; long reasoning is folded to three lines with "Show more"; the
update time reads "12 min ago" and warns when it is over six hours old.
Players show their ESPN photo and D/STs their team logo when ESPN's image
server answers (the run checks first). Add it to your home screen and it
opens full-screen like an app. The lineup editor and trade evaluator are the
other scripts.

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

[`.github/workflows/weekly.yml`](.github/workflows/weekly.yml) is scheduled for
07:17 UTC Tuesday and Wednesday and 08:17 UTC Sunday (03:17 / 04:17 ET). That
looks absurdly early on purpose: GitHub starts scheduled runs when it has
capacity, and this workflow's runs have typically landed 3.5-4.5 hours after
the scheduled time. Scheduled early and off the hour, they still arrive before
you look. The "synced" time at the top of each My-team panel says exactly when
the page was last refreshed.

- **Tuesday** — `update` on the completed week for both leagues, plus
  `benchmark` if you committed a Subvertadown CSV
- **Wednesday** — `sync` both leagues (skipped for any league without
  secrets), then `rank` and `publish` the upcoming week and commit the
  refreshed page
- **Sunday, early morning** — publish once more before the early kickoffs,
  when the sportsbook player props are posted and Friday's injury
  designations are in (official inactives come 90 minutes before kickoff,
  so no morning run can know them -- check those yourself)

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

Once the season starts each league grows its team tabs (Hub, Lineup, Roster,
Waivers, Trades, Season), and four
commands work from a synced snapshot of your roster, your opponent's roster
and the free-agent pool:

| Command | What it does |
|---|---|
| `streamer sync --week N` | Pulls both leagues (`--profile` narrows) into `data/leagues/<profile>/week_N.json`. `--skip-missing` quietly skips a league whose credentials are absent. |
| `streamer lineup --week N` | The lineup that maximises **P(win) against this week's opponent**, with the swaps from what is currently set. |
| `streamer waivers --week N` | Ranked add/drop pairs, each scored by how much it raises your best lineup, depth over what is left on the wire, and rest-of-season upside, with a one-line reason. |
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

The market also feeds the projection: where a book priced a player it is
blended in (40% to start, less with fewer books), and so is the FantasyPros
consensus projection (30% to start). Both weights are refit together each
run from how ours, the market's and the consensus's numbers actually did
this season. Under the lineup, "Where this week's projections come from"
lists each source, its weight and how many of your players it covered --
and says so when a source is missing, as props are until the Sunday run buys
them. The same numbers feed the lineup, P(win), pickups and this week's part
of every title-odds number (waiver adds and drops, trades, the roster).

The panel also says what lineup the market's numbers would start and which
players the two projections disagree about most. Disagreement is the useful
part: the market sees beat-reporter news about a snap count days before it
reaches a box score, so a large gap is worth a look before you set the lineup.

Props are billed per market per event (7 credits a game here), and the
free tier's 500 credits reset on the 1st. A Sunday run buys as many games
as the month can afford: the credits left, less what the midweek runs still
to come will need, over the Sundays left (the event list is free and
reports the balance). That is the whole slate in most months. Games with
your players go first, then the rest (your opponent's players and the free
agents who might start for you). Midweek runs buy at most two games inside
72 hours, usually Thursday night's; a refresh buys nothing and reuses what
was bought. The panel prints the balance the API reports. Set
`odds.props.max_events` to cap a run.

A second feed, SportsGameOdds, is read when the `SPORTSGAMEODDS_API_KEY`
secret is set. It bills each game once, whatever its markets and books
(2,500 games a month free), so it prices the whole slate on any day. Its
books join the same consensus; a price both feeds have counts once, and
pick'em sites are left out. Run the `sgo-probe` job once to see which books
and stat names the free plan carries (it prints no prices, names or key).

**Refreshing by hand.** Each panel has a "refresh" link. It opens the
workflow on GitHub; tap *Run workflow* and leave the job on `refresh`. In
about two minutes the page is rebuilt from a fresh sync of both leagues --
your lineup, your opponent's, the wire -- without spending any Odds API
credits (it reuses the lines and props earlier runs bought).

**Games already played** are locked: a starter whose game has kicked off
keeps his spot, and once the game is final his actual score, not his
projection, goes into P(win).

**Try a lineup.** Under the recommended lineup, a table compares your lineup
as set, the best-P(win) lineup and the most-points lineup, and *Try a lineup*
lets you pick any starters and see P(win) and the projection move.

**Game-time lottery tickets (Yahoo).** Yahoo lets you drop a bench player
after his game starts, so the Yahoo panel suggests a free agent to grab
before each kickoff window still to come -- keep him if his game makes him
worth keeping, otherwise drop him for the next window -- ranked by the chance
he has a top-14 week at his position or inherits a lead back's job, and
names your cheapest roster spots to open for it.

**Read the lineup on Sunday morning.** The page is published Tuesday and
Wednesday for waivers, and again early Sunday. Only that last one has the
props posted and the week's injury designations in, so it is the one to set a
lineup from -- then glance at the official inactives, which come out 90
minutes before kickoff.

### How players are projected

D/ST and K use the streaming rankings above. Every other position gets a
skill projection built as **opportunity x efficiency**: how much work he is
getting (nflverse `ff_opportunity` expected points -- targets, carries, where
on the field), which is sticky and so reacts within a game or two, times how
well he converts it, which mostly regresses and so moves slowly. That is
then scaled by this week's Vegas implied total relative to the team's recent
average, and averaged with the platform's own projection when it publishes
one. A scoring streak on the same workload barely moves him; a jump in
workload does. Walk-forward on 2022-2025 it picks the higher scorer of two
players more often than the previous formula in every season (DECISIONS.md
has the numbers).

When a starter is ruled out (IR, OUT, suspended, or the platform projecting
him for zero), the teammate at his position with the most opportunity is
projected to inherit part of the gap: about a third of it for a running
back. That happens before the backup has played a snap in the role -- the
moment a waiver claim is worth making. Injuries scale the mean by the chance
to play; a bye or an OUT tag zeroes the week.

### How the lineup is chosen

The lineup optimiser maximises the chance of **winning the week**, not
expected points. It draws 20,000 joint outcomes for every player on both
rosters, enumerates every valid lineup and picks the one that beats your
opponent's in the most draws. What makes that more than a slogan is how the
draws are made:

- **Correlated**, as measured 2021-2025: a quarterback and his WR1 (+0.34),
  WR2 (+0.29) and tight end (+0.26); a quarterback and the defence facing
  him (-0.44); the two quarterbacks in the same game (+0.18). A stack widens
  your range, which an underdog wants. Starting the receiver who catches
  your *opponent's* quarterback's passes narrows the range of the
  difference, which a favourite wants.
- **Skewed**: a 6-point projection has a floor near zero and a long right
  tail, so the draws follow the measured shape, not a bell curve.
- **Injury tags are a coin flip**: a Questionable player either plays
  (his full range) or scores zero.

The best candidates are re-scored on a fresh set of draws, and a lineup that
gives up projected points is only recommended when it wins more by a margin
the simulation can resolve (at least half a percentage point). Most weeks
that means the highest-projected lineup: favourite-or-underdog effects are
real but usually smaller than a point of projection. When the recommendation
does differ, the panel says why ("you are the underdog, and he stacks with
your quarterback ..., P(win) 38% -> 40%"). The Range column is the middle 70%
of each player's simulated outcomes.

One thing tested and *not* used: labelling players "boom-or-bust" or
"high-floor" from touchdown dependence, depth of target or their own past
volatility. On 2024-25 none of it predicted weekly spread beyond position
and projection level.

### How the projections are doing

Each league panel ends with a scorecard: our projections, the platform's,
the betting market's and (with a key) FantasyPros', each graded every week
on the players who played -- start/sit accuracy and average miss -- with
ours on the same players beside it. It builds up over the season.

### Championship hub

The top of each My-team panel ranks every move you can make -- lineup
changes, D/ST and K streams, waiver claims and blocks, multi-move plans,
trades -- by how many points of title odds it adds, with a link to the
section that explains each. The detail sections fold away below it.

### Must-win weeks and upside plays

Under the season outlook, "Must-win weeks" shows what each remaining game is
worth to your playoff odds (win against loss, in every simulated season)
and the record that gets in. Waiver moves always price next-man-up backs,
rookies and rising roles; title odds decide how much their upside is worth,
which is more the further behind you are.

### Waiver plans

Below the single waiver moves, the panel lists plans of two or three moves
made together, priced in title odds against the best single move -- "make
the plan" when it wins by more than the simulation noise, "close call" when
by less -- with the order to put the claims in. See
DECISIONS.md, "Waiver plans".

### Trades

When the league's structure has been read, the My-team panel lists trades
in three views: **top** (chance of a yes x title odds gained), **best for
you** (title odds gained) and **most likely yes**. Your side is priced in
title odds; his is judged the way the market values players -- the
platform projection, name value from last season, this season so far,
injuries -- in proportions measured from Yahoo's rostership. See
DECISIONS.md, "Trades: what helps you, and what he will accept".

Below the lists, **Evaluate a trade** takes any players with any team --
an offer you received, or one you are thinking of making -- and shows,
as you tick them, both teams' title and playoff odds before and after,
the weekly points each gains or loses, forced drops (you can pick yours),
how he would see it and his chance of a yes. It runs in the browser on
data from the same simulation, and says how close it came to the full
simulation on the trades priced that run. See DECISIONS.md, "Evaluate any
trade, on the page".

### Pickups for this week

On the Waivers tab, under "Pickups for this matchup" (and in one line under the
P(win) on the Lineup tab), every free agent who could start
for you this week is added to your roster, your lineup rebuilt around him,
and scored against your opponent on the same simulated games: the chance
you win this week with him, against without. Beside it, what the whole
move -- the drop included -- does to your title odds, since a pickup that
wins this week can cost more later. See DECISIONS.md, "Pickups for this
week, at every position". When none helps, the closest few are listed with
what they would have done, so an empty answer still shows who was tried.

### Your roster

The Roster tab values everyone you have for the rest of the season: points
a game, simulated points a week with its 10th-90th percentile, the points
of title odds you would lose without him, and what other managers see him
worth. Waiver drops are tried cheapest to your title first. See
DECISIONS.md, "Your roster, valued on the same seasons".

### How waivers are priced

A pickup is priced by what it does to your roster: this week, and the rest
of the season. Rest of season also counts **upside** as an option -- a
bench player is worth what he adds if his projection climbs past a starter
he could replace, and nothing if it does not, so the one whose role could
move (a rookie, a back one injury from the job) is worth more than a steady
veteran with the same projection. Projections typically move 1.5-2.5 points
a game over a month; rookies' move a quarter more and tend to rise.

Two other lists sit under the moves. **Upside stashes** are free agents who
do not clear the bar yet but have the most room to grow *for your lineup*: a
back behind a bell-cow (lead backs miss about 8.5% of games, and the next
man up has averaged 13.9 points when they do), a rookie, a player whose
opportunity just jumped. **Drop watch** is the roster spots that cost least.

The waiver reasons say what moved: "next man up: X is on IR (+3.1
projected)", "opportunity up: 13.1 expected pts/game over his last 2 vs 9.0
before", "rookie". A points streak with no more opportunity behind it is not
flagged -- measured, those mostly fade. The reverse of the next man up is
priced too: when a teammate comes back from an absence, the games a player
had without him count for less ("X is back: the games X missed count for
less (8.6 -> 7.9 a game)") -- a backup hands most of the job back, a lead
gives back little. A free agent on no NFL roster is worth nothing until the
news has him near a team: ESPN's player news (and Sleeper, which moves a
signing first) is read for talks, a deal close, or a signing, and he is then
priced on his new team at the chance he signs and plays ("news Oct 05: ...;
priced at 11.0 a game for KC when he plays, 38% to play this season").
A receiver or backfield group is held to what its team gives it: when one is
out the rest take his work in proportion, and a group projected for more
than the team's usual total is trimmed ("more of his team's targets this
week: a teammate is out or questionable (+8%)").
Title odds count who quarterbacks and backs face in the weeks to come
(predicted game environment and the opponent's points allowed to the
position); a card notes a clearly easier or harder road, and weeks 15-17.
A questionable player's chance to play comes from his tag and last practice;
FantasyPros' own read ("Are they playing?") is logged beside it every run and
blended in only as far as it proves more accurate on players who did or did
not suit up.

**Breaking news** sits at the bottom of each league's hub: the last 48 hours
of news about every player in the league -- yours first, then your
opponent's, then everyone rostered, then free agents -- tagged (out, injury,
healthy, role up, role down, move, legal) and linked. A player ruled out this
week is set out before the platform's tag catches up. The page refreshes
every two hours through the day and hourly on Sundays.
priced at 11.1 a game for KC when he plays, 38% to play this season").

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

Each sync reads the league page, the settings page, every team's roster,
your matchup and a few pages of free agents -- about twenty page loads, a
short pause apart. The first sync of a season also pages through the rest
of the regular-season schedule (one league page per week); it is kept in
the snapshots and not read again.

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

**FantasyPros (optional).** With a FantasyPros API key, the consensus
rankings are graded against our projections, used as a second opinion on
waiver and trade cards, and folded into the model of what other managers
see. They also get a share of every season value, weighted by how well they
have predicted this season's games against our model (a 30% starting guess
until enough games are graded). Add the key as the `FANTASYPROS_API_KEY` repository secret (Settings ->
Secrets and variables -> Actions), and in `.env` locally. Run the
`fantasypros-probe` job once from the Actions tab to check it works. Their
data stays on the workflow runner and is never committed.

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
