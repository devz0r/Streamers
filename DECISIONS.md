# Decisions

Every non-obvious choice made while building `streamer`, with the reasoning and
the evidence. Where a number was chosen, it was chosen by walk-forward backtest,
not by taste.

---

## Data

### nflreadpy, not nfl_data_py
`nfl_data_py`'s last release was 0.3.3 in **September 2024**. The nflverse
maintainers now publish **`nflreadpy`** (0.1.5, November 2025; Python 3.10+,
polars-backed), which is the actively maintained successor and reads the same
nflverse release artefacts. The brief said to check which is current, so:
`nflreadpy`. It is wrapped behind `streamer.data.nflverse` so a future swap
touches one module.

### Training window: 2021-2025
Five seasons, ~2,850 D/ST team-games and ~2,820 kicker-games. Going further back
crosses the 2020 empty-stadium season and the 17-game schedule change, and older
seasons are decayed to near-irrelevance by the recency weighting anyway.

### Three fallbacks for betting lines, not one
The weekly job must never hard-fail on a missing or rate-limited API key, so
`odds.fallback_order` walks: **The Odds API** → **`data/lines_week_N.csv`**
(hand-entered) → **nflverse schedule closing lines** → **last cached pull**.
The third of these was a genuine find: nflverse ships `spread_line`/`total_line`
for *upcoming* games, so even a completely key-less install produces real Vegas
numbers. Whatever source was used is recorded per game and flagged on the
published page.

### Sign conventions
A sportsbook prints the home favourite as `-3.5`; nflverse stores that as
`+3.5`. Everything internally uses the nflverse convention (`team_spread`
positive = this team favoured), and the manual CSV reader flips the sign on the
way in so the file can be typed the way a book displays it.

### Weather is manual-or-neutral by design
No free weather API is reliable and key-free enough to hang an automated weekly
job on. Historical rows train on the observed `temp`/`wind` nflverse ships on
the schedule; upcoming games use `data/weather_week_N.csv` if present, else
neutral. Domes and closed roofs are forced to neutral automatically regardless
of what any CSV claims. Wind is the only weather variable modelled, because it
is the only one with a robust published effect on field goals.

---

## Scoring

### Two profiles, not a scoring "mode"
ESPN and Yahoo differ in ways that reach past arithmetic into the model, so a
profile owns the league size *and* both scoring sheets, and each keeps its own
trained state under `results/<profile>/`. Sharing a fitted model between them
would be wrong: ESPN pays 2.5 for a sack against Yahoo's 1 and adds a
yards-allowed ladder, which changes what the D/ST target *is*, not just its
scale. The raw nflverse cache is shared, because play-by-play does not depend
on scoring.

Both profiles were re-tuned and re-validated independently after the change:

| Profile | Position | Rank corr | Vegas-only | Edge |
|---|---|---|---|---|
| ESPN | D/ST | +0.359 | +0.344 | **+0.015** |
| ESPN | Kicker | +0.171 | +0.122 | **+0.049** |
| Yahoo | D/ST | +0.333 | +0.318 | **+0.015** |
| Yahoo | Kicker | +0.178 | +0.128 | **+0.051** |

### One gap in the supplied ESPN sheet
**18-27 points allowed was absent.** The sheet ran 14-17 at +1 then jumped to
28-34 at -1. ESPN's default for that band is **0**, which is also the only
value that keeps the ladder monotonic across the gap, so 0 is what is
configured. The ladder validator would have rejected the gap outright.

The sheet also listed "Fumble Return TD" and "Fumble Recovered for TD"
separately at 6 apiece; those are the same event and are scored once. Likewise
"Interception Return TD" and "Fumble Return TD" are both the generic
`defensive_td`.

### What FUML means for a D/ST unit
`Total Fumbles Lost (FUML) -2` was supplied among the D/ST lines. A defence
cannot lose an *offensive* fumble -- those belong to the offence -- so the only
reading under which it affects a D/ST score is the unit losing the ball itself:

- a muffed punt or kickoff return, or a botched snap on a punt (special teams
  is part of the D/ST unit); or
- a defender coughing the ball back up after a takeaway.

Both are counted; the second is vanishingly rare (one occurrence in 2024 against
36 return fumbles). League-wide this runs about **0.06 per team-game**, so it is
worth roughly -0.13 points a week on average but a full -2 in the games where it
lands. It is `dst_scoring.fumble_lost` in config, so if the line was actually
ESPN's *offensive* FUML setting that got pasted along -- which would have no
bearing on D/ST or K rankings at all -- setting it to `0.0` reverts it.

### Field-goal buckets are split from accuracy buckets
Scoring needs 0-39 / 40-49 / 50-59 / 60+, because ESPN pays a premium for a
60-yarder and Yahoo does not. Modelling a *kicker's* 60+ accuracy separately
would be meaningless -- there are a handful of such attempts league-wide per
season -- so `FG_FEATURE_BUCKETS` collapses 50-59 and 60+ back into one 50+
accuracy rate. The registries are asserted to partition each other.

### Nothing is hard-coded, and the ladders are validated on load
No scoring value lives anywhere but `config.yaml`; `streamer.scoring` reads them
and `tests/test_scoring.py` asserts that changing the config changes the result.
Both ladders are validated on load for gaps, overlaps and a missing open-ended
top tier -- which is what would have caught the missing 18-27 band above if it
had been left out.

Adding a third league (Sleeper, a custom league) is a config block: a name, a
league size and two scoring sheets. The switch on the published page, the CLI's
`--profile` choices and the per-profile storage all follow from
`config.yaml`.

### Missed field goals are per-bucket, even where a league is uniform
ESPN exposes a single "Total Field Goals Missed" line at -1 and Yahoo penalises
nothing, but both are expressed per distance bucket so a league that only
punishes short misses needs no code change.

### Points allowed includes everything the opponent scored
`points_allowed_excludes_opponent_dst_st: false`. The common public
interpretation charges a D/ST with the opponent's full score, including a
pick-six thrown by your own offence. Hosts vary, so the exclusion is
implemented and configurable — just not the default.

### Attribution rules that are easy to get wrong
These are pinned by tests because they are silent failure modes:
- On a **kickoff**, nflverse sets `posteam` to the *receiving* team, so a
  kick-return touchdown has `td_team == posteam`. On a **punt**, `posteam` is
  the *punting* team. A naive "`td_team != posteam` means the defence scored"
  rule silently drops every kickoff return touchdown.
- A **blocked** field goal counts as a miss for the kicker and a blocked kick
  for the defence.
- Blocked punts, field goals and PATs are three separate code paths feeding one
  `blocked_kicks` column; they are accumulated and summed once.
- Team sacks are counted as sack *plays*, not player credits, so half-sacks
  don't double-count.

---

## Modelling

### The architecture: a Vegas anchor plus regularised adjustments
The brief's first principle is "start from a Vegas-derived baseline, then apply
adjustments", and it turned out to matter enormously that this be implemented
*literally* rather than as a flat regression that happens to include Vegas
features.

- **Stage 1** fits a lightly-regularised ridge on the Vegas block alone
  (implied total, game total, spread, home; plus dome and wind for kickers).
- **Stage 2** fits every factor against what stage 1 leaves behind, heavily
  regularised and scaled by the factor ledger's multipliers.

The property that makes this work: as the stage-2 penalty rises the model
degrades *gracefully back to the Vegas baseline* rather than falling apart. A
flat ridge over the same 25 features does not have that property — it splits
weight across correlated noisy factors and, out of sample, ranked **worse** than
simply sorting by opponent implied total. Measured on 2023-2025:

| Structure | D/ST rank corr | vs Vegas baseline |
|---|---|---|
| Flat ridge, 25 features, alpha 3 | +0.3143 | **-0.0033** |
| Vegas anchor + adjustments | +0.3296 | **+0.0119** |

Every one of the 32 anchor+adjustment configurations tried (2 residual scopes x
2 weighting modes x 4 penalties x 2 positions) beat the baseline. That
robustness, not the single best cell, is why the architecture was adopted.

### Shrinkage is per-position, and D/ST wants very little
This was the largest single improvement to D/ST and it was counter-intuitive.
Rolling priors already apply exponential decay (0.94/game, ~11-game half-life),
which is itself regularisation. Layering a 6-game league-average prior on top
was washing out real defensive identity — exactly the sack-rate signal the
brief calls the most stable weekly D/ST stat. Reducing it is monotonically
better for D/ST:

| `team_rate_prior_games` (DST) | rank corr edge vs Vegas |
|---|---|
| 6.0 | -0.0032 |
| 4.0 | +0.0032 |
| 2.0 | +0.0080 |
| 1.0 | **+0.0119** |
| 0.5 | +0.0130 |

`1.0` is shipped rather than the nominally-best `0.5` because 0.5 sits at the
edge of the tested grid and the two are within noise of each other; 1.0 keeps a
real (if small) prior for week-1 rows. Kickers go the other way and keep `6.0` —
team field-goal rates are noisier than they look.

### Yards allowed gets its own distribution, not a point estimate
ESPN scores total yards allowed on a nine-step ladder, which has exactly the
same `E[tier(x)] != tier(E[x])` problem as points allowed, so it is modelled the
same way by the same `LadderModel`. Its mean regression leans on volume
(opponent plays, drives, pace) and on the defence's own yards-allowed history
rather than on the betting market -- yards accumulate with snaps, and a defence
that concedes efficiency concedes yardage whatever the scoreboard says. It
predicts a mean of 343 against a historical 337, and it is the main reason
ESPN's D/ST rank correlation (+0.359) runs above Yahoo's (+0.333): yardage is a
more predictable quantity than takeaways, so adding it to the target adds more
signal than noise.

### Points-allowed tiers are a distribution, never a point estimate
The ESPN ladder is a step function, so `E[tier(PA)] != tier(E[PA])`. A defence
projected to allow 20.5 points is not "0 points"; it is a mixture that is mostly
0 and -1 with real mass on +3 and +4. `TierModel` predicts mean points allowed,
then samples historical residuals — bucketed by predicted level, because
high-scoring spots are also more variable — and bins the draws onto the ladder.
This is what produces `P(shutout)` and `P(under 14)`.

### Ridge, not gradient boosting
The brief said to fit both and keep whichever backtests better. Ridge wins for
both positions, and not narrowly -- the tree model fails the baseline gate
outright:

| Position | Estimator | Rank corr | Edge vs Vegas | MAE | Fit time |
|---|---|---|---|---|---|
| D/ST | ridge | **+0.330** | **+0.012** | **4.32** | 5s |
| D/ST | gbm | +0.262 | -0.056 | 4.50 | 228s |
| Kicker | ridge | **+0.169** | **+0.049** | **3.72** | 4s |
| Kicker | gbm | +0.041 | -0.078 | 3.89 | 241s |

This is the expected outcome for the shape of the problem: ~2,800 rows, 25
correlated features and a target dominated by variance the features cannot see.
Trees find structure that does not replicate, while a heavily-penalised linear
adjustment stage degrades toward the Vegas anchor instead. Both candidates stay
in the code and `streamer backtest` still evaluates both, because the right
answer could change if the feature set grows.

### Things that were tried and rejected
Recorded because negative results are results:

- **Component-wise D/ST** (separate regressions for sacks, INTs, fumble
  recoveries, touchdowns, then summed with the exact scoring weights). Better
  MAE (4.313 vs 4.317) but clearly worse ranking (-0.0044 vs +0.0119). Each
  component is cleaner, but the summation compounds five sets of errors.
- **EPA features** (`off_epa`, `def_epa_allowed`, `opp_off_epa`). No
  improvement — the information is already in points-per-drive and the pressure
  metrics. Adding features that don't earn their place is a cost, not a hedge.
- **Adaptive anchor-blend λ**, re-estimated weekly from a rolling holdout.
  Helped kickers slightly, but λ chosen by argmax over a short holdout is too
  noisy for D/ST and gave up as much in good seasons as it saved in bad ones.
- **QB-level opponent priors.** The most promising remaining idea and the reason
  it is not shipped: nflverse populates `home_qb_id`/`away_qb_id` only *after*
  games are played (100% null for 2026), so a Week 1 projection would have to
  guess the starter from last season — precisely when that guess is least
  reliable. Noted as future work behind a depth-chart source.

### The kicker model is honest about its ceiling
Weekly kicker scoring is close to irreducible noise: the strongest single public
factor (team implied total) correlates about **0.12** with actual points. The
model still beats a Vegas-only ranking by a wide relative margin (+0.048 rank
correlation, top-5 hit rate 57% vs 50%), but nobody should expect D/ST-like
ordering. Reporting a floor/ceiling band matters more here than the point
estimate.

---

## Self-calibration

### The ledger's multiplier has two parts
`multiplier = signal_prior x divergence`.

- **`signal_prior`** = `clip(|r_full| / median|r_full|, 0.2, 2.0)` — how strong
  this factor is *relative to the others*. Measured over every completed game
  the refit can see, not just prior seasons: it is a claim about absolute
  reliability, and starving it of the current season only makes it noisier.
  This is what stops twenty weak factors collectively drowning out the implied
  total.
- **`divergence`** = `clip(|r_blend| / |r_hist|, 0.25, 3.0)` — how far the
  current season has moved the factor from its historical value. This is the
  required automatic response to a rules or meta shift.

The blend `r = (n_cur * r_cur + k * r_hist) / (n_cur + k)` with `k = 250`
team-week observations means the historical prior dominates in Week 2 and the
current season dominates by Week 12, with no switch to throw.

For ridge the multiplier scales the standardised column, so it lands directly on
the coefficient. Tree models are invariant to monotone rescaling, so there the
ledger acts by *pruning* factors whose multiplier collapses below 0.3.

### Reported weights are total influence, not just the adjustment
A Vegas factor appears in both stages. The ledger reports its anchor coefficient
plus its adjustment coefficient, so "model weight" means what a reader assumes
it means.

### Per-team adjustments update Bayesianly
A team's season-to-date rate is blended with its own multi-season prior at a
strength of 8 pseudo-games, so a defence running hot earns credit in proportion
to the evidence it has produced. One three-sack game moves the number about a
ninth of the way, not all the way.

### Recency weighting
Current-season rows carry 2x weight (the brief's requirement), with an
additional 0.85 decay per season of age on top so 2021 doesn't out-vote 2025.

---

## Validation

### Leakage is prevented structurally, not by discipline
The rolling-prior recursion hands each row the accumulator state *before* that
row is folded in, so a feature cannot see its own result. Rows for unplayed
games carry `is_future = 1` and observe the accumulator without updating it —
otherwise projecting Week N would corrupt the priors used for Week N+1.
`tests/test_features.py` asserts this directly: putting a 1000x spike in the
final game must not change that game's own feature.

### The backtest measures the shipped model
`walk_forward` recomputes the ledger multipliers at every step, exactly as the
weekly refit does, rather than measuring a plain ridge and shipping something
else.

### Reported metrics
Rank correlation (Spearman) is the gate, because streaming only needs the
ordering. MAE and top-5 hit rate are reported alongside because they are what a
user actually experiences — and for D/ST the top-5 hit rate edge (66% vs 62%) is
larger and more stable than the rank-correlation edge.

---

## Interface

### The redesign: an app, not a report
The page read like a log: the four numbers that matter (this week's P(win),
title and playoff odds, the record) were inside the tabs, every card printed
its whole reasoning, tables were plain, and on a phone the tab strip ran off
the screen. Redesigned without dropping anything -- the text of every tab in
both leagues was compared word for word, old against new, built from the
same data; the only differences were trade notes where the other manager's
lineup is a tie (which bench player he "would start over"), which breaks
either way between runs:

- **Summary tiles** above the tabs, each a link to the tab that explains it.
- **A design system** in `static/page.css`: Inter, tabular numbers, raised
  panels, light and dark themes, colour-coded position chips, gains as
  pills, ranges drawn as floor-to-ceiling bars on one scale so they compare
  down a table, and ESPN photos and team logos.
- **On a phone** the tabs are a bottom bar with the four most-used and a
  "More" sheet (still CSS: a checkbox), the Hub drops its row numbers, and
  tables pin the player column.
- **A script that only adds** (`static/page.js`, `data-enhance`): the
  remembered league and tab and `#league-tab` links, search over the open
  tab, position filters, long reasoning folded to three lines behind "Show
  more", "updated 12 min ago" with a stale warning, and the home-screen
  manifest. With JavaScript off every tab, the switch and all the text are
  still there; the tests hold any script to being data or this enhancement.

Photos load from ESPN's image CDN in the reader's browser. The publish step
checks the CDN answers before using any, so a run that cannot reach it ships
none rather than empty frames, and the page drops any single image that
fails.

### The profile switch is CSS-only
The page carries both leagues and a segmented switch between them, built from
one hidden radio per profile plus `:checked` sibling rules. No JavaScript, so it
switches instantly on a phone, survives a stale cache, and keeps the page a
single self-contained file. The rules are generated per profile, so adding a
third league needs only config. A page rendered with one profile omits the
switch entirely.

### A tab per section
The page had grown to one long scroll per league: the hub, then the lineup,
waivers, trades, the season, D/ST and K, and the grading, each a fold or a
heading further down. Each league now has a strip of tabs -- Hub, Lineup,
Roster, Waivers, Trades, D/ST & K, Season, Model -- that sticks under the
league switch, so any section is one tap from anywhere on the page.

They are built like the league switch: a hidden radio per tab, a label per
radio, and generated `:checked` rules, so they need no script and work on a
stale cache. A section with nothing to show gets no tab; a league without a
snapshot shows only D/ST & K and Model. The hub's "details" links are labels
for the tab that explains the move, so they switch tabs rather than
scrolling; tapping a tab returns to the top of the page (the browser brings
the radio, which sits at the top of the league's panel, into view). The
rankings, their line-source badges and notices moved to D/ST & K; the
D/ST and K grading, the benchmark and the factor ledger to Model, beside the
player projections' own grading.

The selected tab is not remembered between visits: the page opens on the
hub, which is the summary, and remembering would take the script the page
has avoided.

### Plain HTML, no framework, no JavaScript
*(Amended: the My-team lineup editor is the one piece of script -- see "Try a lineup" below. Everything else,
including the three-lineup comparison, still renders without it.)*
The published page has to render instantly on a phone over a bad connection and
still work in five years. It is a single self-contained file with inline CSS,
system fonts, and `prefers-color-scheme` for dark mode. Card layout rather than
a table, because a 9-column table on a 390px screen is unreadable. Wide tables
scroll inside their own container so the page body never scrolls sideways.

### Predictions are stored when ranked
`streamer rank` persists what it published to `results/predictions.parquet`, so
`streamer update` scores *what was actually shown* rather than a reconstruction.
If a week was never published, `update` re-fits on games before that week and
says so in the review — still out-of-sample, but labelled as a reconstruction.

### The workflow is scheduled early, and off the hour
It was scheduled for 11:00 UTC (07:00 ET), and the docs promised 07:00 ET. It
never once ran then: every scheduled run through September landed 3.5-4.5
hours late, between 10:38 and 11:26 ET. GitHub treats a cron schedule as a
request, queues scheduled runs behind everyone else's, and is most congested
exactly on the hour. The schedule is now 07:17 UTC Tuesday and Wednesday and
08:17 UTC Sunday: off the hour, and early enough that the same delay still
lands the page by breakfast, and the Sunday page ahead of the 09:30 ET London
games. The promise in the docs is now "early morning", with the page's own
"synced" time as the record of when it actually ran.

Tuesday scores the completed week and publishes the next; Wednesday publishes
again once waiver claims have processed and lines have settled; Sunday
publishes with player props and injury designations. None of them can see
official inactives, which are announced 90 minutes before kickoff -- an
earlier version of these docs claimed the Sunday run knew them.

## In-season suite

### One normalised snapshot, two thin adapters
ESPN and Yahoo expose very different objects (ESPN a typed `League` with box
scores; Yahoo nested JSON keyed by string indices). Both are flattened into one
`LeagueSnapshot` — teams, rosters, slots, the week's matchup and the free-agent
pool — and everything downstream (projections, optimiser, waivers, panel, CLI)
only ever sees that. Network access is confined to the two `fetch_snapshot`
functions, so the whole engine is testable offline from a fixture snapshot, and
the Wednesday workflow, which is where the secrets live, is the only place the
platforms are actually called. Adding a platform means one adapter file.

### Skill projections: opportunity blended with production, damped by Vegas
*(Superseded by "Opportunity moves fast, efficiency slowly" below; kept for the history.)*
D/ST and K reuse the streaming rankings. Every other position gets a blend of
trailing-4 actual PPR and trailing-4 nflverse *opportunity-expected* points
(`ff_opportunity`), shrunk toward the position mean with 3 pseudo-games, then
scaled by `(implied total / trailing implied total) ** 0.5`. Walk-forward on
2021-2025 (13,414 startable player-weeks, trailing >= 5 PPR), rank correlation
with the following week's actual:

| | trailing avg | opportunity only | blend | blend + Vegas (d=0.5) |
|---|---|---|---|---|
| QB | 0.347 | 0.352 | 0.364 | 0.374 |
| RB | 0.451 | 0.467 | 0.473 | 0.479 |
| WR | 0.411 | 0.430 | 0.442 | 0.437 |
| TE | 0.332 | 0.355 | 0.365 | 0.376 |

The blend beats trailing average at every position; the damping exponent was
swept (0, 0.25, 0.5, 0.75, 1.0) and 0.5 is the overall optimum — full Vegas
scaling over-reacts. Residual spread is calibrated per (position, projection
bucket) rather than assumed: it is 7-8 points for a mid-range starter, and on
real 2025 week-10 rosters 73% of actuals landed within one sd and 95% within
two. When the platform publishes its own projection it is averaged in at equal
weight, because it carries injury and depth-chart news the trailing window
cannot see.

### The lineup objective is P(win), not expected points
Maximising expected points is the right answer only when you are evenly
matched. The optimiser draws a shared matrix of 20,000 joint samples for every
player on both rosters, enumerates every valid lineup (flex included, with
low-projection bench pruned), and picks the lineup that beats the opponent's
best lineup in the most draws. An underdog is correctly steered toward
high-variance plays, a favourite toward the floor. The current lineup on the
platform is scored the same way so the panel can say what the swaps are worth.
Injury tags shrink the mean by the chance to play (Q 0.75, D 0.25) and widen
the spread; OUT and bye contribute zero.

### Waivers are priced by what they do to the roster, not by raw points
The first live run against a real ten-team league recommended six backup
quarterbacks as +10 to +15 moves: a 16-point QB "beats" a 5-point bench RB on
raw projection, but neither would start, so the true gain was nil. A move is
now scored by the change in **roster value**: the expected points of the best
lineup the roster can field (greedy, dedicated slots first, exact for a
one-flex structure) plus a small credit for bench depth -- the first and
second bench body at RB/WR and the first at TE/QB, weighted 0.25/0.10 and
0.10 for the chance they are needed in a given week -- measured over
**replacement level**, the best free agent at that position other than the
one being considered. That last part is what makes the backup QB worth almost
nothing when a 15-point QB is always on the wire, and an RB who would start
over your flex worth every point of the difference. The two horizons blend as
before: `w · next-week + (1-w) · rest-of-season`, `w = 0.35 + 0.5 · week/18`.
Drops never go below one QB/TE/K/DST unless the pickup plays that position,
never touch the best player at another position, and are never taken from an
IR slot (dropping an IR player frees no bench spot). Moves are assigned
greedily and re-scored after each accepted move, so the list is a sequence
that can all be made together, each priced on top of the last; ties between
zero-cost drops go to the pickup's own position, so a new defence replaces the
old one rather than a dead receiver. The bar is 0.5 points of roster value,
lower than before because lineup-marginal gains are inherently smaller than
raw ones.

### Injured reserve is a roster state, not an injury tag
ESPN reports `INJURY_RESERVE` and `DAY_TO_DAY`, neither of which the first
status tables knew, so an IR stash was valued as a healthy bench body and
offered as a drop. Long-term statuses now zero rest-of-season value; a player
in an IR *slot* is excluded from lineup enumeration (activating them is a
roster move that needs a free spot) and from drop candidates. When such a
player's tag has cleared to day-to-day the report says so, because the
platform will refuse further moves until the roster is made valid. ESPN also
publishes a projection of 0.0 for anyone it does not project; that is treated
as no projection unless the player is out, so it cannot halve ours.

## Staying current during the season

### The week is decided by the calendar, not by whether scores have landed
The first in-season bug: the published page sat on Week 1 after Week 1 had
been played. The detector asked "what is the highest week with a final
score?", which is a question about the *data feed*, not about the season. Two
things then went wrong at once. nflverse had not yet posted results, so the
answer was "none", and a `max(1, completed)` floor dressed that zero up as
"week 1 is complete" -- so the job both republished week 1 and tried to score
a week that had not been played. Weeks are now read off the schedule's
kickoff dates: a week is complete when its last kickoff is in the past, and
the upcoming week is the first whose last game has not started. Scores are
still consulted, but only to move the week *forward* (a feed ahead of the
calendar, or a rescheduled game), never to hold it back. `completed_week=0`
is now reported honestly and the scoring step is skipped, rather than being
rounded up to a week that never happened.

### Live feeds get a cache TTL; completed seasons keep their cache forever
The same bug had a second cause that the calendar fix alone would have
hidden. Every raw pull is memoised to `data/raw/` for reproducible backtests,
`load_schedules` was documented as refreshing "whenever the caller asks", and
no caller ever asked. Because the weekly job restores `data/raw` from the
previous run, `schedules.parquet` was still the copy fetched before the
season opened -- no scores, and stale closing lines feeding every projection.
`cached_frame` now takes `max_age_hours`: the schedule and the current
season's play-by-play, rosters, player stats and opportunity data expire
after six hours, while finished seasons are immutable and cached
indefinitely. A failed re-fetch still falls back to the cached copy, so an
upstream outage degrades to stale rather than breaking the run.

### Tuesday publishes too
Publishing only on Wednesday meant that from Monday night until Wednesday
morning the page showed recommendations for a week that was already over --
the window where a manager is actually setting up the next week. The Tuesday
run now scores the finished week *and* publishes the next one; Wednesday
publishes again once waivers have processed and the lines have settled.


## Sportsbook player props

### A prop line is a median, and fantasy scoring wants a mean
The naive reading of "over 84.5 rushing yards" is 84.5 expected yards. That is
the *median*, and using it as the mean throws away everything the price says.
Given de-vigged P(X > L) = p and a normal approximation, the mean is
`L + sd * z(p)`: the line, plus however far the market leans. So the spread of
each stat has to come from somewhere, and it is fitted rather than guessed --
2021-2025 nflverse weekly stats, players bucketed by their trailing-4 average
(the level a book would be pricing), sd of the actual result taken within each
bucket, then a line through the buckets. Receiving yards run
`sd ~ 15.1 + 0.357 x line` (so ~37 at a 60-yard line), receptions
`1.15 + 0.277 x line`, passing yards flatten out near 80 because the low end
is backups playing partial games. The fits live in `odds.props.spread`.

### Touchdown markets are Poisson, and anytime-TD is a rate
`player_pass_tds` is a count with a half-point line, so the rate is solved by
bisection on the Poisson survival function -- exact, and it inverts back to
the price it came from (a test asserts that). Anytime-TD is the one that
invites a real error: the price gives P(scores at least one), and using it as
an expected touchdown count silently deletes every multi-touchdown game. The
rate is `-ln(1 - p)`, which at p = 0.45 is 0.598 touchdowns rather than 0.45 --
a third of a touchdown, or two fantasy points, per player.

### De-vig before anything else
Two -110 prices are not a 52.4% chance each; they are a coin flip with the
book's margin on top. Normalising the pair back to 1 recovers the market's
actual view, and the test asserts a de-vigged pair sums to exactly 1. Pinnacle
carries double weight in the consensus because its margin is the thinnest of
the three books; the others are there to cover players Pinnacle has not posted.

### Only pay for players who could start
Props are billed per market per *event*, so a full slate at seven markets is
~112 credits and the free tier is 500 a month. Only games featuring a player
who could actually start for you are fetched -- not the opponent's roster,
whose implied points are never displayed and whose contribution to P(win) is
scored with our own projections, and not players already out, on bye or on IR.
Capped at eight events, that is ~56 credits a run and ~480 a month across the
two weekly publishes. The reported balance is printed on the page, because a
quota that runs out silently in week 11 is worse than one you can watch.

### Spend the month's credits, not a fixed eight games
Props were capped at eight games a run to stay inside the free 500 credits.
On the week-4 Sunday that bought 8 of 16 games, ranked by how many of
your players were in each across both leagues, and left the ESPN lineup
with props for 7 of 14 startable players: Ollie Gordon (MIA), Dalton
Schultz (HOU), Drake Maye (NE) and KC Concepcion (CLE) were in games
nobody bought. The month had 448 credits left and three more Sundays,
with the midweek runs costing about 14 credits a week.

Credits reset on the first of the month, and the free event list reports
the balance. A Sunday run now buys
`(credits left - midweek runs to come x 1 game x 7) / (Sundays left x 7)`
games: 14 on 4 October, and the whole slate on the last Sunday. Your
games go first, then the rest. Your opponent's players and the free
agents who could start for you are in those, and the blend uses their
props too. Other days buy at most two games, and a credit-free refresh
reuses everything already bought rather than the first eight.

Other free sources were looked at:

- **SportsGameOdds** (free plan: 2,500 games a month, all props per game
  counted once, nine bookmakers on the free plan per its own pages). It is
  read as a second feed when its key is set (`streamer.data.sgo`), merged
  into the same consensus: a price both feeds have for a player, stat and
  book counts once, and pick'em sites (PrizePicks and the like) are left
  out, since their lines are not sportsbook prices. The reference covered
  yardage and reception stat IDs but not touchdowns or interceptions, so
  several spellings are accepted and an `sgo-probe` job prints the stat
  IDs, bet types and books the feed really carries before its numbers
  are relied on. A cache of six hours keeps refreshes within the plan.
  The first probe (week 4): six books on player markets on the free plan
  (FanDuel, DraftKings, ESPN Bet, Bovada, Caesars, BetMGM; no Pinnacle),
  receptions under `receiving_receptions`, and touchdowns priced both as
  yes/no and as over 0.5 by the same book -- kept once, as the over/under,
  so one book is one opinion.
  Its first real fetch returned 173 games: every NFL game already priced,
  weeks ahead, each counted against the 2,500. Props for later weeks
  were also averaged into this week's for the players in them. The fetch
  now asks only for games starting in the next 96 hours (`startsAfter`,
  `startsBefore`), drops any outside that window from the cache too, and
  keeps a month's tally that stops fetching at 2,300.
- ESPN's public API documents no player-props endpoint, and nflverse
  carries none.
- Scraping sportsbooks or pick'em sites directly is against their terms.

### Do not buy props nobody has posted yet
The first live pull spent its budget on a Tuesday and came back with two
Buffalo bench players. The event cap was taking the earliest kickoffs, which
that far out means Thursday night football -- but fixing the ranking barely
helped, because the real cause is that books post a game's props a day or two
before kickoff. A midweek request about Sunday is not just wasteful, it
returns nothing. Events further out than `window_hours` (72) are now not
requested at all, which makes the Tuesday and Wednesday runs nearly free and
concentrates the spend on a new early-Sunday publish -- the run where the
props exist, the injury designations are in, and the lineup is actually being set.
Coverage is reported on the page ("priced 6 of your 14 startable players"), so
a column of dashes is explained rather than mysterious.

### The market gets a column, not the last word
The Vegas number sits beside ours rather than replacing it, and the panel
names the players they disagree about. The books know things the trailing
window cannot -- a snap count from Thursday practice -- but they are pricing a
betting market, not a fantasy lineup: no prop exists for most kickers and
defences, and a blank is shown as a blank rather than a zero so a missing
market never quietly benches anyone.


### Whether a game has been played is a fact about the game, not the week
From Thursday evening until the following Tuesday, the published rankings
showed two teams. The slate builder asked "has this week been played?" by
checking whether any play-by-play existed for that `(season, week)` -- and
Thursday Night Football answered yes for the whole week. No placeholder rows
were built for the thirty teams still to kick off, so the inner join against
play-by-play kept only the two teams who had. The window the bug covered is
precisely the window the tool exists for: Sunday morning, setting a lineup.

The question is now asked per `game_id`, against the schedule's final scores
rather than the presence of play-by-play. A game in progress has partial
counts and no final score, so it is treated as unplayed and its half-finished
play-by-play is dropped before the priors are built -- feeding a first quarter
into a team's season rates would be worse than ignoring it, and would also
collide with the placeholder row for the same game.

Worth noting what made this hard to see: nothing failed. No error, no warning,
no empty frame -- just a shorter table, on a page that legitimately varies in
length. The regression tests now pin the partly-played week directly, because
the failure mode is silence.


## Reading Yahoo through the website

### The developer API is closed, the website is not
Yahoo stopped provisioning its Fantasy Sports API in 2026: the permission was
removed from the app form and entitlement moved to a manual review with no
published turnaround. The fantasy website still serves every manager their own
roster, to their own logged-in session, which is exactly how the ESPN adapter
has always worked (with `espn_s2`/`SWID`). So Yahoo is read the same way: the
Cookie header copied from a browser, stored as the `YAHOO_COOKIE` secret. The
OAuth adapter is kept for anyone whose Client ID has been granted access; when
a cookie is configured it takes precedence.

The trade is stated plainly in the README: Yahoo's session cookies are scoped
to all of yahoo.com, so the secret is equivalent to being signed in; they
expire every few months; and a markup change can break the parser. Signing out
of Yahoo invalidates the secret.

### Parsed against probed structure, looked up by header
The parser was not written from memory of what Yahoo's pages used to look
like. `streamer yahoo-probe --detail` fetched the real pages from the workflow
runner and printed their structure -- tag and class outlines, Yahoo's
`data-tst` hooks, header rows with spans -- with all identifying text scrubbed,
since workflow logs on a public repository are public. A test asserts a name,
an email, a crumb token and a player id cannot survive into that output.

The anchors it found: the lineup slot in `span.pos-label[data-pos]`, the
player id and name on `a.name[data-ys-playerid]`, team and position as
`LAR - QB` in the detail span. Every numeric column (projection, % rostered,
points for) is located by its header label rather than its position, so Yahoo
adding or reordering a column does not silently shift values.

### Only trust a projection the page says is a projection
The free-agent list shows whatever stat type it was asked for in its Fan Pts
column. It is asked for the weekly projection, but the value is only recorded
as one when the page's stat selector confirms "Projected ... Week N" -- a
season total read as a weekly projection would quietly skew the blend with
our own model. When the label is missing, free agents simply carry no Yahoo
projection and are projected by our model alone.

### Eight page loads, at a person's pace
League, your team, your matchup, your opponent's team, and four pages of the
free-agent list, with a short pause between each. Your team and opponent are
found from the "My Team" link and the matchup page, so no team-id secret is
needed. An expired session fails with an error that says to re-copy the
cookie, which lands in `sync_status.json` and on the page.


### Recent form, anchored to the player's own record
*(Superseded by "Opportunity moves fast, efficiency slowly" below, which keeps the
long anchor for efficiency and lets opportunity move.)*
The first real Yahoo waiver list recommended picking up Bryce Young to start
over Matthew Stafford, and rated Drake London (a WR1) level with Ray Davis (a
bench back). Neither was a parsing error. The skill model averaged the last
four games and shrank toward the league mean by *career* game count -- so a
veteran was barely shrunk at all, and four games drove the number outright.
Young had two big games; London a cold spell; Davis one garbage-time week-18
game.

The original validation measured within-position ranking, which a streaky
model can still do passably. Lineups and waivers make a different decision:
*of these two players, who scores more?*, across positions, on absolute
numbers. So the test was rebuilt around that -- pairwise start/sit accuracy on
identical random pairs, calibration slope, top-tier bias -- against an exact
replica of production, walk-forward 2022-2025, 18.5k player-weeks:

| | production | + own 17-game prior (k=6) |
|---|---|---|
| picks the higher scorer (any two players, same week) | 70.4% | **71.1%**, better in all 4 seasons |
| calibration slope (1.0 = right) | 0.89 | **1.01** |
| bias on the top 15% of projections | -1.48 | **-0.37** |
| MAE | 5.15 | **5.10** |
| within-position rank corr QB / RB / WR / TE | .427 / .590 / .536 / .475 | **.451 / .601 / .558 / .504** |

The fix: shrink the four-game average toward the player's own ~17-game average
(itself shrunk toward the league mean when thin), with six games of weight.
k=4 and k=8 were within noise of k=6; six is the one that lands the slope at
1.0. Excluding week 18 from the windows made no consistent difference and was
not adopted. The slope of 0.89 is worth dwelling on: the earlier diagnosis in
conversation was "compression toward the middle", which is what the London
example looked like, but measured against production the error ran the other
way -- overreaction -- and the fix corrects both ends at once.

### Waivers compare like with like
Yahoo's player list does not reliably say which stat its points column shows,
so free agents carry no Yahoo projection while rostered players carry one
blended into ours. When that happens, waiver moves value every player on the
model alone (`model_projection`) rather than pitting a blended number against
a model-only one. Rest-of-season value was already model-only for everyone.


### A projection needs evidence the player is playing
The ESPN waiver list recommended Amari Cooper, 0% rostered, to start over a
current receiver. He had not played an NFL game since 2024. History windows
count games played, not time elapsed, so a player out of the league keeps his
old average indefinitely -- a flaw that predated the long-run prior, which
would only have made it worse. Joe Mixon, with no NFL team, came through the
same way under an injury tag.

The rule: a player's history only counts if something current backs it. If
his last game is more than `inactive_weeks` (4) back, the platform's
projection is used when there is one -- a return from injury -- and otherwise
he is treated as inactive. An injured player *on an NFL roster* keeps his
history for rest-of-season value, since his tag already zeroes this week. A
player with no NFL team is inactive unless the platform projects him: ESPN's
free-agent pool carries 242 unsigned skill players, and projects one.


### A platform's zero is a statement about this week
The ESPN lineup recommended starting Josh Jacobs while he was on the
commissioner's exempt list. ESPN has no status for that and listed him as
DAY_TO_DAY, which reads as "probably plays". ESPN *did* project him for 0.0 --
but an earlier rule treated ESPN's 0.0 as "no projection", to stop injured
stashes losing their rest-of-season value, and so discarded the one signal
that was right.

Those were two different questions run together. A zero from a platform that
projects nearly everyone is a statement about *this week*: on the live league
ESPN projected 154 of 160 active rostered players, and the six zeros were
Jacobs and five Doubtful players it had already ruled out. So a platform zero,
for a player on an NFL team and not on bye, now sets this week's projection to
zero -- and says so on the page -- while rest-of-season value is untouched. It
is only trusted when the platform covers at least 80% of active rostered
players, so a feed that has not populated yet cannot bench a whole roster.


### Buy each game's props once
By late September the Odds API balance had fallen from 398 to 134 in eight
days. Most of that was manual runs during development, but part was
structural: each league pulled its own props, so a game holding players from
both leagues was bought twice, and a re-run an hour after a scheduled run
bought everything again.

Now the publish step loads every league first and makes one pull, with the
event budget going to the games that hold the most of your players across both
leagues (kickers and defences excluded, since books do not post props for
them). Each game's payload is cached in memory and under `data/raw/props` for
`cache_minutes` (180); the weekly job restores `data/raw` between runs, so a
manual re-run shortly after a scheduled one pays nothing, while scheduled runs,
a day or more apart, always fetch fresh. A run is capped at 56 credits
whatever the number of leagues. The page says how many games were bought
versus reused.


## Winning the week, and finding the breakout first

The request was three things: lineups that account for correlation (stacks,
players in the same game), lineups that chase upside when behind and protect
the floor when ahead, and waivers that find a breakout -- a backup who
inherits the job, a player whose role just grew -- before his projection
catches up. Each was measured on 2021-2025 before anything was built. The
research scripts' results are summarised here; the shipped numbers are refit
by `scripts/fit_outcome_model.py`.

### Opportunity moves fast, efficiency slowly
The previous projection shrank a four-game blend of points and expected
points toward the player's 17-game record. It treated a touchdown streak and
a new role the same way. Splitting the projection into **volume** (expected
points per game from nflverse `ff_opportunity`: targets, carries, field
position) and **efficiency** (points per expected point) and giving each its
own speed did better. Volume is an exponentially weighted mean (half-life 2.5
games) anchored to the 17-game volume with 2 games of weight. Efficiency is
the 17-game rate shrunk to the position's with 150 expected points of prior,
with a quarter of the weight on the last four games. Walk-forward, same
evaluation as before (19,257 player-weeks, identical random pairs):

| | previous | volume x efficiency |
|---|---|---|
| picks the higher scorer, 2022 / 2023 / 2024 / 2025 | .715 / .733 / .735 / .724 | **.718 / .734 / .739 / .728** |
| MAE, all seasons | 5.019 | **4.958** |
| calibration slope | 1.019 | 0.977 |
| rank correlation QB / RB / TE / WR | .474 / .618 / .505 / .577 | **.481 / .627 / .513 / .582** |
| bias on players whose opportunity fell sharply | -1.30 | **-0.94** |

Better in every season and at every position. Tried along the way:
- half-life 1.5 matched 2.5 on accuracy but over-reacted (slope 0.95);
  half-life 4 was less accurate;
- a volume prior of 1 game was equally accurate but over-reacted, while 3-4
  games lagged real role changes;
- efficiency priors of 40 and 80 points did slightly worse than 150;
- recent efficiency at weights 0 / 0.25 / 0.5 / 1: 0.25 was best by a hair.
  Scoring streaks carry a little signal, not much.

Two things that were **not** adopted:

- *Chasing one-game jumps.* After a single game with 10+ expected points from
  a player averaging under 6, his next four games averaged 6.6, and the
  previous model projected 6.6. Most one-game spikes are game script. A
  faster model over-projected those players by a point.
- *A youth bonus in the mean.* Rookies beat their projections by ~0.3 a game
  where veterans fall ~0.2 short. A 5% rookie bonus removed the bias but
  worsened MAE in 2023 and 2024. Youth goes into waiver upside instead
  (below).

### The next man up
The case the trailing numbers cannot see is a backup who is about to get the
job, because he has never had it. From every game 2021-2025 where a regular
(5+ expected points in his last game for the team) was missing, for the
teammate at his position with the most opportunity:

- First game of the absence, running backs: the next man up averaged **13.9
  points**. Backs projected the same with no absence averaged 10.7. He hit 20+
  25% of the time, against 11%.
- Modelled as inheriting a share of the gap between the two players'
  opportunity, least squares on 2021-23 and tested on 2024-25: RB 0.35
  (bias +4.1 -> +2.8, RMSE 9.6 -> 9.0), QB 0.30, TE 0.10.
- Receivers: a missing WR's targets spread across the WRs, the TE and the
  backs. No single receiver gained enough to beat noise, and every share
  above zero worsened out-of-sample error, so receivers get none.

In the tool, "missing" is read from the snapshot: IR, PUP, NFI and suspension
apply for this week and the rest of the season; OUT or a platform zero for
this week (and 30% of rest of season); Doubtful and Questionable at their
chance of sitting. It moves only the model half of the projection, because
the platform's half already carries depth-chart news. The waiver reason says
it: "next man up: X is on IR (+3.1 projected)".

### Correlated draws, a measured shape, and a coin flip for injury tags
The lineup optimiser already maximised P(win). The simulation underneath it
did not know that players in the same game move together. Residual
correlations by role, 2021-2025, stable between 2021-23 and 2024-25:

| same team | r | opponents | r |
|---|---|---|---|
| QB - WR1 | +0.34 | QB - opposing D/ST | -0.44 |
| QB - WR2 | +0.29 | K - opposing D/ST | -0.29 |
| QB - TE | +0.26 | RB1 - opposing D/ST | -0.26 |
| K - D/ST | +0.24 | WR1 - opposing D/ST | -0.20 |
| QB - WR3 | +0.19 | QB - QB | +0.18 |
| QB - RB1, K - RB1 | +0.11 | D/ST - D/ST | -0.17 |

Receivers on the same team are uncorrelated with each other (targets
compete, but the passing game rises together), and so are the two backs.
Draws are correlated normals (a Gaussian copula) mapped through each
position-and-level's measured residual shape. That shape is right-skewed:
the median sits below the mean and the 99th percentile is 3-4 sd out for
low projections. A Questionable or Doubtful player plays with the tag's
probability and otherwise scores zero, instead of being a narrower normal
around a discounted mean.

Tested and rejected: **player-specific volatility**. Touchdown share of
expected points, receiving share, average depth of target, the player's own
past residuals, youth. Fit on 2022-23, tested on 2024-25: the realised
spread by predicted quartile was flat at every position, and the Gaussian
log-likelihood gain was zero or negative. So "boom-or-bust" labels are not
used. Beyond position and projection level, spread comes from correlation,
injury tags and skew.

### Ties go to projected points
Picking the best of a few hundred lineups on the draws that scored them
favours whichever got lucky. So the top 30 are re-scored on a fresh 20,000
draws. Every lineup the simulation cannot separate from the leader -- within
max(0.5 points of P(win), 2 paired standard errors) -- counts as tied, and a
tie goes to the higher projection. On the synthetic league a favourite at
70-80% keeps a one-point-better player over a steadier one, because a point
of mean is worth more than the variance. The steady player wins only when
projections are level (tested). When the recommendation does differ from
the max-points lineup, the panel names the reason (stack, hedge, range,
injury tag) and the P(win) change.

### Waiver upside is an option, and handcuffs are priced as one
Rest-of-season value adds, for each bench player, `E[max(0, V - bar)]`. V is
his per-game projection a month from now (normal around today's, spread
measured) and bar is the weakest starter whose slot he could fill. How far
projections move over four games, 2021-2025: typically 1.5-2.5 points a
game by position and level. Rookies move 1.23x as far and rise 0.37 a game
more than veterans; second-year players 1.04x and +0.31. The
option is honest about its size: for most free agents it is a few tenths of
a point a game, which is enough to break a tie toward the rookie, not to
invent a move.

The **upside stashes** list covers what a single number cannot: free agents
with a reason to jump who do not yet clear the bar. It is ranked by value to
*your* lineup, so a backup quarterback in a one-QB league does not top it.
Handcuffs -- the RB2 behind a lead back with 12+ a game -- are valued at
8.5% (how often lead backs miss the next game, 2021-2025) times how far the
inherited job would put him above your weakest eligible starter.

### Loose name matches need the same team
The first live run recommended stashing "Jalon Daniels", a TB rookie
quarterback with no NFL snaps. The matcher's fallback -- first initial plus
surname, for "DJ" vs "D.J." -- had matched him uniquely to Jayden Daniels,
and he inherited Jayden's history. The fallback exists for spelling
variants of one player, and one player is on one team. It now requires the
same NFL team; a real spelling variant still matches.


## Sunday, live

### A played game is a fact
Once a player's game kicks off, his lineup spot is frozen; once it is final,
his score is known. The simulator used to draw Drake London's Thursday game
like any other, so on Sunday morning his 28.4 counted as "13.8 plus noise"
and the Yahoo P(win) read 62% where it was really 79%. Now every player on
either roster whose game has started is **locked**: a locked starter keeps
his slot in every candidate lineup and a locked bench player cannot come
in. A player whose game is final plays in every draw at his actual score.
Finals come from nflverse -- weekly player stats for skill players (full PPR),
play-by-play through the training-label scoring code for kickers and
defences, per league. nflverse posts a game some hours after it ends; until
then a finished game's players stay simulated and the panel says so. Kickoff
times come from the schedule in Eastern time.

### Try a lineup
The panel shows the lineup as set, the best-P(win) lineup and the most-points
lineup side by side, and a small editor to pick any starters and watch P(win)
and the projection move. It computes in the browser from 5,000 of the very
draws the optimiser used (the user's players and the opponent's total,
stored as int16 tenths of a point, ~200KB a league), so a hand-built lineup is
scored exactly as the optimiser scores one, to about +-0.7 points of P(win).
Locked players cannot be moved in the editor either. It is the one piece of
script on the page; without it the comparison table still shows.

### A refresh that costs nothing
A static page cannot start a workflow without carrying a GitHub token, and
this repository is public, so the "refresh" link opens the workflow's Run
button instead (one more tap, already signed in on the phone). Its default
job, `refresh`, re-syncs both leagues -- rosters, set lineups, the wire, the
opponent -- and rebuilds the page with `STREAMER_CREDIT_FREE=1`: game lines
fall back to the last pull, and props are served only from what earlier runs
bought (any age within the week); the events list, which is free, is the
only Odds API call. Manual runs default to it, so a stray tap cannot spend
credits; `publish` and `both` still buy what they need.

### Ours, the platform's, or the market's?
Today the lineup maths (correlations, favourite/underdog, P(win)) runs on our
projection blended 50/50 with the platform's; the sportsbook number is shown
beside it and drives the "market would start" line, but does not enter the
simulation. Which deserves more weight cannot be measured yet: there is no
free archive of player props, so there is no history to backtest. From this
week every publish records ours, the platform's and the market's number for
every priced or rostered player before kickoff
(`results/<profile>/skill_log.parquet`). After four or five weeks that is
~1,000 player-games with actuals, enough to fit blend weights the same way
the rest of the model was fit.

### The market gets a weight, and the weight learns
Where a book priced a player, the projection the lineup maths uses is
`(1 - w) * ours + w * market`, on the if-he-plays number with the injury
discount re-applied (props assume he suits up). Locked, out and
platform-ruled-out players are untouched. The weight starts from a prior of
0.4 -- sharp, Pinnacle-weighted prices deserve real weight, but ours carries
opportunity and next-man-up information the prop may not yet reflect -- and
is scaled by book depth (0.6x with one book, 0.85x with two), because a
one-book price is noisier. Every publish then fits it: the projection log
joined to actual points, the weight minimising squared error, shrunk toward
the prior with 300 games of weight, and only once 150 logged games have
results. So it moves toward whichever source has actually been more right
this season, without anyone retuning it. The "market would start" line and
the disagreement list compare against our number before the blend, and the
market lineup now respects locks like the optimiser does.

### Injury tags, measured
The play probabilities for injury tags were guesses (Questionable 75%,
Doubtful 25%). Measured on 2021-2025 official reports against snap counts,
for regular contributors (40%+ of snaps earlier that season): Questionable
RB/WR/TE played **72%** (n=1,238), Questionable QBs **42%** (n=156), and
Doubtful players **0.5%** (n=183) -- a Doubtful tag means out. Practice
status separates Questionable further, but the platforms do not carry it, so
at first it was not used. The page shows a tagged player's projection both
ways: "9.7 (13.5 if he plays, 72%)".

Later, the official reports themselves (nflverse, refreshed every few hours)
supply the last practice once the final report with the game status is out.
Refitted on 2022-2025 the same way: Questionable RB/WR/TE played **57%** after
no practice (n=141), **73%** after a limited one (552), **88%** after a full
one (147); QBs 29% / 39% / 82% (14 / 84 / 11). Against the single rate it was
better in every held-out season (Brier 0.184->0.182, 0.216->0.204,
0.219->0.211, 0.190->0.185). A questionable player's card says which applies.
Before the final report, the single rate stands: a Wednesday rest day is not
a Friday miss.

FantasyPros publishes its own chance to play ("Are they playing?", the
`/injuries` endpoint's `probability_of_playing`), which can carry what a
practice report cannot -- a coach's word, a beat reporter's. There is no
archive of it to test on, so it starts with no say: each run logs ours and
theirs for every questionable or doubtful player whose game is still to come
(`data/raw/fantasypros/play_log.parquet`, with their other data on the runner -- never committed --
the last read before kickoff kept), each
played week grades both against who has a box score, and ours moves the
Brier-best share of the way toward theirs, shrunk toward 0 by 40 cases and
not at all below 40 (`fantasypros.play_odds`). Their number is never shown;
the card notes when it helped set ours.

### How much this week's number leans on the platform's
Our model and the platform's projection were blended 50/50 by assumption.
Each league's log now holds ours and the platform's number for every rostered
or priced player, so the split is fitted on healthy players (both numbers
mean "if he plays") against what they scored, least squares, shrunk toward
0.5 with 300 player-weeks of weight. Weeks 3-4 of 2026 put it at 0.40 for
ESPN (519 player-weeks) and 0.41 for Yahoo (236), but 0.5 was within noise
of either (RMSE 6.173 vs 6.177 on ESPN) and the weeks disagreed (0.22, 0.53),
so it is refitted every run rather than fixed; the Lineup tab says what it is
and on how many games. The betting market and the FantasyPros projection are
blended on top of this number, with their own fitted weights.

### Are the ranges and P(win) honest?
Checked walk-forward: for each season 2022-2025, the spread and outcome
shape were fit on earlier seasons only, then scored on that season's games
(21,721 player-weeks).

| claimed | actually inside | below | above |
|---|---|---|---|
| 70% range (15th-85th) | **71.1%** | 14.9% | 13.9% |
| 90% range (5th-95th) | **90.4%** | | |

Every season lands within 1.1 points of 70%. By position: RB 73%, TE 72%,
WR 70%, QB 67.5% (QB ranges run slightly narrow: a quarterback's week is a
little less predictable than his level says). By projection level: 69-73%.

P(win) was checked on 3,000 synthetic head-to-heads built from real players
in the same week of 2024-25 (QB, 2 RB, 2 WR, TE, flex each side), scored
with their actual points:

| predicted | 0-20% | 20-35% | 35-50% | 50-65% | 65-80% | 80-100% |
|---|---|---|---|---|---|---|
| mean predicted | 14% | 28% | 43% | 58% | 72% | 86% |
| actually won | 15% | 29% | 42% | 57% | 70% | 84% |

Brier 0.214; predicted spread of the difference 26.6 points against 27.0
actual. Correlation made no measurable difference here, because random
lineups rarely stack; it matters for real rosters that do.

Limits: this validates the model's own projection. The live number also
blends in the platform and the market, whose spread has no history yet (the
projection log will provide it), and kicker/defence ranges come from the
streaming models' own error bands.

### Game-time lottery tickets (Yahoo)
Yahoo lets a bench player be dropped once his game has started, so a spare
roster spot can be rotated: a free agent before each kickoff window
(Thursday, Sunday 1 PM, Sunday 4 PM, Sunday night, Monday), kept if the game
changes his value, otherwise dropped for the next window's ticket. The Yahoo
panel lists the best tickets per window still to come, ranked by the chance
he gives a reason to keep him:

- a **top-N week at his position**, N the number of teams (in a 14-team
  league: QB 17+, RB 16+, WR 18+, TE 10+, measured 2021-2025), read off the
  same outcome distribution as the lineup simulator, injury tag included;
- or, for a back next in line behind a lead back, the job opening up (lead
  backs miss the next game 8.5% of the time).

Only positions where such a week would beat the weakest starter he could
replace are considered, so a backup quarterback behind an established
starter is never a ticket. A merely "startable" week was tried as the bar
and rejected: with two flex spots in a 14-team league it is 6 points for a
back, which is not a reason to keep anyone. The section also names the
cheapest roster spots to open for the rotation. Shown for platforms listed
in `roster.lottery_tickets.platforms` (Yahoo only; ESPN locks players at
kickoff).

### Targets, touches and catch rate: already in, now shown
Fantasy production is opportunity plus talent, and the projection is built
that way: opportunity is nflverse expected points, which prices every target
by depth and field position and every carry by yard line; talent is points
per expected point, which includes catching more than expected. Tested
walk-forward whether the raw ingredients add anything on top (2022-2025,
fit 2022-23, tested 2024-25):

- Trailing targets, carries, target share and weighted opportunity correlate
  with the projection's residual at only -0.07 to -0.09. That is the model
  slightly over-projecting high-volume players, which a plain calibration
  captures equally well. On top of it they moved next-game MAE by at most
  0.015 and made the four-game horizon worse for backs.
- Catch rate over expected, receiving and rushing yards over expected,
  touchdowns over expected: residual correlation about zero.
- Splitting efficiency into non-touchdown skill (light shrinkage) and
  touchdown luck (heavy shrinkage): every setting within noise of the
  single efficiency term (pairwise .7209-.7214 vs .7211).
- For lottery tickets, predicting a top-14 week among waiver-level players:
  accuracy (AUC) .700 / .712 / .715 for RB / WR / TE from the projection
  alone, .697 / .711 / .715 with targets, carries, share and catch rate added.

So the model is unchanged. Usage is shown beside waiver, stash and lottery
picks ("7.3 targets (22% share), 1.0 carries a game over his last 3"),
because it is the evidence a manager wants to see for an opportunity call.

### The red-zone role is opportunity; finishing is not a skill
The case for a real touchdown skill: some players become the red-zone
favourite because of size and ability. Measured 2021-2025 on RB/WR/TE:

| | within a season (odd vs even games) | year to year |
|---|---|---|
| red-zone role: expected TDs a game | r = 0.62-0.75 | r = 0.55-0.68 |
| converting above expected: TDs over expected a game | r = 0.02-0.05 | r = 0.02-0.07 |

Size matters to the role (receiver height vs expected TDs, r = +0.28) but
not to conversion (height vs conversion rate r = -0.23 to +0.06). So the
favourite-in-the-red-zone effect is real, sticky, and already in the fast
opportunity half of the projection -- a target at the 5 is worth far more
expected points than one at midfield -- while the finishing rate is noise
and right to shrink. Expected touchdowns a game now appear in the usage
line beside waiver, stash and lottery picks, so the red-zone role is
visible.

### Streams are ranked by this week's P(win), not projection alone
The D/ST and kicker rankings are league-wide; which one to stream is a
question about *your* matchup. The panel now swaps each free-agent unit
(and your own) into the recommended lineup and scores it against the
opponent on the same correlated simulations. That carries the correlations
the ranking cannot: a defence facing your own quarterback (-0.44) or a
kicker whose offence your defence faces is a hedge -- it narrows your
total's spread, which helps a favourite and hurts an underdog -- and a
defence facing *their* quarterback works the other way. Gaps under half a
point of P(win) are shown as ties, because that is what they are; the
effect is real but usually tenths of a point, so projection still leads.
Waiver D/ST and K moves quote the same P(win) figures. Found while testing:
the same-game check only matched one direction (a's opponent is b's team);
it is now symmetric.

### Pickups for this week, at every position
The stream check covered only D/ST and kickers. Every free agent who could
start for you this week is now tried the same way: added to your roster,
your best lineup rebuilt around him (he has to start to help this week),
and scored against your opponent on draws shared with your lineup as it
stands -- chosen on one sample and scored on a fresh one, as the optimiser
does, so the change is the player and not the luck of the draw. Tried: up
to six per position whose projection comes within three points of the
weakest starter they could replace. Each shows every lineup change he
brings, not just who he starts over, because one addition can reshuffle a
flex.

Winning this week is not the whole question. The drop is the bench
player with the least season value who would not start this week (never
one whose game has kicked off, never one that leaves a position short),
and the season engine then prices the whole move -- this week and the
rest of the season, the drop included -- choosing among the cheapest few
drops. On the ESPN league at week 4 the best pickup (Khalil Shakir, in
for Mike Evans) raised P(win) by 1.2 points, clear of the noise, but
cost 0.4 points of title odds through the drop: the table shows both,
and the hub lists a pickup only when the whole move raises title odds (or,
where the season engine could not price him, by this week's win alone,
and says so). On the Yahoo league no free agent beat a starter.

No odds credits are spent on this. Player props are billed per game, not
per player: one request buys every player's lines in a game, and the
weekly pull is capped at the eight games holding most of your startable
players. A free agent in one of those games carries the market's numbers
for free; one elsewhere is projected by the model alone, as before.
Game lines are one request for the whole slate.

### Which units are actually available
Each tab's D/ST and kicker rankings mark every unit as yours, available or
taken in that league, and dim the taken ones, so the top overall stays
visible and the streamable ones stand out. ESPN syncs every roster, so taken
is explicit. Yahoo syncs only yours and your opponent's; but when its
free-agent list for a position is shorter than a page (25) it is the whole
pool, so a unit missing from it is taken. A full page (kickers, usually)
leaves unlisted units unlabelled rather than guessed.

## Winning the championship

### The objective
Lineups maximise P(win the week). Waivers and add/drops should maximise
P(win the title), which is not a projection comparison. It needs three
things: how each player's outlook could move (speculation), whether a
pickup is worth more to *this* roster than the player he replaces, and
whether the claim is worth the waiver priority (or FAAB) it costs. The
first is built and validated; the season simulator and the move/priority
valuation sit on top of it.

### How a player's outlook can move
Measured on 2021-2025 with the production projection:

- **The projection is a random walk.** Weekly steps have sd 0.8-1.15 by
  position, fat-tailed (kurtosis ~1.8), with **no momentum** (this week's
  step vs last week's: r = 0.00-0.09). Spread over k games grows as
  sqrt(k): 0.92 / 1.33 / 1.87 / 2.48 at 1 / 2 / 4 / 8 games (slight mean
  reversion by 8). A trend is already in the projection; the honest
  forecast is a widening cone around it, not a continued line.
- **Absences** (byes excluded), per game, by projection level
  (<8 / 8-12 / 12-16 / 16+): RB 13 / 9 / 6 / 6%, WR 14 / 8 / 6 / 6%,
  TE 11 / 7 / 9 / 2%, QB 46 / 32 / 12 / 5% (a low-level QB is mostly a
  backup losing the job). Half of absences last one game; a quarter four or
  more.
- **A persistent error.** Simulating only drift, absences and weekly noise
  was too confident: 65% of actual six-game averages inside the 80% band.
  The projection's error persists across weeks (a player is better or
  worse than his number, every week). Adding a persistent error of
  1.5 x sqrt(level) and scaling absences 1.2x for role loss the injury count
  misses -- both fit on 2022-23 -- gave, on 2024-25: **82% inside the 80%
  band, 52% inside the 50% band, 9% below / 9% above, mean 8.75 vs 8.87
  actual**, and 81-83% at every position.

`roster/futures.py` simulates that, week by week, with byes and the next
man up inheriting part of a missing lead's job; the parameters are refit by
`scripts/fit_outcome_model.py` into `outcome_model.json`. A bug caught while
porting: the first fit counted weeks a player played below 5 projected
points as missed games; the played-week set now includes every game.

### Rest of season, graded
Season-long moves -- the drop that turns out to be a breakout, the add who
never plays -- run on the simulated futures, so they were graded the way
they are used. `streamer.roster.ros_backtest` replays weeks 3, 4, 5, 6, 8
and 10 of 2022-2025: every QB/RB/WR/TE on an NFL roster that week (active,
game-day inactive or on a reserve list) gets the status a platform would
have shown (IR, or the week's injury report), is projected by the
production `project_snapshot` on the games before that week, simulated to
week 17 with the real byes and every lead/backup pair, and scored on what
holding him returned: points per team game, a missed game counting zero.
11,848 player-weeks.

What it found, before any change:

| Season value | Simulated | Actual | |
|---|---|---|---|
| under 4 | 2.15 | 1.44 | too high |
| 4-7 (the waiver tier) | 4.11 | 3.37 | too high |
| 14-18 | 11.86 | 12.83 | too low |
| 18+ | 15.72 | 16.65 | too low |

and 16% of players finished below the simulated 10th percentile. Waiver
fodder looked better than it was and stars worse -- the bias that adds bad
players and lets good ones go. Taken apart:

- **Reserve lists.** 63% of players on IR at a checkpoint never played
  again that season (50% of those just placed there, 80% after six games
  out; 70% below 6 points a game, 43% above 12); the ones who came back
  took a median four games. The simulator brought nearly all of them back
  after about four: 2.5 points a team game against 1.1 real.
- **Availability by level.** Healthy stars played 90% of their remaining
  games; the simulator said 78%. Fringe players played 55%; it said 68%.
  The absence hazard was looked up by the *hidden true level*, which
  carries a persistent error of about +-6.7 points for a 20-point player --
  so in some seasons Josh Allen was simulated as a backup and missed games
  at a backup's 46% a week (his 10th percentile was 3 points a game).
- **Points per game played.** Zero floors on levels and scores lifted a
  low player's simulated average (a 4-7 point back: 6.2 simulated against
  5.4 projected and 5.1 real); stars' outlooks were too wide.

What changed (`scripts/fit_ros.py`, each piece fitted on two seasons and
graded on the other two, then the reverse):

- a measured reserve-list model: the chance of never coming back by season
  value and games already missed, and the wait for those who do;
- the hazard reads the projection (where it has drifted to), with a
  multiplier by level fitted to how often players actually played (x2.1
  below 4 points, x0.7-0.9 above it);
- the persistent error is multiplicative (mean-preserving, cannot go below
  zero), scaled by position and level so the middle half of the simulated
  range holds half the outcomes (x2 for a fringe back, x0.6 for a star
  receiver);
- an offset by position and level so simulated points per game played
  match what players like him scored.

| Held out | 2024-25 before | after | 2022-23 before | after |
|---|---|---|---|---|
| average miss, points a team game | 2.76 | **2.48** | 2.64 | **2.40** |
| bias | +0.34 | -0.17 | +0.45 | +0.08 |
| below the 10th percentile (10%) | 17% | 10% | 17% | 11% |
| above the 90th (10%) | 6% | 12% | 5% | 9% |
| inside the middle half (50%) | 49% | 48% | 50% | 49% |
| players on IR, bias | +1.42 | -0.09 | +1.27 | -0.25 |

Breakouts in the waiver tier (season value under 10, not on IR) -- the
chance the simulation gave against how often it happened:

| | before | after | happened |
|---|---|---|---|
| RB averages 12+ | 5.3% | 4.3% | 4.2% |
| WR averages 10+ | 8.7% | 5.9% | 5.1% |
| WR averages 12+ | 4.8% | 3.2% | 2.1% |
| TE averages 8+ | 11.1% | 10.6% | 9.2% |

Receiver upside was overstated by more than half and is now much closer;
which players break out is ranked as well as before (AUC 0.84-0.90). By
segment: backups whose lead is out now went from +0.84 to +0.44 points a
team game too high (the bands now hold: 8% below, 10% above), rookies from
+0.45 to -0.15, everyone else from +0.26 to +0.02.

Tried and left out: pulling high season values toward the position mean.
Stars do score below their number on average (about 1.3 a game above 18),
but out of sample the shrink lowered the miss only for receivers above 12
(2.99 to 2.92) and raised it for every other position -- the error at the
top is a few collapses, not a uniform overstatement.

Still imperfect: what is left by level changes sign from one pair of
seasons to the other (season-to-season variation, not a fixable lean);
backups whose lead is out still run about 0.4 high; backup quarterbacks
who become starters are under-forecast (0.9% simulated, 2.4% real, for
14+). Players with no NFL games yet are not in the replay -- the platform's
projection stands in for them on the page. Rerun
`python scripts/fit_ros.py` after a season is added.

### Form that fades, and a read that lags
The hub talked the ESPN team into three quarterbacks. Drake Maye was the
only one through week 3. Then Jacoby Brissett and Jordan Love were
recommended as close-call adds (+0.5 to +0.9 points of title odds each) and
both were claimed. The simulation said the three were worth 21.2 points a
week in the QB slot, against 17.9 for Brissett alone with a waiver-wire
quarterback covering his absences. That is a 3.3-point weekly edge from
holding QBs you mostly do not start.

That was checked against what happened. In 2022-2025, from weeks 3-6, every
trio of healthy quarterbacks within two points of each other (14.5-18.5 a
game; 2,037 trios) was played out week by week. Each week the manager
started whichever had the best mix of his projection and his points since,
and a wire QB (14.9) when none could play. Then the same trios were run
through the simulator:

| QB slot, points a week | Real | Simulated (before) |
|---|---|---|
| second QB adds | +0.57 (+0.33 covering absences, +0.24 picking) | +2.05 (+0.28, +1.77) |
| third QB adds | +0.29 | +1.56 |

Covering for an absent starter was right; picking was not. Simulated
managers picked the better of two near-identical quarterbacks far better
than real ones did. Following our own production projections instead of
points-so-far, real managers did worse still (+0.48 and +0.20). Taken apart
on all positions, it was three things:

- **Form lasted too long.** A player's points per game over his next four
  games, against his season value, said this much about the games after
  them: QB 0.33 real, 0.75 simulated; RB 0.46 / 0.65; WR 0.32 / 0.49; TE
  0.29 / 0.57. Per-game levels were also too spread out: half of real QBs
  should land in the middle half of the simulated points per game played,
  and 66% did. The per-player error had been fitted to the spread of
  rest-of-season points per *team* game, which mixes in absences. A level
  error widened to cover them also made form permanent.
- **The read was too clean.** Simulated managers learned how wrong a
  projection was from games without the luck of those games. Now a read
  after k games carries k / (k + 4) of what the games showed, luck
  included.
- **Simulated managers saw too much.** When our production projection for
  a player moved between checkpoints three to five weeks apart, the move
  showed up in what he scored afterwards only in part: QB 0.35 of it, RB
  0.94, WR 0.63, TE 0.66. The moves themselves were smaller than the
  simulator's: QB 2.9 points against 4.1, RB 2.4 / 3.8, WR 1.9 / 3.0, TE
  1.6 / 2.7. A real projection is slow and partial, and it moves on noise.

`scripts/fit_ros.py` now fits four things together (each fitted on two
seasons, graded on the other two). It already fitted availability by level
and points per game played by position and level. The per-player error now
targets the spread of points per game played among players who kept
playing. Two new dials are added:

- **Visibility.** The share of a player's real change (drift, and what
  games revealed) a manager's read catches. Fitted: QB 0.33, RB 0.46,
  WR 0.32, TE 0.34.
- **Projection noise.** Moves in the read that are not moves in the player,
  fading at the projection's own 2.5-game half-life. Fitted: QB 1.7, RB
  0.9, WR 1.0, TE 0.9 a week.

A simulated move is visibility times the real change plus the noise, so
the move's size and its share that holds up give both dials directly.

Held out, against the previous model:

| | Real | Before | After |
|---|---|---|---|
| early form carries on, QB / RB / WR / TE | 0.23-0.46 / 0.45-0.47 / 0.26-0.40 / 0.23-0.33 | 0.75 / 0.65 / 0.49 / 0.57 | 0.32-0.44 / 0.49-0.52 / 0.33-0.41 / 0.31-0.44 |
| projection move, size QB / RB / WR / TE | 2.5-3.1 / 2.4 / 1.8-1.9 / 1.6-1.7 | 4.0-4.2 / 3.7-3.8 / 3.0 / 2.7 | 2.7-3.0 / 2.3-2.7 / 1.8-1.9 / 1.6-1.7 |
| points per game in the middle half (50%) | | 50-72% | 43-57% |
| QB slot: second / third QB adds | +0.57 / +0.29 | +2.05 / +1.56 | +0.60 / +0.25 |
| waiver-tier breakouts, WR 12+ a game | 2.1% | 3.3% | 2.3% |

The average miss on rest-of-season points per team game is unchanged
(2.48 and 2.41 held out). One thing got worse in appearance: 73-76% of
rest-of-season outcomes now fall inside the 80% band, against 78-79%, and
the misses are low. Of the 15% below the 10th percentile, 11 points are
players who played less than half of their remaining games. The simulator
loses the right share of half-seasons overall (WR 10% against 10%, QB 34%
against 37%), but not to the right players: who is about to lose his job
or his season is not in its inputs. The old, wider error hid that by
widening every band, at the cost of the persistence above. Telling those
players apart is the next thing to model.

On the ESPN roster, the third quarterback is now worth 0.13 points of title
odds. The QB slot gets 17.8 a week from all three against 17.0 from
Brissett alone, and no waiver add is a quarterback. Title odds as a whole
fell with the change. A long shot's title rests on upside, and upside
that lasts is rarer than the simulator thought.

### A back whose work has collapsed
Tyrone Tracy Jr. topped the ESPN waiver list in week 4 at 7.6 a game: the
Giants' lead back late in 2025 (13-27 expected points a game), he had 2, 0
and 1 carries in 2026 and was the No. 3 behind Skattebo and Najee Harris.
Three games are three of the seventeen the season value leans on, so it read
a demotion as a dip. In the replay, backs valued 4+ whose last two games
kept under half their earlier opportunity (by 3+ expected points a game)
returned about 86% of what the model gave them -- 0.82-0.93 in every
held-out season -- while receivers, tight ends and quarterbacks in the same
spot showed no such lean (and backs below 4 swung both ways). Those backs'
season values are now cut to 86% (`roster.usage_collapse`), unless the
drop is an injury the simulation already plays out; the card says so.
Re-run through the replay, the corrected backs are simulated at 4.07 points
a team game against 4.17 real (5.0 before). Tracy goes from 7.6 to 6.5.

### The real depth chart decides who is next in line
The lead at QB, RB and TE is still the player with the most season value;
who comes *next* -- the handcuff the simulation hands the job to if the
lead goes down -- now follows Sleeper's depth chart where it has the team's
room (matched by name, team and position: its ids for the other platforms
are mostly blank), and value otherwise. The model ranked the Giants' backs
Skattebo, Singletary, Tracy, Harris; Sleeper has Harris second. Sleeper
lists an injured starter below his backup (Braelon Allen first, Breece
Hall second), which is why the lead stays the lead by value.

When the lead is *already* out, the backup standing in is still the one
getting the work: the share he inherits was measured on snaps, and a depth
chart can lag them. The first version let the depth chart choose there
too, and moved Miami's job from Ollie Gordon II (8.3 expected points a game
over his last two, with Achane on IR) to Jaylen Wright (8% of the other
snaps) because Sleeper listed Wright first -- Gordon's value fell from 9.4
to 5.8 and the page told you to drop him. That backup now stays next in
line ahead of the depth chart. The depth chart cannot be replayed --
Sleeper keeps no history -- so the replay runs without it.

### The season simulator
Every rostered player (and the top free agents) gets simulated futures;
each simulated season, every team starts the best lineup it could *see*
that week, the regular season is played on the league's real schedule on
top of current records (median game if played), seeds go by record with
the league's tiebreak, and the playoffs are the fixed bracket with byes and
multi-week rounds. On the real ESPN league (10 teams, 14-week season, 6-team
playoffs) 6,000 seasons take about 6 seconds; simulated weekly team scores
average 122 against 124 in real games so far.

Three corrections found by running it on the real league, each of which
would have made depth look far more valuable than it is:

- **Replacement level.** With rosters frozen, a team whose only QB was on
  bye or hurt scored zero at QB, so every backup QB looked like +2 points
  of title odds. Real managers pick up a replacement. A slot whose best
  rostered option falls below the second-best free agent at the position
  (the best D/ST or kicker) is now filled at that level.
- **What managers can see.** The first version let every manager know each
  player's true level from week 1, so they always started the right one.
  The simulation now keeps a visible level -- the projection plus what the
  games have revealed, learning half of its error in ~4 games -- and
  lineups and claims are made on that, while scores come from the truth.
  The six-game calibration (82% in the 80% band) is of scores, so it is
  unchanged.
- **Locks are about this week.** Players whose game had kicked off were
  excluded as drops; waivers process after the games, so they are not.

### Waiver moves in title odds
For each top free agent (per-position quotas, so backup quarterbacks cannot
crowd out backs and receivers) and each plausible drop, on the same
simulated seasons:

- **now**: P(title) adding him and dropping the other from now on;
- **wait**: P(title) holding, and claiming him the week after he breaks out
  (his visible level beats the drop's by 2 and his own start by 2) -- won
  only if your priority beats every rival who also claims;
- **priority cost**: on a rolling waiver order, claiming sends you to the
  back; the cost is the most valuable future wait-and-claim among the other
  top free agents, at your rank against last.

Rival claims: each rival claims a breakout with probability (share of teams
averaging a pickup a week) x 0.35 -- about 2-3 claims per breakout in an
active ten-team league. The 0.35 is an assumption, stated on the page:
failed claims are not published, so it cannot be fitted. Verdicts: **claim
now** when now beats max(wait, standing pat) by more than the priority cost
and by twice the paired simulation noise; **close call** when the edge is
positive but inside the noise; **worth adding, but wait** when waiting is as
good. On the real league at week 3 the best move was Hunter Henry for
Kaelon Black: +1.0 points of title odds now, +0.4 waiting, priority worth
0.3 -- an edge of +0.35 against noise of +-0.40, a close call.

Not yet modelled: rivals getting stronger when they land a breakout you
passed on; FAAB bidding (neither league uses it); trades.

A toss-up never includes a drop priced below standing pat: inside the noise of
the best drop is not a fair alternative to it if it lowers your odds (a card
once offered "Gadsden (12.2%) or Higbee (11.7%)" against 12.0% standing pat).

### Behind in the standings: upside, and the games that matter
A team behind needs seasons where something breaks right, so a player's
spread is worth more to it than to a leader. No multiplier is bolted on for
that: waiver moves are priced in P(title), which already counts only the
seasons that end in a title, so it weighs upside by how far behind you are
-- more at 5% than at 25%. What was missing was the upside itself:

- **Rookies' futures were drawn like veterans'.** Their projections move
  23% more over a month (second-year players 4%), measured with the drift
  in `outcome_model.json`. The simulated walk is now widened by that
  factor. Their expected rise is already in today's projection, so the
  mean is left alone.
- **Upside plays never reached the list.** Candidates were the best free
  agents by average plus spread, so a back one injury from a lead role,
  whose average is low, was never priced. Up to three of each kind are now
  priced on purpose -- next-man-up backs behind a healthy lead, rookies,
  players whose opportunity just rose -- and each card says which kind.

**Must-win weeks.** For each remaining game, every simulated season is
replayed with that one result forced to a win and then a loss, everything
else held as it fell: the swing is what that result is worth to your
playoff and title odds, including what it does to the opponent's record,
so a game against a rival for the same spots swings more. A week is
flagged when its swing is at least 1.25 times the average and 10 points.
With it, the playoff odds by final win total show the record that gets in.
In the ESPN league at week 3 (0-2, about to be 0-3) no game stood out --
each was worth about 14 points of playoff odds -- and the record told the
story: 6 wins got in 29% of the time, 7 wins 87%, on a pace of 5.4.

Pricing the extra candidates is kept affordable: the priority cost is
computed once rather than per candidate, and each candidate is priced
against his four likeliest drops rather than six.

### When drops tie, keep the player others still value
A waiver plan (Meyers, Brissett, Sadiq) dropped Xavier Worthy and Brian
Thomas Jr., both widely rostered. On our projections they were the
weakest receivers, but pricing the alternatives showed the choice of the
last drop was a coin flip for the title: dropping Terrance Ferguson instead
of Thomas Jr. came out at 7.00% against 6.80% (noise +-0.26). The proxy
that picks drop-sets had broken the tie by our projection alone, ignoring
that a player the market still values (Thomas Jr. 9.7 a game, Ferguson 6.3)
can be traded -- and that cutting him hands him to a rival for nothing.

Now, for single moves and plans, drops within 0.2 points of title odds of
the best are treated as ties, and the tie goes to cutting the player the
market (the perception model, consensus included) values least. It is a
tie-break, not a trade-off: a drop that costs title odds is never swapped
in for asset value -- Kenny Gainwell, valued less by the market than either
receiver but more by us, would cost 0.8 points to cut, and stays. The plan
card says when it kept someone and why.

**Superseded: toss-ups are shown, not decided.** The tie-break then cut Kyle
Pitts, and when that was priced directly, keeping him and dropping Worthy
came out slightly ahead (4.62% against 4.27%, noise +-0.32). A fixed
0.2-point tie band is narrower than the simulation's noise, so which way it
fell was luck. Now a drop is a toss-up when the paired difference in title
odds is within two standard errors: the simulation cannot tell the options
apart. All of them are shown with their title odds ("drop Worthy (4.6%) or
Pitts (4.3%)"), and the card adds what the market sees each at, for a
manager picking by feel (a name that can still be traded, a hunch about a
role). Single moves list the best first. A plan changes the drop the
screening chose only for one clearly better, beyond the noise: taking the
luckiest of eight noisy estimates would have overstated the plan (it
turned a +0.8 plan into +1.6). Each alternative is offered once, at the
step where it is worth most. At week 4 in the ESPN league most drops are
toss-ups between two to four players, which is the honest answer: at that
level of detail the season is not predictable.

### Your roster, valued on the same seasons
The Roster tab lists every player you have with four numbers. Season value
(points a game when he plays) and points a week -- his simulated average per
remaining week once injuries, byes, a lost job and the next man up have
played out, with the 10th-90th percentile of that average, his floor and his
upside. Then his **title worth**: how many points of P(title) you lose
without him, his spot filled from the wire, on the same simulated seasons
the hub prices every move on. That is the number to compare across
positions -- a kicker or a third quarterback who projects well is worth next
to nothing to a title, because the wire or the bench covers him -- and the
cost of dropping or trading him. Last, what other managers see him worth (the
trade model's view): far above ours is a player worth more traded than
held, far below a player worth more to you than in any offer. A worth inside
twice its standard error is shown grey.

Title worth also decides which drops get priced. A waiver add used to try
the four players with the lowest season value plus spread; a third
quarterback at 15.7 a game was never one of them, though he barely moves
your title odds. Drops are now tried cheapest to the title first, with the
cheapest at the add's own position always among them (the one he replaces,
whose loss he covers), and the simulation still decides between them. The
same order picks this week's pickup drops, the drops a waiver plan draws
from, the drop that prices waiver priority, and the drop watch. On the
week-4 rosters it changed no recommendation -- the drops it newly priced
lost to the ones already chosen -- but it is now the title, not a proxy for
it, that decides who is even considered.

Cheap to your title is not the same as worthless. On the live week-4 page
Josh Jacobs -- three games missed, his work halved -- was the third
cheapest on the ESPN roster at +0.6 points, while the market still saw him
at 14.1 a game and the hub's best move was trading him. A drop candidate
the market values a point or more above us is marked "trade, don't drop",
in the drop watch and on the Roster tab alike.

### A player gone for more than a season is not injured
Brandon Aiyuk was recommended as a pickup. He last played in October 2024
and is on the 49ers' reserve/left-squad list, but ESPN tags him OUT, and a
player with an injury tag kept his history's value (11.9 a game) and was
simulated as a short absence. The rule was right for a player hurt a few
weeks ago and wrong for a holdout, the reserve/left-squad list or a long
suspension. Now a player who has not played in more than a season (18+
weeks) has no rest-of-season value unless his platform projects him this
week, with a note saying so. A player hurt partway through last season
keeps his value. Beyond Aiyuk this zeroed Joe Mixon, Tank Dell and James
Conner (all tagged, none having played in over a season); the rest of the
players it touches had no value already.

### Grading the projections, and the FantasyPros consensus
"How good are the projections?" deserves a running answer, not an
impression. Every refresh already logs, before kickoff, our model's number,
our final number (model, platform and betting market blended), the
platform's projection and the betting props for each player. The scorecard
joins that log to what the players scored and grades every source two ways,
each on the players it covered with our final number graded on the same
players beside it:

- **start/sit** -- of any two players at the same position, did it rank the
  one who scored more higher (rankings can take this test too);
- **average miss** in points, for sources that give points.

Week 3 of 2026, first grade: in the ESPN league our final number went
73.0% on start/sit against ESPN's 72.3% on the same 217 players, missing by
4.68 against 4.83; in the Yahoo league 69.3% against Yahoo's 67.8% (76
players). Betting props were level with our final number on the players
they priced. One week is noise; the table on the page accumulates.

(An earlier quick check that had ESPN slightly ahead graded the wrong
column -- our number before the market blend -- and only projections logged
before the 1pm games, so late games were graded on stale numbers.)

**FantasyPros** (with the `FANTASYPROS_API_KEY` secret) adds three things.
Their rest-of-season consensus rank, this week's rank and this week's
projection are attached to players at publish time:

- **graded** in the same scorecard (weekly), and our season values are
  graded against their rest-of-season ranks by rank correlation with the
  points a game each player scored since, once three weeks have passed;
- **folded into the market view** used for trade acceptance -- managers
  read these rankings -- as the market value of a player's consensus rank,
  at a provisional half weight until `scripts/fit_perception.py` measures
  it against Yahoo's rostership;
- **flagged** on hub rows and waiver cards where the consensus is well off
  our view (5+ ranks and 30%), a second opinion before acting.

Their data is theirs: it is cached on the runner only
(`data/raw/fantasypros/`, restored between runs, ignored by git), never
committed, and the public page shows only analysis derived from it --
grades and "consensus is much lower on him" -- credited to FantasyPros.
Their ranks and projections themselves are never displayed: under their
API licence (personal, non-commercial use; August 2026 guidance) that is
redistribution, which requires a Commercial agreement, while published
analysis based on the data is allowed with conspicuous credit. There is no
switch to show them, so it cannot happen by accident. The probe
(`streamer fantasypros-probe`, or the workflow's `fantasypros-probe` job)
prints status, field names and counts, never values or the key.

### The next man up, speculated on
Waiver pickups are bets on a change in opportunity. Braelon Allen was #1 on
RotoBaller's and FantasyPros' lists the week Breece Hall hurt his quad
(week-to-week, maybe a game or two, maybe more). The tool did not rank him at
all, and the comparison turned up three problems.

1. **He was never priced.** The title engine considered the best free
   agents by season value, plus handcuffs behind healthy starters. A backup
   starting now because the lead is out fits neither: his season value
   assumes the lead comes back. Free agents stepping in for an absent lead
   are now always priced (two per position; the reason reads "stepping in:
   Breece Hall is not expected to play").
2. **Backups take more of the job than the model gave them.** From every
   2021-2025 absence of a lead (the team plays, he does not), the share of
   the gap in opportunity that best predicts the next man up's first game,
   each season held out in turn: RB 0.55, QB 0.60, TE 0.35 (it had been
   0.35, 0.30, 0.10). Every held-out season picked about the same share, and
   the old values had left these players about 2 points a game
   under-projected (RB bias +2.1 -> +0.4, QB +2.2 -> +0.2, TE +2.0 -> +0.2).
3. **How much of the job he gets is itself a gamble, and it can stick.**
   The share varies a lot from one absence to the next, beyond game-to-game
   noise (spread 0.8 at RB): sometimes he becomes the bell cow, sometimes a
   third back gets the work. When the lead returns, the backup kept about a
   quarter of what he took over (pooled, 99 returns). Backups who took 80%+
   of the gap kept half or more of it a third of the time, the Rico
   Dowdle case. Leads came back at 90% of their volume.

The season simulation now plays this out. The lead's absence is drawn from
the measured lengths (half last one game, a quarter four or more), so how
long Hall is out stays uncertain. The backup's share is drawn fresh for each
absence. When the lead returns, the backup keeps a quarter of his takeover
and the lead loses it. The part of the backup's season value that came from
the absence is taken out of his starting level, because the simulation now
plays the absence out week by week and would otherwise count it twice.
What it is worth still depends on your lineup. For the ESPN team, Allen at
about 9.5 a game while Hall is out competes with your flex options, so he
priced about even. A team with a hole at RB would see it differently.

### Who else is in the backfield
Every backup was expected to take the same share of an absent lead's work
(0.55 at RB, with a wide spread). The tool read box scores and platform
tags, and a back who never touches the ball has no row in a box score. It
could not see that Isaiah Davis had not played an offensive snap for the
Jets, which left Braelon Allen as the only other back.

Snap counts can see it. They come free from nflverse (Pro Football
Reference), the same provider as the box scores, and cost no API credits.
The next man up's **concentration** is his share of the position's offensive
snaps that did not go to the lead, over the team's last four games. On
2021-2025 running back absences (93 with snap data, each season held out in
turn), predicting his first game as `(volume + share x gap) x efficiency`
with `share = 0.62 + 1.38 x (concentration - 0.66)` beat one share for
every backup: RMSE 7.93 -> 7.46, better in all five seasons. Over whole
absences, backups with 75%+ of the other snaps took 0.80 of the gap, and
those under 50% took 0.33. The share is kept between 0.1 and 0.9.

Tight ends did not pass. The relationship ran backwards (the second tight
end with the most snaps is often a blocker) and held in only 3 of 5
seasons, so they keep the one position share. A position is kept only when
it is better overall and worse in at most one season
(`scripts/fit_next_man_up.py` -> `next_man_up.json`). A backup whose name
the snap data does not have is treated as unknown, not as a back who never
plays.

The concentration sets his share in this week's projection and his mean
share in the season simulation (where it is still drawn per absence, and
he keeps a quarter of it once the lead is back). Allen, with 100% of the
Jets' other RB snaps, now projects 12.5 this week against 11.1 (ESPN has
12.8). Ollie Gordon, with 92% of Miami's, is worth 12.0 a game for the
season as the lead back once Achane is out for the year. The card says it:
"next man up: Breece Hall is not expected to play, and 100% of his team's
other RB snaps have been his".

Sleeper's free player database (no key, about once a day) has each team's
current depth-chart order and injury details. It is next, once a probe
(`streamer sleeper-probe`, the workflow's `sleeper-probe` job) confirms
what it returns: this sandbox cannot reach it, but the runner can.

### This week plays at this week's numbers
The season simulation valued every week, the one in progress included, at
each player's season level. That undersold exactly the pickup a waiver
list is built around: with Breece Hall and Josh Jacobs both out, Braelon
Allen projects 11.1 this week and would start for the ESPN team over Kenny
Gainwell (7.7), but the simulation had him at 9.5 against Gainwell's 8.4.
The week in progress now plays at each player's projection for it --
matchup, the platform's and the betting market's numbers, the chance he
sits (the simulation flips that coin itself). Later weeks keep the season
level. For a next man up, the share of the lead's work the simulation
already expects of him that week is netted out, so it is not counted twice.

Allen still priced at about +0.1 for that team. Starting one week at RB2 is
worth about three points in a week worth about a point of title odds per
win, and once Hall is back he does not start there. The upside plays that
were priced and fell short are now listed under the waiver moves with what
each adds ("also priced"), so a name on every waiver list can be seen to
have been considered, and what he would do for this roster.

### An absence that has already run keeps running
Josh Jacobs has not played this season (the commissioner's exempt list).
ESPN tags him day-to-day and projects zero. The simulation drew his absence
fresh from the measured lengths, so he was back next week 45% of the time,
and the ESPN team's backfield looked healthier than it is. Every player now
carries how many of his team's games in a row he has missed, and an absence
in progress draws its remaining length from the same measured distribution
conditioned on having lasted that long. Three games in, an absence ends the
next week about 31% of the time instead of 36%, and runs about 3.4 more
games on average instead of 2.8. The effect is modest because the measured
lengths are not very heavy-tailed. A suspension or an exempt list with no
set end is probably longer than an injury of the same age, but nothing in
the feeds tells them apart, so a note says how long he has been out.

### Claim order spends priority where you would lose the player
Plans listed claims in order of each pickup's own value, with Jacoby
Brissett first. But on a rolling list the first successful claim sends you
to the back, and every claim after it is processed from there: you get
those players only if nobody else claims them. Priority is worth most on
the player someone else would take. A backup quarterback in a one-QB league
mostly clears without it.

Each rival now claims a free agent with a chance that depends on need and
news, scaled by how active he has been:
- he would start for them (this week or on season value): 50%;
- he is starting for an injured lead, and on every waiver list: 25%;
- otherwise: 5%.

These rates are assumptions, like the breakout-claim rate: failed claims
are never published, so they cannot be fitted. The first claim goes to the
pickup whose value times (chance someone claims him minus chance someone
ahead of you does) is largest. The rest follow by value times the chance
nobody else claims them. In the ESPN league at #3 of 10: Sadiq would start
for 6 of 9 rivals (someone claims him 97%, someone ahead of you 43%), so he
goes first; Brissett would start for 1 (63%, 5%). Single-move cards now
give the chance another manager claims him. Plans show what to claim, in
order, and what to drop as a set ("Thomas Jr., Worthy and one of Ferguson
(5.7%) or Shakir (5.6%)"), because which add is paired with which drop does
not change the roster that results. A plan's percentages are the whole
plan's title odds; a single move's card prices that move alone.

Two corrections after the first run. "Would start for them" had counted
any edge at all: Brissett at 17.0 against Jalen Hurts at 16.9 made Team
Morris a likely claimant, and his claim chance came out at 63%. A claim now
needs a clear upgrade of 1.5+ points a game, and Brissett drops to 30%. The
order also leaned on each pickup's value on its own, which carries the
simulation's noise (+-0.3 points). In one run that noise put Meyers or
Brissett first, when what priority is worth on each was 0.15-0.19 points
either way. Every add in a plan helps, so each value is taken halfway to
the plan's average, and in a close call the contested player goes first:
Sadiq (87% claimed, 34% by someone ahead of you) before Brissett (30%, 5%).
The card shows what priority is worth on each.

### A season-ending injury is not a long absence
De'Von Achane tore his ACL and was a free agent in the ESPN league, still
worth 13.7 a game to the model: ESPN tags a season-ending injury and a
four-week one the same way (IR), and the simulation brought him back after a
few games. Nothing in the platforms' feeds says "season over", but the
FantasyPros rest-of-season rankings do: experts stop ranking a player who
will not play again this season and keep ranking one who will be back. So
an injured player on a long-term tag whose value would put him in the top
half of the players ranked at his position, and whom the consensus does not
rank, is now out for the season: no value, with a note. It only applies
when the consensus is seen ranking some injured players (two or more, a
quarter of those judged), so a feed that leaves out every injured player
cannot zero them all. Players on shorter tags keep the simulated absence,
whose length stays a draw.

### A new team is a new role
Zach Ertz was the top waiver add in the ESPN league and on nobody's list.
The Eagles signed him to their practice squad as a stopgap for Dallas
Goedert; in his one game he had 2 targets, and ESPN projected 3.4 points.
We had him at 8.9 a game, because his volume came from his last 17 games,
16 of them as Washington's starter. A player's role belongs to his team,
and the model did not know he had changed teams.

On 2022-2025, players with games for another team in their window (2,580
player-weeks projected 5+) were projected 0.9 a game too high; players who
had not moved were on target (+0.1). The miss was largest before they had
played for the new team: 1.6 with no games yet, 2.2 for a mid-season
signing. So their old-team games now count less toward their volume: at a
fifth of the weight, the best in every held-out season, average miss went
from 3.13 to 2.99, better in every group by games played for the new team.
Efficiency (points per opportunity) is the player's own and keeps every
game. The platform's current team overrides the box scores, so a player who
has just signed is caught before his first game. Ertz now projects 6.5, and
the note says why. Movers are still a little high after the change (about
0.5 a game): some move because they are declining, which the discount does
not try to read.

### A teammate back from an absence
Asked about a trade for Jaylen Warren: Rico Dowdle had missed weeks 3-4,
Warren's carries had gone from 10-11 a game to 17, and Dowdle was due
back. The volume model reads a player's own games, the recent ones most;
it did not know who else was on the field. The next man up covered one
direction -- a lead out now, his backup inherits part of the gap -- but not
the other: when the man comes back he takes his work back, and the games
his teammates had without him kept them valued as if he were still out.

On 2022-2025, against players of the same projected volume, players whose
last four games included one a regular teammate at their position (5+
expected points a game) missed this season, and who was playing again:

- a player who had stood in for the bigger role -- the next man up handing
  the job back -- was projected too high by 1.2 a game at RB (+-0.2), 0.8
  at TE, 0.3 at WR;
- a lead whose second back returned, 0.4 (+-0.35), not clearly different
  from zero: the 17-game anchor and the 2.5-game half-life already fade a
  two-game bump. Leads whose work rose 2+ expected points while the other
  man was out ran 0.8-1.0 high, on few cases.

What a teammate absorbs in the first game of an absence: the busiest
remaining back takes 0.52 of the gap when the lead is out (the next man
up's 0.55), and the lead takes 0.32 of a missing second back's volume --
about 2 expected points a game, which nothing priced. Receivers spread a
missing receiver's targets too thinly to see in any one of them.

So those games are now read less what he was expected to absorb in them:
`bigger` of the gap when the man coming back had the bigger role (measured
from the player's volume in the games they shared), `smaller` of the
returning man's volume when he had the smaller one, scaled by the returning
man's chance of playing -- this week from his tag, the rest of the season
nearly all of it (`roster.teammate_back`). Fitted walk-forward
(`scripts/fit_teammate_back.py`), each position's pair picked on three
seasons and scored on the fourth, on the player-weeks it moves:

| | bigger | smaller | next-week RMSE | bias |
|---|---|---|---|---|
| RB | 0.65 | 0.10 | 6.14 -> 6.06 | -0.67 -> -0.07 |
| WR | 0.35 | 0 | 6.61 -> 6.58 | -0.17 -> +0.11 |
| TE | 0.35 | 0.10 | 5.13 -> 5.09 | -0.35 -> -0.06 |

each better in all four seasons. Replayed through the season checkpoints
(weeks 3-10, real injury tags), against rest-of-season points per game
played: RB bias -0.75 -> -0.18 (RMSE 3.32 -> 3.20), WR -0.55 -> -0.11
(3.05 -> 2.97), TE -0.23 -> +0.12 (2.52 -> 2.48), every season better at
every position; for the players cut 0.3+ -- the ones whose card says so --
-0.80 -> -0.03. TE first fitted at 0.5 overshot in the replay (+0.22) and
was set to 0.35, which was better in every season in both tests.

Two guards. Quarterbacks are left out: a backup who does not play leaves no
box score, so his absence cannot be told from a seat on the bench. And only
a teammate who has played for the team this season counts: Salvon Ahmed,
last seen in 2023 and listed by Miami without a tag, was read as a back
coming back to take Ollie Gordon II's work.

Warren goes from 13.6 to 13.3 a game (Dowdle is questionable, so this week
counts 72% of it). Davante Adams, who drew 10 and 13 targets while Puka
Nacua missed weeks 2-3 and 6 and 9 with him, goes from 16.6 to 16.3.
Michael Mayer, with Brock Bowers back, from 8.6 to 7.9. In the season
simulation a lead still gains nothing when his second back misses a future
game (only a backup inherits from a lead); his season value, measured on
games that include his partner's usual absences, carries that on average,
and adding it did not hold up (below).

### The next man up, later in an absence: tried, not adopted
Once a backup has three or more games in the job, his own recent games carry
the takeover and the inherited share is added on top: on 2022-2025 (2021
included for the fit), running backs in the fourth game of an absence or later
were projected 1.6 a game too high, while in games one and two they were
still 1.4 and 0.5 too low. Three fixes were tried on the same absences:

- the share taken from his volume before the absence began (0.55-0.8 of
  that gap): bias by game evened out, but the error grew (RMSE 7.90 ->
  8.04-8.23 at RB, 4.83 -> 5.07-5.41 at TE) -- what he has actually been
  given since the lead went down says who got the job, and throwing it away
  costs more than the double count;
- the larger of that and his current volume: also worse (7.94, 4.97);
- a share by game of the absence, fitted on four seasons and scored on the
  fifth (RB about 0.6, 0.4, then 0): RB 7.90 -> 7.79 overall but better in
  only 2 of 5 seasons; TE worse (4.83 -> 4.89).

None passed the rule used everywhere else (better overall, worse in at most
one season), so the next man up stays as it is. About 60 backs a season
reach a fourth game in the job; revisit with more seasons.

### A team's targets are shared
Each receiver's volume is projected on its own, so nothing kept a group's
projections adding up to what the team actually throws them: a receiver out
left his targets nowhere, and a group of healthy receivers could be projected
well past the team's usual total. For each team and position, the usual total
is the mean of the group's expected points over the team's last 8 games
(last season's games fill in early on); this week's projected total is the
sum of the projected volumes of the players expected to play (in the team's
last game, or listed by the league and back from a miss, each at his chance of
playing). Each player's volume is scaled by (usual / projected), the ratio
capped at +-30%, to a power.

Walk-forward on 2022-2025, weeks 4-18, the players who played:

| Position | Power | RMSE | Seasons better |
|---|---|---|---|
| WR | 0.3 | 6.034 -> 6.016 | 4/4 |
| WR | 0.5 | 6.034 -> 6.016 | 4/4 |
| WR | 0.75 | 6.034 -> 6.031 | 1/4 |
| TE | 0.5 | 4.996 -> 5.012 | 1/4 |
| RB | 0.5 | 6.161 -> 6.107 | 4/4 |

Receivers take it. The 0.3 power is used: the same gain as 0.5, but no
receiver moves more than about 10% (against 16%), and 13% move 8% or more
(against 31%). Counting only this season's games once four are in, or
weighting last season's at half, changed nothing.

Tight ends were worse: one tight end's role is not the next one's.

Backs were then tested on top of the next man up, which already hands a
missing lead's work to the backup -- the worry was counting it twice. Same
replay, 5,033 RB player-weeks, the next man up applied as production does
(171 team-weeks with a lead out), the group scaled around his boost:

| | RMSE | heirs: bias |
|---|---|---|
| next man up alone (production) | 6.100 | +2.38 |
| team share alone | 6.107 (1/4 seasons) | |
| both, power 0.3 | 6.068 (4/4) | |
| both, power 0.5 | 6.058 (4/4) | +1.56 |

No double count: with the next man up the heir is still projected 2.4 a
game low, and the other backs on those weeks about 1.0 low, with nothing
handing them anything. Leaving the heir out of the scaling was about 0.01
worse. Power picked on three seasons and scored on the fourth chose 0.5
every time (2022 6.268 -> 6.245, 2023 5.941 -> 5.872, 2024 5.941 -> 5.894,
2025 6.226 -> 6.195), so backs use 0.5 and receivers 0.3. Backs move more:
about a quarter of RB weeks by 8%+, at most 16%. The replay knew who
played; live, a questionable back counts at his chance of playing, so a
backfield of two questionable backs lifts each a little in case the other
sits.

This week only. The season value is unchanged: who is out by then is not
known, and the season simulator already plays each absence.

### A lead's gain when his second back is out: tried, not adopted
Looking back, the effect is there: on 2022-2025 checkpoints (weeks 3-10,
534 lead/second-back pairs), a lead's rest-of-season error rises 0.33 a
game (se 0.10) per expected point of his partner's volume times the share
of games the partner went on to miss -- the same 0.32 measured in a single
game. Leads whose partner never missed ran 0.77 high; those whose partner
missed 6+ games, 1.04 low. On average it washes out (-0.10): the season
value, measured on games that include the usual absences, already carries
it.

Played through the season simulator -- while the partner is out, the lead
gains a share of his level, taken back out of his baseline at the
partner's expected absence rate -- shares 0.20/0.32/0.45 moved rest-of-
season RMSE 4.617 -> 4.570-4.588 overall but, picked on three seasons and
scored on the fourth, were worse in 2023 and 2025; a paired bootstrap
puts every change inside noise. Skipping the gain for an absence already
under way, or counting it in the rate, was never picked. Adding the gain
without taking it back out simply raises every lead (+0.33 a game played
too high). Not adopted; the simulator reproduces the effect correctly
(within-pair slope 2.0 against 2.2 real) if more seasons make the case.

One thing it surfaced: simulated leads fall 0.59 a team game short of what
they scored but only 0.15 a game played -- the gap is mostly in how often
they play, so the absence rate for lead backs may be a little high.
Rechecked after "Who loses his season" refitted the hazard by level: lead
backs are simulated to play 86.8% of their team's games against 87.6%
real. Leads valued 16+ still play 2-5% more than simulated in each season,
but that is about 15 players a season and it does not show in their
points (season-value error -1.3 to +1.7 by season), so no separate rate
for them.

### Who loses his season
Rest-of-season 80% bands held 75% of outcomes, short on both sides (12.2%
below the 10th percentile, 12.4% above the 90th), and 9.5 of the 12.2 low
points were players who played under half their remaining games. The
simulator lost about the right share of seasons (32.8% simulated, 34.7%
real) but not to the right players (AUC 0.75-0.79).

A logistic on "plays under half of his team's remaining games", over the
simulator's own chance, fitted on three seasons and scored on the fourth
(10,135 healthy or questionable player-checkpoints, 2022-2025, weeks 3-10),
lifts that to AUC 0.82-0.85 in every season. What says he will lose it,
beyond what the simulator already expected:

- the share of his team's games he has played: the biggest (odds about 6x
  for none against all; a third or fewer, 67% lost against 46% simulated);
- games in a row missed while healthy -- benched or scratched (1/2/3+: 42,
  62, 73% real against 32, 38, 55%);
- the share of last season he missed (3.1x for all of it), or no games last
  season (1.6x);
- tight end (1.9x; 30% real, 19% simulated), quarterback (0.58x);
- depth rank (1.14x a spot) and season value (1.8x per log unit: the
  simulator's level effect was too steep).

Little once these are in: recent snap share, falling opportunity,
efficiency, a team change, draft capital, injury tags and practice status
(recently injured players lose fewer seasons -- they tend to be starters).
Age helped a little (1.09x a year) but is not in the data production
reads, so it is left out.

How it enters the simulator mattered. Scaling each player's absence hazard
moved availability by level (stars playing 88% against 81% real) and
narrowed quarterback bands to 60%: a backup made to sit more never gets
the upside of taking over. Instead each new absence ends his season with
his own chance (`futures.season_loss`, base 12%, capped at 90%), and the
hazard by level was refitted so simulated availability still matches.
Held out:

| | in 80% band | below q10 / above q90 | RMSE | bias |
|---|---|---|---|---|
| before | 75.4% | 12.2 / 12.4% | 3.432 | -0.107 |
| low side widened only | 76.7% | 11.6 / 11.7% | 3.432 | -0.098 |
| one 12% for everyone | 79.0% | 9.5 / 11.6% | 3.429 | -0.093 |
| hazard scaled by player | 77.0% | 11.3 / 11.7% | 3.403 | +0.020 |
| **season-ending chance by player** | **79.7%** | **8.8 / 11.5%** | **3.391** | **-0.090** |

Coverage by season 79.5/81.9/77.3/80.1% (from 75.4/77.2/72.9/76.0%), RMSE
better in every season, bias level or better in three (2025 +0.086 ->
+0.102). By position: QB 66 -> 72%, RB 79 -> 81%, WR 77 -> 80%, TE 76 ->
84%. Production reads only what it already has: this season's and last
season's games, the depth role, and the games missed in a row; a player
out or doubtful has his absence drawn from his tag instead, and a bye is
not an absence. The high tail (11.5% above the 90th percentile, QB 15%) is
what is left.

`scripts/fit_season_loss.py` refits it from the rest-of-season backtest's
checkpoints, on the same inputs production reads
(`futures.season_loss_inputs`, counted by `projections.season_loss_counts`),
over a simulator already at the base share. Run in week 5 of 2026, it
reproduced the study (the same 10,135 cases; held-out AUC 0.835, 0.849,
0.826, 0.826 against the simulator's 0.764-0.790) with coefficients close to
the shipped ones except QB (-0.54 -> -0.19), RB (-0.19 -> -0.05) and season
value (0.80 -> 0.67) -- the base share in the simulator already covers part
of what those carried, and depth rank is read from the role. The shipped
values stay: they are the ones graded end to end above. After a refit,
`scripts/fit_ros.py` matches the hazard by level again.

### Publishing without committing the page
Every refresh committed the rebuilt page, its data and the team views to
`main`: 126 commits in a week. Packed, the whole repository was 47 MB after
three weeks (about 16 MB a week, 0.8 GB a year -- far less than the raw
5 MB-a-commit it looks like, since the pages compress well against each
other), but it is all churn: `docs/index.html`, `docs/latest.json`,
`docs/views/` and the current week's archive page change on every run, and
every clone carries every version.

With Settings > Pages > Source set to **GitHub Actions**, the workflow
(`weekly.yml`) deploys `docs/` as a Pages artifact and no longer commits
those files. It asks GitHub how the site is served each run
(`build_type`: `workflow` or `legacy`), so nothing changes until the setting
is switched and either way can be restored by switching it back.

- Not committed: `docs/index.html`, `docs/latest.json`, `docs/views/`
  (gitignored; the first deployed run takes them out of git). The current
  week's archive page is committed only when the week is published, not on
  refreshes.
- Still committed: `results/`, `reports/`, `data/leagues/` -- the state the
  next run reads. (`data/leagues` is the largest part of the raw churn; its
  past weeks never change again, so it packs well, but it could be next.)
- `docs/views/` is state, not just output: a game-day refresh reuses the
  last run's panels (`publish.read_me_cache`). It is kept between runs with
  Actions cache. If the cache is evicted, the next run rebuilds the views
  from scratch -- slower, same result.
- A run that does not rank (probes, scoring only) deploys nothing, so the
  last deployed page stays up.

### A record of missed games
The season simulator started new absences at one rate for everyone at a
position and projection level. Two things a manager can see say more,
measured over every game a contributor played, 2022-2025 (did he miss his
team's next game?), each season fitted on the other three:

- the share of his team's games he has missed this season: the biggest
  (odds about 3.5x for a player who has missed most of them against one who
  has missed none) -- re-injury, a role already slipping;
- the share of last season he missed (1.8x for all of it).

Age said nothing once the level was known (older players a touch higher,
younger a touch lower, worse held out in 2022), so it is left out.

`futures.hazard_mult` turns the record into a multiplier on the odds of a
new absence (fitted as a logistic over the position-and-level rate, centred
so the average is unchanged, clipped to 0.5-3x); a healthy player with a
full last season sits at about 0.75x. Applied to the rate instead of the
odds it made two seasons worse. Next-game log loss, held out, fell in every
season (0.324/0.292/0.278/0.283 -> 0.318/0.286/0.278/0.281).

Quarterbacks are left out: with them in, availability improved but their
80% bands narrowed (2024 76% -> 70%) -- a backup made to sit more never gets
a starter's upside, as "Who loses his season" found. Without them, in the
replay (each season simulated with the fit from the other three), for
healthy and questionable players:

| | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| games-played share, squared error | .0697 -> .0646 | .0667 -> .0600 | .0697 -> .0656 | .0690 -> .0670 |
| points per team game, miss | 2.456 -> 2.426 | 2.315 -> 2.264 | 2.660 -> 2.629 | 2.485 -> 2.446 |
| in 80% band | 77.6 -> 78.6% | 80.4 -> 81.4% | 74.7 -> 75.9% | 76.2 -> 77.2% |
| quarterbacks in 80% band | 83.0 -> 83.6% | 78.4 -> 78.7% | 75.7 -> 75.2% | 73.1 -> 73.6% |

Refitting the level calibrations on top (`scripts/fit_ros.py`) evened the
bias but missed points per team game by more in both held-out halves
(2.398 -> 2.422, 2.218 -> 2.276), so the calibrations stand. Left: players
valued under 4 now play about 6% fewer of their games in the simulation
than they did (3% more before) -- waiver fodder, where it matters least.

Cards show the note when the odds are 1.5x or more. `scripts/fit_hazard_history.py`
refits it (and grades it in the replay first; it writes only if the rule holds).

### The top of the season range, and the backup who takes the job
With who loses his season in, 11.6% of rest-of-season outcomes still fell
above the 90th percentile (QB 15.3%). Replayed on 2022-2025 (11,848
player-checkpoints):

- **Not the level.** Points per game played were above the simulated 75th
  percentile 25.5% of the time, below the 25th 23.6%: the spread of how
  well a player plays is right.
- **The games.** Players above the range played 86% of their team's games
  against 59% simulated, at a season value of 6.1: depth players who won a
  role. Depth players who went on to play 90%+ of games scored 1.3-1.5x
  their season value a game; those who played under half, 0.75x. The
  simulator draws whether he plays separately from how good he is.
- **One season.** Above the range by season: 11.2, 10.1, 14.8, 10.2%. In
  2024 nearly everyone outscored his projection (+0.39 a game); a shift in
  scoring across the league is not something a player's range can know.

Tried, each a lasting tilt per simulated season on a player's absence
hazard (`role_spread`, a mean-one lognormal), tied by `role_corr` to how
wrong his projection is, with `role_scale` putting back the games a spread
adds. Scored by quantile loss over the whole range (a proper score: a
range too wide costs as well as one too narrow), picked on three seasons
and scored on the fourth:

| | quantile loss | above / below range | held-out seasons better |
|---|---|---|---|
| before | 1.7171 | 11.6 / 8.7% | |
| RB, WR spread 0.5, tied to level | 1.7141 | 10.4 / 7.8% | 2 of 4 |
| all positions 0.5-0.8 | 1.7153-1.7199 | 9.0-10.1 / 7.0-7.7% | |
| a backup's hazard read at his stepped-up level | 1.7185 | 11.3 / 8.8% | |
| **QB spread 0.8, scale 1.3, tie 0.5** | QB 2.508 -> 2.483 | QB 15.3 / 13.5% -> 11.4 / 9.9% | **3 of 4** |

At running back and receiver the gain did not hold (2023 and 2025 worse)
and the bottom of the range, already short of 10%, got shorter. At
quarterback it did: a backup takes the job and keeps it (Jaxson Dart, Drake
Maye, Bryce Young, Jacoby Brissett in the replay) or barely plays, and the
simulator had scattered his games instead. QB ranges now hold 79% of
outcomes (71% before); simulated QB availability is unchanged on average
(-0.018 -> 0.000), the mean error flat (4.803 -> 4.805). Overall 80.7% fall
inside the 80% range, 11.0% above it. Backs and receivers stay as they were.

### Who he plays the rest of the way
A season value is a level measured on the games a player has played, against
whoever he played; the season simulator then played every week at that
level, so a back with three soft run defences in weeks 15-17 was worth the
same as one with three hard ones (only byes and the league's own schedule
were counted). Measured on 2022-2025 checkpoints (weeks 3-10, 69,499 future
player-games, points per game played), each week's level scaled by
`factor ** b`, `b` fitted on three seasons and scored on the fourth:

| signal | QB | RB | WR | TE |
|---|---|---|---|---|
| the future games' closing implied totals (a ceiling: not known in advance) | 4/4 | 4/4 | 3/4 | 2/4 |
| implied totals predicted from past lines (offence + defence + home, ridge) | 4/4 | 3/4 | 0/4 | 2/4 |
| the opponent's points allowed to the position, shrunk 6 games | 2/4 | 4/4 | 0/4 | 3/4 |
| both, fitted together | **4/4** | **3/4** (4/4 simpler form) | 0/4 | 2/4 |

Even the ceiling moves little (overall 6.52 -> 6.49 a game): one game's
opponent is a small part of a week's noise. But it is consistent at
quarterback (7.53 -> 7.42, 7.86 -> 7.85, 8.53 -> 8.51, 8.96 -> 8.89) and for
backs, and weeks 15-17 improve with it (QB 8.71 -> 8.65). Receivers do not
follow their opponents at all, and tight ends only weakly, so they stay
schedule-neutral. In use: QB `implied 0.6, allowed 0.3`, RB `0.2, 0.3`
(`roster.schedule`), a factor per future week (this week's own lines already
price this week) that spreads QBs by about +-7% over the rest of the season
and backs by +-4.5%, averaging 1.007 and 1.002 as in the replay. It enters
the season simulator only -- title odds, title-priced waivers and trades,
roster values -- so the season value stays the schedule-neutral level it is
measured as; a card says when the rest of the way, or weeks 15-17, are 4%+
easier or harder.

### Playoff-week matchups for D/ST and kickers
Quarterbacks and backs follow their opponents; defences and kickers had never
been tested, and they are the positions whose points depend most on who is
across the field. Same test (`scripts/fit_schedule_dstk.py`): checkpoints at
weeks 3-10 of 2023-2025, every unit's walk-forward season value against its
points in each later week, the week scaled by `environment ** a * allowed **
b`, fitted on two seasons and scored on the third (RMSE per game):

| signal | D/ST | K |
|---|---|---|
| predicted implied total (a defence: its opponent's) | 3/3, 8.47 -> 8.34 | 2/3, 4.674 -> 4.663 |
| opponent's points allowed to the position, shrunk 6 games | **3/3, 8.47 -> 8.23** | 2/3, 4.674 -> 4.673 |
| both | 3/3, 8.47 -> 8.25 | 2/3, 4.674 -> 4.662 |

Defences follow what their opponents give up to defences, steadily (fitted
weight 0.8-1.0 in every fold), and weeks 15-17 improve with it (8.81 ->
8.55). Kickers pass the rule on their own team's predicted total, barely.
In use: DST `allowed 0.9`, K `implied 0.5` (`roster.schedule`), from the D/ST
and kicker points the rankings pipeline already computes. At week 5 of
2026 a defence's playoff weeks range from about -20% to +23%; a kicker's
+-7%.

The title engine now prices the three free-agent defences and two kickers
with the best playoff weeks (season value times the factor over the weeks
after the regular season), as holds against streaming the slot from the wire
every week -- the replacement level the season simulator already fills it
with.

### D/ST and K against the expert consensus
A public benchmark for our D/ST and K rankings: the FantasyPros weekly
consensus (`/consensus-rankings`, positions DST and K), pulled every run for
the upcoming week and for any graded week the API still serves -- taken only
when the response names that week -- and kept with FantasyPros' other data
on the runner. Each graded week both are scored on the units both ranked:
Spearman correlation with actual points, and the top five's average points
and how many finished in the top `startable_rank`, with a record of weeks
our five outscored theirs. Their rankings are never shown. The consensus
for the current week refreshes until the week turns over, so on Sunday it
may carry later news than ours did -- a slight edge to them, not us.

### Blending the expert consensus into D/ST and K: tried, not adopted
Four weeks of 2026 had the FantasyPros consensus ahead on top-five points, so
the obvious move -- mix it into ours -- was tested on 2023-2025, where the
API still serves each week's consensus (`streamer dstk-blend`, run on the
runner: their rankings stay in its cache, the log carries aggregates only).
Our walk-forward predictions with the shipped settings, against theirs, on
the units both ranked:

| 2023-2025 | rank corr | top-5 points | top 5 finishing top 12 |
|---|---|---|---|
| D/ST, ours | **0.356** | 14.89 | 65% |
| D/ST, consensus | 0.332 | **15.03** | 66% |
| D/ST, Vegas baseline | 0.341 | 14.46 | 63% |
| K, ours | **0.201** | **10.08** | **75%** |
| K, consensus | 0.143 | 9.60 | 71% |
| K, Vegas baseline | 0.102 | 9.25 | 70% |

Blends (percentile mix, weight picked on two seasons and scored on the
third, on correlation or on top-five points) were worse in two of three
held-out seasons at both positions (D/ST 2024 0.373 -> 0.365, 2025 0.355 ->
0.350; K 2025 0.215 -> 0.203). Over three seasons ours ranks both fields
better and picks better kickers; their D/ST top five edges ours by 0.14 a
unit, inside the noise. This season's four weeks are a small sample; the
Model tab keeps grading both.

### Breaking news, for everyone
Injury tags, depth charts and box scores reach the model through the
platforms and nflverse, hours or days after the news. The page now carries
the news itself, about every player in the league, and refreshes often
enough for it to matter.

**Who has news.** No free source lists every player's news in one call:
ESPN's fantasy news answers one player at a time (its bulk forms return
500/400/404), ESPN's NFL headlines are 50 stories not tied to players,
Pro Football Rumors' and RotoWire's RSS carry 15 and 5 items. FantasyPros'
news feed does -- its latest 100 items league-wide, each with a player id
-- and the key is already there. It is used only to know *who* has news:
FantasyPros' text is never shown (its licence), so a player it flags who
cannot be found on ESPN appears as "news at FantasyPros" with a link.

**What the news says.** ESPN's player feed (RotoWire's blurbs and ESPN's
headlines, free) for every player FantasyPros flags in the last two days,
re-asked when the flag is newer than the copy kept, and for your roster and
your opponent's every run. Each headline is tagged -- out, injury, healthy,
role up, role down, move, legal -- and listed on the hub: yours first, then
your opponent's, then everyone rostered, then free agents, newest first.

**What it changes.** One thing, because it is unambiguous: a RotoWire
headline published this NFL week that rules a player out of his next game
("Jefferson (ankle) ruled out for Sunday") sets him OUT (IR when it says
so) before the platform's tag catches up, unless his game has started or
the headline is about an in-game injury ("won't return"). The rest -- a
limited practice, a new starter, a benching -- is shown, not priced: the
injury-tag odds, depth charts and opportunity model already read those
once the platforms and box scores have them, and a keyword is not evidence
enough to move a projection on its own.

**How often.** A credit-free refresh every two hours through the US day and
hourly on Sunday mornings and afternoons. GitHub has started this
workflow's scheduled runs hours late, so the times drift; the page stays
within a few hours of the news, and the Refresh link is immediate.

### Free agents the news has close to signing
Tyreek Hill, unsigned after a dislocated knee in 2025, was on a bench in
both leagues and worth exactly nothing to the model: no team, last game
more than a season ago. Meanwhile the Chiefs had reached out to his agent,
his agent was negotiating with several teams, and his Yahoo ownership had
gone 5% -> 22% -> 31% in three weeks. The box scores and the platforms only
catch up once he has signed.

The news now feeds the price (`data/news.py`). For every player on no NFL
roster who is on a fantasy roster, 2%+ owned or among Sleeper's most-added
of the day, ESPN's news about him -- RotoWire's blurbs and ESPN's headlines,
free, no key -- is read for its most advanced stage in the last two weeks:
**signed** (a headline naming a team), **close** ("expected to sign",
"closing in", "finalizing"), **talks** (negotiations, visits, workouts, a
team reaching out). Sleeper's database, which moves a signing within hours,
can say signed before the platforms do. A legal case in the news is
flagged. Pro Football Rumors' and RotoWire's RSS feeds were probed and left
out: they carry only the last 15 and 5 items.

He is then valued as if on the new team (his old games at the old-team
weight, so Hill's 12.5 becomes 11.1 for Kansas City, or the same discount
for a team not yet named), worth nothing this week, and played in the
season simulation only if he signs and gets on the field:

- what follows a signing is measured: of 17 veterans signed in season after
  weeks on no roster (2022-2025), 13 played that season; 8 of those within a
  week of signing, the rest 4-9 weeks later. After their debut they scored
  7.7 a game against 8.8 projected at signing -- a small sample, so no
  further discount is applied;
- the chance he signs by stage (signed 1.0, close 0.8, talks 0.5) and the
  weeks before he does (0, 1, 2) are judgment, not measured: there is no
  free archive of this news to fit them on. They are in the config and on
  the card. Each run now logs every stage the news reaches for a free agent
  (`results/signing_log.parquet`, first sighting of each) and resolves it:
  he turns up on a team (Sleeper or the league), or is still unsigned four
  weeks on. Once a stage has 5 resolved sightings its odds become the
  observed rate shrunk toward the judgment by 10 cases
  (`free_agent_news.learn`), so the guess gives way to the record. The weeks
  before signing stay judgment. (The FantasyPros weekly weight already
  learns the same way: refitted each run on the logged games, 0.15 on 286
  in week 5.)

The waiver and trade screens, which read season value without the
simulation, count him for the share of the rest of the season he is
expected to play. Hill today: talks with Kansas City, 11.1 a game when he
plays, 38% to play this season, about a quarter of the rest of it -- 2.7 a
game to the screens, against 0 before. The card says what the news said.

### The consensus as a second forecast of season value
Ownership and name value track what people expect a player to score, so
when the model rates a well-known receiver far below the market (Brian
Thomas Jr. 7.9 a game against a market 9.7; Xavier Worthy 7.6 against
9.4), it helps to know which view has usually been right. For name value,
we checked on 2022-2025. When our season value for a receiver sat 1.5+
points below name value in weeks 3-6 (159 cases), ours missed what they
went on to score by 3.08 a game on average. Name value missed by 3.66, and
an even average of the two by 3.27. Later weeks look the same. In the 27
cases most like these two (we said 7.8, name value 9.4), they averaged 6.6
a game, and 30% reached the name-value number. Name value leans on last
year: Thomas Jr. scored 16.7 a game as a rookie, then 9.9, and 5.3 so far
this season. The model leans on the role now, and that role has shrunk.

Expert consensus is a different forecast. The experts read news, depth
charts and coaching changes that none of our inputs see. There is no free
archive of their rankings to backtest, so the consensus now gets a share of
every season value, and has to earn it:

- **Order, not level.** A player's consensus value is our season value of
  whoever holds his consensus rank at his position. The blend moves players
  past each other but never shifts a whole position.
- **Healthy players only.** An injured player's rest-of-season rank also
  counts the games he will miss; our per-game value leaves those to the
  season simulation, so the two are not on the same footing.
- **Earned weight.** Every refresh logs our own season value (before the
  blend) and the consensus rank before kickoff. The weight is the one that
  best predicts what the logged players then scored that week, shrunk
  toward a prior of 0.3 by 300 games of evidence. The prior alone stands
  until 150 games are graded. The same scheme sets the betting-prop weight.
  The prior is below an even split because our values beat name value
  where the two disagreed, and the experts are untested here.
- **Visible.** A player moved 0.3+ points carries a note ("season value
  raised from 7.9 to 8.6: the FantasyPros expert consensus rates him
  higher than our model does"), and the scorecard says what share the
  consensus has and what it rests on. Their ranks are never shown.

Everything built on season values follows: drops, adds, plans, trades and
title odds. The rest-of-season benchmark still grades our own values
against their ranks, so the weight and the benchmark answer the same
question from two sides.

### The consensus projection as a second forecast of this week
The FantasyPros weekly projection was pulled, logged and graded on the
scorecard, but no decision used it: this week's numbers blended our model,
the platform's projection and the betting market, and stopped there. It now
joins the market as a second forecast of this week, the same way:

- **One fit for both.** This week's projection is
  `ours + w_m (market - ours) + w_c (consensus - ours)`, where a player has
  each, so the two weights are fitted together on the logged games -- two
  sources that agree are not counted twice. Each is shrunk toward its prior
  by its own graded games (300 games of evidence), and the prior stands
  until 150 are graded: 0.4 for the market as before, 0.3 for the
  consensus, matching its season weight. At week 4 the market's weight
  had been measured down to 0.26; the consensus had 13 graded games, so its
  prior stands.
- **If he plays.** Both are blended on the if-he-plays number and the
  injury discount re-applied, as props were. A consensus projection near
  zero says he sits; the platform's tag carries that, so it is not blended.
- **Everywhere this week's number is used.** The lineup, P(win), this
  week's pickups and streams, and the week in progress of every simulated
  season -- so waiver adds and drops, trades and the roster's title worth
  all see it. The season values carry the rest-of-season consensus as
  before; the betting market has no rest-of-season number, so it reaches
  season-long decisions only through this week.

The Lineup tab says what went in: each source, its weight, how many of your
players it covered, and what the weight rests on. A source missing from a
run is said rather than dropped silently. That came from a real gap. From
Thursday night to Sunday morning of week 4, every page ran without betting
props and showed no Vegas column, with no reason given. Refreshes spend no
credits; Wednesday's run had bought only the Thursday game, the one inside
the 72-hour window, and the Sunday run had not yet fired. GitHub was running
it four hours late.

### Where our season values defer to the market
A trade the hub ranked first (Drake London for Cam Skattebo and Michael
Wilson, +2.5 points of title odds) rested on our model valuing Wilson at
16.1 a game against about 12.6 in the market, after one 17-target game;
FantasyPros called it a big loss. Rather than trust either, the question
was put to history: 2022-2025, weeks 3-13, 8,158 player-weeks of players
with a name (4+ games last season), comparing our projection and the
name-value view (reputation and season so far in the measured 46:13
proportions) with what each player actually scored per game the rest of
the season.

Our projection won overall (average miss 3.01 against 3.15) and won big on
big disagreements (3.9 against 5.8 at 3.5+ points apart): blanket
shrinking toward the market made it worse out of sample. Two cells were
different, out of sample as well as in (fitted on three seasons, tested on
the fourth, each in turn):

| Case | Cases | Our miss | Calibrated | Kept of our gap |
|---|---|---|---|---|
| Receiver we mark up, weeks 3-6 | 95 | 2.75 | 2.57 | 24% |
| Quarterback we mark down, weeks 3-6 | 64 | 3.10 | 2.85 | 38% |

Both are early-season overreactions to a few games, which is what a
2.5-game opportunity half-life risks. In those cells only, the season value
becomes market + k x (ours - market), with a note on the player; the
weekly projection (lineups) is untouched. A receiver we mark *down* -- the
London case -- stands: in 22 such cases the player scored 15.3 a game
against our 14.9 and the market's 17.4.

Wilson's season value became 13.4 and the trade +1.3. Cells are refit by
`scripts/fit_market_calibration.py`, which keeps one only if it beats our
projection by 3%+ on held-out seasons, with 60+ cases, and shrinks (k < 1).

### The championship hub
The panel had grown a section per idea. The hub, at the top, puts every
move on one list in one unit -- points of title odds -- best first:
lineup changes, D/ST and K streams, waiver claims (blocks marked), multi-
move waiver plans and trades.

Season moves are already priced in title odds. A move that only changes
this week moves P(win the week), and a win this week is worth a measured
amount of title odds: the must-win calculation forces this week's game to
a win and to a loss in every simulated season. The move's value is the
change in P(win) times that swing. Waiver claims are listed net of the
waiver priority they spend. A trade adds its gain only if he says yes, so
it is ranked by the expected gain (gain x chance of a yes), with the gain if
accepted beside it; offers of the same players to different teams are one
decision, listed once. (Ranked by raw gain, long-shot offers filled the top
of the list on the first live run.)

Each move is priced on its own against the roster as it stands, so the
numbers do not add up; the advice is to make the top one and refresh. The
detail sections -- season outlook and must-win weeks, lineup extras,
waivers and plans, trades, streams and stashes -- fold away below, and
each hub row links to the one that explains it. The matchup and the
recommended lineup stay open.

### Rivals on the wire
Standing pat used to mean the free agent stays on the wire all season. In
a real league, when he breaks out somebody claims him. Now, in each
simulated season, each rival claims a breakout at the league's activity
rate (the same assumption as before); among the rivals who claim, the best
waiver priority wins, and that team gets him from the following week in
place of its weakest player. The draws are per player, so the same rival
does not land every breakout in a season.

That changes three things. **Standing pat** includes a rival landing him.
**Waiting** gets him only when your priority beats every rival who also
claims; otherwise that rival has him. **Adding now** also denies him to
everyone. A pure block -- a player who barely helps you but would help
the team you are chasing -- is priced the same way.

Measured on both leagues at week 4, it matters less than expected: a rival
landing one of the top breakout candidates moved your title odds by -0.15
to +0.23 points -- inside the noise, and sometimes negative, since the
rival who lands him can beat a team you are chasing. With 10-14 teams, one
player moves one rival a little. Blocking is priced into every move and
named on a card only when it is worth at least 0.3 points; it should
surface late in a tight race, against the team just above you.

To keep the extra pricing affordable, the lineup simulation now keeps
players on the last axis of its arrays (the per-slot best-player search
runs about a third faster), and candidates too far short on their own
merits for blocking to rescue them are not priced for it.

### A mid-week move cannot reach back into played games
Between Sunday and the Tuesday turnover, the simulator's first week is the
one in progress, with finished scores fixed. Pricing a pickup by putting
him on the roster for that whole week credited you with points he scored
on the wire -- a free agent with 30 on Sunday looked like a 30-point
upgrade for a week you cannot change. Every roster a move produces is now
settled against the team's current one: a newcomer whose game has kicked
off joins from next week, and a departing player whose game has kicked off
still counts this week. Waiver moves, plans and both sides of a trade use
it.

### The week turns over when Monday night is over
The week detector compared the schedule's Eastern game dates with the UTC
date, so the week turned at midnight UTC -- 8pm Eastern on Monday, during
the Monday night game. A refresh then synced the next week while the
platform's standings still showed the old record, and the simulator,
starting at the new week, dropped the unfinished one: no loss counted,
none simulated, title odds too high. Two fixes. A week is over at its last
kickoff plus four hours. And if the standings still lag the calendar (the
platform finalises overnight), the simulator plays the uncounted week out
with today's rosters, says so on the page, and treats it as history for
waiver and trade pricing: nobody added now can play in it.

### Waiver plans: several moves priced together
Single moves are priced one at a time, and moves interact: two backs
chasing one lineup spot are worth less together than apart; two dead
roster spots turned into a starter and a handcuff can be worth more. A plan
is two or three (add, drop) pairs made together.

Every combination of the ten most promising free agents and your eight
weakest skill players is screened on roster value (the waiver engine's
measure: lineup, bench depth, bench upside), keeping the best drop-set for
each group of adds and only plans that beat the best single move on paper.
The best twelve are priced in P(title) on the title engine's seasons,
against standing pat and against the best single move on the same seasons,
and a plan is listed only when it beats that single move by more than
twice the paired noise ("make the plan"); one that beats it by less is
shown as a close call, since the single move then gets most of the gain. A plan one pickup away from a better one is folded into it as
an alternative ("about as good: X instead of Y").

Claims go in order of what each is worth alone: on a rolling list the
first successful claim spends your priority and the rest land when nobody
ahead of you wants the same player. A drop the market still values well
above our projection is flagged to shop in a trade first. On the real
leagues at week 3: ESPN's best plan (Aiyuk, Love, Henry) reached 7.1% title
odds against 6.0% for the best single move and 5.5% standing pat (noise
+-0.26); Yahoo's (Dobbins, Otton, Bernard) 16.6% against 14.6% and 13.2%.
A later ESPN sync listed Aiyuk out, and its best plan fell to +0.5 over the
best single move against +-0.58 of noise: a close call, which is what the
page now says instead of hiding it.
Screening takes about a second, pricing about five.

### IR stashes

Both leagues give every roster IR slots, and the waiver engine used them only
to keep injured players off the drop list. A free agent on a reserve list
(IR, PUP, suspended) was never priced at all -- the candidate pool skipped
anyone out long-term -- yet he costs no roster spot while he sits in an empty
IR slot, and on the ESPN wire at week 5 that meant De'Von Achane (13.6 a game
when healthy) and Jaxson Dart (16.0) went unseen. Yahoo's wire did not even
list them: its free-agent pages are sorted by this week's projection, and an
injured player's is zero.

- **How many slots.** ESPN states them (``position_slot_counts``); Yahoo
  lists them in the settings page's roster positions. An older snapshot
  falls back on the most any team has filled.
- **Who is priced.** The six free agents on a reserve list with the best
  season value join the waiver candidates. The season simulator already
  draws whether and when each comes back -- the reserve-list model fitted
  on 1,277 cases, 2022-2025 (most never return that season; the rest miss
  about four more games) -- so nothing new is assumed about the injury.
- **How he joins the roster**, each priced on the same seasons, the best
  taken: an open IR slot (nobody goes now; your least valuable player goes
  the week he is back on the field, when he needs a bench spot); a filled
  IR slot (its occupant is dropped now, and the same later drop applies);
  or a bench spot for one of the usual drops. Any IR-eligible candidate
  (OUT counts in both leagues) gets the IR ways, not only the long-term out.
- **Waiting** means claiming him the week he is back, against the rivals who
  want him then -- the breakout rule, with his return as the breakout.
- **Yahoo** reads one more page of free agents, ordered by Yahoo's overall
  rank, and keeps only the players on a reserve list or out from it.

At week 5 in the ESPN league (both IR slots full), Achane is the top waiver
move, a close call: about +0.4 to +0.9 points of title odds for a bench drop
the simulation cannot separate. Swapping him for Dylan Sampson's IR slot
priced a little lower, because it costs Sampson now *and* a bench player when
Achane returns.

### Trades: what helps you, and what he will accept
The waiver wire in a ten-team league is shallow; a roster short on points
has to find them on other rosters. Every 1-for-1, 2-for-1 and 1-for-2 of
skill players with every team is screened, and each trade answers two
questions about two different people.

**Does it help you?** Our projections and our simulated futures: your
P(title) with the new roster, priced for both teams at once on the title
engine's seasons, kept when the gain clears 0.3 points and twice the paired
noise.

**Will he say yes?** He cannot see this tool, so his view has to be
modelled, and it was measured rather than guessed. Yahoo's "% rostered" is
the whole market's revealed opinion of every player. On 237 players at
week 3 (logit of % rostered, position intercepts), it is explained by:

| What he sees | Weight | 90% interval |
|---|---|---|
| The platform's projection | 41% | 18-73% |
| Reputation: last season's points a game (the one before at half weight), shrunk to the projection over 4 games | 46% | 24-64% |
| This season so far, shrunk over 2 games | 13% | -2 to 27% |

Out of sample the blend explains 68% of rostership; our projection alone
explains 60% and adds nothing once the three are in -- the market does not
value players the way we do, and that gap is what makes a trade work.
Selling a player the market rates above us (a big name in a shrinking role)
and buying one it rates below us helps you and looks good to him. Being on
IR reads about 1.4 points a game lower; each game missed last season about
0.08 (injury history barely registers). `scripts/fit_perception.py` refits
the weights into `perception.json`; the season-so-far weight should grow as
the sample does.

His perceived gain is the change in his roster value on that blend (his
best lineup plus bench depth). The chance he accepts is a logistic in it,
even money at +1 point a game (people value what they own above what they
are offered), shifted by up to 0.75 when an uneven deal gives him -- or
takes from him -- a clearly best player, scaled by engagement (0.6 for a
team with no pickups, 1.0 at one a week), and topping out at 80%. That
shape is an assumption: offers and refusals are not published, so it
cannot be fitted. It ranks offers; the percentage is not a measured rate.

Three views of the same priced trades: **top** by expected gain (chance of
a yes x title odds gained), **best for you** by title odds among offers
with at least a 15% chance, **most likely yes** by the chance among offers
that clearly help. Each card says why for both sides: sell-high and
buy-low gaps, who he would start over, name value, a hot or cold start,
injuries, how active he is. On the ESPN league at week 3, 999 trades
screened and 66 priced took 20 seconds; the top offer (London for Swift)
had an 86% yes-chance and +1.3 points of title odds.

### The trade deadline, two-for-twos, and the spot a deal frees
Trades are the biggest lever a manager has until the deadline, and the tool
did not know when that was. ESPN states it (`tradeSettings.deadlineDate`),
Yahoo's settings page lists a "Trade End Date"; both are read into the
league rules. The trades tab and every trade on the hub say how many days
are left once it is two weeks out, and after it passes no trades are
searched and the tab says trading has closed.

The search was 1-for-1, 2-for-1 and 1-for-2. It now adds 2-for-2 among each
side's eight most valuable players (28 x 28 pairs a team, beside the smaller
shapes' ~1,300): a deal that swaps two starters for two can fit both rosters
where no smaller one does. And a deal that frees one of your roster spots
used to leave it empty -- the simulator filled the slot at replacement level
only when a lineup had a hole -- so a 2-for-1 was priced a bench spot short.
The free agent who adds most to your roster now fills it, is named on the
offer ("then add X") and is priced in; the page's trade evaluator, which
swaps players only, leaves those deals out of its exact list and accuracy
check.

At week 5 in the ESPN league: 3,306 offers screened, 84 priced, 43 seconds;
45 of the priced offers and the top of the list are two-for-twos.

### Grading our own season odds
Nothing had checked whether "41% to make the playoffs" means 41%. Every run
now appends each team's playoff chance, title chance and expected wins to
`results/<profile>/odds_log.parquet` (the last run of a week replaces earlier
ones). When the regular season is over and the standings have counted every
game, `odds_log.grade` scores each logged week's playoff chances against who
actually got in (Brier score, against always saying the league rate),
calibration by bucket, and expected wins against actual; the run log prints
the by-week scores. Nothing is shown on the page until a season has been
graded -- that is the point of keeping the record from now on.

### D/ST and the quarterback across from it: tested, no
The D/ST model reads the opponent's *team* sack, interception and pressure
rates, which describe whoever played quarterback before; a backup starting
this week keeps his starter's numbers. Tested walk-forward 2023-2025
(`scripts/fit_dst_opp_qb.py`), the opposing starter (most dropbacks in the
game) rated from his own earlier dropbacks only -- sack rate, interception
rate, EPA per dropback, each shrunk 150 dropbacks toward the league, and the
gaps between his rates and his team's:

| | MAE | rank correlation | top-5 hit rate | top-5 points |
|---|---|---|---|---|
| shipped | 6.2014 | 0.3564 | 0.652 | 14.89 |
| + opposing QB's own rates | 6.1970 | 0.3542 | 0.648 | 14.96 |
| + "new starter this week" flag | 6.2015 | 0.3572 | 0.652 | 14.85 |

Neither is better in two of three seasons on both MAE and rank correlation.
In the 236 games against a starter who did not start the team's previous
game, the QB's own rates changed nothing (MAE 6.7508 against 6.7506); those
defences did beat their projection by 0.65 a game, but on 236 games with a
7-point spread that is inside one standard error (about 0.45), and the flag
that halved it made those games' MAE worse. The opponent's implied total,
which falls when a backup starts, already carries most of it. Not used.

### The factor ledger: tested against a plain ridge
The ledger scales each standardised D/ST and K feature by `clip(|r_blend| /
|r_hist|, 0.25, 3)` after standardising, which changes its effective ridge
penalty by up to 144x between two features. A review asked whether it earns
that. Walk-forward 2023-2025 with the shipped settings, ledger on against a
plain standardised ridge (`scripts/test_ledger.py`):

| | MAE | rank correlation | top-5 hit rate | top-5 points |
|---|---|---|---|---|
| **D/ST, ledger** | 6.2014 | 0.3564 | 0.652 | 14.89 |
| D/ST, plain | 6.2076 | 0.3501 | 0.633 | 14.73 |
| K, ledger | 3.7187 | 0.1743 | 0.552 | 9.50 |
| K, plain | 3.7184 | 0.1777 | 0.552 | 9.48 |

D/ST: the ledger is better in all three seasons on MAE and rank correlation,
and in two of three on top-5 points; kept. Kicker: a tie (MAE 3.7187 against
3.7184, rank correlation 0.003 worse, top-5 identical), well inside what one
season's noise moves; it neither helps nor hurts, so it stays rather than
change a setting on no evidence.

### Kicker wind forecast
Typical wind recovers about three quarters of what recorded wind is worth
(see "Kicker wind"); a forecast should do better in a windy week. Built, but
off until it has proved itself:

- `data/forecast_wind.py`: stadium coordinates for the 32 home teams (and
  neutral sites by stadium name), Open-Meteo's free forecast (no key) for the
  kickoff hour and the next three, mean wind and gust. Domes are never asked.
- A forecast model's 10 m wind is not nflverse's recorded wind, so it is
  mapped onto the recorded scale (`recorded = a + b x forecast`) and blended
  with the stadium's typical wind by lead time: full inside 2 days, falling
  linearly to zero at 7. The lead-time shape is a judgment: Open-Meteo's
  archive of forecasts as issued holds only near-term ones, so it cannot be
  fitted.
- `scripts/fit_wind_forecast.py` (the `wind-fit` job on the runner, which can
  reach Open-Meteo; the development sandbox cannot) calibrates `a`, `b`
  leave-one-season-out on 2022-2025 games and backtests kickers against
  typical wind, overall and on windy games. With no `wind_forecast.json` the
  module does nothing; the file is written only if the forecast beats typical
  wind on MAE and rank correlation in at least two of three seasons.

### Kicker wind
The kicker model trains on the wind recorded at each game, but a game still
to be played has none -- nflverse's schedule has wind for 0 of the 386
games not yet played -- and production filled it with `weather.default_wind_mph`
(0, calm), about 7 mph under the outdoor average. Walk-forward 2023-2025
(`scripts/fit_kicker_wind.py`, the backtest with the test games' wind
replaced by what could have been known):

| wind given to the model | MAE | rank correlation | top-5 hit rate |
|---|---|---|---|
| recorded (cannot be known ahead) | 3.7155 | 0.1705 | 0.556 |
| calm (production's fill) | 3.7480 | 0.1488 | 0.522 |
| **the home stadium's typical wind** | **3.7229** | **0.1609** | **0.541** |

Typical wind recovers about three quarters of what the recorded wind is
worth, better than calm on MAE and rank correlation in all three seasons;
calm also biased outdoor kickers high (+0.23 against -0.08). A game with no
recorded or given wind now gets the mean wind of its home team's outdoor
games over the last two seasons and this one so far (`weather_source`
"typical"); domes stay neutral and a team with no history keeps the 0 default.
A manual `data/weather_week_N.csv` still overrides. A real forecast (a free
service such as Open-Meteo) would beat the typical wind; it is not wired in.

### Do the platforms already discount questionable players? No
Our last step multiplies by the chance he plays; if a source had already
done so, a questionable player would be discounted twice. Checked on the
logs (weeks 3-5): among players tagged questionable the platforms' average
projections are *higher* than for healthy players (ESPN 9.2 against 7.9,
Yahoo 17.8 against 14.0), so they are if-he-plays numbers, the same meaning
as ours, and betting props are voided when a player sits. FantasyPros'
projection is logged on the runner only and was not checked there. Only 23
questionable players per league: no evidence of a double discount, not proof.

### Fitting every blend weight together: not yet
The projection is a chain: our model blended with the platform (weight
0.53, fitted on healthy players), then the betting market and FantasyPros
blended on top of that result, each stage fitted against the previous
stage's output, so the second stage already accounts for the first. Taken
together the effective shares are roughly our model 18%, the platform 20%,
the market 22%, FantasyPros 40% -- so a gain to our model alone shows up in
the final number at about a fifth of its size (the gains reported for
individual changes were measured on the model alone). The platform weight is
stable across held-out weeks (0.40-0.60 in both leagues). A single joint fit
cannot be checked fairly yet: the market numbers logged before the shape and
fill fix ran 1.5 a game low and no longer count, and FantasyPros' log is on
the runner. Once 150 rows built the new way are graded it can be tried the
way the position weights were (hold each week out; use it only if it wins).

### Grading the number we publish, on everyone it covered
The scorecard graded each source on players who played, and mostly on
numbers easier than the decision they inform. Now three views of the same
logged, pre-kickoff final numbers (this season's logs are untouched by any
change fitted on 2022-2025, so they are the clean test):

- **Players who played** (as before);
- **Counting the projected players who did not play, as zero** -- a
  projection for someone who ends up inactive is a miss; a week still being
  played is never counted this way (a late game's players have not had their
  chance), only weeks the history has moved past;
- **Likely starters only** (top 12 QB, 24 RB, 30 WR, 12 TE by our final
  number each week), inactive counted.

Weeks 3-5 of 2026, ESPN league: miss 4.43 (played) -> 4.33 (inactive counted,
50 projected players, our number already discounts doubtful ones); ESPN's
projection 4.62 -> 4.55. On likely starters alone the edge nearly vanishes:
miss 6.70 against ESPN's 6.78, start/sit 59.8% against 59.6%; in the Yahoo
league 6.46 against 6.47 and start/sit 61.8% against Yahoo's 62.5%. The
earlier lead over the platforms was mostly on the bench and waiver players
the platforms treat loosely; among the players who matter each week the
models are close. The page shows all three tables.

### Turning a prop line into a mean
A review of the design pointed at the Vegas number, and the logs agreed: for
healthy players with props, weeks 3-5 of 2026, actual points less the Vegas
number were +1.49 at RB, +1.61 at WR, +1.14 at TE (QB -0.05), while our
model and the platforms sat near zero. Two causes, both in how a line
became a mean:

- **A bell curve around the line.** A balanced line is the median, and
  yardage is skewed -- a long touchdown sits far above the line, a quiet
  game can only fall to zero -- so the mean is above it. Measured on nflverse
  2021-2025 player-games bucketed by the trailing-4 level a book prices
  (`scripts/fit_prop_shape.py`, fitted 2021-2023 and scored on 2024-2025),
  the bias of each conversion against the real mean:

  | | bell curve | measured shape |
  |---|---|---|
  | rushing yards | -6.30 | -0.20 |
  | receiving yards | -6.46 | +0.25 |
  | receptions | -0.30 | -0.09 |
  | passing yards | +3.20 | +0.68 |

  About a point a game for a receiver, 0.6-1 for a back. Lines are now
  converted on the measured shape (`streamer/data/prop_shape.json`: the
  median and the quantiles around it at each level, interpolated): the line
  is the `1 - P(over)` quantile, the mean is the median plus that level's
  mean-less-median. Touchdowns and interceptions keep the count conversion.
- **Markets no book posted counted as zero.** A back with no receptions prop
  lost his catches. Each priced player's unposted stats (`COMPONENTS` by
  position) are now filled with his own per-game average over his last 8
  games, and his points re-scored on the whole line.

The market's weight had been fitted on the biased numbers: 0.10 on these
logs, 0.35 with each position's bias removed (in-sample, indicative only).
Rows built the old way no longer set it: each logged Vegas number carries
its build (`vegas_version` 2) and the line behind it, and the prior (0.4)
stands until 150 rows built the new way are graded.

### Snap share as an early sign
Expected points move once targets and carries do; who is on the field moves
first. Replayed on 2022-2025 (`scripts/fit_snaps.py`, 12,953 back, receiver
and tight-end weeks with 4+ games for the team that season, through the
production projection with its implied-total scaling). Offensive snap share
(Pro Football Reference via nflverse, joined by team, week and name):

- backs whose snap share over the last 2 games was 15+ points above their
  earlier games beat the projection by 1.06 a game; 15+ below, fell 1.56
  short (all backs +0.08); tight ends +1.19 / +0.08; receivers +0.39 / -0.69;
- two signals: `delta` (last 2 games less earlier) and `gap` (snap-share
  percentile at the position less his share-of-group-work percentile: on
  the field more than he is used), as `exp(trend x delta + gap x gap)`,
  picked on three seasons and scored on the fourth:

| | trend | snaps vs use | both |
|---|---|---|---|
| **RB** | 2 of 4 | 4 of 4 | **6.355 -> 6.322, 4 of 4; (0.25, 0.5) every fold** |
| WR | 1 of 4 | 0 of 4 | 1 of 4 |
| **TE** | 3 of 4 | **5.259 -> 5.253, 4 of 4; 0.25 every fold** | 3 of 4 |

The largest single gain of this round, for backs. In use (`roster.snap_trend`):
RB `trend 0.25, gap 0.5`, TE `gap 0.25`, this week only; backs move about
+-10% (5th-95th percentile), and a card says which signal moved him. At week
6 of 2026: Braelon Allen (41% -> 90% of snaps) +13%, CJ Donaldson +21%,
Najee Harris -12% (30% of the backfield's work on 18% of the snaps). Route
participation is not used: no free archive covers these seasons.

### Blend weights by position
The weights on the betting market and the FantasyPros projection are one
pair for every position. Fitted position by position on the 3 graded weeks
so far (779 market rows; QB 72, RB 200, TE 186, WR 321) they differ a lot
-- market 0.3 at QB, 0.25 at WR, 0.1 at RB, 0 at TE -- but each graded week
held out in turn, the position weights (shrunk toward the pooled one by
0-600 games) scored *worse* than the pooled weight at every shrinkage
(34.89 -> 34.95-35.12 squared error), more so the less they were shrunk:
three weeks cannot tell a real position difference from noise, and there is
no archive of props to replay.

So the fitter now decides for itself, every run (`vegas._position_weights`):
each graded week is held out, pooled and per-position pairs (shrunk 300
games toward the pooled one) are fitted on the rest and scored on it, and a
position gets its own weights only when they win by 0.1% of squared error
overall across at least 3 weeks, with at least 100 graded rows of its own.
Today it finds -0.16% and everyone keeps the pooled pair; the run log says
so. As the season grows, and with the whole waiver wire now graded, it can
switch on without anyone remembering to look.

### Each player's own range
A week's range -- the 10th to 90th percentile on the page, the boom chance,
and the spread every lineup and P(win) simulation draws from -- came from
position and projection level alone: a touchdown-or-nothing back and a
twelve-catch back at the same projection got the same range. Replayed on
2022-2025 (`scripts/fit_player_spread.py`, 20,512 player-weeks through the
production projection and range): each player's spread multiplied by
`exp(c + beta x z)` over his last 17 games -- touchdown dependence (6 x
expected TDs / expected points), targets / (targets + carries), his own
swing (sd of points less expected points, shrunk 6 games toward the
position's), and this week's team total -- fitted on three seasons to the
quantile loss at the 10th and 90th percentiles, scored on the fourth, with
the Brier score of a top-12 week as a second grade:

| | width only | width + style | top-12 Brier | |
|---|---|---|---|---|
| QB | 1 of 4 | 2.675 -> 2.688, 0 of 4 | worse | no |
| **RB** | 3 of 4 | **2.0331 -> 2.0213, 4 of 4** | **0.09451 -> 0.09433** | **kept** |
| **WR** | 2 of 4 | **1.9899 -> 1.9844, 4 of 4** | **0.06611 -> 0.06609** | **kept** |
| TE | 3 of 4 | 1.6234 -> 1.6213, 4 of 4 | 0.11402 -> 0.11419 (worse) | no |

His own swing matters most (about +5% per standard deviation); for backs,
a bigger share of catches narrows the week. Fitted on all four seasons and
written to `outcome_model.json` (`player_spread`, RB and WR); z-scores are
capped at +-3. At week 5 of 2026 it widens Jahmyr Gibbs, James Cook and
Davante Adams 11-16% and narrows backups like Tyjae Spears 10-15%.

### This week's defence and game script
This week's projection was scaled by the implied team total only -- the
game's scoring, not who the opponent is at the player's position, and not
whether his team should be ahead or behind. Replayed on 2022-2025
(`scripts/fit_game_context.py`, 20,524 player-weeks through the production
projection *with* the implied-total scaling, so the spread's effect on the
total is already in): `pred x allowed ** a x exp(b x spread / 7)`, `allowed`
the opponent's points allowed to the position (shrunk 6 games, last season
at half weight, as the season simulator reads it), spread positive when
favoured; `(a, b)` picked on three seasons, scored on the fourth.

| | defence only | spread only | both |
|---|---|---|---|
| QB | 7.663 -> 7.647, 2 of 4 | 0 of 4 | 2 of 4 |
| **RB** | **6.181 -> 6.173, 3 of 4 (fourth tied); a = 0.3-0.4** | 0 of 4 | same as defence |
| **WR** | 0 of 4 | **6.084 -> 6.081, 3 of 4; b = -0.03 every fold** | 2 of 4 |
| TE | 3 of 4 but worse overall | 0 of 4 | -- |

Backs follow the run defence they face beyond what the total says;
receivers on underdogs do slightly better (trailing teams throw). Both are
small -- hundredths of a point a week -- but steady in every fold. In use
(`roster.game_context`): RB `allowed 0.3` (the low end), WR `spread -0.03`,
about -3% for a receiver on a 7-point favourite. A card says so when the
defence moves a back 3%+ or the spread a receiver 2%+.

### A new quarterback for the receivers

*Superseded: switched off once graded inside the full pipeline ("Edges that did not hold up in the full pipeline").*
When a backup starts, this week's implied team total falls and every
projection on the team scales with it (`vegas_damping`). Replayed on
2022-2025 (`scripts/fit_qb_change.py`, 17,947 back/receiver/tight-end weeks
through the production projection *with* that scaling): each team-week's
starter is the quarterback with the most attempts, a player's own
quarterback the starter of his last six games for the team, and quality is
passing EPA per attempt over every earlier game, shrunk 200 attempts toward
the league.

With a starter half a league spread or more below his recent quarterbacks,
receivers still scored 0.75 a game under projection (0.16 otherwise); tight
ends and backs showed nothing. A smooth factor on the quality gap failed
(shrinkage flatters backups with few attempts), but a plain cut on those
weeks -- `cut` x the share of his recent games someone else threw --
passed, picked on three seasons and scored on the fourth:

| new starter | weeks | RMSE | seasons better | cut picked |
|---|---|---|---|---|
| worse by 0.25+ sd | 1,024 | 5.405 -> 5.319 | 3 of 4 | 0.15-0.20 |
| **worse by 0.5+ sd** | 643 | 5.372 -> 5.235 | **4 of 4** | 0.15-0.20 |
| worse by 0.75+ sd | 434 | 5.239 -> 5.078 | 4 of 4 | 0.15-0.225 |
| TE, worse by 0.5+ sd | 382 | 4.622 -> 4.641 | 2 of 4 | -- |

In use for receivers: 0.5 sd, cut 0.15 (the low end of what every fold
picked), this week only (`roster.qb_change`). This week's starter is the
first quarterback on the Sleeper depth chart who is not out or on a bye.
At week 5 of 2026 it moves Baltimore's receivers 10-15% with Tyler Huntley
in for Lamar Jackson; the card names the starter.

### Every player we project is graded
The projection log (`results/<profile>/skill_log.parquet`) kept every
rostered player and the free agents with betting props -- 100-435 player-weeks
a week -- so the waiver players you actually choose between, the ones with
no props, were never graded. It now keeps the whole wire except players on
no team, out for the season or unsigned. Grading already counts only players
with a box score that week, so the extra rows cost nothing in accuracy; they
give the blend weights and the scorecard two to three times the player-weeks
to learn from, and show how the model does on the waiver pool.

### Correcting the projection in-season? Tested, no
The blend weights and every input re-fit each week, but our own projection
was never corrected by how it had been missing this season. Tested
(`scripts/fit_in_season.py`): 2022-2025 replayed week by week through the
production volume model and its 80% range, each week corrected only from
that season's earlier weeks at the position, the shrinkage picked on three
seasons and scored on the fourth:

| | bias shift, RMSE | seasons | spread scale, quantile loss | seasons |
|---|---|---|---|---|
| QB | 7.708 -> 7.708 | 0 of 4 | 2.6723 -> 2.6722 | 3 of 4 |
| RB | 6.211 -> 6.211 | 0 of 4 | 2.0326 -> 2.0335 | 0 of 4 |
| WR | 6.129 -> 6.128 | 2 of 4 | 2.0039 -> 2.0039 | 0 of 4 |
| TE | 4.948 -> 4.948 | 0 of 4 | 1.6134 -> 1.6135 | 0 of 4 |

A season's lean was small every year (within about +-0.5 a game) and did not
carry from early weeks to later ones; the fits chose heavy shrinkage. The
80% ranges already held 78-85% of scores. The QB spread "pass" is a
1-in-27,000 change: not shipped. What adapts in-season stays as it is: the
position rates, each player's window, and the blend weights.

### Last season's stand-in games
Colby Parkinson projected 6.7 on 1-2 targets a game: late in 2025 he stood
in for an injured Tyler Higbee (4-7 targets, nine touchdowns in twelve
games), and with three thin games this season those still carried his
17-game window. The teammate-return correction read only this season's
absences. It now reads last season's too (`teammate_back.prior_season`), and
only where it is the same situation:

- his own game, for the team he plays for now;
- a regular teammate at his position (5+ expected points a game) who is on
  that team now and has played for it this season;
- who was on that team's **injured reserve or game-day inactive list that
  week** (nflverse weekly rosters) -- a documented absence while on the team,
  so weeks before a signing or after a trade or cut never count.

On/off at the shares already fitted (nothing new tuned), walk-forward
2023-2025 (`scripts/fit_teammate_back_prior.py`), on the player-weeks it
moves:

| | moved | RMSE | seasons better |
|---|---|---|---|
| RB | 498 of 3,274 | 6.029 -> 5.956 | 3 of 3 (bias -1.32 -> -1.06, -0.42 -> -0.06, -0.44 -> -0.18) |
| WR | 1,075 of 5,334 | 5.951 -> 5.933 | 3 of 3 |
| TE | 272 of 2,304 | 5.125 -> 5.094 | 2 of 3 |

At week 5 of 2026 the players it moves are last season's stand-ins whose
starter is back: Devin Singletary (Cam Skattebo) 7.7 -> 4.5, Audric Estime
(Alvin Kamara) 10.5 -> 7.7, Darius Slayton (Malik Nabers) 8.1 -> 6.3, Luke
Musgrave (Tucker Kraft), Parkinson (Higbee) 7.4 -> 6.8.

### A receiver back takes targets from the tight end? Tested, no
Tyler Higbee's role jumped to 11 targets in a week Puka Nacua missed, then 6
with Nacua back; the teammate-return correction only reads returns at the
player's own position, so Nacua's return did not lower Higbee. A `cross`
share was added -- a returning regular at the other receiving position takes
back that share of his volume -- and fitted the usual way
(`scripts/fit_teammate_back.py --cross`, same-position shares held at the
config's, picked on three seasons, scored on the fourth):

| | player-weeks it moves | held-out RMSE | seasons better | best share |
|---|---|---|---|---|
| TE (a receiver back) | 2,015 | 5.562 -> 5.562 | 0 of 4 | 0 in every season |
| WR (a tight end back) | 2,183 | 6.662 -> 6.660 | 1 of 4 | 0.05-0.15 |

Tight ends' projections were not too high after a receiver came back -- if
anything slightly low (bias +0.2) -- so the two-game form the volume model
reads already fades a one-week bump enough. Rejected; `cross` is left unset
(the code reads it, so a later refit can turn it on).

### Trades with a team you pick
The trade lists priced the league's 24 most promising offers on each of
three orderings, at most five per team, so a team that never made those
lists had nothing priced -- and you could not ask "what is the best I can
do with Team Morris?". Each team's six most promising screened offers (by
the chance of a yes times what they add to your roster) are now priced as
well, and the Trades tab has a fourth view, **By team**: one card per team,
best offer first, its up-to-five best trades by expected gain, or a line
saying none clearly raises your title odds. On the ESPN league that prices
82 offers instead of about 50, in under half a minute.

### An offer his owner would laugh at
The Yahoo board's second trade was Xavier Worthy for Kenyon Sadiq and
Tyreek Hill, +3.7 points of title odds, "chance of a yes ~62%" -- for a
rookie tight end the waiver columns called the top add of the week (20% to
29% rostered overnight) and a receiver they called droppable. Four things
were wrong, three of them in how he was read:

- **Hill was not on a team.** ESPN listed him with none; Yahoo tags such
  players NA (not active) and keeps their old team, so on Yahoo he was a
  Dolphin with an injury tag, projected from his 2025 games at 12.4 a
  game. That value made the trade look like +3.7 for you, and put him in
  four more of the top offers as a "buy low" throw-in. NA now means not on
  an NFL roster, like no team at all: no value until the platform projects
  him (or he signs), and the card says so.
- **"The best player in the deal" was any lead at all.** Worthy was 0.4 a
  game ahead of Sadiq in the market's eyes and earned the full consolidation
  shift. It now grows with the lead and is full only at 1.7 points a game
  (about the spread of managers' opinions of one player, the scale of the
  rostership fit).
- **His lineup was ignored.** Punisher started Sadiq and Darren Waller in
  his flex spots over a healthy Brian Thomas Jr.; to him each is worth at
  least what the market gives Thomas, whatever it gives them. A player he
  starts over a healthy bench option who could fill that slot is now valued
  at least at that option's market value. Only raised: a lineup can be a
  week old, set before the news, so a bench player is never marked down for
  sitting there. The card says it ("he starts Kenyon Sadiq over Brian Thomas
  Jr., so he rates him above the market's 8.9 a game").
- **No offer is a sure thing.** At a large perceived gain the curve reached
  the manager's engagement, 97% for an active one. Offers go unseen, get
  countered, and read worse because a rival proposed them (reactive
  devaluation), so the ceiling is now 80% -- an assumption, like the rest
  of the curve.

On the same snapshot, without the FantasyPros blend the runner adds, the
offer now comes out at 17%: 30% on the market's view alone, and his lineup
takes it the rest of the way. What the market still says is worth keeping
in view: Worthy is rostered in 65% of Yahoo leagues and Sadiq in 29%, so the
broad market has not caught up with the columns. A manager who reads them
is harder to sell to than the blend says, and the rostership trend is not
in it yet (every sync's snapshot is committed, so the history to fit it on
is building).

### Evaluate any trade, on the page
The trade lists are the engine's picks. A trade you are offered, or one you
want to offer, is usually not on them, so the Trades section also has an
evaluator: pick a team, tick players on both sides (any number), and both
teams' title and playoff odds, weekly points, lineups, forced drops, his
view of the deal and his chance of a yes appear as you tick.

A page cannot play 6,000 seasons per click, so the work is split:

- **Each team's odds against its weekly points** come from the full
  simulation at publish time: every team's scores shifted by -8 to +8
  points a week, every team's title and playoff odds read at each shift.
- **What a trade does to a team's weekly points** is simulated in the
  browser, 1,000 seasons in about 50 ms, from each player's week-by-week
  chance of playing, expected points when he does, and how far that
  expectation spreads across the seasons -- all summarised from the same
  simulation, with the best lineup set each week.

A trade moves each team's odds by its own weekly change and the other
team's (the games they play, the places they chase), read off the curves.

The obvious shortcut -- the change in static season values -- does not
work: tested on 120 trades priced exactly (Yahoo, week 4), it tracked the
exact change in your title odds at a correlation of only 0.45, because it
ignores byes and injuries and overstated a trade's weekly effect by about
half. The simulation's own weekly change, read off the curves, tracked it
at 0.89. Shipping the simulation itself would need about 2 MB per league
per refresh for a similar result (400 seasons: 0.84), committed with every
page. The summary above, 110-160 KB, comes to 0.81 (Yahoo) and 0.89 (ESPN);
without the spread of each player's expectation it was 0.53, so the spread
is what carries it. Each refresh re-measures
it on the trades the finder priced exactly, and the page shows the typical
error: at publish on week 4, 0.6-0.8 points of title odds against a full
simulation whose own noise on a trade is about 0.4. When the ticked trade
is one the finder priced, the exact figures are shown beside the estimate.

The page's script mirrors a Python reference (`trade_eval.py`) line for
line, on the same data and the same random numbers (a hash of player,
season and week, so both languages draw identical seasons); a test runs
both on the same trades and requires every number and every reason to
match. Building it found one bug in the trade finder: a two-for-one could
drop a team's only D/ST to make room, because roster minimums were checked
only at the skill positions. They now cover every position, capped at what
the roster already had (a league without kickers), as waiver plans do.

### Yahoo, on the same engine
The Yahoo sync now reads what the simulator needs, from pages a logged-in
browser already gets:

- **Settings.** The rules table states the playoffs in one line -- "6
  teams - Week 15, 16 and 17" -- which gives the field, the regular
  season (everything before the first playoff week) and the round length.
  Also the waiver type (a continual rolling list), reseeding, and whether
  a median game is played.
- **Waiver order and activity.** The standings table carries each team's
  waiver priority and its pickups so far -- the same two numbers ESPN
  gives, so priority cost and rival claim odds work unchanged.
- **Every roster**, one team page each.
- **The schedule**, from the league page's matchup list, one page per
  remaining week. It is read once a season and reused from earlier
  snapshots, and kept only when every remaining week is complete: a week
  with games missing would hand out too few wins and skew every team.

Two rules the ESPN league does not have, now modelled for both:

- **Reseeding.** After each playoff round the best seed left plays the
  worst seed left. A fixed bracket would send the top seed into the 4/5
  winner even after the 6 seed pulls an upset.
- **Only a rolling list charges for a claim.** On an order that resets by
  standings each week (or FAAB), a claim costs no future priority.

A fourteen-team league is about 260 simulated players; 6,000 seasons and the
candidate moves take under a minute and about 1 GB.

The probe that found these pages printed table headers and page titles
without scrubbing them, which put two managers' first names and the league
name in a public workflow log. Both now pass the same whitelist as the rest
of the probe output, and a test holds them to it.


### Playoff-week factors for held units, checked on their own

The D/ST and kicker cards that say "held for the playoff weeks" rest on the
schedule factor for weeks 15-17 as it stands in October. The original fit
(`fit_schedule_dstk.py`) scored every later week; `scripts/fit_playoff_units.py`
scores only weeks 15-17, from checkpoints 4-9, 2023-2025:

- **D/ST: keeps.** RMSE 8.784 flat -> 8.551 at the production exponents
  (better 2 of 3 seasons, 2023 a hair worse); a shrink fitted held-out on
  the other seasons' playoff weeks comes out near 1 (0.95 on all), so the
  factor is not overstated. The five units with the best playoff schedule
  scored 15%, 20% and 38% over their baseline relative to the rest, in each
  season.
- **K: thin.** RMSE 4.741 -> 4.728 (2 of 3), but the factor barely moves
  (sd 0.045) and the best-schedule kickers beat the rest only in 2025. A
  kicker card leaning on its playoff weeks is leaning on very little; left
  as is because it rarely decides one.

Even where the matchups are real, a defence stashed in October holds a
bench spot for two months, and the simulation prices that spot too cheaply.
So playoff-week defences and kickers are now priced only from three weeks
before the regular season ends (week 12 of 14), or earlier once the team's
playoff odds reach 95% (`roster.playoff_units` in config.yaml). Before that
they show up only as next-week streamers.

A card whose edge is inside the simulation's noise was labelled "Close
call: lean claim", which read as advice. It is now "Toss-up: no clear
edge", and its note says the simulation cannot tell it from standing pat.

### Kicker wind forecast: tested on the runner, dropped

`wind-fit` (Open-Meteo archived forecasts, 2023-2025, calibrated
leave-one-season-out: RMSE vs recorded wind 5.17 raw, 4.96 calibrated)
against the stadium's typical wind: MAE 3.7197 -> 3.7193, rank correlation
0.171 -> 0.173, better in 0 of 3 seasons on both measures. Windy games
(forecast 12+ mph) improved, but there were 9. No `wind_forecast.json` is
committed, so the forecast module stays inert and kickers keep the typical
wind fill.

### Kicker drive stalls: tested, not kept

Subvertadown frames kickers by where drives stall. The kicker model
already carries field-goal tries per drive, red-zone touchdown rate and
field goals' share of scores; `scripts/fit_kicker_stall.py` adds the split
behind tries per drive, each a shrunk, leak-free team rate set against the
opposing defence:

- **range**: drives reaching the opponent's 35 (about 45% of drives);
- **stall**: the share of those that end without a touchdown;
- **kick**: the share of those stalls that end in a field-goal try, which
  captures fourth-down habits and how far the coach trusts the kicker;
- **long stalls**: drives dying between the 25 and the 35.

From these it builds expected tries per game. Walk-forward 2023-2025:

| Variant | MAE | Rank corr | Top-5 hit |
|---|---|---|---|
| Production | 3.7187 | 0.1743 | 0.552 |
| Plus expected tries | 3.7185 (2 of 3 better) | 0.1764 | 0.552 |
| Plus every component | 3.7198 (0 of 3 better) | 0.1766 | 0.541 |

The composite clears the letter of the rule by two ten-thousandths of a
point, which is noise. Its correlation with the tries actually taken is
-0.006, so it has no signal for the thing it estimates; the team's implied
total manages only +0.011. A kicker's tries in any one game are close to
unpredictable from what is known beforehand, and what can be known is
already in the Vegas total. Not shipped.

The first run counted kickoffs, which nflverse attaches to the kicking
team's drive at the 35, so 72% of drives "reached range". The script uses
snaps, tries and punts only. The production red-zone flag (the 20) is
essentially untouched by this.

### Similar past matchups: kept for kickers, not for defences

Subvertadown shows each kicker and defence the most similar historical
matchups and what they scored. `scripts/fit_analogs.py` tests that as a
forecast. For every week, each unit is matched to its `k` nearest earlier
games on standardised pre-game features, its own and the opponent's, and
takes their mean score. That mean is scored alone and blended with the
shipped model's walk-forward projection; `k` and the weight were picked on
two seasons and scored on the third.

- **D/ST: dropped.** Every variant (Vegas features only, or with team
  ratings; pool from 2021 or from 2015) was worse than the model in all
  three seasons, or no better. The held-out weight came out at 0-0.15.
- **K: kept, as half and half.** Matching on implied total, spread, total,
  dome, field-goal tries per drive, red-zone touchdown rate for and
  against, and the kicker's long-try share, with 100 neighbours from 2015
  on. Production features, MAE (rank correlation):

| Season | Model | Half and half | Analog alone |
|---|---|---|---|
| 2023 | 3.606 (0.156) | 3.571 (0.181) | 3.558 (0.162) |
| 2024 | 3.823 (0.189) | 3.796 (0.227) | 3.779 (0.230) |
| 2025 | 3.737 (0.177) | 3.733 (0.164) | 3.748 (0.136) |
| 2026, weeks 1-5 | 3.810 (0.017) | 3.789 (0.034) | 3.774 (0.041) |

  The held-out weight chose 0.8-1.0, but the analog alone lost in 2025, the
  latest full season. Half and half is better on MAE in all four seasons
  and on rank correlation in three, so that is what ships. It is not a bias
  fix: a linear recalibration of the model alone made it worse (3.737 over
  2023-2025). The pool runs from 2015 because the 2021-on pool lost.

The pool is built each run from play-by-play back to `since` (six extra
seasons, cached with the rest). If those cannot load, the pool falls back to
the training seasons. A kicker without lines keeps the model's number. Each
kicker's line on the page names the similar-matchup average. The backtest
and model selection still grade the model alone. Config `model.analogs`.

### Sleeper projections: kept, as a last blend step

Sleeper publishes weekly PPR projections for every skill player, free, and
serves past weeks. `scripts/fit_sleeper_projections.py` (runner job
`sleeper-fit`; the sandbox cannot reach the host) graded them on 2023-2025
against our production model replayed on the games before each week
(weeks 4-10, 12, 14; 1,205 players who played and were projected 3+):

| | MAE | Rank corr | Bias |
|---|---|---|---|
| Our model alone | 5.168 | 0.608 | -0.11 |
| Sleeper | 5.119 | 0.625 | +0.50 |
| Blend, weight fitted on the other seasons (0.45-0.60) | 5.045 | 0.628 | |

The blend is better in all three seasons (5.146 -> 5.088, 5.202 -> 5.082,
5.154 -> 4.966). By position Sleeper is better for RB, WR and TE and worse
for QB (6.90 against 6.53), where the 50/50 is level with ours.

This season, weeks 3-5 (108 graded rows), against the published blend
(model, platforms, betting props and FantasyPros): published 4.633, Sleeper
alone 4.592, and the published blend moved toward Sleeper by 10% / 20% /
30% / 50% gave 4.622 / 4.612 / 4.603 / 4.586.

Two reasons for only 0.25 in the config, not the fitted 0.5:

- **The historical test compared against our model alone.** The platform,
  betting and FantasyPros numbers cannot be reproduced for past weeks, so how
  much Sleeper adds *on top of them* rests on 108 rows.
- **The archived projections may not be strictly pre-kickoff.** Past weeks
  were pulled after the fact; Sleeper may have updated them for who actually
  played. Players who did not play are excluded from the grade, which limits
  but does not remove that. The live number is pulled fresh at each refresh,
  before kickoff.

So it enters as one more step in `blend_market`: after the betting and
FantasyPros steps, on the if-he-plays number, the blended projection moves
`roster.sleeper_projection.weight` of the way to Sleeper's. The number it
saw (`sleeper_projection`) and its weight are logged with every projection
(`skill_log.parquet`) and graded on the scorecard as another source, so
the weight can be refit on genuine pre-kickoff values once enough weeks
are in (the same gate as the betting weight). A player Sleeper does not
cover is left as before.

#### Matching by name: Sleeper had covered one player in five

Cade Otton was recommended as a start while he was a free agent, then lost
to Tyler Higbee the moment he was added. His own number never moved (7.67
before and after). Higbee's did: 7.25 -> 8.52, when Sleeper raised its
projection for him from 4.4 to 9.4. Otton had no Sleeper number at all. The
reason: Sleeper's projections were matched through the `gsis_id` in its
player database, which is filled mostly for veterans. In the week-6 log,
only 20% of projected players had a Sleeper number (Higbee, a 2016 draftee,
did; Otton, 2022, did not), so the blend's last step moved some players and
not others.

Rows without a `gsis_id` are now matched by name on the same team and
position, as the depth charts already were (`sleeper.match`); a name shared
by two players on one team and position is left unmatched. The fit script
uses the same matcher, and was rerun on everyone (2023-2025, 4,063 graded
rows, 20-35% matched by id, the rest by name):

| | MAE | Rank corr |
|---|---|---|
| Our model alone | 5.071 | 0.599 |
| Sleeper | 4.922 | 0.640 |
| Blend, weight fitted on the other seasons (0.70) | 4.886 | 0.640 |

Better in all three seasons (4.898 -> 4.740, 5.105 -> 4.901, 5.210 -> 5.016).
The newly matched players gain more than the veterans did (5.017 -> 4.793,
against 5.186 -> 5.098). This season, weeks 3-6 (618 rows), moving the
published blend toward Sleeper by 0 / 10 / 20 / 30 / 50% gave 5.332 / 5.327 /
5.326 / 5.325 / 5.329. The weight stays at 0.25: the best published-blend
gain is at 20-30%.

Free agents and rostered players otherwise go through the same pipeline:
the same model, platform blend, FantasyPros projection and betting props. The
props are bought for the games holding your players in either league, which
this week covered 26 teams; a free agent elsewhere is blended without them.

### Player comparisons: kept for quarterbacks only

The kicker comparison worked (similar past matchups), so
`scripts/fit_player_comps.py` asks the same of every skill position. Each
player-week is described by what was known before it: expected points per
game over his last 4 and 8 games, PPR points, targets, carries and target
share over the last 4, and his team's implied total and spread. Its
forecast is the mean score of the `k` nearest earlier player-weeks at his
position, in a pool back to 2015. It was judged against our production
model replayed on the games before each week, on weeks 4-14 (every other
week) of 2023-2025, for players who played and were projected 3+
(4,145 rows). `k` and the weight were picked on two seasons and scored on
the third.

- **All positions together: dropped.** MAE 5.036 -> 5.036, better in 1 of 3
  seasons. The comparisons alone were worse than the model at every `k`.
- **By position:** QB 6.334 -> 6.187 (2 of 3 seasons; rank correlation
  0.42 -> 0.45). RB, WR and TE were each slightly worse.

QB was found by slicing, so it was then tested on weeks it had not been
seen on: the odd weeks 5-13 of 2023-2025 and weeks 3-6 of 2026, with the
setting fixed in advance at `k` 100 and weight 0.8:

| Season | Model | 80% comparison | Comparison alone |
|---|---|---|---|
| 2023 | 5.904 | 5.834 | 5.839 |
| 2024 | 6.256 | 6.115 | 6.139 |
| 2025 | 7.023 | 7.045 | 7.096 |
| 2026 | 6.222 | 5.933 | 5.881 |

That is better in three of four seasons (6.351 -> 6.232), with rank
correlation up in three. It ships for quarterbacks. The model's own number
moves 80% of the way to the comparison mean, before the platform, betting,
FantasyPros and Sleeper blends, which is where the test measured it. Replayed
through the production code, 2024 went 6.256 -> 6.070 and 2026 went
6.222 -> 5.821, with 97% of quarterbacks getting a comparison. Each
quarterback's card says what similar past weeks averaged. Config
`roster.comps`.

Because the final number blends in other sources, the change to what is
published is smaller than the model-level gain. The quarterback model also
under-projected in these replays (bias -0.5 to -1.2), which is worth a look
on its own.

### Yahoo sign-in expiry alert

The Yahoo cookie lapses about weekly, and until then the Yahoo tab silently
showed the last good sync. The `Sync leagues` step now keeps its output; if
the sync failed with "Yahoo session has expired", a follow-up step opens one
issue titled "Yahoo sign-in expired" (never a second while one is open), which
the GitHub mobile app pushes to a phone, and a later successful sync closes
it. The workflow gains `issues: write`. The issue holds instructions only,
no cookie value.

### Rest-of-season comparisons: kept, at 20%

The weekly player comparisons helped only quarterbacks; this asks the same
of the season value, which drives waivers, trades and title odds.
`scripts/fit_ros_comps.py` describes each player at a checkpoint by his
games before it: games played, points and expected points per game, targets,
carries and target share; his last four games; last season's points and
expected points per game; career games since 2015; and the week. Its
forecast is how the 100 player-checkpoints most like him at his position
finished (rest-of-season points per game). Only earlier seasons are used,
because a same-season player's finish is still in the future at the
checkpoint. Graded against our season value replayed through `ros_backtest`
at weeks 3, 4, 5, 6, 8 and 10 of 2022-2025: players without an out or IR
tag, valued 3+, with 3+ games left (5,801 player-checkpoints). Weight and
`k` were picked on three seasons and scored on the fourth.

| Season | Ours | Blend | Rank corr |
|---|---|---|---|
| 2022 | 2.633 | 2.617 | 0.708 -> 0.713 |
| 2023 | 2.623 | 2.628 | 0.751 -> 0.751 |
| 2024 | 2.905 | 2.900 | 0.714 -> 0.716 |
| 2025 | 2.693 | 2.670 | 0.680 -> 0.683 |

That is MAE 2.713 -> 2.704 per game, better in three of four seasons. The
held-out weight was 0.20-0.25 every time. The comparisons alone were worse
than ours (2.83 at best).

The gain is small and borderline: 0.0125 a game, about two standard errors
when clustered by player-season. It is concentrated where it matters. For
players we value at 14+ a game the gain is 0.058; under 7 a game it is
-0.009. By checkpoint it is largest at weeks 3-5 and gone by week 10. So
it is applied as tested and no further:

- 20% toward the comparison mean;
- weeks 3-10 only;
- players without an out or IR tag;
- on the model's season value, before the FantasyPros consensus blend.

Replayed through the production code, 2025 went 2.693 -> 2.670 and 2024
went 2.905 -> 2.894, moving 98% of players. A card notes the comparison when
it is a point or more from our value. The finished-season pool is cached in
`data/raw` and costs about 1.3 seconds a projection. Config
`roster.ros_comps`. By position only WR passed on its own; the pooled
version is what ships. `scripts/fit_ros_comps.py` and
`scripts/fit_player_comps.py` turn their own comparison off for the
baseline, since the page now applies it.

### Late swaps: a covered questionable starter is worth more

The lineup simulation treated a questionable starter as a coin flip: in the
draws where he sat, his slot scored zero. But inactives come out about 90
minutes before each game, and both leagues lock players game by game. If
your bench has an eligible player whose game has not started by then, a
starter ruled out is simply swapped. So a questionable player in a 4:25
game with a 4:25 or Sunday-night backup costs almost nothing when he sits.
A questionable player in the 1:00 window, with only 1:00 backups, still
costs his whole slot.

The optimiser now keeps, for every draw, whether each questionable starter
sat (`sample_points(..., sits=)`). For every lineup it pairs each
questionable starter with the best bench player who could still replace
him: eligible for his slot, not out, not in an IR slot, and kicking off no
earlier than 90 minutes before him. Each bench player covers at most one
starter. In the draws where the starter sat, the bench player's draw is
added. This applies to your side only; the opponent's lineup is taken as
set. The recommended lineup's swaps are spelled out on the page ("if X is
ruled out, start Y in his place"). The win odds count on the user making
that swap.

Kickoffs come from the nflverse schedule already used for locks
(`PlayerRow.kickoff`). One limitation: the page's hand lineup editor scores
from the same draws without the swap, so for a lineup with a covered
questionable starter it shows slightly lower odds than the recommendation.

### Live scores in the browser

During games the page used each starter's pre-game projection until nflverse
published the final stats, often a day later. More full refreshes would have
been the wrong fix: 6-8 minutes each, more Yahoo page loads on your cookie,
and every projection rebuilt when only the scores moved.

The page now reads live scores itself, with no workflow runs. ESPN's public NFL
scoreboard and game summaries answer browsers on other sites
(`Access-Control-Allow-Origin: *`, probed with the `live-probe` job).
`roster/live.py` puts a live card at the top of the Lineup tab. It carries both
sides' starters (your lineup as set and your opponent's), their ESPN athlete
ids (ESPN's league ids are ESPN's; Yahoo's are matched through nflverse's
roster table, with the name as a fallback), the league's scoring rules, and
2,000 of the draws that priced the matchup, starter by starter.
`static/live.js` turns each box score into fantasy points with those rules
and rescores the draws:

- a finished game counts what he scored;
- one not started counts the draw;
- one in progress counts his points so far plus the draw's share of the time
  left: the mean scaled by the share left, the spread by its square root;
- a defence in progress is blended instead: the current points (its
  points-allowed tier moves with the score) weighed against the draw by the
  time left.

P(win) is the share of draws you finish ahead, shown beside the pre-game
figure, with both scores so far and the projected finals. It polls once a
minute while the page is open and a game is on, and stops when every game is
final. The card stays hidden until a game has started.

Checked against week 5's fifteen finished games (`live-check` job,
agreement only):

- skill players against nflverse's PPR points: 99.4% exact (336 players); two
  were 2 points low, probably two-point conversions written differently;
- kickers against our recorded actuals: 100% (30);
- defences: 90% (30); three were 2 points high.

Blocked kicks and the return unit's own muffed punts are not in the box score,
so they are read from the play-by-play. The card says the platform's own score
is the final word.

### Live P(win): early pace does not predict the rest of the game

One idea was to weight each live draw by how well it fits the game so far: a
quarterback on a hot pace would make the draws where he is good count more,
and through them, his receivers' draws. Tested on 2022-2025 play-by-play
(13,864 player-games with a projection of 5 or more; fantasy points rebuilt
quarter by quarter matched nflverse's finals exactly for 98.8% of games). At
the end of each quarter, the final score was predicted from the points so
far and the pre-game projection, and graded by CRPS (the whole predicted
range, in points):

| end of | as built | weighted, strongest tested | weakest tested |
|---|---|---|---|
| Q1 | **3.44** | 4.32 | 3.44 (a tie) |
| Q2 | **2.70** | 3.78 | 2.70 (a tie) |
| Q3 | **1.93** | 2.90 | 1.93 (a tie) |

Every weighting did worse, in every season; the lighter the weight, the
closer it came to the method as built, and it never beat it. A player's early
points say almost nothing about the rest of his game beyond his projection:
the points already scored count, and what is left is about what was expected.

The same data checked how the method as built shrinks the spread for the time
left. Scaling it by the square root of the share remaining beat powers of
0.35, 0.65, 0.8 and 1.0 in every season (CRPS 2.66-2.77 by season, against
2.67-2.84 for the alternatives). So the live P(win) is unchanged.

### Yahoo's free agents: the top 150 offensive players, not 50

The Yahoo sync read two pages of the offensive free-agent list (the top 50
by this week's projection), plus the top kickers and defences. In a 14-team
league the wire is thin: the Week 6 snapshot held 100 free agents, with only
6 running backs and 13 tight ends, so a sleeper with a modest projection
never reached the Yahoo waiver advice or the title engine's pickup list. It
now reads up to six pages (`yahoo_web.OFFENSE_PAGES`, offsets 0 to 125) and
stops at the first page with nobody new, so the list ends where Yahoo's does.
Players are de-duplicated by id. About four more page loads per sync, at the
same pace as before. Nothing else about how free agents are priced changed.

### Players on IR can be traded

The trade finder and the page's trade evaluator left out every player in an
IR slot, on both sides, though both platforms let an injured player be
traded. Now they are offered and asked for like anyone else, and roster
room is counted in active spots (`trades._active_change`, mirrored in
`trade_eval.active_change` and its script): an incoming player who can sit
on IR takes a free IR slot instead of forcing a drop, and an outgoing one
from IR frees no bench spot. The leagues' IR-slot count is not in the
synced settings, so it is the most any team is using (`ir_capacity`: 2 in
both leagues today), a lower bound. Season values and the simulation already
carry an injured player's return (his reported timeline or the measured
absence), so what he is worth in a trade is priced as before.

### A power ranking beside the title odds

Title odds are not "the best teams": they mix roster strength with the
record already banked, the schedule left, byes and seeding. A team at 6-1
with an average roster can out-rate a 3-4 team with the league's best.
`SeasonOdds.power` is the record-blind answer: each team's all-play win rate
over the rest of the regular season (the share of the other teams' scores it
beats in an average simulated week, ties half), from the same simulation and
the same `strength_fade` as the odds, so early in the season it does not
exaggerate gaps that waivers will close. The season table gains a Power
column, rank and percentage. It orders teams the way the weekly score
distribution does, so a steady team and a boom-or-bust one with the same
mean differ only a little; expected points a week would not separate them at
all. Graded loosely: real all-play results each week are available to check
it against as the season goes (not done yet).

The table also shows the expected points a week for the rest of the regular season
(`pts_ahead`, after the fade) and the points a week scored so far (`pts_so_far`): a big gap
between them means the roster has changed (injuries, a trade) or the points came on luck.
CMoney (Yahoo) on 2026-10-26: 134.5 so far, 110.0 ahead, from a roster that lost no one but
has Samuel, Croskey-Merritt and Montgomery out.

### Logging the lottery tickets, to test prop prices and FantasyPros ranges later

The ticket's big-week chance already rests on the blended projection (our
model, the betting market's props where he has them, FantasyPros, Sleeper).
What could still sharpen it is the market's touchdown price for a low-
projected back or tight end, or FantasyPros' high/low range for a backup
with an uncertain role. Neither has history for past seasons (the props log
began this season and skips most deep players), so they cannot be validated
on 2022-2025 and are not used.

Every shown ticket is now logged (`lottery.log_tickets`, from the publish
run for your team: `results/<profile>/ticket_log.parquet`, one row per
season, week, window and player, replaced until kickoff) with the model
projection, FantasyPros' projection, the props mean and stat lines, %
rostered, the bar and the three chances. `lottery.grade_tickets` joins the
week's real points and reports the big-week chance said against the hit rate
by band. After a few hundred ticket-weeks (about 4-6 weeks), test whether
the prop price or the expert range beat the current shape held-out by
season, and switch over only if they do.

### Lottery tickets: where the percentages come from, and two fixes

"Chance of a top-14 week" is read off the player's projection and the same
outcome distribution the lineup simulator uses, against a bar: the 14th-best
score at his position in an average week, 2021-2025 (RB 16.0, WR 18.4, TE
9.8, QB 16.7), injury tag included. "Next in line" is the league-wide chance
a lead back misses the next game (8.5%, 2021-2025) and "Keep?" combines the two.

**Graded** on 2022-2025 replay projections of 0.5-12 points (n = 17,229),
predicted against hit, by position and projection band: receivers, tight
ends and backs within about a point, QBs about half too high (QB projected
4-10: 7-16% predicted, 5-9% hit).

**Two changes.**
- `boom_scale: {QB: 0.65}` (config). Scale picked on three seasons and scored
  on the fourth: 0.60-0.70, log loss 0.2391 -> 0.2306, better in 4 of 4.
  Backs (best scale 0.95, better 1 of 4), tight ends (1.1, 3 of 4) and
  receivers (1.15, 3 of 4) gained under 0.2% and are left alone.
- The handcuff's 8.5% is now the lead back's own: odds-scaled by his
  `hazard_mult` (the season simulation's per-player absence hazard, from his
  record of missed games). A lead who has missed games lately says so on the
  card.

Still a league-wide average per lead back for the job-opens part otherwise,
and "keep" assumes a spike without more opportunity behind it fades.

### Playoff and title odds during your game

The live card (static/live.js) already rescored your matchup as games went
on. It now also shows your playoff and title odds with this game going the
way the live P(win) says. The season model prices each remaining game both
ways (`Stakes`: P(playoffs) and P(title) if you win it and if you lose it,
everything else held as it fell). The card carries this week's pair
(`live.season_stakes`) and moves the odds along the line between them by how
far the live P(win) has come from its kickoff value. Nothing moves before
kickoff, a certain win or loss lands on the "if you win/lose" figure, and
it never goes outside 0-100%.

It is this game only. Other matchups and the rest of the season are held as
modelled until the next refresh counts finished games (`locked.py`: a game
final and published locks its players' scores into the season simulation).
Not shown in a playoff week, for a team the season model does not cover, or
when the stakes are missing. Not graded: the live P(win) is the raw draw,
without the 0.78 calibration of `lineup.calibrated`, so it reads a little
surer than the lineup tab's.

### Defences and kickers in the replay, and a calibrated P(win)

The replayed leagues now have a D/ST and a K slot. Each team streams both
every week, worst record first, best projected first, from the shipped
streaming models' walk-forward projections. Every 2022-2025 unit-week is
projected from earlier games only (`backtest.walk_forward`, the selected
estimator and penalty; `league_replay.unit_rows`). Free units are the
wire's replacement level in the season model, and a unit's link to the QB
facing it is drawn as in production.

**The units themselves hold up.** Points on projection within a week: D/ST
1.07 +- 0.06, K 1.03 +- 0.15. The ten best projected each week scored
above their projection (D/ST 13.6 against 13.2, K 8.8 against 8.6), so
streaming the top of the list carries no selection bias.

**This week's P(win) was too sure of the favourite.** With full lineups,
favourites won 59.1% against 61.4% predicted; the 60-70% bin won 59.6%
(95% interval 57.6-61.6%) against 64.3%. The margin's spread is right
(9.8 / 24.5 / 50.3 / 75.1 / 89.9% of results under the predicted 10 / 25 /
50 / 75 / 90th percentiles), but the projected gap between two lineups holds
up only 0.83 of the way. That gap was already there on the skill side (0.84);
the units add variance, not bias. Skill-only, a pull toward 50% did not pass;
with the units in, it does.

`lineup.calibrated` maps p to Phi(k Phi^-1(p)), `WIN_SHRINK` k = 0.78. Held
out by season, each fold picked 0.76-0.82, Brier 0.2373 -> 0.2366 and log
loss 0.6673 -> 0.6657, better in all four. It applies to every P(win) the
lineup code reports: the optimiser, swap explanations, stream and pickup
options, and the in-page lineup editor (the same mapping in its script).
It is monotone, so the chosen lineup never changes, though gains near 50%
read about a fifth smaller. The live in-game P(win) is not mapped: it was not
graded.

The odds check now streams units too (`--units`; `--no-units` for the old
setup), and its baseline was rewritten: P(win) 0.2366, P(playoffs) 0.1653,
P(title) 0.0878 over 6 leagues a season. The first run on GitHub took 6
minutes, not 20.

### Rivals competing for the same waiver players: need still does not predict

Re-tested "Who actually gets picked up" with the weeks collected since. Every
unowned QB/RB/WR/TE in week w, and whether another team held him in
week w+1 (ESPN 2,735 player-weeks, 2.2% added; Yahoo 316, 10.1%). The claim
chance by % rostered, against the same plus the share of rivals he would
start for (`would_start`), leave-one-week-out log loss:

| | % rostered | + rival need | + season value |
|---|---|---|---|
| ESPN | 0.08912 | 0.08909 | 0.08909 |
| Yahoo | 0.33595 | 0.33491 | 0.33491 |

Need adds nothing in ESPN (0.00003) and the coefficient is negative in both
(ESPN -0.9, Yahoo -4.0), the wrong way for competition for the same players.
Yahoo's 0.001 comes from a negative sign on 316 player-weeks, not demand.
Managers follow the platform's % rostered, not their own holes. So no
rival-need term is added to `claim_odds`; each target's chance stays the
fitted rate by % rostered, the ones ahead of you in the waiver order taking
him in proportion. Worth rerunning at the end of the season (the Yahoo
sample is small).

### Late scratches and in-game injuries: nothing to add

Does the model need to treat an active player as possibly missing the game
(ruled out late) or leaving it hurt? Checked on the replay's weekly
projections for untagged players projected at 4+ (a game he missed scoring 0):

- **They play.** Lead players with no injury tag played 99.5% (RB1), 99.6%
  (WR1), 97.6% (WR2), 97.7% (TE) and 93.6% (QB) of their team's games.
- **Our numbers are not too high.** The average miss against the
  projection is +0.4 to +0.5 for RB1, WR and TE (the projection runs a little
  low), -0.35 +- 0.19 for the lead QB. Tagged (Q) players: played 72.3%,
  projected 7.65 against 7.63 scored: the chance-to-play mixture already
  prices them.
- **Backups look like scratches but are not.** Untagged QBs projected at 4+
  played 73.6% of the time (bias -2.0); these are quarterbacks behind the
  starter, and the depth chart already handles them.
- **The tails are right.** Of played games by players projected 12+, 3.7%
  scored under 3 points, and the weekly margin's spread calibrates
  (9.2 / 24.0 / 50.3 / 75.8 / 90.6% under the predicted 10 / 25 / 50 / 75 /
  90th percentiles).

A late-scratch hazard on top would lower means that are already unbiased or
low and would double count the tag mixture, so none is added. In-game
injuries are inside the outcome shapes (fitted on every game he was on the
field for).

### An odds check for model changes

`scripts/grade_league_odds.py --leagues 6 --check odds_baseline.json` plays
24 replayed leagues and fails if weekly P(win), P(playoffs) or P(title) Brier
scores are worse than the baseline by more than 0.003, 0.006 and 0.004 (set
above the spread between a few leagues and 25). The `odds-check` workflow
runs it by hand (it reads the weekly run's nflverse cache and rebuilds the
replay's weekly projections and the D/ST and K backtest); it is not wired to every push.
`--write-baseline` refreshes the file when a change is meant to move the
scores. Baseline (6 leagues a season): P(win) 0.2268, P(playoffs) 0.1589,
P(title) 0.0859.

### Early-season strength: no preseason pull, and production now logs its own

On the replayed leagues, team strength at week 1 was the weakest point. With
each roster frozen at week 1, real rest-of-season points rose only 0.71 for
each point of the model's expected points (0.84-1.03 at weeks 4-13). Before
adding a preseason pull toward the average, I looked for where the extra
spread starts:

- **Players' season values at week 1 are not it.** The simulator's
  per-player expectation (points per team game) against what followed:
  slope 0.99 at week 1, 0.86-0.97 at weeks 2-5, QBs on the narrow side
  (1.0-1.2), TEs a little wide (about 0.8). Within tiers of similar players
  (6, 12 or 24 at a position) it is 0.9-1.1. So the small gaps between
  players drafted near each other are as real as the large ones.
- **The team model mostly corrects the plain sum.** A "best lineup by
  season value" total is far more spread out (slope 0.53-0.74). The season
  model brings it to 0.84-1.03 from week 4 on with absences, byes and the
  wire.
- **What is left is one checkpoint.** Week 1 alone, about +-0.08, on replay
  values that lack the FantasyPros consensus production blends in at
  week 1 (weight about 0.5).

That is not enough to fit a pull, so none is applied. What production
lacked was a way to grade its own numbers:

- The projection log (`skill_log`) gains `ros_value`, the final season value
  after the consensus blend (`vegas.log_season_values`, after
  `blend_season`).
- The odds log gains each team's expected points for the rest of the
  regular season (`SeasonOdds.exp_points`, after the strength fade) and its
  points so far. At season's end `odds_log.grade` reports a
  `strength_slope` for each logged week: real points the rest of the way on
  expected, both against the league average, the replay's measure. The
  ESPN and Yahoo leagues then say for themselves whether week-1 and
  early-season strength needs a pull.

### Edges that did not hold up in the full pipeline

The league replay ("Grading the odds on replayed leagues") showed rosters
built on our numbers came out weaker than those numbers said. Graded alone,
players' weekly projections looked right, so the question was which part of
a projection a lineup or a claim leans on that does not hold up.

**The replay now reads the closing lines.** `ros_backtest.project(...,
lines=True)` scales each past week by its closing implied totals from the
nflverse schedule (`historical_implied`), as production scales by the
market's. Without them, the matchup adjustment stood in for the line and
looked twice as strong as it is (2.0 -> 0.98 once the lines were in), so
fits on the old replay would have misled production. The season-value fits
keep the old default (`lines=False`).

**Each part of the weekly number, graded.** One rebuild of 2022-2025 with
every part of our model's weekly number recorded. Actual points were
regressed on the season value plus each part's points, within position and
week; 1 means the part holds up as built. A game he missed scores 0, as the
weekly number prices it. (Dropping missed games first made the first-game-back
cut look wrong: it is right about backups who do not play.)

| Part of the weekly number | Slope |
|---|---|
| Season value | 0.97 +- 0.02 |
| The betting line | 1.04 +- 0.09 |
| Matchup and game script | 0.98 +- 0.27 |
| Snap-share trend | 0.93 +- 0.22 |
| Next man up | 1.01 +- 0.16 |
| Teammates' share (a receiver or back out) | 0.65 +- 0.13; RB 0.50 |
| A clearly worse quarterback (`qb_change`, WR) | -0.08 +- 0.14 |

**Two changes, held out by season** (each fitted on three seasons, scored on
the fourth):
- `qb_change` off. k = 0 in all four seasons, RMSE on the receivers it
  moved 7.27 -> 7.11, better in 4 of 4. Inside the full projection
  (snaps, shares, the line) the cut double counted. Its own fit graded
  against the volume model alone. The mechanism stays, tested; the season
  simulator's version was dropped earlier for the same reason.
- RB `team_share` power 0.5 -> 0.33. About two thirds of a back's share swing
  held up (k 0.60-0.70); better in 4 of 4, small.

Rebuilt with both, this week's projection over every player-week projected
at 3+: RMSE 6.833 -> 6.817, better in all four seasons (6.871 -> 6.824 on the
5,806 that moved; receivers 7.055 -> 6.913).

**Fading team strength.** On the same leagues with each roster frozen at the
checkpoint, the model's strength estimate holds up better (slope about 0.87
against 0.68 with claims). So about half of the gap is churn: managers level
rosters through the wire, the weakest claiming first. `SeasonModel` now
fades each team's expected edge over the league by `strength_fade` per week
ahead (`_fade`, computed from today's rosters and the same for any override,
so what a move adds passes through whole; this week untouched). Held out by
season, 0.95 was picked in three of four, P(playoffs) + P(title) Brier better
in 3 of 4 (0.2519 -> 0.2516: small). Real leagues churn more than the replay:
ESPN kept 70.5% of its week-1 skill players by week 7, the replay 85.2%.

**Re-graded**, the same 100 replayed leagues with lines in, before and after:

| | Before | After | Seasons better |
|---|---|---|---|
| This week's P(win), Brier | 0.2323 | 0.2312 | 3 of 4 |
| P(playoffs), Brier | 0.1704 | 0.1665 | 3 of 4 |
| P(title), Brier | 0.0881 | 0.0854 | 3 of 4 |
| P(title), standings-only on the same leagues | 0.0888 | 0.0871 | |

P(title) now beats the standings-only baseline: by 0.0017 against 0.0007
before.

**Still open.**
- Heavy favourites are a little overconfident (70-80%: 74.3% predicted,
  70.9% won).
- Season strength still regresses (slope 0.55-0.83).
- Week-1 values in the replay lack the consensus blend that production
  has.

### Grading the odds on replayed leagues

Every part of the season engine had been validated on its own; the numbers
a manager reads -- this week's P(win), P(playoffs), P(title), expected wins --
had not been graded as a whole. `roster/league_replay.py` drafts and plays
ten-team leagues on 2022-2025 with real players and real scores, as our
leagues run: QB/RB/RB/WR/WR/TE/FLEX, six bench spots, full PPR, weeks 1-14
regular season, six playoff teams over weeks 15-17 with byes for the top
two. Each week's projections come from the games before it only
(`ros_backtest.project`). Managers draft on their own noisy read of season
values, fill the starting spots they cannot field from the wire, make at most
one upgrade claim a week (worst record first), and start their best projected
lineup. P(win) comes from the lineup tab's correlated draws; the
`SeasonModel` odds come at weeks 1, 4, 7, 10 and 13. The baseline knows only
the standings (every remaining game a coin flip). `scripts/grade_league_odds.py`
runs 25 leagues a season: 7,000 games and 5,000 team-checkpoints.

**This week's P(win)** -- informative and close to calibrated: Brier 0.2316
(coin flip 0.25), log loss 0.655, the same in every season. Favourites won
61.4% against 62.0% predicted. The 70-80% bin is a little overconfident
(74.2% predicted, 70.9% won, 95% interval 68.0-73.6%); the margin's spread
is right (9.4/24.9/50.1/75.5/90.3% of results under the predicted
10/25/50/75/90th percentiles). At the checkpoints the season model's own
P(win) for the week (Brier 0.2338) was worse than the lineup draws' (0.2278)
on the same games.

**P(playoffs)** -- informative and calibrated: Brier 0.1658 against 0.1731 for
the standings alone, better at every checkpoint and in three of four seasons
(2024 even). Calibration bins all within or near their intervals.

**P(title)** -- calibrated but carrying almost nothing beyond the standings:
Brier 0.0861 against 0.0859, log loss 0.3030 against 0.3009 (frozen rosters:
0.0864 against 0.0873). What the model adds over the standings is worth
about half its weight (a logistic fit puts 0.5 +- 0.2 on it at every
checkpoint; 0.6-0.9 for P(playoffs)).

**Why: the optimizer's curse, not miscalibrated players.** Graded alone, the
replay's weekly projections are if anything a little cautious (slope of
actual on projection 1.07 +- 0.02 over 13,140 player-weeks). But a lineup
starts whoever projects highest, and a draft or claim takes whoever's season
value is highest, and the players picked that way are the ones whose numbers
ran high. So rosters built on our numbers are weaker than our numbers say:

| | Slope (1 = calibrated) |
|---|---|
| Weekly projected margin, real on projected | 0.87 +- 0.03 (0.75 in weeks 1-4) |
| Rest-of-season team points, real on expected | 0.58-0.84 by checkpoint (frozen rosters 0.52-1.01) |

Churn adds a little: the standings-only baseline does better when managers
make upgrade claims, which pull weak teams back toward the pack. A blanket
recalibration fitted on three seasons and scored on the fourth barely helps
(title Brier 0.0861 -> 0.0855; playoffs unchanged), so nothing is applied
on top of the odds. The fix belongs where the bias starts: discounting a
selected player's edge over the ones passed over.

**Limits.**
- Replay projections are our model alone. Production blends in the betting
  market, FantasyPros and Sleeper: on 2026 weeks 1-6 the blend's slope was
  0.95 +- 0.06 against 0.90 +- 0.06 for our model alone.
- Rival managers in real leagues pick by the platform's numbers, not ours,
  so the curse hits our own team harder than our rivals'.
- The replay has no D/ST, K or trades.
- Parts of the model (outcome shapes, correlations, hazards) were fitted on
  these same seasons.

### Teammates and opponents in the season simulation

Until now the season simulator (`futures.simulate`) drew every player's week
on its own, while the week's lineup optimiser already linked players in the
same game. A quarterback's big week is usually his receivers' big week too,
and a bad week for the defence facing him. Independent draws made a stacked
team's weekly total too narrow. That matters for P(win) each week, and so
for the standings and the title odds.

Now each simulated week, the players in each NFL game are drawn together
through the fitted residual correlations by role in `outcome_model.json`,
the same table the lineup optimiser uses (QB and his WR1 +0.34, QB and TE
+0.26, QB and the defence facing him -0.44, the two QBs +0.18). It is a
Gaussian copula, so each player's own distribution is unchanged. The games
come from the NFL schedule (`season.nfl_opponents`). The setting is
`roster.season_correlation`.

**Held out**, on the 2022-2025 replay (24 checkpoints), with the
correlations refit leaving out each test season. Graded on weekly totals
against what happened:

| Weekly total | 80% band, independent -> linked | Dispersion ratio | CRPS |
|---|---|---|---|
| Team stack (QB+WR1+WR2+TE), n=8,089 | 75.5% -> 78.8% | 1.30 -> 1.10 | 10.823 -> 10.823 |
| Whole game (both stacks), n=3,842 | 71.8% -> 78.3% | 1.50 -> 1.16 | 16.523 -> 16.468 |
| Single player, n=8,089 | 88.4% -> 88.5% | 0.93 -> 0.93 | 4.578 -> 4.577 |

The bands and the dispersion improve in every season, for both stacks. CRPS
is dominated by misses in the mean, which this change does not touch. It is
better overall for games (worse in 2023 only) and flat for team stacks
(worse in 2023 and 2024, each by under 0.3%). Kept on the band and
dispersion grades, which are what a correlation changes. The remaining
dispersion above 1 suggests teammates' longer-run errors are linked too (a
whole offence better than expected). That is not modelled yet.

The replay has no D/ST or kickers, so their links (most strongly QB and the
defence facing him) rest on the fitted table alone, as in the lineup
optimiser.

**Tried and dropped: receivers losing part of their level while their lead
QB is out.** Graded against the volume model alone, with no betting line
(`fit_qb_change.py --no-line`; the season simulation's future weeks have
none), receivers with a clearly worse new starter fall short by 22.5% times
the share of their recent games the old starter threw. That was better in
all four held-out seasons on those weeks (5.43 -> 5.19 RMSE); tight ends and
backs showed nothing. In the season simulation, though, it made receivers'
rest-of-season points per team game worse on the replay:
- all receivers: bias +0.39 -> +0.48, CRPS 1.587 -> 1.595;
- receivers whose lead QB is out at the checkpoint: bias +0.30 -> +0.64,
  CRPS 1.363 -> 1.407, worse in three of four seasons.

Their season values already carry it, and the weekly projection keeps its
own cut (`qb_change`) for the week a backup starts. Not shipped.

### A review of the title odds: intervals, the second run, dropped players

An outside review of how P(title) is computed; the first fixes:

* **Intervals.** Each title and playoff percentage now carries a 95%
  Wilson interval for the simulation's own noise (a 10% title chance is
  +-0.8 points at 6,000 seasons, +-1.5 at the 1,500 of a quick refresh; "+-0.4"
  had been one standard error, not a margin). It says what a rerun could
  move, not whether the model is right.
* **The second run only prices the choice.** Moves, plans and trades are
  chosen and ordered on one simulation and priced on an independent one --
  but the shortlist had been re-ranked and filtered on the second run too,
  which picks its favourable noise in turn. Now the first run chooses and
  orders; the second only supplies the numbers and the verdict (a move it
  reads as no gain shows as "wait", a plan as "no clear edge on a second
  run", a trade as a toss-up).
* **A dropped player goes back to the wire.** One worth more than the free
  agents there is claimed by a rival the next week (the claim draw used for
  breakouts). Doing that exposed a bug: a rival landing a player made room
  by cutting his lowest season value, which could be his only tight end --
  an empty slot, so a rival landing a breakout read as a gain for you. He
  now cuts a bench body (not in his best lineup), or failing that his
  weakest at the new man's position.
* **ESPN's playoff-game tiebreak** is read from the league
  (playoffMatchupTieRule): season points where it says so, else the higher
  seed. Standings tiebreaks and playoff-game tiebreaks are separate rules.
* **Deterministic tests**: a fixed bracket against a reseeded one on
  hand-set scores, tied playoff games both ways, a two-week round, exactly
  one champion and a full field every season, title odds summing to 100%,
  and a move that changes nothing moving the title odds by exactly zero.

Still to do, in order: link teammates' and opponents' outcomes in the
season simulation (shared game effects, a QB's absence for his receivers);
late scratches and in-game injuries that bench depth cannot always cover,
and shared ownership of the wire; then grade weekly P(win) and title odds
on replayed leagues.

### ESPN: division winners get the byes

In the ESPN league each division's winner is seeded first (1 and 2 by
record, then points) and takes a first-round bye; the other four spots go
by record, divisions aside. The simulator seeded everyone by record. Now,
where the profile says so (``league.division_byes``, ESPN only) and the
sync has each team's division (ESPN's divisionId), each simulated season
seeds the division winners first. Yahoo is unchanged. On this week's
snapshot it moves little -- the two favourites are in different divisions
-- but it is the league's rule, and a weak division's winner gets the bye.

### Head to head, the opponent's bench, and the latest game in the flex

The Lineup tab and the live card now pair your starters with your
opponent's slot by slot (best against best within a slot, as FantasyPros'
matchup page does), with a check by the higher projection -- live, by
whoever is on track to finish higher: his points plus the projection's
share of the time left.

The opponent's bench is now in the simulation where a manager would use it:
a starter already out or on bye in his set lineup is taken to be replaced
by his best healthy bench player whose game has not started (nearly every
manager makes that fix; the page lists it), and a questionable starter who
sits in a draw is replaced from his bench by the same rule as yours (his
inactives, a bench player still to play). Small projection edges are not
assumed chased: there is no record of how often managers reshuffle. P(win)
had been counting an opponent's ruled-out starter as a zero.

Within the lineup chosen, each flex slot holds the latest kickoff among the
starters at its occupant's position (games not yet started only): if
someone is ruled out late, the flex can still take a player from any later
game, a dedicated slot only its own position. Same starters, same odds; the
page says which two to swap.

### A teammate's "ruled out" is not his

The breaking-news step sets a player out when this week's RotoWire headline
starts with his name and says he is ruled out. "Marks is lined up to start
... with David Montgomery (groin) ruled out" set Woody Marks out -- the
Yahoo opponent's starting back, zeroed, and the matchup read 80% instead of
about 68% -- and "Wilson will be the lead back ... after Zach Charbonnet
(knee) was ruled out" did the same to Emanuel Wilson. An out tag that
follows another player's "Name (injury)" is now that player's (the rule the
timeline reader already used), and "lined up / set / in line to start"
reads as a bigger role.

### "Is back" only when he plays this week

Darius Cooper's card said "DeVonta Smith is back" with Smith ruled out. The
teammate-return correction reads an out tag as 0% to play this week and
70% of the rest of the season, so it rightly left Cooper's week alone and
trimmed only his season value, but the card said "back" either way. Now,
under 50% to play this week, it reads "out this week but expected back
later ... this week unchanged". And where the news reports a timeline
(reported_share), that, not the flat 70% (or 0% for IR), is the share of
the season the absent player is taken to play, for both the next man up and
the teammate-return correction: a season-ending injury hands his backup the
whole season, a 2-week one most of nothing. The flat shares were judgment
too; no history of news exists to grade either against.

### Who actually gets picked up

The chance another manager takes a player this week was a guess: each
active rival claims with chance 0.5 if he would start for him. Over 13
rivals that read 98-99% for Kendre Miller and Darren Waller, who sat on the
Yahoo wire for days. Measured on the stored weekly snapshots (3,051 unowned
player-weeks: was he on a rival's roster the next week?), "would start for
rivals" says nothing (ESPN, would start for 3-5 rivals: 1.6% picked up);
platform-wide % rostered does. Fitted per league (scripts/fit_claim_odds.py,
claim_odds.json): ESPN picks up 2.2% of its unowned players a week, from
about 1% under 10% rostered to a third above 60% (log loss 0.087 against a
flat 0.106); Yahoo 10.1%, barely moved by % rostered (0.322 against 0.328).
Miller (45% rostered) now reads about 22% on Yahoo. The ones ahead of you in
the order are taken to get him in proportion to their number. Refit as the
snapshots accumulate. The separate assumption of how many managers chase a
real breakout (CLAIM_INTEREST) is unchanged: that is a different event,
and no record of it exists to fit.

### Free agents are added, not claimed

Every unowned player was priced as a waiver claim: "98% chance another
manager claims him", a claim order, waiver priority spent. Most of them are
free agents, added at once with no claim and no priority spent; only players
recently dropped (or, in some leagues, everyone until waivers clear) are on
waivers. Both platforms list the two together, so the sync now reads which
is which: ESPN by asking for the players on waivers on their own (its
free-agent call keeps no status; the bare filter was refused, so it is
built as espn-api builds its own), Yahoo from its waivers-only player list
(status=W; the main list has no owner column, as the first run showed). A free agent has no claim odds, costs no priority, and
goes first in a plan ("add now"); unknown status keeps the old behaviour.
The chance a rival adds him later, if he breaks out, is unchanged.

### The best of many is the luckiest

Waivers price twenty-odd free agents (each with several drops) and trades
dozens of deals on one simulation of the season, then show the best. The
best of many noisy estimates is the luckiest: Malik Willis showed +1.2 title
points on the Yahoo page, a third quarterback behind Stafford and Brissett.

Measured on the Yahoo league, 12 seeds x 10 moves: each run's own error bar
is about right (spread between runs 1.14x the stated standard error), but
among the nine moves with no real edge, the one each run ranked first read
+0.55 on its own run and +0.09 across the other eleven. The fix is the
standard one: the shortlist (the moves shown and three more; the trades on
any board and a few more) is priced again on a second, independent
simulation (seed + 7919), and the numbers, the claim/lean verdict and the
order come from that. A move or trade that falls short there is not shown.
The card says so, with what the first pass read. Cost: about 40 seconds per
league per full build. On the same snapshot Willis reads "wait" either way;
Pierre Strong Jr. (the one clear edge) stays a claim.

Waiver plans were at first left as they were (they must beat the best
single move, itself the luckiest of its pool), but a plan of three adds
carried Malik Willis as a third quarterback on Yahoo: the best of many
plans on one set of seasons is as lucky as the best single move. Plans
shown are now priced again on the same independent simulation, against the
confirmed single moves, and kept only if they still beat them there.

### The market's view, refit at week 7

The perception weights (how managers value a player) were fitted at week 3,
when this season was 13% of the blend. Refit on Yahoo's % rostered at week 7
(338 players): platform 0.41 -> 0.11, reputation 0.46 -> 0.43, this season
0.13 -> 0.46 (90% interval 0.29-0.67); out-of-sample R^2 0.824 against our
projection alone at 0.814. A breakout is now read as managers read it:
Kenneth Walker III (25.2 a game this season, 13.5 his old reputation) 17.3 ->
19.5 in the market's eyes, Puka Nacua 23.6 -> 23.3.

The chance a manager accepts is still an assumed curve (no record of offers
and answers exists to fit it); it ranks offers, and the percentage is not a
measured rate.

### Back from a season-start reserve list

The teammate-return correction (a back's work while the lead was out is
read at what he keeps once the lead is back) only counted a teammate who had
played for the team this season. That guard was for players carried on a
roster without a snap (Salvon Ahmed), but it also skipped anyone back from a
season-start IR: Zach Charbonnet, on Seattle's IR weeks 1-6 and questionable
after a full practice in week 7, did not count, so Emanuel Wilson's 14-21
carries a game were priced as his to keep (10.3 a game for the season) and
he was recommended as a pickup. Now a teammate who has not played this season
counts when his team carried him on a reserve list this season (weekly
rosters, weeks before this one only).

Walk-forward 2023-2025 at the shares already fitted (scripts/
fit_teammate_back_reserve.py), on the player-weeks it moves: RB RMSE 6.75 ->
6.67 (27 player-weeks, better in 2024 and 2025, worse in 2023), TE 6.25 ->
6.20 (10, better in 1 of 2), WR 4.94 -> 4.95 (worse overall, left out).
Small samples, but the rule held for backs and tight ends; it is on for
those (``teammate_back.reserve_return``). Wilson: 10.3 -> 8.5 a game for the
season, 9.7 -> 8.8 this week; George Holani 5.4 -> 2.5.

### The league's most valuable players

A Players tab ranks every player the season simulation carries (all rosters
and the free agents worth pricing) on value over replacement: the points he
is simulated to score above a waiver pickup at his position, summed over the
weeks left (this week through 17). Replacement is the second-best free
agent's season value a game (the best for D/ST and K), as waivers and trades
already use. A week he does not play counts nothing, since you would start the
pickup either way: byes, injuries, reported timelines and the next man up all
come from the same simulated seasons as the title odds, so Adams out 6-10
weeks ranks on the few weeks he has left. Raw points were the alternative and
were rejected: they rank every quarterback first and make a kicker look like
a receiver. Filters by position and FLEX (RB/WR/TE); sortable by season
value, this week, games, range and the market's view. No fitting, so nothing
to validate: it reads numbers already graded elsewhere.

Its first day showed Jared Goff above Josh Allen, and that exposed a bug in
the season simulation: a player on bye this week has a 0% chance to play
this week, and the simulator read that as an injury, drawing several more
missed games after the bye. Allen played 6.1 of his 10 games left instead of
about 9. Every player on bye was undervalued in the title odds, trades and
waivers from week 5 (the first byes) until the fix. The replays the
simulator was fitted on never mark a bye this way, so no fitted setting
changed; the live pages were wrong.

### Reported injury timelines

The recommendation to trade for Davante Adams exposed a gap: he was out with a
reported 6-10 weeks, tagged plain OUT (his team had stashed him in an IR slot,
but the platform tag is not an IR status), and the season simulator treated a
plain OUT as an ordinary absence: back in about two games (45% one game, 19%
two). Nothing read the news for how long. The status tag was all it knew.
Anyone with a reported multi-week injury and no IR tag had the same error,
and the rest-of-season value, title odds and trade pricing built on it.

`data.news.timelines` now reads the time out of the news about every player
who is out (or whom the news flags with an injury) and ESPN's feed carries a
recent RotoWire or ESPN item for: "expected to miss 6-10 weeks", "out four to
six weeks", "about a month", "a couple of weeks", "season-ending", "out for the
season", a torn ACL or Achilles. Past counts ("has missed the last three
weeks"), "Week 7" and a number with nothing about missing time are not read.
A report counts down from the day it was written, rounded to whole weeks, and
a newer item that says he practiced fully or was cleared cancels it. It sets:

- `reported_back`: the earliest and latest NFL week he is back (99 for a
  season-ending injury). The simulator draws his return week evenly from the
  range, in every simulated season, byes not counted as games, instead of the
  generic draw or the reserve-list table. A confirmed return
  (`roster.known_returns`) still wins.
- `reported_share`: the share of the rest of the season he plays on
  average, which scales his value in the quick screens that rank trades and
  claims before the simulation prices them.
- a note on his card, and a row in `results/<profile>/timeline_log.parquet`
  (player, report time, the range, when he was logged), so each report can
  be graded against the week he actually came back.

The first live run found real problems, all fixed before relying on it. It
read an injured-reserve stint's floor ("required to miss *at least four
games*") as a four-game timeline, "the Packers' *Week 7 game*" as seven
games, a signing story's mention of a *past* torn Achilles as a
season-ending injury, an ESPN article about forty players as news about each
of them, and a team that "*fears* a season-ending injury" before an MRI as
a confirmed one. Each is now not a timeline: a floor (at least, minimum,
requisite), a hedge (fear, hope, possible, could, may, if, without a
timeline), a week label, a past count. ESPN's articles are read only for a
sentence that names the player, from the last seven days. The log keeps the
sentence it read from, so a wrong one can be found. Hurts, whom the first run
also marked out for the season, was right (blood clots, "miss the rest of the
season"). Adams's 6-10 weeks is in an ESPN article, not a blurb, which is why
the first version missed him; the RotoWire blurb said only that he
ruptured a hamstring and would have a procedure.

On the Adams case: the generic draw had him back in week 8 in 48% of
simulated seasons and playing 79% of the time by week 12. With the report he
plays 0% until week 13, 20% that week, and 92% by week 17.

Not validated held out: there is no history of past news to test on. Reports
are used as given, evenly across their range; the log exists to find out
whether to shade them (teams return players early and late). Until then a
reported range is the best information there is, and the generic draw was
known to be wrong for it.

The second live run still read 13 of 26 wrong, every one a number or phrase
that was not about his time away: stat lines ("over six games", "sat out a
combined 19 games in 2023", "three Week 6 practices"), past counts ("the
Chargers' prior two games"), a list article whose "out for the season"
belonged to the item before his, and a blurb about him that gave another
player's injury ("after Jayden Reed (neck) will undergo season-ending
surgery"). Now a count needs words for time away right before it ("will
miss", "out", "ruled out for the next", "sidelined for") or after it ("a
six-week recovery"); a list in an article is read item by item (split at
semicolons); a time that follows another player's "Name (injury)" is his,
not ours; a time given before his name in a sentence that names him is
someone else's (Ekeler got Jonathon Brooks' six weeks); and "6-to-8 weeks"
reads as a range. Every sentence from those runs
is a test case.

### Bye weeks thin the wire

The season simulation filled any hole from the wire at one fixed
replacement level for every week: the second-best free agent today. Two
things it missed. A free agent on bye that week fills nothing. And when
several teams need the same fill-in, they cannot all get the second-best
one.

`SeasonModel` now builds the wire week by week (`_wire_by_week`). For each
starting slot and future week it lists the free agents off bye that week,
best first, starting where `replacement_levels` starts: each position's
second best (the best for D/ST and K), merged at FLEX. With no byes and no
competition it reproduces the old level exactly. With the switch off it
reproduces the old numbers to the bit (checked on the live ESPN league).

The competition step works in two passes. First, every team's holes are
counted per simulated week on today's rosters. A hole is a slot whose
best rostered option falls below the wire. Then each lineup is scored
against what the others leave (`_wire_level`). A team's fill comes from
as far down the list as the rival claimants ahead of it reach: the other
teams' holes at that slot that week, times the share of them ahead in
waiver order. Today's order is used for the next two weeks
(`PRIORITY_WEEKS`), because the order turns over as claims are made.
After that every team sits in the middle.

Claims compete at QB, RB, WR, TE and FLEX only (`roster.wire_compete`).
Competing at D/ST and K as well cost a team that streams them every week
4 points of playoff odds (66.1% -> 62.1%, ESPN). That number is not
believable. The list ranks streamers by season value, but a real stream
follows the week's matchups, so the 4th-best option in a given week is
much closer to the best than the season list suggests. D/ST and K
therefore get byes only, until there is a matchup-aware version.

What it moved (4,000 simulated seasons, today's rosters):

- ESPN: at most 0.7 points of playoff odds. Yours went 66.1% -> 65.7%
  (title 12.0% -> 11.7%). Almost all of that is your week-11 receiver byes.
- Yahoo: concentrated where byes stack. Near Ja'marr 67.8% -> 64.9%, CMoney
  +1.7, Macky +2.0. Yours went 80.4% -> 81.3%, because rivals' holes now
  cost them more.

There is no held-out test. Past weekly free-agent pools are not kept, so
this is a modelling correction checked for sense, not a fitted gain.
`roster.wire_thinning: false` turns it off.

### Late swaps from the wire, cover pickups and the night-game refresh

A questionable starter in the last game of your week, such as Mike Evans on
Monday night, had no swap at all. By his inactives every bench player's game
is over, so none of them can come in. On ESPN they cannot be dropped either
once their games have begun. The only replacement then is a free agent
playing that night, and only if the roster has room to add him.

- **Free agents count as late swaps** (`lineup.Wire`, `swap_plan`). The
  matchup report keeps the best few free agents per position
  (`matchup.late_wire`) and the room the roster will have when the
  inactives come out: empty bench spots, and empty IR slots (a starter
  ruled out can move into one; ESPN lets an OUT player sit there). A
  free agent covers a starter only if his game starts no earlier than 90
  minutes before the starter's, he is eligible for the slot, and there is
  room. Each spot is used once. A bench player is preferred unless the
  free agent projects higher. A drop is never counted on: by then the
  bench is locked, and the drop is the manager's call anyway. Config
  `roster.late_wire` switches it off.
- **The start/sit choice sees the real fallback.** With a Monday free agent
  available, a questionable Monday starter is worth more than a sure
  Sunday player of the same expected points. The optimiser now starts him
  where that holds; before, he only counted as a coin flip against zero.
- **Cover pickups are priced** (`pickup_options`). A free agent no longer
  has to start to help: he can be held on the bench as cover for a
  questionable starter in a game that kicks off before his. He is priced
  the same way, as P(win) against the opponent on shared draws, with late
  swaps counted on both sides, so his worth is what he adds over the
  fallback you already had. The page says "hold him on the bench to
  replace X if he is ruled out".
- **The Lineup tab shows "If a questionable starter is ruled out".** For
  each one: when his inactives come out (ET), and his replacements, best
  first, with what each takes (from your bench, your open roster spot, or
  moving him to IR and then adding a free agent). It also marks which one
  the win odds count on, and warns that the free-agent pool cannot tell a
  free agent from a player on waivers.
- **The night games get a refresh after their inactives.** It runs at 6:55
  PM ET on Thursday, Sunday and Monday. The cron is in UTC, so it has two
  entries (22:55 and 23:55), one for each side of the clock change. Before
  this, the 22:41 UTC run came just before the inactives and the next one
  after kickoff.

### Quarterback bias: not systematic, no correction

The weekly replays showed quarterbacks projected 0.5 to 1.2 points low in
some seasons, which looked like a standing under-projection.
`scripts/check_qb_bias.py` replays the production projection at weeks 5-13
(every other week) of 2022-2025 and 3-6 of 2026, with the quarterback
comparisons on and off. Mean error (projection minus actual) with the
comparisons on / off:

| Season | On | Off |
|---|---|---|
| 2022 | +0.59 | +0.21 |
| 2023 | -0.07 | -0.08 |
| 2024 | -0.78 | -0.95 |
| 2025 | +0.10 | +0.19 |
| 2026 (weeks 3-6) | -1.18 | -0.85 |

Over 2022-2025 it averages about zero (-0.04 on). It is a scoring-environment
swing from season to season (quarterbacks averaged 15.2 in 2024 and 16.2 so
far in 2026, against 14.4-14.9 projected), not a model flaw. Within a season,
shifting each week's projections by the shrunk mean error of the earlier
checkpoints (prior strength 40, 80, 160 games; shift 50% or 100%) made the
miss worse in every variant (6.002 -> 6.007 at best) and better in at most
two of five seasons. The error is not persistent enough week to week to
learn from. No correction was added.

The comparison blend moved 2026 the wrong way (-0.85 to -1.18): it leans on
past seasons, and this season's quarterbacks are scoring above them. Over
2022-2025 it still helped (the earlier test), and it is 20% of the way off
the model's own number for quarterbacks at most, so it stays.

### First game back from an absence: a discount, 80% and 69%

Players coming off an absence often play limited snaps in their first game.
`scripts/fit_first_game_back.py` replays the production weekly projection
for every week 3-15 of 2023-2025 and splits the players who played by how
many of their team's games in a row they missed before it (only those who
had played earlier that season, so a rookie or a new signing is not a
return). Judged against the projection's if-he-plays number, because the
injury-tag discount hides the comparison among players who did play.

| Games missed | n | Projected (if plays) | Scored | Scored / projected |
|---|---|---|---|---|
| 0 | 8,510 | 9.59 | 10.04 | 1.05 |
| 1 | 487 | 6.29 | 5.06 | 0.80 |
| 2-3 | 252 | 7.17 | 4.85 | 0.70 |
| 4+ | 27 | 9.45 | 7.57 | 0.80 |

Over-projected in every one of the three seasons for the 1 and 2-3 groups,
at every position (QB +4.5 points, WR +2.0, TE +1.7, RB +1.3 for 2+
missed). With a multiplier per group (1 missed; 2 or more) fitted on two
seasons and scored on the third, the miss fell 5.039 -> 5.003 overall, better
in 3 of 3 seasons, and 4.45 -> 4.01 for the 766 players back, bias +1.61 ->
-0.02. Fitted on all three seasons: 0.80 for one missed game, 0.69 for two or
more. Applied to the model's weekly number before the platform, betting,
FantasyPros and Sleeper blends, for players who have played this season; a
card says so ("first game back after missing his team's last N").

Replayed through the production code on weeks 4-14: 2025 5.164 -> 5.120 on the
if-he-plays basis and 2024 flat (4.934 -> 4.936): the effect was weakest that
year (bias +0.5 to +1.0 against +1.5 to +3.5). The platform and betting blends
dilute the discount on the published number, since the platforms do not apply
it. `roster.first_game_back` in config.yaml; the season value is untouched.

### D/ST and K ranges in the stream picks

The ranking cards already show floor-ceiling and P(top-12) for each unit. The
D/ST & K tab, where free-agent units are swapped into your lineup and scored on
this week's win odds, now shows the same 15th-85th percentile range (with its
bar) from the simulated draws, so a pick can be read as a floor against a
ceiling as well as a win probability.

### Yahoo bye weeks read "% rostered" as the projection

On 2026-10-13 the Yahoo title odds collapsed: one team 67%, two others 16%
each, and every other playoff team under 1% (Drakes on a Plane: 4-1, 66%
playoffs, 0.3% title). Those three teams were simulated at 155-170 points a
week and 180-200 in the playoffs, against 110-125 for everyone else.

The cause was in the Yahoo page reader. On a player's bye, the row has one
"Bye" cell spanning the game and points columns, so the cells are fewer
than the headers. Looking a column up by position then landed the
projection on "% Rostered": Justin Jefferson projected 100, the Vikings
D/ST 95, Evan McPherson 73. For skill players on bye this did no harm, since
their week is zeroed anyway. But a D/ST or kicker on bye is missing from this
week's own rankings, so its season value fell back to the platform number.
That handed those three rosters a 70-95-point unit for every remaining week,
playoffs included. It had been there every week; week 6 was the first with
rostered D/ST and K on bye (Minnesota and Cincinnati).

Fixed three ways:

- **The reader.** `column_cells` expands spanned cells so every column
  lines up, and a bye row gets no projection, in both the roster and
  free-agent readers.
- **The projection.** It ignores a platform projection for a player on bye,
  or one above what the position can plausibly score (`PLATFORM_MAX`).
- **Next week's ranking.** A D/ST or kicker on bye takes its season value
  from our own ranking for next week, not from a default.

With the week-6 snapshot the Yahoo odds return to normal: the best team is
about 21%, and Drakes on a Plane is about 15% title and 81% playoffs.

### An empty D/ST or K slot still gets stream options

The "Best D/ST and K for this matchup" table priced each free-agent unit
against the one in your recommended lineup. With no D/ST rostered (ESPN,
week 6, after the Jets were dropped), there was no unit to price against,
and the D/ST table vanished. That is exactly when it matters most, because
the slot scores zero. Now an empty slot is shown as "yours" at zero, every
free-agent unit is priced against it, and the hub says "your defence slot is
empty". The kicker table, sorted by this week's win odds rather than by
projection, is otherwise consistent with the full rankings. The order
differs where a unit's link to either lineup moves the odds. A platform can
list a different kicker for a team than nflverse's starter (ESPN shows Blake
Grupe for LA, our rankings Harrison Mevis); the row carries the team's
kicker projection either way.

### Try a pickup: any free agent against every sensible drop

The Waivers tab gains a "Try a pickup" section. Pick a free agent, and each
sensible drop is shown with its change in title odds and in this week's win
odds. A static page cannot run the season simulation per click, so every pair
is priced during the refresh (`roster/add_drop.py`):

- **Which players.** The top 120 skill-position free agents and the top 6
  defences and kickers by season value. The top 40 skill players and every
  unit are priced against the 6 drops cheapest to your title (always
  including the cheapest at his position), the next 80 against the 3
  cheapest. Every other free agent is listed unpriced, with the reason
  (below those priced at his position, out long term, or not on an NFL
  roster). Players whose
  game has started, and players in IR slots, are not drop options.
- **Title odds.** The add and the drop on the full season simulation
  (`TitleEngine.value_now`), with the simulation's own noise. A change
  inside it is marked "toss-up".
- **This week's win odds.** Your best lineup rebuilt with him in and the
  drop gone. It is chosen on one sample and scored on a fresh one, against
  your opponent's lineup, on draws shared by every pair.

You find a player by **searching** (name, team or position, accents and
punctuation ignored), not from a menu: the first version's menu of 30
rarely held the player you were looking for. Pricing about 130 players
costs about a minute per league per refresh (about 0.1 s per add/drop
pair); the data is about 150 KB.
It does not account for waiver priority or for the chance another team
claims him first; the waiver cards above still do.

### Any team's best lineup

The Lineup tab gains "Any team's best lineup". Pick a team (yours is the
default) and see the lineup that maximises its P(win) this week against its
opponent, two ways:

- **Current:** against the lineup the opponent has set (their best projected
  lineup if none is set yet).
- **Ideal:** against the opponent's best projected lineup, respecting games
  already started (`optimise(..., opponent_mode="ideal")`).

Each view gives P(win), the odds with the lineup as set, both projected
totals, the changes from the set lineup, and the lineup slot by slot. Opponents
come from the league schedule, so every team this week is covered. Your team
uses the full simulation; the others use 4,000 draws, so their odds are good
to about a point. It all runs in the refresh (a few seconds per league).

### Win/win trades, as the other manager sees them

The Trades tab gains a "Win/win" view. A trade is win/win when both of these
hold:

- **Your side:** it clearly raises your title odds. The gain must clear the
  simulation's noise, the same bar as the other lists.
- **His side:** he sees his roster gain at least 0.3 points a game
  (`SEEN_WIN`), judged the way managers value players (`perception`: the
  platform's projection, a player's name and last season, this season so
  far).

He cannot see title odds, so his side is judged on the market view, not
on ours. The finder now also prices the trades he sees as best, best for
you first, so the list is not limited to deals that made the other views.
Each win/win card also says what our projections think of his side. In the
week-6 ESPN run, all five win/win trades raised your title odds by 1.1-3.9
points while lowering his by 0.3-2.6 on our numbers. He gains on how the
market rates players, not on what we project, which is exactly why he would
say yes. The hub's "win-win" tag now means the same thing; it used to mean
his title odds rose on our model. With a fifth tab, the trade tab labels
wrap onto a second row on a phone.

#### Win/win back to both teams' title odds

The perception version above looked little different from the other lists:
a deal he sees as a gain is mostly a deal he is likely to accept, which "Most
likely yes" already shows. So win/win is back to its first meaning: **both
teams' title odds rise on our season simulation**, yours by more than the
noise. To find them, the finder now also prices the deals that raise his
roster value on our projections (best for you first), not only those that
made the other lists. The chance he says yes is still judged on how he sees
it, and each card still shows it.

### Confirmed returns: Josh Jacobs off the exempt list

Josh Jacobs (ESPN roster) had missed GB's first five games on the
commissioner's exempt list, tagged day-to-day. The simulator read that as an
injury absence five games old: the measured absence lengths, conditioned on
lasting more than five games, kept him out for most of the season, and
`season_loss` (low share of games played, none this season) gave a 54% chance
that any later absence would end his season. The news (removed from the list
Monday, eligible for week 7) settles both.

`roster.known_returns` (config) takes name -> the week he is back. While that
week is still to come, the simulator sits him for exactly the weeks before it
(byes not counted), the season-loss read is replaced by the base share (a
suspension ending says nothing about a lost job or a fragile body), and his
card says so. Everything else is unchanged: his week-6 zero, his season value
(11.5 a game, with MarShawn Lloyd back), and the first-game-back discount in
week 7 (measured on all returners, 2023-2025; suspension returns were not
separated). On the week-6 ESPN snapshot this moves your title odds 11.4% ->
12.0% and playoff odds 63.6% -> 66.1%. An entry is ignored once its week
arrives, so it can be left in place.

### Kicker props and the opposing QB's interception line: logged, graded first

The kicker and D/ST models already rest on Vegas through the game lines
(team and opponent implied totals) and beat a ranking by those alone. Two
player-prop lines add something the game line does not know:

- **A kicker's kicking points** (`player_kicking_points`: field goals x 3
  plus extra points): drives that stall into field goals, his range, the
  weather. Bought with the other props from this week on, one more credit
  per game (8 markets, not 7; the monthly pacing buys a game or so fewer on
  a Sunday). The line is converted to a mean on a normal with sd 3.5, flat
  across levels (kicker games 2019-2024: 3.4-3.8 at every level). It is not
  shown as Vegas points: the leagues score field goals by distance, so
  kicking points are not fantasy points.
- **The opposing quarterback's interceptions** (`player_pass_interceptions`,
  already bought for the skill-player blend): the starter's posted line (a
  line filled in from his recent games does not count).

There is no free history of props, so neither can be backtested, and the
skill log only started keeping each quarterback's lines in week 6. So both
are logged each week beside our projection for the unit
(`results/<profile>/unit_log.parquet`, `roster/unit_props.py`), replaced
until his game starts, and shown on the unit's card, without moving its
projection. `scripts/grade_unit_props.py` grades them against what the
units scored, leaving one week out:

- **kickers:** `(1 - w) * ours + w * c * line`, with `c` (kicking points to
  fantasy points) and `w` fitted on the other weeks;
- **defences:** `ours + b * (his line - that week's average)`, with `b`
  fitted on the other weeks.

Once there are 60 graded rows over 3 weeks or more, each is kept only if
it is better overall and worse in at most one week in four, the usual rule.
About 25-30 of each are priced a week, so the first verdict is about three
weeks out.

#### Free agents: browse by position, and adds with no drop

"Try a pickup" became "Free agents": every free agent the page knows, with
position filters (All, QB, RB, WR, TE, D/ST, K), a name/team search, and
columns for season value a game and this week's projection (tap either to
sort, 30 rows at a time). Tapping a player opens what adding him does: each
sensible drop with its change in title odds and this week's win odds, as
before. When your roster has an open spot (fewer players outside IR than
starting slots plus bench), the first line is adding him with nobody
dropped, priced on the same simulations (`TitleEngine._won` on the roster
plus him; this week's best lineup rebuilt with him in). Players beyond the
~130 priced are listed and say why they are not priced.

### Any team's view: one "Viewing as" picker for every tab

Each league's panel now opens with "Viewing as", defaulting to your team.
Picking another team swaps the team tabs (Hub, Lineup, Roster, Waivers,
Trades, Season, the summary tiles and the matchup part of D/ST & K) for that
team's, worked out as if it were yours: its lineup against its opponent this
week, its pickups and drops, its title odds and moves, its trades. The
league-wide parts (the D/ST and K rankings, the Model tab) do not change.

- **How:** the snapshot is copied with that team marked as yours and its
  opponent from the schedule (`publish.view_as`), and the same code builds
  its tabs (`publish._team_panel`). Your team's view alone writes the logs
  (season odds, scorecard) and carries the sections only you use (every
  team's lineup, the trade evaluator, waiver plans).
- **Lighter for the others** (`publish.LIGHT`): 5,000 lineup draws instead of
  20,000, 1,500 season simulations instead of 6,000, and 40 skill free agents
  plus 3 defences and kickers, priced against 3 drops each. Their odds are
  good to a point or two, and the page says so. **Let Kevin Cook** (config
  `roster.views.full`) gets the full settings.
- **Page weight:** each other team's tabs are written to
  `views/<league>/<team id>.json` and loaded only when picked, so the page
  itself is no heavier. The weekly archive copy has no picker (its views
  would be overwritten by later weeks).
- **Time:** views stop being started past `roster.views.budget_minutes` (20)
  per league; a team left out is listed as not built this refresh.

#### Trimming the refresh: faster scoring, one simulation, game day

The first refresh with every team's view took 16 minutes (ESPN: your panel
2 min, Let Kevin Cook 2 min, eight light views 25 s each; Yahoo: 1.5 min and
thirteen light views). Profiled, 105 of a full panel's 186 seconds went to
one function: scoring a roster's best lineup in every simulated week
(`SeasonModel.team_scores`), called by the title moves, the free-agent
table and the trades.

- **Faster lineup scoring.** Each slot took its best player by building
  three full-size arrays (every simulation x week x player). It now marks a
  player used in place and takes each slot's best among only the players
  eligible for it. Bit-identical on every roster tried in both leagues,
  about 1.6x faster.
- **Each roster priced once.** A roster's title odds are cached in the
  engine (the free-agent table and the title moves price many of the same
  add/drop pairs).
- **One simulation per league for the odds.** Every team's title and
  playoff odds in its view are read off your season simulation, so all the
  views agree with yours exactly; the light views also share one lighter
  simulation to price their moves on, instead of building one each.
- **Game day** (`publish.is_game_day`: Sunday 12:30 to Monday 4:00 US
  Eastern), when the page refreshes about every hour: other teams rebuild
  only their lineup, this week's win odds and the summary, keeping the rest
  from the last full build; your trade search and free-agent pricing are
  kept from the last full build too (`views/<league>/_me.json`), and those
  sections say when they were priced. Your lineup, win odds, late swaps and
  title odds still update every run.

Light views no longer search trades at all: the search was about 70% of
each one's time, and only your trades and the full views' (Let Kevin Cook)
are wanted. Your own trade search never used the other teams' views.

### Games on the field with no stat line

nflverse's weekly player stats have no row for a skill player who played
offensive snaps and carried, caught and scored nothing. Every trailing
average in the projection (volume, efficiency, the last-17-games windows)
skipped those games, so backups and rotational players ran high: in weeks 3-5
of 2026 players projected under 3 points by the model alone were projected
2.65 and scored 1.79 (the published blend, which also uses the platform,
Vegas, FantasyPros and Sleeper, was at 1.77). The same gap made the
touchdown model overrate backup receivers until it was trained on every game
a player was on the field.

`roster/idle.py` adds a zero-point row for every game a QB/RB/WR/TE had an
offensive snap and no stat line (matched by name and position from the snap
counts), at the top of `project_snapshot`, so everything downstream sees
them. `scripts/check_active_zero.py` replays the production projection on the
games before weeks 4, 6, 8, 10, 12 and 14 of 2023-2025, the switch off and on
(`roster.idle_games`), graded on every snap-active player, a zero counting as
a zero (5,859 player-weeks):

| | MAE | Bias | Rank corr |
|---|---|---|---|
| Before | 4.091 | -0.15 | 0.717 |
| After | 4.008 | -0.33 | 0.722 |

Better in all three seasons (3.981 -> 3.893, 4.095 -> 4.014, 4.204 -> 4.123)
and at every position (tight ends most, 2.985 -> 2.814). Players projected
under 3 went from 1.93 to 1.53 against 1.61 scored; those between 3 and 6
from 4.32 to 3.90 against 3.83. The mid and high groups are unchanged. It
moves live values modestly: fringe players' season values fall 0.2-0.6 a
game (Tyler Higbee 9.4 -> 8.8), starters not at all.

A variant that kept the position averages (the shrinkage target) to real
stat lines only was no better (4.0245). **Known limit:** a game he left
injured after a few snaps also counts as a zero game (Rashee Rice, 6 snaps in
week 4, 14.6 -> 12.2); injury exits are a small share of the games and the
net is positive, but the weeks after a long absence read low until he is back
on the field.

## Lineup tab: against the opponent's optimal lineup

The Lineup tab has an **Opponent: As set / Optimal** toggle. "Optimal" faces
the opponent's best projected lineup among players whose games have not
started (`optimise(..., opponent_mode="ideal")`), and your recommended lineup,
P(win), head to head, bench, fallbacks and lineup lab are all re-optimised
for it during the refresh. A page cannot redo that per click: P(win) is
scored against the opponent's whole correlated distribution, and the best
lineup can change with it (more upside as an underdog), so both versions are
computed in Python and the toggle only switches between them. The toggle
appears only when the optimal lineup differs from the set one; the choice is
remembered per league in the browser. The extra optimisation also feeds your
own "ideal" row in "Any team's best lineup", so nothing is simulated twice.

**Late swaps count in "optimal".** First version picked the opponent's most
projected lineup. On week 8 ESPN that benched Matthew Golden (questionable,
10.0 with the sit folded in) for Jameson Williams (11.3) -- worse than their
set lineup, which starts Golden and swaps Williams in if he sits, so your
P(win) went *up* against the "optimal" one (65% -> 67%). Optimal now ranks
lineups by projected points plus each questionable starter's chance of
sitting times his bench replacement (`lineup.best_expected`, the same swap
the simulation makes); the top 200 by projection are re-ranked. The
"already optimal" note now sits at the top of the Lineup tab, not the bottom.

## Pickups for this matchup: this week's win, the rest, and the net

The single "Season" column is split in three, all in points of P(title):
**Wk** = the P(win) gain times this game's title swing (title odds if you
win it less if you lose it, from the stakes); **Rest** = the whole move's
title odds less that -- the weeks after, his value net of the player you
drop (who may be claimed by a rival); **Net** = the whole move on the season
simulation (add him, drop that player, every week this one included), less
the waiver priority a claim spends (`TitleEngine.priority_cost`, 0 for a
free agent). The verdict -- claim/add, toss-up, pass -- goes on Net against
the same bar as the title moves (0.2 points or twice the noise). Above the
table: which of your starters an available player out-projects this week,
and by whom.
