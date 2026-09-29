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

### The profile switch is CSS-only
The page carries both leagues and a segmented switch between them, built from
one hidden radio per profile plus `:checked` sibling rules. No JavaScript, so it
switches instantly on a phone, survives a stale cache, and keeps the page a
single self-contained file. The rules are generated per profile, so adding a
third league needs only config. A page rendered with one profile omits the
switch entirely.

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
status separates Questionable further (DNP 48%, limited 71%, full 84%),
but the platforms do not carry it, so it is not used. The page now shows a
tagged player's projection both ways: "9.7 (13.5 if he plays, 72%)".

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
are offered), shifted by 0.75 when an uneven deal gives him -- or takes
from him -- the best player in it, and scaled by engagement (0.6 for a team
with no pickups, 1.0 at one a week). That shape is an assumption: offers
and refusals are not published, so it cannot be fitted. It ranks offers;
the percentage is not a measured rate.

Three views of the same priced trades: **top** by expected gain (chance of
a yes x title odds gained), **best for you** by title odds among offers
with at least a 15% chance, **most likely yes** by the chance among offers
that clearly help. Each card says why for both sides: sell-high and
buy-low gaps, who he would start over, name value, a hot or cold start,
injuries, how active he is. On the ESPN league at week 3, 999 trades
screened and 66 priced took 20 seconds; the top offer (London for Swift)
had an 86% yes-chance and +1.3 points of title odds.

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

