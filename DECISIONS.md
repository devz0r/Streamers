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

### The workflow runs at 11:00 UTC
07:00 ET during EDT. Tuesday scores the completed week; Wednesday ranks and
publishes the upcoming one. Wednesday is deliberate: waiver claims process
Wednesday morning, and lines have settled by then.

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
concentrates the spend on a new Sunday 07:00 ET publish -- the run where the
props exist, the inactives are known, and the lineup is actually being set.
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
