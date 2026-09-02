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

### Waivers price each pickup against the spot it would cost
A pickup is only as good as the player it displaces, so every free agent is
scored as an (add, drop) pair: `w · next-week gain + (1-w) · rest-of-season
gain`, with `w = 0.35 + 0.5 · week/18` so stashes matter in September and only
this week matters in December. Drops never go below one QB/TE/K/DST unless the
pickup plays that position, and never touch the best player at a position.
Drops are assigned greedily in score order so the list reads as a set of moves
that can all be made together; a stash that only cleared the bar against the
dead roster spot disappears once that spot is spent on a better pickup, which
is the honest answer. Moves under a 1.0-point blended gain are not shown.
