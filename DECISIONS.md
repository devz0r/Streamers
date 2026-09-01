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

### ESPN defaults, all in `config.yaml`
No scoring value is hard-coded anywhere in the codebase; `streamer.scoring`
reads them and `tests/test_scoring.py` asserts that changing the config changes
the result. The points-allowed ladder is validated on load for gaps, overlaps
and a missing open-ended top tier.

### Missed field goals default to -1 at every distance
ESPN's default sheet exposes a single "Total Field Goals Missed (FGM)" line at
-1 rather than per-distance penalties. Leagues that only penalise short misses
can override per bucket (`fg_missed_50_plus: 0.0`) without touching code.

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

### Points-allowed tiers are a distribution, never a point estimate
The ESPN ladder is a step function, so `E[tier(PA)] != tier(E[PA])`. A defence
projected to allow 20.5 points is not "0 points"; it is a mixture that is mostly
0 and -1 with real mass on +3 and +4. `TierModel` predicts mean points allowed,
then samples historical residuals — bucketed by predicted level, because
high-scoring spots are also more variable — and bins the draws onto the ladder.
This is what produces `P(shutout)` and `P(under 14)`.

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
