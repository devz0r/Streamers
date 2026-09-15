# Week 1 review (2026)

_Generated 2026-09-15 11:04 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 32 | 6.37 | 6.12 | 0.213 | 0.338 | 60% | 60% | 12.5 | 9.6 |
| K | 22 | 3.68 | 3.43 | -0.174 | 0.034 | 60% | 60% | 5.8 | 7.8 |

## D/ST: what it got wrong

Rank correlation 0.213 against a Vegas-only 0.338 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 12.5 points against a slate average of 9.6, and 60% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| TEN | NYJ | 12.9 | -3.0 | -15.9 |
| LAC | ARI | 13.7 | 1.5 | -12.2 |
| CAR | CHI | 7.6 | -4.0 | -11.6 |
| HOU | BUF | 9.8 | 0.0 | -9.8 |
| IND | BAL | 8.2 | 0.0 | -8.2 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| CIN | TB | 8.9 | 26.0 | +17.1 |
| PIT | ATL | 11.6 | 26.0 | +14.4 |
| KC | DEN | 10.1 | 20.5 | +10.4 |
| LV | MIA | 12.0 | 22.0 | +10.0 |
| ARI | LAC | 7.5 | 17.0 | +9.5 |

## Kicker: what it got wrong

Rank correlation -0.174 against a Vegas-only 0.034 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 5.8 points against a slate average of 7.8, and 60% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| H.Mevis | SF | 9.0 | 1.0 | -8.0 |
| C.Dicker | ARI | 9.6 | 2.0 | -7.6 |
| B.Aubrey | NYG | 8.4 | 1.5 | -6.9 |
| W.Lutz | KC | 7.7 | 4.0 | -3.7 |
| J.Slye | NYJ | 8.4 | 5.0 | -3.4 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| E.McPherson | TB | 8.6 | 19.0 | +10.4 |
| C.Ryland | LAC | 7.7 | 15.0 | +7.3 |
| T.Loop | IND | 8.3 | 14.0 | +5.7 |
| C.Santos | CAR | 8.2 | 12.0 | +3.8 |
| C.McLaughlin | CIN | 7.6 | 11.0 | +3.4 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.316 | -0.384 | -0.384 | -0.324 | -1.301 | - |
| Point spread | 0.273 | 0.367 | 0.367 | 0.284 | 1.178 | - |
| Opp points per drive | -0.219 | -0.342 | -0.342 | -0.233 | 0.090 | - |
| Opp sack rate allowed (O-line) | 0.210 | 0.037 | 0.037 | 0.190 | 0.163 | - |
| Sack opportunity (def x O-line x volume) | 0.204 | -0.159 | -0.159 | 0.163 | 0.100 | - |
| Opp pressure rate allowed | 0.191 | -0.086 | -0.086 | 0.159 | 0.239 | - |
| Game total | -0.161 | -0.148 | -0.148 | -0.159 | -0.582 | - |
| Takeaway opportunity | 0.119 | -0.021 | -0.021 | 0.103 | 0.077 | - |
| Opp INT rate | 0.109 | 0.117 | 0.117 | 0.110 | 0.068 | - |
| Defense points allowed per drive | -0.108 | 0.043 | 0.043 | -0.091 | -0.008 | - |
| Opp turnover-worthy play rate | 0.096 | 0.104 | 0.104 | 0.097 | -0.052 | - |
| Opp drives per game | 0.092 | 0.333 | 0.333 | 0.119 | 0.109 | - |

**Biggest divergences from the historical prior:**

- **Sack opportunity (def x O-line x volume)** is running weaker this season (-0.159 vs 0.204 historically). Blended to 0.163, so its weight multiplier is 1.60x.
- **Defense sack rate** is running weaker this season (-0.238 vs 0.054 historically). Blended to 0.021, so its weight multiplier is 0.26x.
- **Opp pressure rate allowed** is running weaker this season (-0.086 vs 0.191 historically). Blended to 0.159, so its weight multiplier is 1.67x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.128 | 0.015 | 0.015 | 0.115 | 0.394 | - |
| Team implied total | 0.117 | 0.077 | 0.077 | 0.113 | 0.253 | - |
| Implied total x FG rate | 0.105 | -0.079 | -0.079 | 0.084 | -0.002 | - |
| Implied total x red-zone stall | 0.101 | -0.053 | -0.053 | 0.084 | -0.004 | - |
| Dome / indoors | 0.095 | 0.083 | 0.083 | 0.094 | 0.272 | - |
| Opponent implied total | -0.094 | 0.053 | 0.053 | -0.077 | -0.000 | - |
| Wind (mph) | -0.092 | -0.191 | -0.191 | -0.103 | -0.318 | - |
| Wind x 50+ attempt share | -0.079 | -0.163 | -0.163 | -0.089 | 0.006 | - |
| Kicker 50+ attempt share | 0.054 | 0.135 | 0.135 | 0.063 | 0.056 | - |
| Home field | 0.046 | -0.176 | -0.176 | 0.021 | 0.040 | - |
| TDs per drive | 0.037 | 0.060 | 0.060 | 0.040 | -0.006 | - |
| Red-zone trips per drive | 0.035 | 0.003 | 0.003 | 0.032 | -0.011 | - |

**Biggest divergences from the historical prior:**

- **Opp points per drive allowed** is running weaker this season (-0.318 vs 0.031 historically). Blended to -0.008, so its weight multiplier is 0.24x.
- **Opp drives per game allowed** is running stronger this season (0.293 vs -0.010 historically). Blended to 0.025, so its weight multiplier is 0.28x.
- **Kicker accuracy 40-49** is running weaker this season (-0.248 vs 0.030 historically). Blended to -0.002, so its weight multiplier is 0.23x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| CIN | 9.58 | 24.00 | 11.18 | +1.60 | 1 |
| TEN | 8.95 | -2.00 | 7.74 | -1.22 | 1 |
| DET | 9.07 | 19.50 | 10.23 | +1.16 | 1 |
| PIT | 11.09 | 21.00 | 12.19 | +1.10 | 1 |
| LV | 8.21 | 17.00 | 9.19 | +0.98 | 1 |

**Points Allowed**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| CAR | 24.91 | 59.00 | 28.70 | +3.79 | 1 |
| GB | 21.24 | 39.00 | 23.22 | +1.97 | 1 |
| IND | 24.07 | 41.00 | 25.95 | +1.88 | 1 |
| NYJ | 24.49 | 10.00 | 22.88 | -1.61 | 1 |
| HOU | 22.08 | 36.00 | 23.62 | +1.55 | 1 |

**Sack Rate**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| JAX | 2.01 | 5.00 | 2.34 | +0.33 | 1 |
| LV | 2.15 | 5.00 | 2.47 | +0.32 | 1 |
| DET | 2.29 | 5.00 | 2.59 | +0.30 | 1 |
| LA | 2.63 | 0.00 | 2.34 | -0.29 | 1 |
| MIA | 2.59 | 0.00 | 2.30 | -0.29 | 1 |

## Season-to-date calibration

| Position | Weeks | MAE | Rank corr | Top-5 hit rate |
|---|---|---|---|---|
| DST | 1 | 6.37 | 0.213 | 60% |
| K | 1 | 3.68 | -0.174 | 60% |

