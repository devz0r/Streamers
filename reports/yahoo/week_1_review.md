# Week 1 review (2026)

_Generated 2026-09-15 11:04 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 32 | 4.41 | 4.34 | 0.208 | 0.294 | 60% | 60% | 8.4 | 6.6 |
| K | 22 | 3.72 | 3.55 | -0.168 | 0.001 | 60% | 60% | 5.8 | 8.0 |

## D/ST: what it got wrong

Rank correlation 0.208 against a Vegas-only 0.294 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 8.4 points against a slate average of 6.6, and 60% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| LAC | ARI | 9.7 | 1.0 | -8.7 |
| TEN | NYJ | 8.6 | 0.0 | -8.6 |
| HOU | BUF | 7.0 | -1.0 | -8.0 |
| GB | MIN | 7.2 | 1.0 | -6.2 |
| CLE | JAX | 5.7 | 0.0 | -5.7 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| CIN | TB | 6.6 | 18.0 | +11.4 |
| PIT | ATL | 8.3 | 18.0 | +9.7 |
| ARI | LAC | 4.8 | 13.0 | +8.2 |
| LV | MIA | 8.4 | 14.0 | +5.6 |
| KC | DEN | 7.4 | 13.0 | +5.6 |

## Kicker: what it got wrong

Rank correlation -0.168 against a Vegas-only 0.001 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 5.8 points against a slate average of 8.0, and 60% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| H.Mevis | SF | 9.3 | 1.0 | -8.3 |
| C.Dicker | ARI | 10.0 | 2.0 | -8.0 |
| B.Aubrey | NYG | 8.7 | 2.0 | -6.7 |
| W.Lutz | KC | 8.0 | 4.0 | -4.0 |
| J.Slye | NYJ | 8.7 | 5.0 | -3.7 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| E.McPherson | TB | 8.9 | 19.0 | +10.1 |
| C.Ryland | LAC | 8.0 | 16.0 | +8.0 |
| T.Loop | IND | 8.6 | 14.0 | +5.4 |
| C.Santos | CAR | 8.6 | 12.0 | +3.4 |
| C.McLaughlin | CIN | 8.0 | 11.0 | +3.0 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.289 | -0.313 | -0.313 | -0.292 | -0.907 | - |
| Point spread | 0.265 | 0.337 | 0.337 | 0.274 | 0.915 | - |
| Opp points per drive | -0.185 | -0.285 | -0.285 | -0.196 | 0.077 | - |
| Opp sack rate allowed (O-line) | 0.161 | -0.086 | -0.086 | 0.133 | 0.070 | - |
| Sack opportunity (def x O-line x volume) | 0.160 | -0.236 | -0.236 | 0.115 | 0.034 | - |
| Opp pressure rate allowed | 0.152 | -0.139 | -0.139 | 0.119 | 0.136 | - |
| Game total | -0.124 | -0.070 | -0.070 | -0.118 | -0.272 | - |
| Takeaway opportunity | 0.119 | -0.042 | -0.042 | 0.101 | 0.064 | - |
| Defense points allowed per drive | -0.098 | 0.071 | 0.071 | -0.078 | 0.006 | - |
| Opp INT rate | 0.093 | 0.014 | 0.014 | 0.084 | 0.031 | - |
| Defense INT rate | 0.085 | -0.064 | -0.064 | 0.068 | 0.055 | - |
| Opp drives per game | 0.084 | 0.298 | 0.298 | 0.108 | 0.095 | - |

**Biggest divergences from the historical prior:**

- **Sack opportunity (def x O-line x volume)** is running weaker this season (-0.236 vs 0.160 historically). Blended to 0.115, so its weight multiplier is 1.44x.
- **Defense sack rate** is running weaker this season (-0.259 vs 0.041 historically). Blended to 0.007, so its weight multiplier is 0.13x.
- **Opp pressure rate allowed** is running weaker this season (-0.139 vs 0.152 historically). Blended to 0.119, so its weight multiplier is 1.56x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.134 | -0.050 | -0.050 | 0.113 | 0.401 | - |
| Team implied total | 0.122 | 0.008 | 0.008 | 0.109 | 0.255 | - |
| Implied total x FG rate | 0.109 | -0.116 | -0.116 | 0.084 | -0.001 | - |
| Implied total x red-zone stall | 0.106 | -0.089 | -0.089 | 0.084 | -0.003 | - |
| Opponent implied total | -0.099 | 0.089 | 0.089 | -0.077 | -0.000 | - |
| Dome / indoors | 0.094 | 0.081 | 0.081 | 0.093 | 0.266 | - |
| Wind (mph) | -0.090 | -0.204 | -0.204 | -0.103 | -0.305 | - |
| Wind x 50+ attempt share | -0.077 | -0.170 | -0.170 | -0.087 | 0.006 | - |
| Kicker 50+ attempt share | 0.058 | 0.132 | 0.132 | 0.067 | 0.058 | - |
| Home field | 0.046 | -0.229 | -0.229 | 0.014 | 0.019 | - |
| TDs per drive | 0.038 | 0.026 | 0.026 | 0.037 | -0.007 | - |
| Red-zone trips per drive | 0.037 | -0.029 | -0.029 | 0.029 | -0.011 | - |

**Biggest divergences from the historical prior:**

- **Opp points per drive allowed** is running weaker this season (-0.340 vs 0.033 historically). Blended to -0.009, so its weight multiplier is 0.25x.
- **Opp red-zone TD rate allowed** is running weaker this season (-0.287 vs 0.025 historically). Blended to -0.011, so its weight multiplier is 0.30x.
- **Opp drives per game allowed** is running stronger this season (0.289 vs -0.009 historically). Blended to 0.025, so its weight multiplier is 0.25x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| CIN | 6.53 | 18.00 | 7.80 | +1.27 | 1 |
| PIT | 7.25 | 14.00 | 8.00 | +0.75 | 1 |
| ARI | 6.01 | 12.00 | 6.68 | +0.67 | 1 |
| CLE | 6.64 | 1.00 | 6.01 | -0.63 | 1 |
| TEN | 5.59 | 0.00 | 4.97 | -0.62 | 1 |

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
| DST | 1 | 4.41 | 0.208 | 60% |
| K | 1 | 3.72 | -0.168 | 60% |

