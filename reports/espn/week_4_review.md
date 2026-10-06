# Week 4 review (2026)

_Generated 2026-10-06 14:14 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 32 | 4.31 | 4.42 | 0.512 | 0.521 | 60% | 60% | 13.2 | 8.3 |
| K | 32 | 3.70 | 3.66 | 0.244 | 0.260 | 40% | 20% | 10.8 | 8.9 |

## D/ST: what it got wrong

Rank correlation 0.512 against a Vegas-only 0.521 -- essentially level with the Vegas-only ranking. The five recommended units averaged 13.2 points against a slate average of 8.3, and 60% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| NO | ATL | 9.1 | -2.5 | -11.6 |
| KC | LV | 11.3 | 2.5 | -8.8 |
| LV | KC | 7.8 | 0.0 | -7.8 |
| DEN | SF | 6.8 | -1.0 | -7.8 |
| BUF | NE | 11.7 | 4.0 | -7.7 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| NYG | ARI | 8.9 | 20.5 | +11.6 |
| SEA | LAC | 14.3 | 20.0 | +5.7 |
| CLE | PIT | 11.1 | 16.5 | +5.4 |
| ARI | NYG | 10.6 | 15.0 | +4.4 |
| WAS | IND | 7.5 | 10.5 | +3.0 |

## Kicker: what it got wrong

Rank correlation 0.244 against a Vegas-only 0.260 -- essentially level with the Vegas-only ranking. The five recommended units averaged 10.8 points against a slate average of 8.9, and 40% of them finished startable (Vegas-only: 20%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| C.McLaughlin | GB | 8.2 | 2.0 | -6.2 |
| C.Boswell | CLE | 7.9 | 3.0 | -4.9 |
| A.Borregales | BUF | 7.5 | 3.0 | -4.5 |
| E.Pineiro | DEN | 9.0 | 5.0 | -4.0 |
| D.Carlson | ATL | 9.0 | 5.0 | -4.0 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| W.Reichard | MIA | 9.8 | 19.0 | +9.2 |
| J.Bates | CAR | 8.9 | 18.0 | +9.1 |
| M.Gay | KC | 8.6 | 16.0 | +7.4 |
| S.Shrader | WAS | 9.0 | 16.0 | +7.0 |
| K.Fairbairn | DAL | 9.2 | 16.0 | +6.8 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.316 | -0.331 | -0.331 | -0.321 | -1.290 | -0.010 |
| Point spread | 0.273 | 0.205 | 0.205 | 0.250 | 1.093 | -0.024 |
| Opp points per drive | -0.219 | -0.202 | -0.202 | -0.213 | 0.119 | +0.005 |
| Opp sack rate allowed (O-line) | 0.210 | 0.182 | 0.182 | 0.201 | 0.176 | +0.006 |
| Sack opportunity (def x O-line x volume) | 0.204 | 0.179 | 0.179 | 0.195 | 0.150 | +0.002 |
| Opp pressure rate allowed | 0.191 | -0.010 | -0.010 | 0.123 | 0.192 | -0.008 |
| Game total | -0.161 | -0.281 | -0.281 | -0.201 | -0.689 | -0.054 |
| Takeaway opportunity | 0.119 | 0.131 | 0.131 | 0.123 | 0.090 | +0.007 |
| Opp INT rate | 0.109 | -0.029 | -0.029 | 0.062 | 0.010 | -0.020 |
| Defense points allowed per drive | -0.108 | -0.142 | -0.142 | -0.120 | -0.030 | +0.001 |
| Opp turnover-worthy play rate | 0.096 | 0.016 | 0.016 | 0.069 | -0.046 | -0.002 |
| Opp drives per game | 0.092 | 0.169 | 0.169 | 0.118 | 0.127 | -0.007 |

**Biggest divergences from the historical prior:**

- **Opp pressure rate allowed** is running weaker this season (-0.010 vs 0.191 historically). Blended to 0.123, so its weight multiplier is 1.29x.
- **Opp pressure-to-sack rate** is running stronger this season (0.272 vs 0.080 historically). Blended to 0.145, so its weight multiplier is 1.97x.
- **Opp INT rate** is running weaker this season (-0.029 vs 0.109 historically). Blended to 0.062, so its weight multiplier is 0.74x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.128 | -0.013 | -0.013 | 0.080 | 0.353 | +0.004 |
| Team implied total | 0.117 | 0.041 | 0.041 | 0.091 | 0.244 | +0.013 |
| Implied total x FG rate | 0.105 | -0.028 | -0.028 | 0.060 | -0.004 | -0.002 |
| Implied total x red-zone stall | 0.101 | -0.041 | -0.041 | 0.053 | -0.003 | -0.002 |
| Dome / indoors | 0.095 | -0.014 | -0.014 | 0.058 | 0.225 | +0.003 |
| Opponent implied total | -0.094 | 0.061 | 0.061 | -0.041 | -0.001 | -0.001 |
| Wind (mph) | -0.092 | 0.019 | 0.019 | -0.055 | -0.275 | -0.011 |
| Wind x 50+ attempt share | -0.079 | 0.036 | 0.036 | -0.040 | 0.006 | +0.002 |
| Kicker 50+ attempt share | 0.054 | 0.116 | 0.116 | 0.075 | 0.069 | +0.006 |
| Home field | 0.046 | -0.102 | -0.102 | -0.004 | -0.002 | +0.009 |
| TDs per drive | 0.037 | 0.062 | 0.062 | 0.046 | -0.007 | -0.004 |
| Red-zone trips per drive | 0.035 | 0.009 | 0.009 | 0.026 | -0.009 | -0.002 |

**Biggest divergences from the historical prior:**

- **Opponent implied total** is running stronger this season (0.061 vs -0.094 historically). Blended to -0.041, so its weight multiplier is 0.88x.
- **Home field** is running weaker this season (-0.102 vs 0.046 historically). Blended to -0.004, so its weight multiplier is 0.29x.
- **Implied total x red-zone stall** is running weaker this season (-0.041 vs 0.101 historically). Blended to 0.053, so its weight multiplier is 1.05x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 11.80 | 3.50 | 9.03 | -2.77 | 4 |
| PHI | 10.81 | 4.25 | 8.62 | -2.19 | 4 |
| MIA | 10.29 | 4.38 | 8.32 | -1.97 | 4 |
| MIN | 11.04 | 16.00 | 12.69 | +1.65 | 4 |
| SF | 9.36 | 4.88 | 7.87 | -1.50 | 4 |

**Points Allowed**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| NO | 21.04 | 32.00 | 24.69 | +3.65 | 4 |
| JAX | 23.10 | 13.25 | 19.82 | -3.28 | 4 |
| MIN | 22.28 | 12.75 | 19.10 | -3.18 | 4 |
| CHI | 23.92 | 16.25 | 21.36 | -2.56 | 4 |
| DET | 24.37 | 31.75 | 26.83 | +2.46 | 4 |

**Sack Rate**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| PHI | 2.72 | 1.00 | 2.14 | -0.57 | 4 |
| DAL | 2.61 | 1.00 | 2.07 | -0.54 | 4 |
| MIA | 2.59 | 1.00 | 2.06 | -0.53 | 4 |
| KC | 2.55 | 1.00 | 2.03 | -0.52 | 4 |
| NYG | 2.23 | 0.75 | 1.74 | -0.49 | 4 |

## Season-to-date calibration

| Position | Weeks | MAE | Rank corr | Top-5 hit rate |
|---|---|---|---|---|
| DST | 4 | 5.58 | 0.304 | 35% |
| K | 4 | 3.79 | -0.097 | 30% |

