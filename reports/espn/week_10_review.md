# Week 10 review (2026)

_Generated 2026-11-17 16:55 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 28 | 5.16 | 5.10 | 0.589 | 0.589 | 60% | 60% | 12.2 | 8.6 |
| K | 27 | 3.44 | 3.28 | -0.283 | -0.089 | 60% | 40% | 9.0 | 8.1 |

## D/ST: what it got wrong

Rank correlation 0.589 against a Vegas-only 0.589 -- slightly ahead of ranking on the Vegas number alone. The five recommended units averaged 12.2 points against a slate average of 8.6, and 60% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| ATL | KC | 6.7 | -4.0 | -10.7 |
| PIT | CIN | 6.4 | -4.0 | -10.4 |
| SF | DAL | 6.6 | -0.5 | -7.1 |
| BAL | LAC | 12.8 | 6.0 | -6.8 |
| JAX | TEN | 12.1 | 5.5 | -6.6 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| MIN | GB | 8.9 | 29.5 | +20.6 |
| DET | NE | 8.3 | 18.5 | +10.2 |
| LA | ARI | 14.5 | 22.0 | +7.5 |
| CAR | NO | 7.3 | 13.5 | +6.2 |
| SEA | LV | 12.5 | 18.0 | +5.5 |

## Kicker: what it got wrong

Rank correlation -0.283 against a Vegas-only -0.089 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 9.0 points against a slate average of 8.1, and 60% of them finished startable (Vegas-only: 40%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| J.Myers | LV | 8.8 | 2.0 | -6.8 |
| D.Zvada | WAS | 8.6 | 2.0 | -6.6 |
| Z.Gonzalez | IND | 8.7 | 3.5 | -5.2 |
| A.Borregales | DET | 8.4 | 4.0 | -4.4 |
| J.Bates | NE | 8.8 | 5.0 | -3.8 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| J.Slye | JAX | 7.0 | 17.0 | +10.0 |
| C.Little | TEN | 9.1 | 19.0 | +9.9 |
| J.Sanders | BUF | 7.3 | 14.0 | +6.7 |
| W.Reichard | GB | 8.2 | 12.0 | +3.8 |
| C.Dicker | BAL | 7.3 | 11.0 | +3.7 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.322 | -0.340 | -0.391 | -0.332 | -1.251 | -0.027 |
| Point spread | 0.278 | 0.256 | 0.384 | 0.266 | 1.097 | +0.013 |
| Opp points per drive | -0.221 | -0.219 | -0.255 | -0.220 | 0.141 | -0.003 |
| Opp sack rate allowed (O-line) | 0.204 | 0.218 | 0.314 | 0.212 | 0.167 | -0.006 |
| Sack opportunity (def x O-line x volume) | 0.199 | 0.169 | 0.194 | 0.183 | 0.089 | -0.002 |
| Opp pressure rate allowed | 0.186 | 0.083 | 0.164 | 0.130 | 0.175 | -0.013 |
| Game total | -0.163 | -0.224 | -0.153 | -0.196 | -0.615 | -0.029 |
| Takeaway opportunity | 0.120 | 0.153 | 0.298 | 0.138 | 0.097 | +0.003 |
| Opp INT rate | 0.109 | 0.042 | 0.082 | 0.073 | 0.013 | +0.007 |
| Defense points allowed per drive | -0.108 | -0.141 | -0.168 | -0.126 | -0.010 | -0.003 |
| Opp turnover-worthy play rate | 0.096 | 0.097 | 0.124 | 0.097 | -0.027 | +0.006 |
| Opp drives per game | 0.093 | 0.127 | 0.042 | 0.111 | 0.121 | -0.002 |

**Biggest divergences from the historical prior:**

- **Opp fumble-lost rate** is running stronger this season (0.157 vs 0.000 historically). Blended to 0.085, so its weight multiplier is 0.60x.
- **Opp pass rate** is running weaker this season (-0.093 vs 0.042 historically). Blended to -0.032, so its weight multiplier is 0.31x.
- **Opp pressure-to-sack rate** is running stronger this season (0.210 vs 0.077 historically). Blended to 0.150, so its weight multiplier is 2.19x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.128 | 0.017 | 0.110 | 0.068 | 0.346 | -0.024 |
| Team implied total | 0.117 | 0.028 | 0.138 | 0.068 | 0.230 | -0.007 |
| Implied total x FG rate | 0.105 | -0.006 | 0.123 | 0.045 | -0.005 | +0.000 |
| Implied total x red-zone stall | 0.101 | 0.023 | 0.193 | 0.059 | 0.005 | +0.003 |
| Wind (mph) | -0.099 | -0.019 | -0.014 | -0.055 | -0.403 | -0.018 |
| Dome / indoors | 0.095 | -0.033 | -0.001 | 0.025 | 0.028 | -0.059 |
| Opponent implied total | -0.094 | 0.001 | -0.029 | -0.042 | 0.001 | +0.000 |
| Wind x 50+ attempt share | -0.083 | -0.000 | 0.005 | -0.038 | 0.008 | -0.002 |
| Kicker 50+ attempt share | 0.054 | 0.072 | 0.044 | 0.064 | 0.067 | -0.001 |
| Home field | 0.046 | -0.115 | -0.028 | -0.042 | -0.098 | +0.016 |
| TDs per drive | 0.037 | -0.019 | -0.053 | 0.007 | -0.004 | +0.001 |
| Red-zone trips per drive | 0.035 | -0.038 | -0.045 | -0.005 | -0.006 | -0.001 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.115 vs 0.046 historically). Blended to -0.042, so its weight multiplier is 0.92x.
- **Dome / indoors** is running weaker this season (-0.033 vs 0.095 historically). Blended to 0.025, so its weight multiplier is 0.53x.
- **Implied total x FG rate** is running weaker this season (-0.006 vs 0.105 historically). Blended to 0.045, so its weight multiplier is 0.85x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 10.49 | 6.00 | 8.00 | -2.50 | 10 |
| PHI | 9.45 | 5.06 | 7.12 | -2.33 | 9 |
| MIN | 9.69 | 13.44 | 11.68 | +1.99 | 9 |
| MIA | 9.00 | 5.94 | 7.38 | -1.62 | 9 |
| BAL | 8.71 | 5.80 | 7.09 | -1.62 | 10 |

**Points Allowed**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| MIN | 22.28 | 14.00 | 17.89 | -4.38 | 9 |
| JAX | 23.10 | 16.00 | 19.34 | -3.76 | 9 |
| BAL | 20.42 | 25.90 | 23.47 | +3.04 | 10 |
| NO | 21.04 | 26.22 | 23.78 | +2.75 | 9 |
| LV | 24.19 | 28.90 | 26.80 | +2.62 | 10 |

**Sack Rate**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| MIN | 2.70 | 4.11 | 3.45 | +0.75 | 9 |
| DET | 2.29 | 3.56 | 2.96 | +0.67 | 9 |
| MIA | 2.59 | 1.44 | 1.98 | -0.60 | 9 |
| HOU | 2.59 | 3.67 | 3.16 | +0.57 | 9 |
| PHI | 2.72 | 1.78 | 2.22 | -0.50 | 9 |

## Season-to-date calibration

| Position | Weeks | MAE | Rank corr | Top-5 hit rate |
|---|---|---|---|---|
| DST | 10 | 5.44 | 0.358 | 56% |
| K | 10 | 3.82 | 0.023 | 46% |

