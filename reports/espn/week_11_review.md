# Week 11 review (2026)

_Generated 2026-11-24 18:56 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 26 | 4.61 | 4.62 | 0.615 | 0.671 | 100% | 100% | 13.7 | 8.3 |
| K | 26 | 3.88 | 3.85 | 0.104 | 0.155 | 60% | 60% | 9.7 | 9.4 |

## D/ST: what it got wrong

Rank correlation 0.615 against a Vegas-only 0.671 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 13.7 points against a slate average of 8.3, and 100% of them finished startable (Vegas-only: 100%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| TEN | DAL | 3.2 | -7.0 | -10.2 |
| LAC | NYJ | 10.8 | 3.0 | -7.8 |
| MIN | SF | 7.5 | 0.0 | -7.5 |
| TB | DET | 7.4 | 0.0 | -7.4 |
| JAX | NYG | 11.9 | 6.0 | -5.9 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| CHI | NO | 12.0 | 22.5 | +10.5 |
| SF | MIN | 10.7 | 16.5 | +5.8 |
| MIA | BUF | 3.6 | 9.0 | +5.4 |
| NO | CHI | 4.3 | 9.5 | +5.2 |
| NYJ | LAC | 9.2 | 14.0 | +4.8 |

## Kicker: what it got wrong

Rank correlation 0.104 against a Vegas-only 0.155 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 9.7 points against a slate average of 9.4, and 60% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| S.Shrader | HOU | 7.6 | 1.0 | -6.6 |
| C.McLaughlin | DET | 8.5 | 3.0 | -5.5 |
| C.Ryland | KC | 6.3 | 1.0 | -5.3 |
| W.Reichard | SF | 8.0 | 3.0 | -5.0 |
| E.Pineiro | MIN | 8.4 | 5.0 | -3.4 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| D.Stevens | CIN | 7.4 | 18.0 | +10.6 |
| J.Elliott | PIT | 7.7 | 18.0 | +10.3 |
| T.Bass | MIA | 8.8 | 17.0 | +8.2 |
| D.Zvada | JAX | 7.7 | 15.0 | +7.3 |
| D.Carlson | BUF | 6.8 | 13.0 | +6.2 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.322 | -0.365 | -0.511 | -0.347 | -1.262 | -0.010 |
| Point spread | 0.278 | 0.290 | 0.486 | 0.285 | 1.113 | +0.016 |
| Opp points per drive | -0.221 | -0.240 | -0.390 | -0.232 | 0.140 | -0.001 |
| Opp sack rate allowed (O-line) | 0.204 | 0.237 | 0.400 | 0.223 | 0.163 | -0.004 |
| Sack opportunity (def x O-line x volume) | 0.199 | 0.188 | 0.252 | 0.193 | 0.090 | +0.001 |
| Opp pressure rate allowed | 0.186 | 0.092 | 0.184 | 0.133 | 0.166 | -0.009 |
| Game total | -0.163 | -0.222 | -0.198 | -0.197 | -0.609 | +0.006 |
| Takeaway opportunity | 0.120 | 0.152 | 0.314 | 0.138 | 0.096 | -0.000 |
| Opp INT rate | 0.109 | 0.080 | 0.255 | 0.093 | 0.028 | +0.014 |
| Defense points allowed per drive | -0.108 | -0.146 | -0.208 | -0.130 | -0.009 | +0.001 |
| Opp turnover-worthy play rate | 0.096 | 0.136 | 0.297 | 0.119 | -0.012 | +0.015 |
| Opp drives per game | 0.093 | 0.128 | 0.103 | 0.113 | 0.127 | +0.006 |

**Biggest divergences from the historical prior:**

- **Opp fumble-lost rate** is running stronger this season (0.175 vs 0.000 historically). Blended to 0.099, so its weight multiplier is 0.60x.
- **Opp pressure-to-sack rate** is running stronger this season (0.227 vs 0.077 historically). Blended to 0.162, so its weight multiplier is 2.44x.
- **Opp pass rate** is running weaker this season (-0.082 vs 0.042 historically). Blended to -0.028, so its weight multiplier is 0.28x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.128 | 0.039 | 0.085 | 0.077 | 0.342 | -0.004 |
| Team implied total | 0.117 | 0.051 | 0.124 | 0.080 | 0.238 | +0.008 |
| Implied total x FG rate | 0.105 | 0.006 | 0.066 | 0.049 | -0.007 | -0.002 |
| Implied total x red-zone stall | 0.101 | 0.023 | 0.078 | 0.057 | 0.001 | -0.004 |
| Wind (mph) | -0.099 | 0.014 | 0.048 | -0.035 | -0.345 | +0.059 |
| Dome / indoors | 0.095 | -0.038 | -0.035 | 0.020 | 0.045 | +0.016 |
| Opponent implied total | -0.094 | -0.010 | -0.015 | -0.046 | -0.000 | -0.001 |
| Wind x 50+ attempt share | -0.083 | 0.036 | 0.079 | -0.016 | 0.006 | -0.002 |
| Kicker 50+ attempt share | 0.054 | 0.083 | 0.135 | 0.070 | 0.075 | +0.009 |
| Home field | 0.046 | -0.081 | 0.018 | -0.026 | -0.074 | +0.023 |
| TDs per drive | 0.037 | 0.003 | 0.039 | 0.018 | -0.008 | -0.004 |
| Red-zone trips per drive | 0.035 | -0.031 | -0.007 | -0.002 | -0.006 | -0.000 |

**Biggest divergences from the historical prior:**

- **Dome / indoors** is running weaker this season (-0.038 vs 0.095 historically). Blended to 0.020, so its weight multiplier is 0.50x.
- **Home field** is running weaker this season (-0.081 vs 0.046 historically). Blended to -0.026, so its weight multiplier is 0.56x.
- **Wind x 50+ attempt share** is running stronger this season (0.036 vs -0.083 historically). Blended to -0.016, so its weight multiplier is 0.50x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 10.49 | 6.55 | 8.21 | -2.29 | 11 |
| PHI | 9.45 | 5.35 | 7.17 | -2.28 | 10 |
| CHI | 8.10 | 11.00 | 9.71 | +1.61 | 10 |
| IND | 9.07 | 6.50 | 7.58 | -1.49 | 11 |
| MIN | 9.69 | 12.30 | 11.14 | +1.45 | 10 |

**Points Allowed**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| BAL | 20.42 | 26.73 | 24.07 | +3.65 | 11 |
| JAX | 23.10 | 16.70 | 19.55 | -3.56 | 10 |
| MIN | 22.28 | 16.10 | 18.84 | -3.43 | 10 |
| LV | 24.19 | 29.73 | 27.39 | +3.21 | 11 |
| TEN | 23.69 | 29.00 | 26.64 | +2.95 | 10 |

**Sack Rate**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DET | 2.29 | 3.60 | 3.02 | +0.73 | 10 |
| CHI | 2.06 | 3.20 | 2.69 | +0.63 | 10 |
| MIN | 2.70 | 3.80 | 3.31 | +0.61 | 10 |
| MIA | 2.59 | 1.60 | 2.04 | -0.55 | 10 |
| LAC | 2.55 | 1.60 | 2.02 | -0.53 | 10 |

## Season-to-date calibration

| Position | Weeks | MAE | Rank corr | Top-5 hit rate |
|---|---|---|---|---|
| DST | 11 | 5.38 | 0.382 | 60% |
| K | 11 | 3.83 | 0.030 | 47% |

