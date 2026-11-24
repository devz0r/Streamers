# Week 11 review (2026)

_Generated 2026-11-24 18:56 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 26 | 3.82 | 3.82 | 0.552 | 0.582 | 100% | 100% | 13.4 | 6.3 |
| K | 26 | 3.76 | 3.74 | 0.097 | 0.152 | 60% | 60% | 9.6 | 9.8 |

## D/ST: what it got wrong

Rank correlation 0.552 against a Vegas-only 0.582 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 13.4 points against a slate average of 6.3, and 100% of them finished startable (Vegas-only: 100%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| MIN | SF | 5.9 | -3.0 | -8.9 |
| LAC | NYJ | 8.3 | 1.0 | -7.3 |
| TEN | DAL | 3.1 | -4.0 | -7.1 |
| JAX | NYG | 9.3 | 3.0 | -6.3 |
| TB | DET | 6.2 | 0.0 | -6.2 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| SF | MIN | 8.3 | 17.0 | +8.7 |
| CHI | NO | 9.6 | 17.0 | +7.4 |
| HOU | IND | 10.6 | 16.0 | +5.4 |
| DEN | LV | 10.7 | 15.0 | +4.3 |
| MIA | BUF | 2.9 | 7.0 | +4.1 |

## Kicker: what it got wrong

Rank correlation 0.097 against a Vegas-only 0.152 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 9.6 points against a slate average of 9.8, and 60% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| C.McLaughlin | DET | 8.8 | 3.0 | -5.8 |
| C.Ryland | KC | 6.6 | 1.0 | -5.6 |
| S.Shrader | HOU | 7.9 | 3.0 | -4.9 |
| E.Pineiro | MIN | 8.7 | 5.0 | -3.7 |
| K.Fairbairn | IND | 9.7 | 6.0 | -3.7 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| D.Stevens | CIN | 7.8 | 18.0 | +10.2 |
| J.Elliott | PIT | 8.1 | 18.0 | +9.9 |
| T.Bass | MIA | 9.1 | 18.0 | +8.9 |
| D.Zvada | JAX | 8.1 | 16.0 | +7.9 |
| D.Carlson | BUF | 7.2 | 13.0 | +5.8 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.289 | -0.326 | -0.502 | -0.310 | -0.940 | -0.013 |
| Point spread | 0.265 | 0.275 | 0.493 | 0.271 | 0.917 | +0.013 |
| Opp points per drive | -0.185 | -0.189 | -0.366 | -0.187 | 0.133 | +0.002 |
| Opp sack rate allowed (O-line) | 0.161 | 0.194 | 0.371 | 0.180 | 0.101 | -0.000 |
| Sack opportunity (def x O-line x volume) | 0.160 | 0.137 | 0.180 | 0.147 | 0.035 | -0.006 |
| Opp pressure rate allowed | 0.152 | 0.082 | 0.180 | 0.112 | 0.102 | -0.003 |
| Game total | -0.124 | -0.178 | -0.172 | -0.154 | -0.329 | -0.004 |
| Takeaway opportunity | 0.119 | 0.125 | 0.300 | 0.123 | 0.081 | +0.009 |
| Defense points allowed per drive | -0.098 | -0.132 | -0.193 | -0.117 | -0.009 | -0.000 |
| Opp INT rate | 0.093 | 0.046 | 0.227 | 0.066 | 0.008 | +0.012 |
| Defense INT rate | 0.085 | 0.094 | 0.170 | 0.090 | 0.064 | -0.023 |
| Opp drives per game | 0.084 | 0.108 | 0.085 | 0.097 | 0.124 | +0.007 |

**Biggest divergences from the historical prior:**

- **Opp pass rate** is running weaker this season (-0.093 vs 0.052 historically). Blended to -0.030, so its weight multiplier is 0.32x.
- **Opp fumble-lost rate** is running stronger this season (0.136 vs -0.006 historically). Blended to 0.074, so its weight multiplier is 0.60x.
- **Opp pressure-to-sack rate** is running stronger this season (0.181 vs 0.053 historically). Blended to 0.126, so its weight multiplier is 2.08x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.134 | 0.031 | 0.080 | 0.075 | 0.342 | -0.007 |
| Team implied total | 0.122 | 0.045 | 0.120 | 0.079 | 0.238 | +0.007 |
| Implied total x FG rate | 0.109 | 0.006 | 0.068 | 0.051 | -0.006 | -0.002 |
| Implied total x red-zone stall | 0.106 | 0.022 | 0.076 | 0.059 | 0.002 | -0.005 |
| Opponent implied total | -0.099 | -0.003 | -0.011 | -0.044 | -0.000 | -0.001 |
| Wind (mph) | -0.097 | 0.015 | 0.050 | -0.034 | -0.319 | +0.055 |
| Dome / indoors | 0.094 | -0.035 | -0.033 | 0.021 | 0.057 | +0.012 |
| Wind x 50+ attempt share | -0.080 | 0.036 | 0.082 | -0.014 | 0.007 | -0.002 |
| Kicker 50+ attempt share | 0.058 | 0.079 | 0.146 | 0.070 | 0.077 | +0.009 |
| Home field | 0.046 | -0.090 | 0.022 | -0.031 | -0.089 | +0.023 |
| TDs per drive | 0.038 | 0.003 | 0.036 | 0.018 | -0.008 | -0.004 |
| Red-zone trips per drive | 0.037 | -0.027 | -0.011 | 0.000 | -0.006 | -0.000 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.090 vs 0.046 historically). Blended to -0.031, so its weight multiplier is 0.63x.
- **Dome / indoors** is running weaker this season (-0.035 vs 0.094 historically). Blended to 0.021, so its weight multiplier is 0.50x.
- **Wind x 50+ attempt share** is running stronger this season (0.036 vs -0.080 historically). Blended to -0.014, so its weight multiplier is 0.50x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 8.18 | 4.55 | 6.08 | -2.10 | 11 |
| PHI | 7.35 | 3.80 | 5.38 | -1.97 | 10 |
| MIA | 6.89 | 4.60 | 5.62 | -1.27 | 10 |
| IND | 7.11 | 5.09 | 5.94 | -1.17 | 11 |
| SEA | 7.60 | 5.50 | 6.43 | -1.16 | 10 |

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
| DST | 11 | 4.20 | 0.345 | 71% |
| K | 11 | 3.72 | 0.029 | 49% |

