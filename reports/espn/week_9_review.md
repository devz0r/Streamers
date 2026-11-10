# Week 9 review (2026)

_Generated 2026-11-10 15:38 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 30 | 4.92 | 5.10 | 0.633 | 0.582 | 80% | 100% | 14.9 | 10.3 |
| K | 30 | 3.81 | 4.00 | 0.347 | 0.015 | 100% | 40% | 14.2 | 9.8 |

## D/ST: what it got wrong

Rank correlation 0.633 against a Vegas-only 0.582 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 14.9 points against a slate average of 10.3, and 80% of them finished startable (Vegas-only: 100%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| LAC | HOU | 7.9 | -1.5 | -9.4 |
| NE | GB | 9.1 | 0.0 | -9.1 |
| BAL | JAX | 7.0 | 0.0 | -7.0 |
| NO | CLE | 11.9 | 7.0 | -4.9 |
| ARI | SEA | 7.9 | 3.0 | -4.9 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| SF | LV | 9.4 | 20.5 | +11.1 |
| HOU | LAC | 13.1 | 21.5 | +8.4 |
| GB | NE | 8.3 | 16.5 | +8.2 |
| CHI | TB | 10.3 | 18.0 | +7.7 |
| BUF | MIN | 10.3 | 18.0 | +7.7 |

## Kicker: what it got wrong

Rank correlation 0.347 against a Vegas-only 0.015 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 14.2 points against a slate average of 9.8, and 100% of them finished startable (Vegas-only: 40%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| R.Fitzgerald | DEN | 7.7 | 2.0 | -5.7 |
| A.Borregales | GB | 7.9 | 3.0 | -4.9 |
| N.Folk | CIN | 8.7 | 4.0 | -4.7 |
| H.Butker | NYJ | 8.6 | 4.0 | -4.6 |
| C.Ryland | SEA | 6.7 | 3.0 | -3.7 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| T.Smack | NE | 7.5 | 16.0 | +8.5 |
| D.Zvada | PHI | 8.6 | 17.0 | +8.4 |
| J.Sanders | KC | 6.8 | 15.0 | +8.2 |
| S.Shrader | DAL | 7.9 | 16.0 | +8.1 |
| K.Fairbairn | LAC | 9.0 | 17.0 | +8.0 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.322 | -0.315 | -0.286 | -0.319 | -1.224 | +0.058 |
| Point spread | 0.278 | 0.241 | 0.293 | 0.259 | 1.083 | -0.019 |
| Opp points per drive | -0.221 | -0.194 | -0.160 | -0.207 | 0.144 | -0.004 |
| Opp sack rate allowed (O-line) | 0.204 | 0.224 | 0.309 | 0.214 | 0.172 | -0.013 |
| Sack opportunity (def x O-line x volume) | 0.199 | 0.164 | 0.162 | 0.181 | 0.091 | -0.040 |
| Opp pressure rate allowed | 0.186 | 0.091 | 0.187 | 0.136 | 0.188 | -0.035 |
| Game total | -0.163 | -0.203 | -0.098 | -0.184 | -0.587 | +0.074 |
| Takeaway opportunity | 0.120 | 0.126 | 0.224 | 0.123 | 0.094 | -0.002 |
| Opp INT rate | 0.109 | 0.020 | 0.088 | 0.063 | 0.006 | -0.005 |
| Defense points allowed per drive | -0.108 | -0.119 | -0.071 | -0.114 | -0.007 | -0.009 |
| Opp turnover-worthy play rate | 0.096 | 0.077 | 0.138 | 0.086 | -0.032 | -0.001 |
| Opp drives per game | 0.093 | 0.127 | 0.081 | 0.111 | 0.123 | -0.018 |

**Biggest divergences from the historical prior:**

- **Opp pass rate** is running weaker this season (-0.116 vs 0.042 historically). Blended to -0.040, so its weight multiplier is 0.38x.
- **Opp fumble-lost rate** is running stronger this season (0.153 vs 0.000 historically). Blended to 0.079, so its weight multiplier is 0.60x.
- **Opp pressure-to-sack rate** is running stronger this season (0.206 vs 0.077 historically). Blended to 0.144, so its weight multiplier is 2.07x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.128 | 0.032 | 0.020 | 0.078 | 0.370 | +0.010 |
| Team implied total | 0.117 | 0.032 | 0.024 | 0.073 | 0.237 | +0.008 |
| Implied total x FG rate | 0.105 | -0.003 | -0.003 | 0.049 | -0.006 | +0.001 |
| Implied total x red-zone stall | 0.101 | 0.025 | 0.120 | 0.061 | 0.002 | -0.002 |
| Wind (mph) | -0.099 | -0.033 | -0.078 | -0.065 | -0.386 | -0.000 |
| Dome / indoors | 0.095 | -0.004 | 0.058 | 0.043 | 0.087 | +0.013 |
| Opponent implied total | -0.094 | -0.017 | -0.007 | -0.053 | 0.001 | +0.000 |
| Wind x 50+ attempt share | -0.083 | -0.012 | -0.057 | -0.046 | 0.010 | +0.002 |
| Kicker 50+ attempt share | 0.054 | 0.071 | 0.047 | 0.063 | 0.068 | +0.006 |
| Home field | 0.046 | -0.128 | -0.116 | -0.045 | -0.114 | -0.063 |
| TDs per drive | 0.037 | -0.004 | -0.088 | 0.016 | -0.005 | -0.001 |
| Red-zone trips per drive | 0.035 | -0.023 | -0.055 | 0.005 | -0.005 | -0.001 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.128 vs 0.046 historically). Blended to -0.045, so its weight multiplier is 0.97x.
- **Implied total x FG rate** is running weaker this season (-0.003 vs 0.105 historically). Blended to 0.049, so its weight multiplier is 0.93x.
- **Dome / indoors** is running weaker this season (-0.004 vs 0.095 historically). Blended to 0.043, so its weight multiplier is 0.91x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 10.49 | 5.56 | 7.88 | -2.61 | 9 |
| PHI | 9.45 | 5.06 | 7.12 | -2.33 | 9 |
| MIA | 9.00 | 5.06 | 7.03 | -1.97 | 8 |
| IND | 9.07 | 5.89 | 7.39 | -1.68 | 9 |
| LA | 9.03 | 6.17 | 7.51 | -1.51 | 9 |

**Points Allowed**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| JAX | 23.10 | 15.12 | 19.11 | -3.99 | 8 |
| MIN | 22.28 | 14.50 | 18.39 | -3.89 | 8 |
| BAL | 20.42 | 26.22 | 23.49 | +3.07 | 9 |
| NO | 21.04 | 27.00 | 24.02 | +2.98 | 8 |
| LV | 24.19 | 29.78 | 27.15 | +2.96 | 9 |

**Sack Rate**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| MIA | 2.59 | 1.50 | 2.04 | -0.54 | 8 |
| DET | 2.29 | 3.38 | 2.83 | +0.54 | 8 |
| HOU | 2.59 | 3.62 | 3.11 | +0.52 | 8 |
| PHI | 2.72 | 1.78 | 2.22 | -0.50 | 9 |
| MIN | 2.70 | 3.62 | 3.16 | +0.46 | 8 |

## Season-to-date calibration

| Position | Weeks | MAE | Rank corr | Top-5 hit rate |
|---|---|---|---|---|
| DST | 9 | 5.47 | 0.332 | 56% |
| K | 9 | 3.86 | 0.057 | 44% |

