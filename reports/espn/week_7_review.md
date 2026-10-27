# Week 7 review (2026)

_Generated 2026-10-27 13:26 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 28 | 6.05 | 6.15 | 0.255 | 0.208 | 60% | 40% | 8.5 | 6.9 |
| K | 28 | 3.44 | 3.49 | 0.284 | 0.371 | 100% | 60% | 11.4 | 7.1 |

## D/ST: what it got wrong

Rank correlation 0.255 against a Vegas-only 0.208 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 8.5 points against a slate average of 6.9, and 60% of them finished startable (Vegas-only: 40%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| CAR | TB | 10.8 | -4.0 | -14.8 |
| PHI | DAL | 7.8 | -5.0 | -12.8 |
| KC | SEA | 10.1 | 0.5 | -9.6 |
| TEN | CLE | 13.6 | 4.0 | -9.6 |
| BAL | CIN | 8.0 | -1.5 | -9.5 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| TB | CAR | 8.9 | 34.5 | +25.6 |
| ATL | SF | 6.3 | 12.0 | +5.7 |
| MIN | IND | 13.5 | 17.5 | +4.0 |
| CIN | BAL | 11.0 | 14.5 | +3.5 |
| IND | MIN | 9.5 | 12.5 | +3.0 |

## Kicker: what it got wrong

Rank correlation 0.284 against a Vegas-only 0.371 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 11.4 points against a slate average of 7.1, and 100% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| R.Fitzgerald | TB | 8.2 | 0.5 | -7.7 |
| E.Pineiro | ATL | 8.8 | 1.5 | -7.3 |
| D.Zvada | HOU | 8.2 | 1.0 | -7.2 |
| D.Carlson | PIT | 8.5 | 3.0 | -5.5 |
| S.Shrader | MIN | 8.1 | 3.0 | -5.1 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| C.Santos | NE | 8.0 | 14.0 | +6.0 |
| B.Aubrey | PHI | 8.5 | 14.0 | +5.5 |
| W.Lutz | ARI | 9.0 | 14.0 | +5.0 |
| C.McLaughlin | CAR | 7.9 | 12.0 | +4.1 |
| W.Reichard | IND | 9.1 | 12.0 | +2.9 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.316 | -0.280 | -0.271 | -0.300 | -1.285 | +0.009 |
| Point spread | 0.273 | 0.165 | 0.133 | 0.223 | 1.082 | +0.002 |
| Opp points per drive | -0.219 | -0.165 | -0.158 | -0.194 | 0.136 | +0.003 |
| Opp sack rate allowed (O-line) | 0.210 | 0.152 | 0.160 | 0.183 | 0.167 | +0.002 |
| Sack opportunity (def x O-line x volume) | 0.204 | 0.155 | 0.157 | 0.181 | 0.151 | +0.001 |
| Opp pressure rate allowed | 0.191 | 0.047 | 0.136 | 0.124 | 0.209 | -0.001 |
| Game total | -0.161 | -0.238 | -0.252 | -0.196 | -0.696 | +0.018 |
| Takeaway opportunity | 0.119 | 0.058 | 0.015 | 0.091 | 0.076 | +0.000 |
| Opp INT rate | 0.109 | -0.009 | -0.015 | 0.054 | 0.011 | -0.016 |
| Defense points allowed per drive | -0.108 | -0.107 | -0.066 | -0.108 | -0.020 | +0.004 |
| Opp turnover-worthy play rate | 0.096 | 0.042 | 0.054 | 0.071 | -0.029 | -0.010 |
| Opp drives per game | 0.092 | 0.140 | 0.057 | 0.114 | 0.149 | +0.002 |

**Biggest divergences from the historical prior:**

- **Opp pressure rate allowed** is running weaker this season (0.047 vs 0.191 historically). Blended to 0.124, so its weight multiplier is 1.31x.
- **Opp pass rate** is running weaker this season (-0.094 vs 0.037 historically). Blended to -0.023, so its weight multiplier is 0.24x.
- **Opp fumble-lost rate** is running stronger this season (0.129 vs 0.000 historically). Blended to 0.059, so its weight multiplier is 0.59x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.128 | 0.010 | 0.091 | 0.073 | 0.369 | +0.015 |
| Team implied total | 0.117 | 0.006 | 0.047 | 0.066 | 0.229 | +0.011 |
| Implied total x FG rate | 0.105 | -0.030 | 0.026 | 0.043 | -0.004 | -0.002 |
| Implied total x red-zone stall | 0.101 | -0.005 | 0.073 | 0.052 | 0.001 | +0.002 |
| Wind (mph) | -0.099 | -0.008 | -0.078 | -0.057 | -0.386 | +0.031 |
| Dome / indoors | 0.095 | -0.038 | -0.007 | 0.033 | 0.080 | +0.015 |
| Opponent implied total | -0.094 | -0.009 | -0.086 | -0.054 | -0.000 | +0.001 |
| Wind x 50+ attempt share | -0.083 | 0.011 | -0.062 | -0.040 | 0.009 | +0.000 |
| Kicker 50+ attempt share | 0.054 | 0.069 | 0.043 | 0.061 | 0.064 | -0.004 |
| Home field | 0.046 | -0.131 | -0.098 | -0.036 | -0.080 | +0.007 |
| TDs per drive | 0.037 | -0.016 | -0.060 | 0.012 | -0.003 | -0.002 |
| Red-zone trips per drive | 0.035 | -0.040 | -0.078 | 0.000 | -0.004 | -0.001 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.131 vs 0.046 historically). Blended to -0.036, so its weight multiplier is 0.82x.
- **Implied total x FG rate** is running weaker this season (-0.030 vs 0.105 historically). Blended to 0.043, so its weight multiplier is 0.81x.
- **Dome / indoors** is running weaker this season (-0.038 vs 0.095 historically). Blended to 0.033, so its weight multiplier is 0.70x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 11.80 | 4.57 | 8.43 | -3.37 | 7 |
| PHI | 10.81 | 4.21 | 7.73 | -3.08 | 7 |
| MIA | 10.29 | 4.50 | 7.81 | -2.48 | 6 |
| LA | 10.34 | 5.64 | 8.15 | -2.19 | 7 |
| IND | 10.26 | 5.93 | 8.24 | -2.02 | 7 |

**Points Allowed**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| MIN | 22.28 | 12.33 | 18.01 | -4.26 | 6 |
| BUF | 20.01 | 28.50 | 23.65 | +3.64 | 6 |
| NO | 21.04 | 27.71 | 24.15 | +3.12 | 7 |
| JAX | 23.10 | 17.33 | 20.63 | -2.47 | 6 |
| ARI | 25.24 | 29.71 | 27.33 | +2.09 | 7 |

**Sack Rate**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| PHI | 2.72 | 1.14 | 1.98 | -0.73 | 7 |
| MIA | 2.59 | 1.00 | 1.91 | -0.68 | 6 |
| LA | 2.63 | 1.43 | 2.07 | -0.56 | 7 |
| DAL | 2.61 | 1.43 | 2.06 | -0.55 | 7 |
| CIN | 2.23 | 3.50 | 2.77 | +0.54 | 6 |

## Season-to-date calibration

| Position | Weeks | MAE | Rank corr | Top-5 hit rate |
|---|---|---|---|---|
| DST | 7 | 5.86 | 0.285 | 49% |
| K | 7 | 3.89 | 0.016 | 40% |

