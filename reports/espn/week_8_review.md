# Week 8 review (2026)

_Generated 2026-11-03 15:15 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 28 | 5.84 | 6.01 | 0.320 | 0.263 | 80% | 80% | 19.0 | 11.4 |
| K | 28 | 3.72 | 3.70 | 0.049 | 0.095 | 20% | 40% | 6.4 | 7.4 |

## D/ST: what it got wrong

Rank correlation 0.320 against a Vegas-only 0.263 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 19.0 points against a slate average of 11.4, and 80% of them finished startable (Vegas-only: 80%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| NYJ | LV | 9.6 | 0.5 | -9.1 |
| SEA | CHI | 10.5 | 2.5 | -8.0 |
| CIN | TEN | 13.6 | 6.0 | -7.6 |
| DEN | KC | 11.4 | 4.0 | -7.4 |
| CLE | PIT | 11.2 | 4.0 | -7.2 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| JAX | IND | 13.1 | 28.0 | +14.9 |
| PIT | CLE | 14.4 | 29.0 | +14.6 |
| BUF | BAL | 8.0 | 22.5 | +14.5 |
| WAS | PHI | 12.6 | 22.0 | +9.4 |
| GB | CAR | 10.2 | 19.5 | +9.3 |

## Kicker: what it got wrong

Rank correlation 0.049 against a Vegas-only 0.095 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 6.4 points against a slate average of 7.4, and 20% of them finished startable (Vegas-only: 40%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| N.Folk | TB | 7.6 | -1.0 | -8.6 |
| W.Reichard | DET | 8.6 | 0.0 | -8.6 |
| J.Myers | CHI | 8.5 | 1.5 | -7.0 |
| S.Shrader | JAX | 6.7 | 1.0 | -5.7 |
| T.Smack | CAR | 8.2 | 2.5 | -5.7 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| T.Bass | BAL | 8.2 | 16.0 | +7.8 |
| D.Stevens | PHI | 8.4 | 15.0 | +6.6 |
| J.Sanders | LV | 8.6 | 15.0 | +6.4 |
| C.McLaughlin | ATL | 8.0 | 12.0 | +4.0 |
| B.Aubrey | ARI | 8.5 | 12.0 | +3.5 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.316 | -0.288 | -0.250 | -0.302 | -1.282 | +0.002 |
| Point spread | 0.273 | 0.198 | 0.192 | 0.236 | 1.103 | +0.021 |
| Opp points per drive | -0.219 | -0.172 | -0.139 | -0.196 | 0.148 | +0.013 |
| Opp sack rate allowed (O-line) | 0.210 | 0.180 | 0.183 | 0.195 | 0.186 | +0.019 |
| Sack opportunity (def x O-line x volume) | 0.204 | 0.146 | 0.118 | 0.175 | 0.131 | -0.020 |
| Opp pressure rate allowed | 0.191 | 0.074 | 0.170 | 0.133 | 0.223 | +0.014 |
| Game total | -0.161 | -0.212 | -0.160 | -0.186 | -0.661 | +0.036 |
| Takeaway opportunity | 0.119 | 0.100 | 0.070 | 0.109 | 0.096 | +0.020 |
| Opp INT rate | 0.109 | 0.013 | 0.064 | 0.062 | 0.011 | -0.000 |
| Defense points allowed per drive | -0.108 | -0.086 | -0.025 | -0.097 | 0.002 | +0.022 |
| Opp turnover-worthy play rate | 0.096 | 0.067 | 0.115 | 0.082 | -0.031 | -0.002 |
| Opp drives per game | 0.092 | 0.133 | 0.090 | 0.112 | 0.141 | -0.007 |

**Biggest divergences from the historical prior:**

- **Opp fumble-lost rate** is running stronger this season (0.143 vs 0.000 historically). Blended to 0.070, so its weight multiplier is 0.60x.
- **Opp pass rate** is running weaker this season (-0.103 vs 0.037 historically). Blended to -0.032, so its weight multiplier is 0.32x.
- **Opp dropbacks per game** is running weaker this season (-0.114 vs 0.010 historically). Blended to -0.051, so its weight multiplier is 0.51x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.128 | 0.022 | 0.057 | 0.076 | 0.360 | -0.009 |
| Team implied total | 0.117 | 0.025 | 0.013 | 0.072 | 0.229 | +0.000 |
| Implied total x FG rate | 0.105 | -0.020 | -0.008 | 0.043 | -0.006 | -0.002 |
| Implied total x red-zone stall | 0.101 | 0.020 | 0.080 | 0.061 | 0.004 | +0.002 |
| Wind (mph) | -0.099 | -0.018 | -0.048 | -0.059 | -0.385 | +0.000 |
| Dome / indoors | 0.095 | -0.030 | -0.044 | 0.034 | 0.074 | -0.007 |
| Opponent implied total | -0.094 | -0.008 | -0.071 | -0.052 | 0.000 | +0.000 |
| Wind x 50+ attempt share | -0.083 | 0.000 | -0.035 | -0.042 | 0.009 | -0.000 |
| Kicker 50+ attempt share | 0.054 | 0.068 | 0.004 | 0.061 | 0.062 | -0.001 |
| Home field | 0.046 | -0.084 | -0.066 | -0.018 | -0.051 | +0.029 |
| TDs per drive | 0.037 | -0.013 | -0.090 | 0.013 | -0.004 | -0.001 |
| Red-zone trips per drive | 0.035 | -0.029 | -0.068 | 0.004 | -0.004 | -0.000 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.084 vs 0.046 historically). Blended to -0.018, so its weight multiplier is 0.45x.
- **Implied total x FG rate** is running weaker this season (-0.020 vs 0.105 historically). Blended to 0.043, so its weight multiplier is 0.82x.
- **Dome / indoors** is running weaker this season (-0.030 vs 0.095 historically). Blended to 0.034, so its weight multiplier is 0.71x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| PHI | 10.81 | 4.56 | 7.69 | -3.12 | 8 |
| DAL | 11.80 | 6.12 | 8.96 | -2.84 | 8 |
| MIA | 10.29 | 5.21 | 7.92 | -2.37 | 7 |
| LA | 10.34 | 6.19 | 8.26 | -2.08 | 8 |
| IND | 10.26 | 6.38 | 8.32 | -1.94 | 8 |

**Points Allowed**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| MIN | 22.28 | 13.29 | 18.08 | -4.20 | 7 |
| JAX | 23.10 | 15.86 | 19.72 | -3.38 | 7 |
| NO | 21.04 | 27.71 | 24.15 | +3.12 | 7 |
| BUF | 20.01 | 25.86 | 22.74 | +2.73 | 7 |
| BAL | 20.42 | 25.38 | 22.90 | +2.48 | 8 |

**Sack Rate**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| PHI | 2.72 | 1.25 | 1.98 | -0.73 | 8 |
| MIA | 2.59 | 1.29 | 1.98 | -0.61 | 7 |
| KC | 2.55 | 1.29 | 1.96 | -0.59 | 7 |
| LA | 2.63 | 1.50 | 2.07 | -0.57 | 8 |
| DET | 2.29 | 3.43 | 2.82 | +0.53 | 7 |

## Season-to-date calibration

| Position | Weeks | MAE | Rank corr | Top-5 hit rate |
|---|---|---|---|---|
| DST | 8 | 5.86 | 0.289 | 52% |
| K | 8 | 3.87 | 0.020 | 38% |

