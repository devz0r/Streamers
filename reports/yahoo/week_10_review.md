# Week 10 review (2026)

_Generated 2026-11-17 16:55 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 28 | 4.17 | 4.12 | 0.557 | 0.568 | 60% | 60% | 7.8 | 6.4 |
| K | 27 | 3.38 | 3.18 | -0.259 | -0.106 | 60% | 60% | 9.2 | 8.5 |

## D/ST: what it got wrong

Rank correlation 0.557 against a Vegas-only 0.568 -- essentially level with the Vegas-only ranking. The five recommended units averaged 7.8 points against a slate average of 6.4, and 60% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| PIT | CIN | 5.3 | -3.0 | -8.3 |
| BAL | LAC | 9.5 | 2.0 | -7.5 |
| ATL | KC | 5.5 | -2.0 | -7.5 |
| SF | DAL | 5.7 | 0.0 | -5.7 |
| JAX | TEN | 9.7 | 4.0 | -5.7 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| MIN | GB | 7.0 | 20.0 | +13.0 |
| KC | ATL | 6.7 | 13.0 | +6.3 |
| LA | ARI | 11.1 | 17.0 | +5.9 |
| BUF | NYJ | 8.5 | 13.0 | +4.5 |
| MIA | IND | 7.5 | 12.0 | +4.5 |

## Kicker: what it got wrong

Rank correlation -0.259 against a Vegas-only -0.106 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 9.2 points against a slate average of 8.5, and 60% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| J.Myers | LV | 9.2 | 3.0 | -6.2 |
| D.Zvada | WAS | 8.9 | 3.0 | -5.9 |
| A.Borregales | DET | 8.7 | 4.0 | -4.7 |
| J.Bates | NE | 9.1 | 5.0 | -4.1 |
| H.Butker | ATL | 9.1 | 5.0 | -4.1 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| J.Slye | JAX | 7.3 | 17.0 | +9.7 |
| C.Little | TEN | 9.4 | 19.0 | +9.6 |
| J.Sanders | BUF | 7.6 | 14.0 | +6.4 |
| W.Reichard | GB | 8.5 | 12.0 | +3.5 |
| B.Aubrey | SF | 8.6 | 12.0 | +3.4 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.289 | -0.297 | -0.373 | -0.293 | -0.927 | -0.022 |
| Point spread | 0.265 | 0.240 | 0.401 | 0.252 | 0.904 | +0.010 |
| Opp points per drive | -0.185 | -0.169 | -0.231 | -0.176 | 0.131 | -0.005 |
| Opp sack rate allowed (O-line) | 0.161 | 0.179 | 0.298 | 0.171 | 0.101 | +0.000 |
| Sack opportunity (def x O-line x volume) | 0.160 | 0.125 | 0.151 | 0.141 | 0.040 | -0.007 |
| Opp pressure rate allowed | 0.152 | 0.071 | 0.180 | 0.108 | 0.106 | -0.013 |
| Game total | -0.124 | -0.175 | -0.106 | -0.152 | -0.325 | -0.024 |
| Takeaway opportunity | 0.119 | 0.123 | 0.280 | 0.121 | 0.072 | +0.002 |
| Defense points allowed per drive | -0.098 | -0.127 | -0.167 | -0.114 | -0.009 | -0.001 |
| Opp INT rate | 0.093 | 0.004 | 0.041 | 0.045 | -0.004 | +0.004 |
| Defense INT rate | 0.085 | 0.122 | 0.308 | 0.105 | 0.087 | -0.001 |
| Opp drives per game | 0.084 | 0.104 | 0.018 | 0.095 | 0.117 | -0.004 |

**Biggest divergences from the historical prior:**

- **Opp pass rate** is running weaker this season (-0.107 vs 0.052 historically). Blended to -0.035, so its weight multiplier is 0.37x.
- **Opp fumble-lost rate** is running stronger this season (0.126 vs -0.006 historically). Blended to 0.066, so its weight multiplier is 0.60x.
- **Opp pressure-to-sack rate** is running stronger this season (0.170 vs 0.053 historically). Blended to 0.117, so its weight multiplier is 1.88x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.134 | 0.010 | 0.101 | 0.066 | 0.349 | -0.024 |
| Team implied total | 0.122 | 0.023 | 0.130 | 0.068 | 0.232 | -0.008 |
| Implied total x FG rate | 0.109 | -0.005 | 0.128 | 0.047 | -0.004 | +0.000 |
| Implied total x red-zone stall | 0.106 | 0.024 | 0.198 | 0.061 | 0.007 | +0.003 |
| Opponent implied total | -0.099 | 0.008 | -0.022 | -0.041 | 0.001 | -0.000 |
| Wind (mph) | -0.097 | -0.018 | -0.025 | -0.054 | -0.374 | -0.015 |
| Dome / indoors | 0.094 | -0.027 | 0.019 | 0.028 | 0.045 | -0.053 |
| Wind x 50+ attempt share | -0.080 | -0.001 | -0.008 | -0.037 | 0.009 | -0.002 |
| Kicker 50+ attempt share | 0.058 | 0.067 | 0.051 | 0.063 | 0.069 | -0.001 |
| Home field | 0.046 | -0.124 | -0.029 | -0.047 | -0.112 | +0.017 |
| TDs per drive | 0.038 | -0.018 | -0.059 | 0.008 | -0.004 | +0.002 |
| Red-zone trips per drive | 0.037 | -0.033 | -0.046 | -0.002 | -0.006 | -0.001 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.124 vs 0.046 historically). Blended to -0.047, so its weight multiplier is 0.96x.
- **Point spread** is running weaker this season (0.010 vs 0.134 historically). Blended to 0.066, so its weight multiplier is 0.99x.
- **Dome / indoors** is running weaker this season (-0.027 vs 0.094 historically). Blended to 0.028, so its weight multiplier is 0.59x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 8.18 | 4.30 | 6.02 | -2.16 | 10 |
| PHI | 7.35 | 3.56 | 5.34 | -2.01 | 9 |
| MIA | 6.89 | 4.33 | 5.53 | -1.35 | 9 |
| TB | 6.71 | 9.00 | 7.92 | +1.21 | 9 |
| BAL | 6.54 | 4.40 | 5.35 | -1.19 | 10 |

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
| DST | 10 | 4.23 | 0.324 | 68% |
| K | 10 | 3.72 | 0.022 | 48% |

