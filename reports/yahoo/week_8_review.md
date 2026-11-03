# Week 8 review (2026)

_Generated 2026-11-03 15:16 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 28 | 4.49 | 4.64 | 0.373 | 0.294 | 80% | 80% | 14.0 | 7.6 |
| K | 28 | 3.55 | 3.54 | 0.020 | 0.076 | 0% | 40% | 5.0 | 7.9 |

## D/ST: what it got wrong

Rank correlation 0.373 against a Vegas-only 0.294 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 14.0 points against a slate average of 7.6, and 80% of them finished startable (Vegas-only: 80%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| NYJ | LV | 6.7 | -2.0 | -8.7 |
| KC | DEN | 6.7 | 0.0 | -6.7 |
| SEA | CHI | 7.5 | 2.0 | -5.5 |
| DEN | KC | 7.9 | 3.0 | -4.9 |
| ATL | TB | 6.8 | 2.0 | -4.8 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| TB | ATL | 7.9 | 21.0 | +13.1 |
| JAX | IND | 9.5 | 22.0 | +12.5 |
| PIT | CLE | 9.7 | 21.0 | +11.3 |
| BUF | BAL | 5.8 | 15.0 | +9.2 |
| GB | CAR | 7.3 | 13.0 | +5.7 |

## Kicker: what it got wrong

Rank correlation 0.020 against a Vegas-only 0.076 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 5.0 points against a slate average of 7.9, and 0% of them finished startable (Vegas-only: 40%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| N.Folk | TB | 7.9 | 0.0 | -7.9 |
| W.Reichard | DET | 8.9 | 1.0 | -7.9 |
| S.Shrader | JAX | 7.1 | 1.0 | -6.1 |
| J.Myers | CHI | 8.9 | 3.0 | -5.9 |
| T.Smack | CAR | 8.6 | 3.0 | -5.6 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| T.Bass | BAL | 8.5 | 16.0 | +7.5 |
| D.Stevens | PHI | 8.7 | 16.0 | +7.3 |
| J.Sanders | LV | 8.9 | 15.0 | +6.1 |
| B.Aubrey | ARI | 8.8 | 13.0 | +4.2 |
| C.McLaughlin | ATL | 8.4 | 12.0 | +3.6 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.289 | -0.239 | -0.223 | -0.264 | -0.885 | -0.004 |
| Point spread | 0.265 | 0.172 | 0.191 | 0.219 | 0.848 | +0.018 |
| Opp points per drive | -0.185 | -0.110 | -0.089 | -0.148 | 0.131 | +0.009 |
| Opp sack rate allowed (O-line) | 0.161 | 0.130 | 0.190 | 0.145 | 0.084 | +0.017 |
| Sack opportunity (def x O-line x volume) | 0.160 | 0.106 | 0.127 | 0.133 | 0.064 | -0.017 |
| Opp pressure rate allowed | 0.152 | 0.060 | 0.215 | 0.107 | 0.123 | +0.012 |
| Game total | -0.124 | -0.167 | -0.123 | -0.145 | -0.332 | +0.018 |
| Takeaway opportunity | 0.119 | 0.083 | 0.105 | 0.102 | 0.071 | +0.018 |
| Defense points allowed per drive | -0.098 | -0.085 | -0.064 | -0.091 | -0.001 | +0.018 |
| Opp INT rate | 0.093 | -0.024 | 0.079 | 0.036 | -0.005 | -0.001 |
| Defense INT rate | 0.085 | 0.090 | 0.069 | 0.088 | 0.085 | +0.013 |
| Opp drives per game | 0.084 | 0.111 | 0.065 | 0.097 | 0.123 | -0.002 |

**Biggest divergences from the historical prior:**

- **Opp pass rate** is running weaker this season (-0.108 vs 0.052 historically). Blended to -0.027, so its weight multiplier is 0.30x.
- **Opp fumble-lost rate** is running stronger this season (0.124 vs -0.006 historically). Blended to 0.058, so its weight multiplier is 0.58x.
- **Opp INT rate** is running weaker this season (-0.024 vs 0.093 historically). Blended to 0.036, so its weight multiplier is 0.44x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.134 | 0.014 | 0.050 | 0.075 | 0.364 | -0.009 |
| Team implied total | 0.122 | 0.021 | 0.006 | 0.072 | 0.232 | +0.000 |
| Implied total x FG rate | 0.109 | -0.019 | -0.010 | 0.046 | -0.005 | -0.002 |
| Implied total x red-zone stall | 0.106 | 0.020 | 0.084 | 0.064 | 0.005 | +0.002 |
| Opponent implied total | -0.099 | -0.001 | -0.068 | -0.051 | 0.000 | +0.000 |
| Wind (mph) | -0.097 | -0.017 | -0.047 | -0.058 | -0.359 | +0.001 |
| Dome / indoors | 0.094 | -0.025 | -0.035 | 0.036 | 0.085 | -0.005 |
| Wind x 50+ attempt share | -0.080 | -0.002 | -0.037 | -0.041 | 0.009 | +0.000 |
| Kicker 50+ attempt share | 0.058 | 0.058 | -0.009 | 0.058 | 0.063 | -0.001 |
| Home field | 0.046 | -0.095 | -0.080 | -0.024 | -0.070 | +0.029 |
| TDs per drive | 0.038 | -0.009 | -0.086 | 0.015 | -0.005 | -0.001 |
| Red-zone trips per drive | 0.037 | -0.021 | -0.055 | 0.008 | -0.004 | -0.000 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.095 vs 0.046 historically). Blended to -0.024, so its weight multiplier is 0.55x.
- **Implied total x FG rate** is running weaker this season (-0.019 vs 0.109 historically). Blended to 0.046, so its weight multiplier is 0.84x.
- **Point spread** is running weaker this season (0.014 vs 0.134 historically). Blended to 0.075, so its weight multiplier is 1.12x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| PHI | 7.35 | 3.00 | 5.17 | -2.17 | 8 |
| DAL | 8.18 | 4.00 | 6.09 | -2.09 | 8 |
| MIA | 6.89 | 3.14 | 5.14 | -1.75 | 7 |
| IND | 7.11 | 4.00 | 5.55 | -1.55 | 8 |
| LA | 6.83 | 4.50 | 5.67 | -1.17 | 8 |

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
| DST | 8 | 4.29 | 0.254 | 68% |
| K | 8 | 3.75 | 0.016 | 40% |

