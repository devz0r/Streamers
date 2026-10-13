# Week 5 review (2026)

_Generated 2026-10-13 15:06 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 30 | 4.51 | 4.49 | 0.370 | 0.403 | 80% | 60% | 12.6 | 7.1 |
| K | 30 | 4.20 | 4.14 | 0.314 | 0.095 | 60% | 40% | 9.4 | 8.9 |

## D/ST: what it got wrong

Rank correlation 0.370 against a Vegas-only 0.403 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 12.6 points against a slate average of 7.1, and 80% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| LV | NE | 6.2 | -1.0 | -7.2 |
| IND | PIT | 6.9 | 0.0 | -6.9 |
| DAL | TB | 8.2 | 2.0 | -6.2 |
| CHI | GB | 6.9 | 1.0 | -5.9 |
| PIT | IND | 8.6 | 3.0 | -5.6 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| GB | CHI | 6.1 | 21.0 | +14.9 |
| SF | SEA | 5.7 | 19.0 | +13.3 |
| HOU | TEN | 9.8 | 17.0 | +7.2 |
| NYJ | CLE | 8.9 | 16.0 | +7.1 |
| DEN | LAC | 8.9 | 14.0 | +5.1 |

## Kicker: what it got wrong

Rank correlation 0.314 against a Vegas-only 0.095 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 9.4 points against a slate average of 8.9, and 60% of them finished startable (Vegas-only: 40%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| D.Carlson | MIN | 8.7 | 1.0 | -7.7 |
| C.Santos | GB | 8.6 | 1.0 | -7.6 |
| C.Ryland | DET | 8.4 | 3.0 | -5.4 |
| B.Aubrey | TB | 10.2 | 5.0 | -5.2 |
| H.Mevis | BUF | 9.1 | 4.0 | -5.1 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| J.Myers | SF | 8.9 | 23.0 | +14.1 |
| S.Shrader | PIT | 8.4 | 21.0 | +12.6 |
| K.Fairbairn | TEN | 9.3 | 17.0 | +7.7 |
| W.Reichard | NO | 9.4 | 16.0 | +6.6 |
| T.Loop | ATL | 8.3 | 14.0 | +5.7 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.289 | -0.278 | -0.271 | -0.285 | -0.907 | -0.013 |
| Point spread | 0.265 | 0.162 | 0.124 | 0.225 | 0.847 | +0.002 |
| Opp points per drive | -0.185 | -0.146 | -0.114 | -0.170 | 0.104 | +0.004 |
| Opp sack rate allowed (O-line) | 0.161 | 0.092 | 0.137 | 0.134 | 0.074 | -0.008 |
| Sack opportunity (def x O-line x volume) | 0.160 | 0.107 | 0.199 | 0.139 | 0.081 | +0.004 |
| Opp pressure rate allowed | 0.152 | -0.030 | 0.001 | 0.081 | 0.091 | +0.009 |
| Game total | -0.124 | -0.243 | -0.278 | -0.170 | -0.371 | -0.021 |
| Takeaway opportunity | 0.119 | 0.007 | 0.019 | 0.076 | 0.051 | -0.004 |
| Defense points allowed per drive | -0.098 | -0.118 | -0.165 | -0.106 | -0.025 | -0.020 |
| Opp INT rate | 0.093 | -0.081 | -0.102 | 0.026 | -0.006 | -0.000 |
| Defense INT rate | 0.085 | 0.034 | 0.057 | 0.065 | 0.064 | -0.023 |
| Opp drives per game | 0.084 | 0.140 | 0.101 | 0.106 | 0.120 | +0.006 |

**Biggest divergences from the historical prior:**

- **Opp pressure rate allowed** is running weaker this season (-0.030 vs 0.152 historically). Blended to 0.081, so its weight multiplier is 1.01x.
- **Opp INT rate** is running weaker this season (-0.081 vs 0.093 historically). Blended to 0.026, so its weight multiplier is 0.32x.
- **Game total** is running weaker this season (-0.243 vs -0.124 historically). Blended to -0.170, so its weight multiplier is 2.36x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.134 | 0.032 | 0.048 | 0.094 | 0.390 | +0.030 |
| Team implied total | 0.122 | 0.040 | 0.046 | 0.090 | 0.253 | +0.003 |
| Implied total x FG rate | 0.109 | -0.002 | 0.019 | 0.066 | -0.002 | +0.001 |
| Implied total x red-zone stall | 0.106 | -0.044 | -0.037 | 0.048 | -0.002 | +0.000 |
| Opponent implied total | -0.099 | -0.010 | -0.029 | -0.064 | -0.001 | +0.000 |
| Wind (mph) | -0.097 | 0.004 | 0.041 | -0.058 | -0.357 | -0.097 |
| Dome / indoors | 0.094 | -0.050 | -0.080 | 0.038 | 0.105 | -0.116 |
| Wind x 50+ attempt share | -0.080 | 0.023 | 0.057 | -0.040 | 0.008 | +0.003 |
| Kicker 50+ attempt share | 0.058 | 0.084 | 0.071 | 0.069 | 0.066 | -0.005 |
| Home field | 0.046 | -0.147 | -0.128 | -0.029 | -0.069 | -0.053 |
| TDs per drive | 0.038 | 0.067 | 0.076 | 0.049 | -0.008 | +0.000 |
| Red-zone trips per drive | 0.037 | 0.012 | 0.017 | 0.027 | -0.009 | +0.001 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.147 vs 0.046 historically). Blended to -0.029, so its weight multiplier is 0.63x.
- **Implied total x red-zone stall** is running weaker this season (-0.044 vs 0.106 historically). Blended to 0.048, so its weight multiplier is 0.90x.
- **Dome / indoors** is running weaker this season (-0.050 vs 0.094 historically). Blended to 0.038, so its weight multiplier is 0.81x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 8.18 | 2.40 | 5.96 | -2.22 | 5 |
| MIA | 6.89 | 2.60 | 5.24 | -1.65 | 5 |
| PHI | 7.35 | 3.60 | 5.91 | -1.44 | 5 |
| IND | 7.11 | 3.80 | 5.83 | -1.27 | 5 |
| TEN | 5.59 | 2.80 | 4.52 | -1.07 | 5 |

**Points Allowed**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| MIN | 22.28 | 12.80 | 18.63 | -3.64 | 5 |
| NO | 21.04 | 30.20 | 24.56 | +3.52 | 5 |
| JAX | 23.10 | 14.00 | 19.60 | -3.50 | 5 |
| BUF | 20.01 | 28.20 | 23.16 | +3.15 | 5 |
| ARI | 25.24 | 32.40 | 28.00 | +2.75 | 5 |

**Sack Rate**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| MIA | 2.59 | 0.80 | 1.90 | -0.69 | 5 |
| DET | 2.29 | 4.00 | 2.95 | +0.66 | 5 |
| DAL | 2.61 | 1.00 | 1.99 | -0.62 | 5 |
| PHI | 2.72 | 1.20 | 2.13 | -0.58 | 5 |
| TB | 2.57 | 1.20 | 2.04 | -0.53 | 5 |

## Season-to-date calibration

| Position | Weeks | MAE | Rank corr | Top-5 hit rate |
|---|---|---|---|---|
| DST | 5 | 4.19 | 0.263 | 60% |
| K | 5 | 3.78 | -0.006 | 36% |

