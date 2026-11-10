# Week 9 review (2026)

_Generated 2026-11-10 15:39 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 30 | 3.81 | 3.93 | 0.659 | 0.584 | 80% | 100% | 11.4 | 7.8 |
| K | 30 | 3.75 | 3.96 | 0.352 | 0.028 | 100% | 60% | 14.4 | 10.1 |

## D/ST: what it got wrong

Rank correlation 0.659 against a Vegas-only 0.584 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 11.4 points against a slate average of 7.8, and 80% of them finished startable (Vegas-only: 100%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| LAC | HOU | 6.4 | -1.0 | -7.4 |
| NE | GB | 7.4 | 0.0 | -7.4 |
| ARI | SEA | 6.0 | 1.0 | -5.0 |
| BAL | JAX | 5.5 | 1.0 | -4.5 |
| ATL | CIN | 4.8 | 1.0 | -3.8 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| SF | LV | 7.7 | 18.0 | +10.3 |
| BUF | MIN | 8.0 | 17.0 | +9.0 |
| CHI | TB | 8.2 | 15.0 | +6.8 |
| NYG | PHI | 8.8 | 15.0 | +6.2 |
| TB | CHI | 5.2 | 11.0 | +5.8 |

## Kicker: what it got wrong

Rank correlation 0.352 against a Vegas-only 0.028 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 14.4 points against a slate average of 10.1, and 100% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| A.Borregales | GB | 8.2 | 3.0 | -5.2 |
| R.Fitzgerald | DEN | 8.0 | 3.0 | -5.0 |
| C.Ryland | SEA | 7.1 | 3.0 | -4.1 |
| N.Folk | CIN | 9.0 | 5.0 | -4.0 |
| H.Butker | NYJ | 9.0 | 5.0 | -4.0 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| J.Sanders | KC | 7.1 | 16.0 | +8.9 |
| T.Smack | NE | 7.9 | 16.0 | +8.1 |
| D.Zvada | PHI | 8.9 | 17.0 | +8.1 |
| S.Shrader | DAL | 8.3 | 16.0 | +7.7 |
| K.Fairbairn | LAC | 9.4 | 17.0 | +7.6 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.289 | -0.272 | -0.267 | -0.280 | -0.905 | -0.020 |
| Point spread | 0.265 | 0.226 | 0.304 | 0.245 | 0.893 | +0.045 |
| Opp points per drive | -0.185 | -0.142 | -0.132 | -0.163 | 0.136 | +0.006 |
| Opp sack rate allowed (O-line) | 0.161 | 0.182 | 0.303 | 0.172 | 0.101 | +0.017 |
| Sack opportunity (def x O-line x volume) | 0.160 | 0.129 | 0.164 | 0.144 | 0.047 | -0.017 |
| Opp pressure rate allowed | 0.152 | 0.085 | 0.241 | 0.117 | 0.119 | -0.004 |
| Game total | -0.124 | -0.154 | -0.055 | -0.139 | -0.301 | +0.031 |
| Takeaway opportunity | 0.119 | 0.104 | 0.234 | 0.111 | 0.070 | -0.001 |
| Defense points allowed per drive | -0.098 | -0.111 | -0.103 | -0.105 | -0.008 | -0.007 |
| Opp INT rate | 0.093 | -0.018 | 0.070 | 0.035 | -0.008 | -0.003 |
| Defense INT rate | 0.085 | 0.109 | 0.217 | 0.097 | 0.088 | +0.002 |
| Opp drives per game | 0.084 | 0.112 | 0.059 | 0.099 | 0.121 | -0.002 |

**Biggest divergences from the historical prior:**

- **Opp pass rate** is running weaker this season (-0.120 vs 0.052 historically). Blended to -0.038, so its weight multiplier is 0.40x.
- **Opp fumble-lost rate** is running stronger this season (0.126 vs -0.006 historically). Blended to 0.062, so its weight multiplier is 0.60x.
- **Opp dropbacks per game** is running weaker this season (-0.094 vs 0.021 historically). Blended to -0.039, so its weight multiplier is 0.38x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.134 | 0.025 | 0.017 | 0.077 | 0.372 | +0.009 |
| Team implied total | 0.122 | 0.030 | 0.018 | 0.074 | 0.240 | +0.008 |
| Implied total x FG rate | 0.109 | 0.001 | 0.003 | 0.053 | -0.004 | +0.001 |
| Implied total x red-zone stall | 0.106 | 0.028 | 0.126 | 0.065 | 0.004 | -0.001 |
| Opponent implied total | -0.099 | -0.009 | -0.008 | -0.052 | 0.001 | +0.000 |
| Wind (mph) | -0.097 | -0.033 | -0.083 | -0.064 | -0.359 | +0.001 |
| Dome / indoors | 0.094 | 0.001 | 0.071 | 0.046 | 0.099 | +0.014 |
| Wind x 50+ attempt share | -0.080 | -0.014 | -0.065 | -0.045 | 0.011 | +0.001 |
| Kicker 50+ attempt share | 0.058 | 0.065 | 0.044 | 0.062 | 0.070 | +0.006 |
| Home field | 0.046 | -0.138 | -0.125 | -0.050 | -0.129 | -0.059 |
| TDs per drive | 0.038 | -0.002 | -0.093 | 0.017 | -0.006 | -0.001 |
| Red-zone trips per drive | 0.037 | -0.017 | -0.053 | 0.008 | -0.005 | -0.001 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.138 vs 0.046 historically). Blended to -0.050, so its weight multiplier is 1.00x.
- **Point spread** is running weaker this season (0.025 vs 0.134 historically). Blended to 0.077, so its weight multiplier is 1.15x.
- **Implied total x FG rate** is running weaker this season (0.001 vs 0.109 historically). Blended to 0.053, so its weight multiplier is 0.96x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 8.18 | 4.22 | 6.08 | -2.10 | 9 |
| PHI | 7.35 | 3.56 | 5.34 | -2.01 | 9 |
| MIA | 6.89 | 3.38 | 5.13 | -1.76 | 8 |
| IND | 7.11 | 4.33 | 5.64 | -1.47 | 9 |
| TB | 6.71 | 9.00 | 7.92 | +1.21 | 9 |

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
| DST | 9 | 4.24 | 0.299 | 69% |
| K | 9 | 3.75 | 0.053 | 47% |

