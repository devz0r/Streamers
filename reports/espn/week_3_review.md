# Week 3 review (2026)

_Generated 2026-09-29 14:07 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 32 | 5.89 | 6.05 | 0.142 | 0.062 | 0% | 0% | 8.1 | 8.6 |
| K | 32 | 4.33 | 4.26 | -0.299 | -0.236 | 0% | 20% | 4.5 | 8.5 |

## D/ST: what it got wrong

Rank correlation 0.142 against a Vegas-only 0.062 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 8.1 points against a slate average of 8.6, and 0% of them finished startable (Vegas-only: 0%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| GB | ATL | 11.4 | -4.5 | -15.9 |
| PHI | CHI | 11.7 | -1.0 | -12.7 |
| SF | ARI | 10.9 | 1.0 | -9.9 |
| CIN | PIT | 11.6 | 2.5 | -9.1 |
| DAL | BAL | 6.2 | -2.0 | -8.2 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| MIN | TB | 11.4 | 29.0 | +17.6 |
| LAC | BUF | 5.9 | 20.0 | +14.1 |
| JAX | NE | 11.3 | 19.5 | +8.2 |
| CHI | PHI | 9.0 | 17.0 | +8.0 |
| DEN | LA | 10.3 | 18.0 | +7.7 |

## Kicker: what it got wrong

Rank correlation -0.299 against a Vegas-only -0.236 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 4.5 points against a slate average of 8.5, and 0% of them finished startable (Vegas-only: 20%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| J.Slye | NYG | 7.6 | 0.0 | -7.6 |
| J.Elliott | CHI | 8.5 | 1.0 | -7.5 |
| T.Smack | ATL | 8.4 | 1.0 | -7.4 |
| W.Lutz | LA | 7.7 | 1.0 | -6.7 |
| D.Carlson | LV | 8.8 | 2.5 | -6.3 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| S.Shrader | HOU | 8.0 | 17.0 | +9.0 |
| H.Mevis | DEN | 7.4 | 16.0 | +8.6 |
| W.Reichard | TB | 7.9 | 16.0 | +8.1 |
| R.Fitzgerald | CLE | 7.8 | 15.0 | +7.2 |
| C.Ryland | SF | 7.5 | 14.0 | +6.5 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.316 | -0.293 | -0.293 | -0.310 | -1.281 | +0.032 |
| Point spread | 0.273 | 0.202 | 0.202 | 0.253 | 1.117 | -0.048 |
| Opp points per drive | -0.219 | -0.174 | -0.174 | -0.206 | 0.113 | +0.013 |
| Opp sack rate allowed (O-line) | 0.210 | 0.138 | 0.138 | 0.190 | 0.169 | +0.008 |
| Sack opportunity (def x O-line x volume) | 0.204 | 0.147 | 0.147 | 0.188 | 0.148 | +0.036 |
| Opp pressure rate allowed | 0.191 | -0.043 | -0.043 | 0.126 | 0.199 | -0.030 |
| Game total | -0.161 | -0.223 | -0.223 | -0.178 | -0.635 | -0.014 |
| Takeaway opportunity | 0.119 | 0.106 | 0.106 | 0.115 | 0.083 | +0.020 |
| Opp INT rate | 0.109 | -0.005 | -0.005 | 0.077 | 0.030 | -0.014 |
| Defense points allowed per drive | -0.108 | -0.156 | -0.156 | -0.121 | -0.031 | -0.017 |
| Opp turnover-worthy play rate | 0.096 | 0.027 | 0.027 | 0.077 | -0.044 | +0.004 |
| Opp drives per game | 0.092 | 0.247 | 0.247 | 0.135 | 0.134 | +0.012 |

**Biggest divergences from the historical prior:**

- **Opp pressure rate allowed** is running weaker this season (-0.043 vs 0.191 historically). Blended to 0.126, so its weight multiplier is 1.32x.
- **Opp pressure-to-sack rate** is running stronger this season (0.258 vs 0.080 historically). Blended to 0.129, so its weight multiplier is 1.71x.
- **Opp drives per game** is running stronger this season (0.247 vs 0.092 historically). Blended to 0.135, so its weight multiplier is 1.76x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.128 | -0.103 | -0.103 | 0.064 | 0.348 | -0.026 |
| Team implied total | 0.117 | -0.060 | -0.060 | 0.068 | 0.231 | -0.017 |
| Implied total x FG rate | 0.105 | -0.120 | -0.120 | 0.043 | -0.002 | +0.001 |
| Implied total x red-zone stall | 0.101 | -0.114 | -0.114 | 0.042 | -0.002 | -0.000 |
| Dome / indoors | 0.095 | -0.083 | -0.083 | 0.046 | 0.222 | -0.012 |
| Opponent implied total | -0.094 | 0.109 | 0.109 | -0.037 | -0.001 | -0.000 |
| Wind (mph) | -0.092 | 0.095 | 0.095 | -0.040 | -0.264 | +0.037 |
| Wind x 50+ attempt share | -0.079 | 0.115 | 0.115 | -0.025 | 0.004 | -0.001 |
| Kicker 50+ attempt share | 0.054 | 0.117 | 0.117 | 0.072 | 0.063 | +0.003 |
| Home field | 0.046 | -0.179 | -0.179 | -0.016 | -0.010 | -0.011 |
| TDs per drive | 0.037 | 0.049 | 0.049 | 0.040 | -0.003 | +0.003 |
| Red-zone trips per drive | 0.035 | 0.018 | 0.018 | 0.031 | -0.008 | -0.000 |

**Biggest divergences from the historical prior:**

- **Point spread** is running weaker this season (-0.103 vs 0.128 historically). Blended to 0.064, so its weight multiplier is 1.00x.
- **Implied total x FG rate** is running weaker this season (-0.120 vs 0.105 historically). Blended to 0.043, so its weight multiplier is 0.81x.
- **Home field** is running weaker this season (-0.179 vs 0.046 historically). Blended to -0.016, so its weight multiplier is 0.44x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 11.80 | 3.00 | 9.40 | -2.40 | 3 |
| MIN | 11.04 | 19.00 | 13.21 | +2.17 | 3 |
| MIA | 10.29 | 2.50 | 8.17 | -2.13 | 3 |
| PHI | 10.81 | 3.17 | 8.73 | -2.08 | 3 |
| LV | 8.21 | 15.67 | 10.24 | +2.03 | 3 |

**Points Allowed**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| JAX | 23.10 | 12.00 | 20.07 | -3.03 | 3 |
| GB | 21.24 | 30.33 | 23.72 | +2.48 | 3 |
| MIN | 22.28 | 13.67 | 19.93 | -2.35 | 3 |
| DET | 24.37 | 31.67 | 26.36 | +1.99 | 3 |
| NO | 21.04 | 27.67 | 22.84 | +1.81 | 3 |

**Sack Rate**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| MIA | 2.59 | 0.00 | 1.88 | -0.71 | 3 |
| MIN | 2.70 | 4.67 | 3.24 | +0.54 | 3 |
| DAL | 2.61 | 0.67 | 2.08 | -0.53 | 3 |
| NYG | 2.23 | 0.33 | 1.71 | -0.52 | 3 |
| PHI | 2.72 | 1.00 | 2.25 | -0.47 | 3 |

## Season-to-date calibration

| Position | Weeks | MAE | Rank corr | Top-5 hit rate |
|---|---|---|---|---|
| DST | 3 | 6.01 | 0.235 | 27% |
| K | 3 | 3.82 | -0.210 | 27% |

