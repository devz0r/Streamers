# Week 2 review (2026)

_Generated 2026-09-22 15:25 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 32 | 5.76 | 5.79 | 0.350 | 0.340 | 20% | 40% | 9.2 | 10.1 |
| K | 32 | 3.41 | 3.30 | -0.157 | -0.103 | 20% | 20% | 6.0 | 8.0 |

## D/ST: what it got wrong

Rank correlation 0.350 against a Vegas-only 0.340 -- slightly ahead of ranking on the Vegas number alone. The five recommended units averaged 9.2 points against a slate average of 10.1, and 20% of them finished startable (Vegas-only: 40%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| KC | IND | 11.5 | 2.5 | -9.0 |
| TB | CLE | 15.4 | 7.0 | -8.4 |
| MIA | SF | 4.3 | -4.0 | -8.3 |
| IND | KC | 8.0 | 0.0 | -8.0 |
| DAL | WAS | 8.8 | 1.0 | -7.8 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| CAR | ATL | 11.0 | 31.0 | +20.0 |
| MIN | CHI | 7.5 | 22.5 | +15.0 |
| NE | PIT | 12.2 | 26.5 | +14.3 |
| LV | LAC | 9.2 | 18.5 | +9.3 |
| NO | BAL | 7.3 | 14.0 | +6.7 |

## Kicker: what it got wrong

Rank correlation -0.157 against a Vegas-only -0.103 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 6.0 points against a slate average of 8.0, and 20% of them finished startable (Vegas-only: 20%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| C.Dicker | LV | 9.5 | 1.0 | -8.5 |
| C.Ryland | SEA | 7.9 | 1.0 | -6.9 |
| H.Mevis | NYG | 9.5 | 3.0 | -6.5 |
| N.Folk | CAR | 8.2 | 3.0 | -5.2 |
| E.Pineiro | MIA | 9.7 | 5.0 | -4.7 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| B.Aubrey | WAS | 8.8 | 17.0 | +8.2 |
| H.Butker | IND | 8.8 | 16.0 | +7.2 |
| C.McLaughlin | CLE | 9.2 | 16.0 | +6.8 |
| S.Shrader | KC | 7.8 | 14.0 | +6.2 |
| D.Carlson | BAL | 7.5 | 13.0 | +5.5 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.316 | -0.366 | -0.366 | -0.326 | -1.312 | -0.012 |
| Point spread | 0.273 | 0.280 | 0.280 | 0.274 | 1.165 | -0.013 |
| Opp points per drive | -0.219 | -0.264 | -0.264 | -0.228 | 0.101 | +0.011 |
| Opp sack rate allowed (O-line) | 0.210 | 0.110 | 0.110 | 0.190 | 0.161 | -0.002 |
| Sack opportunity (def x O-line x volume) | 0.204 | 0.009 | 0.009 | 0.164 | 0.112 | +0.012 |
| Opp pressure rate allowed | 0.191 | -0.010 | -0.010 | 0.150 | 0.229 | -0.010 |
| Game total | -0.161 | -0.240 | -0.240 | -0.177 | -0.621 | -0.038 |
| Takeaway opportunity | 0.119 | -0.029 | -0.029 | 0.088 | 0.063 | -0.014 |
| Opp INT rate | 0.109 | -0.001 | -0.001 | 0.086 | 0.043 | -0.025 |
| Defense points allowed per drive | -0.108 | -0.078 | -0.078 | -0.102 | -0.014 | -0.006 |
| Opp turnover-worthy play rate | 0.096 | 0.006 | 0.006 | 0.078 | -0.048 | +0.004 |
| Opp drives per game | 0.092 | 0.296 | 0.296 | 0.133 | 0.123 | +0.014 |

**Biggest divergences from the historical prior:**

- **Opp drives per game** is running stronger this season (0.296 vs 0.092 historically). Blended to 0.133, so its weight multiplier is 1.82x.
- **Opp pressure rate allowed** is running weaker this season (-0.010 vs 0.191 historically). Blended to 0.150, so its weight multiplier is 1.57x.
- **Sack opportunity (def x O-line x volume)** is running weaker this season (0.009 vs 0.204 historically). Blended to 0.164, so its weight multiplier is 1.61x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.128 | -0.046 | -0.046 | 0.093 | 0.374 | -0.020 |
| Team implied total | 0.117 | 0.017 | 0.017 | 0.097 | 0.249 | -0.004 |
| Implied total x FG rate | 0.105 | -0.069 | -0.069 | 0.070 | -0.003 | -0.002 |
| Implied total x red-zone stall | 0.101 | -0.028 | -0.028 | 0.075 | -0.001 | +0.003 |
| Dome / indoors | 0.095 | -0.044 | -0.044 | 0.067 | 0.233 | -0.039 |
| Opponent implied total | -0.094 | 0.094 | 0.094 | -0.055 | -0.000 | -0.000 |
| Wind (mph) | -0.092 | 0.012 | 0.012 | -0.071 | -0.301 | +0.017 |
| Wind x 50+ attempt share | -0.079 | 0.038 | 0.038 | -0.055 | 0.005 | -0.001 |
| Kicker 50+ attempt share | 0.054 | 0.174 | 0.174 | 0.079 | 0.060 | +0.004 |
| Home field | 0.046 | -0.206 | -0.206 | -0.006 | 0.000 | -0.039 |
| TDs per drive | 0.037 | -0.004 | -0.004 | 0.029 | -0.007 | -0.000 |
| Red-zone trips per drive | 0.035 | -0.050 | -0.050 | 0.018 | -0.007 | +0.004 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.206 vs 0.046 historically). Blended to -0.006, so its weight multiplier is 0.34x.
- **Opp drives per game allowed** is running stronger this season (0.241 vs -0.010 historically). Blended to 0.042, so its weight multiplier is 0.42x.
- **Kicker PAT accuracy** is running weaker this season (-0.210 vs 0.022 historically). Blended to -0.025, so its weight multiplier is 0.72x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| LV | 8.21 | 16.75 | 9.92 | +1.71 | 2 |
| MIA | 10.29 | 2.50 | 8.73 | -1.56 | 2 |
| CIN | 9.58 | 17.00 | 11.07 | +1.48 | 2 |
| CAR | 7.70 | 15.00 | 9.16 | +1.46 | 2 |
| DAL | 11.80 | 4.50 | 10.34 | -1.46 | 2 |

**Points Allowed**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| SEA | 21.53 | 8.50 | 18.92 | -2.61 | 2 |
| IND | 24.07 | 37.00 | 26.66 | +2.59 | 2 |
| NE | 20.61 | 8.00 | 18.09 | -2.52 | 2 |
| DET | 24.37 | 35.50 | 26.60 | +2.23 | 2 |
| BUF | 20.01 | 31.00 | 22.21 | +2.20 | 2 |

**Sack Rate**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| MIA | 2.59 | 0.00 | 2.07 | -0.52 | 2 |
| LV | 2.15 | 4.00 | 2.52 | +0.37 | 2 |
| CIN | 2.23 | 4.00 | 2.58 | +0.35 | 2 |
| NYG | 2.23 | 0.50 | 1.88 | -0.35 | 2 |
| LA | 2.63 | 1.00 | 2.31 | -0.33 | 2 |

## Season-to-date calibration

| Position | Weeks | MAE | Rank corr | Top-5 hit rate |
|---|---|---|---|---|
| DST | 2 | 6.07 | 0.282 | 40% |
| K | 2 | 3.52 | -0.166 | 40% |

