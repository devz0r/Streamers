# Week 2 review (2026)

_Generated 2026-09-22 15:26 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 32 | 4.80 | 4.67 | 0.222 | 0.253 | 20% | 40% | 4.6 | 6.7 |
| K | 32 | 3.20 | 3.07 | -0.143 | -0.097 | 20% | 20% | 6.6 | 8.3 |

## D/ST: what it got wrong

Rank correlation 0.222 against a Vegas-only 0.253 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 4.6 points against a slate average of 6.7, and 20% of them finished startable (Vegas-only: 40%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| TB | CLE | 10.5 | 3.0 | -7.5 |
| KC | IND | 8.3 | 1.0 | -7.3 |
| MIA | SF | 3.0 | -4.0 | -7.0 |
| PHI | TEN | 9.8 | 3.0 | -6.8 |
| WAS | DAL | 4.5 | -2.0 | -6.5 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| CAR | ATL | 7.8 | 27.0 | +19.2 |
| MIN | CHI | 5.3 | 18.0 | +12.7 |
| NE | PIT | 8.7 | 21.0 | +12.3 |
| CIN | HOU | 6.4 | 15.0 | +8.6 |
| LV | LAC | 5.8 | 12.0 | +6.2 |

## Kicker: what it got wrong

Rank correlation -0.143 against a Vegas-only -0.097 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 6.6 points against a slate average of 8.3, and 20% of them finished startable (Vegas-only: 20%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| C.Dicker | LV | 9.8 | 2.0 | -7.8 |
| C.Ryland | SEA | 8.3 | 1.0 | -7.3 |
| H.Mevis | NYG | 9.8 | 4.0 | -5.8 |
| E.Pineiro | MIA | 10.1 | 5.0 | -5.1 |
| N.Folk | CAR | 8.5 | 4.0 | -4.5 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| H.Butker | IND | 9.2 | 17.0 | +7.8 |
| B.Aubrey | WAS | 9.2 | 16.0 | +6.8 |
| C.McLaughlin | CLE | 9.6 | 16.0 | +6.4 |
| S.Shrader | KC | 8.1 | 14.0 | +5.9 |
| D.Carlson | BAL | 7.8 | 13.0 | +5.2 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.289 | -0.291 | -0.291 | -0.289 | -0.918 | -0.010 |
| Point spread | 0.265 | 0.217 | 0.217 | 0.255 | 0.903 | -0.013 |
| Opp points per drive | -0.185 | -0.219 | -0.219 | -0.192 | 0.083 | +0.006 |
| Opp sack rate allowed (O-line) | 0.161 | -0.006 | -0.006 | 0.127 | 0.064 | -0.007 |
| Sack opportunity (def x O-line x volume) | 0.160 | -0.073 | -0.073 | 0.112 | 0.044 | +0.009 |
| Opp pressure rate allowed | 0.152 | -0.069 | -0.069 | 0.107 | 0.123 | -0.013 |
| Game total | -0.124 | -0.200 | -0.200 | -0.139 | -0.308 | -0.035 |
| Takeaway opportunity | 0.119 | -0.059 | -0.059 | 0.083 | 0.050 | -0.014 |
| Defense points allowed per drive | -0.098 | -0.015 | -0.015 | -0.081 | 0.009 | +0.003 |
| Opp INT rate | 0.093 | -0.074 | -0.074 | 0.059 | 0.011 | -0.020 |
| Defense INT rate | 0.085 | -0.012 | -0.012 | 0.065 | 0.058 | +0.003 |
| Opp drives per game | 0.084 | 0.316 | 0.316 | 0.131 | 0.114 | +0.019 |

**Biggest divergences from the historical prior:**

- **Opp neutral-script pace** is running stronger this season (0.267 vs 0.009 historically). Blended to 0.062, so its weight multiplier is 0.60x.
- **Sack opportunity (def x O-line x volume)** is running weaker this season (-0.073 vs 0.160 historically). Blended to 0.112, so its weight multiplier is 1.41x.
- **Opp drives per game** is running stronger this season (0.316 vs 0.084 historically). Blended to 0.131, so its weight multiplier is 1.86x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.134 | -0.067 | -0.067 | 0.093 | 0.382 | -0.020 |
| Team implied total | 0.122 | -0.006 | -0.006 | 0.096 | 0.252 | -0.004 |
| Implied total x FG rate | 0.109 | -0.085 | -0.085 | 0.070 | -0.003 | -0.002 |
| Implied total x red-zone stall | 0.106 | -0.043 | -0.043 | 0.076 | 0.000 | +0.003 |
| Opponent implied total | -0.099 | 0.106 | 0.106 | -0.057 | -0.000 | -0.000 |
| Dome / indoors | 0.094 | -0.048 | -0.048 | 0.065 | 0.229 | -0.037 |
| Wind (mph) | -0.090 | 0.013 | 0.013 | -0.069 | -0.285 | +0.019 |
| Wind x 50+ attempt share | -0.077 | 0.038 | 0.038 | -0.053 | 0.005 | -0.001 |
| Kicker 50+ attempt share | 0.058 | 0.156 | 0.156 | 0.078 | 0.061 | +0.003 |
| Home field | 0.046 | -0.226 | -0.226 | -0.010 | -0.016 | -0.035 |
| TDs per drive | 0.038 | -0.014 | -0.014 | 0.028 | -0.007 | +0.000 |
| Red-zone trips per drive | 0.037 | -0.059 | -0.059 | 0.017 | -0.007 | +0.003 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.226 vs 0.046 historically). Blended to -0.010, so its weight multiplier is 0.32x.
- **Opp drives per game allowed** is running stronger this season (0.246 vs -0.009 historically). Blended to 0.043, so its weight multiplier is 0.43x.
- **Kicker PAT accuracy** is running weaker this season (-0.192 vs 0.021 historically). Blended to -0.022, so its weight multiplier is 0.63x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| CAR | 5.08 | 12.50 | 6.57 | +1.48 | 2 |
| CIN | 6.53 | 13.00 | 7.82 | +1.29 | 2 |
| LV | 5.22 | 10.50 | 6.28 | +1.06 | 2 |
| DAL | 8.18 | 3.00 | 7.14 | -1.04 | 2 |
| MIA | 6.89 | 2.00 | 5.91 | -0.98 | 2 |

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
| DST | 2 | 4.60 | 0.215 | 40% |
| K | 2 | 3.41 | -0.156 | 40% |

