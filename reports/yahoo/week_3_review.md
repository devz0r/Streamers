# Week 3 review (2026)

_Generated 2026-09-29 14:07 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 32 | 4.69 | 4.69 | 0.006 | -0.004 | 40% | 40% | 5.4 | 6.4 |
| K | 32 | 4.13 | 4.06 | -0.300 | -0.218 | 0% | 20% | 5.0 | 8.8 |

## D/ST: what it got wrong

Rank correlation 0.006 against a Vegas-only -0.004 -- slightly ahead of ranking on the Vegas number alone. The five recommended units averaged 5.4 points against a slate average of 6.4, and 40% of them finished startable (Vegas-only: 40%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| GB | ATL | 8.1 | -2.0 | -10.1 |
| SEA | WAS | 9.6 | 1.0 | -8.6 |
| PHI | CHI | 8.4 | 1.0 | -7.4 |
| NO | LV | 8.3 | 1.0 | -7.3 |
| NE | JAX | 5.8 | -1.0 | -6.8 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| MIN | TB | 7.8 | 19.0 | +11.2 |
| LAC | BUF | 4.2 | 14.0 | +9.8 |
| JAX | NE | 7.9 | 17.0 | +9.1 |
| CHI | PHI | 6.4 | 14.0 | +7.6 |
| LV | NO | 6.4 | 13.0 | +6.6 |

## Kicker: what it got wrong

Rank correlation -0.300 against a Vegas-only -0.218 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 5.0 points against a slate average of 8.8, and 0% of them finished startable (Vegas-only: 20%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| J.Elliott | CHI | 8.8 | 1.0 | -7.8 |
| J.Slye | NYG | 7.9 | 1.0 | -6.9 |
| T.Smack | ATL | 8.7 | 2.0 | -6.7 |
| D.Carlson | LV | 9.2 | 3.0 | -6.2 |
| W.Lutz | LA | 8.1 | 2.0 | -6.1 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| S.Shrader | HOU | 8.3 | 17.0 | +8.7 |
| H.Mevis | DEN | 7.8 | 16.0 | +8.2 |
| W.Reichard | TB | 8.3 | 16.0 | +7.7 |
| R.Fitzgerald | CLE | 8.2 | 15.0 | +6.8 |
| C.Ryland | SF | 7.9 | 14.0 | +6.1 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.289 | -0.224 | -0.224 | -0.271 | -0.893 | +0.025 |
| Point spread | 0.265 | 0.138 | 0.138 | 0.230 | 0.861 | -0.042 |
| Opp points per drive | -0.185 | -0.112 | -0.112 | -0.165 | 0.094 | +0.011 |
| Opp sack rate allowed (O-line) | 0.161 | 0.044 | 0.044 | 0.128 | 0.079 | +0.015 |
| Sack opportunity (def x O-line x volume) | 0.160 | 0.063 | 0.063 | 0.133 | 0.075 | +0.031 |
| Opp pressure rate allowed | 0.152 | -0.113 | -0.113 | 0.078 | 0.087 | -0.036 |
| Game total | -0.124 | -0.194 | -0.194 | -0.143 | -0.325 | -0.017 |
| Takeaway opportunity | 0.119 | 0.045 | 0.045 | 0.099 | 0.059 | +0.008 |
| Defense points allowed per drive | -0.098 | -0.082 | -0.082 | -0.093 | 0.006 | -0.004 |
| Opp INT rate | 0.093 | -0.089 | -0.089 | 0.043 | 0.001 | -0.010 |
| Defense INT rate | 0.085 | 0.098 | 0.098 | 0.088 | 0.079 | +0.021 |
| Opp drives per game | 0.084 | 0.226 | 0.226 | 0.123 | 0.117 | +0.003 |

**Biggest divergences from the historical prior:**

- **Opp pressure rate allowed** is running weaker this season (-0.113 vs 0.152 historically). Blended to 0.078, so its weight multiplier is 0.97x.
- **Opp neutral-script pace** is running stronger this season (0.197 vs 0.009 historically). Blended to 0.061, so its weight multiplier is 0.60x.
- **Opp INT rate** is running weaker this season (-0.089 vs 0.093 historically). Blended to 0.043, so its weight multiplier is 0.53x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.134 | -0.115 | -0.115 | 0.065 | 0.356 | -0.026 |
| Team implied total | 0.122 | -0.069 | -0.069 | 0.069 | 0.236 | -0.016 |
| Implied total x FG rate | 0.109 | -0.122 | -0.122 | 0.045 | -0.002 | +0.001 |
| Implied total x red-zone stall | 0.106 | -0.121 | -0.121 | 0.043 | -0.001 | -0.001 |
| Opponent implied total | -0.099 | 0.120 | 0.120 | -0.038 | -0.001 | -0.000 |
| Dome / indoors | 0.094 | -0.090 | -0.090 | 0.043 | 0.216 | -0.013 |
| Wind (mph) | -0.090 | 0.103 | 0.103 | -0.037 | -0.248 | +0.037 |
| Wind x 50+ attempt share | -0.077 | 0.121 | 0.121 | -0.022 | 0.004 | -0.002 |
| Kicker 50+ attempt share | 0.058 | 0.109 | 0.109 | 0.073 | 0.064 | +0.003 |
| Home field | 0.046 | -0.190 | -0.190 | -0.020 | -0.025 | -0.008 |
| TDs per drive | 0.038 | 0.045 | 0.045 | 0.040 | -0.005 | +0.002 |
| Red-zone trips per drive | 0.037 | 0.013 | 0.013 | 0.030 | -0.008 | -0.001 |

**Biggest divergences from the historical prior:**

- **Point spread** is running weaker this season (-0.115 vs 0.134 historically). Blended to 0.065, so its weight multiplier is 0.97x.
- **Home field** is running weaker this season (-0.190 vs 0.046 historically). Blended to -0.020, so its weight multiplier is 0.51x.
- **Implied total x FG rate** is running weaker this season (-0.122 vs 0.109 historically). Blended to 0.045, so its weight multiplier is 0.83x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| LV | 5.22 | 11.33 | 6.89 | +1.67 | 3 |
| MIN | 7.48 | 13.00 | 8.99 | +1.50 | 3 |
| DAL | 8.18 | 2.67 | 6.68 | -1.50 | 3 |
| PHI | 7.35 | 2.00 | 5.89 | -1.46 | 3 |
| CAR | 5.08 | 10.00 | 6.42 | +1.34 | 3 |

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
| DST | 3 | 4.63 | 0.145 | 40% |
| K | 3 | 3.68 | -0.204 | 27% |

