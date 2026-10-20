# Week 6 review (2026)

_Generated 2026-10-20 12:46 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 28 | 4.16 | 4.19 | 0.132 | 0.103 | 80% | 100% | 9.8 | 7.0 |
| K | 29 | 4.25 | 4.39 | -0.097 | -0.296 | 40% | 20% | 7.2 | 9.4 |

## D/ST: what it got wrong

Rank correlation 0.132 against a Vegas-only 0.103 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 9.8 points against a slate average of 7.0, and 80% of them finished startable (Vegas-only: 100%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| JAX | HOU | 8.1 | 1.0 | -7.1 |
| IND | TEN | 8.2 | 2.0 | -6.2 |
| LA | ARI | 8.8 | 3.0 | -5.8 |
| BUF | LV | 6.8 | 1.0 | -5.8 |
| PHI | CAR | 7.2 | 2.0 | -5.2 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| TEN | IND | 5.5 | 23.0 | +17.5 |
| KC | LAC | 9.8 | 18.0 | +8.2 |
| CHI | ATL | 6.0 | 14.0 | +8.0 |
| NO | NYG | 7.1 | 14.0 | +6.9 |
| GB | DAL | 4.6 | 11.0 | +6.4 |

## Kicker: what it got wrong

Rank correlation -0.097 against a Vegas-only -0.296 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 7.2 points against a slate average of 9.4, and 40% of them finished startable (Vegas-only: 20%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| E.Evans | ARI | 9.4 | 1.0 | -8.4 |
| N.Folk | CHI | 9.0 | 1.0 | -8.0 |
| B.Aubrey | GB | 8.8 | 1.0 | -7.8 |
| J.Elliott | CAR | 7.6 | 2.0 | -5.6 |
| B.Grupe | ARI | 9.4 | 4.0 | -5.4 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| C.Ryland | LA | 8.4 | 19.0 | +10.6 |
| J.Slye | IND | 8.4 | 18.0 | +9.6 |
| C.McLaughlin | PIT | 7.7 | 17.0 | +9.3 |
| C.Boswell | TB | 8.7 | 18.0 | +9.3 |
| T.Bass | LV | 9.3 | 17.0 | +7.7 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.289 | -0.243 | -0.219 | -0.269 | -0.891 | +0.000 |
| Point spread | 0.265 | 0.129 | 0.075 | 0.207 | 0.827 | +0.000 |
| Opp points per drive | -0.185 | -0.130 | -0.082 | -0.161 | 0.116 | +0.000 |
| Opp sack rate allowed (O-line) | 0.161 | 0.099 | 0.163 | 0.134 | 0.069 | +0.000 |
| Sack opportunity (def x O-line x volume) | 0.160 | 0.106 | 0.220 | 0.137 | 0.082 | +0.000 |
| Opp pressure rate allowed | 0.152 | 0.004 | 0.054 | 0.089 | 0.103 | +0.000 |
| Game total | -0.124 | -0.226 | -0.244 | -0.167 | -0.372 | +0.000 |
| Takeaway opportunity | 0.119 | 0.014 | 0.060 | 0.074 | 0.062 | +0.000 |
| Defense points allowed per drive | -0.098 | -0.101 | -0.153 | -0.099 | -0.016 | +0.000 |
| Opp INT rate | 0.093 | -0.020 | 0.013 | 0.045 | 0.006 | +0.000 |
| Defense INT rate | 0.085 | 0.002 | 0.009 | 0.049 | 0.044 | +0.000 |
| Opp drives per game | 0.084 | 0.157 | 0.066 | 0.115 | 0.126 | +0.000 |

**Biggest divergences from the historical prior:**

- **Opp pressure rate allowed** is running weaker this season (0.004 vs 0.152 historically). Blended to 0.089, so its weight multiplier is 1.13x.
- **Opp fumble-lost rate** is running stronger this season (0.133 vs -0.006 historically). Blended to 0.053, so its weight multiplier is 0.53x.
- **Point spread** is running weaker this season (0.129 vs 0.265 historically). Blended to 0.207, so its weight multiplier is 1.56x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.134 | -0.042 | -0.032 | 0.059 | 0.361 | +0.000 |
| Team implied total | 0.122 | -0.038 | -0.051 | 0.053 | 0.223 | +0.000 |
| Implied total x FG rate | 0.109 | -0.077 | -0.086 | 0.030 | -0.002 | +0.000 |
| Implied total x red-zone stall | 0.106 | -0.065 | -0.077 | 0.033 | 0.000 | +0.000 |
| Opponent implied total | -0.099 | 0.026 | -0.004 | -0.045 | -0.001 | +0.000 |
| Wind (mph) | -0.097 | -0.014 | -0.044 | -0.062 | -0.389 | +0.000 |
| Dome / indoors | 0.094 | -0.052 | -0.036 | 0.032 | 0.070 | +0.000 |
| Wind x 50+ attempt share | -0.080 | 0.001 | -0.030 | -0.046 | 0.009 | +0.000 |
| Kicker 50+ attempt share | 0.058 | 0.067 | 0.048 | 0.062 | 0.068 | +0.000 |
| Home field | 0.046 | -0.178 | -0.159 | -0.050 | -0.105 | +0.000 |
| TDs per drive | 0.038 | 0.004 | 0.008 | 0.024 | -0.002 | +0.000 |
| Red-zone trips per drive | 0.037 | -0.029 | -0.020 | 0.009 | -0.003 | +0.000 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.178 vs 0.046 historically). Blended to -0.050, so its weight multiplier is 1.03x.
- **Implied total x FG rate** is running weaker this season (-0.077 vs 0.109 historically). Blended to 0.030, so its weight multiplier is 0.54x.
- **Point spread** is running weaker this season (-0.042 vs 0.134 historically). Blended to 0.059, so its weight multiplier is 0.88x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 8.18 | 2.83 | 5.89 | -2.29 | 6 |
| PHI | 7.35 | 3.33 | 5.63 | -1.72 | 6 |
| MIA | 6.89 | 2.60 | 5.24 | -1.65 | 5 |
| IND | 7.11 | 3.50 | 5.56 | -1.55 | 6 |
| BUF | 7.37 | 4.33 | 6.07 | -1.30 | 6 |

**Points Allowed**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| NO | 21.04 | 29.67 | 24.73 | +3.70 | 6 |
| MIN | 22.28 | 12.80 | 18.63 | -3.64 | 5 |
| BUF | 20.01 | 28.50 | 23.65 | +3.64 | 6 |
| JAX | 23.10 | 17.33 | 20.63 | -2.47 | 6 |
| DET | 24.37 | 29.60 | 26.38 | +2.01 | 5 |

**Sack Rate**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| MIA | 2.59 | 0.80 | 1.90 | -0.69 | 5 |
| DET | 2.29 | 4.00 | 2.95 | +0.66 | 5 |
| PHI | 2.72 | 1.33 | 2.12 | -0.59 | 6 |
| LA | 2.63 | 1.33 | 2.08 | -0.56 | 6 |
| DAL | 2.61 | 1.33 | 2.06 | -0.55 | 6 |

## Season-to-date calibration

| Position | Weeks | MAE | Rank corr | Top-5 hit rate |
|---|---|---|---|---|
| DST | 6 | 4.19 | 0.241 | 63% |
| K | 6 | 3.86 | -0.021 | 37% |

