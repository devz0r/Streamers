# Week 4 review (2026)

_Generated 2026-10-06 14:15 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 32 | 2.58 | 2.67 | 0.512 | 0.603 | 100% | 100% | 8.4 | 5.5 |
| K | 32 | 3.67 | 3.62 | 0.265 | 0.259 | 40% | 20% | 11.0 | 9.2 |

## D/ST: what it got wrong

Rank correlation 0.512 against a Vegas-only 0.603 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 8.4 points against a slate average of 5.5, and 100% of them finished startable (Vegas-only: 100%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| NO | ATL | 6.6 | -3.0 | -9.6 |
| DET | CAR | 6.5 | 2.0 | -4.5 |
| DAL | HOU | 5.5 | 1.0 | -4.5 |
| BUF | NE | 8.1 | 4.0 | -4.1 |
| SF | DEN | 6.5 | 3.0 | -3.5 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| NYG | ARI | 6.1 | 15.0 | +8.9 |
| IND | WAS | 7.3 | 11.0 | +3.7 |
| JAX | CIN | 5.2 | 8.0 | +2.8 |
| SEA | LAC | 9.7 | 12.0 | +2.3 |
| CLE | PIT | 7.6 | 9.0 | +1.4 |

## Kicker: what it got wrong

Rank correlation 0.265 against a Vegas-only 0.259 -- slightly ahead of ranking on the Vegas number alone. The five recommended units averaged 11.0 points against a slate average of 9.2, and 40% of them finished startable (Vegas-only: 20%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| C.McLaughlin | GB | 8.4 | 2.0 | -6.4 |
| A.Borregales | BUF | 7.8 | 3.0 | -4.8 |
| E.McPherson | JAX | 9.3 | 5.0 | -4.3 |
| C.Boswell | CLE | 8.3 | 4.0 | -4.3 |
| R.Patterson | MIN | 8.1 | 4.0 | -4.1 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| W.Reichard | MIA | 10.1 | 19.0 | +8.9 |
| J.Bates | CAR | 9.2 | 18.0 | +8.8 |
| M.Gay | KC | 8.9 | 16.0 | +7.1 |
| S.Shrader | WAS | 9.3 | 16.0 | +6.7 |
| K.Fairbairn | DAL | 9.5 | 16.0 | +6.5 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.289 | -0.258 | -0.258 | -0.278 | -0.894 | -0.001 |
| Point spread | 0.265 | 0.154 | 0.154 | 0.228 | 0.846 | -0.015 |
| Opp points per drive | -0.185 | -0.130 | -0.130 | -0.166 | 0.100 | +0.005 |
| Opp sack rate allowed (O-line) | 0.161 | 0.076 | 0.076 | 0.132 | 0.082 | +0.003 |
| Sack opportunity (def x O-line x volume) | 0.160 | 0.088 | 0.088 | 0.136 | 0.078 | +0.003 |
| Opp pressure rate allowed | 0.152 | -0.078 | -0.078 | 0.074 | 0.082 | -0.005 |
| Game total | -0.124 | -0.226 | -0.226 | -0.158 | -0.350 | -0.025 |
| Takeaway opportunity | 0.119 | 0.065 | 0.065 | 0.101 | 0.055 | -0.004 |
| Defense points allowed per drive | -0.098 | -0.104 | -0.104 | -0.100 | -0.005 | -0.011 |
| Opp INT rate | 0.093 | -0.114 | -0.114 | 0.023 | -0.006 | -0.007 |
| Defense INT rate | 0.085 | 0.115 | 0.115 | 0.095 | 0.086 | +0.007 |
| Opp drives per game | 0.084 | 0.150 | 0.150 | 0.106 | 0.114 | -0.003 |

**Biggest divergences from the historical prior:**

- **Opp pressure rate allowed** is running weaker this season (-0.078 vs 0.152 historically). Blended to 0.074, so its weight multiplier is 0.92x.
- **Opp INT rate** is running weaker this season (-0.114 vs 0.093 historically). Blended to 0.023, so its weight multiplier is 0.29x.
- **Opp turnover-worthy play rate** is running weaker this season (-0.076 vs 0.080 historically). Blended to 0.027, so its weight multiplier is 0.34x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.134 | -0.021 | -0.021 | 0.081 | 0.359 | +0.003 |
| Team implied total | 0.122 | 0.039 | 0.039 | 0.094 | 0.250 | +0.014 |
| Implied total x FG rate | 0.109 | -0.025 | -0.025 | 0.064 | -0.003 | -0.002 |
| Implied total x red-zone stall | 0.106 | -0.045 | -0.045 | 0.055 | -0.002 | -0.001 |
| Opponent implied total | -0.099 | 0.073 | 0.073 | -0.041 | -0.001 | -0.000 |
| Dome / indoors | 0.094 | -0.014 | -0.014 | 0.058 | 0.221 | +0.005 |
| Wind (mph) | -0.090 | 0.017 | 0.017 | -0.054 | -0.260 | -0.012 |
| Wind x 50+ attempt share | -0.077 | 0.034 | 0.034 | -0.039 | 0.006 | +0.002 |
| Kicker 50+ attempt share | 0.058 | 0.115 | 0.115 | 0.078 | 0.071 | +0.007 |
| Home field | 0.046 | -0.111 | -0.111 | -0.007 | -0.016 | +0.008 |
| TDs per drive | 0.038 | 0.066 | 0.066 | 0.048 | -0.008 | -0.004 |
| Red-zone trips per drive | 0.037 | 0.012 | 0.012 | 0.028 | -0.010 | -0.002 |

**Biggest divergences from the historical prior:**

- **Opponent implied total** is running stronger this season (0.073 vs -0.099 historically). Blended to -0.041, so its weight multiplier is 0.82x.
- **Home field** is running weaker this season (-0.111 vs 0.046 historically). Blended to -0.007, so its weight multiplier is 0.28x.
- **Point spread** is running weaker this season (-0.021 vs 0.134 historically). Blended to 0.081, so its weight multiplier is 1.22x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 8.18 | 2.50 | 6.29 | -1.89 | 4 |
| PHI | 7.35 | 3.00 | 5.90 | -1.45 | 4 |
| MIA | 6.89 | 2.75 | 5.51 | -1.38 | 4 |
| LV | 5.22 | 9.25 | 6.56 | +1.34 | 4 |
| MIN | 7.48 | 10.75 | 8.57 | +1.09 | 4 |

**Points Allowed**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| NO | 21.04 | 32.00 | 24.69 | +3.65 | 4 |
| JAX | 23.10 | 13.25 | 19.82 | -3.28 | 4 |
| MIN | 22.28 | 12.75 | 19.10 | -3.18 | 4 |
| CHI | 23.92 | 16.25 | 21.36 | -2.56 | 4 |
| DET | 24.37 | 31.75 | 26.83 | +2.46 | 4 |

**Sack Rate**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| PHI | 2.72 | 1.00 | 2.14 | -0.57 | 4 |
| DAL | 2.61 | 1.00 | 2.07 | -0.54 | 4 |
| MIA | 2.59 | 1.00 | 2.06 | -0.53 | 4 |
| KC | 2.55 | 1.00 | 2.03 | -0.52 | 4 |
| NYG | 2.23 | 0.75 | 1.74 | -0.49 | 4 |

## Season-to-date calibration

| Position | Weeks | MAE | Rank corr | Top-5 hit rate |
|---|---|---|---|---|
| DST | 4 | 4.12 | 0.237 | 55% |
| K | 4 | 3.68 | -0.087 | 30% |

