# Week 5 review (2026)

_Generated 2026-10-13 15:06 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 30 | 6.46 | 6.51 | 0.355 | 0.363 | 80% | 60% | 18.5 | 10.3 |
| K | 30 | 4.35 | 4.28 | 0.303 | 0.085 | 40% | 40% | 9.3 | 8.4 |

## D/ST: what it got wrong

Rank correlation 0.355 against a Vegas-only 0.363 -- essentially level with the Vegas-only ranking. The five recommended units averaged 18.5 points against a slate average of 10.3, and 80% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| DAL | TB | 11.3 | -1.5 | -12.8 |
| LV | NE | 9.3 | -2.5 | -11.8 |
| IND | PIT | 10.0 | 0.5 | -9.5 |
| PIT | IND | 12.0 | 3.5 | -8.5 |
| JAX | PHI | 13.3 | 5.5 | -7.8 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| SF | SEA | 8.0 | 26.5 | +18.5 |
| GB | CHI | 8.2 | 26.5 | +18.3 |
| NYJ | CLE | 13.1 | 28.0 | +14.9 |
| DEN | LAC | 13.6 | 22.0 | +8.4 |
| DET | ARI | 8.7 | 16.0 | +7.3 |

## Kicker: what it got wrong

Rank correlation 0.303 against a Vegas-only 0.085 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 9.3 points against a slate average of 8.4, and 40% of them finished startable (Vegas-only: 40%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| C.Santos | GB | 8.3 | 0.0 | -8.3 |
| D.Carlson | MIN | 8.4 | 0.5 | -7.9 |
| H.Mevis | BUF | 8.8 | 2.0 | -6.8 |
| C.Ryland | DET | 8.1 | 2.0 | -6.1 |
| T.Smack | CHI | 8.1 | 2.5 | -5.6 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| J.Myers | SF | 8.6 | 23.0 | +14.4 |
| S.Shrader | PIT | 8.0 | 21.0 | +13.0 |
| K.Fairbairn | TEN | 9.0 | 17.0 | +8.0 |
| W.Reichard | NO | 9.0 | 16.0 | +7.0 |
| T.Loop | ATL | 7.9 | 14.0 | +6.1 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.316 | -0.340 | -0.332 | -0.325 | -1.309 | -0.019 |
| Point spread | 0.273 | 0.207 | 0.170 | 0.247 | 1.100 | +0.007 |
| Opp points per drive | -0.219 | -0.217 | -0.187 | -0.218 | 0.118 | -0.001 |
| Opp sack rate allowed (O-line) | 0.210 | 0.171 | 0.209 | 0.195 | 0.166 | -0.009 |
| Sack opportunity (def x O-line x volume) | 0.204 | 0.177 | 0.277 | 0.193 | 0.154 | +0.004 |
| Opp pressure rate allowed | 0.191 | 0.034 | 0.072 | 0.130 | 0.205 | +0.014 |
| Game total | -0.161 | -0.286 | -0.319 | -0.209 | -0.712 | -0.023 |
| Takeaway opportunity | 0.119 | 0.058 | 0.078 | 0.095 | 0.073 | -0.018 |
| Opp INT rate | 0.109 | -0.022 | -0.057 | 0.058 | 0.009 | -0.001 |
| Defense points allowed per drive | -0.108 | -0.146 | -0.198 | -0.123 | -0.049 | -0.019 |
| Opp turnover-worthy play rate | 0.096 | 0.027 | 0.007 | 0.069 | -0.041 | +0.005 |
| Opp drives per game | 0.092 | 0.160 | 0.113 | 0.118 | 0.134 | +0.007 |

**Biggest divergences from the historical prior:**

- **Opp pressure rate allowed** is running weaker this season (0.034 vs 0.191 historically). Blended to 0.130, so its weight multiplier is 1.36x.
- **Opp fumble-lost rate** is running stronger this season (0.132 vs 0.000 historically). Blended to 0.051, so its weight multiplier is 0.51x.
- **Opp INT rate** is running weaker this season (-0.022 vs 0.109 historically). Blended to 0.058, so its weight multiplier is 0.69x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.128 | 0.041 | 0.046 | 0.094 | 0.386 | +0.034 |
| Team implied total | 0.117 | 0.039 | 0.032 | 0.087 | 0.247 | +0.003 |
| Implied total x FG rate | 0.105 | -0.004 | 0.008 | 0.063 | -0.003 | +0.001 |
| Implied total x red-zone stall | 0.101 | -0.044 | -0.046 | 0.045 | -0.003 | +0.001 |
| Wind (mph) | -0.099 | 0.001 | 0.033 | -0.060 | -0.380 | -0.105 |
| Dome / indoors | 0.095 | -0.050 | -0.078 | 0.039 | 0.104 | -0.121 |
| Opponent implied total | -0.094 | -0.025 | -0.039 | -0.067 | -0.001 | +0.000 |
| Wind x 50+ attempt share | -0.083 | 0.021 | 0.050 | -0.043 | 0.008 | +0.002 |
| Kicker 50+ attempt share | 0.054 | 0.091 | 0.080 | 0.069 | 0.064 | -0.006 |
| Home field | 0.046 | -0.137 | -0.129 | -0.025 | -0.055 | -0.054 |
| TDs per drive | 0.037 | 0.059 | 0.059 | 0.045 | -0.007 | -0.000 |
| Red-zone trips per drive | 0.035 | 0.002 | -0.001 | 0.022 | -0.008 | +0.001 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.137 vs 0.046 historically). Blended to -0.025, so its weight multiplier is 0.58x.
- **Implied total x red-zone stall** is running weaker this season (-0.044 vs 0.101 historically). Blended to 0.045, so its weight multiplier is 0.89x.
- **Dome / indoors** is running weaker this season (-0.050 vs 0.095 historically). Blended to 0.039, so its weight multiplier is 0.82x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 11.80 | 2.90 | 8.38 | -3.42 | 5 |
| MIA | 10.29 | 3.90 | 7.83 | -2.46 | 5 |
| PHI | 10.81 | 4.90 | 8.54 | -2.27 | 5 |
| IND | 10.26 | 5.90 | 8.59 | -1.68 | 5 |
| LA | 10.34 | 6.40 | 8.83 | -1.52 | 5 |

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
| DST | 5 | 5.75 | 0.315 | 44% |
| K | 5 | 3.90 | -0.017 | 32% |

