# Week 6 review (2026)

_Generated 2026-10-20 12:46 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 28 | 6.31 | 6.38 | 0.166 | 0.141 | 60% | 60% | 14.0 | 10.2 |
| K | 29 | 4.25 | 4.42 | -0.086 | -0.283 | 20% | 20% | 7.1 | 9.1 |

## D/ST: what it got wrong

Rank correlation 0.166 against a Vegas-only 0.141 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 14.0 points against a slate average of 10.2, and 60% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| JAX | HOU | 11.3 | -0.5 | -11.8 |
| HOU | JAX | 10.1 | -0.5 | -10.6 |
| BUF | LV | 9.5 | 0.5 | -9.0 |
| NYG | NO | 8.8 | 0.0 | -8.8 |
| LA | ARI | 12.2 | 3.5 | -8.7 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| TEN | IND | 7.8 | 29.0 | +21.2 |
| CHI | ATL | 8.2 | 20.5 | +12.3 |
| ARI | LA | 3.2 | 12.5 | +9.3 |
| NO | NYG | 10.4 | 19.0 | +8.6 |
| DEN | SEA | 11.4 | 19.5 | +8.1 |

## Kicker: what it got wrong

Rank correlation -0.086 against a Vegas-only -0.283 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 7.1 points against a slate average of 9.1, and 20% of them finished startable (Vegas-only: 20%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| E.Evans | ARI | 9.1 | 1.0 | -8.1 |
| N.Folk | CHI | 8.6 | 1.0 | -7.6 |
| B.Aubrey | GB | 8.4 | 1.0 | -7.4 |
| J.Elliott | CAR | 7.3 | 2.0 | -5.3 |
| T.Loop | CLE | 8.1 | 3.0 | -5.1 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| C.Ryland | LA | 8.1 | 19.0 | +10.9 |
| J.Slye | IND | 8.1 | 18.0 | +9.9 |
| C.McLaughlin | PIT | 7.3 | 17.0 | +9.7 |
| C.Boswell | TB | 8.4 | 17.0 | +8.6 |
| T.Bass | LV | 9.0 | 17.0 | +8.0 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.316 | -0.305 | -0.275 | -0.311 | -1.294 | +0.000 |
| Point spread | 0.273 | 0.174 | 0.113 | 0.231 | 1.079 | +0.000 |
| Opp points per drive | -0.219 | -0.198 | -0.166 | -0.210 | 0.133 | +0.000 |
| Opp sack rate allowed (O-line) | 0.210 | 0.161 | 0.187 | 0.189 | 0.165 | +0.000 |
| Sack opportunity (def x O-line x volume) | 0.204 | 0.154 | 0.236 | 0.183 | 0.151 | +0.000 |
| Opp pressure rate allowed | 0.191 | 0.042 | 0.076 | 0.127 | 0.210 | +0.000 |
| Game total | -0.161 | -0.268 | -0.285 | -0.206 | -0.714 | +0.000 |
| Takeaway opportunity | 0.119 | 0.054 | 0.100 | 0.091 | 0.075 | +0.000 |
| Opp INT rate | 0.109 | 0.020 | 0.030 | 0.071 | 0.027 | +0.000 |
| Defense points allowed per drive | -0.108 | -0.117 | -0.140 | -0.112 | -0.024 | +0.000 |
| Opp turnover-worthy play rate | 0.096 | 0.080 | 0.118 | 0.089 | -0.019 | +0.000 |
| Opp drives per game | 0.092 | 0.188 | 0.131 | 0.133 | 0.147 | +0.000 |

**Biggest divergences from the historical prior:**

- **Opp fumble-lost rate** is running stronger this season (0.161 vs 0.000 historically). Blended to 0.069, so its weight multiplier is 0.60x.
- **Opp pressure rate allowed** is running weaker this season (0.042 vs 0.191 historically). Blended to 0.127, so its weight multiplier is 1.33x.
- **Opp pass rate** is running weaker this season (-0.074 vs 0.037 historically). Blended to -0.010, so its weight multiplier is 0.11x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.128 | -0.037 | -0.034 | 0.058 | 0.354 | +0.000 |
| Team implied total | 0.117 | -0.036 | -0.057 | 0.052 | 0.217 | +0.000 |
| Implied total x FG rate | 0.105 | -0.075 | -0.090 | 0.028 | -0.002 | +0.000 |
| Implied total x red-zone stall | 0.101 | -0.066 | -0.085 | 0.030 | -0.000 | +0.000 |
| Wind (mph) | -0.099 | -0.021 | -0.054 | -0.066 | -0.417 | +0.000 |
| Dome / indoors | 0.095 | -0.049 | -0.034 | 0.033 | 0.065 | +0.000 |
| Opponent implied total | -0.094 | 0.021 | -0.007 | -0.045 | -0.001 | +0.000 |
| Wind x 50+ attempt share | -0.083 | -0.005 | -0.039 | -0.050 | 0.008 | +0.000 |
| Kicker 50+ attempt share | 0.054 | 0.076 | 0.054 | 0.064 | 0.067 | +0.000 |
| Home field | 0.046 | -0.165 | -0.150 | -0.044 | -0.087 | +0.000 |
| TDs per drive | 0.037 | -0.001 | -0.003 | 0.021 | -0.001 | +0.000 |
| Red-zone trips per drive | 0.035 | -0.037 | -0.035 | 0.005 | -0.003 | +0.000 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.165 vs 0.046 historically). Blended to -0.044, so its weight multiplier is 0.97x.
- **Implied total x FG rate** is running weaker this season (-0.075 vs 0.105 historically). Blended to 0.028, so its weight multiplier is 0.53x.
- **Implied total x red-zone stall** is running weaker this season (-0.066 vs 0.101 historically). Blended to 0.030, so its weight multiplier is 0.59x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 11.80 | 4.08 | 8.49 | -3.31 | 6 |
| PHI | 10.81 | 4.92 | 8.28 | -2.53 | 6 |
| MIA | 10.29 | 3.90 | 7.83 | -2.46 | 5 |
| IND | 10.26 | 5.33 | 8.15 | -2.11 | 6 |
| LA | 10.34 | 5.75 | 8.37 | -1.97 | 6 |

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
| DST | 6 | 5.83 | 0.290 | 47% |
| K | 6 | 3.96 | -0.028 | 30% |

