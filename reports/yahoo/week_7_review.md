# Week 7 review (2026)

_Generated 2026-10-27 13:26 UTC._

## How the model did

| Position | n | MAE | Vegas-only MAE | Rank corr | Vegas-only rank corr | Top-5 hit rate | Vegas-only top-5 | Top-5 avg pts | Slate avg |
|---|---|---|---|---|---|---|---|---|---|
| DST | 28 | 4.78 | 4.93 | 0.206 | 0.156 | 80% | 60% | 6.0 | 5.1 |
| K | 28 | 3.26 | 3.33 | 0.235 | 0.338 | 100% | 60% | 11.6 | 7.7 |

## D/ST: what it got wrong

Rank correlation 0.206 against a Vegas-only 0.156 -- clearly better than ranking on the Vegas number alone. The five recommended units averaged 6.0 points against a slate average of 5.1, and 80% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| CAR | TB | 7.6 | -3.0 | -10.6 |
| PHI | DAL | 5.7 | -4.0 | -9.7 |
| BAL | CIN | 5.7 | -3.0 | -8.7 |
| SF | ATL | 6.4 | -1.0 | -7.4 |
| HOU | NYG | 10.2 | 3.0 | -7.2 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| TB | CAR | 6.3 | 27.0 | +20.7 |
| ATL | SF | 4.8 | 12.0 | +7.2 |
| CIN | BAL | 7.7 | 12.0 | +4.3 |
| GB | DET | 4.5 | 8.0 | +3.5 |
| SEA | KC | 8.1 | 11.0 | +2.9 |

## Kicker: what it got wrong

Rank correlation 0.235 against a Vegas-only 0.338 -- behind the Vegas-only ranking, which is a bad week. The five recommended units averaged 11.6 points against a slate average of 7.7, and 100% of them finished startable (Vegas-only: 60%).

**Biggest overshoots** (projected high, scored low):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| R.Fitzgerald | TB | 8.6 | 1.0 | -7.6 |
| E.Pineiro | ATL | 9.1 | 2.0 | -7.1 |
| D.Zvada | HOU | 8.5 | 3.0 | -5.5 |
| D.Carlson | PIT | 8.9 | 4.0 | -4.9 |
| J.Elliott | DAL | 7.4 | 3.0 | -4.4 |

**Biggest undershoots** (projected low, scored high):

| Unit | Opp | Projected | Actual | Miss |
|---|---|---|---|---|
| C.Santos | NE | 8.4 | 15.0 | +6.6 |
| B.Aubrey | PHI | 8.9 | 14.0 | +5.1 |
| W.Lutz | ARI | 9.3 | 14.0 | +4.7 |
| C.McLaughlin | CAR | 8.3 | 12.0 | +3.7 |
| W.Reichard | IND | 9.5 | 13.0 | +3.5 |

## Factor ledger: which weights moved and why

Each factor's correlation with actual fantasy points, over the full historical window, the current season to date, and the trailing four weeks. The model weight is derived from a shrinkage blend of the historical and current-season numbers -- the historical prior's pull decays automatically as the season's sample grows.

### D/ST

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Opponent implied total | -0.289 | -0.216 | -0.210 | -0.255 | -0.880 | +0.011 |
| Point spread | 0.265 | 0.125 | 0.113 | 0.201 | 0.830 | +0.004 |
| Opp points per drive | -0.185 | -0.092 | -0.075 | -0.142 | 0.121 | +0.005 |
| Opp sack rate allowed (O-line) | 0.161 | 0.094 | 0.136 | 0.130 | 0.066 | -0.003 |
| Sack opportunity (def x O-line x volume) | 0.160 | 0.109 | 0.145 | 0.136 | 0.081 | -0.001 |
| Opp pressure rate allowed | 0.152 | 0.027 | 0.173 | 0.094 | 0.111 | +0.009 |
| Game total | -0.124 | -0.186 | -0.184 | -0.152 | -0.350 | +0.023 |
| Takeaway opportunity | 0.119 | 0.024 | 0.003 | 0.075 | 0.053 | -0.009 |
| Defense points allowed per drive | -0.098 | -0.099 | -0.115 | -0.098 | -0.019 | -0.003 |
| Opp INT rate | 0.093 | -0.054 | -0.022 | 0.025 | -0.004 | -0.010 |
| Defense INT rate | 0.085 | 0.048 | 0.005 | 0.068 | 0.073 | +0.029 |
| Opp drives per game | 0.084 | 0.111 | 0.016 | 0.097 | 0.125 | -0.001 |

**Biggest divergences from the historical prior:**

- **Opp pass rate** is running weaker this season (-0.102 vs 0.052 historically). Blended to -0.019, so its weight multiplier is 0.22x.
- **Opp INT rate** is running weaker this season (-0.054 vs 0.093 historically). Blended to 0.025, so its weight multiplier is 0.31x.
- **Point spread** is running weaker this season (0.125 vs 0.265 historically). Blended to 0.201, so its weight multiplier is 1.51x.

### Kicker

| Factor | Historical r | Season r | Last 4wk r | Blended r | Weight | Moved |
|---|---|---|---|---|---|---|
| Point spread | 0.134 | -0.000 | 0.082 | 0.072 | 0.372 | +0.011 |
| Team implied total | 0.122 | 0.000 | 0.042 | 0.066 | 0.232 | +0.009 |
| Implied total x FG rate | 0.109 | -0.031 | 0.024 | 0.044 | -0.003 | -0.001 |
| Implied total x red-zone stall | 0.106 | -0.005 | 0.078 | 0.055 | 0.003 | +0.002 |
| Opponent implied total | -0.099 | 0.001 | -0.077 | -0.053 | 0.000 | +0.001 |
| Wind (mph) | -0.097 | -0.008 | -0.081 | -0.056 | -0.361 | +0.028 |
| Dome / indoors | 0.094 | -0.034 | 0.006 | 0.035 | 0.090 | +0.020 |
| Wind x 50+ attempt share | -0.080 | 0.009 | -0.068 | -0.039 | 0.009 | +0.001 |
| Kicker 50+ attempt share | 0.058 | 0.058 | 0.031 | 0.058 | 0.065 | -0.003 |
| Home field | 0.046 | -0.146 | -0.116 | -0.043 | -0.099 | +0.006 |
| TDs per drive | 0.038 | -0.014 | -0.053 | 0.014 | -0.004 | -0.002 |
| Red-zone trips per drive | 0.037 | -0.033 | -0.062 | 0.004 | -0.004 | -0.002 |

**Biggest divergences from the historical prior:**

- **Home field** is running weaker this season (-0.146 vs 0.046 historically). Blended to -0.043, so its weight multiplier is 0.94x.
- **Implied total x FG rate** is running weaker this season (-0.031 vs 0.109 historically). Blended to 0.044, so its weight multiplier is 0.81x.
- **Point spread** is running weaker this season (-0.000 vs 0.134 historically). Blended to 0.072, so its weight multiplier is 1.07x.

## Per-team in-season adjustments

Season-to-date form blended against each team's own multi-season prior. A team only earns a large adjustment once it has the games to back it up.

**Big Play Points**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| DAL | 8.18 | 3.14 | 5.83 | -2.35 | 7 |
| PHI | 7.35 | 2.86 | 5.25 | -2.10 | 7 |
| MIA | 6.89 | 2.83 | 5.15 | -1.74 | 6 |
| IND | 7.11 | 3.86 | 5.59 | -1.52 | 7 |
| BUF | 7.37 | 4.33 | 6.07 | -1.30 | 6 |

**Points Allowed**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| MIN | 22.28 | 12.33 | 18.01 | -4.26 | 6 |
| BUF | 20.01 | 28.50 | 23.65 | +3.64 | 6 |
| NO | 21.04 | 27.71 | 24.15 | +3.12 | 7 |
| JAX | 23.10 | 17.33 | 20.63 | -2.47 | 6 |
| ARI | 25.24 | 29.71 | 27.33 | +2.09 | 7 |

**Sack Rate**

| Team | Prior | Season | Posterior | Move | Games |
|---|---|---|---|---|---|
| PHI | 2.72 | 1.14 | 1.98 | -0.73 | 7 |
| MIA | 2.59 | 1.00 | 1.91 | -0.68 | 6 |
| LA | 2.63 | 1.43 | 2.07 | -0.56 | 7 |
| DAL | 2.61 | 1.43 | 2.06 | -0.55 | 7 |
| CIN | 2.23 | 3.50 | 2.77 | +0.54 | 6 |

## Season-to-date calibration

| Position | Weeks | MAE | Rank corr | Top-5 hit rate |
|---|---|---|---|---|
| DST | 7 | 4.26 | 0.236 | 66% |
| K | 7 | 3.78 | 0.015 | 46% |

