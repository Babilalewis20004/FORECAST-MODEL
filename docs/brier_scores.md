# Brier scores

Per-league, per-market Brier scores from each league's `ModelRun.metrics`
(held-out test split; see `backend/app/ml/ensemble.py`'s `train_ensemble`).
Lower is better. **Calibrated** is what's actually served (the final
blended + calibrated probabilities); **Model A** / **Model B** are the two
base models before blending (Model B is `—` for the 8 goal-range markets,
which only Model A predicts); **Baseline** is always predicting the
league's historical base rate.

This is a snapshot as of the trained_at timestamps below -- retraining a
league updates its `ModelRun` row, so these numbers will drift. To
regenerate, query `ModelRun` directly:

```python
from app.db import SessionLocal
from app.orm_models import ModelRun

db = SessionLocal()
for r in db.query(ModelRun).order_by(ModelRun.league).all():
    print(r.league, r.trained_at, r.metrics["markets"])
```

## Match result -- summary across leagues

| League | Calibrated | vs. baseline | Trained |
|---|---|---|---|
| Scottish Premiership | 0.564 | 0.637 | 2026-09-27 |
| International | 0.574 | 0.634 | 2026-09-25 |
| Serie A | 0.582 | 0.653 | 2026-09-27 |
| La Liga | 0.589 | 0.631 | 2026-09-25 |
| League One | 0.621 | 0.657 | 2026-09-25 |
| League Two | 0.632 | 0.644 | 2026-09-25 |
| Ligue 1 | 0.635 | 0.657 | 2026-09-27 |
| Bundesliga | 0.636 | 0.649 | 2026-09-27 |
| Premier League | 0.648 | 0.675 | 2026-09-25 |
| EFL Championship | 0.651 | 0.663 | 2026-09-25 |

Every trained league beats its baseline on `match_result`.

## Full breakdown, per league

### Bundesliga (1,268 matches, trained 2026-09-27)

| Market | Calibrated | Model A | Model B | Baseline |
|---|---|---|---|---|
| match_result | 0.636 | 0.612 | 0.668 | 0.649 |
| btts | 0.233 | 0.254 | 0.253 | 0.232 |
| over_1.5 | 0.131 | 0.158 | 0.140 | 0.131 |
| over_2.5 | 0.224 | 0.269 | 0.243 | 0.228 |
| over_3.5 | 0.244 | 0.283 | 0.249 | 0.250 |
| range_1-3 | 0.254 | 0.272 | — | 0.251 |
| range_1-4 | 0.214 | 0.212 | — | 0.199 |
| range_2-3 | 0.239 | 0.247 | — | 0.241 |
| range_2-4 | 0.237 | 0.248 | — | 0.238 |
| range_2-5 | 0.193 | 0.206 | — | 0.190 |
| range_2-6 | 0.155 | 0.175 | — | 0.152 |
| range_3-5 | 0.248 | 0.270 | — | 0.250 |
| range_3-6 | 0.234 | 0.272 | — | 0.238 |

### EFL Championship (2,323 matches, trained 2026-09-25)

| Market | Calibrated | Model A | Model B | Baseline |
|---|---|---|---|---|
| match_result | 0.651 | 0.667 | 0.670 | 0.663 |
| btts | 0.244 | 0.262 | 0.261 | 0.250 |
| over_1.5 | 0.169 | 0.180 | 0.179 | 0.172 |
| over_2.5 | 0.251 | 0.260 | 0.265 | 0.253 |
| over_3.5 | 0.203 | 0.206 | 0.211 | 0.203 |
| range_1-3 | 0.230 | 0.229 | — | 0.230 |
| range_1-4 | 0.136 | 0.137 | — | 0.136 |
| range_2-3 | 0.250 | 0.251 | — | 0.251 |
| range_2-4 | 0.214 | 0.223 | — | 0.217 |
| range_2-5 | 0.191 | 0.202 | — | 0.195 |
| range_2-6 | 0.177 | 0.188 | — | 0.180 |
| range_3-5 | 0.250 | 0.263 | — | 0.254 |
| range_3-6 | 0.251 | 0.262 | — | 0.254 |

### International (4,278 matches, trained 2026-09-25)

| Market | Calibrated | Model A | Model B | Baseline |
|---|---|---|---|---|
| match_result | 0.574 | 0.587 | 0.577 | 0.634 |
| btts | 0.247 | 0.252 | 0.249 | 0.250 |
| over_1.5 | 0.186 | 0.188 | 0.190 | 0.187 |
| over_2.5 | 0.248 | 0.256 | 0.255 | 0.250 |
| over_3.5 | 0.215 | 0.230 | 0.221 | 0.216 |
| range_1-3 | 0.240 | 0.250 | — | 0.240 |
| range_1-4 | 0.175 | 0.180 | — | 0.175 |
| range_2-3 | 0.247 | 0.252 | — | 0.246 |
| range_2-4 | 0.241 | 0.241 | — | 0.239 |
| range_2-5 | 0.218 | 0.217 | — | 0.216 |
| range_2-6 | 0.202 | 0.202 | — | 0.202 |
| range_3-5 | 0.252 | 0.250 | — | 0.249 |
| range_3-6 | 0.252 | 0.253 | — | 0.250 |

### La Liga (1,589 matches, trained 2026-09-25)

| Market | Calibrated | Model A | Model B | Baseline |
|---|---|---|---|---|
| match_result | 0.589 | 0.606 | 0.593 | 0.631 |
| btts | 0.240 | 0.271 | 0.234 | 0.247 |
| over_1.5 | 0.178 | 0.191 | 0.190 | 0.180 |
| over_2.5 | 0.243 | 0.263 | 0.244 | 0.252 |
| over_3.5 | 0.209 | 0.212 | 0.209 | 0.213 |
| range_1-3 | 0.229 | 0.226 | — | 0.228 |
| range_1-4 | 0.187 | 0.181 | — | 0.181 |
| range_2-3 | 0.254 | 0.251 | — | 0.249 |
| range_2-4 | 0.257 | 0.251 | — | 0.243 |
| range_2-5 | 0.216 | 0.220 | — | 0.214 |
| range_2-6 | 0.195 | 0.204 | — | 0.195 |
| range_3-5 | 0.248 | 0.254 | — | 0.250 |
| range_3-6 | 0.250 | 0.261 | — | 0.253 |

### League One (2,311 matches, trained 2026-09-25)

| Market | Calibrated | Model A | Model B | Baseline |
|---|---|---|---|---|
| match_result | 0.621 | 0.625 | 0.621 | 0.657 |
| btts | 0.246 | 0.260 | 0.255 | 0.251 |
| over_1.5 | 0.190 | 0.196 | 0.192 | 0.192 |
| over_2.5 | 0.247 | 0.266 | 0.253 | 0.251 |
| over_3.5 | 0.225 | 0.238 | 0.232 | 0.225 |
| range_1-3 | 0.249 | 0.250 | — | 0.245 |
| range_1-4 | 0.168 | 0.166 | — | 0.166 |
| range_2-3 | 0.252 | 0.248 | — | 0.248 |
| range_2-4 | 0.240 | 0.241 | — | 0.239 |
| range_2-5 | 0.221 | 0.223 | — | 0.219 |
| range_2-6 | 0.203 | 0.207 | — | 0.202 |
| range_3-5 | 0.250 | 0.262 | — | 0.250 |
| range_3-6 | 0.251 | 0.267 | — | 0.252 |

### League Two (2,312 matches, trained 2026-09-25)

| Market | Calibrated | Model A | Model B | Baseline |
|---|---|---|---|---|
| match_result | 0.632 | 0.634 | 0.644 | 0.644 |
| btts | 0.251 | 0.257 | 0.259 | 0.250 |
| over_1.5 | 0.208 | 0.211 | 0.216 | 0.207 |
| over_2.5 | 0.251 | 0.262 | 0.262 | 0.250 |
| over_3.5 | 0.199 | 0.197 | 0.205 | 0.194 |
| range_1-3 | 0.227 | 0.231 | — | 0.227 |
| range_1-4 | 0.155 | 0.158 | — | 0.155 |
| range_2-3 | 0.247 | 0.248 | — | 0.248 |
| range_2-4 | 0.241 | 0.243 | — | 0.240 |
| range_2-5 | 0.218 | 0.220 | — | 0.218 |
| range_2-6 | 0.212 | 0.215 | — | 0.212 |
| range_3-5 | 0.249 | 0.255 | — | 0.249 |
| range_3-6 | 0.250 | 0.260 | — | 0.250 |

### Ligue 1 (1,348 matches, trained 2026-09-27)

| Market | Calibrated | Model A | Model B | Baseline |
|---|---|---|---|---|
| match_result | 0.635 | 0.640 | 0.656 | 0.657 |
| btts | 0.248 | 0.264 | 0.258 | 0.248 |
| over_1.5 | 0.180 | 0.197 | 0.193 | 0.180 |
| over_2.5 | 0.244 | 0.259 | 0.258 | 0.247 |
| over_3.5 | 0.221 | 0.221 | 0.229 | 0.221 |
| range_1-3 | 0.242 | 0.247 | — | 0.248 |
| range_1-4 | 0.184 | 0.187 | — | 0.183 |
| range_2-3 | 0.248 | 0.249 | — | 0.248 |
| range_2-4 | 0.240 | 0.244 | — | 0.235 |
| range_2-5 | 0.214 | 0.225 | — | 0.210 |
| range_2-6 | 0.203 | 0.217 | — | 0.200 |
| range_3-5 | 0.252 | 0.261 | — | 0.250 |
| range_3-6 | 0.253 | 0.262 | — | 0.250 |

### Premier League (1,570 matches, trained 2026-09-25)

| Market | Calibrated | Model A | Model B | Baseline |
|---|---|---|---|---|
| match_result | 0.648 | 0.659 | 0.679 | 0.675 |
| btts | 0.247 | 0.269 | 0.250 | 0.245 |
| over_1.5 | 0.177 | 0.181 | 0.186 | 0.178 |
| over_2.5 | 0.249 | 0.265 | 0.256 | 0.251 |
| over_3.5 | 0.209 | 0.220 | 0.223 | 0.214 |
| range_1-3 | 0.236 | 0.245 | — | 0.236 |
| range_1-4 | 0.170 | 0.172 | — | 0.170 |
| range_2-3 | 0.250 | 0.252 | — | 0.250 |
| range_2-4 | 0.233 | 0.232 | — | 0.232 |
| range_2-5 | 0.198 | 0.202 | — | 0.198 |
| range_2-6 | 0.193 | 0.196 | — | 0.193 |
| range_3-5 | 0.250 | 0.263 | — | 0.250 |
| range_3-6 | 0.250 | 0.264 | — | 0.252 |

### Scottish Premiership (978 matches, trained 2026-09-27)

| Market | Calibrated | Model A | Model B | Baseline |
|---|---|---|---|---|
| match_result | 0.564 | 0.566 | 0.587 | 0.637 |
| btts | 0.258 | 0.271 | 0.272 | 0.249 |
| over_1.5 | 0.181 | 0.196 | 0.204 | 0.178 |
| over_2.5 | 0.251 | 0.255 | 0.296 | 0.247 |
| over_3.5 | 0.210 | 0.216 | 0.242 | 0.213 |
| range_1-3 | 0.233 | 0.236 | — | 0.232 |
| range_1-4 | 0.188 | 0.185 | — | 0.179 |
| range_2-3 | 0.255 | 0.257 | — | 0.249 |
| range_2-4 | 0.264 | 0.251 | — | 0.241 |
| range_2-5 | 0.222 | 0.226 | — | 0.216 |
| range_2-6 | 0.202 | 0.211 | — | 0.196 |
| range_3-5 | 0.247 | 0.249 | — | 0.250 |
| range_3-6 | 0.245 | 0.254 | — | 0.250 |

### Serie A (1,571 matches, trained 2026-09-27)

| Market | Calibrated | Model A | Model B | Baseline |
|---|---|---|---|---|
| match_result | 0.582 | 0.583 | 0.595 | 0.653 |
| btts | 0.250 | 0.255 | 0.267 | 0.251 |
| over_1.5 | 0.196 | 0.198 | 0.203 | 0.198 |
| over_2.5 | 0.252 | 0.266 | 0.270 | 0.250 |
| over_3.5 | 0.176 | 0.174 | 0.181 | 0.179 |
| range_1-3 | 0.216 | 0.212 | — | 0.214 |
| range_1-4 | 0.177 | 0.171 | — | 0.173 |
| range_2-3 | 0.254 | 0.250 | — | 0.250 |
| range_2-4 | 0.246 | 0.242 | — | 0.245 |
| range_2-5 | 0.213 | 0.216 | — | 0.214 |
| range_2-6 | 0.204 | 0.208 | — | 0.206 |
| range_3-5 | 0.250 | 0.264 | — | 0.249 |
| range_3-6 | 0.251 | 0.268 | — | 0.250 |

## Notes

- `match_result` shows the biggest, most consistent calibration gain over
  baseline across every league.
- Bundesliga, Serie A, and Ligue 1 were added after verifying their
  API-Football stats coverage (shots/xG/possession) matched the existing
  xG-rich leagues -- see `XG_RICH_LEAGUES` in `football_goal_range_model.py`.
- Leagues with smaller training sets (Scottish Premiership at 978 matches,
  Bundesliga/Ligue 1/Serie A around 1,300-1,600) tend to show noisier
  calibrated-vs-baseline gaps on the smaller binary/goal-range markets than
  the larger leagues (International at 4,278, EFL Championship at 2,323).