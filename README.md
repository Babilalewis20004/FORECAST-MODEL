# Predict Model

Football prediction app: trains a per-league Poisson/XGBoost +
Negative-Binomial model on historical results and serves, per upcoming
fixture, probabilities (and their fair decimal-odds equivalent) for:

- **Goal ranges** (1-3, 2-4, 3-6, etc.)
- **Match Result / 1X2** (Home Win / Draw / Away Win)
- **Over/Under Goals** (1.5, 2.5, 3.5)
- **Both Teams To Score (BTTS)** (Yes / No)

All four markets come from the same simulated joint score distribution for
a fixture — nothing is trained per market, so they're internally consistent
with each other (see [How data flows](#how-data-flows)).

- [`backend/app/ml/football_goal_range_model.py`](backend/app/ml/football_goal_range_model.py) — the model itself (feature engineering, training, Monte Carlo simulation, all four markets)
- [`backend/app/`](backend/app/) — FastAPI service: Postgres-backed API, model caching/persistence, manual scrape/retrain background jobs
- [`frontend/`](frontend/) — React + Vite UI: upcoming fixtures with goal-range, 1X2, Over/Under and BTTS predictions, league filter, refresh trigger
- [`legacy/`](legacy/) — the original standalone CLI scripts (`api_football_scraper.py`, `predict_upcoming.py`), superseded by the backend's service layer; kept for reference, not run directly anymore

Local/single-user app — no auth, no scheduler. Data refresh is a manual
button in the UI (or a `POST /api/refresh/*` call) rather than a cron job.

## Quickstart

Prerequisites: Python 3.10+, Node 18+, Docker Desktop running.

```powershell
docker compose up -d   # Postgres, from the project root
```

Then follow **Backend setup** and **Frontend setup** below (separate
terminals — both need to stay running). Once both are up, open
http://localhost:5173.

### Backend setup

```powershell
cd backend
python -m venv venv
venv\Scripts\Activate.ps1              # bash/WSL: source venv/Scripts/activate
pip install -r requirements.txt

copy .env.example .env                 # then edit .env:
#   API_FOOTBALL_KEY=...                 free key at api-football.com
#   DATABASE_URL=...                     defaults to host port 5433 (see note below)

python scripts\init_db.py              # creates tables (safe to re-run)
python scripts\import_csv_to_postgres.py   # one-off: seeds Postgres from the root CSVs
python scripts\migrate_add_markets.py  # adds the 1X2/O-U/BTTS columns (safe to re-run)
python scripts\migrate_add_ensemble.py # adds ensemble/calibration columns (safe to re-run)

uvicorn app.main:app --reload --port 8000
```

Verify it's up: http://localhost:8000/api/health should return `{"status":
"ok"}`, and http://localhost:8000/docs gives the interactive (Swagger)
API reference for every endpoint below.

> **Port note:** `docker-compose.yml` maps Postgres to host port **5433**,
> not 5432 — the default `.env.example` already points at 5433. If you
> don't have another Postgres running locally, feel free to change both
> the compose file's port mapping and `DATABASE_URL` back to 5432.

### Frontend setup

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — `vite.config.ts` proxies `/api` to
`localhost:8000`, so no CORS setup or `.env` is needed for local dev as
long as the backend is running on port 8000.

### First run

The seed import gives you a populated fixtures list immediately (imported
straight from the root CSVs, including their existing predictions). To
refresh against live data, click **Refresh data** in the UI (or `POST
/api/refresh/full`) — see [Manual refresh](#manual-refresh) below for what
that does and roughly how long it takes.

## How data flows

`api_football_scraper` (now `backend/app/services/scrape_service.py`) pulls
match history + upcoming fixtures from the API-Football API for all 6
leagues (Premier League, La Liga, EFL Championship, League One, League Two,
Scottish Premiership) and upserts them into Postgres, keyed on
`(league, date, home_team, away_team)`. `model_service.py` trains one
`LeagueGoalModel` per league on that history, persists it to
`backend/app/ml/model_store/*.joblib`, and writes predictions for every
upcoming fixture back to the `predictions` table. The API layer just reads
from Postgres and the in-memory/disk model cache — it never retrains on a
GET request.

For each fixture, the model predicts expected goals (`lambda_home`,
`lambda_away`) via two Poisson-objective XGBoost regressors, then Monte
Carlo simulates 20,000 *jointly* correlated home/away score draws from
Negative Binomial distributions parameterised by those lambdas
(`simulate_joint_score_distribution` in
[football_goal_range_model.py](backend/app/ml/football_goal_range_model.py)).
Every market is then just a different aggregation over that one sample set:

| Market | Derivation |
|---|---|
| Goal range (e.g. "2-4") | `P(lo <= home+away <= hi)` |
| Match Result / 1X2 | `P(home>away)` / `P(home==away)` / `P(home<away)` |
| Over/Under *N* | `P(home+away > N)` |
| BTTS | `P(home>0 AND away>0)` |

Because they share one simulation, the markets agree with each other (e.g.
a fixture with a high home-win probability will also skew its BTTS/O-U
numbers accordingly) rather than being four independently-calibrated
models that can quietly contradict one another.

`PredictionOut.match_result` / `.over_under` / `.btts` are `null` for any
prediction generated before this feature shipped, until that fixture's
league is next retrained — the frontend's `FixtureCard` falls back to
showing just the goal-range chart in that case.

Every market above also gets a decimal-odds counterpart (`range_odds`,
`match_result_odds`, `over_under_odds`, `btts_odds`) — fair, no-margin
odds (`1/probability`, via `decimal_odds()`/`probabilities_to_odds()` in
[football_goal_range_model.py](backend/app/ml/football_goal_range_model.py)),
same keys as their probability counterparts, `null` where a probability
is exactly 0. These are **computed at API-response time from the stored
probabilities, not persisted** — there's nothing to migrate, since odds
carry no information a probability doesn't already have. `range_odds`
is available for every prediction (goal-range probabilities have always
been stored); the other three odds fields inherit the same `null`-until-
retrained rule as their probabilities.

### Ensemble and calibration

Model A (the simulation above) is one of two models. The final probabilities
served for 1X2, BTTS and Over/Under are a blend of both, then calibrated:

| Layer | Module | What it does |
|---|---|---|
| 1. Features | [`features.py`](backend/app/ml/features.py) | Rolling 5/10-match goals, xG, shots and shots on target (for and against), shot conversion, xG-adjusted Elo, rest days, head-to-head record. Training and serving share one code path, so a fixture can't see its own or later results |
| 2a. Model A | [`football_goal_range_model.py`](backend/app/ml/football_goal_range_model.py) | Negative-Binomial copula simulation (unchanged) |
| 2b. Model B | [`gbdt_models.py`](backend/app/ml/gbdt_models.py) | XGBoost classifiers trained directly on outcomes: `multi:softprob` for 1X2, `binary:logistic` for BTTS and each Over line |
| 3. Blend + calibrate | [`calibration.py`](backend/app/ml/calibration.py), [`ensemble.py`](backend/app/ml/ensemble.py) | `P = w_A·P_A + (1-w_A)·P_B` per market, then Platt (sigmoid) calibration. Isotonic is also implemented. Log loss, Brier score and ECE metrics |

Each league's history is split by date: the oldest 70% trains both models,
the next 15% fits each market's blend weight `w_A` (grid search on log loss,
falling back to 0.45 when there's too little data) and its calibrator, and
the newest 15% is only used to report metrics. Both base models are then
refit on 100% of history for serving, reusing the fitted weights and
calibrators. Goal ranges come from Model A only; they're calibrated but not
blended, since Model B has no range classifiers.

Two things to know when reading the output:

- **Markets are no longer guaranteed to agree.** Each market is blended and
  calibrated on its own, so e.g. the final P(Over 2.5) need not exactly
  match the goal-range numbers. That's the price of per-market calibration.
- **The card's "skill vs base rate" tag is per league, not per fixture.**
  It's `1 − Brier(calibrated) / Brier(always predicting the base rate)` on
  that league's test split. It can be negative, and for lower-league goal
  markets it often sits near zero: those markets are close to unpredictable
  from pre-match data. Hover the tag for the full Brier breakdown and blend
  weights.

The market columns in `predictions` hold the final calibrated numbers (so
odds and the UI use them unchanged). `raw_probabilities` keeps the Model A,
Model B and blended values they came from, `ensemble_weights` the per-market
`w_A`, and `model_runs.metrics` the league's test-split Brier, log loss and
ECE for every source. `GET /api/predict/{fixture_id}` returns all of it.
Predictions made before this shipped have these fields `null` and show
"Uncalibrated · Model A only" until their league is retrained. A retrain now
takes about 3 minutes per league (about 20 minutes for all six).

### Tests

```powershell
cd backend
pip install -r requirements-dev.txt
python -m pytest
```

Covers feature leakage (rewriting later results can't change earlier
features), train/serve feature parity, rest-day and head-to-head
correctness, metric formulas against scikit-learn, calibrator and
blend-weight behaviour, and end-to-end ensemble training on synthetic data.
No database needed.

### Manual refresh

| Action | Endpoint |
|---|---|
| Scrape fixtures + retrain (the UI's main button) | `POST /api/refresh/full` |
| Scrape fixtures only | `POST /api/refresh/scrape` `{"include_stats": false}` |
| Scrape + backfill shots/corners/cards/xG | `POST /api/refresh/scrape` `{"include_stats": true}` |
| Retrain only (uses whatever's already in Postgres) | `POST /api/refresh/retrain` |

All of these run as a background job (single-flight — only one at a time);
poll `GET /api/refresh/status/{job_id}` for progress. A full retrain across
all 6 leagues takes roughly 20 minutes; a single league is about 3 minutes
(ensemble training plus Monte Carlo simulation over every upcoming fixture). Stats backfill costs one API request per match and is
budget-limited on the free plan (100/day), so it's kept separate from a
routine refresh — run it on its own and re-run daily until it reports
nothing left to fetch.

## Notes

- `tier="rich"` (Premier League/La Liga/EFL Championship) uses xG + shots
  features when present; other leagues fall back to goals + Elo only.
- The model file's `if __name__ == "__main__"` block still runs standalone
  (`python -m app.ml.football_goal_range_model` from `backend/`) — trains on
  synthetic data if no `matches.csv` is found, or the real seed CSV
  otherwise. Useful for quick experimentation without the API/DB running.
- `downloaded_files/api_football_cache/` caches every raw API response on
  disk (shared with the archived CLI scripts' original cache) — a
  re-scrape of already-fetched league-seasons costs no API budget.
- No Alembic in this project — a schema change to an *existing* table
  (like adding the 1X2/O-U/BTTS columns) needs its own small idempotent
  script (`backend/scripts/migrate_add_markets.py` is the pattern to
  follow for the next one), since `init_db.py`'s `create_all()` only
  creates tables that don't exist yet, never new columns on one that does.
