from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.ml.football_goal_range_model import (
    GOALS_ONLY_LEAGUES,
    XG_RICH_LEAGUES,
    LeagueGoalModel,
    compute_team_state,
    predict_fixture_fast,
)
from app.orm_models import Match, ModelRun, Prediction, UpcomingFixture
from app.schemas import LABEL_TO_FIELD

MODEL_STORE = Path(__file__).resolve().parents[1] / "ml" / "model_store"
MODEL_STORE.mkdir(parents=True, exist_ok=True)

ALL_LEAGUES = sorted(XG_RICH_LEAGUES | GOALS_ONLY_LEAGUES)

_model_cache: dict[str, LeagueGoalModel] = {}


class ModelNotTrainedError(RuntimeError):
    pass


def league_slug(league: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", league.lower()).strip("_")


def league_tier(league: str) -> str:
    return "rich" if league in XG_RICH_LEAGUES else "lower"


def _model_path(league: str) -> Path:
    return MODEL_STORE / f"{league_slug(league)}.joblib"


def load_cached_models() -> None:
    """Called once at startup: populate the in-memory cache from whatever
    .joblib files already exist on disk, so the server never forces a
    retrain just because the process restarted."""
    for league in ALL_LEAGUES:
        path = _model_path(league)
        if path.exists():
            try:
                _model_cache[league] = joblib.load(path)
            except Exception:  # noqa: BLE001 -- a corrupt/stale file just means "not cached"
                continue


def get_model(league: str) -> LeagueGoalModel:
    if league in _model_cache:
        return _model_cache[league]
    path = _model_path(league)
    if path.exists():
        model = joblib.load(path)
        _model_cache[league] = model
        return model
    raise ModelNotTrainedError(f"No trained model for '{league}' yet -- trigger a retrain first.")


def _matches_df(db: Session, league: str) -> pd.DataFrame:
    rows = db.execute(select(Match).where(Match.league == league)).scalars().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        [
            {
                "date": r.date,
                "league": r.league,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "home_goals": r.home_goals,
                "away_goals": r.away_goals,
                "home_shots": r.home_shots,
                "away_shots": r.away_shots,
                "home_shots_on_target": r.home_shots_on_target,
                "away_shots_on_target": r.away_shots_on_target,
                "home_corners": r.home_corners,
                "away_corners": r.away_corners,
                "home_cards": r.home_cards,
                "away_cards": r.away_cards,
                "home_xg": r.home_xg,
                "away_xg": r.away_xg,
            }
            for r in rows
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def train_league(db: Session, league: str) -> dict:
    history = _matches_df(db, league)
    if history.empty:
        raise ValueError(f"No matches in DB for league '{league}' -- run a scrape first.")

    tier = league_tier(league)
    model = LeagueGoalModel(league=league, tier=tier)
    model.fit(history, test_size=0.0)  # full history, no holdout -- this is for real predictions

    joblib.dump(model, _model_path(league))
    _model_cache[league] = model

    trained_at = datetime.now(timezone.utc)
    stmt = pg_insert(ModelRun).values(
        league=league,
        tier=tier,
        trained_at=trained_at,
        n_matches=len(history),
        model_path=str(_model_path(league)),
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[ModelRun.league],
        set_={
            "tier": stmt.excluded.tier,
            "trained_at": stmt.excluded.trained_at,
            "n_matches": stmt.excluded.n_matches,
            "model_path": stmt.excluded.model_path,
        },
    )
    db.execute(stmt)
    db.commit()

    return {"league": league, "n_matches": len(history), "trained_at": trained_at}


def generate_predictions(db: Session, league: str) -> dict:
    model = get_model(league)
    history = _matches_df(db, league)
    upcoming = db.execute(select(UpcomingFixture).where(UpcomingFixture.league == league)).scalars().all()

    # Team rolling-stat/Elo state depends only on real match history, which
    # is identical for every fixture in this league -- compute it once
    # instead of rebuilding the whole feature pipeline per fixture (that
    # per-fixture rebuild was previously the dominant cost of a retrain:
    # several minutes for a single mid-size league).
    team_state = compute_team_state(history, league_tier(league))

    n_ok = n_failed = 0
    for fx in upcoming:
        if fx.home_team not in team_state or fx.away_team not in team_state:
            n_failed += 1
            continue
        try:
            result = predict_fixture_fast(model, team_state, fx.home_team, fx.away_team)
        except (ValueError, IndexError):
            n_failed += 1
            continue
        n_ok += 1

        values = {
            "fixture_id": fx.fixture_id,
            "league": league,
            "lambda_home": result["lambda_home"],
            "lambda_away": result["lambda_away"],
            **{LABEL_TO_FIELD[label]: p for label, p in result["range_probabilities"].items()},
            "match_result": result["match_result"],
            "over_under": result["over_under"],
            "btts": result["btts"],
        }
        stmt = pg_insert(Prediction).values(**values)
        update_cols = {k: getattr(stmt.excluded, k) for k in values if k != "fixture_id"}
        update_cols["predicted_at"] = datetime.now(timezone.utc)
        stmt = stmt.on_conflict_do_update(index_elements=[Prediction.fixture_id], set_=update_cols)
        db.execute(stmt)

    db.commit()
    return {"league": league, "predicted": n_ok, "skipped": n_failed}


def retrain_leagues(leagues: list[str] | None) -> dict:
    """Entry point for the background job -- opens its own DB session since
    it runs outside the request lifecycle (see job_manager)."""
    targets = leagues or ALL_LEAGUES
    db = SessionLocal()
    results = []
    try:
        for league in targets:
            train_result = train_league(db, league)
            predict_result = generate_predictions(db, league)
            results.append({**train_result, **predict_result})
        return {"leagues": results}
    finally:
        db.close()


def predict_single(db: Session, fixture_id: int) -> dict | None:
    """Live recompute (not persisted) including the full 0..8+ goal-count
    breakdown, for the fixture-detail view."""
    fx = db.execute(select(UpcomingFixture).where(UpcomingFixture.fixture_id == fixture_id)).scalar_one_or_none()
    if fx is None:
        return None
    model = get_model(fx.league)
    history = _matches_df(db, fx.league)
    team_state = compute_team_state(history, league_tier(fx.league))
    return predict_fixture_fast(model, team_state, fx.home_team, fx.away_team)
