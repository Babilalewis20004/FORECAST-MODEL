from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.orm_models import ModelRun, Prediction, UpcomingFixture
from app.schemas import FixtureOut, PredictionOut
from app.services.model_service import model_group

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.get("/{fixture_id}", response_model=FixtureOut)
def get_prediction(fixture_id: int, db: Session = Depends(get_db)) -> FixtureOut:
    """The blended, calibrated prediction for one fixture, as persisted by the
    last retrain -- including raw Model A/B/blended probabilities, ensemble
    weights, and the league's test-split Brier scores. Served from the
    database; never retrains or re-simulates."""
    fx = db.query(UpcomingFixture).filter(UpcomingFixture.fixture_id == fixture_id).one_or_none()
    if fx is None:
        raise HTTPException(status_code=404, detail="Fixture not found")
    pred = db.query(Prediction).filter(Prediction.fixture_id == fixture_id).one_or_none()
    if pred is None:
        raise HTTPException(status_code=404, detail="No prediction for this fixture yet -- trigger a retrain")

    run = db.query(ModelRun).filter(ModelRun.league == model_group(fx.league)).one_or_none()
    return FixtureOut(
        fixture_id=fx.fixture_id,
        date=fx.date,
        league=fx.league,
        home_team=fx.home_team,
        away_team=fx.away_team,
        status=fx.status,
        prediction=PredictionOut.from_orm_row(pred, run.metrics if run else None),
    )
