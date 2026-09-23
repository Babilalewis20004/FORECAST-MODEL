from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.orm_models import Match, ModelRun, Prediction, UpcomingFixture
from app.schemas import FixtureDetailOut, FixtureOut, MatchOut, PredictionOut
from app.services.model_service import ModelNotTrainedError, model_group, predict_single


def league_metrics(db: Session) -> dict[str, dict | None]:
    """Test-split metrics keyed by model group -- look up with model_group(league)."""
    return {r.league: r.metrics for r in db.query(ModelRun).all()}

router = APIRouter(prefix="/api", tags=["fixtures"])

# Only "NS" (not started) actually means "will be played, hasn't happened
# yet" -- the API-Football feed's other non-finished statuses in this data
# are PST (postponed, no new date) and CANC (cancelled), neither of which
# belongs in a default "what's coming up" view. Pass status= explicitly to
# see those.
DEFAULT_UPCOMING_STATUS = "NS"


@router.get("/fixtures/upcoming", response_model=list[FixtureOut])
def list_upcoming(
    league: str | None = None,
    start: date | None = None,
    end: date | None = None,
    status: str | None = None,
    limit: int = Query(default=500, le=2000),
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[FixtureOut]:
    q = db.query(UpcomingFixture)
    if league:
        q = q.filter(UpcomingFixture.league == league)
    if start:
        q = q.filter(UpcomingFixture.date >= start)
    if end:
        q = q.filter(UpcomingFixture.date <= end)
    q = q.filter(UpcomingFixture.status == (status or DEFAULT_UPCOMING_STATUS))
    fixtures = q.order_by(UpcomingFixture.date).offset(offset).limit(limit).all()

    fixture_ids = [f.fixture_id for f in fixtures]
    preds = {
        p.fixture_id: p
        for p in db.query(Prediction).filter(Prediction.fixture_id.in_(fixture_ids)).all()
    } if fixture_ids else {}

    metrics = league_metrics(db)
    out = []
    for f in fixtures:
        pred = preds.get(f.fixture_id)
        out.append(
            FixtureOut(
                fixture_id=f.fixture_id,
                date=f.date,
                league=f.league,
                home_team=f.home_team,
                away_team=f.away_team,
                status=f.status,
                prediction=PredictionOut.from_orm_row(pred, metrics.get(model_group(f.league))) if pred else None,
            )
        )
    return out


@router.get("/fixtures/{fixture_id}", response_model=FixtureDetailOut)
def get_fixture(fixture_id: int, full_distribution: bool = False, db: Session = Depends(get_db)) -> FixtureDetailOut:
    fx = db.query(UpcomingFixture).filter(UpcomingFixture.fixture_id == fixture_id).one_or_none()
    if fx is None:
        raise HTTPException(status_code=404, detail="Fixture not found")

    pred = db.query(Prediction).filter(Prediction.fixture_id == fixture_id).one_or_none()
    goal_distribution = None
    if full_distribution:
        try:
            live = predict_single(db, fixture_id)
            if live:
                goal_distribution = {str(k): v for k, v in live["goal_distribution"].items()}
        except ModelNotTrainedError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

    return FixtureDetailOut(
        fixture_id=fx.fixture_id,
        date=fx.date,
        league=fx.league,
        home_team=fx.home_team,
        away_team=fx.away_team,
        status=fx.status,
        prediction=PredictionOut.from_orm_row(pred, league_metrics(db).get(model_group(fx.league))) if pred else None,
        goal_distribution=goal_distribution,
    )


@router.get("/matches", response_model=list[MatchOut])
def list_matches(
    league: str | None = None,
    start: date | None = None,
    end: date | None = None,
    team: str | None = None,
    limit: int = Query(default=200, le=2000),
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[MatchOut]:
    q = db.query(Match)
    if league:
        q = q.filter(Match.league == league)
    if start:
        q = q.filter(Match.date >= start)
    if end:
        q = q.filter(Match.date <= end)
    if team:
        q = q.filter((Match.home_team == team) | (Match.away_team == team))
    return q.order_by(Match.date.desc()).offset(offset).limit(limit).all()
