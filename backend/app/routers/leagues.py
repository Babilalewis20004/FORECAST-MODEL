from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.orm_models import ModelRun
from app.schemas import LeagueOut
from app.services.model_service import ALL_LEAGUES, league_tier

router = APIRouter(prefix="/api/leagues", tags=["leagues"])


@router.get("", response_model=list[LeagueOut])
def list_leagues(db: Session = Depends(get_db)) -> list[LeagueOut]:
    runs = {r.league: r for r in db.query(ModelRun).all()}
    return [
        LeagueOut(
            league=league,
            tier=league_tier(league),
            last_trained_at=runs[league].trained_at if league in runs else None,
            n_matches=runs[league].n_matches if league in runs else None,
        )
        for league in ALL_LEAGUES
    ]
