from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.orm_models import ModelRun
from app.schemas import LeagueOut
from app.services.model_service import ALL_LEAGUES, league_tier, model_group

router = APIRouter(prefix="/api/leagues", tags=["leagues"])


@router.get("", response_model=list[LeagueOut])
def list_leagues(db: Session = Depends(get_db)) -> list[LeagueOut]:
    runs = {r.league: r for r in db.query(ModelRun).all()}
    out = []
    for league in ALL_LEAGUES:
        # International competitions share one model: training status and
        # n_matches are the pooled International model's.
        group = model_group(league)
        run = runs.get(group)
        out.append(
            LeagueOut(
                league=league,
                tier=league_tier(group),
                model_group=group,
                last_trained_at=run.trained_at if run else None,
                n_matches=run.n_matches if run else None,
            )
        )
    return out
