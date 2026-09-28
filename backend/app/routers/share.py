from datetime import date, timedelta

import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.routers import fixtures
from app.services.telegram_service import TelegramNotConfigured, build_digest, send_digest

router = APIRouter(prefix="/api/share", tags=["share"])


@router.post("/telegram")
def share_telegram(
    league: str | None = None,
    start: date | None = None,
    end: date | None = None,
    db: Session = Depends(get_db),
) -> dict:
    """Publishes every fixture/prediction in [start, end] (same filters the
    UI's DateRangeFilter uses) as a Telegram digest. No combined
    upcoming+finished query exists server-side -- this calls
    fixtures.list_upcoming/list_matches directly (they're plain functions
    FastAPI also exposes as routes) rather than adding a 4th copy of the
    prediction-joining pattern already duplicated across fixtures.py/predict.py."""
    today = date.today()
    yesterday = today - timedelta(days=1)

    wants_upcoming = end is None or end >= today
    wants_past = start is None or start <= yesterday

    # list_upcoming/list_matches's `limit` params default to a FastAPI
    # Query(...) sentinel that's only resolved by the DI system on a real
    # request -- calling them directly here bypasses that, so limit must be
    # passed explicitly (the same le=2000 ceiling those endpoints enforce;
    # a digest shouldn't silently truncate the way the UI's fetch can).
    upcoming = (
        fixtures.list_upcoming(
            league=league,
            start=start if start and start > today else None,
            end=end,
            limit=2000,
            db=db,
        )
        if wants_upcoming
        else []
    )
    finished = (
        fixtures.list_matches(
            league=league,
            start=start,
            end=end if end and end < today else yesterday,
            with_predictions=True,
            limit=2000,
            db=db,
        )
        if wants_past
        else []
    )

    chunks = build_digest(upcoming, finished)
    if not chunks:
        return {"sent": False, "fixtures": 0, "messages": 0, "detail": "No fixtures to share for this range"}

    try:
        n_messages = send_digest(chunks)
    except TelegramNotConfigured as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Telegram request failed: {e}") from e

    return {"sent": True, "fixtures": len(upcoming) + len(finished), "messages": n_messages}
