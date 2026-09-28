from datetime import date, timedelta

import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.routers import fixtures
from app.schemas import FixtureOut, MatchOut
from app.services.card_image_service import send_card_albums
from app.services.telegram_service import TelegramNotConfigured, build_digest, send_digest

router = APIRouter(prefix="/api/share", tags=["share"])


def _fixtures_for_range(
    league: str | None, start: date | None, end: date | None, db: Session
) -> tuple[list[FixtureOut], list[MatchOut]]:
    """Same [start, end] fixture set the UI's DateRangeFilter shows (split
    at today, upcoming vs. finished). No combined upcoming+finished query
    exists server-side -- calls fixtures.list_upcoming/list_matches
    directly (plain functions FastAPI also exposes as routes) rather than
    adding another copy of the prediction-joining pattern already
    duplicated across fixtures.py/predict.py. Their `limit` params default
    to a FastAPI Query(...) sentinel only resolved by the DI system on a
    real request, so it must be passed explicitly here (le=2000, the same
    ceiling those endpoints enforce -- a share shouldn't silently truncate
    the way the UI's fetch can)."""
    today = date.today()
    yesterday = today - timedelta(days=1)

    wants_upcoming = end is None or end >= today
    wants_past = start is None or start <= yesterday

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
    return upcoming, finished


@router.post("/telegram")
def share_telegram(
    league: str | None = None,
    start: date | None = None,
    end: date | None = None,
    db: Session = Depends(get_db),
) -> dict:
    """Publishes every fixture/prediction in [start, end] as a Telegram text digest."""
    upcoming, finished = _fixtures_for_range(league, start, end, db)

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


@router.post("/telegram/cards")
def share_telegram_cards(
    league: str | None = None,
    start: date | None = None,
    end: date | None = None,
    db: Session = Depends(get_db),
) -> dict:
    """Publishes every fixture/prediction in [start, end] as image cards,
    one per fixture, sent as Telegram albums (10 per message)."""
    upcoming, finished = _fixtures_for_range(league, start, end, db)
    items = [*upcoming, *finished]
    items.sort(key=lambda f: (f.date, f.league))
    if not items:
        return {"sent": False, "fixtures": 0, "albums": 0, "detail": "No fixtures to share for this range"}

    try:
        n_albums = send_card_albums(items)
    except TelegramNotConfigured as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Telegram request failed: {e}") from e

    return {"sent": True, "fixtures": len(items), "albums": n_albums}
