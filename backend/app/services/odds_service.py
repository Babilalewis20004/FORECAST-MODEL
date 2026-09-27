"""Phase 2: live closing-line-value (CLV) tracking.

capture_placed_snapshot() freezes the odds for a fixture/market/selection the
moment it's first flagged as a value bet (called from
scrape_service.scrape_odds() right after fresh odds are saved). poll_closing_
lines() later fills in the closing line from a sharp book (Pinnacle/Bet365)
in the 10-15 minutes before kickoff, and the resulting CLV -- it's meant to be
run on a schedule (see main.py's lifespan, which wires it into APScheduler)
but takes no scheduling dependency itself, so it stays a plain, testable
function. See orm_models.BetTracking for the row shape.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.ml.value_engine import BOOKS, assess_value, calculate_clv, flat_model_probabilities
from app.orm_models import BetTracking, Prediction, UpcomingFixture
from app.services.scrape_service import Budget, SHARP_BOOKMAKERS, _get, _session, parse_odds

# How often main.py schedules poll_closing_lines to run.
CLOSING_POLL_INTERVAL_MINUTES = 5
# A fixture is "due" once its kickoff falls inside this many minutes from now.
CLOSING_WINDOW_MINUTES = 15


def _book_for_selection(selection: str) -> str | None:
    return next((book for book, outcomes in BOOKS.items() if selection in outcomes), None)


def capture_placed_snapshot(db: Session, pred: Prediction, bookmaker: str, odds: dict[str, float]) -> None:
    """Insert a BetTracking row the first time this fixture's current model
    probabilities + these odds flag a value bet. A no-op if it's not a value
    bet, or if a row for this (fixture, market, selection) already exists --
    CLV is measured from the price available when the bet would first have
    been placed, not whatever it moves to afterwards."""
    probs = flat_model_probabilities(pred.match_result, pred.over_under, pred.btts)
    if not probs:
        return
    assessment = assess_value(probs, odds)
    if not assessment["is_value_bet"]:
        return
    selection = assessment["value_selection"]
    market = _book_for_selection(selection)
    if market is None:
        return

    values = {
        "fixture_id": pred.fixture_id,
        "market": market,
        "selection": selection,
        "bookmaker": bookmaker,
        "is_sharp_book": bookmaker in SHARP_BOOKMAKERS,
        "odds_placed_time": datetime.now(timezone.utc),
        "odds_placed_value": odds[selection],
        "ev_at_placement": assessment["expected_value"][selection],
    }
    stmt = pg_insert(BetTracking).values(**values).on_conflict_do_nothing(
        index_elements=["fixture_id", "market", "selection"]
    )
    db.execute(stmt)


def poll_closing_lines(job=None) -> dict:
    """For every tracked bet with no closing line yet whose fixture kicks off
    within CLOSING_WINDOW_MINUTES, fetch a fresh /odds response, take
    whichever of SHARP_BOOKMAKERS is present, and fill in the closing line +
    CLV. One request per fixture (not per tracked row), same as scrape_odds."""
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(minutes=CLOSING_WINDOW_MINUTES)
        open_rows = db.execute(
            select(BetTracking, UpcomingFixture)
            .join(UpcomingFixture, UpcomingFixture.fixture_id == BetTracking.fixture_id)
            .where(BetTracking.closing_odds_value.is_(None), UpcomingFixture.status == "NS")
        ).all()
        due = [(bt, fx) for bt, fx in open_rows if fx.kickoff_time is not None and now <= fx.kickoff_time <= cutoff]
        if not due:
            return {"open": len(open_rows), "due": 0, "updated": 0}

        by_fixture: dict[int, list[BetTracking]] = {}
        for bt, _fx in due:
            by_fixture.setdefault(bt.fixture_id, []).append(bt)

        session = _session()
        budget = Budget()
        updated = 0
        for fixture_id, rows in by_fixture.items():
            if job is not None:
                job.detail = f"closing lines: {updated} updated, fixture {fixture_id}"
            try:
                data = _get(session, budget, "odds", None, fixture=fixture_id)
            except (requests.HTTPError, RuntimeError):
                continue
            parsed = parse_odds(data["response"][0]) if data.get("response") else None
            if parsed is None:
                continue
            _best_bm, _best_odds, sharp_odds = parsed
            closing_bookmaker = next((name for name in SHARP_BOOKMAKERS if name in sharp_odds), None)
            if closing_bookmaker is None:
                continue
            closing_odds = sharp_odds[closing_bookmaker]

            for bt in rows:
                if bt.selection not in closing_odds:
                    continue
                clv = calculate_clv(bt.odds_placed_value, closing_odds, bt.selection)
                if clv is None:
                    continue
                bt.closing_bookmaker = closing_bookmaker
                bt.closing_odds_value = closing_odds[bt.selection]
                bt.closing_captured_at = datetime.now(timezone.utc)
                bt.clv_percentage = clv
                updated += 1
            db.commit()
        return {"open": len(open_rows), "due": len(due), "updated": updated}
    finally:
        db.close()
