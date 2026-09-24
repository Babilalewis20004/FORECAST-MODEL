"""One-off migration: imports the root matches.csv / upcoming_fixtures.csv /
upcoming_predictions.csv into Postgres. Safe to re-run (upserts on natural
keys). The CSVs are left in place afterward as frozen seed/backup data --
the app never writes back to them."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import SessionLocal
from app.orm_models import Match, Prediction, UpcomingFixture
from app.schemas import LABEL_TO_FIELD

ROOT = Path(__file__).resolve().parents[2]

MATCH_STAT_COLS = [
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners", "home_cards", "away_cards", "home_xg", "away_xg",
]


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def import_matches(db) -> int:
    df = pd.read_csv(ROOT / "matches.csv", parse_dates=["date"])
    # Rows without an API-Football fixture_id came from an older source with
    # its own team spellings ("Nottingham" vs "Nottingham Forest"): they
    # duplicated scraped matches under phantom teams, so never import them.
    df = df[df["fixture_id"].notna()]
    n = 0
    for _, r in df.iterrows():
        fixture_id = _clean(r.get("fixture_id"))
        values = {
            "fixture_id": int(fixture_id) if fixture_id is not None else None,
            "date": r["date"].date(),
            "league": r["league"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_goals": int(r["home_goals"]),
            "away_goals": int(r["away_goals"]),
            **{c: _clean(r.get(c)) for c in MATCH_STAT_COLS},
        }
        stmt = pg_insert(Match).values(**values)
        update_cols = {k: getattr(stmt.excluded, k) for k in values if k not in ("league", "date", "home_team", "away_team")}
        stmt = stmt.on_conflict_do_update(
            index_elements=["league", "date", "home_team", "away_team"], set_=update_cols
        )
        db.execute(stmt)
        n += 1
    db.commit()
    return n


def import_upcoming(db) -> int:
    df = pd.read_csv(ROOT / "upcoming_fixtures.csv", parse_dates=["date"])
    n = 0
    for _, r in df.iterrows():
        values = {
            "fixture_id": int(r["fixture_id"]),
            "date": r["date"].date(),
            "league": r["league"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "status": r["status"],
        }
        stmt = pg_insert(UpcomingFixture).values(**values)
        update_cols = {k: getattr(stmt.excluded, k) for k in values if k != "fixture_id"}
        stmt = stmt.on_conflict_do_update(index_elements=[UpcomingFixture.fixture_id], set_=update_cols)
        db.execute(stmt)
        n += 1
    db.commit()
    return n


def import_predictions(db) -> int:
    path = ROOT / "upcoming_predictions.csv"
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    n = 0
    for _, r in df.iterrows():
        values = {
            "fixture_id": int(r["fixture_id"]),
            "league": r["league"],
            "lambda_home": float(r["lambda_home"]),
            "lambda_away": float(r["lambda_away"]),
            **{field: float(r[f"p_{label}"]) for label, field in LABEL_TO_FIELD.items()},
        }
        stmt = pg_insert(Prediction).values(**values)
        update_cols = {k: getattr(stmt.excluded, k) for k in values if k != "fixture_id"}
        stmt = stmt.on_conflict_do_update(index_elements=[Prediction.fixture_id], set_=update_cols)
        db.execute(stmt)
        n += 1
    db.commit()
    return n


def main() -> None:
    db = SessionLocal()
    try:
        n_matches = import_matches(db)
        n_upcoming = import_upcoming(db)
        n_predictions = import_predictions(db)
        print(f"Imported {n_matches} matches, {n_upcoming} upcoming fixtures, {n_predictions} predictions.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
