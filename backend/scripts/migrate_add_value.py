"""One-off, idempotent migration for the value-bet layer: adds the bookmaker
odds snapshot and value fields to predictions (all nullable) without touching
existing data. Same pattern as migrate_add_ensemble.py; safe on a fresh
database and safe to re-run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.db import engine

NEW_COLUMNS = [
    ("bookmaker_odds", "JSONB"),
    ("odds_bookmaker", "VARCHAR"),
    ("odds_fetched_at", "TIMESTAMPTZ"),
    ("expected_value", "JSONB"),
    ("is_value_bet", "BOOLEAN"),
    ("value_selection", "VARCHAR"),
    ("recommended_stake", "DOUBLE PRECISION"),
]


def main() -> None:
    with engine.begin() as conn:
        for col, sql_type in NEW_COLUMNS:
            conn.execute(text(f"ALTER TABLE predictions ADD COLUMN IF NOT EXISTS {col} {sql_type}"))
    print(f"Ensured columns exist: {[f'predictions.{c}' for c, _ in NEW_COLUMNS]}")


if __name__ == "__main__":
    main()
