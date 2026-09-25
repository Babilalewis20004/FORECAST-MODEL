"""One-off, idempotent migration for the tactical/situational feature layer:
adds possession and referee columns to matches, and referee to
upcoming_fixtures (all nullable), without touching existing data. Same
pattern as migrate_add_value.py, safe on a fresh database and safe to
re-run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.db import engine

MATCHES_COLUMNS = [
    ("home_possession", "DOUBLE PRECISION"),
    ("away_possession", "DOUBLE PRECISION"),
    ("referee", "VARCHAR"),
]
UPCOMING_FIXTURES_COLUMNS = [
    ("referee", "VARCHAR"),
]


def main() -> None:
    with engine.begin() as conn:
        for col, sql_type in MATCHES_COLUMNS:
            conn.execute(text(f"ALTER TABLE matches ADD COLUMN IF NOT EXISTS {col} {sql_type}"))
        for col, sql_type in UPCOMING_FIXTURES_COLUMNS:
            conn.execute(text(f"ALTER TABLE upcoming_fixtures ADD COLUMN IF NOT EXISTS {col} {sql_type}"))
    print(f"Ensured columns exist: {[f'matches.{c}' for c, _ in MATCHES_COLUMNS]}"
          f" {[f'upcoming_fixtures.{c}' for c, _ in UPCOMING_FIXTURES_COLUMNS]}")


if __name__ == "__main__":
    main()
