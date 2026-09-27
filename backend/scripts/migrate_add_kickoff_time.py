"""One-off, idempotent migration: adds upcoming_fixtures.kickoff_time (the
full kickoff instant -- `date` is truncated to a day) needed by
odds_service.poll_closing_lines() to find fixtures inside its
10-15-minute-before-kickoff closing-line window. Nullable: existing rows
backfill on their next scrape. Same pattern as migrate_add_value.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.db import engine


def main() -> None:
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE upcoming_fixtures ADD COLUMN IF NOT EXISTS kickoff_time TIMESTAMPTZ"))
    print("Ensured column exists: upcoming_fixtures.kickoff_time")


if __name__ == "__main__":
    main()
