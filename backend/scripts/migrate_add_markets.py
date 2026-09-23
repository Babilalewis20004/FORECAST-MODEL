"""One-off, idempotent migration: adds the match_result/over_under/btts
JSONB columns to an EXISTING predictions table without touching any other
data. Base.metadata.create_all() (see init_db.py) only creates missing
TABLES, not missing columns on a table that already exists -- this project
has no Alembic, so a new column on an existing table needs a script like
this one instead.

Safe to run against a fresh database too (ADD COLUMN IF NOT EXISTS is a
no-op if init_db.py already created the columns), and safe to re-run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.db import engine

NEW_COLUMNS = ["match_result", "over_under", "btts"]


def main() -> None:
    with engine.begin() as conn:
        for col in NEW_COLUMNS:
            conn.execute(text(f"ALTER TABLE predictions ADD COLUMN IF NOT EXISTS {col} JSONB"))
    print(f"Ensured columns exist on predictions: {NEW_COLUMNS}")


if __name__ == "__main__":
    main()
