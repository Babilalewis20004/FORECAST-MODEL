"""Creates all tables. Single-user local app -- no Alembic; future schema
changes are a manual ALTER TABLE (or drop + re-run import_csv_to_postgres.py).
Note: this only creates tables that don't exist yet -- it does NOT add new
columns to an existing table. For that, see e.g. migrate_add_markets.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import orm_models  # noqa: F401 -- import registers models on Base.metadata
from app.db import Base, engine


def main() -> None:
    Base.metadata.create_all(engine)
    print(f"Created tables: {sorted(Base.metadata.tables.keys())}")


if __name__ == "__main__":
    main()
