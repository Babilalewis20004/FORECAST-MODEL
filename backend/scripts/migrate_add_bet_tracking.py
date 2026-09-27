"""One-off, idempotent migration for live CLV tracking (Phase 2): creates the
bet_tracking table (see orm_models.BetTracking). create_all only creates
tables that don't exist yet, so this is safe to run against a database that
already has matches/predictions/etc. Same pattern as init_db.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import orm_models  # noqa: F401 -- import registers BetTracking on Base.metadata
from app.db import Base, engine


def main() -> None:
    Base.metadata.create_all(engine)
    print("Ensured table exists: bet_tracking")


if __name__ == "__main__":
    main()
