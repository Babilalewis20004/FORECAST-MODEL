"""One-off, idempotent migration: adds predictions.extra_markets (team totals,
1X2/O-U combined with BTTS, odd/even total -- see orm_models.Prediction and
football_goal_range_model.extra_market_probabilities). Nullable: existing
rows backfill on their next retrain. Same pattern as migrate_add_correct_score.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.db import engine


def main() -> None:
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS extra_markets JSONB"))
    print("Ensured column exists: predictions.extra_markets")


if __name__ == "__main__":
    main()
