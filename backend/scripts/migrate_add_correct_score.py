"""One-off, idempotent migration: adds predictions.correct_score (top-N
exact-scoreline probabilities, Model A only -- see orm_models.Prediction and
football_goal_range_model.correct_score_probabilities). Nullable: existing
rows backfill on their next retrain. Same pattern as migrate_add_value.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.db import engine


def main() -> None:
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS correct_score JSONB"))
    print("Ensured column exists: predictions.correct_score")


if __name__ == "__main__":
    main()
