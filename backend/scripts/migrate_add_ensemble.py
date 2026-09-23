"""One-off, idempotent migration for the ensemble/calibration layer: adds
predictions.raw_probabilities, predictions.ensemble_weights and
model_runs.metrics (all JSONB, nullable) without touching existing data.
Same pattern as migrate_add_markets.py; safe on a fresh database and safe
to re-run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.db import engine

NEW_COLUMNS = [
    ("predictions", "raw_probabilities"),
    ("predictions", "ensemble_weights"),
    ("model_runs", "metrics"),
]


def main() -> None:
    with engine.begin() as conn:
        for table, col in NEW_COLUMNS:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} JSONB"))
    print(f"Ensured columns exist: {[f'{t}.{c}' for t, c in NEW_COLUMNS]}")


if __name__ == "__main__":
    main()
