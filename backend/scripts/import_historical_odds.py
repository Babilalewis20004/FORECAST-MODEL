"""Backfills match_odds with Bet365 opening/closing odds from
football-data.co.uk for every domestic-league match in the DB (seasons
2022/23 onwards). Safe to re-run: upserts on (match_id, bookmaker). The
regular scrape keeps the current season up to date afterwards.

  python scripts/init_db.py                 # creates match_odds if missing
  python scripts/import_historical_odds.py  # then retrain to use it
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import SessionLocal
from app.services.odds_history import sync_match_odds


def main() -> None:
    db = SessionLocal()
    try:
        for league, s in sync_match_odds(db).items():
            print(f"{league:<22} {s['with_odds']:>5} / {s['matches']:>5} matches with odds "
                  f"({s['odds_rows_unmatched']} odds rows unmatched)")
    finally:
        db.close()


if __name__ == "__main__":
    main()
