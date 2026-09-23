"""
Runs the trained model over fixtures in upcoming_fixtures.csv (produced by
`api_football_scraper.py fixtures`) and writes goal-range probabilities for
each to upcoming_predictions.csv.

Fits one LeagueGoalModel per league (on its full match history -- no
holdout here, since this is for real predictions, not calibration
checking), then predicts every upcoming fixture in that league that falls
within --start/--end (inclusive; both optional, default is no filter).

    python predict_upcoming.py                                  # all upcoming fixtures
    python predict_upcoming.py --start 2026-10-01 --end 2026-10-31  # October 2026 only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from football_goal_range_model import (
    XG_RICH_LEAGUES,
    LeagueGoalModel,
    predict_fixture,
    prepare_fixture_features,
)

HERE = Path(__file__).resolve().parent
MATCHES_CSV = HERE / "matches.csv"
UPCOMING_CSV = HERE / "upcoming_fixtures.csv"
OUT_CSV = HERE / "upcoming_predictions.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=str, default=None, help="only predict fixtures on/after this date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="only predict fixtures on/before this date (YYYY-MM-DD)")
    args = parser.parse_args()

    matches = pd.read_csv(MATCHES_CSV, parse_dates=["date"])
    upcoming = pd.read_csv(UPCOMING_CSV, parse_dates=["date"])
    if args.start:
        upcoming = upcoming[upcoming["date"] >= pd.Timestamp(args.start)]
    if args.end:
        upcoming = upcoming[upcoming["date"] <= pd.Timestamp(args.end)]

    rows = []
    for league_name in sorted(upcoming["league"].unique()):
        # Train on the league's FULL history regardless of --start/--end --
        # only which fixtures get predicted is filtered, not what they're trained on.
        league_history = matches[matches["league"] == league_name]
        league_upcoming = upcoming[upcoming["league"] == league_name].sort_values("date")

        if league_history.empty:
            print(f"[{league_name}] no historical matches in matches.csv -- skipping {len(league_upcoming)} fixtures")
            continue

        tier = "rich" if league_name in XG_RICH_LEAGUES else "lower"
        model = LeagueGoalModel(league=league_name, tier=tier)
        model.fit(league_history, test_size=0.0)

        print(f"[{league_name}] trained on {len(league_history)} matches, tier='{tier}' -- predicting {len(league_upcoming)} fixtures...")

        n_ok = n_failed = 0
        for _, fx in league_upcoming.iterrows():
            try:
                row = prepare_fixture_features(model, league_history, fx["home_team"], fx["away_team"], fx["date"])
                result = predict_fixture(model, row)
            except (ValueError, IndexError) as e:
                n_failed += 1
                continue
            n_ok += 1
            rows.append({
                "fixture_id": fx["fixture_id"],
                "date": fx["date"].date(),
                "league": league_name,
                "home_team": fx["home_team"],
                "away_team": fx["away_team"],
                "lambda_home": result["lambda_home"],
                "lambda_away": result["lambda_away"],
                **{f"p_{label}": round(p, 4) for label, p in result["range_probabilities"].items()},
            })
        print(f"  {n_ok} predicted, {n_failed} skipped (no usable history, e.g. brand-new team)")

    out = pd.DataFrame(rows)
    out = out.sort_values(["date", "league"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out)} fixture predictions to {OUT_CSV}")


if __name__ == "__main__":
    main()
