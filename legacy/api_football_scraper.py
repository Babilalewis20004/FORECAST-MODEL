"""
API-Football (api-sports.io) scraper.

Fills the gap left by FBref/soccerdata (which only covers Premier League and
La Liga): pulls match history + upcoming fixtures for all six leagues below,
and can separately backfill shots/corners/cards/xG for matches that already
have a fixture_id.

Two commands:

    python api_football_scraper.py fixtures
        Pulls full match history (+ next unplayed fixtures) for every
        configured league/season and writes/updates matches.csv and
        upcoming_fixtures.csv. One request per league-season -- cheap.

    python api_football_scraper.py stats
        Backfills home_shots/away_shots/..._corners/..._cards/..._xg columns
        onto rows in matches.csv that have a fixture_id and are missing
        them. One request PER MATCH, so it's budget-limited (free plan:
        100 requests/day) and resumable -- it caches every response on disk
        and only spends budget on matches it doesn't already have cached.
        Re-run it daily until it reports nothing left to fetch.

Auth: reads API_FOOTBALL_KEY from a .env file next to this script (or from
the environment directly). Get a free key at api-football.com.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://v3.football.api-sports.io"
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "downloaded_files" / "api_football_cache"
MATCHES_CSV = HERE / "matches.csv"
UPCOMING_CSV = HERE / "upcoming_fixtures.csv"

# Maps this project's league names (as used in matches.csv / football_goal_range_model.py)
# to API-Football league ids, confirmed live against the account's own /leagues response.
LEAGUES: dict[str, int] = {
    "Premier League": 39,
    "La Liga": 140,
    "EFL Championship": 40,
    "League One": 41,
    "League Two": 42,
    "Scottish Premiership": 179,
}

SEASONS = [2022, 2023, 2024, 2025, 2026]  # season = start year, e.g. 2023 -> 2023-2024
# Free plan only granted access to 2022-2024 (confirmed live: requesting
# 2025 returned {"plan": "Free plans do not have access to this season,
# try from 2022 to 2024."}). Paid plans (Pro+) lift that restriction, so
# 2025 (the current 2025-2026 season) is included once upgraded -- that's
# also where genuinely upcoming/unplayed (status "NS") fixtures live.

FINISHED_STATUSES = {"FT", "AET", "PEN"}

# Stat "type" strings as returned by /fixtures/statistics -> our column suffix.
STAT_MAP = {
    "Total Shots": "shots",
    "Shots on Goal": "shots_on_target",
    "Corner Kicks": "corners",
}

DAILY_SAFETY_MARGIN = 3   # stop `stats` once remaining-day budget drops to this


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def _session() -> requests.Session:
    _load_env(HERE / ".env")
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        sys.exit("API_FOOTBALL_KEY not set (expected in .env next to this script).")
    s = requests.Session()
    s.headers.update({"x-apisports-key": key})
    return s


class Budget:
    """Tracks the account's request budget (daily + per-minute) from response headers."""

    def __init__(self):
        self.remaining_day: int | None = None
        self.limit_minute: int | None = None

    def update(self, resp: requests.Response) -> None:
        v = resp.headers.get("x-ratelimit-requests-remaining")
        if v is not None:
            self.remaining_day = int(v)
        lim = resp.headers.get("x-ratelimit-limit")
        if lim is not None:
            self.limit_minute = int(lim)

    def exhausted(self, margin: int = DAILY_SAFETY_MARGIN) -> bool:
        return self.remaining_day is not None and self.remaining_day <= margin

    def sleep_interval(self) -> float:
        """Seconds to sleep between requests to stay safely under the account's
        per-minute limit -- read from live headers rather than hardcoded, since
        it differs a lot between the free plan (10/min) and paid plans (300/min)."""
        limit = self.limit_minute or 10  # unknown yet -> assume the free-plan limit
        return (60.0 / limit) * 1.15  # small safety margin


def _get(session: requests.Session, budget: Budget, path: str, cache_file: Path | None, **params):
    """GET with on-disk caching -- a cache hit costs nothing and doesn't touch budget."""
    if cache_file is not None and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    resp = session.get(f"{BASE_URL}/{path}", params=params, timeout=30)
    resp.raise_for_status()
    budget.update(resp)
    data = resp.json()
    if data.get("errors"):
        raise RuntimeError(f"API error for {path} {params}: {data['errors']}")

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data), encoding="utf-8")

    time.sleep(budget.sleep_interval())
    return data


# ---------------------------------------------------------------------------
# fixtures command
# ---------------------------------------------------------------------------

def _fetch_league_season(session: requests.Session, budget: Budget, league_name: str, league_id: int, season: int) -> list[dict]:
    cache_file = CACHE_DIR / f"fixtures_{league_id}_{season}.json"
    data = _get(session, budget, "fixtures", cache_file, league=league_id, season=season)

    rows = []
    for item in data["response"]:
        fx, teams, goals = item["fixture"], item["teams"], item["goals"]
        rows.append({
            "fixture_id": fx["id"],
            "date": fx["date"][:10],
            "status": fx["status"]["short"],
            "league": league_name,
            "home_team": teams["home"]["name"],
            "away_team": teams["away"]["name"],
            "home_goals": goals["home"],
            "away_goals": goals["away"],
        })
    return rows


def cmd_fixtures(args) -> None:
    session = _session()
    budget = Budget()

    all_rows: list[dict] = []
    stop = False
    for league_name, league_id in LEAGUES.items():
        if stop:
            break
        for season in SEASONS:
            print(f"fetching {league_name} {season}-{season + 1}...", end=" ", flush=True)
            try:
                rows = _fetch_league_season(session, budget, league_name, league_id, season)
            except RuntimeError as e:
                print(f"skipped ({e})")
                continue
            print(f"{len(rows)} fixtures (budget remaining: {budget.remaining_day})")
            all_rows.extend(rows)
            if budget.exhausted():
                print("Daily request budget nearly exhausted -- stopping early. Re-run tomorrow to continue.")
                stop = True
                break

    df = pd.DataFrame(all_rows)
    finished = df[df["status"].isin(FINISHED_STATUSES)].copy()
    upcoming = df[~df["status"].isin(FINISHED_STATUSES)].copy()

    finished["home_goals"] = finished["home_goals"].astype(int)
    finished["away_goals"] = finished["away_goals"].astype(int)
    finished = finished[["fixture_id", "date", "league", "home_team", "away_team", "home_goals", "away_goals"]]
    finished = finished.sort_values("date")

    key_cols = ["league", "date", "home_team", "away_team"]
    if MATCHES_CSV.exists():
        existing = pd.read_csv(MATCHES_CSV)
        if "fixture_id" not in existing.columns:
            existing["fixture_id"] = pd.NA
        # Union by (league, date, home_team, away_team): existing rows the
        # API doesn't cover (e.g. the 2025-2026 season, outside the free
        # plan's [2022, 2024] window) are kept as-is; where the same match
        # exists in both, the freshly-fetched API row wins (adds fixture_id).
        combined = pd.concat([existing, finished], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = finished

    combined = combined.sort_values(["league", "date"]).reset_index(drop=True)
    combined.to_csv(MATCHES_CSV, index=False)
    print(f"\nWrote {len(combined)} historical matches to {MATCHES_CSV}")

    upcoming = upcoming[["fixture_id", "date", "league", "home_team", "away_team", "status"]].sort_values("date")
    upcoming.to_csv(UPCOMING_CSV, index=False)
    print(f"Wrote {len(upcoming)} upcoming fixtures to {UPCOMING_CSV}")


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------

def _fetch_fixture_stats(session: requests.Session, budget: Budget, fixture_id: int, home_team: str) -> dict | None:
    cache_file = CACHE_DIR / f"stats_{fixture_id}.json"
    data = _get(session, budget, "fixtures/statistics", cache_file, fixture=fixture_id)
    if not data["response"]:
        return None  # stats not available for this fixture (e.g. too old)

    out: dict[str, float | None] = {}
    for block in data["response"]:
        side = "home" if block["team"]["name"] == home_team else "away"
        values = {s["type"]: s["value"] for s in block["statistics"]}
        for api_type, suffix in STAT_MAP.items():
            v = values.get(api_type)
            out[f"{side}_{suffix}"] = float(v) if v is not None else None
        yellow = values.get("Yellow Cards") or 0
        red = values.get("Red Cards") or 0
        out[f"{side}_cards"] = float(yellow) + float(red)
        xg = values.get("expected_goals")
        out[f"{side}_xg"] = float(xg) if xg is not None else None
    return out


def cmd_stats(args) -> None:
    if not MATCHES_CSV.exists():
        sys.exit(f"{MATCHES_CSV} not found -- run `fixtures` first.")

    df = pd.read_csv(MATCHES_CSV)
    if "fixture_id" not in df.columns:
        sys.exit("matches.csv has no fixture_id column -- run `fixtures` first so matches are traceable to API-Football fixtures.")

    stat_cols = [f"{side}_{suffix}" for side in ("home", "away") for suffix in ("shots", "shots_on_target", "corners", "cards", "xg")]
    for c in stat_cols:
        if c not in df.columns:
            df[c] = pd.NA

    needs_stats = df[df["fixture_id"].notna() & df[stat_cols].isna().any(axis=1)]
    print(f"{len(needs_stats)} matches missing stats (of {len(df)} total).")
    if needs_stats.empty:
        return

    session = _session()
    budget = Budget()
    fetched = 0

    for idx, row in needs_stats.iterrows():
        if budget.exhausted():
            print("Daily request budget nearly exhausted -- stopping early. Re-run tomorrow to continue.")
            break
        try:
            stats = _fetch_fixture_stats(session, budget, int(row["fixture_id"]), row["home_team"])
        except requests.HTTPError as e:
            print(f"  fixture {row['fixture_id']}: HTTP error {e}, skipping")
            continue
        if stats is None:
            # No stats available upstream. Nothing to write, but the response
            # is now cached on disk, so re-checking it next run costs no budget.
            continue
        for c, v in stats.items():
            df.loc[idx, c] = v
        fetched += 1
        if fetched % 200 == 0:
            df.to_csv(MATCHES_CSV, index=False)
            print(f"  ...{fetched} fetched, progress saved (budget remaining: {budget.remaining_day})")

    df.to_csv(MATCHES_CSV, index=False)
    print(f"\nFetched stats for {fetched} matches this run. Updated {MATCHES_CSV}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fixtures", help="pull match history + upcoming fixtures").set_defaults(func=cmd_fixtures)
    sub.add_parser("stats", help="backfill shots/corners/cards/xg for matches with a fixture_id").set_defaults(func=cmd_stats)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
