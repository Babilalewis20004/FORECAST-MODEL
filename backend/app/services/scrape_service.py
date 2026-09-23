"""API-Football (api-sports.io) fetch logic, ported from the archived
legacy/api_football_scraper.py -- same fetch/parse/budget/cache behavior,
but upserts into Postgres instead of rewriting matches.csv/upcoming_fixtures.csv.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.orm_models import Match, UpcomingFixture

BASE_URL = "https://v3.football.api-sports.io"
ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "downloaded_files" / "api_football_cache"

LEAGUES: dict[str, int] = {
    "Premier League": 39,
    "La Liga": 140,
    "EFL Championship": 40,
    "League One": 41,
    "League Two": 42,
    "Scottish Premiership": 179,
    "UEFA Nations League": 5,
    "International Friendlies": 10,
    "CONCACAF Nations League": 536,
    "Gulf Cup of Nations": 25,
    "UEFA U21 Championship": 38,
    "UEFA U21 Championship - Qualification": 850,
    "Africa Cup of Nations - Qualification": 36,
    "World Cup": 1,
    "Africa Cup of Nations": 6,
    "Copa America": 9,
}

SEASONS = [2022, 2023, 2024, 2025, 2026]
# International competitions don't run on a yearly club calendar: API-Football
# only has seasons for the editions that exist (e.g. AFCON 2027 qualifying is
# season 2027). Listing them avoids wasted requests on missing years.
LEAGUE_SEASONS: dict[str, list[int]] = {
    "UEFA Nations League": [2022, 2024, 2026],
    "CONCACAF Nations League": [2022, 2023, 2024, 2025],
    "Gulf Cup of Nations": [2023, 2024, 2026],
    "UEFA U21 Championship": [2023, 2025],
    "UEFA U21 Championship - Qualification": [2023, 2025, 2027],
    "Africa Cup of Nations - Qualification": [2023, 2025, 2027],
    "World Cup": [2026],
    "Africa Cup of Nations": [2025],  # the 2025 edition, played Dec 2025 - Jan 2026
    "Copa America": [2024],  # latest edition; the next is 2028
}
FINISHED_STATUSES = {"FT", "AET", "PEN"}
# Statuses that will never change again (cancelled, abandoned, awarded).
SETTLED_STATUSES = FINISHED_STATUSES | {"CANC", "ABD", "AWD", "WO"}
STAT_MAP = {
    "Total Shots": "shots",
    "Shots on Goal": "shots_on_target",
    "Corner Kicks": "corners",
}
DAILY_SAFETY_MARGIN = 3


class Budget:
    def __init__(self) -> None:
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
        limit = self.limit_minute or 10
        return (60.0 / limit) * 1.15


def _session() -> requests.Session:
    if not settings.api_football_key:
        raise RuntimeError("API_FOOTBALL_KEY not set in backend/.env")
    s = requests.Session()
    s.headers.update({"x-apisports-key": settings.api_football_key})
    return s


def _get(session: requests.Session, budget: Budget, path: str, cache_file: Path | None, **params) -> dict:
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


def _season_complete(cache_file: Path) -> bool:
    """A cached season is final only once none of its fixtures can still
    change -- otherwise upcoming fixtures and new results would never be
    picked up after the first scrape."""
    if not cache_file.exists():
        return False
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    return all(item["fixture"]["status"]["short"] in SETTLED_STATUSES for item in data["response"])


def _fetch_league_season(session: requests.Session, budget: Budget, league_name: str, league_id: int, season: int) -> list[dict]:
    cache_file = CACHE_DIR / f"fixtures_{league_id}_{season}.json"
    if not _season_complete(cache_file):
        cache_file.unlink(missing_ok=True)
    data = _get(session, budget, "fixtures", cache_file, league=league_id, season=season)

    rows = []
    for item in data["response"]:
        fx, teams, goals = item["fixture"], item["teams"], item["goals"]
        # Every market settles on 90 minutes, but "goals" includes extra time
        # -- knockout ties (Nations League finals, Gulf Cup, U21) use the
        # regular-time score instead.
        fulltime = (item.get("score") or {}).get("fulltime") or {}
        if fx["status"]["short"] in ("AET", "PEN") and fulltime.get("home") is not None:
            goals = fulltime
        rows.append(
            {
                "fixture_id": fx["id"],
                "date": fx["date"][:10],
                "status": fx["status"]["short"],
                "league": league_name,
                "home_team": teams["home"]["name"],
                "away_team": teams["away"]["name"],
                "home_goals": goals["home"],
                "away_goals": goals["away"],
            }
        )
    return rows


def _upsert_matches(db: Session, rows: list[dict]) -> int:
    finished = [r for r in rows if r["status"] in FINISHED_STATUSES]
    for r in finished:
        values = {
            "fixture_id": r["fixture_id"],
            "date": r["date"],
            "league": r["league"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_goals": int(r["home_goals"]),
            "away_goals": int(r["away_goals"]),
        }
        stmt = pg_insert(Match).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["league", "date", "home_team", "away_team"],
            set_={
                "fixture_id": stmt.excluded.fixture_id,
                "home_goals": stmt.excluded.home_goals,
                "away_goals": stmt.excluded.away_goals,
            },
        )
        db.execute(stmt)
    db.commit()
    return len(finished)


def _upsert_upcoming(db: Session, rows: list[dict]) -> int:
    upcoming = [r for r in rows if r["status"] not in FINISHED_STATUSES]
    for r in upcoming:
        values = {
            "fixture_id": r["fixture_id"],
            "date": r["date"],
            "league": r["league"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "status": r["status"],
        }
        stmt = pg_insert(UpcomingFixture).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[UpcomingFixture.fixture_id],
            set_={
                "date": stmt.excluded.date,
                "status": stmt.excluded.status,
            },
        )
        db.execute(stmt)
    db.commit()
    return len(upcoming)


def scrape_fixtures(leagues: list[str] | None, job=None) -> dict:
    session = _session()
    budget = Budget()
    db = SessionLocal()
    targets = {name: LEAGUES[name] for name in (leagues or LEAGUES.keys())}

    summary = {}
    try:
        for league_name, league_id in targets.items():
            n_finished = n_upcoming = 0
            for season in LEAGUE_SEASONS.get(league_name, SEASONS):
                if job is not None:
                    job.detail = f"fetching {league_name} {season}-{season + 1}..."
                try:
                    rows = _fetch_league_season(session, budget, league_name, league_id, season)
                except RuntimeError:
                    continue
                n_finished += _upsert_matches(db, rows)
                n_upcoming += _upsert_upcoming(db, rows)
                if budget.exhausted():
                    break
            summary[league_name] = {"matches_upserted": n_finished, "upcoming_upserted": n_upcoming}
            if budget.exhausted():
                break
    finally:
        db.close()
    return summary


def _fetch_fixture_stats(session: requests.Session, budget: Budget, fixture_id: int, home_team: str) -> dict | None:
    cache_file = CACHE_DIR / f"stats_{fixture_id}.json"
    data = _get(session, budget, "fixtures/statistics", cache_file, fixture=fixture_id)
    if not data["response"]:
        return None

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


def scrape_stats(job=None) -> dict:
    session = _session()
    budget = Budget()
    db = SessionLocal()
    stat_cols = [f"{side}_{suffix}" for side in ("home", "away") for suffix in ("shots", "shots_on_target", "corners", "cards", "xg")]

    try:
        needs_stats = (
            db.query(Match)
            .filter(Match.fixture_id.isnot(None))
            .filter(
                (Match.home_shots.is_(None))
                | (Match.away_shots.is_(None))
                | (Match.home_xg.is_(None))
                | (Match.away_xg.is_(None))
            )
            .all()
        )
        fetched = 0
        for m in needs_stats:
            if budget.exhausted():
                break
            if job is not None:
                job.detail = f"stats backfill: {fetched} fetched, fixture {m.fixture_id}"
            try:
                stats = _fetch_fixture_stats(session, budget, int(m.fixture_id), m.home_team)
            except requests.HTTPError:
                continue
            if stats is None:
                continue
            for c, v in stats.items():
                setattr(m, c, v)
            fetched += 1
            if fetched % 200 == 0:
                db.commit()
        db.commit()
        return {"matches_needing_stats": len(needs_stats), "fetched": fetched, "stat_cols": stat_cols}
    finally:
        db.close()


def run_scrape(leagues: list[str] | None, include_stats: bool, job=None) -> dict:
    result = {"fixtures": scrape_fixtures(leagues, job=job)}
    if include_stats:
        result["stats"] = scrape_stats(job=job)
    return result
