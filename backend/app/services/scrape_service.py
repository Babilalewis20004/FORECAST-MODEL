"""API-Football (api-sports.io) fetch logic, ported from the archived
legacy/api_football_scraper.py -- same fetch/parse/budget/cache behavior,
but upserts into Postgres instead of rewriting matches.csv/upcoming_fixtures.csv.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from itertools import groupby
from pathlib import Path

import requests
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.ml.football_goal_range_model import OU_LINES
from app.orm_models import Match, Prediction, UpcomingFixture
from app.services.model_service import ModelNotTrainedError, generate_predictions, model_group
from app.services.odds_history import current_season_start, sync_match_odds

BASE_URL = "https://v3.football.api-sports.io"
ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "downloaded_files" / "api_football_cache"

LEAGUES: dict[str, int] = {
    "Premier League": 39,
    "La Liga": 140,
    "Bundesliga": 78,
    "Serie A": 135,
    "Ligue 1": 61,
    "Primeira Liga": 94,
    "Eredivisie": 88,
    "Liga Profesional Argentina": 128,
    "Süper Lig": 203,
    "Jupiler Pro League": 144,
    "Super League 1": 197,
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

# /odds -> value_engine outcome keys, per API-Football bet name.
ODDS_MARKETS: dict[str, dict[str, str]] = {
    "Match Winner": {"Home": "home_win", "Draw": "draw", "Away": "away_win"},
    "Goals Over/Under": {
        f"{side.capitalize()} {line}": f"{side}_{line}" for line in OU_LINES for side in ("over", "under")
    },
    "Both Teams Score": {"Yes": "btts_yes", "No": "btts_no"},
}
# Bet365 first: Model B's market features and the shrinkage weights were
# fitted on Bet365's historical prices, so serving on the same book avoids
# train/serve skew. Pinnacle (sharpest) is the fallback.
PREFERRED_BOOKMAKERS = ("Bet365", "Pinnacle", "Marathonbet", "Betfair")
# The books Phase 2's CLV tracking treats as "sharp" -- closing lines are
# only ever captured from these, regardless of which book a bet was "placed"
# at, since a soft book's closing price is a poor benchmark for fair value.
SHARP_BOOKMAKERS = ("Pinnacle", "Bet365")
# API-Football publishes pre-match odds roughly 1-14 days out; nearer
# kickoff is closer to the closing price and costs fewer requests.
ODDS_DAYS_AHEAD = 3


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


def _parse_kickoff(date_str: str) -> datetime:
    """API-Football's fixture date, e.g. "2024-05-19T15:00:00+00:00" (or a
    "Z" suffix) -- the full instant, unlike the day-only `date` column."""
    if date_str.endswith("Z"):
        date_str = date_str[:-1] + "+00:00"
    return datetime.fromisoformat(date_str)


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
                "kickoff_time": _parse_kickoff(fx["date"]),
                "status": fx["status"]["short"],
                "league": league_name,
                "home_team": teams["home"]["name"],
                "away_team": teams["away"]["name"],
                "home_goals": goals["home"],
                "away_goals": goals["away"],
                "referee": fx.get("referee"),
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
            "referee": r.get("referee"),
        }
        stmt = pg_insert(Match).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["league", "date", "home_team", "away_team"],
            set_={
                "fixture_id": stmt.excluded.fixture_id,
                "home_goals": stmt.excluded.home_goals,
                "away_goals": stmt.excluded.away_goals,
                "referee": stmt.excluded.referee,
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
            "kickoff_time": r.get("kickoff_time"),
            "league": r["league"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "status": r["status"],
            "referee": r.get("referee"),
        }
        stmt = pg_insert(UpcomingFixture).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[UpcomingFixture.fixture_id],
            set_={
                "date": stmt.excluded.date,
                "kickoff_time": stmt.excluded.kickoff_time,
                "status": stmt.excluded.status,
                "referee": stmt.excluded.referee,
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
            if job is not None:
                job.detail = f"Scraping {league_name} fixtures..."
            n_finished = n_upcoming = n_rows = 0
            for season in LEAGUE_SEASONS.get(league_name, SEASONS):
                try:
                    rows = _fetch_league_season(session, budget, league_name, league_id, season)
                except RuntimeError:
                    continue
                n_rows += len(rows)
                n_finished += _upsert_matches(db, rows)
                n_upcoming += _upsert_upcoming(db, rows)
                if budget.exhausted():
                    break
            summary[league_name] = {"matches_upserted": n_finished, "upcoming_upserted": n_upcoming}
            if job is not None:
                job.log_line(f"Scraping {league_name} fixtures… {n_rows} matches")
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
        poss = values.get("Ball Possession")
        out[f"{side}_possession"] = float(poss.rstrip("%")) if isinstance(poss, str) and poss.rstrip("%") else None
    return out


def scrape_stats(job=None) -> dict:
    session = _session()
    budget = Budget()
    db = SessionLocal()
    stat_cols = [
        f"{side}_{suffix}"
        for side in ("home", "away")
        for suffix in ("shots", "shots_on_target", "corners", "cards", "xg", "possession")
    ]

    try:
        needs_stats = (
            db.query(Match)
            .filter(Match.fixture_id.isnot(None))
            .filter(
                (Match.home_shots.is_(None))
                | (Match.away_shots.is_(None))
                | (Match.home_xg.is_(None))
                | (Match.away_xg.is_(None))
                | (Match.home_possession.is_(None))
                | (Match.away_possession.is_(None))
            )
            .order_by(Match.league)
            .all()
        )
        fetched = 0
        # Grouped by league so progress reads as one clean line per league
        # ("Backfilling Premier League match stats… done") instead of one
        # line per fixture -- the per-fixture fetch loop underneath is
        # unchanged, only what's surfaced to job.log_line() is coarser.
        for league_name, group in groupby(needs_stats, key=lambda m: m.league):
            if job is not None:
                job.detail = f"Backfilling {league_name} match stats..."
            if budget.exhausted():
                break
            for m in group:
                if budget.exhausted():
                    break
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
            if job is not None:
                job.log_line(f"Backfilling {league_name} match stats… done")
        db.commit()
        return {"matches_needing_stats": len(needs_stats), "fetched": fetched, "stat_cols": stat_cols}
    finally:
        db.close()


def parse_odds(item: dict) -> tuple[str, dict[str, float], dict[str, dict[str, float]]] | None:
    """(best bookmaker, its flat odds, sharp-book odds) from one /odds
    response item. "best" is the first PREFERRED_BOOKMAKERS entry that prices
    any market we model, else the bookmaker pricing the most outcomes --
    unchanged from before Phase 2, still what's served for Model B's market
    features. sharp_odds maps any of SHARP_BOOKMAKERS (Pinnacle, Bet365)
    present in this fixture to their own flat odds, independent of which book
    is "best" -- CLV's closing line is always measured against a sharp book,
    regardless of where the bet was placed. Keys match value_engine.BOOKS."""
    parsed = []
    for bm in item.get("bookmakers", []):
        odds: dict[str, float] = {}
        for bet in bm.get("bets", []):
            names = ODDS_MARKETS.get(bet.get("name"))
            if names is None:
                continue
            for v in bet.get("values", []):
                key = names.get(str(v.get("value")))
                if key is not None:
                    try:
                        odds[key] = float(v["odd"])
                    except (KeyError, TypeError, ValueError):
                        continue
        if odds:
            parsed.append((bm.get("name", "unknown"), odds))
    if not parsed:
        return None

    sharp_odds = {name: odds for name, odds in parsed if name in SHARP_BOOKMAKERS}
    for name in PREFERRED_BOOKMAKERS:
        for bm_name, odds in parsed:
            if bm_name == name:
                return bm_name, odds, sharp_odds
    best_name, best_odds = max(parsed, key=lambda p: len(p[1]))
    return best_name, best_odds, sharp_odds


def scrape_odds(leagues: list[str] | None, job=None, days_ahead: int = ODDS_DAYS_AHEAD) -> dict:
    """Latest pre-match odds for every predicted, not-started fixture in the
    next `days_ahead` days, stored on its prediction row. One uncached
    request per fixture (odds move), nearest kickoff first so the budget goes
    where the odds are closest to closing. Odds are model inputs (Model B's
    market features, shrinkage toward the market), so every affected model
    group is then re-predicted, which also re-derives the value fields."""
    # Deferred: odds_service imports Budget/_session/_get/parse_odds/
    # SHARP_BOOKMAKERS from this module at load time, so importing it back
    # here at module scope would be circular. By call time both modules are
    # already fully loaded, so this is just a normal, cheap cached import.
    from app.services.odds_service import capture_placed_snapshot

    session = _session()
    budget = Budget()
    db = SessionLocal()
    today = date.today()
    try:
        q = (
            db.query(Prediction, UpcomingFixture)
            .join(UpcomingFixture, UpcomingFixture.fixture_id == Prediction.fixture_id)
            .filter(UpcomingFixture.status == "NS")
            .filter(UpcomingFixture.date >= today, UpcomingFixture.date <= today + timedelta(days=days_ahead))
        )
        if leagues:
            q = q.filter(UpcomingFixture.league.in_(leagues))
        targets = q.order_by(UpcomingFixture.date).all()

        fetched = with_odds = 0
        groups: set[str] = set()
        for pred, fx in targets:
            if budget.exhausted():
                break
            if job is not None:
                # Per-fixture -- too fine-grained for the console (could be
                # hundreds of lines), so this only updates the coarse
                # current-phase label, not job.log.
                job.detail = f"Fetching odds… {fetched}/{len(targets)} fixtures, {fx.home_team} vs {fx.away_team}"
            try:
                data = _get(session, budget, "odds", None, fixture=fx.fixture_id)
            except (requests.HTTPError, RuntimeError):
                continue
            fetched += 1
            parsed = parse_odds(data["response"][0]) if data.get("response") else None
            if parsed is None:
                continue
            bm_name, best_odds, _sharp_odds = parsed
            pred.odds_bookmaker, pred.bookmaker_odds = bm_name, best_odds
            pred.odds_fetched_at = datetime.now(timezone.utc)
            # Freeze this fixture/market/selection's price the first time the
            # *current* model probabilities + these odds flag it as a value
            # bet -- generate_predictions() below will shortly re-derive
            # is_value_bet from the same odds anyway, so this just captures
            # the moment first, before it can be overwritten by a later odds
            # move or retrain.
            capture_placed_snapshot(db, pred, bm_name, best_odds)
            groups.add(model_group(fx.league))
            with_odds += 1
            if with_odds % 50 == 0:
                db.commit()
        db.commit()
        if job is not None:
            job.log_line(f"Fetching odds… {with_odds}/{len(targets)} fixtures updated")

        for group in sorted(groups):
            if job is not None:
                job.detail = f"Re-predicting {group} with market inputs..."
            try:
                generate_predictions(db, group)
            except ModelNotTrainedError:
                continue
            if job is not None:
                job.log_line(f"Re-predicting {group} with market inputs… done")
        n_value = (
            db.query(Prediction)
            .filter(Prediction.fixture_id.in_([fx.fixture_id for _, fx in targets]), Prediction.is_value_bet.is_(True))
            .count()
        ) if targets else 0
        return {"fixtures_in_window": len(targets), "fetched": fetched, "with_odds": with_odds, "value_bets": n_value}
    finally:
        db.close()


def run_scrape(leagues: list[str] | None, include_stats: bool, job=None, include_odds: bool = False) -> dict:
    result = {"fixtures": scrape_fixtures(leagues, job=job)}
    # Historical odds for newly finished matches (model inputs at the next
    # retrain) -- football-data.co.uk, not the API-Football budget.
    if job is not None:
        job.detail = "Syncing historical odds (football-data.co.uk)..."
    db = SessionLocal()
    try:
        result["historical_odds"] = sync_match_odds(db, leagues, seasons=[current_season_start()])
    except requests.RequestException as exc:
        result["historical_odds"] = f"skipped: {exc}"
    finally:
        db.close()
    if job is not None:
        job.log_line("Syncing historical odds (football-data.co.uk)… done")
    if include_stats:
        result["stats"] = scrape_stats(job=job)
    if include_odds:
        result["odds"] = scrape_odds(leagues, job=job)
    return result
