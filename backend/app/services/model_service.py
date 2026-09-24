from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.ml.ensemble import LeagueEnsemble, flat_from_prediction, market_flat_from_odds, nest, train_ensemble
from app.ml.features import market_features_from_odds
from app.ml.football_goal_range_model import (
    GOALS_ONLY_LEAGUES,
    INTERNATIONAL_COMPETITIONS,
    INTERNATIONAL_GROUP,
    XG_RICH_LEAGUES,
    LeagueGoalModel,
    compute_team_state,
    predict_fixture_fast,
)
from app.ml.value_engine import assess_value, flat_model_probabilities
from app.orm_models import Match, MatchOdds, ModelRun, Prediction, UpcomingFixture

# Model B's market features and the shrinkage fit were trained on this
# bookmaker's historical prices (see scripts/import_historical_odds.py).
HISTORICAL_ODDS_BOOKMAKER = "bet365"
from app.schemas import LABEL_TO_FIELD

MODEL_STORE = Path(__file__).resolve().parents[1] / "ml" / "model_store"
MODEL_STORE.mkdir(parents=True, exist_ok=True)

# Competitions fixtures are listed/filtered by, vs. the model groups actually
# trained: each domestic league is its own group, and every international
# competition shares INTERNATIONAL_GROUP (see football_goal_range_model).
ALL_LEAGUES = sorted(XG_RICH_LEAGUES | GOALS_ONLY_LEAGUES) + sorted(INTERNATIONAL_COMPETITIONS)
ALL_MODEL_GROUPS = sorted(XG_RICH_LEAGUES | GOALS_ONLY_LEAGUES) + [INTERNATIONAL_GROUP]

_model_cache: dict[str, LeagueGoalModel] = {}
_ensemble_cache: dict[str, LeagueEnsemble] = {}


class ModelNotTrainedError(RuntimeError):
    pass


def league_slug(league: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", league.lower()).strip("_")


def league_tier(league: str) -> str:
    return "rich" if league in XG_RICH_LEAGUES else "lower"


def model_group(league: str) -> str:
    """The trained model (and model_runs row) that serves a competition."""
    return INTERNATIONAL_GROUP if league in INTERNATIONAL_COMPETITIONS else league


def group_competitions(group: str) -> list[str]:
    return sorted(INTERNATIONAL_COMPETITIONS) if group == INTERNATIONAL_GROUP else [group]


def _model_path(league: str) -> Path:
    return MODEL_STORE / f"{league_slug(league)}.joblib"


def _ensemble_path(league: str) -> Path:
    # Separate file from Model A's so joblib files from before the ensemble
    # shipped still load as plain LeagueGoalModels.
    return MODEL_STORE / f"{league_slug(league)}_ensemble.joblib"


def load_cached_models() -> None:
    """Called once at startup: populate the in-memory caches from whatever
    .joblib files already exist on disk, so the server never forces a
    retrain just because the process restarted."""
    for league in ALL_MODEL_GROUPS:
        for path, cache in ((_model_path(league), _model_cache), (_ensemble_path(league), _ensemble_cache)):
            if path.exists():
                try:
                    cache[league] = joblib.load(path)
                except Exception:  # noqa: BLE001 -- a corrupt/stale file just means "not cached"
                    continue


def get_ensemble(league: str) -> LeagueEnsemble | None:
    """None if the league hasn't been trained since the ensemble shipped --
    callers then fall back to Model A only."""
    if league not in _ensemble_cache and _ensemble_path(league).exists():
        _ensemble_cache[league] = joblib.load(_ensemble_path(league))
    return _ensemble_cache.get(league)


def get_model(league: str) -> LeagueGoalModel:
    if league in _model_cache:
        return _model_cache[league]
    path = _model_path(league)
    if path.exists():
        model = joblib.load(path)
        _model_cache[league] = model
        return model
    raise ModelNotTrainedError(f"No trained model for '{league}' yet -- trigger a retrain first.")


def _matches_df(db: Session, group: str) -> pd.DataFrame:
    """All finished matches of a model group, pooled across its competitions
    so a team's rolling form and Elo span every competition it plays in."""
    rows = db.execute(
        select(Match, MatchOdds.opening)
        .outerjoin(MatchOdds, (MatchOdds.match_id == Match.id) & (MatchOdds.bookmaker == HISTORICAL_ODDS_BOOKMAKER))
        .where(Match.league.in_(group_competitions(group)))
    ).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        [
            {
                **market_features_from_odds(opening),
                "date": r.date,
                "league": r.league,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "home_goals": r.home_goals,
                "away_goals": r.away_goals,
                "home_shots": r.home_shots,
                "away_shots": r.away_shots,
                "home_shots_on_target": r.home_shots_on_target,
                "away_shots_on_target": r.away_shots_on_target,
                "home_corners": r.home_corners,
                "away_corners": r.away_corners,
                "home_cards": r.home_cards,
                "away_cards": r.away_cards,
                "home_xg": r.home_xg,
                "away_xg": r.away_xg,
            }
            for r, opening in rows
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def train_league(db: Session, league: str) -> dict:
    history = _matches_df(db, league)
    if history.empty:
        raise ValueError(f"No matches in DB for league '{league}' -- run a scrape first.")

    tier = league_tier(league)
    # Model A (refit on full history) + Model B + per-market blend weights and
    # calibrators, with test-split metrics -- see app/ml/ensemble.py.
    model, ensemble = train_ensemble(history, league, tier)

    joblib.dump(model, _model_path(league))
    joblib.dump(ensemble, _ensemble_path(league))
    _model_cache[league] = model
    _ensemble_cache[league] = ensemble

    trained_at = datetime.now(timezone.utc)
    stmt = pg_insert(ModelRun).values(
        league=league,
        tier=tier,
        trained_at=trained_at,
        n_matches=len(history),
        model_path=str(_model_path(league)),
        metrics=ensemble.metrics,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[ModelRun.league],
        set_={
            "tier": stmt.excluded.tier,
            "trained_at": stmt.excluded.trained_at,
            "n_matches": stmt.excluded.n_matches,
            "model_path": stmt.excluded.model_path,
            "metrics": stmt.excluded.metrics,
        },
    )
    db.execute(stmt)
    db.commit()

    return {"league": league, "n_matches": len(history), "trained_at": trained_at}


def value_fields(markets: dict, odds: dict | None) -> dict:
    """Prediction columns for the value layer, from nested market
    probabilities (match_result / over_under / btts) and flat bookmaker odds.
    All null without odds, so stale value flags never outlive their odds."""
    if not odds:
        return {"expected_value": None, "is_value_bet": None, "value_selection": None, "recommended_stake": None}
    probs = flat_model_probabilities(markets.get("match_result"), markets.get("over_under"), markets.get("btts"))
    return assess_value(probs, odds)


def generate_predictions(db: Session, league: str) -> dict:
    model = get_model(league)
    ensemble = get_ensemble(league)
    history = _matches_df(db, league)
    upcoming = (
        db.execute(select(UpcomingFixture).where(UpcomingFixture.league.in_(group_competitions(league))))
        .scalars()
        .all()
    )

    # Team rolling-stat/Elo state depends only on real match history, which
    # is identical for every fixture in this league -- compute it once
    # instead of rebuilding the whole feature pipeline per fixture (that
    # per-fixture rebuild was previously the dominant cost of a retrain:
    # several minutes for a single mid-size league).
    team_state = compute_team_state(history, league_tier(league))
    fixtures = [fx for fx in upcoming if fx.home_team in team_state and fx.away_team in team_state]
    n_failed = len(upcoming) - len(fixtures)

    # Odds live on the prediction row (fetched by scrape_service.scrape_odds)
    # and outlive a retrain. They feed Model B's market features and the
    # shrinkage toward the market, and value is re-assessed against them.
    stored_odds = dict(
        db.execute(
            select(Prediction.fixture_id, Prediction.bookmaker_odds).where(
                Prediction.fixture_id.in_([fx.fixture_id for fx in fixtures]),
                Prediction.bookmaker_odds.isnot(None),
            )
        ).all()
    ) if fixtures else {}

    # Model B scores every fixture in one batch.
    b_probs = None
    if ensemble is not None and fixtures:
        targets = pd.DataFrame(
            [
                {
                    "date": fx.date,
                    "home_team": fx.home_team,
                    "away_team": fx.away_team,
                    **market_features_from_odds(stored_odds.get(fx.fixture_id)),
                }
                for fx in fixtures
            ]
        )
        b_probs = ensemble.predict_model_b(history, targets)

    n_ok = 0
    for i, fx in enumerate(fixtures):
        try:
            result = predict_fixture_fast(model, team_state, fx.home_team, fx.away_team)
        except (ValueError, IndexError):
            n_failed += 1
            continue
        n_ok += 1

        a_flat = flat_from_prediction(result)
        if ensemble is not None:
            combined = ensemble.combine(
                a_flat,
                {m: v[i] for m, v in b_probs.items()},
                market_flat_from_odds(stored_odds.get(fx.fixture_id)),
            )
            final, raw, weights = nest(combined["calibrated"]), combined["raw"], combined["weights"]
        else:
            final, raw, weights = nest(a_flat), None, None

        # The market columns hold the final (calibrated when available)
        # probabilities, so odds and the existing UI pick them up unchanged.
        values = {
            "fixture_id": fx.fixture_id,
            "league": fx.league,
            "lambda_home": result["lambda_home"],
            "lambda_away": result["lambda_away"],
            **{LABEL_TO_FIELD[label]: p for label, p in final["goal_ranges"].items()},
            "match_result": final["match_result"],
            "over_under": final["over_under"],
            "btts": final["btts"],
            "raw_probabilities": raw,
            "ensemble_weights": weights,
            **value_fields(final, stored_odds.get(fx.fixture_id)),
        }
        stmt = pg_insert(Prediction).values(**values)
        update_cols = {k: getattr(stmt.excluded, k) for k in values if k != "fixture_id"}
        update_cols["predicted_at"] = datetime.now(timezone.utc)
        stmt = stmt.on_conflict_do_update(index_elements=[Prediction.fixture_id], set_=update_cols)
        db.execute(stmt)

    db.commit()
    return {"league": league, "predicted": n_ok, "skipped": n_failed}


def retrain_leagues(leagues: list[str] | None) -> dict:
    """Entry point for the background job -- opens its own DB session since
    it runs outside the request lifecycle (see job_manager)."""
    # Competitions map to their model group; asking for two international
    # competitions retrains the shared International model once.
    targets = list(dict.fromkeys(model_group(lg) for lg in leagues)) if leagues else ALL_MODEL_GROUPS
    db = SessionLocal()
    results = []
    try:
        for league in targets:
            train_result = train_league(db, league)
            predict_result = generate_predictions(db, league)
            results.append({**train_result, **predict_result})
        return {"leagues": results}
    finally:
        db.close()


def predict_single(db: Session, fixture_id: int) -> dict | None:
    """Live recompute (not persisted) including the full 0..8+ goal-count
    breakdown, for the fixture-detail view."""
    fx = db.execute(select(UpcomingFixture).where(UpcomingFixture.fixture_id == fixture_id)).scalar_one_or_none()
    if fx is None:
        return None
    group = model_group(fx.league)
    model = get_model(group)
    history = _matches_df(db, group)
    team_state = compute_team_state(history, league_tier(group))
    return predict_fixture_fast(model, team_state, fx.home_team, fx.away_team)
