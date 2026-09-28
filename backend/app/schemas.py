from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.ml.football_goal_range_model import probabilities_to_odds
from app.ml.value_engine import market_probabilities

# Goal-range labels use hyphens ("1-3") in the model/CSV output, but hyphens
# aren't valid Postgres/Python identifiers, so DB columns and ORM/ORM-derived
# fields use underscores (p_1_3). These two maps translate between them so
# the API's JSON shape matches what upcoming_predictions.csv already used.
RANGE_LABELS = ["1-3", "1-4", "2-3", "2-4", "2-5", "2-6", "3-5", "3-6"]
LABEL_TO_FIELD = {label: f"p_{label.replace('-', '_')}" for label in RANGE_LABELS}
FIELD_TO_LABEL = {field: label for label, field in LABEL_TO_FIELD.items()}


class LeagueOut(BaseModel):
    league: str
    tier: str
    model_group: str
    last_trained_at: datetime | None = None
    n_matches: int | None = None


class PredictionOut(BaseModel):
    lambda_home: float
    lambda_away: float
    predicted_at: datetime
    range_probabilities: dict[str, float]
    # Nullable: rows predicted before these markets shipped won't have them
    # until the next retrain (see backend/scripts/migrate_add_markets.py).
    match_result: dict[str, float] | None = None
    over_under: dict[str, float] | None = None
    btts: dict[str, float] | None = None
    # Top-N most likely exact scorelines ("2-1": 0.11, ...), Model A only --
    # see football_goal_range_model.correct_score_probabilities. Doesn't sum
    # to 1 (display-only, not a complete distribution).
    correct_score: dict[str, float] | None = None
    # Decimal odds (1/probability, no bookmaker margin) -- computed from the
    # probabilities above at response time rather than stored, since they're
    # a pure derived value with no new information. Same keys as their
    # probability counterparts.
    range_odds: dict[str, float | None]
    match_result_odds: dict[str, float | None] | None = None
    over_under_odds: dict[str, float | None] | None = None
    btts_odds: dict[str, float | None] | None = None
    correct_score_odds: dict[str, float | None] | None = None
    # Ensemble provenance -- all null for predictions made before the
    # ensemble shipped (Model A only, uncalibrated) until the next retrain.
    #   raw_probabilities: {"model_a": {...}, "model_b": {...}, "blended": {...}}
    #   calibrated_probabilities: the final numbers (same as the market fields
    #     above), grouped the same way as raw_probabilities
    #   ensemble_weights: per-market w_A (Model A's share; null = Model A only)
    #   brier_scores: per-market test-split Brier for this fixture's *league*
    #     (it measures the model on a holdout set, not this one fixture)
    raw_probabilities: dict | None = None
    calibrated_probabilities: dict | None = None
    ensemble_weights: dict[str, float | None] | None = None
    brier_scores: dict[str, dict[str, float | None]] | None = None
    # Value layer -- all null until bookmaker odds have been fetched. Flat
    # outcome keys (home_win, over_2.5, btts_yes, ...).
    #   bookmaker_odds: latest pre-match snapshot (odds_bookmaker, odds_fetched_at)
    #   market_probabilities: bookmaker_odds de-margined with Shin's method
    #   expected_value: p_model * odds - 1 per outcome (+0.052 = 5.2% edge)
    #   is_value_bet / value_selection: best outcome with EV >= 3%, if any
    #   recommended_stake: quarter-Kelly stake for value_selection, in units
    #     (1 unit = 1% of bankroll)
    bookmaker_odds: dict[str, float] | None = None
    odds_bookmaker: str | None = None
    odds_fetched_at: datetime | None = None
    market_probabilities: dict[str, float] | None = None
    expected_value: dict[str, float] | None = None
    is_value_bet: bool | None = None
    value_selection: str | None = None
    recommended_stake: float | None = None

    @staticmethod
    def from_orm_row(row, league_metrics: dict | None = None) -> "PredictionOut":
        range_probabilities = {label: getattr(row, field) for label, field in LABEL_TO_FIELD.items()}
        calibrated = None
        if row.raw_probabilities:
            calibrated = {
                "match_result": row.match_result,
                "over_under": row.over_under,
                "btts": row.btts,
                "goal_ranges": range_probabilities,
            }
        return PredictionOut(
            lambda_home=row.lambda_home,
            lambda_away=row.lambda_away,
            predicted_at=row.predicted_at,
            range_probabilities=range_probabilities,
            match_result=row.match_result,
            over_under=row.over_under,
            btts=row.btts,
            correct_score=row.correct_score,
            range_odds=probabilities_to_odds(range_probabilities),
            match_result_odds=probabilities_to_odds(row.match_result) if row.match_result else None,
            over_under_odds=probabilities_to_odds(row.over_under) if row.over_under else None,
            btts_odds=probabilities_to_odds(row.btts) if row.btts else None,
            correct_score_odds=probabilities_to_odds(row.correct_score) if row.correct_score else None,
            raw_probabilities=row.raw_probabilities,
            calibrated_probabilities=calibrated,
            ensemble_weights=row.ensemble_weights,
            brier_scores=brier_scores_from_metrics(league_metrics) if row.raw_probabilities else None,
            bookmaker_odds=row.bookmaker_odds,
            odds_bookmaker=row.odds_bookmaker,
            odds_fetched_at=row.odds_fetched_at,
            market_probabilities=market_probabilities(row.bookmaker_odds) if row.bookmaker_odds else None,
            expected_value=row.expected_value,
            is_value_bet=row.is_value_bet,
            value_selection=row.value_selection,
            recommended_stake=row.recommended_stake,
        )


def brier_scores_from_metrics(metrics: dict | None) -> dict[str, dict[str, float | None]] | None:
    """Per-market test-split Brier (calibrated model vs. base-rate baseline, plus
    each base model) from a model_runs.metrics blob."""
    if not metrics or "markets" not in metrics:
        return None
    return {market: m["brier"] for market, m in metrics["markets"].items()}


class FixtureOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    fixture_id: int
    date: date
    league: str
    home_team: str
    away_team: str
    status: str
    prediction: PredictionOut | None = None


class FixtureDetailOut(FixtureOut):
    goal_distribution: dict[str, float] | None = None


class MatchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    fixture_id: int | None
    date: date
    league: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    prediction: PredictionOut | None = None


class RefreshRequest(BaseModel):
    leagues: list[str] | None = Field(default=None, description="Subset of leagues; omit for all")
    include_stats: bool = False
    # Bookmaker odds for predicted fixtures in the next ODDS_DAYS_AHEAD days
    # (one API request per fixture) -- feeds the value-bet fields.
    include_odds: bool = True


class LogEntryOut(BaseModel):
    at: datetime
    message: str


class JobOut(BaseModel):
    job_id: str
    type: str
    status: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    detail: str | None = None
    error: str | None = None
    # Approximate expected len(log) at completion (see refresh.py's
    # _scrape_total_steps/_retrain_total_steps) -- the frontend's progress
    # bar is len(log)/total_steps, not a fixed timer.
    total_steps: int | None = None
    log: list[LogEntryOut] = []
