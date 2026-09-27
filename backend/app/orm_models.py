from datetime import date, datetime

from sqlalchemy import BigInteger, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("league", "date", "home_team", "away_team", name="ux_matches_natural_key"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    fixture_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    league: Mapped[str] = mapped_column(String, nullable=False, index=True)
    home_team: Mapped[str] = mapped_column(String, nullable=False)
    away_team: Mapped[str] = mapped_column(String, nullable=False)
    home_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    away_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    home_shots: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_shots: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_shots_on_target: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_shots_on_target: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_corners: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_corners: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_cards: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_cards: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_possession: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_possession: Mapped[float | None] = mapped_column(Float, nullable=True)
    referee: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class MatchOdds(Base):
    """Historical bookmaker odds for a finished match (football-data.co.uk,
    via scripts/import_historical_odds.py). Flat outcome keys as in
    value_engine: home_win / draw / away_win / over_2.5 / under_2.5.
    `opening` (the pre-closing snapshot) feeds Model B's market features and
    the shrinkage fit; `closing` is kept for backtests and closing-line value."""

    __tablename__ = "match_odds"
    __table_args__ = (UniqueConstraint("match_id", "bookmaker", name="ux_match_odds_match_bookmaker"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id", ondelete="CASCADE"), nullable=False, index=True)
    bookmaker: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    opening: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    closing: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UpcomingFixture(Base):
    __tablename__ = "upcoming_fixtures"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    # Full kickoff instant -- `date` above is truncated to a day. Nullable:
    # existing rows predate this column and backfill on their next scrape.
    # Needed by odds_service.poll_closing_lines() to find fixtures inside its
    # 10-15-minute-before-kickoff closing-line window.
    kickoff_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    league: Mapped[str] = mapped_column(String, nullable=False, index=True)
    home_team: Mapped[str] = mapped_column(String, nullable=False)
    away_team: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    referee: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False, index=True)
    league: Mapped[str] = mapped_column(String, nullable=False, index=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    lambda_home: Mapped[float] = mapped_column(Float, nullable=False)
    lambda_away: Mapped[float] = mapped_column(Float, nullable=False)
    p_1_3: Mapped[float] = mapped_column(Float, nullable=False)
    p_1_4: Mapped[float] = mapped_column(Float, nullable=False)
    p_2_3: Mapped[float] = mapped_column(Float, nullable=False)
    p_2_4: Mapped[float] = mapped_column(Float, nullable=False)
    p_2_5: Mapped[float] = mapped_column(Float, nullable=False)
    p_2_6: Mapped[float] = mapped_column(Float, nullable=False)
    p_3_5: Mapped[float] = mapped_column(Float, nullable=False)
    p_3_6: Mapped[float] = mapped_column(Float, nullable=False)
    # Extra markets -- stored as JSONB rather than one flat column per outcome
    # (unlike p_1_3 etc.) since each is a small, self-contained dict
    # (match_result_probabilities() etc.) with keys that may grow (e.g. more
    # O/U lines) without needing another migration. Nullable: existing rows
    # predicted before this feature shipped won't have them until retrained.
    match_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    over_under: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    btts: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Top-N most likely exact scorelines ("2-1": 0.11, ...), Model A only --
    # see football_goal_range_model.correct_score_probabilities. Not a
    # complete distribution (doesn't sum to 1); display-only, for the
    # correct-score market card.
    correct_score: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Ensemble provenance. When present, the market columns above hold the
    # final *calibrated* probabilities, and raw_probabilities holds the
    # pre-calibration Model A / Model B / blended values they came from.
    # Null for predictions made before the ensemble shipped (Model A only).
    raw_probabilities: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ensemble_weights: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Value layer (app/ml/value_engine.py). bookmaker_odds is the latest
    # pre-match snapshot from API-Football, flat outcome keys (home_win,
    # over_2.5, btts_yes, ...) -- it's only "closing" if the last odds refresh
    # ran just before kickoff. The value fields are re-derived from it and the
    # market columns above whenever either changes. All null until odds are
    # fetched (they're only published ~2 weeks before a fixture).
    bookmaker_odds: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    odds_bookmaker: Mapped[str | None] = mapped_column(String, nullable=True)
    odds_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expected_value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    is_value_bet: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    value_selection: Mapped[str | None] = mapped_column(String, nullable=True)
    recommended_stake: Mapped[float | None] = mapped_column(Float, nullable=True)


class BetTracking(Base):
    """Live closing-line-value tracking (Phase 2). One row per (fixture,
    market, selection) the value engine first flags as a value bet --
    odds_placed_* freezes the price at that moment (later flags on the same
    key, from odds moving or a retrain, don't overwrite it). closing_* is
    filled in by odds_service.poll_closing_lines() from a sharp book
    (Pinnacle/Bet365) in the 10-15 minutes before kickoff, and clv_percentage
    is calculate_clv()'s Shin no-vig CLV of the placed price against it."""

    __tablename__ = "bet_tracking"
    __table_args__ = (
        UniqueConstraint("fixture_id", "market", "selection", name="ux_bet_tracking_fixture_market_selection"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    market: Mapped[str] = mapped_column(String, nullable=False)
    selection: Mapped[str] = mapped_column(String, nullable=False)
    bookmaker: Mapped[str] = mapped_column(String, nullable=False)
    is_sharp_book: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    odds_placed_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    odds_placed_value: Mapped[float] = mapped_column(Float, nullable=False)
    ev_at_placement: Mapped[float] = mapped_column(Float, nullable=False)
    closing_bookmaker: Mapped[str | None] = mapped_column(String, nullable=True)
    closing_odds_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    clv_percentage: Mapped[float | None] = mapped_column(Float, nullable=True)


class ModelRun(Base):
    __tablename__ = "model_runs"

    league: Mapped[str] = mapped_column(String, primary_key=True)
    tier: Mapped[str] = mapped_column(String, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    n_matches: Mapped[int] = mapped_column(Integer, nullable=False)
    model_path: Mapped[str] = mapped_column(String, nullable=False)
    # Test-split evaluation from ensemble.train_ensemble (Brier / log loss /
    # ECE per market). Per league, not per prediction, so it lives here.
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
