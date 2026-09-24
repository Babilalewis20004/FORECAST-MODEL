"""
Ensemble orchestration: Extract features -> Model A + Model B -> Blend -> Calibrate.

Training (train_ensemble), per league, on a chronological split of history:
  train (oldest 70%)   -> fit Model A and Model B
  calibration (15%)    -> fit per-market blend weight w_A and calibrator, then
                          the shrinkage weight toward the de-vigged market
                          (markets with historical odds: 1X2, O/U 2.5)
  test (newest 15%)    -> report Brier / log loss / ECE (never used to fit anything)
then refit both base models on 100% of history for serving, reusing the
calibration-split weights and calibrators (standard practice: the most
recent matches matter most in football, so they aren't withheld from the
production models, at the cost of the calibrators having been fitted
against slightly weaker models).

Market keys are "flat": match_result -> (3,) or (n, 3) home/draw/away;
btts -> P(yes); over_{line} -> P(over); range_{label} -> P(in range).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.ml.calibration import (
    DEFAULT_SHRINKAGE_WEIGHT,
    BinaryCalibrator,
    IdentityCalibrator,
    MarketCalibration,
    TemperatureCalibrator,
    base_rate_prediction,
    evaluate,
    fit_blend_weight,
    fit_calibrator_or_identity,
    fit_shrinkage_weight,
    shrink_to_market,
)
from app.ml.features import FEATURE_COLUMNS, MARKET_FEATURES, build_feature_frame, build_training_frame
from app.ml.football_goal_range_model import (
    GOAL_RANGES,
    OU_LINES,
    LeagueGoalModel,
    btts_probabilities,
    build_features as model_a_build_features,
    goal_range_probabilities,
    match_result_probabilities,
    over_under_probabilities,
    simulate_joint_score_distribution,
)
from app.ml.gbdt_models import RESULT_CLASSES, GBDTMarketModels
from app.ml.value_engine import market_probabilities

TRAIN_FRAC = 0.70
CALIBRATION_FRAC = 0.15  # the remaining 0.15 is the test split
VALIDATION_SIMS = 5000  # Model A sims per validation match (serving uses the full 20k)

BLENDED_MARKETS = ["match_result", "btts"] + [f"over_{line}" for line in OU_LINES]
RANGE_MARKETS = [f"range_{label}" for label in GOAL_RANGES]
ALL_MARKETS = BLENDED_MARKETS + RANGE_MARKETS


def _label_col(market: str) -> str:
    return "y_result" if market == "match_result" else f"y_{market}"


# ---------------------------------------------------------------------------
# Flat <-> nested market representations
# ---------------------------------------------------------------------------

def model_a_flat(home_goals: np.ndarray, away_goals: np.ndarray) -> dict:
    """Model A market probabilities from joint score samples, in flat form."""
    total = home_goals + away_goals
    mr = match_result_probabilities(home_goals, away_goals)
    ou = over_under_probabilities(total)
    flat = {
        "match_result": np.array([mr[c] for c in RESULT_CLASSES]),
        "btts": btts_probabilities(home_goals, away_goals)["yes"],
    }
    for line in OU_LINES:
        flat[f"over_{line}"] = ou[f"over_{line}"]
    for label, p in goal_range_probabilities(total).items():
        flat[f"range_{label}"] = p
    return flat


def flat_from_prediction(result: dict) -> dict:
    """Flat form of football_goal_range_model.predict_fixture()'s output."""
    flat = {
        "match_result": np.array([result["match_result"][c] for c in RESULT_CLASSES]),
        "btts": result["btts"]["yes"],
    }
    for line in OU_LINES:
        flat[f"over_{line}"] = result["over_under"][f"over_{line}"]
    for label, p in result["range_probabilities"].items():
        flat[f"range_{label}"] = p
    return flat


def nest(flat: dict) -> dict:
    """Flat market probabilities -> the API's nested shape (only markets present)."""
    out: dict = {}
    if "match_result" in flat:
        out["match_result"] = {c: round(float(v), 5) for c, v in zip(RESULT_CLASSES, flat["match_result"])}
    if "btts" in flat:
        yes = float(flat["btts"])
        out["btts"] = {"yes": round(yes, 5), "no": round(1.0 - yes, 5)}
    ou = {}
    for line in OU_LINES:
        if f"over_{line}" in flat:
            over = float(flat[f"over_{line}"])
            ou[f"over_{line}"] = round(over, 5)
            ou[f"under_{line}"] = round(1.0 - over, 5)
    if ou:
        out["over_under"] = ou
    ranges = {label: round(float(flat[f"range_{label}"]), 5) for label in GOAL_RANGES if f"range_{label}" in flat}
    if ranges:
        out["goal_ranges"] = ranges
    return out


# ---------------------------------------------------------------------------
# Ensemble artifact
# ---------------------------------------------------------------------------

@dataclass
class LeagueEnsemble:
    league: str
    tier: str
    model_b: GBDTMarketModels
    calibrations: dict[str, MarketCalibration]
    metrics: dict = field(default_factory=dict)
    calibration_method: str = "sigmoid"

    def weights(self) -> dict[str, float | None]:
        return {m: c.w_a for m, c in self.calibrations.items()}

    def predict_model_b(self, history: pd.DataFrame, targets: pd.DataFrame) -> dict[str, np.ndarray]:
        """Model B probabilities for a batch of targets (date, home_team, away_team)."""
        frame = build_feature_frame(history, targets[["date", "home_team", "away_team"]])
        return self.model_b.predict_proba(frame)

    def combine(self, a_flat: dict, b_flat: dict | None, market_flat: dict | None = None) -> dict:
        """One fixture: raw Model A/B + blended + calibrated probabilities,
        the calibrated ones shrunk toward the de-vigged market for every
        market with a price in market_flat (market_flat_from_odds)."""
        blended, calibrated, unshrunk = {}, {}, {}
        for market, cal in self.calibrations.items():
            p_a = np.atleast_1d(a_flat[market])[None, ...] if market == "match_result" else np.atleast_1d(a_flat[market])
            p_b = None
            if b_flat is not None and market in b_flat:
                p_b = np.asarray(b_flat[market])[None, ...] if market == "match_result" else np.atleast_1d(b_flat[market])
            raw, cal_p = cal.apply(p_a, p_b)
            blended[market] = raw[0]
            calibrated[market] = cal_p[0]
            if market_flat and market in market_flat:
                unshrunk[market] = cal_p[0]
                p_mkt = np.atleast_1d(market_flat[market])[None, ...] if market == "match_result" else np.atleast_1d(market_flat[market])
                calibrated[market] = shrink_to_market(cal_p, p_mkt, cal.shrinkage_weight())[0]
        raw = {"model_a": nest(a_flat), "blended": nest(blended)}
        if b_flat is not None:
            raw["model_b"] = nest(b_flat)
        if unshrunk:
            raw["calibrated_unshrunk"] = nest(unshrunk)
            raw["market"] = nest({m: market_flat[m] for m in unshrunk})
        return {"raw": raw, "calibrated": calibrated, "weights": self.weights()}


def market_flat_from_odds(odds: dict | None) -> dict:
    """Flat market-key (match_result / btts / over_{line}) de-vigged
    probabilities from flat bookmaker odds, complete books only."""
    if not odds:
        return {}
    fair = market_probabilities(odds)
    flat: dict = {}
    if all(c in fair for c in RESULT_CLASSES):
        flat["match_result"] = np.array([fair[c] for c in RESULT_CLASSES])
    if "btts_yes" in fair:
        flat["btts"] = fair["btts_yes"]
    for line in OU_LINES:
        if f"over_{line}" in fair:
            flat[f"over_{line}"] = fair[f"over_{line}"]
    return flat


def market_flat_from_frame(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """The same, batched, from MARKET_FEATURES columns (NaN = no price)."""
    if not set(MARKET_FEATURES) <= set(frame.columns):
        return {}
    return {
        "match_result": frame[["opening_prob_home", "opening_prob_draw", "opening_prob_away"]].to_numpy(dtype=float),
        "over_2.5": frame["opening_prob_over_2.5"].to_numpy(dtype=float),
    }


def _has_market(p_mkt: np.ndarray) -> np.ndarray:
    return ~(np.isnan(p_mkt).any(axis=1) if p_mkt.ndim == 2 else np.isnan(p_mkt))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _model_a_holdout_flat(
    model_a: LeagueGoalModel, history: pd.DataFrame, rows: pd.DataFrame, n_sims: int
) -> dict[str, np.ndarray]:
    """Model A probabilities for holdout rows, from a Model A fitted only on
    the train split. Model A's own features are built over the full history
    but are strictly backward-looking (shift(1) rolling / sequential Elo), so
    holdout rows never see their own result."""
    feats = model_a_build_features(history, model_a.tier)
    feats["date"] = pd.to_datetime(feats["date"]).astype("datetime64[ns]")
    keyed = rows[["date", "home_team", "away_team"]].merge(
        feats.drop_duplicates(["date", "home_team", "away_team"]),
        on=["date", "home_team", "away_team"],
        how="left",
    )
    X = keyed[model_a.feature_cols]
    lam_h = np.maximum(model_a.home_model.predict(X), 0.05)
    lam_a = np.maximum(model_a.away_model.predict(X), 0.05)

    per_row = [
        model_a_flat(*simulate_joint_score_distribution(
            lh, la, model_a.home_dispersion, model_a.away_dispersion, n_sims=n_sims
        ))
        for lh, la in zip(lam_h, lam_a)
    ]
    return {m: np.array([r[m] for r in per_row]) for m in ALL_MARKETS}


def _round_metrics(d: dict) -> dict:
    return {k: (round(float(v), 5) if v is not None else None) for k, v in d.items()}


def train_ensemble(
    history: pd.DataFrame,
    league: str,
    tier: str,
    calibration_method: str = "sigmoid",
    validation_sims: int = VALIDATION_SIMS,
) -> tuple[LeagueGoalModel, LeagueEnsemble]:
    """Returns (Model A refit on full history, fitted LeagueEnsemble)."""
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"]).astype("datetime64[ns]")
    frame = build_training_frame(history).sort_values("date", kind="mergesort").reset_index(drop=True)

    n = len(frame)
    cut_cal = frame["date"].iloc[int(n * TRAIN_FRAC)]
    cut_test = frame["date"].iloc[int(n * (TRAIN_FRAC + CALIBRATION_FRAC))]
    train = frame[frame["date"] < cut_cal]
    cal = frame[(frame["date"] >= cut_cal) & (frame["date"] < cut_test)].reset_index(drop=True)
    test = frame[frame["date"] >= cut_test].reset_index(drop=True)

    # Base models on the train split only.
    model_a_val = LeagueGoalModel(league=league, tier=tier)
    model_a_val.fit(history[history["date"] < cut_cal], test_size=0.0)
    model_b_val = GBDTMarketModels(feature_cols=FEATURE_COLUMNS).fit(train)

    a_cal = _model_a_holdout_flat(model_a_val, history, cal, validation_sims)
    a_test = _model_a_holdout_flat(model_a_val, history, test, validation_sims)
    b_cal = model_b_val.predict_proba(cal)
    b_test = model_b_val.predict_proba(test)
    m_cal = market_flat_from_frame(cal)
    m_test = market_flat_from_frame(test)

    calibrations: dict[str, MarketCalibration] = {}
    market_metrics: dict[str, dict] = {}
    for market in ALL_MARKETS:
        label = _label_col(market)
        y_cal, y_test, y_train = cal[label].to_numpy(), test[label].to_numpy(), train[label].to_numpy()
        is_multiclass = market == "match_result"
        has_b = market in b_cal

        w_a = fit_blend_weight(a_cal[market], b_cal[market], y_cal, market) if has_b else None
        blended_cal = MarketCalibration(w_a=w_a, calibrator=IdentityCalibrator()).blended(
            a_cal[market], b_cal.get(market)
        )
        # 1X2: temperature scaling, kept only if it beats the raw blend
        # out-of-fold. Binary markets: Platt unconditionally -- on held-out
        # data the same gate discarded Platt where it helped.
        calibrator = (
            fit_calibrator_or_identity(TemperatureCalibrator(), blended_cal, y_cal)
            if is_multiclass
            else BinaryCalibrator(calibration_method).fit(blended_cal, y_cal)
        )
        mc = MarketCalibration(w_a=w_a, calibrator=calibrator)
        calibrations[market] = mc

        # Shrinkage weight toward the de-vigged market, on calibration rows
        # that have odds (markets without historical odds keep the default).
        vs_market = None
        if market in m_cal:
            mc.w_market = fit_shrinkage_weight(calibrator.transform(blended_cal), m_cal[market], y_cal)

        blended_test, calibrated_test = mc.apply(a_test[market], b_test.get(market))
        if market in m_test:
            has = _has_market(m_test[market])
            if has.sum() > 0:
                sources = {
                    "calibrated": calibrated_test[has],
                    # "shrunk" = the weight actually served; "shrunk_fitted" = diagnostic.
                    "shrunk": shrink_to_market(calibrated_test[has], m_test[market][has], mc.shrinkage_weight()),
                    "shrunk_fitted": shrink_to_market(calibrated_test[has], m_test[market][has], mc.w_market),
                    "market": m_test[market][has],
                }
                scored_m = {src: evaluate(p, y_test[has]) for src, p in sources.items()}
                vs_market = {
                    "n": int(has.sum()),
                    "w_market": mc.shrinkage_weight(),
                    "w_market_fitted": mc.w_market,
                    **{metric: _round_metrics({s: v[metric] for s, v in scored_m.items()}) for metric in ("brier", "log_loss")},
                }
        baseline = base_rate_prediction(y_train, len(y_test), 3 if is_multiclass else None)
        scored = {
            "model_a": evaluate(a_test[market], y_test),
            "model_b": evaluate(b_test[market], y_test) if has_b else None,
            "blended": evaluate(blended_test, y_test),
            "calibrated": evaluate(calibrated_test, y_test),
            "baseline": evaluate(baseline, y_test),
        }
        market_metrics[market] = {
            "w_a": w_a,
            "calibrator": type(mc.calibrator).__name__,
            **{
                metric: _round_metrics({src: (s[metric] if s else None) for src, s in scored.items()})
                for metric in ("brier", "log_loss", "ece")
            },
            # Test rows with odds only: calibrated vs. shrunk vs. market itself.
            "vs_market": vs_market,
        }

    metrics = {
        "calibration_method": calibration_method,
        "n_train": int(len(train)),
        "n_calibration": int(len(cal)),
        "n_test": int(len(test)),
        "test_from": str(cut_test.date()),
        "markets": market_metrics,
    }

    # Serving models: refit on 100% of history.
    model_a = LeagueGoalModel(league=league, tier=tier)
    model_a.fit(history, test_size=0.0)
    model_b = GBDTMarketModels(feature_cols=FEATURE_COLUMNS).fit(frame)

    ensemble = LeagueEnsemble(
        league=league,
        tier=tier,
        model_b=model_b,
        calibrations=calibrations,
        metrics=metrics,
        calibration_method=calibration_method,
    )
    return model_a, ensemble
