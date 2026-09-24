import numpy as np
import pandas as pd
import pytest

from app.ml.calibration import (
    DEFAULT_SHRINKAGE_WEIGHT,
    MarketCalibration,
    IdentityCalibrator,
    fit_shrinkage_weight,
    shrink_to_market,
)
from app.ml.ensemble import market_flat_from_odds
from app.ml.features import MARKET_FEATURES, build_feature_frame, market_features_from_odds

rng = np.random.default_rng(1)


def test_shrink_formula_and_missing_market_passthrough():
    p = np.array([0.6, 0.2])
    m = np.array([0.5, np.nan])
    np.testing.assert_allclose(shrink_to_market(p, m, 0.7), [0.3 * 0.6 + 0.7 * 0.5, 0.2])
    P = np.array([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3]])
    M = np.array([[0.4, 0.3, 0.3], [np.nan, np.nan, np.nan]])
    out = shrink_to_market(P, M, 0.7)
    np.testing.assert_allclose(out[0], 0.3 * P[0] + 0.7 * M[0])
    np.testing.assert_allclose(out[1], P[1])
    assert out[0].sum() == pytest.approx(1.0)


def _truth_and_noisy(n: int, noise_model: float, noise_market: float):
    p_true = rng.uniform(0.2, 0.8, n)
    logit = np.log(p_true / (1 - p_true))
    noisy = lambda s: 1 / (1 + np.exp(-(logit + rng.normal(0, s, n))))  # noqa: E731
    y = rng.binomial(1, p_true)
    return noisy(noise_model), noisy(noise_market), y


def test_fit_shrinkage_weight_trusts_the_better_source():
    p_model, p_mkt, y = _truth_and_noisy(5000, noise_model=1.0, noise_market=0.1)
    assert fit_shrinkage_weight(p_model, p_mkt, y) >= 0.8  # noisy model -> lean on market
    p_model, p_mkt, y = _truth_and_noisy(5000, noise_model=0.1, noise_market=1.0)
    assert fit_shrinkage_weight(p_model, p_mkt, y) <= 0.2  # noisy market -> lean on model


def test_fit_shrinkage_weight_defaults_without_enough_odds():
    p = rng.uniform(0.2, 0.8, 500)
    m = np.full(500, np.nan)
    m[:10] = 0.5
    assert fit_shrinkage_weight(p, m, rng.binomial(1, p)) == DEFAULT_SHRINKAGE_WEIGHT
    # Ensembles pickled before shrinkage have no w_market -> default.
    assert MarketCalibration(w_a=None, calibrator=IdentityCalibrator()).shrinkage_weight() == DEFAULT_SHRINKAGE_WEIGHT


def test_fixed_weight_is_served_unless_fitted_opted_in(monkeypatch):
    mc = MarketCalibration(w_a=None, calibrator=IdentityCalibrator(), w_market=0.15)
    assert mc.shrinkage_weight() == DEFAULT_SHRINKAGE_WEIGHT == 0.70
    monkeypatch.setattr("app.ml.calibration.USE_FITTED_SHRINKAGE", True)
    assert mc.shrinkage_weight() == 0.15


def test_market_features_devig_and_pass_through(synthetic_matches):
    feats = market_features_from_odds({"home_win": 2.0, "draw": 3.4, "away_win": 4.0, "over_2.5": 1.9, "under_2.5": 1.95})
    assert feats["opening_prob_home"] + feats["opening_prob_draw"] + feats["opening_prob_away"] == pytest.approx(1.0)
    assert 0.45 < feats["opening_prob_over_2.5"] < 0.55
    assert np.isnan(market_features_from_odds(None)["opening_prob_home"])

    targets = synthetic_matches[["date", "home_team", "away_team"]].tail(3).assign(**feats)
    out = build_feature_frame(synthetic_matches, targets)
    assert list(out.columns).count("opening_prob_home") == 1
    assert (out["opening_prob_home"] == feats["opening_prob_home"]).all()
    # Targets without odds get NaN market features, not an error.
    bare = build_feature_frame(synthetic_matches, synthetic_matches[["date", "home_team", "away_team"]].tail(3))
    assert bare[MARKET_FEATURES].isna().all().all()


def test_market_flat_from_odds():
    flat = market_flat_from_odds({"home_win": 2.0, "draw": 3.4, "away_win": 4.0, "btts_yes": 1.8, "btts_no": 2.0})
    assert flat["match_result"].shape == (3,) and flat["match_result"].sum() == pytest.approx(1.0, abs=1e-4)
    assert 0.5 < flat["btts"] < 0.55
    assert "over_2.5" not in flat
    assert market_flat_from_odds(None) == {}
