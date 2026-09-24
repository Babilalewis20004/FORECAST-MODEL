import numpy as np
import pandas as pd
import pytest

from app.ml.calibration import DEFAULT_SHRINKAGE_WEIGHT
from app.ml.ensemble import ALL_MARKETS, BLENDED_MARKETS, RANGE_MARKETS, model_a_flat, nest, train_ensemble
from app.ml.features import FEATURE_COLUMNS, build_training_frame
from app.ml.football_goal_range_model import simulate_joint_score_distribution
from app.ml.gbdt_models import GBDTMarketModels


def test_gbdt_probabilities_are_valid(synthetic_matches):
    frame = build_training_frame(synthetic_matches)
    probs = GBDTMarketModels(feature_cols=FEATURE_COLUMNS).fit(frame).predict_proba(frame.tail(20))
    assert probs["match_result"].shape == (20, 3)
    assert np.allclose(probs["match_result"].sum(axis=1), 1.0)
    for market in ("btts", "over_1.5", "over_2.5", "over_3.5"):
        assert probs[market].shape == (20,)
        assert ((probs[market] >= 0) & (probs[market] <= 1)).all()


@pytest.fixture(scope="module")
def trained(synthetic_matches):
    return train_ensemble(synthetic_matches, "League Two", "lower", validation_sims=500)


def test_train_ensemble_reports_honest_splits_and_metrics(trained):
    _, ensemble = trained
    m = ensemble.metrics
    assert m["n_train"] > m["n_calibration"] > 0 and m["n_test"] > 0
    assert set(m["markets"]) == set(ALL_MARKETS)
    for market in BLENDED_MARKETS:
        brier = m["markets"][market]["brier"]
        assert all(brier[k] is not None for k in ("model_a", "model_b", "blended", "calibrated", "baseline"))
        assert 0.0 <= m["markets"][market]["w_a"] <= 1.0
    for market in RANGE_MARKETS:
        assert m["markets"][market]["w_a"] is None  # Model A only
        assert m["markets"][market]["brier"]["model_b"] is None


def test_combine_produces_consistent_calibrated_markets(trained, synthetic_matches):
    model_a, ensemble = trained
    target = pd.DataFrame([{"date": "2099-01-01", "home_team": "Lea_Team0", "away_team": "Lea_Team1"}])
    b = ensemble.predict_model_b(synthetic_matches, target)
    a = model_a_flat(*simulate_joint_score_distribution(1.4, 1.1, model_a.home_dispersion, model_a.away_dispersion, n_sims=2000))

    out = ensemble.combine(a, {k: v[0] for k, v in b.items()})
    final = nest(out["calibrated"])
    assert sum(final["match_result"].values()) == pytest.approx(1.0, abs=1e-4)
    assert final["btts"]["yes"] + final["btts"]["no"] == pytest.approx(1.0)
    for line in (1.5, 2.5, 3.5):
        assert final["over_under"][f"over_{line}"] + final["over_under"][f"under_{line}"] == pytest.approx(1.0)
    assert set(out["raw"]) == {"model_a", "model_b", "blended"}
    assert "goal_ranges" not in out["raw"]["model_b"]
    assert out["weights"]["range_2-4"] is None


def test_market_shrinkage_fitted_in_training_and_applied_in_serving(synthetic_matches):
    # Market columns derived from the actual outcome with noise: informative,
    # so the calibration split should give the market real weight.
    rng = np.random.default_rng(3)
    df = synthetic_matches.copy()
    home_win = (df["home_goals"] > df["away_goals"]).to_numpy()
    p_home = np.clip(np.where(home_win, 0.62, 0.30) + rng.normal(0, 0.05, len(df)), 0.05, 0.85)
    df["opening_prob_home"] = p_home
    df["opening_prob_draw"] = (1 - p_home) * 0.45
    df["opening_prob_away"] = (1 - p_home) * 0.55
    df["opening_prob_over_2.5"] = np.nan  # no O/U odds -> that market keeps the default weight
    model_a, ensemble = train_ensemble(df, "League Two", "lower", validation_sims=500)

    mr = ensemble.metrics["markets"]["match_result"]
    assert mr["vs_market"]["n"] > 0 and 0.0 <= mr["vs_market"]["w_market_fitted"] <= 1.0
    assert mr["vs_market"]["w_market"] == DEFAULT_SHRINKAGE_WEIGHT  # fixed weight is what's served
    assert set(mr["vs_market"]["brier"]) == {"calibrated", "shrunk", "shrunk_fitted", "market"}
    # The market columns are also Model B features: Model B learns them, the
    # blend leans on it, and the calibrated model beats the raw base models.
    assert mr["w_a"] < 0.5
    assert mr["brier"]["calibrated"] < mr["brier"]["model_a"]
    assert ensemble.metrics["markets"]["over_2.5"]["vs_market"] is None

    target = pd.DataFrame([{"date": "2099-01-01", "home_team": "Lea_Team0", "away_team": "Lea_Team1"}])
    b = ensemble.predict_model_b(df, target)
    a = model_a_flat(*simulate_joint_score_distribution(1.4, 1.1, model_a.home_dispersion, model_a.away_dispersion, n_sims=2000))
    market = {"match_result": np.array([0.9, 0.06, 0.04])}
    plain = ensemble.combine(a, {k: v[0] for k, v in b.items()})
    shrunk = ensemble.combine(a, {k: v[0] for k, v in b.items()}, market)
    assert shrunk["calibrated"]["match_result"][0] > plain["calibrated"]["match_result"][0]
    assert shrunk["calibrated"]["match_result"].sum() == pytest.approx(1.0)
    assert np.allclose(shrunk["calibrated"]["btts"], plain["calibrated"]["btts"])  # no BTTS price -> untouched
    assert {"calibrated_unshrunk", "market"} <= set(shrunk["raw"])
