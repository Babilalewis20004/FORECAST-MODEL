import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss

from app.ml.calibration import (
    DEFAULT_WEIGHTS,
    BinaryCalibrator,
    MulticlassCalibrator,
    brier_binary,
    brier_multiclass,
    expected_calibration_error,
    fit_blend_weight,
    log_loss_binary,
    log_loss_multiclass,
)

rng = np.random.default_rng(0)


def test_binary_metrics_match_sklearn():
    p = rng.uniform(0.05, 0.95, 500)
    y = rng.binomial(1, p)
    assert log_loss_binary(p, y) == pytest.approx(log_loss(y, p))
    assert brier_binary(p, y) == pytest.approx(brier_score_loss(y, p))


def test_multiclass_metrics():
    P = rng.dirichlet([2, 2, 2], 400)
    y = np.array([rng.choice(3, p=row) for row in P])
    assert log_loss_multiclass(P, y) == pytest.approx(log_loss(y, P, labels=[0, 1, 2]))
    # multiclass Brier = sum of per-class one-vs-rest Brier
    per_class = sum(brier_score_loss(y == k, P[:, k]) for k in range(3))
    assert brier_multiclass(P, y) == pytest.approx(per_class)


def test_ece_small_when_calibrated_large_when_not():
    p = rng.uniform(0, 1, 20000)
    y = rng.binomial(1, p)
    assert expected_calibration_error(p, y) < 0.02
    assert expected_calibration_error(p**3, y) > 0.1


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_fixes_miscalibration_out_of_sample(method):
    p_true = rng.uniform(0.02, 0.98, 20000)
    y = rng.binomial(1, p_true)
    p_bad = p_true**2  # systematically underconfident on positives
    fit, held = slice(0, 10000), slice(10000, None)
    cal = BinaryCalibrator(method).fit(p_bad[fit], y[fit])
    before = expected_calibration_error(p_bad[held], y[held])
    after = expected_calibration_error(cal.transform(p_bad[held]), y[held])
    assert after < before / 3


def test_multiclass_calibrator_rows_sum_to_one():
    P = rng.dirichlet([3, 2, 2], 1000)
    y = np.array([rng.choice(3, p=row) for row in P])
    Q = MulticlassCalibrator(3).fit(P**1.5 / (P**1.5).sum(1, keepdims=True), y).transform(P)
    assert np.allclose(Q.sum(axis=1), 1.0)
    assert (Q >= 0).all() and (Q <= 1).all()


def test_single_class_labels_fall_back_to_identity():
    p = rng.uniform(0, 1, 100)
    cal = BinaryCalibrator("sigmoid").fit(p, np.zeros(100))
    assert np.allclose(cal.transform(p), p)


def test_blend_weight_prefers_the_informative_model():
    p_true = rng.uniform(0.1, 0.9, 3000)
    y = rng.binomial(1, p_true)
    noise = rng.uniform(0.1, 0.9, 3000)
    assert fit_blend_weight(p_true, noise, y, "btts") >= 0.9
    assert fit_blend_weight(noise, p_true, y, "btts") <= 0.1


def test_blend_weight_falls_back_to_default_on_small_samples():
    p = rng.uniform(0.1, 0.9, 20)
    y = rng.binomial(1, p)
    assert fit_blend_weight(p, 1 - p, y, "over_2.5") == DEFAULT_WEIGHTS["over_under"]
