"""
Layer 3: ensemble blending, probability calibration, and evaluation metrics.

Blend:      P_blended = w_A * P_A + (1 - w_A) * P_B, with w_A fitted per market
            on the calibration split (grid search minimising log loss),
            falling back to DEFAULT_WEIGHTS when that split is too small.
Calibrate:  Platt scaling (sigmoid, the default) or isotonic regression,
            fitted on the *blended* calibration-split probabilities. Note
            sklearn's CalibratedClassifierCV wraps a single estimator, but
            the thing being calibrated here is a blend of two models, so
            Platt is implemented directly -- a logistic regression on the
            logit of the probability, the same mapping CalibratedClassifierCV's
            method="sigmoid" fits internally.
Shrink:     P_final = (1 - w) * P_calibrated + w * P_market, where P_market
            is the bookmaker price de-vigged with Shin's method and w is a
            fixed 0.70 (a per-market fit on the calibration split is
            recorded as a diagnostic; USE_FITTED_SHRINKAGE serves it instead).
            Applied only where a market price exists.
Metrics:    log loss, Brier score, and expected calibration error (ECE).

Binary markets take/return P(event) arrays of shape (n,); the 1X2 market
takes/returns (n, 3) arrays (home/draw/away), calibrated by multinomial
temperature scaling, kept only if it beats the raw blend out-of-fold on the
calibration split (fit_calibrator_or_identity).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-12

# w_A (Model A's share); Model B gets 1 - w_A.
DEFAULT_WEIGHTS = {"match_result": 0.45, "btts": 0.45, "over_under": 0.45, "goal_range": 0.80}
WEIGHT_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)
MIN_FIT_SAMPLES = 50
# Market shrinkage: the final probability's weight on the de-vigged market.
DEFAULT_SHRINKAGE_WEIGHT = 0.70
# Serve the fixed weight, not the per-league fitted one. On the 2025/26
# out-of-sample backtest the fitted weights (fitted on a few hundred
# pre-season matches, ranging 0.00-1.00) were no better than 0.70 and far
# less stable, so they're kept only as a diagnostic in model_runs.metrics.
USE_FITTED_SHRINKAGE = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def log_loss_binary(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def log_loss_multiclass(P: np.ndarray, y: np.ndarray) -> float:
    P = np.clip(np.asarray(P, dtype=float), EPS, 1.0)
    y = np.asarray(y, dtype=int)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def brier_binary(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def brier_multiclass(P: np.ndarray, y: np.ndarray) -> float:
    """Sum over classes of squared error, averaged over rows (range 0..2)."""
    P = np.asarray(P, dtype=float)
    Y = np.eye(P.shape[1])[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE: sum over bins of |mean predicted - observed rate|,
    weighted by the share of predictions in that bin."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def expected_calibration_error_multiclass(P: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Class-wise ECE: mean of one-vs-rest ECE across classes."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    return float(np.mean([expected_calibration_error(P[:, k], y == k, n_bins) for k in range(P.shape[1])]))


def evaluate(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Brier / log loss / ECE for a binary (n,) or multiclass (n, k) prediction."""
    if np.asarray(p).ndim == 2:
        return {
            "brier": brier_multiclass(p, y),
            "log_loss": log_loss_multiclass(p, y),
            "ece": expected_calibration_error_multiclass(p, y),
        }
    return {"brier": brier_binary(p, y), "log_loss": log_loss_binary(p, y), "ece": expected_calibration_error(p, y)}


def base_rate_prediction(y_train: np.ndarray, n: int, n_classes: int | None = None) -> np.ndarray:
    """Climatology baseline: predict the training base rate for every row."""
    y_train = np.asarray(y_train)
    if n_classes:
        rates = np.bincount(y_train.astype(int), minlength=n_classes) / len(y_train)
        return np.tile(rates, (n, 1))
    return np.full(n, y_train.mean())


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------

class BinaryCalibrator:
    def __init__(self, method: str = "sigmoid"):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"unknown calibration method: {method}")
        self.method = method
        self._model = None  # None = identity (e.g. single-class calibration labels)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "BinaryCalibrator":
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            self._model = None
            return self
        if self.method == "sigmoid":
            self._model = LogisticRegression(C=100.0).fit(_logit(p).reshape(-1, 1), y)
        else:
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self._model is None:
            return p
        if self.method == "sigmoid":
            return self._model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return self._model.predict(p)


class MulticlassCalibrator:
    """One-vs-rest BinaryCalibrator per class, rows renormalised to sum to 1.
    Superseded for 1X2 by TemperatureCalibrator (six parameters on a few
    hundred matches overfit); kept so older saved ensembles still load."""

    def __init__(self, n_classes: int, method: str = "sigmoid"):
        self.calibrators = [BinaryCalibrator(method) for _ in range(n_classes)]

    def fit(self, P: np.ndarray, y: np.ndarray) -> "MulticlassCalibrator":
        y = np.asarray(y, dtype=int)
        for k, cal in enumerate(self.calibrators):
            cal.fit(P[:, k], y == k)
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        Q = np.column_stack([cal.transform(P[:, k]) for k, cal in enumerate(self.calibrators)])
        Q = np.clip(Q, EPS, None)
        return Q / Q.sum(axis=1, keepdims=True)


class TemperatureCalibrator:
    """Multinomial temperature scaling: softmax(log P / T). A single parameter
    that sharpens (T < 1) or softens (T > 1) the whole distribution without
    reordering classes, and rows sum to 1 by construction."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, P: np.ndarray, y: np.ndarray) -> "TemperatureCalibrator":
        y = np.asarray(y, dtype=int)
        self.temperature = 1.0
        if len(np.unique(y)) < 2:
            return self
        log_p = np.log(np.clip(np.asarray(P, dtype=float), EPS, 1.0))
        res = minimize_scalar(
            lambda t: log_loss_multiclass(_softmax(log_p / t), y), bounds=(0.2, 5.0), method="bounded"
        )
        self.temperature = float(res.x)
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        return _softmax(np.log(np.clip(np.asarray(P, dtype=float), EPS, 1.0)) / self.temperature)


class IdentityCalibrator:
    """Passes probabilities through unchanged: used when no calibrator beats
    the raw blend out-of-fold."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IdentityCalibrator":
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p, dtype=float)


def fit_calibrator_or_identity(calibrator, p: np.ndarray, y: np.ndarray, n_folds: int = 5):
    """Fit `calibrator` on (p, y) and return it only if it lowers log loss
    out-of-fold (contiguous folds of the calibration split) versus the raw
    probabilities; otherwise return an IdentityCalibrator. In-sample, any
    fitted calibrator looks better, so the check has to be out-of-fold."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    if len(y) < MIN_FIT_SAMPLES:
        return IdentityCalibrator()
    loss = log_loss_multiclass if p.ndim == 2 else log_loss_binary
    oof = np.empty_like(p)
    for fold in np.array_split(np.arange(len(y)), n_folds):
        rest = np.setdiff1d(np.arange(len(y)), fold)
        oof[fold] = copy.deepcopy(calibrator).fit(p[rest], y[rest]).transform(p[fold])
    if loss(oof, y) >= loss(p, y):
        return IdentityCalibrator()
    return calibrator.fit(p, y)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

def default_weight(market: str) -> float:
    if market.startswith("over_"):
        return DEFAULT_WEIGHTS["over_under"]
    if market.startswith("range_"):
        return DEFAULT_WEIGHTS["goal_range"]
    return DEFAULT_WEIGHTS[market]


def blend(p_a: np.ndarray, p_b: np.ndarray, w_a: float) -> np.ndarray:
    return w_a * np.asarray(p_a, dtype=float) + (1.0 - w_a) * np.asarray(p_b, dtype=float)


def fit_blend_weight(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray, market: str) -> float:
    """w_A minimising log loss on the calibration split; the market's default
    weight when there are too few samples to fit reliably."""
    if len(y) < MIN_FIT_SAMPLES:
        return default_weight(market)
    loss = log_loss_multiclass if np.asarray(p_a).ndim == 2 else log_loss_binary
    losses = [loss(blend(p_a, p_b, w), y) for w in WEIGHT_GRID]
    return float(WEIGHT_GRID[int(np.argmin(losses))])


# ---------------------------------------------------------------------------
# Market de-vigging and shrinkage
# ---------------------------------------------------------------------------

def _shin(odds: np.ndarray) -> tuple[np.ndarray, float]:
    pi = 1.0 / odds
    booksum = pi.sum()
    if booksum <= 1.0:
        # No margin (or an arbitrage across books): nothing for Shin to remove.
        return pi / booksum, 0.0

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / booksum) - z) / (2.0 * (1.0 - z))

    # sum(probs(0)) = sqrt(booksum) > 1 and it falls monotonically in z.
    z = brentq(lambda z: probs(z).sum() - 1.0, 0.0, 0.999, xtol=1e-12)
    p = probs(z)
    return p / p.sum(), float(z)


def shin_implied_probabilities(decimal_odds: Sequence[float] | Mapping[str, float]):
    """Fair (de-vigged) probabilities for one complete book of mutually
    exclusive outcomes, via Shin's method: the overround is modelled as
    protection against a fraction z of insider money, so longshots are
    shaded more than favourites (the favourite-longshot bias). Returns the
    same shape as the input: a dict for a dict, an array otherwise."""
    if isinstance(decimal_odds, Mapping):
        keys = list(decimal_odds)
        p, _ = _shin(np.array([decimal_odds[k] for k in keys], dtype=float))
        return {k: float(v) for k, v in zip(keys, p)}
    odds = np.asarray(decimal_odds, dtype=float)
    if odds.ndim != 1 or len(odds) < 2 or np.any(odds <= 1.0):
        raise ValueError("need >= 2 decimal odds, all > 1.0")
    return _shin(odds)[0]


def shrink_to_market(p_model: np.ndarray, p_fair_market: np.ndarray, w: float = DEFAULT_SHRINKAGE_WEIGHT) -> np.ndarray:
    """Bayesian shrinkage toward the de-vigged market: (1 - w) * p_model +
    w * p_fair_market. The market is the prior (it is hard to beat), the
    model the evidence; w is the prior's weight. Rows where the market is
    missing (NaN) keep the model's probability."""
    p_model = np.asarray(p_model, dtype=float)
    p_mkt = np.asarray(p_fair_market, dtype=float)
    shrunk = (1.0 - w) * p_model + w * p_mkt
    missing = np.isnan(p_mkt) if p_mkt.ndim == 1 else np.isnan(p_mkt).any(axis=-1, keepdims=True)
    return np.where(missing, p_model, shrunk)


def fit_shrinkage_weight(p_model: np.ndarray, p_fair_market: np.ndarray, y: np.ndarray) -> float:
    """w minimising log loss of the shrunk probabilities on the calibration
    split (rows with a market price only); DEFAULT_SHRINKAGE_WEIGHT when
    there are too few such rows to fit reliably."""
    p_model = np.asarray(p_model, dtype=float)
    p_mkt = np.asarray(p_fair_market, dtype=float)
    y = np.asarray(y)
    has = ~(np.isnan(p_mkt) if p_mkt.ndim == 1 else np.isnan(p_mkt).any(axis=1))
    if has.sum() < MIN_FIT_SAMPLES:
        return DEFAULT_SHRINKAGE_WEIGHT
    loss = log_loss_multiclass if p_model.ndim == 2 else log_loss_binary
    losses = [loss(shrink_to_market(p_model[has], p_mkt[has], w), y[has]) for w in WEIGHT_GRID]
    return float(WEIGHT_GRID[int(np.argmin(losses))])


@dataclass
class MarketCalibration:
    """Everything needed to turn one market's raw Model A (+ Model B)
    probabilities into a final probability. w_a is None for Model-A-only
    markets (goal ranges), which are calibrated but not blended. w_market is
    the fitted shrinkage weight toward the de-vigged market (None for
    ensembles saved before shrinkage). Shrinkage is applied only when a
    market price exists, with DEFAULT_SHRINKAGE_WEIGHT unless
    USE_FITTED_SHRINKAGE opts into w_market."""

    w_a: float | None
    calibrator: BinaryCalibrator | MulticlassCalibrator | TemperatureCalibrator | IdentityCalibrator
    w_market: float | None = None

    def shrinkage_weight(self) -> float:
        if USE_FITTED_SHRINKAGE and self.w_market is not None:
            return self.w_market
        return DEFAULT_SHRINKAGE_WEIGHT

    def blended(self, p_a: np.ndarray, p_b: np.ndarray | None) -> np.ndarray:
        if self.w_a is None or p_b is None:
            return np.asarray(p_a, dtype=float)
        return blend(p_a, p_b, self.w_a)

    def apply(self, p_a: np.ndarray, p_b: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        raw = self.blended(p_a, p_b)
        return raw, self.calibrator.transform(raw)
