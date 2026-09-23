"""
Layer 2, Model B: direct GBDT classifiers trained on market outcomes.

Where Model A (the Negative-Binomial copula simulation) predicts expected
goals and derives every market from one simulated score distribution,
Model B learns each market's outcome directly from the Layer 1 features --
different inductive bias, which is what makes blending the two worthwhile.

Per league (matching Model A's per-league training):
  - 1X2:         XGBClassifier(objective="multi:softprob"), classes home/draw/away
  - BTTS:        XGBClassifier(objective="binary:logistic")
  - Over N.5:    one XGBClassifier(objective="binary:logistic") per line in OU_LINES
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

from app.ml.football_goal_range_model import OU_LINES

RESULT_CLASSES = ("home_win", "draw", "away_win")

# Shallow, slow-learning, heavily-subsampled trees: each league has only
# ~1-2k matches, so capacity is deliberately low to avoid fitting noise.
_BASE_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
)


def _binary_classifier() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **_BASE_PARAMS)


def _require_classes(y: pd.Series, expected: int, name: str) -> None:
    present = y.nunique()
    if present < expected:
        raise ValueError(f"{name}: training labels contain {present} class(es), need {expected}")


@dataclass
class GBDTMarketModels:
    feature_cols: list[str]
    result_model: xgb.XGBClassifier | None = None
    btts_model: xgb.XGBClassifier | None = None
    ou_models: dict[float, xgb.XGBClassifier] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "GBDTMarketModels":
        """frame: output of features.build_training_frame (features + y_* labels)."""
        X = frame[self.feature_cols]

        _require_classes(frame["y_result"], 3, "match_result")
        self.result_model = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss", **_BASE_PARAMS
        )
        self.result_model.fit(X, frame["y_result"])

        _require_classes(frame["y_btts"], 2, "btts")
        self.btts_model = _binary_classifier()
        self.btts_model.fit(X, frame["y_btts"])

        self.ou_models = {}
        for line in OU_LINES:
            y = frame[f"y_over_{line}"]
            _require_classes(y, 2, f"over_{line}")
            model = _binary_classifier()
            model.fit(X, y)
            self.ou_models[line] = model
        return self

    def predict_proba(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Flat market -> probabilities: 'match_result' is (n, 3) in
        RESULT_CLASSES order; 'btts' / 'over_{line}' are (n,) P(yes)/P(over)."""
        X = frame[self.feature_cols]
        out: dict[str, np.ndarray] = {
            "match_result": self.result_model.predict_proba(X),
            "btts": self.btts_model.predict_proba(X)[:, 1],
        }
        for line, model in self.ou_models.items():
            out[f"over_{line}"] = model.predict_proba(X)[:, 1]
        return out
