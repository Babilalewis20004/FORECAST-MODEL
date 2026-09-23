import numpy as np
import pandas as pd
import pytest

from app.ml.features import FEATURE_COLUMNS, build_feature_frame, build_training_frame


def test_features_never_see_same_day_or_later_results(synthetic_matches):
    """Rewriting every result from date D onwards must not change the features
    of any match on or before D."""
    cutoff = synthetic_matches["date"].iloc[len(synthetic_matches) // 2]
    tampered = synthetic_matches.copy()
    later = tampered["date"] >= cutoff
    tampered.loc[later, "home_goals"] += 3
    tampered.loc[later, "away_goals"] = 0

    original = build_training_frame(synthetic_matches)
    changed = build_training_frame(tampered)
    upto = original["date"] <= cutoff
    pd.testing.assert_frame_equal(
        original.loc[upto, FEATURE_COLUMNS].reset_index(drop=True),
        changed.loc[upto, FEATURE_COLUMNS].reset_index(drop=True),
    )
    # sanity: the tampering did change features after the cutoff
    assert not original.loc[~upto, FEATURE_COLUMNS].equals(changed.loc[~upto, FEATURE_COLUMNS])


def test_prediction_features_match_training_features(synthetic_matches):
    """Train/serve parity: predicting a fixture from only the matches before it
    yields exactly the features the training frame computed for that match."""
    training = build_training_frame(synthetic_matches)
    for match_date in synthetic_matches["date"].drop_duplicates().iloc[-5:]:
        history = synthetic_matches[synthetic_matches["date"] < match_date]
        targets = synthetic_matches.loc[
            synthetic_matches["date"] == match_date, ["date", "home_team", "away_team"]
        ]
        served = build_feature_frame(history, targets)
        trained = training[training["date"] == match_date]
        pd.testing.assert_frame_equal(
            served[FEATURE_COLUMNS].reset_index(drop=True),
            trained[FEATURE_COLUMNS].reset_index(drop=True),
        )


def test_rest_days_h2h_perspective_and_elo():
    matches = pd.DataFrame(
        [
            {"date": "2024-01-01", "home_team": "A", "away_team": "B", "home_goals": 3, "away_goals": 0},
            {"date": "2024-01-08", "home_team": "C", "away_team": "A", "home_goals": 1, "away_goals": 1},
        ]
    )
    target = pd.DataFrame([{"date": "2024-01-15", "home_team": "B", "away_team": "A"}])
    row = build_feature_frame(matches, target).iloc[0]

    assert row["h_rest_days"] == 14 and row["a_rest_days"] == 7
    assert row["h_n_matches"] == 1 and row["a_n_matches"] == 2
    assert row["h2h_n"] == 1
    assert row["h2h_gd"] == pytest.approx(-3.0)  # from B's (the target home side's) view
    assert row["h2h_total"] == pytest.approx(3.0)
    assert row["h2h_btts"] == pytest.approx(0.0)
    assert row["a_elo"] > row["h_elo"]


def test_unseen_teams_and_missing_optional_columns(synthetic_matches):
    """Synthetic data has no xG/shots columns at all; brand-new teams have no
    history. Both must yield NaN/default features, not errors."""
    target = pd.DataFrame([{"date": "2099-01-01", "home_team": "Nobody FC", "away_team": "Nowhere Utd"}])
    row = build_feature_frame(synthetic_matches, target).iloc[0]
    assert set(FEATURE_COLUMNS) <= set(row.index)
    assert row["h_n_matches"] == 0 and row["h2h_n"] == 0
    assert row["h_elo"] == 1500.0
    assert np.isnan(row["h_xg_for_r5"])
