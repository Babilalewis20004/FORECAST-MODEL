"""Tiny helpers shared by Model A (football_goal_range_model.py) and Model B
(features.py), which otherwise deliberately keep separate feature-engineering
code paths (see features.py's module docstring)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_match_counts(dates: pd.Series, window_days: int) -> np.ndarray:
    """For one team's own match dates (must be sorted ascending), the count
    of that team's matches falling in (date - window_days, date] for each
    date -- i.e. fixture pile-up, including the match itself. Vectorized via
    searchsorted rather than a per-row date-window scan."""
    arr = pd.to_datetime(dates).to_numpy()
    lo = arr - np.timedelta64(window_days, "D")
    left = np.searchsorted(arr, lo, side="right")
    right = np.searchsorted(arr, arr, side="right")
    return (right - left).astype(float)
