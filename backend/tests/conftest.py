import pandas as pd
import pytest

from app.ml.football_goal_range_model import _make_synthetic_league


@pytest.fixture(scope="session")
def synthetic_matches() -> pd.DataFrame:
    """~450 matches, one per date, Poisson goals from fixed team strengths."""
    df = _make_synthetic_league("League Two", n_teams=8, n_seasons=8, seed=7)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)
