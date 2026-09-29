import numpy as np
import pytest

from app.ml.football_goal_range_model import double_chance_probabilities, extra_market_probabilities


def _samples():
    rng = np.random.default_rng(0)
    return rng.poisson(1.6, 20000), rng.poisson(1.1, 20000)


def test_pairs_sum_to_one_and_partitions_are_exhaustive():
    h, a = _samples()
    m = extra_market_probabilities(h, a)
    for side in ("home", "away"):
        for line in (0.5, 1.5, 2.5):
            assert m["team_totals"][f"{side}_over_{line}"] + m["team_totals"][f"{side}_under_{line}"] == pytest.approx(1)
    assert m["odd_even"]["odd"] + m["odd_even"]["even"] == pytest.approx(1)
    assert sum(m["x12_btts"].values()) == pytest.approx(1)
    for line in (1.5, 2.5, 3.5):
        assert sum(v for k, v in m["ou_btts"].items() if f"_{line}_" in k) == pytest.approx(1)


def test_combos_are_consistent_with_marginals():
    h, a = _samples()
    m = extra_market_probabilities(h, a)
    btts_yes = float(np.mean((h > 0) & (a > 0)))
    assert m["x12_btts"]["home_win_btts_yes"] + m["x12_btts"]["draw_btts_yes"] + m["x12_btts"][
        "away_win_btts_yes"
    ] == pytest.approx(btts_yes)
    assert m["ou_btts"]["over_2.5_btts_yes"] + m["ou_btts"]["under_2.5_btts_yes"] == pytest.approx(btts_yes)
    # a scoreline with 0 goals for one side can't be BTTS
    assert m["x12_btts"]["draw_btts_yes"] <= float(np.mean(h == a))


def test_odd_even_and_team_total_known_values():
    h, a = np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1])  # totals 0,1,3,4
    m = extra_market_probabilities(h, a)
    assert m["odd_even"]["odd"] == 0.5
    assert m["team_totals"]["home_over_1.5"] == 0.5
    assert m["team_totals"]["away_over_0.5"] == 0.5


def test_double_chance():
    dc = double_chance_probabilities({"home_win": 0.5, "draw": 0.3, "away_win": 0.2})
    assert dc == {"home_draw": pytest.approx(0.8), "draw_away": pytest.approx(0.5), "home_away": pytest.approx(0.7)}
