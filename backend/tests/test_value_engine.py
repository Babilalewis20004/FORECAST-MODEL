import numpy as np
import pandas as pd
import pytest

from app.ml.value_engine import (
    BettingBacktestEngine,
    assess_value,
    bootstrap_roi_test,
    calculate_expected_value,
    calculate_kelly_stake,
    market_probabilities,
    shin_implied_probabilities,
)
from app.services.scrape_service import parse_odds


def test_shin_sums_to_one_and_removes_margin():
    odds = [1.80, 3.60, 4.75]  # booksum ~1.04
    p = shin_implied_probabilities(odds)
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p < 1.0 / np.array(odds))


def test_shin_shades_longshots_more_than_proportional():
    odds = np.array([1.25, 6.0, 15.0])
    shin = shin_implied_probabilities(odds)
    proportional = (1 / odds) / (1 / odds).sum()
    assert shin[0] > proportional[0]  # favourite gets more
    assert shin[2] < proportional[2]  # longshot gets less


def test_shin_fair_book_is_unchanged_and_dict_roundtrip():
    fair = {"over_2.5": 2.0, "under_2.5": 2.0}
    assert shin_implied_probabilities(fair) == pytest.approx({"over_2.5": 0.5, "under_2.5": 0.5})


def test_shin_rejects_bad_odds():
    with pytest.raises(ValueError):
        shin_implied_probabilities([1.0, 2.0])


def test_expected_value_and_kelly():
    assert calculate_expected_value(0.55, 2.0) == pytest.approx(0.10)
    # full Kelly (0.55*2-1)/(2-1) = 0.10 -> quarter = 0.025
    assert calculate_kelly_stake(0.55, 2.0) == pytest.approx(0.025)
    assert calculate_kelly_stake(0.40, 2.0) == 0.0  # no edge -> no bet
    assert calculate_kelly_stake(0.9, 3.0, fraction=1.0) == pytest.approx(0.05)  # capped
    np.testing.assert_allclose(calculate_expected_value([0.5, 0.25], [2.2, 3.0]), [0.10, -0.25])


def test_assess_value_picks_best_outcome():
    probs = {"home_win": 0.50, "draw": 0.25, "away_win": 0.25, "over_2.5": 0.52, "under_2.5": 0.48}
    odds = {"home_win": 2.20, "draw": 3.40, "away_win": 3.60, "over_2.5": 1.90, "under_2.5": 1.95}
    out = assess_value(probs, odds)
    assert out["expected_value"]["home_win"] == pytest.approx(0.10)
    assert out["is_value_bet"] and out["value_selection"] == "home_win"
    # quarter Kelly of (0.5*2.2-1)/1.2 = 0.0833 -> 2.08% -> 2.08 units
    assert out["recommended_stake"] == pytest.approx(2.08, abs=0.01)


def test_assess_value_no_value():
    out = assess_value({"home_win": 0.4}, {"home_win": 2.4})
    assert out["is_value_bet"] is False and out["recommended_stake"] is None


def test_market_probabilities_only_complete_books():
    fair = market_probabilities({"home_win": 2.0, "draw": 3.5, "away_win": 4.0, "over_2.5": 1.9})
    assert set(fair) == {"home_win", "draw", "away_win"}
    assert sum(fair.values()) == pytest.approx(1.0, abs=1e-4)


def test_assess_value_edge_cap_blocks_unrealistic_longshot():
    # Fair away ~0.25; model says 0.40 -> EV huge, but a 15-point edge is model error.
    odds = {"home_win": 1.80, "draw": 3.60, "away_win": 3.80}
    out = assess_value({"home_win": 0.35, "draw": 0.25, "away_win": 0.40}, odds)
    assert out["expected_value"]["away_win"] > 0.5
    assert out["is_value_bet"] is False
    assert assess_value({"home_win": 0.35, "draw": 0.25, "away_win": 0.40}, odds, max_edge=0.2)["value_selection"] == "away_win"


def test_assess_value_min_ev_is_2_5_percent():
    odds = {"over_2.5": 2.00, "under_2.5": 1.90}  # fair over ~0.487
    assert assess_value({"over_2.5": 0.515, "under_2.5": 0.485}, odds)["is_value_bet"] is True   # EV +3.0%
    assert assess_value({"over_2.5": 0.51, "under_2.5": 0.49}, odds)["is_value_bet"] is False    # EV +2.0%


def test_assess_value_needs_complete_book_for_edge_check():
    # No fair price without the other side of the book -> can't pass the guardrail.
    assert assess_value({"over_2.5": 0.6}, {"over_2.5": 2.0})["is_value_bet"] is False

    item = {"bookmakers": [
        {"name": "10Bet", "bets": [{"name": "Match Winner", "values": [
            {"value": "Home", "odd": "2.00"}, {"value": "Draw", "odd": "3.30"}, {"value": "Away", "odd": "3.80"}]}]},
        {"name": "Pinnacle", "bets": [
            {"name": "Match Winner", "values": [
                {"value": "Home", "odd": "2.05"}, {"value": "Draw", "odd": "3.40"}, {"value": "Away", "odd": "3.90"}]},
            {"name": "Goals Over/Under", "values": [
                {"value": "Over 2.5", "odd": "1.95"}, {"value": "Under 2.5", "odd": "1.93"},
                {"value": "Over 4.5", "odd": "5.0"}]},
            {"name": "Corners", "values": [{"value": "Over 9.5", "odd": "1.9"}]}]},
    ]}
    bookmaker, odds = parse_odds(item)
    assert bookmaker == "Pinnacle"
    assert odds == {"home_win": 2.05, "draw": 3.40, "away_win": 3.90, "over_2.5": 1.95, "under_2.5": 1.93}


def _synthetic_rows(n_matches: int, edge: float, seed: int = 0) -> pd.DataFrame:
    """O/U books where the true P(over) is known; the market misprices it by
    `edge` in logit space and the 'model' knows the truth."""
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0.35, 0.65, n_matches)
    p_mkt = 1 / (1 + np.exp(-(np.log(p_true / (1 - p_true)) - edge * rng.choice([-1, 1], n_matches))))
    margin = 1.04
    won_over = rng.random(n_matches) < p_true
    rows = []
    for i in range(n_matches):
        for sel, p, pm, won in (("over_2.5", p_true[i], p_mkt[i], won_over[i]),
                                ("under_2.5", 1 - p_true[i], 1 - p_mkt[i], not won_over[i])):
            rows.append({"match_id": i, "date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=i // 10),
                         "league": "L", "book": "over_under_2.5", "selection": sel,
                         "p_model": p, "odds": 1 / (pm * margin), "won": int(won)})
    return pd.DataFrame(rows)


def test_backtest_finds_real_edge():
    # The planted edge (~10 points) is above MAX_EDGE_CAP, so disable the cap here.
    result = BettingBacktestEngine(staking="flat", max_edge=None).run(_synthetic_rows(4000, edge=0.4))
    s = result.summary
    assert s["n_bets"] > 1000
    assert s["roi"] > 0.03
    # at most one bet per book
    assert not result.bets.duplicated(["match_id", "book"]).any()
    boot = bootstrap_roi_test(result.bets["profit"], result.bets["stake"], result.bets["match_id"], n_boot=2000)
    assert boot["p_value"] < 0.01 and boot["ci_low"] > 0
    sc = result.probability_scores["over_under_2.5"]
    assert sc["model"]["brier"] < sc["market"]["brier"]


def test_backtest_no_edge_is_not_significant():
    rows = _synthetic_rows(3000, edge=0.0, seed=3)
    rows["p_model"] = np.where(rows["selection"] == "over_2.5", 0.6, 0.4)  # confident noise
    result = BettingBacktestEngine(staking="flat", max_edge=None).run(rows)
    boot = bootstrap_roi_test(result.bets["profit"], result.bets["stake"], result.bets["match_id"], n_boot=2000)
    assert result.summary["roi"] < 0.05
    assert boot["p_value"] > 0.05


def test_backtest_drops_incomplete_books_and_bootstrap_handles_empty():
    rows = _synthetic_rows(10, edge=0.4)
    rows = rows[~((rows["match_id"] == 0) & (rows["selection"] == "under_2.5"))]
    result = BettingBacktestEngine().run(rows)
    assert 0 not in set(result.bets["match_id"])
    assert result.summary["n_matches"] == 9
    assert bootstrap_roi_test(np.array([]), np.array([]))["p_value"] is None
