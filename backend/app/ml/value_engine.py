"""
Layer 4: value-bet identification, staking, and betting backtests.

Market probabilities: bookmaker odds carry an overround (sum of 1/odds > 1).
Shin's method removes it by modelling the margin as protection against a
fraction z of insider money, which shades longshots more than favourites --
matching the favourite-longshot bias better than proportional normalisation.

Value:   EV = p_model * odds - 1, against the price actually offered (not the
         de-margined fair price -- the margin is a real cost of the bet).
Guards:  a selection is only flagged/bet with EV >= MIN_EV (2.5%) and a
         probability edge over the de-vigged market <= MAX_EDGE_CAP (7 pts).
Staking: fractional Kelly, f = fraction * (p*o - 1) / (o - 1), capped.

Outcome keys are flat and unique across markets so one dict can hold a whole
fixture: home_win / draw / away_win, over_{line} / under_{line}, btts_yes /
btts_no. A "book" is one set of mutually exclusive outcomes (1X2, one O/U
line, BTTS) -- the unit Shin de-margins and the backtest bets at most once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from app.ml.calibration import _shin, evaluate, shin_implied_probabilities  # noqa: F401 -- re-export
from app.ml.football_goal_range_model import OU_LINES

MIN_EV = 0.025
# Guardrail: the largest believable gap between our probability and the
# de-vigged market's. Bigger "edges" are overwhelmingly model error (mostly
# on longshots), so they are never flagged or bet, however large the EV.
MAX_EDGE_CAP = 0.07
KELLY_FRACTION = 0.25
MAX_STAKE = 0.05  # cap on the staked bankroll fraction, whatever Kelly says
UNITS_PER_BANKROLL = 100.0  # 1 unit = 1% of bankroll

BOOKS: dict[str, tuple[str, ...]] = {
    "match_result": ("home_win", "draw", "away_win"),
    **{f"over_under_{line}": (f"over_{line}", f"under_{line}") for line in OU_LINES},
    "btts": ("btts_yes", "btts_no"),
}


# ---------------------------------------------------------------------------
# Core pricing functions (Shin de-vigging lives in calibration.py, since the
# ensemble's market shrinkage needs it too; re-exported here)
# ---------------------------------------------------------------------------

def calculate_expected_value(p_model, decimal_odds):
    """Expected profit per unit staked: p * odds - 1."""
    ev = np.asarray(p_model, dtype=float) * np.asarray(decimal_odds, dtype=float) - 1.0
    return float(ev) if ev.ndim == 0 else ev


def calculate_kelly_stake(p_model, decimal_odds, fraction: float = KELLY_FRACTION, max_stake: float = MAX_STAKE):
    """Fractional-Kelly bankroll fraction; 0 when the bet has no edge."""
    p = np.asarray(p_model, dtype=float)
    o = np.asarray(decimal_odds, dtype=float)
    full = (p * o - 1.0) / (o - 1.0)
    stake = np.clip(fraction * full, 0.0, max_stake)
    return float(stake) if stake.ndim == 0 else stake


# ---------------------------------------------------------------------------
# Serving: one fixture's value assessment
# ---------------------------------------------------------------------------

def flat_model_probabilities(match_result: dict | None, over_under: dict | None, btts: dict | None) -> dict[str, float]:
    """Nested API-shape market probabilities -> flat outcome keys."""
    flat: dict[str, float] = {}
    flat.update(match_result or {})
    flat.update(over_under or {})
    if btts:
        flat["btts_yes"], flat["btts_no"] = btts["yes"], btts["no"]
    return flat


def market_probabilities(odds: Mapping[str, float]) -> dict[str, float]:
    """Shin fair probabilities for every complete book in a flat odds dict."""
    fair: dict[str, float] = {}
    for outcomes in BOOKS.values():
        if all(o in odds and odds[o] and odds[o] > 1.0 for o in outcomes):
            fair.update(shin_implied_probabilities({o: odds[o] for o in outcomes}))
    return {k: round(v, 5) for k, v in fair.items()}


def assess_value(
    model_probs: Mapping[str, float],
    odds: Mapping[str, float],
    min_ev: float = MIN_EV,
    kelly_fraction: float = KELLY_FRACTION,
    max_edge: float = MAX_EDGE_CAP,
) -> dict:
    """EV for every outcome we both price and have odds for, plus the single
    best value selection -- highest EV >= min_ev whose probability edge over
    the de-vigged market is <= max_edge -- and its Kelly stake in units."""
    ev = {
        k: round(calculate_expected_value(model_probs[k], o), 5)
        for k, o in odds.items()
        if k in model_probs and o and o > 1.0
    }
    fair = market_probabilities(odds)
    # An outcome from an incomplete book has no fair price to check the edge
    # against, so it can't pass the guardrail.
    candidates = {
        k: v for k, v in ev.items()
        if v >= min_ev and k in fair and model_probs[k] - fair[k] <= max_edge
    }
    best = max(candidates, key=candidates.get) if candidates else None
    stake = (
        round(calculate_kelly_stake(model_probs[best], odds[best], kelly_fraction) * UNITS_PER_BANKROLL, 2)
        if best
        else None
    )
    return {
        "expected_value": ev or None,
        "is_value_bet": best is not None,
        "value_selection": best,
        "recommended_stake": stake,
    }


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def bootstrap_roi_test(
    profits: np.ndarray,
    stakes: np.ndarray,
    clusters: np.ndarray | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """One-sided bootstrap test of H0: ROI <= 0, plus a 95% percentile CI.

    Resamples whole clusters (matches) rather than bets, since a 1X2 and an
    O/U bet on the same match are not independent. The null distribution
    comes from profits shifted so the sample ROI is exactly 0."""
    profits = np.asarray(profits, dtype=float)
    stakes = np.asarray(stakes, dtype=float)
    if len(profits) == 0 or stakes.sum() <= 0:
        return {
            "roi": None, "p_value": None, "ci_low": None, "ci_high": None,
            "n_boot": n_boot, "boot_rois": np.array([]),
        }
    roi = profits.sum() / stakes.sum()
    null_profits = profits - roi * stakes

    if clusters is None:
        clusters = np.arange(len(profits))
    codes, inv = np.unique(clusters, return_inverse=True)
    k = len(codes)
    c_profit = np.bincount(inv, weights=profits, minlength=k)
    c_null = np.bincount(inv, weights=null_profits, minlength=k)
    c_stake = np.bincount(inv, weights=stakes, minlength=k)

    rng = np.random.default_rng(seed)
    boot_roi, null_roi = [], []
    for size in np.diff(np.r_[0:n_boot:1000, n_boot]):  # chunks bound the (size, k) matrix
        counts = rng.multinomial(k, np.full(k, 1.0 / k), size=size)  # resample weights per cluster
        boot_stake = counts @ c_stake
        ok = boot_stake > 0
        boot_roi.append((counts @ c_profit)[ok] / boot_stake[ok])
        null_roi.append((counts @ c_null)[ok] / boot_stake[ok])
    boot_roi, null_roi = np.concatenate(boot_roi), np.concatenate(null_roi)
    return {
        "roi": float(roi),
        "p_value": float((np.sum(null_roi >= roi) + 1) / (len(null_roi) + 1)),
        "ci_low": float(np.percentile(boot_roi, 2.5)),
        "ci_high": float(np.percentile(boot_roi, 97.5)),
        "n_boot": n_boot,
        "boot_rois": boot_roi,
    }


@dataclass
class BacktestResult:
    bets: pd.DataFrame
    summary: dict
    by_market: pd.DataFrame
    by_league: pd.DataFrame
    probability_scores: dict = field(default_factory=dict)


class BettingBacktestEngine:
    """Replays value betting over historical (prediction, odds, outcome) rows.

    Input: long format, one row per outcome of each book --
      match_id, date, league, book, selection, p_model, odds, won (0/1)
    optionally closing_odds (the market's last price, for closing-line value
    when `odds` is an earlier price). Each book of each match gets at most one
    bet: its highest-EV outcome with EV >= min_ev and a probability edge over
    the de-vigged price (p_model - p_market) <= max_edge.

    Stakes are fractions of a fixed starting bankroll (no compounding), so
    ROI stays comparable across bets and runs. staking="flat" bets 1 unit."""

    REQUIRED = ("match_id", "date", "league", "book", "selection", "p_model", "odds", "won")

    def __init__(
        self,
        min_ev: float = MIN_EV,
        kelly_fraction: float = KELLY_FRACTION,
        max_stake: float = MAX_STAKE,
        staking: str = "kelly",
        max_odds: float | None = None,
        max_edge: float | None = MAX_EDGE_CAP,
    ):
        if staking not in ("kelly", "flat"):
            raise ValueError(f"unknown staking: {staking}")
        self.min_ev = min_ev
        self.max_edge = max_edge
        self.kelly_fraction = kelly_fraction
        self.max_stake = max_stake
        self.staking = staking
        self.max_odds = max_odds

    def _complete_books(self, rows: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.REQUIRED if c not in rows.columns]
        if missing:
            raise ValueError(f"backtest rows missing columns: {missing}")
        df = rows.dropna(subset=["p_model", "odds", "won"]).copy()
        df = df[df["odds"] > 1.0]
        # Shin needs every outcome of a book; drop books with a missing price.
        expected = df["book"].map(lambda b: len(BOOKS.get(b, ())) or np.nan)
        size = df.groupby(["match_id", "book"])["selection"].transform("size")
        return df[size == expected].reset_index(drop=True)

    def _add_fair(self, df: pd.DataFrame, odds_col: str, out_col: str) -> None:
        df[out_col] = np.nan
        for _, idx in df.groupby(["match_id", "book"]).groups.items():
            odds = df.loc[idx, odds_col].to_numpy(dtype=float)
            if np.all(odds > 1.0):
                df.loc[idx, out_col] = _shin(odds)[0]

    def run(self, rows: pd.DataFrame) -> BacktestResult:
        df = self._complete_books(rows)
        self._add_fair(df, "odds", "p_market")
        has_close = "closing_odds" in df.columns and df["closing_odds"].notna().any()
        if has_close:
            self._add_fair(df, "closing_odds", "p_close")

        df["ev"] = calculate_expected_value(df["p_model"].to_numpy(), df["odds"].to_numpy())
        df["edge"] = df["p_model"] - df["p_market"]
        eligible = df[df["ev"] >= self.min_ev]
        if self.max_edge is not None:
            eligible = eligible[eligible["edge"] <= self.max_edge]
        if self.max_odds:
            eligible = eligible[eligible["odds"] <= self.max_odds]
        best = eligible.sort_values("ev", ascending=False).drop_duplicates(["match_id", "book"])
        bets = best.sort_values(["date", "match_id", "book"], kind="mergesort").copy()

        if self.staking == "kelly":
            bets["stake"] = calculate_kelly_stake(
                bets["p_model"].to_numpy(), bets["odds"].to_numpy(), self.kelly_fraction, self.max_stake
            ) * UNITS_PER_BANKROLL
        else:
            bets["stake"] = 1.0
        bets["profit"] = np.where(bets["won"] == 1, bets["stake"] * (bets["odds"] - 1.0), -bets["stake"])
        if has_close:
            # Closing-line value: EV of the price taken, judged by the closing market.
            bets["clv"] = bets["odds"] * bets["p_close"] - 1.0

        return BacktestResult(
            bets=bets.reset_index(drop=True),
            summary=self._summary(bets, df),
            by_market=self._breakdown(bets, "book"),
            by_league=self._breakdown(bets, "league"),
            probability_scores=self._probability_scores(df),
        )

    @staticmethod
    def _stats(bets: pd.DataFrame) -> dict:
        staked = float(bets["stake"].sum())
        profit = float(bets["profit"].sum())
        return {
            "n_bets": int(len(bets)),
            "staked_units": round(staked, 2),
            "profit_units": round(profit, 2),
            "roi": profit / staked if staked > 0 else None,
            "win_rate": float(bets["won"].mean()) if len(bets) else None,
            "avg_odds": float(bets["odds"].mean()) if len(bets) else None,
            "avg_ev": float(bets["ev"].mean()) if len(bets) else None,
            "avg_edge": float(bets["edge"].mean()) if len(bets) else None,
            "avg_clv": float(bets["clv"].mean()) if "clv" in bets and len(bets) else None,
        }

    def _summary(self, bets: pd.DataFrame, evaluated: pd.DataFrame) -> dict:
        equity = bets["profit"].cumsum()
        drawdown = float((equity.cummax().clip(lower=0) - equity).max()) if len(bets) else 0.0
        return {
            "n_matches": int(evaluated["match_id"].nunique()),
            "n_books": int(evaluated.groupby(["match_id", "book"]).ngroups),
            **self._stats(bets),
            "max_drawdown_units": round(drawdown, 2),
            "breakeven_win_rate": float((1.0 / bets["odds"]).mean()) if len(bets) else None,
            "staking": self.staking,
            "min_ev": self.min_ev,
            "max_edge": self.max_edge,
            "kelly_fraction": self.kelly_fraction if self.staking == "kelly" else None,
        }

    def _breakdown(self, bets: pd.DataFrame, key: str) -> pd.DataFrame:
        if bets.empty:
            return pd.DataFrame()
        return pd.DataFrame([{key: k, **self._stats(g)} for k, g in bets.groupby(key)]).set_index(key)

    @staticmethod
    def _probability_scores(df: pd.DataFrame) -> dict:
        """Brier / log loss of the model vs. the de-margined market at the
        price bet (and at closing, when given), per book type, on every
        evaluated book (bet on or not). Two-outcome books are scored on their
        first outcome (over / btts_yes), as in calibration.py."""
        out = {}
        for book, g in df.groupby("book"):
            order = {sel: i for i, sel in enumerate(BOOKS[book])}
            g = g.assign(_k=g["selection"].map(order)).sort_values(["match_id", "_k"])
            k = len(order)
            y = g["won"].to_numpy(dtype=float).reshape(-1, k).argmax(axis=1)
            sources = {"model": "p_model", "market": "p_market"}
            if "p_close" in g and g["p_close"].notna().all():
                sources["closing_market"] = "p_close"
            scores = {}
            for name, col in sources.items():
                P = g[col].to_numpy(dtype=float).reshape(-1, k)
                P = P / P.sum(axis=1, keepdims=True)
                scores[name] = evaluate(P[:, 0], y == 0) if k == 2 else evaluate(P, y)
            out[book] = {"n": int(len(y)), **scores}
        return out
