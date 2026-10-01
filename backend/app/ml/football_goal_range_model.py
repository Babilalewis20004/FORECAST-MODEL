"""
Football Goal-Range Prediction Pipeline
========================================

Predicts probabilities for total-match-goal ranges (1-3, 1-4, 2-3, 2-4, 2-5,
2-6, 3-5, 3-6), plus Match Result (1X2), Over/Under (1.5/2.5/3.5), and Both
Teams To Score (BTTS) -- all derived from the same simulated joint score
distribution, so no separate model is trained per market.

Approach
--------
1. Train two XGBoost regressors per league tier (home goals, away goals)
   using a Poisson objective — this predicts expected goals (lambda), not
   a raw point score.
2. Estimate an overdispersion parameter per league tier from residuals
   (football goals are overdispersed relative to Poisson — variance > mean).
3. For a given fixture, jointly sample home/away goals from correlated
   Negative Binomial distributions parameterised by the predicted lambdas +
   dispersion, via Monte Carlo simulation (simulate_joint_score_distribution).
4. Derive whichever market you need from that one joint sample set — goal
   ranges, 1X2, Over/Under, BTTS — one model serves every market, and
   probabilities are internally consistent with each other.

Data assumptions
-----------------
You need a historical match-level dataframe with (at minimum):

    date, league, home_team, away_team, home_goals, away_goals

Optional but strongly recommended (used if present, skipped if not):

    home_xg, away_xg          -- shot-quality data (PL/La Liga/Championship)
    home_shots, away_shots
    home_shots_on_target, away_shots_on_target
    home_corners, away_corners
    home_cards, away_cards

The pipeline is tier-aware: leagues with sparse/no xG (League One,
League Two, Scottish Premiership) automatically fall back to a
goals/shots-based feature set; leagues with xG use the richer set.

Run this file directly for a worked example on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import xgboost as xgb
from scipy.stats import nbinom, norm

from app.ml.rolling_utils import trailing_match_counts


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Leagues you said you want covered, split by data-richness tier.
# xG-rich tier: full shot-quality features.
# goals-only tier: fall back to goals/shots/cards rolling features.
XG_RICH_LEAGUES = {
    "Premier League", "La Liga", "EFL Championship", "Bundesliga", "Serie A", "Ligue 1",
    "Primeira Liga", "Eredivisie", "Süper Lig", "Jupiler Pro League",
}
# Liga Profesional Argentina: xG coverage (~32% overall, and inconsistent
# year to year -- 99.7% in 2024 vs 50% in 2025) is well below every xG-rich
# league's. Super League 1 (Greece): xG coverage never exceeds ~30% in any
# season checked (2023/2024/2025 alike) -- not a ramp-up like every xG-rich
# league shows, just a persistent near-absence. Both kept here rather than
# overclaiming shot-quality signal that isn't reliably there.
GOALS_ONLY_LEAGUES = {
    "League One", "League Two", "Scottish Premiership", "Liga Profesional Argentina", "Super League 1",
    # xG is absent before 2025 (Ligue 2: never) or patchy, so these stay goals-only.
    # J1 League was tested with xG features on (xG filled from TheStatsAPI to 88%
    # coverage) and scored worse on 24 of 37 markets, so it stays here too.
    "Segunda División", "2. Bundesliga", "Serie B", "Ligue 2", "J1 League",
}

ROLLING_WINDOW_RICH = 10       # PL/La Liga/Championship: longer window OK
ROLLING_WINDOW_LOWER = 6       # lower leagues: shorter window (more squad churn)

GOAL_RANGES = {
    "1-3": (1, 3),
    "1-4": (1, 4),
    "2-3": (2, 3),
    "2-4": (2, 4),
    "2-5": (2, 5),
    "2-6": (2, 6),
    "3-5": (3, 5),
    "3-6": (3, 6),
}

# Over/Under goal lines for the Over/Under market.
OU_LINES: tuple[float, ...] = (1.5, 2.5, 3.5)

MAX_GOALS_SIDE = 10  # cap per-side goals when building the distribution
N_SIMULATIONS = 20000

CONGESTION_WINDOW_DAYS = 14  # fixture pile-up: matches played in the trailing N days
REFEREE_WINDOW = 20  # matches, all-competition, same referee
MIN_REFEREE_PERIODS = 3


# ---------------------------------------------------------------------------
# 1. Feature engineering
# ---------------------------------------------------------------------------

def _rolling_team_stats(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Build rolling (pre-match, no leakage) attacking/defensive features for
    each team, from a long-format match dataframe.

    Expects df sorted by date, with one row per match (home + away teams).
    Returns df with added columns for both home_* and away_* rolling stats.
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Reshape to team-level long format so rolling stats are per-team,
    # not per-fixture (each team's history spans both home and away games).
    home = df[["date", "home_team", "home_goals", "away_goals"]].rename(
        columns={"home_team": "team", "home_goals": "goals_for", "away_goals": "goals_against"}
    )
    home["is_home"] = 1
    away = df[["date", "away_team", "away_goals", "home_goals"]].rename(
        columns={"away_team": "team", "away_goals": "goals_for", "home_goals": "goals_against"}
    )
    away["is_home"] = 0

    optional_cols = [
        ("xg_for", "home_xg", "away_xg"),
        ("xg_against", "away_xg", "home_xg"),
        ("shots_for", "home_shots", "away_shots"),
        ("shots_against", "away_shots", "home_shots"),
        ("sot_for", "home_shots_on_target", "away_shots_on_target"),
        ("sot_against", "away_shots_on_target", "home_shots_on_target"),
        ("corners_for", "home_corners", "away_corners"),
        ("corners_against", "away_corners", "home_corners"),
        ("cards_for", "home_cards", "away_cards"),
        ("cards_against", "away_cards", "home_cards"),
        ("poss_for", "home_possession", "away_possession"),
        ("poss_against", "away_possession", "home_possession"),
    ]
    for out_col, home_src, away_src in optional_cols:
        if home_src in df.columns:
            home[out_col] = df[home_src]
            away[out_col] = df[away_src]

    long_df = pd.concat([home, away], ignore_index=True).sort_values("date")

    stat_cols = [c for c in long_df.columns if c not in ("date", "team", "is_home")]

    long_df = long_df.sort_values(["team", "date"])
    rolled = long_df.groupby("team")[stat_cols].transform(
        lambda s: s.shift(1).rolling(window, min_periods=3).mean()
    )
    rolled.columns = [f"roll_{c}" for c in rolled.columns]
    long_df = pd.concat([long_df[["date", "team", "is_home"]], rolled], axis=1)

    # Fixture congestion: matches played in the trailing CONGESTION_WINDOW_DAYS,
    # shift(1) so a row only sees pile-up strictly before this match.
    long_df["congestion14"] = long_df.groupby("team")["date"].transform(
        lambda s: pd.Series(trailing_match_counts(s, CONGESTION_WINDOW_DAYS), index=s.index)
    )
    long_df["congestion14"] = long_df.groupby("team")["congestion14"].shift(1).fillna(0.0)

    # Split back into home/away lookups keyed by (date, team)
    home_feats = long_df.add_prefix("h_")
    away_feats = long_df.add_prefix("a_")

    df = df.merge(
        home_feats, left_on=["date", "home_team"], right_on=["h_date", "h_team"], how="left"
    )
    df = df.merge(
        away_feats, left_on=["date", "away_team"], right_on=["a_date", "a_team"], how="left"
    )
    drop_cols = [c for c in ("h_date", "h_team", "h_is_home", "a_date", "a_team", "a_is_home") if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def _elo_ratings(df: pd.DataFrame, k: float = 20.0, base: float = 1500.0) -> pd.DataFrame:
    """Simple sequential Elo, pre-match rating attached as a feature (no leakage).

    Iterates via zipped numpy arrays rather than df.iterrows() -- iterrows()
    reconstructs a full pandas Series per row, which dominates runtime once
    this gets called per-fixture over a multi-thousand-row history (e.g. once
    per upcoming fixture when generating predictions for a whole league).
    Same sequential recurrence, same output, just without the per-row Series
    overhead.
    """
    sorted_df = df.sort_values("date")
    homes = sorted_df["home_team"].to_numpy()
    aways = sorted_df["away_team"].to_numpy()
    home_goals_arr = sorted_df["home_goals"].to_numpy(dtype=float)
    away_goals_arr = sorted_df["away_goals"].to_numpy(dtype=float)

    ratings: dict[str, float] = {}
    home_elo = np.empty(len(sorted_df))
    away_elo = np.empty(len(sorted_df))

    for i in range(len(sorted_df)):
        h, a = homes[i], aways[i]
        hg, ag = home_goals_arr[i], away_goals_arr[i]
        rh = ratings.get(h, base)
        ra = ratings.get(a, base)
        home_elo[i] = rh
        away_elo[i] = ra

        exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))
        if hg > ag:
            score_h = 1.0
        elif hg == ag:
            score_h = 0.5
        else:
            score_h = 0.0

        margin_mult = np.log(abs(hg - ag) + 1) + 1
        ratings[h] = rh + k * margin_mult * (score_h - exp_h)
        ratings[a] = ra + k * margin_mult * ((1 - score_h) - (1 - exp_h))

    out = sorted_df.copy()
    out["home_elo"] = home_elo
    out["away_elo"] = away_elo
    return out.sort_index()


def _add_referee_tendency(df: pd.DataFrame) -> pd.DataFrame:
    """ref_avg_goals / ref_btts_rate: rolling (shift(1), no leakage) average of
    total goals and BTTS rate in this referee's past REFEREE_WINDOW matches,
    across every competition they've officiated. NaN until a referee has
    MIN_REFEREE_PERIODS matches, or if the referee is unknown -- XGBoost
    handles missing values natively, same as xG/shots."""
    if "referee" not in df.columns:
        df["ref_avg_goals"] = np.nan
        df["ref_btts_rate"] = np.nan
        return df

    df = df.sort_values("date").reset_index(drop=True)
    ref = pd.DataFrame({
        "referee": df["referee"].where(df["referee"].astype(str).str.strip() != ""),
        "date": df["date"],
        "total_goals": df["home_goals"] + df["away_goals"],
        "btts": ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(float),
    })
    sorted_ref = ref.sort_values(["referee", "date"], kind="mergesort")
    rolled = sorted_ref.groupby("referee")[["total_goals", "btts"]].transform(
        lambda s: s.shift(1).rolling(REFEREE_WINDOW, min_periods=MIN_REFEREE_PERIODS).mean()
    )
    df["ref_avg_goals"] = rolled["total_goals"].reindex(df.index)
    df["ref_btts_rate"] = rolled["btts"].reindex(df.index)
    return df


def build_features(df: pd.DataFrame, league_tier: str) -> pd.DataFrame:
    """
    league_tier: 'rich' or 'lower' — controls rolling window length and
    which optional columns are expected to matter.
    """
    window = ROLLING_WINDOW_RICH if league_tier == "rich" else ROLLING_WINDOW_LOWER
    df = _rolling_team_stats(df, window=window)
    df = _elo_ratings(df)
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df = _add_referee_tendency(df)
    return df


def compute_team_state(df: pd.DataFrame, league_tier: str) -> dict[str, dict[str, float]]:
    """
    Per-team snapshot (rolling stats + Elo) as of *right after* each team's
    most recent match in df -- i.e. exactly the pre-match state
    prepare_fixture_features() would compute for a hypothetical next fixture,
    but for every team in one pass instead of rebuilding the whole
    rolling/elo pipeline from scratch per fixture. Real match history never
    changes across a batch of upcoming fixtures, so this is safe to compute
    once per league and reuse for every fixture in that league.

    Returns {team: {"roll_goals_for": ..., ..., "elo": ...}}.
    """
    window = ROLLING_WINDOW_RICH if league_tier == "rich" else ROLLING_WINDOW_LOWER
    df = df.sort_values("date")

    home = df[["date", "home_team", "home_goals", "away_goals"]].rename(
        columns={"home_team": "team", "home_goals": "goals_for", "away_goals": "goals_against"}
    )
    away = df[["date", "away_team", "away_goals", "home_goals"]].rename(
        columns={"away_team": "team", "away_goals": "goals_for", "home_goals": "goals_against"}
    )
    optional_cols = [
        ("xg_for", "home_xg", "away_xg"),
        ("xg_against", "away_xg", "home_xg"),
        ("shots_for", "home_shots", "away_shots"),
        ("shots_against", "away_shots", "home_shots"),
        ("sot_for", "home_shots_on_target", "away_shots_on_target"),
        ("sot_against", "away_shots_on_target", "home_shots_on_target"),
        ("corners_for", "home_corners", "away_corners"),
        ("corners_against", "away_corners", "home_corners"),
        ("cards_for", "home_cards", "away_cards"),
        ("cards_against", "away_cards", "home_cards"),
        ("poss_for", "home_possession", "away_possession"),
        ("poss_against", "away_possession", "home_possession"),
    ]
    for out_col, home_src, away_src in optional_cols:
        if home_src in df.columns:
            home[out_col] = df[home_src]
            away[out_col] = df[away_src]

    long_df = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"])
    stat_cols = [c for c in long_df.columns if c not in ("date", "team")]

    state: dict[str, dict[str, float]] = {}
    for team, group in long_df.groupby("team", sort=False):
        tail = group.tail(window)
        counts = tail[stat_cols].count()
        means = tail[stat_cols].mean()
        means = means.where(counts >= 3)  # mirrors rolling(..., min_periods=3)
        state[team] = {f"roll_{c}": means[c] for c in stat_cols}
        sorted_dates = group["date"].sort_values()
        state[team]["congestion14"] = float(trailing_match_counts(sorted_dates, CONGESTION_WINDOW_DAYS)[-1])

    # Final Elo rating per team after all matches in df -- same recurrence as
    # _elo_ratings, just keeping the end-of-loop ratings dict instead of the
    # per-row pre-match snapshots.
    ratings: dict[str, float] = {}
    for h, a, hg, ag in zip(df["home_team"], df["away_team"], df["home_goals"], df["away_goals"]):
        rh = ratings.get(h, 1500.0)
        ra = ratings.get(a, 1500.0)
        exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))
        if hg > ag:
            score_h = 1.0
        elif hg == ag:
            score_h = 0.5
        else:
            score_h = 0.0
        margin_mult = np.log(abs(hg - ag) + 1) + 1
        ratings[h] = rh + 20.0 * margin_mult * (score_h - exp_h)
        ratings[a] = ra + 20.0 * margin_mult * ((1 - score_h) - (1 - exp_h))

    for team, elo in ratings.items():
        state.setdefault(team, {})["elo"] = elo

    return state


def compute_referee_state(df: pd.DataFrame, window: int = REFEREE_WINDOW) -> dict[str, dict[str, float]]:
    """Per-referee snapshot (avg total goals, BTTS rate over their last
    `window` matches in df) as of the end of df -- the compute_team_state
    counterpart for referees, used by predict_fixture_fast. Empty dict if
    `df` has no referee column or no known referees."""
    if "referee" not in df.columns:
        return {}
    d = df[df["referee"].notna() & (df["referee"].astype(str).str.strip() != "")]
    if d.empty:
        return {}

    state: dict[str, dict[str, float]] = {}
    for ref, group in d.groupby("referee", sort=False):
        tail = group.tail(window)
        if len(tail) < MIN_REFEREE_PERIODS:
            continue
        state[ref] = {
            "ref_avg_goals": float((tail["home_goals"] + tail["away_goals"]).mean()),
            "ref_btts_rate": float(((tail["home_goals"] > 0) & (tail["away_goals"] > 0)).mean()),
        }
    return state


def fixture_feature_row(
    model: "LeagueGoalModel",
    team_state: dict,
    home_team: str,
    away_team: str,
    referee_state: dict | None = None,
    referee: str | None = None,
) -> dict:
    """Builds one model.feature_cols-aligned row from precomputed team state
    (see compute_team_state) -- the fast counterpart to
    prepare_fixture_features for batch prediction over many fixtures."""
    home_s = team_state.get(home_team, {})
    away_s = team_state.get(away_team, {})

    row: dict[str, float] = {}
    for col in model.feature_cols:
        if col.startswith("h_roll_"):
            row[col] = home_s.get(col[len("h_"):], np.nan)
        elif col.startswith("a_roll_"):
            row[col] = away_s.get(col[len("a_"):], np.nan)

    row["home_elo"] = home_s.get("elo", 1500.0)
    row["away_elo"] = away_s.get("elo", 1500.0)
    row["elo_diff"] = row["home_elo"] - row["away_elo"]
    row["h_congestion14"] = home_s.get("congestion14", 0.0)
    row["a_congestion14"] = away_s.get("congestion14", 0.0)

    ref_s = (referee_state or {}).get(referee, {}) if referee else {}
    row["ref_avg_goals"] = ref_s.get("ref_avg_goals", np.nan)
    row["ref_btts_rate"] = ref_s.get("ref_btts_rate", np.nan)
    return row


# ---------------------------------------------------------------------------
# 2. Model: per-league-tier Poisson XGBoost for home/away expected goals
# ---------------------------------------------------------------------------

@dataclass
class LeagueGoalModel:
    league: str
    tier: str  # 'rich' or 'lower'
    feature_cols: list[str] = field(default_factory=list)
    home_model: Optional[xgb.XGBRegressor] = None
    away_model: Optional[xgb.XGBRegressor] = None
    home_dispersion: float = 1.0  # NB dispersion (variance/mean ratio - 1), fit from residuals
    away_dispersion: float = 1.0
    test_feats_: Optional[pd.DataFrame] = field(default=None, repr=False)

    def fit(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        test_size: fraction of matches (chronologically last) held out of
        training entirely. Rolling/Elo features are still built over the
        full history so the holdout keeps real pre-match context, but
        neither the XGBoost fit nor the dispersion (theta) estimate ever
        sees these rows. Call evaluate() afterwards to get genuine
        out-of-sample calibration numbers instead of in-sample ones.
        """
        feats = build_features(df, self.tier)
        candidate_cols = [c for c in feats.columns if c.startswith(("h_roll_", "a_roll_"))]
        candidate_cols += ["home_elo", "away_elo", "elo_diff", "h_congestion14", "a_congestion14"]
        candidate_cols += [c for c in ("ref_avg_goals", "ref_btts_rate") if c in feats.columns]
        # keep only columns with enough non-null coverage (handles missing xG etc.)
        self.feature_cols = [
            c for c in candidate_cols
            if feats[c].notna().mean() > 0.5
        ]
        feats = feats.dropna(subset=self.feature_cols + ["home_goals", "away_goals"])
        feats = feats.sort_values("date").reset_index(drop=True)

        n_test = int(len(feats) * test_size)
        if n_test > 0:
            train_feats, self.test_feats_ = feats.iloc[:-n_test], feats.iloc[-n_test:].reset_index(drop=True)
        else:
            train_feats, self.test_feats_ = feats, feats.iloc[0:0]

        X_train = train_feats[self.feature_cols]

        self.home_model = xgb.XGBRegressor(
            objective="count:poisson", n_estimators=300, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        )
        self.away_model = xgb.XGBRegressor(
            objective="count:poisson", n_estimators=300, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        )
        self.home_model.fit(X_train, train_feats["home_goals"])
        self.away_model.fit(X_train, train_feats["away_goals"])

        # Fit dispersion from TRAIN-ONLY residuals: NB variance = mean + mean^2/theta
        # theta (dispersion param) estimated via method-of-moments.
        self.home_dispersion = self._fit_dispersion(
            self.home_model.predict(X_train), train_feats["home_goals"].values
        )
        self.away_dispersion = self._fit_dispersion(
            self.away_model.predict(X_train), train_feats["away_goals"].values
        )
        return feats  # for downstream use (e.g. Elo/rolling lookups for new fixtures)

    def evaluate(self) -> dict:
        """
        Out-of-sample calibration check on the holdout carved out by fit()'s
        test_size. For every held-out match (never used to train the
        regressors or the dispersion estimate), predicts range probabilities
        and scores them against the real outcome with a Brier score per
        range (0 = perfect, lower is better; well-calibrated ranges on real
        football data typically land well under ~0.2). Also reports mean
        absolute error of the point lambda vs. actual goals as a secondary
        diagnostic.
        """
        if self.test_feats_ is None or self.test_feats_.empty:
            raise RuntimeError("No holdout set available — call fit(df, test_size>0) first.")

        n = len(self.test_feats_)
        brier_sums = {label: 0.0 for label in GOAL_RANGES}
        abs_err_home = abs_err_away = 0.0

        for i in range(n):
            row_df = self.test_feats_.iloc[[i]]
            lam_h, lam_a = self.predict_lambdas(row_df)
            samples = simulate_total_goals_distribution(
                lam_h, lam_a, self.home_dispersion, self.away_dispersion
            )
            probs = goal_range_probabilities(samples)
            actual_total = row_df["home_goals"].iloc[0] + row_df["away_goals"].iloc[0]
            for label, (lo, hi) in GOAL_RANGES.items():
                outcome = 1.0 if lo <= actual_total <= hi else 0.0
                brier_sums[label] += (probs[label] - outcome) ** 2
            abs_err_home += abs(lam_h - row_df["home_goals"].iloc[0])
            abs_err_away += abs(lam_a - row_df["away_goals"].iloc[0])

        return {
            "n_test_matches": n,
            "brier_by_range": {label: round(s / n, 4) for label, s in brier_sums.items()},
            "mean_abs_error_lambda_home": round(abs_err_home / n, 3),
            "mean_abs_error_lambda_away": round(abs_err_away / n, 3),
        }

    @staticmethod
    def _fit_dispersion(pred_mean: np.ndarray, actual: np.ndarray) -> float:
        resid_var = np.var(actual - pred_mean)
        mean_mean = np.mean(pred_mean)
        excess_var = max(resid_var - mean_mean, 1e-3)
        theta = (mean_mean ** 2) / excess_var
        return float(np.clip(theta, 1.0, 100.0))  # theta -> inf recovers Poisson

    def predict_lambdas(self, X_row: pd.DataFrame) -> tuple[float, float]:
        lam_h = float(self.home_model.predict(X_row[self.feature_cols])[0])
        lam_a = float(self.away_model.predict(X_row[self.feature_cols])[0])
        return max(lam_h, 0.05), max(lam_a, 0.05)


# ---------------------------------------------------------------------------
# 3. Convert lambdas -> full total-goals distribution via NB simulation
# ---------------------------------------------------------------------------

def _nb_params(mean: float, theta: float) -> tuple[float, float]:
    """Convert (mean, theta) to scipy's (n, p) parameterisation for nbinom."""
    p = theta / (theta + mean)
    n = theta
    return n, p


def simulate_joint_score_distribution(
    lam_home: float, lam_away: float,
    theta_home: float, theta_away: float,
    n_sims: int = N_SIMULATIONS,
    rho: float = -0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo simulate the JOINT (home_goals, away_goals) draw, with a
    negative correlation between the two sides (Dixon-Coles style low-score
    adjustment: teams that concede fewer also tend to face teams scoring
    fewer, and low-score draws are slightly more common than independence
    implies).

    rho: target correlation, applied via a Gaussian copula — correlated
    standard normals are mapped through each side's own NB quantile
    function (inverse CDF). This keeps each marginal an *exact*
    NegativeBinomial(mean, theta) draw (so the calibrated dispersion is
    preserved precisely) while inducing real dependence between the two
    sides, including at the 0-0 / 1-1 level. rho=0 recovers independence.

    Returns the joint (home_goals, away_goals) sample arrays -- this is the
    single source of truth every market (goal range, total, 1X2, O/U, BTTS)
    derives from, so they stay mutually consistent with each other.
    """
    rng = np.random.default_rng(42)

    cov = [[1.0, rho], [rho, 1.0]]
    z = rng.multivariate_normal([0.0, 0.0], cov, size=n_sims)
    u_home = np.clip(norm.cdf(z[:, 0]), 1e-9, 1 - 1e-9)
    u_away = np.clip(norm.cdf(z[:, 1]), 1e-9, 1 - 1e-9)

    n_h, p_h = _nb_params(lam_home, theta_home)
    n_a, p_a = _nb_params(lam_away, theta_away)

    home_goals = nbinom.ppf(u_home, n_h, p_h)
    away_goals = nbinom.ppf(u_away, n_a, p_a)

    home_goals = np.clip(home_goals, 0, MAX_GOALS_SIDE).astype(int)
    away_goals = np.clip(away_goals, 0, MAX_GOALS_SIDE).astype(int)

    return home_goals, away_goals


def simulate_total_goals_distribution(
    lam_home: float, lam_away: float,
    theta_home: float, theta_away: float,
    n_sims: int = N_SIMULATIONS,
    rho: float = -0.1,
) -> np.ndarray:
    """Total-goals-only view of simulate_joint_score_distribution, kept for
    callers (e.g. LeagueGoalModel.evaluate()) that only need the sum."""
    home_goals, away_goals = simulate_joint_score_distribution(
        lam_home, lam_away, theta_home, theta_away, n_sims, rho
    )
    return home_goals + away_goals


def goal_range_probabilities(total_goals_samples: np.ndarray) -> dict[str, float]:
    n = len(total_goals_samples)
    probs = {}
    for label, (lo, hi) in GOAL_RANGES.items():
        probs[label] = float(np.mean((total_goals_samples >= lo) & (total_goals_samples <= hi)))
    return probs


def full_goal_distribution(total_goals_samples: np.ndarray, max_k: int = 8) -> dict[int, float]:
    """P(total goals = k) for k = 0..max_k, with max_k+ bucketed together."""
    dist = {}
    for k in range(max_k):
        dist[k] = float(np.mean(total_goals_samples == k))
    dist[f"{max_k}+"] = float(np.mean(total_goals_samples >= max_k))
    return dist


def match_result_probabilities(home_goals_samples: np.ndarray, away_goals_samples: np.ndarray) -> dict[str, float]:
    """1X2: P(home win) / P(draw) / P(away win) from the joint score draw."""
    return {
        "home_win": float(np.mean(home_goals_samples > away_goals_samples)),
        "draw": float(np.mean(home_goals_samples == away_goals_samples)),
        "away_win": float(np.mean(home_goals_samples < away_goals_samples)),
    }


def over_under_probabilities(
    total_goals_samples: np.ndarray, lines: tuple[float, ...] = OU_LINES
) -> dict[str, float]:
    """Over/Under probabilities for each line in `lines` (e.g. 1.5/2.5/3.5),
    keyed 'over_1.5'/'under_1.5' etc. -- derived from the same total-goals
    samples goal_range_probabilities() uses, so any line works, not just
    the three in OU_LINES."""
    probs: dict[str, float] = {}
    for line in lines:
        over = float(np.mean(total_goals_samples > line))
        probs[f"over_{line}"] = over
        probs[f"under_{line}"] = 1.0 - over
    return probs


def btts_probabilities(home_goals_samples: np.ndarray, away_goals_samples: np.ndarray) -> dict[str, float]:
    """Both Teams To Score: P(yes) / P(no) from the joint score draw."""
    yes = float(np.mean((home_goals_samples > 0) & (away_goals_samples > 0)))
    return {"yes": yes, "no": 1.0 - yes}


TEAM_TOTAL_LINES = (0.5, 1.5, 2.5)

# Flat market keys for the derived markets, "<category>:<outcome>" -- one
# entry per *independently calibrated* binary event (a Platt calibrator each,
# see ensemble.train_ensemble). Complements (team-total under, odd/even's
# "even") are implied by 1 - p and don't get their own key; the 1X2/O-U x BTTS
# partitions are calibrated per outcome and then renormalised to sum to 1
# (see nest_extra).
def _ou_btts_impossible(side: str, line: float, btts: str) -> bool:
    return side == "under" and btts == "yes" and line < 2


def extra_flat_masks(home_goals: np.ndarray, away_goals: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean outcome mask per flat derived-market key, from a (home, away)
    goal array pair -- samples for a prediction, or actual results for the
    training labels, so both use exactly the same definitions."""
    total = home_goals + away_goals
    both = (home_goals > 0) & (away_goals > 0)
    masks: dict[str, np.ndarray] = {}
    for side, goals in (("home", home_goals), ("away", away_goals)):
        for line in TEAM_TOTAL_LINES:
            masks[f"team_totals:{side}_over_{line}"] = goals > line
    results = {
        "home_win": home_goals > away_goals,
        "draw": home_goals == away_goals,
        "away_win": home_goals < away_goals,
    }
    for name, mask in results.items():
        masks[f"x12_btts:{name}_btts_yes"] = mask & both
        masks[f"x12_btts:{name}_btts_no"] = mask & ~both
    for line in OU_LINES:
        for side, mask in (("over", total > line), ("under", total <= line)):
            # BTTS needs at least two goals, so "under 1.5 & BTTS yes" can
            # never happen -- not a market, and nothing to calibrate.
            if not _ou_btts_impossible(side, line, "yes"):
                masks[f"ou_btts:{side}_{line}_btts_yes"] = mask & both
            masks[f"ou_btts:{side}_{line}_btts_no"] = mask & ~both
    masks["odd_even:odd"] = total % 2 == 1
    return masks


EXTRA_MARKET_KEYS = list(extra_flat_masks(np.zeros(1, dtype=int), np.zeros(1, dtype=int)))


def extra_flat_probabilities(home_goals: np.ndarray, away_goals: np.ndarray) -> dict[str, float]:
    return {k: float(np.mean(m)) for k, m in extra_flat_masks(home_goals, away_goals).items()}


def nest_extra(flat: dict) -> dict[str, dict[str, float]]:
    """Flat derived-market probabilities -> nested by category, adding the
    implied complements and renormalising the complete partitions:

      team_totals: "home_over_1.5", "home_under_1.5", ...
      x12_btts:    "home_win_btts_yes", "draw_btts_no", ... (sums to 1)
      ou_btts:     "over_2.5_btts_yes", "under_2.5_btts_no", ... (per line sums to 1)
      odd_even:    "odd", "even"

    Only categories present in `flat` are returned."""
    out: dict[str, dict[str, float]] = {}
    for key, p in flat.items():
        category, name = key.split(":", 1)
        out.setdefault(category, {})[name] = float(p)

    tt = out.get("team_totals", {})
    for name in [n for n in tt if "_over_" in n]:
        tt[name.replace("_over_", "_under_")] = 1.0 - tt[name]

    oe = out.get("odd_even", {})
    if "odd" in oe:
        oe["even"] = 1.0 - oe["odd"]

    def renormalise(probs: dict[str, float], names: list[str]) -> None:
        if all(n in probs for n in names):
            total = sum(probs[n] for n in names)
            if total > 0:
                for n in names:
                    probs[n] /= total

    renormalise(out.get("x12_btts", {}), [f"{r}_btts_{b}" for r in ("home_win", "draw", "away_win") for b in ("yes", "no")])
    for line in OU_LINES:
        renormalise(
            out.get("ou_btts", {}),
            [
                f"{s}_{line}_btts_{b}"
                for s in ("over", "under")
                for b in ("yes", "no")
                if not _ou_btts_impossible(s, line, b)
            ],
        )
    return out


def extra_market_probabilities(
    home_goals_samples: np.ndarray, away_goals_samples: np.ndarray
) -> dict[str, dict[str, float]]:
    """Markets derived from the joint score draw that need more than the
    marginals: team totals, 1X2/O-U combined with BTTS, and odd/even total
    (see nest_extra for the key shapes). Raw simulation output; the served
    values come from the ensemble's per-market calibrators when the league
    has them. Double chance is not here -- it's a plain sum of the
    calibrated 1X2 (see double_chance_probabilities)."""
    return nest_extra(extra_flat_probabilities(home_goals_samples, away_goals_samples))


def double_chance_probabilities(match_result: dict[str, float]) -> dict[str, float]:
    """Double chance from a 1X2 dict: 1X (home or draw), X2 (draw or away),
    12 (home or away)."""
    h, d, a = match_result["home_win"], match_result["draw"], match_result["away_win"]
    return {"home_draw": h + d, "draw_away": d + a, "home_away": h + a}


def correct_score_probabilities(
    home_goals_samples: np.ndarray, away_goals_samples: np.ndarray, max_goals: int = 6, top_n: int = 10
) -> dict[str, float]:
    """The `top_n` most likely exact scorelines within a 0..max_goals grid
    (any sample with either side above max_goals is dropped, not bucketed --
    at max_goals=6 that's a vanishingly small tail with no realistic chance
    of making the top N anyway), keyed "h-a", from the same joint score draw
    predict_fixture already produces. Not exhaustive: these top_n
    probabilities don't sum to 1 and are for display only -- callers that
    need a full-coverage probability should use match_result/goal_range_
    probabilities/over_under_probabilities instead."""
    n = len(home_goals_samples)
    in_range = (home_goals_samples <= max_goals) & (away_goals_samples <= max_goals)
    h, a = home_goals_samples[in_range].astype(int), away_goals_samples[in_range].astype(int)
    counts = np.bincount(h * (max_goals + 1) + a, minlength=(max_goals + 1) ** 2)
    probs = counts / n
    entries = [
        (f"{hg}-{ag}", float(probs[hg * (max_goals + 1) + ag]))
        for hg in range(max_goals + 1)
        for ag in range(max_goals + 1)
    ]
    return dict(sorted(entries, key=lambda kv: kv[1], reverse=True)[:top_n])


def decimal_odds(probability: float) -> float | None:
    """Fair (no-margin/no-vig) decimal odds for a probability: 1/p. This is
    the "true" odds implied by the model, not what a bookmaker would price
    (a real bookmaker shades this to build in their margin) -- it's the
    same probability already being served, just in the units bettors read.
    None for a probability of exactly 0 (odds would be infinite)."""
    if probability <= 0:
        return None
    return round(1.0 / probability, 2)


def probabilities_to_odds(probs: dict[str, float]) -> dict[str, float | None]:
    """Maps every value in a probability dict (e.g. from
    goal_range_probabilities/match_result_probabilities/etc.) to its
    decimal_odds() -- same keys, so it's a drop-in parallel to whichever
    probabilities dict it's given."""
    return {label: decimal_odds(p) for label, p in probs.items()}


# ---------------------------------------------------------------------------
# 4. End-to-end predict for a single fixture
# ---------------------------------------------------------------------------

def predict_fixture_fast(
    model: LeagueGoalModel,
    team_state: dict,
    home_team: str,
    away_team: str,
    referee_state: dict | None = None,
    referee: str | None = None,
) -> dict:
    """Fast counterpart to predict_fixture(model, prepare_fixture_features(...))
    for batch use: takes team state precomputed once per league (see
    compute_team_state) instead of rebuilding rolling/elo features from the
    full match history for every single fixture. referee_state (see
    compute_referee_state) + the fixture's own referee are optional -- NaN
    referee features when either is omitted (e.g. no referee assigned yet)."""
    row = fixture_feature_row(model, team_state, home_team, away_team, referee_state, referee)
    row_df = pd.DataFrame([row], columns=model.feature_cols)
    return predict_fixture(model, row_df)


def predict_fixture(
    model: LeagueGoalModel,
    fixture_features_row: pd.DataFrame,
) -> dict:
    lam_h, lam_a = model.predict_lambdas(fixture_features_row)
    home_samples, away_samples = simulate_joint_score_distribution(
        lam_h, lam_a, model.home_dispersion, model.away_dispersion
    )
    total_samples = home_samples + away_samples

    range_probabilities = goal_range_probabilities(total_samples)
    match_result = match_result_probabilities(home_samples, away_samples)
    over_under = over_under_probabilities(total_samples)
    btts = btts_probabilities(home_samples, away_samples)
    # Model A only (like lambda_home/away) -- not part of the ensemble's
    # flat-key blend/shrinkage, since a per-scoreline joint distribution
    # doesn't fit that framework the way the other markets do.
    correct_score = correct_score_probabilities(home_samples, away_samples)
    extra_flat = extra_flat_probabilities(home_samples, away_samples)
    extra_markets = nest_extra(extra_flat)

    return {
        "lambda_home": round(lam_h, 2),
        "lambda_away": round(lam_a, 2),
        "goal_distribution": full_goal_distribution(total_samples),
        "range_probabilities": range_probabilities,
        "match_result": match_result,
        "over_under": over_under,
        "btts": btts,
        "correct_score": correct_score,
        "extra_markets": extra_markets,
        "extra_flat": extra_flat,
        # Fair (no-margin) decimal odds -- 1/probability -- for every market
        # above, same keys as their probability counterparts.
        "odds": {
            "range": probabilities_to_odds(range_probabilities),
            "match_result": probabilities_to_odds(match_result),
            "over_under": probabilities_to_odds(over_under),
            "btts": probabilities_to_odds(btts),
            "correct_score": probabilities_to_odds(correct_score),
        },
    }


def prepare_fixture_features(
    model: LeagueGoalModel,
    history: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date,
) -> pd.DataFrame:
    """
    Build the feature row for a genuinely unplayed fixture (no result yet),
    for use with predict_fixture(). Uses only matches strictly before
    match_date, then runs the *same* build_features pipeline used at
    training time — no separate/divergent feature-construction code path,
    so there's no risk of it drifting out of sync with what the model was
    trained on. home_goals/away_goals for the fixture are unknown (NaN);
    that's safe here because rolling stats/Elo are computed from each
    team's *prior* matches only (shift(1) / sequential update), never from
    the row being predicted.
    """
    match_date = pd.Timestamp(match_date)
    history = history[history["date"] < match_date]
    if history.empty:
        raise ValueError("No historical matches before match_date to build features from.")

    future_row = pd.DataFrame([{
        "date": match_date,
        "league": model.league,
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": np.nan,
        "away_goals": np.nan,
    }])
    augmented = pd.concat([history, future_row], ignore_index=True)
    feats = build_features(augmented, model.tier)

    fixture_feats = feats[
        (feats["date"] == match_date)
        & (feats["home_team"] == home_team)
        & (feats["away_team"] == away_team)
    ]
    if fixture_feats.empty:
        raise ValueError("Could not locate the fixture row after feature building.")

    missing = [c for c in model.feature_cols if c not in fixture_feats.columns]
    if missing:
        raise ValueError(f"Fitted model expects feature columns not present: {missing}")
    return fixture_feats.iloc[[-1]]


# ---------------------------------------------------------------------------
# 5. Worked example on synthetic data (replace with real historical data)
# ---------------------------------------------------------------------------

def _make_synthetic_league(league: str, n_teams: int = 20, n_seasons: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"{league[:3]}_Team{i}" for i in range(n_teams)]
    strengths = rng.normal(1.3, 0.35, n_teams)  # baseline attack rate per team
    dates = pd.date_range("2023-08-01", periods=n_seasons * n_teams * (n_teams - 1), freq="3D")

    rows = []
    d_idx = 0
    for season in range(n_seasons):
        for i, home in enumerate(teams):
            for j, away in enumerate(teams):
                if i == j:
                    continue
                lam_h = max(0.3, strengths[i] * 1.15)  # home advantage bump
                lam_a = max(0.3, strengths[j] * 0.9)
                hg = rng.poisson(lam_h)
                ag = rng.poisson(lam_a)
                rows.append({
                    "date": dates[d_idx % len(dates)] + pd.Timedelta(days=season * 300),
                    "league": league,
                    "home_team": home,
                    "away_team": away,
                    "home_goals": hg,
                    "away_goals": ag,
                })
                d_idx += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # --- Fit on real historical data from matches.csv, one league at a time ---
    from pathlib import Path

    _root_csv = Path(__file__).resolve().parents[3] / "matches.csv"
    df = pd.read_csv(_root_csv, parse_dates=["date"])

    for league_name in sorted(df["league"].unique()):
        history = df[df["league"] == league_name].reset_index(drop=True)
        tier = "rich" if league_name in XG_RICH_LEAGUES else "lower"

        model = LeagueGoalModel(league=league_name, tier=tier)
        fitted_feats = model.fit(history, test_size=0.2)

        n_train = len(fitted_feats) - len(model.test_feats_)
        print(f"\n=== {league_name} ===")
        print(f"trained on {n_train} matches, held out {len(model.test_feats_)} for eval, tier='{tier}'")
        print(f"features used: {model.feature_cols}")
        print(f"home dispersion theta={model.home_dispersion:.1f}, away theta={model.away_dispersion:.1f}")

        # Out-of-sample calibration check on the holdout (never seen during fit)
        eval_results = model.evaluate()
        print(f"\nHoldout evaluation ({eval_results['n_test_matches']} matches):")
        print(f"  mean abs error: home={eval_results['mean_abs_error_lambda_home']}, "
              f"away={eval_results['mean_abs_error_lambda_away']}")
        print("  Brier score by range (0=perfect, lower is better):")
        for label, brier in eval_results["brier_by_range"].items():
            print(f"    {label}: {brier}")

        # Predict a genuinely unplayed fixture: two teams from the league, dated
        # after the last historical match, using only data that precedes it.
        last_date = history["date"].max()
        teams = history["home_team"].unique()
        home_team, away_team = teams[0], teams[1]
        fixture_row = prepare_fixture_features(
            model, history, home_team, away_team, last_date + pd.Timedelta(days=7)
        )
        result = predict_fixture(model, fixture_row)

        print(f"\nExample future-fixture prediction ({home_team} vs {away_team}):")
        print(f"  lambda_home={result['lambda_home']}, lambda_away={result['lambda_away']}")
        print("  Range probabilities:")
        for label, prob in result["range_probabilities"].items():
            print(f"    {label}: {prob:.1%}")
