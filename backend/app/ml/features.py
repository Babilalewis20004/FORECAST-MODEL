"""
Layer 1: feature pipeline for the direct GBDT classifiers (Model B).

One code path serves both training and prediction: every target row (a
historical match being trained on, or an upcoming fixture being predicted)
looks up each side's state as of that team's last *real* match strictly
before the target's date, via pd.merge_asof(allow_exact_matches=False).
Because state is only ever computed from completed matches and joined
strictly-before, a target can never see its own result or anything later
-- and an upcoming fixture gets exactly the features a historical match on
the same date would have had, with no separate "prediction-time" rebuild.

Model A (football_goal_range_model.py) keeps its own feature set; this
module only feeds Model B. MARKET_FEATURES are the exception to "state from
completed matches": bookmaker prices passed in per target row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.calibration import shin_implied_probabilities
from app.ml.football_goal_range_model import GOAL_RANGES, OU_LINES, extra_flat_masks
from app.ml.rolling_utils import trailing_match_counts

WINDOWS = (5, 10)
MIN_PERIODS = 3  # observations needed before a rolling mean is trusted
H2H_WINDOW = 6
REST_DAYS_CAP = 30  # beyond this it's off-season/international-break noise
CONGESTION_WINDOW_DAYS = 14  # fixture pile-up: matches played in the trailing N days
REFEREE_WINDOW = 20  # matches, all-competition, same referee
MIN_REFEREE_PERIODS = 3
SEASON_GAP_DAYS = 75  # a longer gap between a team's matches starts a new season (any league calendar)
VENUE_WINDOW = 10  # home-only / away-only form window

ELO_BASE = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 60.0
ELO_SCORE_SLOPE = 1.5  # logistic slope mapping an (xG-adjusted) goal margin to a 0..1 result

# (long-format column, home-side source, away-side source)
_STAT_SOURCES = [
    ("goals_for", "home_goals", "away_goals"),
    ("goals_against", "away_goals", "home_goals"),
    ("xg_for", "home_xg", "away_xg"),
    ("xg_against", "away_xg", "home_xg"),
    ("shots_for", "home_shots", "away_shots"),
    ("shots_against", "away_shots", "home_shots"),
    ("sot_for", "home_shots_on_target", "away_shots_on_target"),
    ("poss_for", "home_possession", "away_possession"),
]
_STAT_COLS = [c for c, _, _ in _STAT_SOURCES]
_OPTIONAL_SOURCES = [
    "home_xg", "away_xg", "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_possession", "away_possession",
]

_TEAM_FEATURES = (
    [f"{c}_r{w}" for c in _STAT_COLS for w in WINDOWS]
    + ["conv_r10", "elo", "rest_days", "n_matches", "congestion14", "season_ppg", "season_gd_pg", "season_n"]
)
# Venue-split form: the home side's last VENUE_WINDOW *home* matches and the
# away side's last VENUE_WINDOW *away* matches (prefixed h_/a_ like the rest).
_VENUE_FEATURES = ["venue_gf_r10", "venue_ga_r10", "venue_ppg_r10"]
# De-vigged (Shin) pre-match market probabilities. Unlike everything above
# they aren't derived from match history: they arrive as columns on the
# matches/targets frames (historical: match_odds' opening price; upcoming:
# the latest API-Football snapshot), NaN where no odds exist -- XGBoost
# routes missing values natively. Pre-kickoff information, so no leakage.
MARKET_FEATURES = ["opening_prob_home", "opening_prob_draw", "opening_prob_away", "opening_prob_over_2.5"]

FEATURE_COLUMNS: list[str] = (
    [f"h_{f}" for f in _TEAM_FEATURES]
    + [f"a_{f}" for f in _TEAM_FEATURES]
    + [f"h_{f}" for f in _VENUE_FEATURES]
    + [f"a_{f}" for f in _VENUE_FEATURES]
    + ["ppg_diff", "elo_diff", "h2h_n", "h2h_gd", "h2h_total", "h2h_btts", "ref_avg_goals", "ref_btts_rate"]
    + MARKET_FEATURES
)


def market_features_from_odds(odds: dict | None) -> dict[str, float]:
    """MARKET_FEATURES from flat odds (home_win/draw/away_win/over_2.5/
    under_2.5); NaN for any book that isn't completely priced."""
    out = {c: np.nan for c in MARKET_FEATURES}
    if not odds:
        return out
    mr = [odds.get(k) for k in ("home_win", "draw", "away_win")]
    if all(v and v > 1.0 for v in mr):
        out["opening_prob_home"], out["opening_prob_draw"], out["opening_prob_away"] = shin_implied_probabilities(mr)
    ou = [odds.get("over_2.5"), odds.get("under_2.5")]
    if all(v and v > 1.0 for v in ou):
        out["opening_prob_over_2.5"] = float(shin_implied_probabilities(ou)[0])
    return out


def _to_ns(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).astype("datetime64[ns]")


def _prepare_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Completed matches only, sorted by date, with every optional stat column present."""
    df = matches[matches["home_goals"].notna() & matches["away_goals"].notna()].copy()
    df["date"] = _to_ns(df["date"])
    for col in _OPTIONAL_SOURCES:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "referee" not in df.columns:
        df["referee"] = None
    df["home_goals"] = df["home_goals"].astype(float)
    df["away_goals"] = df["away_goals"].astype(float)
    return df.sort_values("date", kind="mergesort").reset_index(drop=True)


def _elo_after(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sequential Elo where the match signal blends the actual goal margin with
    the xG margin when xG is available (so a lucky 1-0 on 0.4 vs 2.1 xG moves
    ratings less than a deserved one). Returns each side's rating *after* the
    match; the pre-match rating for a later target comes from the merge_asof."""
    ratings: dict[str, float] = {}
    h_after = np.empty(len(df))
    a_after = np.empty(len(df))
    cols = zip(df["home_team"], df["away_team"], df["home_goals"], df["away_goals"], df["home_xg"], df["away_xg"])
    for i, (h, a, hg, ag, hx, ax) in enumerate(cols):
        rh = ratings.get(h, ELO_BASE)
        ra = ratings.get(a, ELO_BASE)
        margin = hg - ag
        if not (np.isnan(hx) or np.isnan(ax)):
            margin = 0.5 * margin + 0.5 * (hx - ax)
        score_h = 1.0 / (1.0 + np.exp(-ELO_SCORE_SLOPE * margin))
        exp_h = 1.0 / (1.0 + 10 ** ((ra - rh - ELO_HOME_ADV) / 400))
        delta = ELO_K * (np.log(abs(margin) + 1) + 1) * (score_h - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
        h_after[i] = ratings[h]
        a_after[i] = ratings[a]
    return h_after, a_after


def _team_state(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, completed match): the team's rolling form *including*
    that match -- i.e. its state going into its next game."""
    h_after, a_after = _elo_after(df)

    home = pd.DataFrame({"team": df["home_team"], "date": df["date"], "elo": h_after})
    away = pd.DataFrame({"team": df["away_team"], "date": df["date"], "elo": a_after})
    for col, h_src, a_src in _STAT_SOURCES:
        home[col] = df[h_src].to_numpy()
        away[col] = df[a_src].to_numpy()

    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"], kind="mergesort")
    long["_goals_w_shots"] = long["goals_for"].where(long["shots_for"].notna())
    g = long.groupby("team", sort=False)

    out = long[["team", "date", "elo"]].copy()
    for col in _STAT_COLS:
        for w in WINDOWS:
            out[f"{col}_r{w}"] = g[col].transform(lambda s, w=w: s.rolling(w, min_periods=MIN_PERIODS).mean())

    # Fixture congestion: matches played in the trailing CONGESTION_WINDOW_DAYS,
    # including this one -- consistent with every other column in `out`
    # representing state *going into the team's next game*.
    out["congestion14"] = g["date"].transform(lambda s: pd.Series(trailing_match_counts(s, CONGESTION_WINDOW_DAYS), index=s.index))

    goals_sum = g["_goals_w_shots"].transform(lambda s: s.rolling(10, min_periods=MIN_PERIODS).sum())
    shots_sum = g["shots_for"].transform(lambda s: s.rolling(10, min_periods=MIN_PERIODS).sum())
    out["conv_r10"] = goals_sum / shots_sum.replace(0, np.nan)
    out["n_matches"] = g.cumcount() + 1

    # Season-to-date table form, from results only. A team's season restarts
    # after a SEASON_GAP_DAYS gap, which works for any league calendar.
    gf, ga = long["goals_for"], long["goals_against"]
    long["_pts"] = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
    long["_gd"] = gf - ga
    new_season = g["date"].diff().dt.days.gt(SEASON_GAP_DAYS).fillna(False)
    long["_season"] = new_season.groupby(long["team"], sort=False).cumsum()
    sg = long.groupby(["team", "_season"], sort=False)
    out["season_n"] = sg.cumcount() + 1
    out["season_ppg"] = sg["_pts"].cumsum() / out["season_n"]
    out["season_gd_pg"] = sg["_gd"].cumsum() / out["season_n"]
    out["last_match_date"] = out["date"]
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


def _venue_state(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """One row per (team, match played at `side` venue): that team's rolling
    VENUE_WINDOW goals for/against and points per game at that venue only,
    including this match (same convention as _team_state)."""
    own, opp = ("home_goals", "away_goals") if side == "home" else ("away_goals", "home_goals")
    v = pd.DataFrame({
        "team": df[f"{side}_team"].to_numpy(),
        "date": df["date"].to_numpy(),
        "gf": df[own].to_numpy(),
        "ga": df[opp].to_numpy(),
    })
    v["pts"] = np.where(v["gf"] > v["ga"], 3.0, np.where(v["gf"] == v["ga"], 1.0, 0.0))
    v = v.sort_values(["team", "date"], kind="mergesort")
    g = v.groupby("team", sort=False)
    out = v[["team", "date"]].copy()
    for col, name in (("gf", "venue_gf_r10"), ("ga", "venue_ga_r10"), ("pts", "venue_ppg_r10")):
        out[name] = g[col].transform(lambda s: s.rolling(VENUE_WINDOW, min_periods=MIN_PERIODS).mean())
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


def _h2h_state(df: pd.DataFrame) -> pd.DataFrame:
    """One row per completed match per unordered team pair: the pair's last
    H2H_WINDOW meetings *including* this one. gd is from the perspective of
    the alphabetically-first team (pair_a) so it's venue-independent."""
    pair_a = np.where(df["home_team"] <= df["away_team"], df["home_team"], df["away_team"])
    pair_b = np.where(df["home_team"] <= df["away_team"], df["away_team"], df["home_team"])
    gd = df["home_goals"] - df["away_goals"]
    h2h = pd.DataFrame({
        "pair": pd.Series(pair_a) + "|" + pd.Series(pair_b),
        "pair_a": pair_a,
        "date": df["date"].to_numpy(),
        "gd_a": np.where(df["home_team"].to_numpy() == pair_a, gd, -gd),
        "total": (df["home_goals"] + df["away_goals"]).to_numpy(),
        "btts": ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(float).to_numpy(),
    }).sort_values(["pair", "date"], kind="mergesort")

    g = h2h.groupby("pair", sort=False)
    out = h2h[["pair", "pair_a", "date"]].copy()
    for col in ("gd_a", "total", "btts"):
        out[f"h2h_{col}"] = g[col].transform(lambda s: s.rolling(H2H_WINDOW, min_periods=1).mean())
    out["h2h_n"] = g.cumcount() + 1
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


def _referee_state(df: pd.DataFrame) -> pd.DataFrame:
    """One row per completed match with a known referee: rolling
    REFEREE_WINDOW-match average of that referee's total goals and BTTS rate
    (across every competition they've officiated), *including* this match --
    same "state going into the next game" convention as _team_state/_h2h_state,
    relying on the target join's allow_exact_matches=False for leak safety."""
    empty = pd.DataFrame(columns=["referee", "date", "ref_avg_goals", "ref_btts_rate"])
    if "referee" not in df.columns:
        return empty
    d = df[df["referee"].notna() & (df["referee"].astype(str).str.strip() != "")]
    if d.empty:
        return empty

    ref = pd.DataFrame({
        "referee": d["referee"].to_numpy(),
        "date": d["date"].to_numpy(),
        "total_goals": (d["home_goals"] + d["away_goals"]).to_numpy(),
        "btts": ((d["home_goals"] > 0) & (d["away_goals"] > 0)).astype(float).to_numpy(),
    }).sort_values(["referee", "date"], kind="mergesort")

    g = ref.groupby("referee", sort=False)
    ref["ref_avg_goals"] = g["total_goals"].transform(lambda s: s.rolling(REFEREE_WINDOW, min_periods=MIN_REFEREE_PERIODS).mean())
    ref["ref_btts_rate"] = g["btts"].transform(lambda s: s.rolling(REFEREE_WINDOW, min_periods=MIN_REFEREE_PERIODS).mean())
    return ref[["referee", "date", "ref_avg_goals", "ref_btts_rate"]].sort_values("date", kind="mergesort").reset_index(drop=True)


def build_feature_frame(matches: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """
    Features for arbitrary target rows (date, home_team, away_team), computed
    only from matches strictly before each target's date. Works identically
    for historical matches (training) and unplayed fixtures (prediction).
    Returns `targets` (original order, extra columns kept) plus FEATURE_COLUMNS.
    """
    history = _prepare_matches(matches)
    state = _team_state(history)
    venue_states = {"h": _venue_state(history, "home"), "a": _venue_state(history, "away")}
    h2h = _h2h_state(history)

    t = targets.reset_index(drop=True).copy()
    t["date"] = _to_ns(t["date"])
    t["_tid"] = np.arange(len(t))
    left = t[["_tid", "date", "home_team", "away_team"]].sort_values("date", kind="mergesort")

    state_cols = [c for c in state.columns if c not in ("team", "date")]
    for side, team_col in (("h", "home_team"), ("a", "away_team")):
        merged = pd.merge_asof(
            left[["_tid", "date", team_col]].rename(columns={team_col: "team"}),
            state,
            on="date",
            by="team",
            allow_exact_matches=False,
            direction="backward",
        ).set_index("_tid")
        side_feats = merged[state_cols].copy()
        side_feats["rest_days"] = (merged["date"] - merged["last_match_date"]).dt.days.clip(upper=REST_DAYS_CAP)
        side_feats["n_matches"] = side_feats["n_matches"].fillna(0)
        side_feats["congestion14"] = side_feats["congestion14"].fillna(0)
        side_feats["elo"] = side_feats["elo"].fillna(ELO_BASE)
        side_feats = side_feats.drop(columns=["last_match_date"]).add_prefix(f"{side}_")
        t = t.join(side_feats, on="_tid")

        venue = pd.merge_asof(
            left[["_tid", "date", team_col]].rename(columns={team_col: "team"}),
            venue_states[side],
            on="date",
            by="team",
            allow_exact_matches=False,
            direction="backward",
        ).set_index("_tid")[_VENUE_FEATURES].add_prefix(f"{side}_")
        t = t.join(venue, on="_tid")

    pair_a = np.where(left["home_team"] <= left["away_team"], left["home_team"], left["away_team"])
    pair_b = np.where(left["home_team"] <= left["away_team"], left["away_team"], left["home_team"])
    h2h_left = left[["_tid", "date", "home_team"]].assign(pair=pd.Series(pair_a, index=left.index) + "|" + pd.Series(pair_b, index=left.index))
    h2h_merged = pd.merge_asof(
        h2h_left, h2h, on="date", by="pair", allow_exact_matches=False, direction="backward"
    ).set_index("_tid")
    sign = np.where(h2h_merged["home_team"] == h2h_merged["pair_a"], 1.0, -1.0)
    h2h_feats = pd.DataFrame(
        {
            "h2h_n": h2h_merged["h2h_n"].fillna(0),
            "h2h_gd": h2h_merged["h2h_gd_a"] * sign,  # from the target's home side
            "h2h_total": h2h_merged["h2h_total"],
            "h2h_btts": h2h_merged["h2h_btts"],
        },
        index=h2h_merged.index,
    )
    t = t.join(h2h_feats, on="_tid")

    ref_state = _referee_state(history)
    if not ref_state.empty and "referee" in t.columns:
        ref_left = t[["_tid", "date", "referee"]].sort_values("date", kind="mergesort")
        ref_merged = pd.merge_asof(
            ref_left, ref_state, on="date", by="referee", allow_exact_matches=False, direction="backward"
        ).set_index("_tid")
        t = t.join(ref_merged[["ref_avg_goals", "ref_btts_rate"]], on="_tid")
    else:
        t["ref_avg_goals"] = np.nan
        t["ref_btts_rate"] = np.nan

    t["elo_diff"] = t["h_elo"] - t["a_elo"]
    t["ppg_diff"] = t["h_season_ppg"] - t["a_season_ppg"]
    for col in MARKET_FEATURES:
        if col not in t.columns:
            t[col] = np.nan
    # Stable dtype regardless of whether any target lacked history (a column
    # with no NaNs would otherwise come out int64 in one call, float64 in another).
    t[FEATURE_COLUMNS] = t[FEATURE_COLUMNS].astype("float64")

    # dict.fromkeys: market columns may already be among targets' own columns.
    return t.drop(columns=["_tid"])[list(dict.fromkeys(list(targets.columns) + FEATURE_COLUMNS))]


def build_training_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """Every completed match with its pre-match features plus outcome labels
    for each market Model B (and the ensemble evaluation) needs."""
    history = _prepare_matches(matches)
    market = [c for c in MARKET_FEATURES if c in history.columns]
    targets = history[["date", "home_team", "away_team", "home_goals", "away_goals", "referee", *market]]
    frame = build_feature_frame(history, targets)

    hg, ag = frame["home_goals"], frame["away_goals"]
    total = hg + ag
    frame["y_result"] = np.select([hg > ag, hg == ag], [0, 1], default=2).astype(int)
    frame["y_btts"] = ((hg > 0) & (ag > 0)).astype(int)
    for line in OU_LINES:
        frame[f"y_over_{line}"] = (total > line).astype(int)
    for label, (lo, hi) in GOAL_RANGES.items():
        frame[f"y_range_{label}"] = ((total >= lo) & (total <= hi)).astype(int)
    for market, mask in extra_flat_masks(hg, ag).items():
        frame[f"y_{market}"] = mask.astype(int)
    return frame
