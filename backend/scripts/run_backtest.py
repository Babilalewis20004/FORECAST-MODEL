"""Offline value-betting backtest on a held-out season.

  python scripts/run_backtest.py                       # 2025/26, shrinkage w=0.70, bet Bet365's opening price
  python scripts/run_backtest.py --shrinkage fitted    # per-league fitted weights (or: none, 0.5, ...)
  python scripts/run_backtest.py --price closing       # bet the closing price instead
  python scripts/run_backtest.py --refit-months 1      # monthly walk-forward refits (slow)
  python scripts/run_backtest.py --min-ev 0.05 --staking flat --out bets.csv

1. Out-of-sample predictions: for each domestic league, the full ensemble
   (Model A + Model B with market features + blend + calibration + fitted
   shrinkage weight, exactly as served) is trained only on matches before
   each window start and scores every match in the window. Features are
   strictly pre-match (see features.py), so no prediction sees its own
   result. Cached per season/refit schedule; pass --retrain to rebuild.
2. Shrinkage: calibrated probabilities are pulled toward Bet365's
   de-vigged *opening* price -- the price that exists when a bet would be
   placed. Shrinking toward the closing price and then scoring against it
   would leak the answer.
3. Betting odds: football-data.co.uk (API-Football only keeps odds ~7 days
   after a fixture), joined to our matches on league + date + score + team
   names. BettingBacktestEngine (EV >= min_ev, edge <= MAX_EDGE_CAP) +
   match-clustered bootstrap_roi_test -> report, then a shrinkage
   comparison (none / 0.70 / fitted) for EPL + La Liga and all leagues.

Markets: 1X2 and O/U 2.5 (football-data has no BTTS odds). International
competitions have no odds source and are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from app.db import SessionLocal
from app.ml.calibration import DEFAULT_SHRINKAGE_WEIGHT, shrink_to_market
from app.ml.ensemble import _model_a_holdout_flat, train_ensemble
from app.ml.features import MARKET_FEATURES
from app.ml.gbdt_models import RESULT_CLASSES
from app.ml.value_engine import (
    KELLY_FRACTION,
    MAX_EDGE_CAP,
    MIN_EV,
    BettingBacktestEngine,
    bootstrap_roi_test,
)
from app.services.model_service import _matches_df, league_tier
from app.services.odds_history import FD_LEAGUES, FD_ODDS, join_odds, load_fd_odds, numeric_col

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "downloaded_files" / "backtest"
VALIDATION_SIMS = 5000
CACHE_VERSION = 2  # bump when the prediction columns/pipeline change
HEADLINE_LEAGUES = ("Premier League", "La Liga")

MR_COLS = [f"p_{c}" for c in RESULT_CLASSES]
MR_MARKET = ["opening_prob_home", "opening_prob_draw", "opening_prob_away"]


# ---------------------------------------------------------------------------
# Out-of-sample predictions
# ---------------------------------------------------------------------------

def _windows(start: pd.Timestamp, end: pd.Timestamp, refit_months: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if refit_months <= 0:
        return [(start, end)]
    edges = list(pd.date_range(start, end, freq=pd.DateOffset(months=refit_months)))
    if edges[-1] < end:
        edges.append(end)
    return list(zip(edges[:-1], edges[1:]))


def predict_league_oos(history: pd.DataFrame, league: str, start, end, refit_months: int) -> pd.DataFrame:
    """Calibrated (unshrunk) probabilities per match, plus the market
    probabilities and the fitted shrinkage weights, so every shrinkage
    variant can be applied afterwards without retraining."""
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"]).astype("datetime64[ns]")
    tier = league_tier(league)
    out = []
    windows = _windows(start, end, refit_months)
    for i, (w_start, w_end) in enumerate(windows):
        last = i == len(windows) - 1
        in_window = (history["date"] >= w_start) & ((history["date"] <= w_end) if last else (history["date"] < w_end))
        cols = ["date", "league", "home_team", "away_team", "home_goals", "away_goals", *MARKET_FEATURES]
        targets = history.loc[in_window, cols]
        if targets.empty:
            continue
        train = history[history["date"] < w_start]
        print(f"  {league}: train < {w_start.date()} ({len(train)} matches) -> predict {len(targets)}", flush=True)
        model_a, ensemble = train_ensemble(train, league, tier)

        targets = targets.reset_index(drop=True)
        a = _model_a_holdout_flat(model_a, history, targets, VALIDATION_SIMS)
        b = ensemble.predict_model_b(history, targets)
        cal = {m: c.apply(a[m], b.get(m))[1] for m, c in ensemble.calibrations.items()}

        preds = targets.copy()
        for k, col in enumerate(MR_COLS):
            preds[col] = cal["match_result"][:, k]
        preds["p_over_2.5"] = cal["over_2.5"]
        # The fitted weights (not what's served, see USE_FITTED_SHRINKAGE) --
        # for the "fitted" comparison variant.
        for market, col in (("match_result", "w_match_result"), ("over_2.5", "w_over_2.5")):
            w = ensemble.calibrations[market].w_market
            preds[col] = DEFAULT_SHRINKAGE_WEIGHT if w is None else w
        preds["trained_before"] = w_start
        out.append(preds)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def load_predictions(leagues, start, end, refit_months: int, retrain: bool, cache: Path) -> pd.DataFrame:
    if cache.exists() and not retrain:
        preds = pd.read_csv(cache, parse_dates=["date", "trained_before"])
        missing = [lg for lg in leagues if lg not in set(preds["league"])]
        if not missing:
            print(f"Using cached out-of-sample predictions: {cache}")
            return preds[preds["league"].isin(leagues)]
        print(f"Cache lacks {missing}; building those leagues.")
        leagues_to_run = missing
    else:
        preds, leagues_to_run = pd.DataFrame(), leagues

    db = SessionLocal()
    try:
        frames = [preds] if not preds.empty else []
        for league in leagues_to_run:
            history = _matches_df(db, league)
            if history.empty:
                print(f"  {league}: no matches in DB, skipped")
                continue
            frames.append(predict_league_oos(history, league, start, end, refit_months))
    finally:
        db.close()
    preds = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(cache, index=False)
    return preds[preds["league"].isin(leagues)]


def apply_shrinkage(preds: pd.DataFrame, mode: str) -> pd.DataFrame:
    """mode: "none", "fitted" (each league's calibration-split weight), or a
    fixed weight like "0.70". Matches without an opening price keep the
    model's probabilities."""
    out = preds.copy()
    if mode == "none":
        return out
    n = len(out)
    w_mr = out["w_match_result"].to_numpy(dtype=float) if mode == "fitted" else np.full(n, float(mode))
    w_ou = out["w_over_2.5"].to_numpy(dtype=float) if mode == "fitted" else np.full(n, float(mode))
    out[MR_COLS] = shrink_to_market(
        out[MR_COLS].to_numpy(dtype=float), out[MR_MARKET].to_numpy(dtype=float), w_mr[:, None]
    )
    out["p_over_2.5"] = shrink_to_market(
        out["p_over_2.5"].to_numpy(dtype=float), out["opening_prob_over_2.5"].to_numpy(dtype=float), w_ou
    )
    return out


# ---------------------------------------------------------------------------
# Odds -> engine rows
# ---------------------------------------------------------------------------

def to_long(joined: pd.DataFrame, price: str, bookmaker: str) -> pd.DataFrame:
    """One row per outcome per book, in BettingBacktestEngine's format."""
    mr_cols, ou_cols = FD_ODDS[(price, bookmaker)]
    close_mr, close_ou = FD_ODDS[("closing", bookmaker)]
    hg, ag = joined["home_goals"], joined["away_goals"]
    over = (hg + ag) > 2.5
    books = [
        ("match_result", "home_win", "p_home_win", mr_cols[0], close_mr[0], hg > ag),
        ("match_result", "draw", "p_draw", mr_cols[1], close_mr[1], hg == ag),
        ("match_result", "away_win", "p_away_win", mr_cols[2], close_mr[2], hg < ag),
        ("over_under_2.5", "over_2.5", "p_over_2.5", ou_cols[0], close_ou[0], over),
        ("over_under_2.5", "under_2.5", None, ou_cols[1], close_ou[1], ~over),
    ]
    rows = []
    for book, sel, p_col, odds_col, close_col, won in books:
        rows.append(pd.DataFrame({
            "match_id": joined["match_id"],
            "date": joined["date"],
            "league": joined["league"],
            "home_team": joined["home_team"],
            "away_team": joined["away_team"],
            "book": book,
            "selection": sel,
            "p_model": joined[p_col] if p_col else 1.0 - joined["p_over_2.5"],
            "odds": numeric_col(joined, odds_col),
            # Always attached: CLV when betting earlier prices, and the
            # closing-market Brier benchmark either way.
            "closing_odds": numeric_col(joined, close_col),
            "won": won.astype(int),
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(x, digits: int = 2) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:+.{digits}f}%"


def _num(x, fmt: str) -> str:
    return "n/a" if x is None or pd.isna(x) else format(x, fmt)


def print_report(result, boot: dict, meta: dict) -> None:
    s = result.summary
    line = "=" * 76
    print(f"\n{line}\nVALUE BETTING BACKTEST -- {meta['season_label']} -- {meta['scope']}\n{line}")
    print(f"Probabilities: shrinkage {meta['shrinkage']} toward Bet365 de-vigged opening price")
    print(f"Betting: {meta['bookmaker']} {meta['price']} odds   Staking: {s['staking']}"
          + (f" ({s['kelly_fraction']:g} Kelly, cap {meta['max_stake']:.0%})" if s["kelly_fraction"] else " (1 unit)")
          + f"   Min EV: {s['min_ev']:.1%}   Max edge: {_num(s['max_edge'], '.0%')}")
    print(f"Models: trained before {meta['windows']}")
    print(f"Matches: {meta['joined']} joined to odds ({meta['unmatched']} odds rows unmatched), "
          f"{s['n_matches']} with complete prices   Books evaluated: {s['n_books']}")

    print(f"\n--- Financial performance {'-' * 50}")
    print(f"Bets placed          {s['n_bets']}")
    print(f"Staked / profit      {s['staked_units']:.2f} u / {s['profit_units']:+.2f} u")
    print(f"ROI (yield)          {_pct(s['roi'])}")
    wr = s["win_rate"]
    print(f"Win rate             {'n/a' if wr is None else f'{wr:.1%}'}"
          + ("" if s["breakeven_win_rate"] is None else f"  (breakeven at taken odds {s['breakeven_win_rate']:.1%})"))
    print(f"Avg odds             {_num(s['avg_odds'], '.2f')}")
    print(f"Avg predicted edge   {_pct(s['avg_edge'])}  (p_model - de-vigged price taken)")
    print(f"Avg EV               {_pct(s['avg_ev'])}")
    if s["avg_clv"] is not None:
        print(f"Avg CLV              {_pct(s['avg_clv'])}  (EV of taken price vs. de-vigged closing line)")
    print(f"Max drawdown         {s['max_drawdown_units']:.2f} u")
    print(f"\nBootstrap ({boot['n_boot']} resamples by match), H0: ROI <= 0")
    print(f"  p-value            {_num(boot['p_value'], '.4f')}")
    print(f"  95% CI for ROI     [{_pct(boot['ci_low'])}, {_pct(boot['ci_high'])}]")

    cols = ["n_bets", "staked_units", "profit_units", "roi", "win_rate", "avg_odds", "avg_edge", "avg_ev"]
    for title, df in (("By market", result.by_market), ("By league", result.by_league)):
        if df.empty:
            continue
        print(f"\n--- {title} {'-' * (72 - len(title))}")
        show = df[cols].copy()
        for c in ("roi", "avg_edge", "avg_ev"):
            show[c] = show[c].map(lambda v: _pct(v, 1))
        show["win_rate"] = show["win_rate"].map(lambda v: _num(v, ".1%"))
        show["avg_odds"] = show["avg_odds"].map(lambda v: _num(v, ".2f"))
        print(show.to_string())

    print(f"\n--- Probability quality (all evaluated books) {'-' * 30}")
    _print_scores(result.probability_scores)
    print(line)


def _print_scores(scores: dict) -> None:
    print(f"{'book':<16}{'n':>6}{'Brier model':>13}{'mkt open':>10}{'mkt close':>11}"
          f"{'LogLoss model':>15}{'mkt open':>10}{'mkt close':>11}")
    for book, sc in scores.items():
        close = sc.get("closing_market")
        print(f"{book:<16}{sc['n']:>6}{sc['model']['brier']:>13.4f}{sc['market']['brier']:>10.4f}"
              f"{_num(close and close['brier'], '.4f'):>11}{sc['model']['log_loss']:>15.4f}"
              f"{sc['market']['log_loss']:>10.4f}{_num(close and close['log_loss'], '.4f'):>11}")
    print("(Lower is better; 'mkt' = Bet365 de-vigged with Shin's method.)")


def run_variant(joined: pd.DataFrame, mode: str, args, engine: BettingBacktestEngine):
    long = to_long(apply_shrinkage(joined, mode), args.price, args.bookmaker)
    result = engine.run(long)
    bets = result.bets
    boot = bootstrap_roi_test(bets["profit"], bets["stake"], clusters=bets["match_id"], n_boot=args.n_boot)
    return result, boot


def print_comparison(joined: pd.DataFrame, args, engine: BettingBacktestEngine) -> None:
    scopes = [("EPL + La Liga", joined[joined["league"].isin(HEADLINE_LEAGUES)]), ("All leagues", joined)]
    modes = ["none", f"{DEFAULT_SHRINKAGE_WEIGHT:.2f}", "fitted"]
    w = joined.drop_duplicates("league").set_index("league")[["w_match_result", "w_over_2.5"]]
    line = "=" * 108
    print(f"\n{line}\nSHRINKAGE COMPARISON -- 1X2 Brier vs. Bet365 (de-vigged), edge, ROI\n{line}")
    print("Fitted weights (calibration split, pre-season): "
          + ", ".join(f"{lg} 1X2 {r.w_match_result:.2f} / O-U {r['w_over_2.5']:.2f}" for lg, r in w.iterrows()))
    print(f"\n{'scope':<15}{'shrinkage':<11}{'Brier 1X2':>10}{'open mkt':>10}{'close mkt':>10}"
          f"{'Brier O/U':>10}{'open mkt':>10}{'bets':>7}{'avg edge':>10}{'avg EV':>9}{'ROI':>9}{'p-value':>9}")
    for scope, df in scopes:
        if df.empty:
            continue
        for mode in modes:
            result, boot = run_variant(df, mode, args, engine)
            s, sc = result.summary, result.probability_scores
            mr, ou = sc.get("match_result"), sc.get("over_under_2.5")
            close = mr.get("closing_market") if mr else None
            print(f"{scope:<15}{mode:<11}"
                  f"{_num(mr and mr['model']['brier'], '.4f'):>10}{_num(mr and mr['market']['brier'], '.4f'):>10}"
                  f"{_num(close and close['brier'], '.4f'):>10}"
                  f"{_num(ou and ou['model']['brier'], '.4f'):>10}{_num(ou and ou['market']['brier'], '.4f'):>10}"
                  f"{s['n_bets']:>7}{_pct(s['avg_edge']):>10}{_pct(s['avg_ev'], 1):>9}{_pct(s['roi'], 1):>9}"
                  f"{_num(boot['p_value'], '.3f'):>9}")
        print()
    print("A shrunk model only beats the market if its Brier is below 'open mkt' (the price it shrinks toward\n"
          "and bets at); 'close mkt' is the stricter benchmark. p-value: bootstrap, H0 ROI <= 0.")
    print(line)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default="2526", help="football-data season code, e.g. 2526 = 2025/26")
    ap.add_argument("--leagues", nargs="+", default=list(FD_LEAGUES), choices=list(FD_LEAGUES), metavar="LEAGUE")
    ap.add_argument("--shrinkage", default=f"{DEFAULT_SHRINKAGE_WEIGHT:.2f}",
                    help='a weight such as 0.70 (default: what production serves), "fitted", or "none"')
    ap.add_argument("--price", choices=("closing", "opening"), default="opening",
                    help="price bet at (default opening: the price shrinkage uses and a bettor could take)")
    # Not Pinnacle by default: football-data's Pinnacle columns are only
    # ~40% filled for 2025/26 (Pinnacle closed its public odds feed mid-2025).
    ap.add_argument("--bookmaker", choices=("pinnacle", "bet365", "avg", "max"), default="bet365")
    ap.add_argument("--min-ev", type=float, default=MIN_EV)
    ap.add_argument("--max-edge", type=float, default=MAX_EDGE_CAP, help="probability-edge cap; <= 0 disables")
    ap.add_argument("--staking", choices=("kelly", "flat"), default="kelly")
    ap.add_argument("--kelly-fraction", type=float, default=KELLY_FRACTION)
    ap.add_argument("--max-stake", type=float, default=0.05, help="cap as a bankroll fraction")
    ap.add_argument("--max-odds", type=float, default=None, help="skip bets above these odds")
    ap.add_argument("--refit-months", type=int, default=0, help="0 = one model frozen at season start")
    ap.add_argument("--retrain", action="store_true", help="ignore cached out-of-sample predictions")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--no-compare", action="store_true", help="skip the shrinkage comparison table")
    ap.add_argument("--out", type=Path, default=None, help="write the placed bets to this CSV")
    args = ap.parse_args()
    if args.shrinkage not in ("none", "fitted"):
        args.shrinkage = f"{float(args.shrinkage):.2f}"

    y1 = 2000 + int(args.season[:2])
    start, end = pd.Timestamp(f"{y1}-07-01"), pd.Timestamp(f"{y1 + 1}-06-30")
    cache = CACHE_DIR / f"oos_predictions_{args.season}_refit{args.refit_months}_v{CACHE_VERSION}.csv"

    preds = load_predictions(args.leagues, start, end, args.refit_months, args.retrain, cache)
    odds = load_fd_odds(args.season, args.leagues)
    joined, unmatched = join_odds(preds.reset_index(drop=True), odds)
    if joined.empty:
        sys.exit("No predictions could be matched to odds -- nothing to backtest.")

    engine = BettingBacktestEngine(
        min_ev=args.min_ev,
        kelly_fraction=args.kelly_fraction,
        max_stake=args.max_stake,
        staking=args.staking,
        max_odds=args.max_odds,
        max_edge=args.max_edge if args.max_edge > 0 else None,
    )
    result, boot = run_variant(joined, args.shrinkage, args, engine)

    windows = sorted({str(pd.Timestamp(d).date()) for d in preds["trained_before"]})
    print_report(result, boot, {
        "season_label": f"{y1}/{str(y1 + 1)[2:]}",
        "scope": ", ".join(args.leagues) if len(args.leagues) < len(FD_LEAGUES) else "all domestic leagues",
        "shrinkage": args.shrinkage,
        "bookmaker": args.bookmaker,
        "price": args.price,
        "max_stake": args.max_stake,
        "unmatched": unmatched,
        "joined": len(joined),
        "windows": ", ".join(windows) if len(windows) <= 3 else f"{windows[0]} ... {windows[-1]} ({len(windows)} refits)",
    })
    if not args.no_compare:
        print_comparison(joined, args, engine)
    if args.out:
        result.bets.to_csv(args.out, index=False)
        print(f"Bets written to {args.out}")


if __name__ == "__main__":
    main()
