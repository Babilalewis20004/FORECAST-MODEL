"""
Backtest Evaluation Dashboard
=============================

How to run:
    pip install streamlit plotly pandas numpy
    streamlit run app.py

Runs the real `BettingBacktestEngine` on the 2025/26 season via
backend/scripts/run_backtest.py -- see `load_backtest()`. That needs the
backend's own dependencies (sqlalchemy, psycopg2, fastapi's settings loader,
xgboost, ...; see backend/requirements.txt) importable from this process, and
a reachable Postgres database (app.db.SessionLocal / DATABASE_URL) with match
history loaded -- the same DB the backend API uses. The first run also needs
network access, to fetch football-data.co.uk odds and build the out-of-sample
prediction cache under downloaded_files/backtest.

Expected bet columns (after `_adapt_bets`): date, league, market, odds, ev,
edge, stake, profit, won, p_model (for the reliability curve) and optionally
p_market, p_close.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# backend/scripts/run_backtest.py itself inserts backend/ onto sys.path (so
# its own `from app... import ...` lines resolve) -- importing it here makes
# the whole backend package importable for the rest of this module too.
_BACKEND_SCRIPTS = Path(__file__).resolve().parent / "backend" / "scripts"
if str(_BACKEND_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SCRIPTS))

import run_backtest  # noqa: E402 -- after the sys.path shim above
from app.db import SessionLocal  # noqa: E402
from app.ml.value_engine import BacktestResult, bootstrap_roi_test  # noqa: E402
from app.orm_models import BetTracking  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402

# Validated categorical/diverging steps (blue <-> red poles, gray reference ink).
BLUE, ORANGE, RED, MUTED = "#2a78d6", "#eb6834", "#e34948", "#898781"

# BettingBacktestEngine books -> display labels (football-data has no BTTS
# odds, see run_backtest.py's module docstring; any other book falls back to
# its raw key).
MARKET_LABELS = {"match_result": "1X2", "over_under_2.5": "Over/Under 2.5", "btts": "BTTS"}


def _market_label(book: str) -> str:
    return MARKET_LABELS.get(book, book)


# ---------------------------------------------------------------------------
# Adapters: BettingBacktestEngine's schema -> this dashboard's display schema
# ---------------------------------------------------------------------------

def _adapt_bets(bets: pd.DataFrame) -> pd.DataFrame:
    out = bets.rename(columns={"book": "market"})
    out["market"] = out["market"].map(_market_label)
    return out


def _adapt_segment(seg: pd.DataFrame, key: str) -> pd.DataFrame:
    """engine._breakdown() output (indexed by book/league, engine column
    names) -> the shape `segment()` below produces for filtered views."""
    if seg.empty:
        return pd.DataFrame(columns=[key, "bets", "staked", "profit", "win_rate", "avg_odds", "avg_ev", "roi"])
    out = seg.rename(columns={"n_bets": "bets", "staked_units": "staked", "profit_units": "profit"})
    out = out.rename_axis(key).reset_index()
    if key == "market":
        out["market"] = out["market"].map(_market_label)
    return out[[key, "bets", "staked", "profit", "win_rate", "avg_odds", "avg_ev", "roi"]]


def _adapt_summary(summary: dict) -> dict:
    return {
        "total_bets": summary["n_bets"],
        "staked": summary["staked_units"],
        "profit": summary["profit_units"],
        "roi": summary["roi"] or 0.0,
        "win_rate": summary["win_rate"] or 0.0,
        "avg_odds": summary["avg_odds"] or 0.0,
        "avg_ev": summary["avg_ev"] or 0.0,
        "max_drawdown": summary["max_drawdown_units"],
    }


def _adapt_probability_scores(scores: dict) -> pd.DataFrame:
    """engine.probability_scores (book -> {"n", "model": {...}, "market":
    {...}, "closing_market": {...}?}) -> one row per market, model vs. market
    (and closing line, when priced at an earlier price than closing)."""
    rows = []
    for book, sc in scores.items():
        row = {
            "market": _market_label(book), "n": sc["n"],
            "brier_model": sc["model"]["brier"], "log_loss_model": sc["model"]["log_loss"],
            "brier_market": sc["market"]["brier"], "log_loss_market": sc["market"]["log_loss"],
        }
        if "closing_market" in sc:
            row["brier_close"] = sc["closing_market"]["brier"]
            row["log_loss_close"] = sc["closing_market"]["log_loss"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analytics helpers (used for the filtered views; the unfiltered view uses
# the engine's own summary/by_market/by_league via the adapters above)
# ---------------------------------------------------------------------------

def summarize(bets: pd.DataFrame) -> dict:
    staked, profit = bets["stake"].sum(), bets["profit"].sum()
    cum = bets["profit"].cumsum()
    drawdown = cum - cum.cummax().clip(lower=0)  # peak includes the 0 starting point
    return {
        "total_bets": len(bets),
        "staked": staked,
        "profit": profit,
        "roi": profit / staked if staked else 0.0,
        "win_rate": bets["won"].mean(),
        "avg_odds": bets["odds"].mean(),
        "avg_ev": bets["ev"].mean(),
        "max_drawdown": float(-drawdown.min()) if len(bets) else 0.0,
    }


def segment(bets: pd.DataFrame, key: str) -> pd.DataFrame:
    g = bets.groupby(key).agg(
        bets=("profit", "size"),
        staked=("stake", "sum"),
        profit=("profit", "sum"),
        win_rate=("won", "mean"),
        avg_odds=("odds", "mean"),
        avg_ev=("ev", "mean"),
    )
    g["roi"] = g["profit"] / g["staked"]
    return g.reset_index().sort_values("roi")


def bootstrap_roi(profits, stakes, clusters, n_boot: int = 2000, seed: int = 0) -> dict:
    """Cluster bootstrap of ROI for the interactive (filtered) views -- the
    same match-clustered scheme value_engine.bootstrap_roi_test uses for the
    full backtest, just called with fewer resamples so filter changes stay
    snappy. Its "boot_rois" key carries the raw distribution for plotting."""
    return bootstrap_roi_test(profits, stakes, clusters, n_boot=n_boot, seed=seed)


# ---------------------------------------------------------------------------
# Real backtest, via backend/scripts/run_backtest.py
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Running the 2025/26 backtest (training models, loading odds)…")
def load_backtest() -> tuple[BacktestResult, dict]:
    result, boot, _meta = run_backtest.run_backtest(season="2526")
    adapted = BacktestResult(
        bets=_adapt_bets(result.bets),
        summary=_adapt_summary(result.summary),
        by_market=_adapt_segment(result.by_market, "market"),
        by_league=_adapt_segment(result.by_league, "league"),
        probability_scores=result.probability_scores,  # kept raw; _adapt_probability_scores() at render time
    )
    return adapted, boot


# ---------------------------------------------------------------------------
# Live CLV (Phase 2): bet_tracking, written by odds_service, read directly
# from the same Postgres the backend API uses -- not part of the backtest
# replay above. Short TTL since poll_closing_lines() fills rows in
# continuously as fixtures approach kickoff.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def load_live_bet_tracking() -> pd.DataFrame:
    try:
        db = SessionLocal()
    except SQLAlchemyError:
        return pd.DataFrame()
    try:
        rows = db.query(BetTracking).filter(BetTracking.clv_percentage.isnot(None)).all()
    except SQLAlchemyError:
        # e.g. the migration hasn't been run yet -- treat as "no live data".
        return pd.DataFrame()
    finally:
        db.close()
    return pd.DataFrame([
        {
            "fixture_id": r.fixture_id,
            "market": r.market,
            "selection": r.selection,
            "bookmaker": r.bookmaker,
            "is_sharp_book": r.is_sharp_book,
            "odds_placed_time": r.odds_placed_time,
            "odds_placed_value": r.odds_placed_value,
            "ev_at_placement": r.ev_at_placement,
            "closing_bookmaker": r.closing_bookmaker,
            "closing_odds_value": r.closing_odds_value,
            "clv_percentage": r.clv_percentage,
        }
        for r in rows
    ])


# ---------------------------------------------------------------------------
# Cached filtering (keyed on hashable widget values, not the 50k-row frame)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def filter_bets(leagues: tuple, markets: tuple, start: date, end: date) -> pd.DataFrame:
    bets = load_backtest()[0].bets
    mask = (
        bets["league"].isin(leagues)
        & bets["market"].isin(markets)
        & bets["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    )
    return bets[mask].reset_index(drop=True)


@st.cache_data(show_spinner="Bootstrapping ROI…")
def filtered_bootstrap(leagues: tuple, markets: tuple, start: date, end: date) -> dict:
    b = filter_bets(leagues, markets, start, end)
    return bootstrap_roi(b["profit"], b["stake"], b["match_id"])


@st.cache_data(show_spinner=False)
def daily_equity(leagues: tuple, markets: tuple, start: date, end: date, bankroll: float) -> pd.DataFrame:
    b = filter_bets(leagues, markets, start, end)
    daily = b.groupby(b["date"].dt.normalize())["profit"].sum().rename("profit").to_frame()
    daily["bankroll"] = bankroll + daily["profit"].cumsum()
    peak = daily["bankroll"].cummax().clip(lower=bankroll)
    daily["drawdown"] = daily["bankroll"] - peak
    daily["drawdown_pct"] = daily["drawdown"] / peak
    return daily.reset_index()


@st.cache_data(show_spinner=False)
def reliability(leagues: tuple, markets: tuple, start: date, end: date, n_bins: int) -> pd.DataFrame:
    b = filter_bets(leagues, markets, start, end)
    frames = []
    for col, label in (("p_model", "Model"), ("p_market", "Market (de-vigged)")):
        if col not in b:
            continue
        bins = pd.qcut(b[col], n_bins, duplicates="drop")
        g = b.groupby(bins, observed=True).agg(predicted=(col, "mean"), actual=("won", "mean"), n=("won", "size"))
        frames.append(g.assign(source=label).reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def styled(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.08,
        **layout,
    )
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def histogram(values: pd.Series, title: str, x_title: str, fmt: str) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=60, marker_color=BLUE,
        hovertemplate=f"{x_title}: %{{x:{fmt}}}<br>Bets: %{{y:,}}<extra></extra>",
    ))
    fig.add_vline(x=values.mean(), line=dict(color=MUTED, dash="dot", width=1.5),
                  annotation_text=f"mean {values.mean():{fmt}}", annotation_position="top right")
    return styled(fig, 320, title=title, xaxis_title=x_title, yaxis_title="Bets", showlegend=False,
                  xaxis_tickformat=fmt)


def segment_bars(seg: pd.DataFrame, key: str, metric: str, title: str, fmt: str) -> go.Figure:
    colors = np.where(seg[metric] >= 0, BLUE, RED)  # diverging poles: gain vs loss
    fig = go.Figure(go.Bar(
        x=seg[metric], y=seg[key], orientation="h", marker_color=colors,
        text=seg[metric].map(lambda v: f"{v:{fmt}}"), textposition="outside", cliponaxis=False,
        customdata=seg[["bets", "staked"]],
        hovertemplate=f"%{{y}}<br>{title}: %{{x:{fmt}}}<br>Bets: %{{customdata[0]:,}}"
                      "<br>Staked: %{customdata[1]:,.1f}u<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    return styled(fig, 60 + 42 * len(seg), title=title, showlegend=False, xaxis_tickformat=fmt)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Backtest Evaluation", page_icon="📈", layout="wide")

result, full_bootstrap = load_backtest()
all_bets = result.bets
LEAGUES = sorted(all_bets["league"].unique().tolist())
MARKETS = sorted(all_bets["market"].unique().tolist())
min_date, max_date = all_bets["date"].min().date(), all_bets["date"].max().date()

with st.sidebar:
    st.header("Filters")
    leagues = tuple(st.multiselect("League", LEAGUES, default=LEAGUES))
    markets = tuple(st.multiselect("Market", MARKETS, default=MARKETS))
    picked = st.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
    start, end = (picked if isinstance(picked, (tuple, list)) and len(picked) == 2 else (min_date, max_date))
    st.divider()
    bankroll = st.number_input("Starting bankroll (units)", min_value=1.0, value=1000.0, step=100.0)
    st.caption("Filters apply to every tab. Results are cached per filter combination.")

bets = filter_bets(leagues, markets, start, end)
if bets.empty:
    st.warning("No bets match the current filters.")
    st.stop()

live_bets = load_live_bet_tracking()  # Phase 2 live CLV, unfiltered by the sidebar (separate data source)

unfiltered = set(leagues) == set(LEAGUES) and set(markets) == set(MARKETS) and (start, end) == (min_date, max_date)
summary = result.summary if unfiltered else summarize(bets)
boot = full_bootstrap if unfiltered else filtered_bootstrap(leagues, markets, start, end)
by_league = result.by_league if unfiltered else segment(bets, "league")
by_market = result.by_market if unfiltered else segment(bets, "market")

st.title("Backtest Evaluation")
st.caption(f"{summary['total_bets']:,} bets · {start:%d %b %Y} – {end:%d %b %Y}"
           + ("" if unfiltered else " · filtered view"))

tabs = st.tabs(["Summary", "Significance", "Segments", "Edge & EV", "Calibration & CLV", "Bet Explorer"])

# --- Tab 1: Executive summary --------------------------------------------------
with tabs[0]:
    row1, row2 = st.columns(4), st.columns(4)
    row1[0].metric("Total bets", f"{summary['total_bets']:,}")
    row1[1].metric("Staked", f"{summary['staked']:,.1f}u")
    row1[2].metric("Profit", f"{summary['profit']:+,.1f}u")
    row1[3].metric("ROI", f"{summary['roi']:+.2%}")
    row2[0].metric("Win rate", f"{summary['win_rate']:.1%}")
    row2[1].metric("Avg odds", f"{summary['avg_odds']:.2f}")
    row2[2].metric("Avg EV", f"{summary['avg_ev']:+.2%}")
    row2[3].metric("Max drawdown", f"-{summary['max_drawdown']:,.1f}u")

    row3 = st.columns(4)
    avg_clv_live = live_bets["clv_percentage"].mean() if not live_bets.empty else None
    row3[0].metric(
        "Avg CLV % (live)", f"{avg_clv_live:+.2%}" if avg_clv_live is not None else "—",
        help="Live, in-flight value bets tracked via bet_tracking -- distinct from this page's backtest KPIs above.",
    )

    eq = daily_equity(leagues, markets, start, end, bankroll)
    log_scale = st.toggle("Log scale", value=False, help="Bankroll on a log axis, so equal % moves look equal.")
    if log_scale and (eq["bankroll"] <= 0).any():
        st.info("Bankroll goes to zero or below in this view; log scale unavailable.")
        log_scale = False

    fig = go.Figure(go.Scatter(
        x=eq["date"], y=eq["bankroll"], mode="lines", line=dict(color=BLUE, width=2),
        customdata=eq["profit"],
        hovertemplate="%{x|%d %b %Y}<br>Bankroll: %{y:,.1f}u<br>Day P&L: %{customdata:+,.2f}u<extra></extra>",
    ))
    fig.add_hline(y=bankroll, line=dict(color=MUTED, dash="dot", width=1))
    show(styled(fig, 400, title="Equity curve (bankroll, units)", hovermode="x unified", showlegend=False,
                yaxis_type="log" if log_scale else "linear"))

    fig = go.Figure(go.Scatter(
        x=eq["date"], y=eq["drawdown"], mode="lines", fill="tozeroy",
        line=dict(color=RED, width=1.5), fillcolor="rgba(227,73,72,0.25)",
        customdata=eq["drawdown_pct"],
        hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:,.1f}u (%{customdata:.1%})<extra></extra>",
    ))
    show(styled(fig, 260, title="Drawdown from peak (units)", hovermode="x unified", showlegend=False))

# --- Tab 2: Statistical significance -------------------------------------------
with tabs[1]:
    significant = boot["p_value"] < 0.05
    c = st.columns(4)
    c[0].metric("Observed ROI", f"{boot['roi']:+.2%}")
    c[1].metric("p-value (H0: ROI ≤ 0)", f"{boot['p_value']:.4f}")
    c[2].metric("95% CI low", f"{boot['ci_low']:+.2%}")
    c[3].metric("95% CI high", f"{boot['ci_high']:+.2%}")

    verdict = f"ROI {'is' if significant else 'is not'} statistically significant at 95% confidence."
    (st.success if significant else st.error)(
        f"{verdict} (one-sided p = {boot['p_value']:.4f}; 95% CI {boot['ci_low']:+.2%} to {boot['ci_high']:+.2%}; "
        f"{boot['n_boot']:,} resamples clustered by match)"
    )

    fig = go.Figure(go.Histogram(
        x=boot["boot_rois"], nbinsx=70, marker_color=BLUE, name="Bootstrap ROI",
        hovertemplate="ROI: %{x:.2%}<br>Resamples: %{y:,}<extra></extra>",
    ))
    fig.add_vline(x=boot["roi"], line=dict(color="red", dash="dash", width=2),
                  annotation_text=f"Observed {boot['roi']:+.2%}", annotation_position="top right")
    for x in (boot["ci_low"], boot["ci_high"]):
        fig.add_vline(x=x, line=dict(color=MUTED, dash="dot", width=1.5))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    show(styled(fig, 420, title="Bootstrap ROI distribution (dotted: 95% CI)", xaxis_title="ROI",
                yaxis_title="Resamples", xaxis_tickformat=".1%", showlegend=False))

# --- Tab 3: Segment breakdown ----------------------------------------------------
with tabs[2]:
    for label, key, seg in (("League", "league", by_league), ("Market", "market", by_market)):
        st.subheader(f"By {label.lower()}")
        left, right = st.columns(2)
        with left:
            show(segment_bars(seg, key, "roi", "ROI", "+.1%"))
        with right:
            show(segment_bars(seg.sort_values("profit"), key, "profit", "Profit (units)", "+,.1f"))
        st.dataframe(
            seg.sort_values("roi", ascending=False), hide_index=True, width="stretch",
            column_config={
                key: label,
                "staked": st.column_config.NumberColumn("Staked", format="%.1f"),
                "profit": st.column_config.NumberColumn("Profit", format="%+.1f"),
                "roi": st.column_config.NumberColumn("ROI", format="percent"),
                "win_rate": st.column_config.NumberColumn("Win rate", format="percent"),
                "avg_odds": st.column_config.NumberColumn("Avg odds", format="%.2f"),
                "avg_ev": st.column_config.NumberColumn("Avg EV", format="percent"),
            },
        )

# --- Tab 4: Edge & EV distributions ------------------------------------------------
with tabs[3]:
    st.caption("Use the sidebar to filter by league, market and date range.")
    c = st.columns(3)
    with c[0]:
        show(histogram(bets["ev"], "Expected value", "EV", ".1%"))
    with c[1]:
        show(histogram(bets["edge"], "Probability edge (model − market)", "Edge", ".1%"))
    with c[2]:
        show(histogram(bets["odds"], "Odds taken", "Odds", ".2f"))

# --- Tab 5: Calibration & CLV --------------------------------------------------------
with tabs[4]:
    left, right = st.columns(2)
    with left:
        n_bins = st.slider("Calibration bins", 5, 30, 15)
        rel = reliability(leagues, markets, start, end, n_bins)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                 line=dict(color=MUTED, dash="dash", width=1), hoverinfo="skip"))
        for source, color in (("Model", BLUE), ("Market (de-vigged)", ORANGE)):
            r = rel[rel["source"] == source]
            if r.empty:
                continue
            fig.add_trace(go.Scatter(
                x=r["predicted"], y=r["actual"], mode="lines+markers", name=source,
                line=dict(color=color, width=2), marker=dict(size=8),
                customdata=r["n"],
                hovertemplate=f"{source}<br>Predicted: %{{x:.1%}}<br>Actual: %{{y:.1%}}"
                              "<br>Bets: %{customdata:,}<extra></extra>",
            ))
        hi = float(max(rel["predicted"].max(), rel["actual"].max())) * 1.05
        show(styled(fig, 440, title="Reliability curve", xaxis_title="Predicted probability",
                    yaxis_title="Actual win frequency", xaxis=dict(range=[0, hi], tickformat=".0%"),
                    yaxis=dict(range=[0, hi], tickformat=".0%")))

        scores = _adapt_probability_scores(result.probability_scores)
        has_close = "brier_close" in scores.columns
        st.caption("Probability scores per market (full backtest): model vs. de-vigged market"
                   + (" and closing line" if has_close else ""))
        score_config = {
            "market": "Market",
            "n": st.column_config.NumberColumn("Bets", format="%d"),
            "brier_model": st.column_config.NumberColumn("Brier (model)", format="%.4f"),
            "log_loss_model": st.column_config.NumberColumn("Log loss (model)", format="%.4f"),
            "brier_market": st.column_config.NumberColumn("Brier (market)", format="%.4f"),
            "log_loss_market": st.column_config.NumberColumn("Log loss (market)", format="%.4f"),
        }
        if has_close:
            score_config["brier_close"] = st.column_config.NumberColumn("Brier (close)", format="%.4f")
            score_config["log_loss_close"] = st.column_config.NumberColumn("Log loss (close)", format="%.4f")
        st.dataframe(scores, hide_index=True, width="stretch", column_config=score_config)

    with right:
        if "p_close" not in bets or bets["p_close"].isna().all():
            st.info("No closing prices (p_close) in this backtest, so CLV can't be computed.")
        else:
            clv = bets["odds"] * bets["p_close"] - 1  # value of our price against the fair closing line
            c = st.columns(3)
            c[0].metric("Average CLV", f"{clv.mean():+.2%}")
            c[1].metric("Bets beating close", f"{(clv > 0).mean():.1%}")
            c[2].metric("Corr(EV, CLV)", f"{np.corrcoef(bets['ev'], clv)[0, 1]:.2f}")

            sample = bets.assign(clv=clv).sample(min(len(bets), 5000), random_state=0)
            trend = (bets.assign(clv=clv, bin=pd.qcut(bets["ev"], 20, duplicates="drop"))
                     .groupby("bin", observed=True)[["ev", "clv"]].mean())
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=sample["ev"], y=sample["clv"], mode="markers", name=f"Bets (sample of {len(sample):,})",
                marker=dict(color=BLUE, size=5, opacity=0.35),
                hovertemplate="EV: %{x:.2%}<br>CLV: %{y:.2%}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=trend["ev"], y=trend["clv"], mode="lines+markers", name="Binned mean",
                line=dict(color=ORANGE, width=2), marker=dict(size=8),
                hovertemplate="EV: %{x:.2%}<br>Mean CLV: %{y:.2%}<extra></extra>",
            ))
            fig.add_hline(y=0, line=dict(color=MUTED, width=1))
            show(styled(fig, 520, title="EV at bet time vs closing-line value", xaxis_title="EV",
                        yaxis_title="CLV", xaxis_tickformat=".0%", yaxis_tickformat=".0%"))

    st.divider()
    st.subheader("Live CLV (in-flight value bets)")
    st.caption("From bet_tracking, not the backtest above: odds_service.capture_placed_snapshot() freezes the "
               "price when a value bet is first flagged, poll_closing_lines() fills in the sharp-book closing "
               "line near kickoff and this table's clv_percentage.")
    if live_bets.empty:
        st.info("No live bets with a captured closing line yet.")
    else:
        lc = st.columns(3)
        lc[0].metric("Avg CLV % (live)", f"{live_bets['clv_percentage'].mean():+.2%}")
        lc[1].metric("Bets tracked", f"{len(live_bets):,}")
        lc[2].metric("Beating close", f"{(live_bets['clv_percentage'] > 0).mean():.1%}")

        fig = go.Figure(go.Scattergl(
            x=live_bets["ev_at_placement"], y=live_bets["clv_percentage"], mode="markers",
            marker=dict(color=BLUE, size=7, opacity=0.6),
            customdata=live_bets[["market", "selection", "bookmaker"]],
            hovertemplate="EV: %{x:.2%}<br>CLV: %{y:.2%}<br>%{customdata[0]} / %{customdata[1]} @ "
                          "%{customdata[2]}<extra></extra>",
        ))
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
        show(styled(fig, 420, title="Live: EV at placement vs closing-line value", xaxis_title="EV at placement",
                    yaxis_title="CLV", xaxis_tickformat=".0%", yaxis_tickformat=".0%", showlegend=False))

# --- Tab 6: Bet explorer ---------------------------------------------------------------
with tabs[5]:
    st.caption(f"{len(bets):,} bets. Click a column header to sort; use the table toolbar to search or download.")
    st.dataframe(
        bets, hide_index=True, width="stretch", height=620,
        column_config={
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "odds": st.column_config.NumberColumn("Odds", format="%.2f"),
            "p_model": st.column_config.NumberColumn("P model", format="%.3f"),
            "p_market": st.column_config.NumberColumn("P market", format="%.3f"),
            "p_close": st.column_config.NumberColumn("P close", format="%.3f"),
            "edge": st.column_config.NumberColumn("Edge", format="percent"),
            "ev": st.column_config.NumberColumn("EV", format="percent"),
            "stake": st.column_config.NumberColumn("Stake", format="%.2f"),
            "profit": st.column_config.NumberColumn("Profit", format="%+.2f"),
            "won": st.column_config.CheckboxColumn("Won"),
        },
    )
