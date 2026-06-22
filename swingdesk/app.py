"""Streamlit dashboard. Can be launched either way:

    streamlit run swingdesk/app.py          # works (bootstraps sys.path below)
    python -m streamlit run swingdesk/app.py

Either invocation goes through the same module path.
"""
from __future__ import annotations

import sys
from pathlib import Path

# When Streamlit runs this file as a script, __package__ is empty and relative
# imports fail. Insert the project root so `import swingdesk.*` resolves.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from swingdesk.analyze import chart_signals
from swingdesk.analyze import glossary
from swingdesk.analyze import discovery
from swingdesk.analyze import manipulation
from swingdesk.analyze import market_calendar
from swingdesk.analyze import sectors as sector_rotation
from swingdesk.analyze import score as scoring
from swingdesk.analyze import smallcaps
from swingdesk.analyze import sentiment as sentiment_mod
from swingdesk.analyze import summary as stock_summary
from swingdesk.analyze import spillover as spillover_mod
from swingdesk.analyze import expected_range as erange_mod
from swingdesk.analyze import risk as risk_mod
from swingdesk.analyze import factors as factors_mod
from swingdesk.analyze import ml_direction as ml_mod
from swingdesk.analyze import intraday as intraday_mod
from swingdesk.analyze import signal_analysis as sigan_mod
from swingdesk.analyze import screener as screener_mod
from swingdesk.analyze import liquidity as liquidity_mod
from swingdesk.analyze import execution as execution_mod
from swingdesk.analyze import tca as tca_mod
from swingdesk.analyze import decision as decision_mod
from swingdesk.analyze.setups import scan_all
from swingdesk.analyze.technicals import (
    add_indicators as _add_indicators_raw,
    money_flow_read,
    signal_scoreboard,
    trend_quality,
    volume_profile,
    volume_profile_read,
)
from swingdesk.backtest import engine as bt_engine
from swingdesk.backtest import metrics as bt_metrics
from swingdesk.config import (
    ACCOUNT_CAPITAL,
    BACKTEST_COST_PCT,
    DEFAULT_WATCHLIST,
    MAX_OPEN_POSITIONS,
    RISK_PER_TRADE_PCT,
)
from swingdesk.ingest import fundamentals as fundamentals_ingest
from swingdesk.ingest import macro as macro_ingest
from swingdesk.ingest import news_rss, prices
from swingdesk.ingest import intraday as intraday_ingest
from swingdesk.notify import telegram
from swingdesk.backtest import optimizer as opt
from swingdesk.portfolio import allocate as allocate_mod
from swingdesk.portfolio import holdings as holdings_mod
from swingdesk.portfolio import journal as pj
from swingdesk.portfolio import positions as portfolio
from swingdesk.portfolio import paper_trader as paper_trader_mod
from swingdesk.portfolio import reconcile as reconcile_mod
from swingdesk.storage import (
    add_to_smallcap_watchlist,
    combined_universe,
    get_smallcap_watchlist,
    get_fundamentals,
    get_watchlist,
    holdings_tickers,
    load_holdings,
    load_fundamentals as _load_fundamentals_raw,
    load_plans,
    upsert_plan,
    update_plan,
    delete_plan,
    remove_from_smallcap_watchlist,
    set_smallcap_watchlist,
    init_db,
    list_backtest_runs,
    load_backtest_trades,
    load_news,
    load_positions,
    load_prices as _load_prices_raw,
    load_signals,
    save_backtest_trades,
    save_signals,
    seed_watchlist_if_empty,
    set_watchlist,
)

st.set_page_config(page_title="SwingDesk", layout="wide", page_icon=":chart_with_upwards_trend:")

init_db()
seed_watchlist_if_empty(DEFAULT_WATCHLIST)


# --- Cached hot paths -----------------------------------------------------------
# Streamlit re-runs this whole script (and the body of EVERY tab — tabs are not
# lazy) on every widget interaction. Without caching, that re-reads prices from
# disk and recomputes ~30 pandas-ta indicators per ticker on each click, which
# is what makes the app lag. st.cache_data returns a fresh copy each call, so
# downstream mutation is safe. Caches are cleared explicitly after any refetch
# (see `_clear_data_caches`).
@st.cache_data(ttl=600, show_spinner=False)
def load_prices(ticker: str, days: int | None = None) -> pd.DataFrame:
    return _load_prices_raw(ticker, days=days)


@st.cache_data(ttl=600, show_spinner=False)
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return _add_indicators_raw(df)


@st.cache_data(ttl=600, show_spinner=False)
def load_fundamentals(min_quality: float | None = None) -> pd.DataFrame:
    return _load_fundamentals_raw(min_quality=min_quality)


def _clear_data_caches() -> None:
    """Call after any action that writes fresh prices/fundamentals to disk so the
    UI doesn't keep serving stale cached frames."""
    load_prices.clear()
    add_indicators.clear()
    load_fundamentals.clear()
    _cached_regime.clear()
    _cached_outlook.clear()
    _cached_betas.clear()
    _cached_factor_table.clear()
    _cached_ml_eval.clear()
    _cached_ml_predict.clear()
    _cached_ml_importance.clear()
    _cached_intraday_scan.clear()
    _cached_expected_range.clear()
    _cached_monte_carlo.clear()
    _cached_vol_cone.clear()
    _cached_daily_vol.clear()
    _cached_setup_bt_stats.clear()
    _cached_holding_plan.clear()
    _cached_target_vs_stop.clear()
    _cached_risk_report.clear()
    _cached_portfolio_analysis.clear()
    _cached_manip_cards.clear()
    _cached_decisions.clear()
    _cached_market_pulse.clear()
    _cached_price_coverage.clear()


# Quant modules: macro/portfolio reads are cheap but loop over many tickers, so
# cache them (tabs aren't lazy — every rerun would otherwise recompute them).
@st.cache_data(ttl=900, show_spinner=False)
def _cached_regime():
    return spillover_mod.regime()


@st.cache_data(ttl=900, show_spinner=False)
def _cached_outlook():
    return spillover_mod.next_day_outlook()


@st.cache_data(ttl=900, show_spinner=False)
def _cached_betas():
    return spillover_mod.spillover_betas()


@st.cache_data(ttl=900, show_spinner=False)
def _cached_sensitivities(ticker: str):
    return spillover_mod.stock_sensitivities(ticker)


@st.cache_data(ttl=1800, show_spinner="Ranking universe…")
def _cached_factor_table(tickers: tuple):
    return factors_mod.factor_table(list(tickers))


@st.cache_data(ttl=3600, show_spinner="Walk-forward evaluation (training models)…")
def _cached_ml_eval(tickers: tuple, horizon: int):
    return ml_mod.walk_forward_eval(list(tickers), horizon=horizon)


@st.cache_data(ttl=3600, show_spinner="Training model + scoring…")
def _cached_ml_predict(tickers: tuple, horizon: int):
    return ml_mod.train_and_predict(list(tickers), horizon=horizon)


@st.cache_data(ttl=3600, show_spinner="Computing feature importance…")
def _cached_ml_importance(tickers: tuple, horizon: int):
    return ml_mod.feature_importance(list(tickers), horizon=horizon)


@st.cache_data(ttl=300, show_spinner="Scanning intraday setups…")
def _cached_intraday_scan(tickers: tuple, interval: str):
    return intraday_mod.scan(list(tickers), interval=interval)


@st.cache_data(ttl=900, show_spinner="Analyzing signals (Global · Range · Risk · Rank)…")
def _cached_signal_analysis(key: tuple, capital: float, risk_pct: float, _signals: list):
    return sigan_mod.analyze_signals(_signals, capital=capital, risk_pct=risk_pct)


@st.cache_data(ttl=1800, show_spinner="Screening (factors + liquidity)…")
def _cached_screen(tickers: tuple, min_liquidity: float):
    return screener_mod.screen(list(tickers), min_liquidity=min_liquidity)


@st.cache_data(ttl=600, show_spinner="Comparing execution algos…")
def _cached_algo_compare(ticker: str, side: str, qty: int, notional: float,
                         bucket_minutes: int, participation: float, risk_aversion: float):
    return tca_mod.compare_algos(
        ticker, side, qty=qty or None, notional=notional or None,
        bucket_minutes=bucket_minutes, participation=participation,
        risk_aversion=risk_aversion)


@st.cache_data(ttl=300, show_spinner=False)
def _screen_news_support(tickers: tuple, days: int = 14) -> dict:
    """Per-ticker recent-news read used to flag screener rows the tape is backing.

    For each name: count fresh (≤`days`) bullish vs bearish headlines and keep the
    headline strings for the hover tooltip. `supported` = at least one bullish item
    and bulls outnumber bears — i.e. news agrees with the bullish factor read."""
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    dot = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
    out: dict = {}
    for t in tickers:
        nd = load_news(limit=40, ticker=t)
        if nd.empty:
            out[t] = {"supported": False, "n_bull": 0, "n_bear": 0,
                      "n_total": 0, "headlines": []}
            continue
        # load_news uses a LIKE match (can catch substrings) — keep only rows that
        # actually tag this exact ticker; fall back to the LIKE set if none do.
        tagged = nd[nd["tickers"].fillna("").apply(lambda s: t in s.split(","))]
        if tagged.empty:
            tagged = nd
        pub = pd.to_datetime(tagged["published"], errors="coerce", utc=True)
        recent = tagged[(pub >= cutoff) | pub.isna()]
        n_bull = int((recent["sentiment"] == "bullish").sum())
        n_bear = int((recent["sentiment"] == "bearish").sum())
        heads = []
        for _, r in recent.head(6).iterrows():
            mark = dot.get(r.get("sentiment"), "•")
            when = str(r.get("published") or "")[:10]
            heads.append(f"{mark} {r['title']} ({r.get('source', '')} {when})".strip())
        out[t] = {
            "supported": n_bull >= 1 and n_bull > n_bear,
            "n_bull": n_bull, "n_bear": n_bear, "n_total": len(recent),
            "headlines": heads,
        }
    return out


# Range/Risk tabs: these recompute on EVERY rerun (tabs aren't lazy), so the
# Monte Carlo (thousands of sims) and the portfolio correlation matrix must be
# cached or they lag the whole app on every click.
@st.cache_data(ttl=600, show_spinner=False)
def _cached_expected_range(ticker: str, horizon: int):
    return erange_mod.expected_range(ticker, horizon_days=horizon)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_monte_carlo(ticker: str, horizon: int, method: str):
    return erange_mod.monte_carlo(ticker, horizon_days=horizon, method=method)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_vol_cone(ticker: str):
    return erange_mod.vol_cone(ticker)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_daily_vol(ticker: str):
    return erange_mod.daily_vol(ticker)


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_setup_bt_stats() -> dict:
    """Per-setup performance from the most recent backtest run, keyed by setup
    name — used to annotate live signals with their historical track record."""
    runs = list_backtest_runs()
    if runs.empty:
        return {}
    trades = load_backtest_trades(runs.iloc[0]["run_id"])
    if trades.empty:
        return {}
    # NET of the configured round-trip cost — the signal edge must survive costs.
    summ = bt_metrics.summarize(trades, cost_pct=BACKTEST_COST_PCT)
    out = {}
    for _, row in summ.iterrows():
        if row["setup"] == "ALL":
            continue
        out[row["setup"]] = {
            "n": int(row["n_trades"]),
            "win_rate": float(row["win_rate"]),
            "avg_r": float(row["avg_r"]),
            "expectancy": float(row["expectancy"]),
            "avg_days": float(row["avg_bars_held"]),
        }
    out["_run_id"] = str(runs.iloc[0]["run_id"])
    out["_cost_pct"] = BACKTEST_COST_PCT
    return out


@st.cache_data(ttl=600, show_spinner=False)
def _cached_holding_plan(ticker: str, max_horizon: int, method: str):
    return erange_mod.holding_plan(ticker, max_horizon=max_horizon, method=method)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_target_vs_stop(ticker: str, target_pct: float, stop_pct: float,
                           max_horizon: int, method: str):
    return erange_mod.target_vs_stop(ticker, target_pct=target_pct, stop_pct=stop_pct,
                                     max_horizon=max_horizon, method=method)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_risk_report():
    return risk_mod.portfolio_risk_report()


# Per-holding analysis runs add_indicators for EVERY holding (~27s for 50 names)
# and bypasses the load-level cache, on the default Holdings tab — so it must be
# cached at this level. `key` (holdings snapshot + AI flag) drives the cache;
# the DataFrame arg is underscore-prefixed so Streamlit doesn't try to hash it.
@st.cache_data(ttl=900, show_spinner="Running per-holding analysis…")
def _cached_portfolio_analysis(key: tuple, use_ai_thesis: bool, _df):
    return holdings_mod.analyze_portfolio(_df, use_ai_thesis=use_ai_thesis)


@st.cache_data(ttl=900, show_spinner="Scoring manipulation risk…")
def _cached_manip_cards(tickers: tuple):
    fund_df = load_fundamentals()
    fmap = {r["ticker"]: r for r in fund_df.to_dict("records")} if not fund_df.empty else {}
    cards = []
    for t in tickers:
        cards.append(manipulation.scorecard(t, load_prices(t), fmap.get(t)))
    return cards


@st.cache_data(ttl=900, show_spinner="Fusing all signals into one verdict…")
def _cached_decisions(tickers: tuple, capital: float, risk_pct: float,
                      run_mc: bool, use_ml: bool, plans_key: tuple):
    # plans_key (ticker, amount) pairs are part of the cache key so editing a
    # planned amount re-decides. Rebuilt into a dict for the engine.
    plans = {t: a for t, a in plans_key if a}
    return decision_mod.decide_universe(
        list(tickers), capital=capital, risk_pct=risk_pct,
        run_montecarlo=run_mc, use_ml=use_ml, plans=plans)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_market_pulse():
    return macro_ingest.market_pulse()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_price_coverage(tickers: tuple):
    rows = []
    for t in tickers:
        df = load_prices(t)
        if df.empty:
            rows.append({"ticker": t, "bars": 0, "first": None, "last": None})
        else:
            rows.append({"ticker": t, "bars": len(df),
                         "first": df.index.min().date(), "last": df.index.max().date()})
    return pd.DataFrame(rows)

st.title("SwingDesk — local swing-trading signals (NSE)")
st.caption("Run the daily refresh from the sidebar. All data is stored locally in SQLite.")

# --- Sidebar: actions + watchlist ----------------------------------------------
PAGES = [
    "My Holdings", "💸 Invest", "Discover", "Small Caps", "Signals", "Chart",
    "News", "🔀 Sectors", "🚨 Manipulation", "📅 Calendar", "Backtest", "Optimize",
    "Portfolio", "Reconcile", "Fundamentals", "Raw data",
    "🌐 Global", "📐 Range", "🛡 Risk", "🏆 Rank", "🔎 Screener", "🤖 ML", "⚡ Intraday",
    "🛠 Execution", "📟 Paper Trader",
]

with st.sidebar:
    # Lazy navigation: ONLY the selected section's code runs each rerun (unlike
    # st.tabs, which executes all 22 bodies every time). This is the app's main
    # performance lever. Programmatic jumps set `_jump` before the widget.
    if "_jump" in st.session_state:
        st.session_state["_nav"] = st.session_state.pop("_jump")
    _page = st.selectbox("📍 Section", PAGES, key="_nav")
    st.divider()
    st.header("Actions")
    if st.button("Fetch prices", width="stretch"):
        with st.spinner("Downloading OHLCV..."):
            # Watchlist + holdings (+ small caps) so your portfolio always has
            # chartable price data, not just the curated watchlist.
            res = prices.ingest(combined_universe(include_smallcaps=True),
                                period="2y")
            _clear_data_caches()
        st.success(f"prices: {sum(1 for v in res.values() if v > 0)}/{len(res)} ok")

    if st.button("Fetch news", width="stretch"):
        with st.spinner("Pulling RSS feeds..."):
            # Include small-caps + holdings so headlines naming those names
            # get tagged to the right ticker, not just the main watchlist.
            n = news_rss.ingest(combined_universe(include_smallcaps=True))
        st.success(f"news: {n} new items")

    if st.button("Analyze news (Claude)", width="stretch"):
        _sent_prog = st.progress(0.0, text="Running sentiment analysis…")

        def _sent_cb(done: int, total: int) -> None:
            _sent_prog.progress(done / total if total else 1.0,
                                text=f"Sentiment: {done}/{total} headlines…")

        n = sentiment_mod.ingest(max_items=200, progress=_sent_cb)
        _sent_prog.empty()
        st.success(f"sentiment: {n} headlines classified (Haiku)")

    if st.button("Run scan", width="stretch", type="primary"):
        with st.spinner("Scanning + scoring..."):
            sigs = scan_all(get_watchlist(), persist=False)
            if sigs:
                sigs = scoring.enrich(sigs)
                save_signals(sigs)
        st.success(f"scan: {len(sigs)} signals")

    if st.button("Push to Telegram", width="stretch"):
        sigs_df = load_signals(limit=10)
        if sigs_df.empty:
            st.warning("no signals to push")
        else:
            ok = telegram.send_signals(sigs_df.to_dict(orient="records"))
            st.success("sent" if ok else "failed (check token/chat id)")

    st.divider()
    st.header("Watchlist")
    wl_text = st.text_area(
        "One ticker per line (NSE: use .NS suffix)",
        value="\n".join(get_watchlist()),
        height=300,
    )
    if st.button("Save watchlist", width="stretch"):
        new_wl = [t.strip().upper() for t in wl_text.splitlines() if t.strip()]
        set_watchlist(new_wl)
        st.success(f"saved {len(new_wl)} tickers")
        st.rerun()


# --- Pages: each block runs only when its section is selected (lazy) ------------
# --- Discover -------------------------------------------------------------------
if _page == "Discover":
    st.subheader("Find new stocks to swing-trade")
    st.caption(
        f"Ranked from a curated universe of ~{len(discovery.DISCOVERY_UNIVERSE)} "
        "Indian large + mid caps. Excludes stocks you already hold or have on "
        "your watchlist. Scoring combines fundamentals + trend + momentum + "
        "active setups."
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        min_quality = st.slider("Min quality", 0, 100, 60,
                                  help="Filter out fundamentally weak names")
    with col2:
        only_with_setup = st.checkbox("Only with active setup",
                                       help="Show only stocks with a setup firing today")
    with col3:
        st.write("")
        if st.button("Scan discovery universe", type="primary", width="stretch"):
            with st.spinner("Ranking opportunities..."):
                opps = discovery.scan()
            st.session_state["last_discovery"] = opps
            st.success(f"scanned {len(opps)} candidates")

    opps = st.session_state.get("last_discovery", [])
    if opps:
        # --- High-conviction panel: "invest without thinking too hard"
        hc = discovery.high_conviction(opps)
        if hc:
            st.success(
                f"🎯 **{len(hc)} HIGH-CONVICTION names** — all 4 lenses align "
                "(strong quality + uptrend + volume-confirmed + positive momentum). "
                "These are the ones the engine has fully reviewed and rates as "
                "invest-without-much-thought (still size them sensibly)."
            )
            for o in hc[:5]:
                with st.expander(f"⭐ {o.ticker} · {o.company} · score {o.composite_score:.1f}/100"):
                    st.markdown(f"**Sector:** {o.sector or '?'}  ·  **Price:** ₹{o.price}  "
                                f"·  **Quality:** {o.quality_score:.0f}/100  ·  "
                                f"**Trend:** {getattr(o, 'trend_label', None) or o.technical_state}")
                    if o.reasons:
                        for r in o.reasons:
                            st.markdown(f"- {r}")

        # Apply filters
        filtered = opps
        if min_quality > 0:
            filtered = [o for o in filtered
                         if o.quality_score is not None and o.quality_score >= min_quality]
        if only_with_setup:
            filtered = [o for o in filtered if o.active_setup]

        st.markdown(f"### Top {min(20, len(filtered))} ideas")
        rows = []
        conviction_badge = {"high": "🟢 HIGH", "medium": "🟡 MED", "low": "⚪ low"}
        for o in filtered[:20]:
            rows.append({
                "Conv": conviction_badge.get(getattr(o, "conviction", "low"), "⚪ low"),
                "Rank Score": o.composite_score,
                "Ticker": o.ticker,
                "Company": o.company,
                "Sector": o.sector,
                "Price": f"₹{o.price}",
                "Quality": o.quality_score,
                "Trend": o.technical_state,
                "Trend Quality": getattr(o, "trend_verdict", None) or "—",
                "RSI": o.rsi,
                "20d Mom": f"{o.momentum_20d_pct:+.1f}%" if o.momentum_20d_pct else "—",
                "Vol×Avg": o.volume_x_avg,
                "Setup": o.active_setup or "—",
                "Why": " · ".join(o.reasons[:3]),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # Pick one to chart deep-dive
        st.markdown("### Deep-dive on a specific name")
        if filtered:
            pick = st.selectbox("Stock", [o.ticker for o in filtered[:20]])
            picked_opp = next((o for o in filtered if o.ticker == pick), None)
            if picked_opp:
                col_a, col_b = st.columns([2, 3])
                with col_a:
                    st.markdown(f"**{picked_opp.company}**  ·  {picked_opp.sector}")
                    st.metric("Composite rank score", f"{picked_opp.composite_score}/100")
                    st.markdown(f"- Price: ₹{picked_opp.price}")
                    st.markdown(f"- Quality: {picked_opp.quality_score}")
                    st.markdown(f"- Trend: {picked_opp.technical_state}")
                    st.markdown(f"- RSI: {picked_opp.rsi}")
                    if picked_opp.active_setup:
                        st.success(f"🔔 Active setup: **{picked_opp.active_setup}**")
                    for r in picked_opp.reasons:
                        st.markdown(f"- {r}")
                with col_b:
                    # Mini-summary + chart link hint
                    s = stock_summary.summarize(pick)
                    if s:
                        v_emoji = {"STRONG_BUY": "🟢", "BUY": "🟢",
                                   "WAIT": "⚪", "AVOID": "🔴"}.get(s.verdict, "⚪")
                        st.markdown(f"### {v_emoji} {s.verdict}")
                        st.markdown(s.one_liner)
                        if s.fundamental_brief:
                            st.caption(s.fundamental_brief)
                        if s.why_invest:
                            st.markdown("**Why invest:**")
                            for r in s.why_invest:
                                st.markdown(f"- {r}")
                st.info("→ Switch to the **Chart** tab and pick this ticker for "
                        "full price chart with signal markers.")
    else:
        st.info("Click **Scan discovery universe** above to find new ideas. "
                "Make sure you've run `cli prices` and `cli fundamentals` first.")


# --- Invest tab -----------------------------------------------------------------
if _page == "💸 Invest":
    st.subheader("💸 Invest — decide, time & size your money")
    _invest_mode = st.radio(
        "Mode",
        ["🎯 Decide & time (per stock)", "💸 Split a lump sum"],
        horizontal=True, key="invest_mode",
        help="Decide & time: ONE fused verdict per stock (manipulation + screener + "
             "all factors), a 'buy today vs wait' read, a place to log intended ₹ per "
             "name, and hold/add/trim/exit on what you already own.  ·  Split a lump "
             "sum: spread an amount across the best discovery ideas.",
    )

if _page == "💸 Invest" and _invest_mode == "💸 Split a lump sum":
    st.subheader("💸 Invest fresh money — where & how much")
    st.caption(
        "Tell me how much you want to deploy. I rank the best ideas from the "
        "discovery universe, then split your amount across them — conviction- "
        "weighted, capped per stock, and floored to whole shares — with an "
        "entry, stop and target for each. Rule-based and fully transparent."
    )

    ic1, ic2, ic3 = st.columns([1.3, 1, 1])
    with ic1:
        invest_amt = st.number_input(
            "Amount to invest (₹)", min_value=1000.0, step=5000.0,
            value=float(ACCOUNT_CAPITAL),
            help="Total fresh capital you want to put to work right now.",
        )
    with ic2:
        n_positions = st.slider("Max stocks", 2, 15, min(MAX_OPEN_POSITIONS, 10),
                                help="Diversify across at most this many names.")
        min_conv = st.select_slider(
            "Min conviction", options=["low", "medium", "high"], value="medium",
            help="Only allocate to ideas at or above this conviction band.",
        )
    with ic3:
        max_w = st.slider("Max per stock (%)", 10, 50, 25,
                          help="No single position exceeds this share of the amount.") / 100.0
        st.write("")
        run_alloc = st.button("Build my plan", type="primary", width="stretch")

    # Reuse a prior discovery scan if present; otherwise scan on demand.
    opps_inv = st.session_state.get("last_discovery", [])
    if run_alloc:
        if not opps_inv:
            with st.spinner("Ranking the discovery universe..."):
                opps_inv = discovery.scan()
            st.session_state["last_discovery"] = opps_inv
        with st.spinner("Sizing positions..."):
            plan = allocate_mod.allocate(
                invest_amt, opps_inv, max_positions=n_positions,
                max_weight=max_w, min_conviction=min_conv,
            )
        st.session_state["last_alloc"] = plan

    plan = st.session_state.get("last_alloc")
    if not plan:
        st.info("Set your amount and click **Build my plan**. "
                "(Tip: run **Scan discovery universe** in the Discover tab first "
                "for the freshest rankings.)")
    elif not plan.allocations:
        for n in plan.notes:
            st.warning(n)
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Amount", f"₹{plan.amount:,.0f}")
        m2.metric("Deploying", f"₹{plan.deployed:,.0f}")
        m3.metric("Cash left", f"₹{plan.leftover:,.0f}")
        m4.metric("Stocks", plan.n_positions)

        conv_badge = {"high": "🟢 HIGH", "medium": "🟡 MED", "low": "⚪ low"}
        alloc_rows = []
        for al in plan.allocations:
            alloc_rows.append({
                "Conv": conv_badge.get(al.conviction, "⚪"),
                "Ticker": al.ticker,
                "Company": al.company,
                "Buy shares": al.shares,
                "Invest ₹": f"₹{al.rupees:,.0f}",
                "Weight": f"{al.weight_pct:.1f}%",
                "Price": f"₹{al.price:,.1f}",
                "Entry ≤": f"₹{al.entry:,.1f}" if al.entry else "—",
                "Stop": f"₹{al.stoploss:,.1f}" if al.stoploss else "—",
                "Target": f"₹{al.target:,.1f}" if al.target else "—",
                "R:R": al.rr or "—",
                "₹ at risk": f"₹{al.risk_rupees:,.0f}" if al.risk_rupees else "—",
                "Setup": al.setup or "—",
            })
        st.dataframe(pd.DataFrame(alloc_rows), width="stretch", hide_index=True)

        for n in plan.notes:
            st.caption("• " + n)

        # Pie of the suggested split
        try:
            pie = go.Figure(go.Pie(
                labels=[al.ticker for al in plan.allocations] +
                       (["Cash"] if plan.leftover > 0 else []),
                values=[al.rupees for al in plan.allocations] +
                       ([plan.leftover] if plan.leftover > 0 else []),
                hole=0.45, textinfo="label+percent",
            ))
            pie.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                              title="Suggested allocation")
            st.plotly_chart(pie, width="stretch")
        except Exception:
            pass

        # Per-name reasons
        with st.expander("Why these names?"):
            for al in plan.allocations:
                st.markdown(f"**{al.ticker} · {al.company}** — "
                            f"{' · '.join(al.reasons) if al.reasons else 'ranked by composite score'}")

        st.warning(
            "⚠ Suggestions, not advice. Entry ≤ means try to buy at or below "
            "that level; stops/targets are ATR-based. Re-scan and re-plan as "
            "prices move. Past patterns don't guarantee future returns."
        )

        with st.expander("📖 How the split is decided"):
            st.markdown(glossary.VERDICT_METHODOLOGY)
            st.markdown(
                "**Sizing rules.** Each idea's weight = conviction band "
                "(high×3 / med×2 / low×1) × composite score, normalised, then "
                f"capped at the per-stock limit you set. Top {n_positions} names "
                "by rank are kept (your *Max stocks*), each floored to whole "
                "shares; the remainder stays as cash. Risk per name = "
                "(entry − stop) × shares."
            )


# --- Invest → Decide & time (the unified per-stock decision engine) -------------
_ACTION_BADGE = {"STRONG_BUY": "🟢🟢 STRONG BUY", "BUY": "🟢 BUY",
                 "ACCUMULATE": "🔵 ACCUMULATE", "WAIT": "🟡 WAIT", "AVOID": "🔴 AVOID"}
_TIMING_BADGE = {"ENTER_TODAY": "✅ enter today", "WAIT_PULLBACK": "⏳ wait — pullback",
                 "WAIT_CONFIRMATION": "⏳ wait — confirm", "DONT_ENTER_TODAY": "🚫 don't enter"}
_HOLD_BADGE = {"HOLD": "🟢 HOLD", "ADD": "🔵 ADD", "TRIM": "🟠 TRIM", "EXIT": "🔴 EXIT"}


if _page == "💸 Invest" and _invest_mode == "🎯 Decide & time (per stock)":
    st.caption(
        "ONE verdict per stock, fusing every lens — factor rank, trend, money flow, "
        "expected-range odds, sector & macro regime — with **manipulation risk and "
        "illiquidity as vetoes**. The *action* says whether it's worth owning; the "
        "*timing* says whether **today** is the day to buy or whether to wait. Log "
        "what you intend to invest and I'll size it to your risk and tell you if "
        "today's the day; for stocks you hold I'll say hold / add / trim / exit."
    )

    dc1, dc2, dc3, dc4 = st.columns([1.3, 1, 1, 1])
    with dc1:
        d_scope = st.radio("Universe", ["My holdings", "Watchlist", "Holdings + watchlist"],
                           key="decide_scope")
    with dc2:
        d_capital = st.number_input("Capital (₹)", min_value=10000.0, step=10000.0,
                                    value=float(ACCOUNT_CAPITAL),
                                    help="Used to size positions and the risk per trade.")
    with dc3:
        d_fast = st.checkbox("Fast mode", value=True,
                             help="Skip the Monte-Carlo target-vs-stop odds (faster). "
                                  "Turn off for the expectancy read.")
        d_ml = st.checkbox("Use ML P(up)", value=False,
                           help="Fold the (slow) ML direction model into conviction.")
    with dc4:
        st.write("")
        if st.button("🔄 Re-run", width="stretch",
                     help="Recompute (also clears the 15-min cache)."):
            _cached_decisions.clear()
            st.rerun()

    if d_scope == "My holdings":
        d_univ = holdings_tickers()
    elif d_scope == "Watchlist":
        d_univ = get_watchlist()
    else:
        d_univ = sorted(set(get_watchlist()) | set(holdings_tickers()))

    if not d_univ:
        st.info("No tickers in this universe. Add some to your watchlist (sidebar) "
                "or import holdings.")
    else:
        # --- Plan editor: log intended ₹ per ticker (saved to the `plans` table) ---
        plans_df = load_plans()
        active = plans_df[plans_df["status"] != "dropped"] if not plans_df.empty else plans_df
        plan_amounts = ({r["ticker"]: r["planned_amount"] for r in active.to_dict("records")}
                        if not active.empty else {})

        with st.expander("✍️ Your investment plans — log intended ₹ per stock", expanded=bool(plan_amounts)):
            st.caption("Add a row per stock you're planning to buy. Saved plans drive the "
                       "sizing and the 'invest today?' verdict below.")
            seed_rows = (active[["ticker", "planned_amount", "target_price", "note", "status"]]
                         .to_dict("records") if not active.empty else [])
            editor_df = pd.DataFrame(seed_rows) if seed_rows else pd.DataFrame(
                columns=["ticker", "planned_amount", "target_price", "note", "status"])
            edited = st.data_editor(
                editor_df, num_rows="dynamic", width="stretch", hide_index=True,
                key="plans_editor",
                column_config={
                    "ticker": st.column_config.SelectboxColumn("Ticker", options=d_univ, required=True),
                    "planned_amount": st.column_config.NumberColumn("Intend to invest ₹", min_value=0.0, step=5000.0, format="₹%.0f"),
                    "target_price": st.column_config.NumberColumn("Buy at/below ₹", format="₹%.1f"),
                    "note": st.column_config.TextColumn("Note"),
                    "status": st.column_config.SelectboxColumn("Status", options=["idea", "watching", "entered", "dropped"]),
                },
            )
            if st.button("💾 Save plans", type="primary"):
                saved = 0
                for r in edited.to_dict("records"):
                    t = r.get("ticker")
                    if not t or not r.get("planned_amount"):
                        continue
                    upsert_plan({"ticker": t, "planned_amount": float(r["planned_amount"]),
                                 "target_price": r.get("target_price"),
                                 "note": r.get("note"), "status": r.get("status") or "idea"})
                    saved += 1
                _cached_decisions.clear()
                st.success(f"Saved {saved} plan(s).")
                st.rerun()

        # --- Decide over the universe (cross-sectional inputs computed once) ---
        plans_key = tuple(sorted((t, float(a)) for t, a in plan_amounts.items() if a))
        decisions = _cached_decisions(tuple(sorted(d_univ)), float(d_capital),
                                      float(RISK_PER_TRADE_PCT), not d_fast, d_ml, plans_key)

        if not decisions:
            st.info("Couldn't decide on any of these yet — fetch prices/fundamentals first.")
        else:
            held_set = set(holdings_tickers())
            rows = []
            for d in decisions:
                rows.append({
                    "Ticker": d.ticker,
                    "Action": _ACTION_BADGE.get(d.action, d.action),
                    "Conv": d.conviction,
                    "Today?": _TIMING_BADGE.get(d.timing, d.timing),
                    "Hold": (_HOLD_BADGE.get(d.holding.action, "") if d.holding else ""),
                    "Turn today %": d.today_turnover_pct,
                    "Turn avg %": d.avg_turnover_pct,
                    "today×avg": d.today_vs_avg_value_mult,
                    "Liq": d.liq_tier,
                    "Manip": d.manip_tier,
                    "Entry": d.entry, "Stop": d.stoploss, "Target": d.target, "R:R": d.rr,
                })
            st.markdown(f"**{len(decisions)} stocks · ranked by conviction**  ·  "
                        "*Turn today %* vs *Turn avg %* are the same number the "
                        "Manipulation and Screener tabs show — now from one source.")
            st.dataframe(
                pd.DataFrame(rows), width="stretch", hide_index=True,
                column_config={
                    "Conv": st.column_config.ProgressColumn("Conv", min_value=0, max_value=100, format="%d"),
                    "Turn today %": st.column_config.NumberColumn("Turn today %", format="%.3f",
                        help="Today's traded value ÷ market cap (spike read — what Manipulation shows)."),
                    "Turn avg %": st.column_config.NumberColumn("Turn avg %", format="%.3f",
                        help="60-day average traded value ÷ market cap (tradeability — what the Screener shows)."),
                    "today×avg": st.column_config.NumberColumn("today×avg", format="%.2f×",
                        help="Today's ₹ volume as a multiple of its usual day. ~1 = normal; ≥5 = spike."),
                    "Entry": st.column_config.NumberColumn(format="₹%.1f"),
                    "Stop": st.column_config.NumberColumn(format="₹%.1f"),
                    "Target": st.column_config.NumberColumn(format="₹%.1f"),
                },
            )

            st.divider()
            st.markdown("#### Per-stock detail")
            for d in decisions:
                head = (f"{_ACTION_BADGE.get(d.action, d.action)} · **{d.ticker}** · "
                        f"conviction {d.conviction:.0f} · {_TIMING_BADGE.get(d.timing, d.timing)}")
                with st.expander(head, expanded=(d.action in ("STRONG_BUY", "BUY")
                                                 and d.timing == "ENTER_TODAY")):
                    st.markdown(f"**Today:** {_TIMING_BADGE.get(d.timing, d.timing)} — {d.timing_reason}")

                    # "Invest ₹X today?" verdict, when a plan amount is logged.
                    if d.plan:
                        p = d.plan
                        if d.timing == "ENTER_TODAY" and p.suggested_shares > 0:
                            v = (f"✅ **Yes** — buy ~{p.suggested_shares} shares "
                                 f"(₹{p.suggested_value:,.0f}, {p.pct_of_capital:.0f}% of capital, "
                                 f"₹{p.risk_amount:,.0f} at risk). {p.advice}.")
                        elif d.timing == "DONT_ENTER_TODAY":
                            v = f"🚫 **Not today** — {d.timing_reason}"
                        else:
                            v = (f"⏳ **Hold the cash** — {d.timing_reason} "
                                 f"When it triggers, size ~{p.suggested_shares} shares "
                                 f"(₹{p.suggested_value:,.0f}). {p.advice}.")
                        st.info(f"Plan ₹{p.planned_amount:,.0f}: {v}")

                    # Hold / add / trim / exit for a name you already own.
                    if d.holding:
                        h = d.holding
                        st.warning(f"You hold {h.qty:g} @ ₹{h.avg_price:,.1f} "
                                   f"({h.unrealized_pct:+.1f}%): **{_HOLD_BADGE.get(h.action, h.action)}** — {h.reason}")

                    cpros, ccons = st.columns(2)
                    with cpros:
                        st.caption("Why it's interesting")
                        for x in d.pros:
                            st.markdown(f"- {x}")
                    with ccons:
                        st.caption("Watch-outs")
                        for x in (d.cons or ["—"]):
                            st.markdown(f"- {x}")

                    mc = st.columns(6)
                    mc[0].metric("Conviction", f"{d.conviction:.0f}")
                    mc[1].metric("Tech tilt", d.tech_tilt or "—")
                    mc[2].metric("Factor Q", f"Q{d.factor_quintile}" if d.factor_quintile else "—")
                    mc[3].metric("Trend", d.trend_verdict or "—")
                    mc[4].metric("Exp. R", f"{d.expectancy_r:+.2f}" if d.expectancy_r is not None else "—")
                    mc[5].metric("P(tgt 1st)", f"{d.p_target_first:.0%}" if d.p_target_first is not None else "—")

                    mc2 = st.columns(6)
                    mc2[0].metric("Manip", d.manip_tier)
                    mc2[1].metric("Liquidity", d.liq_tier)
                    mc2[2].metric("Turn today %", f"{d.today_turnover_pct:.3f}" if d.today_turnover_pct is not None else "—")
                    mc2[3].metric("Turn avg %", f"{d.avg_turnover_pct:.3f}" if d.avg_turnover_pct is not None else "—")
                    mc2[4].metric("ADV ₹cr", f"{d.adv_value_cr:.1f}" if d.adv_value_cr is not None else "—")
                    mc2[5].metric("Quality", f"{d.quality_score:.0f}" if d.quality_score is not None else "—")

                    if d.entry:
                        st.caption(f"Levels ({d.level_source}): entry ₹{d.entry:,.1f} · "
                                   f"stop ₹{d.stoploss:,.1f} · target ₹{d.target:,.1f} · R:R {d.rr}")
                    if d.data_gaps:
                        st.caption("Data gaps: " + " · ".join(d.data_gaps))

            st.warning("⚠ Suggestions, not advice. The *action* is a conviction read; the "
                       "*timing* is about today's entry. Re-run as prices move.")


# --- Small Caps tab -------------------------------------------------------------
if _page == "Small Caps":
    st.subheader("Small-cap workbench")
    st.caption(
        "Three sections: **Discover** finds names from the curated universe and "
        "lets you add them to your small-cap watchlist; **My Small-cap Watchlist** "
        "manages that list and pulls data + scans for signals; **Charts & Analysis** "
        "is the deep-dive view. Signals from here are tagged `universe=smallcap` "
        "and appear in the **Signals** tab with that label."
    )

    sc_sub1, sc_sub2, sc_sub3 = st.tabs(
        ["🔍 Discover & Add", "📋 My Small-cap Watchlist", "📊 Charts & Analysis"]
    )

    # ---------- 🔍 Discover & Add ----------
    with sc_sub1:
        st.markdown(
            f"Scan the curated **{len(smallcaps.SMALLCAP_UNIVERSE)}-name** small-cap "
            "universe and add the ones you like to your watchlist."
        )
        d_col1, d_col2, d_col3 = st.columns([1, 1, 2])
        with d_col1:
            d_min_quality = st.slider("Min quality", 0, 100, 55, key="sc_d_minq")
        with d_col2:
            d_active = st.checkbox("Active setup only", key="sc_d_active")
        with d_col3:
            st.write("")
            if st.button("Scan universe", type="primary", width="stretch",
                          key="sc_d_scan"):
                with st.spinner("Ranking..."):
                    st.session_state["sc_disc_opps"] = smallcaps.scan()
                st.success(f"scanned {len(st.session_state['sc_disc_opps'])} candidates")

        opps = st.session_state.get("sc_disc_opps", [])
        if not opps:
            st.info("Click **Scan universe** to discover small-cap candidates.")
        else:
            # Apply filters
            filt = opps
            if d_min_quality > 0:
                filt = [o for o in filt
                         if o.quality_score is not None
                         and o.quality_score >= d_min_quality]
            if d_active:
                filt = [o for o in filt if o.active_setup]

            sc_wl_set = set(get_smallcap_watchlist())

            # High-conviction subset highlighted up top
            hc = [o for o in filt if getattr(o, "conviction", "low") == "high"]
            if hc:
                st.success(f"🎯 **{len(hc)} HIGH-CONVICTION** names below ↓")

            st.markdown(f"### Top {min(20, len(filt))} candidates")
            for o in filt[:20]:
                conv = getattr(o, "conviction", "low")
                badge = {"high": "🟢", "medium": "🟡", "low": "⚪"}.get(conv, "⚪")
                already = o.ticker in sc_wl_set
                col_l, col_m, col_r = st.columns([1, 5, 1])
                with col_l:
                    st.markdown(f"**{badge} {o.composite_score:.0f}**")
                with col_m:
                    st.markdown(
                        f"**{o.ticker}** · {o.company or '—'} · "
                        f"{o.sector or '?'}  ·  ₹{o.price}  ·  "
                        f"Q={o.quality_score or '?'}  ·  "
                        f"trend={o.technical_state}  ·  "
                        f"20d {o.momentum_20d_pct:+.1f}%"
                        if o.momentum_20d_pct is not None else
                        f"**{o.ticker}** · {o.company or '—'} · {o.sector or '?'} · ₹{o.price}"
                    )
                    if o.reasons:
                        st.caption(" · ".join(o.reasons[:3]))
                with col_r:
                    if already:
                        st.caption("✓ in WL")
                    else:
                        if st.button("Add", key=f"sc_add_{o.ticker}"):
                            add_to_smallcap_watchlist(o.ticker)
                            st.toast(f"added {o.ticker} to small-cap WL")
                            st.rerun()
                st.divider()

    # ---------- 📋 My Small-cap Watchlist ----------
    with sc_sub2:
        sc_wl = get_smallcap_watchlist()
        st.markdown(f"### Small-cap watchlist ({len(sc_wl)} tickers)")

        if not sc_wl:
            st.info(
                "Your small-cap watchlist is empty. Go to **🔍 Discover & Add** to "
                "find candidates, or paste tickers manually below."
            )
        else:
            # Render as a removable chip list
            cols = st.columns(4)
            for i, t in enumerate(sc_wl):
                with cols[i % 4]:
                    if st.button(f"❌ {t.replace('.NS', '')}",
                                 key=f"sc_rm_{t}", width="stretch"):
                        remove_from_smallcap_watchlist(t)
                        st.rerun()

        st.divider()

        # Manual edit
        with st.expander("Add tickers manually / edit list"):
            wl_text = st.text_area(
                "One ticker per line (use .NS suffix)",
                value="\n".join(sc_wl),
                height=200,
                key="sc_wl_text",
            )
            cwl1, cwl2 = st.columns(2)
            with cwl1:
                if st.button("💾 Save list", type="primary", width="stretch",
                              key="sc_wl_save"):
                    new_list = [t.strip().upper() for t in wl_text.splitlines() if t.strip()]
                    set_smallcap_watchlist(new_list)
                    st.success(f"saved {len(new_list)} tickers")
                    st.rerun()
            with cwl2:
                if st.button("🗑 Clear list", width="stretch", key="sc_wl_clear"):
                    set_smallcap_watchlist([])
                    st.rerun()

        st.divider()

        # Data fetch buttons — scoped to the SMALL-CAP WL only
        st.markdown("### Refresh data (only for the small-cap watchlist)")
        if not sc_wl:
            st.caption("Add at least one ticker before fetching data.")
        else:
            ba, bb, bc, bd = st.columns(4)
            with ba:
                if st.button("📈 Prices", width="stretch", key="sc_wl_prices",
                              help=f"Pull 1y OHLCV for {len(sc_wl)} tickers"):
                    with st.spinner(f"Fetching prices for {len(sc_wl)} tickers..."):
                        res = prices.ingest(sc_wl, period="1y", workers=6)
                    _clear_data_caches()
                    ok = sum(1 for v in res.values() if v > 0)
                    st.success(f"prices: {ok}/{len(res)} ok")
            with bb:
                if st.button("📊 Fundamentals", width="stretch",
                              key="sc_wl_fund"):
                    with st.spinner("Fetching fundamentals..."):
                        n = fundamentals_ingest.ingest(sc_wl, workers=5)
                    _clear_data_caches()
                    st.success(f"saved {n} fundamentals")
            with bc:
                if st.button("📰 News", width="stretch", key="sc_wl_news",
                              help="Pulls RSS + tags against the wider universe"):
                    with st.spinner("Pulling RSS feeds..."):
                        universe = combined_universe(
                            include_smallcaps=True, include_discovery=True,
                        )
                        n = news_rss.ingest(universe)
                    st.success(f"news: {n} new items")
            with bd:
                if st.button("🧠 Sentiment", width="stretch", key="sc_wl_sent"):
                    with st.spinner("Running Claude sentiment..."):
                        n = sentiment_mod.ingest(max_items=200)
                    st.success(f"sentiment: {n} headlines")

            st.divider()

            # The scan — produces signals tagged universe='smallcap'
            if st.button("🚀 Run scan on small-cap watchlist",
                          type="primary", width="stretch", key="sc_wl_scan"):
                with st.spinner(f"Scanning {len(sc_wl)} small caps for setups..."):
                    sigs = scan_all(sc_wl, persist=False, universe="smallcap")
                    if sigs:
                        sigs = scoring.enrich(sigs)
                        save_signals(sigs, universe="smallcap")
                if sigs:
                    st.success(
                        f"✅ {len(sigs)} small-cap signals generated. "
                        "Open the **Signals** tab and filter by `smallcap` to see them."
                    )
                    for s in sigs[:5]:
                        st.markdown(
                            f"- **{s['ticker']}** · {s['setup']}  ·  "
                            f"entry ₹{s.get('entry')}  ·  SL ₹{s.get('stoploss')}  ·  "
                            f"target ₹{s.get('target')}  ·  R:R {s.get('rr')}"
                        )
                else:
                    st.info("No setups firing on the small-cap watchlist today. "
                            "That's normal — quality setups are rare. Try again tomorrow.")

    # ---------- 📊 Charts & Analysis ----------
    with sc_sub3:
        sc_wl = get_smallcap_watchlist()
        if not sc_wl:
            st.info("Add stocks to your small-cap watchlist (📋 My Small-cap "
                    "Watchlist tab) first to chart them here.")
        else:
            ck1, ck2 = st.columns([1, 4])
            with ck1:
                sc_pick = st.selectbox("Stock", sc_wl, key="sc_ch_pick")
                sc_lb = st.slider("Lookback (days)", 60, 500, 180, key="sc_ch_lb")
                show_markers_sc = st.checkbox("Signal markers",
                                                 value=True, key="sc_ch_marks")

            sc_df = load_prices(sc_pick, days=sc_lb)
            if sc_df.empty:
                with st.spinner(f"Fetching prices for {sc_pick}..."):
                    fresh = prices.fetch_one(sc_pick, period="1y")
                    if fresh is not None and not fresh.empty:
                        from swingdesk.storage import upsert_prices
                        upsert_prices(sc_pick, fresh)
                        load_prices.clear()
                        sc_df = load_prices(sc_pick, days=sc_lb)
            if sc_df.empty:
                st.warning(f"No price data for {sc_pick}. Use **📋 My Small-cap "
                           "Watchlist** → 📈 Prices first.")
            else:
                sc_df = add_indicators(sc_df)

                # Investability summary card
                sc_s = stock_summary.summarize(sc_pick)
                if sc_s:
                    emoji, banner = {
                        "STRONG_BUY": ("🟢", "success"), "BUY": ("🟢", "success"),
                        "WAIT": ("⚪", "info"), "AVOID": ("🔴", "error"),
                    }.get(sc_s.verdict, ("⚪", "info"))
                    getattr(st, banner)(
                        f"{emoji} **{sc_s.verdict}** · {sc_s.company} "
                        f"({sc_s.sector or 'Unknown'}) · ₹{sc_s.current_price:,.2f}\n\n"
                        f"_{sc_s.one_liner}_"
                    )
                    if sc_s.fundamental_brief:
                        st.caption(sc_s.fundamental_brief)
                    sc_c1, sc_c2 = st.columns(2)
                    if sc_s.why_invest:
                        sc_c1.markdown("**✓ Why invest**")
                        for r in sc_s.why_invest:
                            sc_c1.markdown(f"- {r}")
                    if sc_s.why_avoid:
                        sc_c2.markdown("**⚠ Why avoid / wait**")
                        for r in sc_s.why_avoid:
                            sc_c2.markdown(f"- {r}")

                # Candlestick + EMAs + volume (shared x) + reason-rich markers
                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.74, 0.26], vertical_spacing=0.03,
                    subplot_titles=("", "Volume (green=up day, red=down day)"),
                )
                fig.add_trace(go.Candlestick(
                    x=sc_df.index, open=sc_df["open"], high=sc_df["high"],
                    low=sc_df["low"], close=sc_df["close"], name="Price",
                ), row=1, col=1)
                for col, color in [("ema20", "#1f77b4"), ("ema50", "#ff7f0e"),
                                     ("ema200", "#888")]:
                    if col in sc_df.columns:
                        fig.add_trace(go.Scatter(
                            x=sc_df.index, y=sc_df[col], mode="lines",
                            name=col.upper(), line=dict(width=1, color=color),
                        ), row=1, col=1)
                sc_up = (sc_df["close"] >= sc_df["close"].shift(1)).fillna(True)
                fig.add_trace(go.Bar(
                    x=sc_df.index, y=sc_df["volume"],
                    marker_color=np.where(sc_up, "#2ca02c", "#d62728"),
                    name="Volume", showlegend=False,
                    hovertemplate="Vol %{y:,.0f}<extra></extra>",
                ), row=2, col=1)
                if "vol_avg20" in sc_df.columns:
                    fig.add_trace(go.Scatter(
                        x=sc_df.index, y=sc_df["vol_avg20"], mode="lines",
                        name="Vol 20d-avg",
                        line=dict(color="#222", width=1.3, dash="dash"),
                        showlegend=False,
                        hovertemplate="20d-avg %{y:,.0f}<extra></extra>",
                    ), row=2, col=1)
                if show_markers_sc:
                    sc_ev = chart_signals.events_for_ticker(sc_pick, lookback=sc_lb)
                    sc_ev = [e for e in sc_ev
                              if pd.Timestamp(e.date) >= sc_df.index.min()]

                    def _sc_hover(e, tag):
                        parts = [f"<b>{e.setup}</b> · {pd.Timestamp(e.date):%d %b %Y}"]
                        if e.notes:
                            parts.append(f"<i>{e.notes}</i>")
                        lv = []
                        if e.entry is not None:
                            lv.append(f"entry ₹{e.entry:,.1f}")
                        if e.stoploss is not None:
                            lv.append(f"SL ₹{e.stoploss:,.1f}")
                        if e.target is not None:
                            lv.append(f"tgt ₹{e.target:,.1f}")
                        if e.rr is not None:
                            lv.append(f"R:R {e.rr}")
                        if lv:
                            parts.append(" · ".join(lv))
                        parts.append(tag)
                        return "<br>".join(parts)

                    for outcome, symbol, color, size, tagfn, legend in [
                        ("target", "triangle-up", "#2ca02c", 14,
                         lambda e: f"{e.r_multiple:+.1f}R 🎯", "Profitable signal"),
                        ("stop", "triangle-down", "#d62728", 14,
                         lambda e: f"{e.r_multiple:+.1f}R ❌", "Stopped-out"),
                        ("open", "star", "#ffd700", 18,
                         lambda e: "🔔 active", "Active"),
                    ]:
                        evs = [e for e in sc_ev if e.outcome == outcome]
                        if not evs:
                            continue
                        fig.add_trace(go.Scatter(
                            x=[e.date for e in evs], y=[e.price for e in evs],
                            mode="markers",
                            marker=dict(symbol=symbol, color=color, size=size,
                                        line=dict(color="white", width=1)),
                            text=[_sc_hover(e, tagfn(e)) for e in evs],
                            hoverinfo="text", name=legend,
                        ), row=1, col=1)
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                fig.update_yaxes(title_text="Price ₹", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                fig.update_layout(height=640, xaxis_rangeslider_visible=False,
                                    xaxis2_rangeslider_visible=False,
                                    margin=dict(l=10, r=10, t=30, b=10),
                                    showlegend=True, bargap=0)
                st.plotly_chart(fig, width="stretch")

                # Volume profile + OBV + MFI
                vp = volume_profile(sc_df, bins=24, lookback=60)
                if not vp.empty:
                    with st.expander("Volume profile + Money flow over time"):
                        vp_fig = go.Figure()
                        vp_fig.add_trace(go.Bar(
                            x=vp["volume"], y=vp["price_mid"], orientation="h",
                            marker_color="#888",
                            width=(vp["price_high"] - vp["price_low"]),
                        ))
                        last_close = float(sc_df["close"].iloc[-1])
                        vp_fig.add_hline(y=last_close, line_dash="dash",
                                          line_color="red")
                        sc_vp_read = volume_profile_read(sc_df, vp)
                        if sc_vp_read:
                            vp_fig.add_hline(y=sc_vp_read["poc"], line_dash="dot",
                                             line_color="#1f77b4")
                            vp_fig.add_hrect(y0=sc_vp_read["value_area_low"],
                                             y1=sc_vp_read["value_area_high"],
                                             fillcolor="#1f77b4", opacity=0.08,
                                             line_width=0)
                        vp_fig.update_layout(height=320,
                                              margin=dict(l=10, r=10, t=10, b=10),
                                              xaxis_title="Volume",
                                              yaxis_title="Price",
                                              showlegend=False)
                        st.plotly_chart(vp_fig, width="stretch")
                        if sc_vp_read:
                            for r in sc_vp_read["reasons"]:
                                st.markdown(f"- {r}")

                        sc_mfr = money_flow_read(sc_df)
                        if "obv" in sc_df.columns:
                            obvf = go.Figure(go.Scatter(
                                x=sc_df.index, y=sc_df["obv"], mode="lines",
                                name="OBV", line=dict(color="#1f77b4"),
                            ))
                            obvf.update_layout(height=160, title="OBV",
                                                margin=dict(l=10, r=10, t=30, b=10))
                            st.plotly_chart(obvf, width="stretch")
                            if sc_mfr.get("obv"):
                                st.markdown(f"**{sc_mfr['obv']}**")
                            if sc_mfr.get("obv_detail"):
                                st.caption(sc_mfr["obv_detail"])
                        if "mfi14" in sc_df.columns:
                            mff = go.Figure(go.Scatter(
                                x=sc_df.index, y=sc_df["mfi14"], mode="lines",
                                line=dict(color="#ff7f0e"),
                            ))
                            mff.add_hline(y=80, line_dash="dot", line_color="red")
                            mff.add_hline(y=20, line_dash="dot", line_color="green")
                            mff.update_layout(height=160, title="MFI(14)",
                                                yaxis_range=[0, 100],
                                                margin=dict(l=10, r=10, t=30, b=10))
                            st.plotly_chart(mff, width="stretch")
                            if sc_mfr.get("mfi"):
                                st.markdown(f"**{sc_mfr['mfi']}**")
                            if sc_mfr.get("mfi_detail"):
                                st.caption(sc_mfr["mfi_detail"])

                # Methodology reference
                with st.expander("📖 How SwingDesk decides — the norms behind this"):
                    st.markdown(glossary.full_methodology())

                # News for this small cap
                with st.expander(f"Recent news for {sc_pick}"):
                    sc_news = load_news(limit=10, ticker=sc_pick)
                    if sc_news.empty:
                        st.caption("No news yet. Click **📰 News** in the watchlist tab.")
                    else:
                        sent_color = {"bullish": "#1f8a3a", "bearish": "#c0392b",
                                       "neutral": "#888"}
                        for _, row in sc_news.iterrows():
                            badge = ""
                            if row.get("sentiment"):
                                color = sent_color.get(row["sentiment"], "#888")
                                badge = (f" <span style='background:{color};color:white;"
                                         f"padding:2px 6px;border-radius:3px;"
                                         f"font-size:0.75em'>{row['sentiment']}/"
                                         f"{row.get('impact', '?')}</span>")
                            st.markdown(
                                f"**[{row['title']}]({row['link']})**{badge}",
                                unsafe_allow_html=True,
                            )
                            st.caption(f"{row['source']} · {row.get('published', '')}")

                # AI thesis on demand
                with st.expander(f"🧠 Claude AI thesis for {sc_pick}"):
                    key = f"sc_th_{sc_pick}"
                    if st.button(f"Generate thesis", key=f"sc_th_btn_{sc_pick}"):
                        from swingdesk.analyze import thesis as thesis_mod
                        from swingdesk.storage import get_fundamentals
                        fund = get_fundamentals(sc_pick) or {}
                        last = sc_df.iloc[-1]
                        tech_state = {
                            "state": ("uptrend" if last.get("ema50") and last.get("ema200")
                                       and last["close"] > last["ema50"]
                                       and last["close"] > last["ema200"] else "unknown"),
                            "rsi": float(last["rsi14"]) if pd.notna(last.get("rsi14")) else None,
                            "above_50ema": bool(last.get("ema50") and last["close"] > last["ema50"]),
                            "above_200ema": bool(last.get("ema200") and last["close"] > last["ema200"]),
                            "mfi": float(last["mfi14"]) if pd.notna(last.get("mfi14")) else None,
                        }
                        with st.spinner("Asking Claude..."):
                            t_obj = thesis_mod.generate(
                                ticker=sc_pick, qty=0, avg_buy=None,
                                last_price=float(last["close"]), pnl_pct=None,
                                fundamentals=fund, technical_state=tech_state,
                                recent_news=load_news(limit=10, ticker=sc_pick),
                            )
                        st.session_state[key] = t_obj
                    t_obj = st.session_state.get(key)
                    if t_obj is not None:
                        action_emoji = {"BUY_MORE": "🟢", "HOLD": "⚪",
                                         "TRIM": "🟡", "EXIT": "🔴"}.get(t_obj.action, "⚪")
                        st.success(f"{action_emoji} **{t_obj.action}** · "
                                   f"conviction **{t_obj.conviction}/100**")
                        st.markdown(f"**Thesis.** {t_obj.narrative}")
                        if t_obj.catalyst_to_watch:
                            st.markdown(f"**Catalyst.** {t_obj.catalyst_to_watch}")
                        if t_obj.risks:
                            st.markdown("**Top risks:**")
                            for r in t_obj.risks:
                                st.markdown(f"- {r}")


# --- Signals tab ----------------------------------------------------------------
if _page == "Signals":
    st.subheader("Latest signals")

    fcol1, fcol2 = st.columns([1, 4])
    with fcol1:
        sig_universe = st.radio(
            "Universe",
            ["All", "Main (large/mid)", "Small-cap"],
            horizontal=False,
            key="sig_universe_filter",
        )
    universe_filter = {"All": None, "Main (large/mid)": "main",
                        "Small-cap": "smallcap"}[sig_universe]
    sigs = load_signals(limit=200, universe=universe_filter)
    if sigs.empty:
        st.info(
            "No signals match this filter yet. Run **Run scan** in the sidebar "
            "(main universe) or **🚀 Run scan** in the Small Caps → My Watchlist "
            "tab (small caps)."
        )
    else:
        sigs["generated_at"] = pd.to_datetime(sigs["generated_at"])
        latest = sigs.sort_values("generated_at", ascending=False)
        last_batch_ts = latest["generated_at"].max()
        last_batch = latest[latest["generated_at"] == last_batch_ts]

        # Headline counts by universe
        with fcol2:
            counts = sigs["universe"].value_counts().to_dict()
            badges = []
            if counts.get("main"):
                badges.append(f"🟦 main: **{counts['main']}**")
            if counts.get("smallcap"):
                badges.append(f"🟧 smallcap: **{counts['smallcap']}**")
            if badges:
                st.markdown(" · ".join(badges))
            st.caption(f"Most recent batch: {last_batch_ts} ({len(last_batch)} signals)")

        # Color-code the universe column with an emoji badge
        latest = latest.reset_index(drop=True)
        latest_display = latest.copy()
        latest_display["universe"] = latest_display["universe"].map({
            "main": "🟦 main", "smallcap": "🟧 smallcap"
        }).fillna(latest_display["universe"])

        # Est. sessions to reach the target: vol-implied days from the stock's
        # own EWMA volatility — the horizon at which the entry→target move is a
        # typical (1σ) move. A realistic "how long are we targeting" per signal.
        _volmap = {t: _cached_daily_vol(t) for t in latest["ticker"].unique()}

        def _est_days(row):
            d = erange_mod.vol_implied_days(_volmap.get(row["ticker"]),
                                            row.get("entry"), row.get("target"))
            return round(d) if d is not None else None
        latest_display["est_days"] = latest_display.apply(_est_days, axis=1)

        # --- Backtest track record per setup (from the latest backtest run).
        # Tells you whether each signal's SETUP has a historical edge — a signal
        # from a negative-expectancy setup is a warning, not an invitation.
        _bt = _cached_setup_bt_stats()

        def _bt_edge(setup):
            s = _bt.get(setup)
            if not s or s["n"] < 20:
                return "— (insufficient)"
            e = s["expectancy"]
            return ("✅ edge" if e > 0.05 else "⚠️ negative" if e < -0.05 else "➖ flat")
        latest_display["bt_win"] = latest["setup"].map(
            lambda s: (_bt[s]["win_rate"] * 100) if s in _bt else None)
        latest_display["bt_avg_r"] = latest["setup"].map(
            lambda s: _bt.get(s, {}).get("avg_r"))
        latest_display["bt_days"] = latest["setup"].map(
            lambda s: _bt.get(s, {}).get("avg_days"))
        latest_display["bt_edge"] = latest["setup"].map(_bt_edge)

        if not _bt:
            st.warning("No backtest run yet — run one in the **Backtest** tab to "
                       "annotate each signal with its setup's historical edge.")
        else:
            st.caption(
                f"💡 Click a row to open it in the **Chart** tab. *Est. days→target* "
                f"= volatility-implied holding horizon. **bt_** columns = this "
                f"setup's historical track record from the latest backtest "
                f"(run {_bt.get('_run_id','?')}), **NET of {_bt.get('_cost_pct', 0):.2f}% "
                f"round-trip costs**: win-rate, avg R, avg days held, and whether it "
                f"has a positive-expectancy **edge**. Treat ⚠️ negative-expectancy "
                f"setups with caution — most setups lose after costs."
            )
        sig_event = st.dataframe(
            latest_display[
                ["generated_at", "universe", "ticker", "setup", "direction",
                 "entry", "stoploss", "target", "rr", "est_days",
                 "bt_edge", "bt_win", "bt_avg_r", "bt_days", "score", "notes"]
            ],
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="signals_table",
            column_config={
                "est_days": st.column_config.NumberColumn(
                    "Est. days→target", format="%d",
                    help="Volatility-implied sessions to reach the target "
                         "(≈ (ln(target/entry) ÷ daily-vol)²). Typical holding "
                         "horizon if the move plays out — not a guarantee."),
                "bt_edge": st.column_config.TextColumn(
                    "Setup edge",
                    help="From the latest backtest: ✅ positive expectancy, "
                         "➖ ~flat, ⚠️ negative. '—' = <20 trades to judge."),
                "bt_win": st.column_config.NumberColumn(
                    "Setup win%", format="%.0f%%",
                    help="Historical win rate of this setup"),
                "bt_avg_r": st.column_config.NumberColumn(
                    "Setup avg R", format="%.2f",
                    help="Average R-multiple per trade (expectancy). >0 = edge."),
                "bt_days": st.column_config.NumberColumn(
                    "Setup avg days", format="%.0f",
                    help="Avg bars held historically — empirical holding period "
                         "(compare with Est. days→target)."),
            },
        )

        # A clicked row -> preload the Chart tab with that ticker and jump there.
        picked = sig_event.get("selection", {}).get("rows", []) if sig_event else []
        if picked:
            chosen = str(latest.iloc[picked[0]]["ticker"])
            st.session_state["chart_ticker"] = chosen
            st.session_state["_jump"] = "Chart"   # lazy-nav to the Chart page
            st.rerun()

        # --- 🔬 Full analysis: fuse Global · Range · Risk · Rank onto every signal
        st.divider()
        st.markdown("### 🔬 Full analysis — Global · Range · Risk · Rank")
        st.caption(
            "For every signal: factor **Rank** (quintile), **Range** = P(target "
            "before stop) + expectancy from the signal's own levels, **Risk** = "
            "position size at your configured risk, and **Global** = beta to NIFTY "
            "and US-tech. Runs on demand (Monte-Carlo per name)."
        )
        MAX_ANALYZE = 60
        rows_all = latest.to_dict("records")
        rows_an = rows_all[:MAX_ANALYZE]
        if st.button(f"Run full analysis on {len(rows_an)} signal(s)",
                     type="primary", width="stretch"):
            # Market-wide Global context (one regime for all signals).
            reg = _cached_regime()
            if reg is not None:
                rb = {"risk-on": "success", "neutral": "info", "risk-off": "error"}[reg.label]
                getattr(st, rb)(f"🌐 **Market regime: {reg.label.upper()}** "
                                f"(score {reg.score:+d}) — applies to every signal below.")
            if len(rows_all) > MAX_ANALYZE:
                st.caption(f"Analyzing the top {MAX_ANALYZE} of {len(rows_all)} signals "
                           f"(Monte-Carlo is per-name).")
            akey = tuple((r.get("ticker"), r.get("setup"), r.get("entry"),
                          r.get("stoploss"), r.get("target"), r.get("direction"))
                         for r in rows_an)
            adf = _cached_signal_analysis(akey, float(ACCOUNT_CAPITAL),
                                          float(RISK_PER_TRADE_PCT), rows_an)
            if adf.empty:
                st.info("Couldn't analyze — need price/macro history. Fetch data first.")
            else:
                st.dataframe(
                    adf, width="stretch", hide_index=True,
                    column_config={
                        "rank_q": st.column_config.NumberColumn(
                            "Rank Q", help="Factor quintile across these names (1=best)"),
                        "rank_score": st.column_config.NumberColumn(
                            "Rank score", format="%.2f", help="Composite factor z-score"),
                        "p_target_first": st.column_config.ProgressColumn(
                            "P(target 1st)", format="%.0f%%", min_value=0.0, max_value=1.0,
                            help="P(hit target before stop) from the signal's own levels"),
                        "exp_R": st.column_config.NumberColumn(
                            "Reward:risk exp (R)", format="%.2f",
                            help="Expectancy of the target-vs-stop bet in R. >0 = favourable"),
                        "exp_move_10d_pct": st.column_config.NumberColumn(
                            "±1σ 10d %", format="%.1f", help="Expected 10-session move"),
                        "shares": st.column_config.NumberColumn("Shares", format="%d"),
                        "pct_cap": st.column_config.NumberColumn("% cap", format="%.0f%%"),
                        "risk_amt": st.column_config.NumberColumn("Risk ₹", format="%.0f"),
                        "beta_nifty": st.column_config.NumberColumn(
                            "β NIFTY", format="%.2f", help="Sensitivity to the index"),
                        "beta_nasdaq": st.column_config.NumberColumn(
                            "β NASDAQ", format="%.2f", help="US-tech overnight sensitivity"),
                    },
                )
                st.caption("Sorted by reward-vs-risk expectancy. Cross-check: a "
                           "high P(target 1st) + positive exp R + top Rank Q + ✅ "
                           "setup edge above is the strongest stack. Size with **% cap**.")

# --- Chart ----------------------------------------------------------------------
if _page == "Chart":
    # Universe = watchlist + holdings, so users can chart anything they own
    universe = combined_universe()
    held_set = set(holdings_tickers())
    if not universe:
        st.info("Add tickers to your watchlist or import holdings first.")
    else:
        col1, col2 = st.columns([1, 4])
        with col1:
            # Sort holdings first (★ emoji indicator), then watchlist
            options = sorted(universe, key=lambda t: (t not in held_set, t))
            # A ticker arriving from a signal click may not be in the base
            # universe (e.g. a small-cap signal) — make it selectable anyway.
            sel = st.session_state.get("chart_ticker")
            if sel and sel not in options:
                options = [sel] + options
            display = {t: f"★ {t}" if t in held_set else t for t in options}
            ticker = st.selectbox("Ticker", options, key="chart_ticker",
                                   format_func=lambda t: display.get(t, t))
            lookback = st.slider("Lookback (days)", 60, 500, 180)
            show_markers = st.checkbox("Show signal labels", value=True,
                                        help="Plot historical entry signals as markers")
            overlays = st.multiselect(
                "Overlays", ["Bollinger Bands", "Supertrend"],
                default=["Supertrend"],
                help="Bollinger = volatility envelope; Supertrend = buy/sell regime line",
            )

        df = load_prices(ticker, days=lookback)
        if df.empty:
            st.warning(
                f"No price data for {ticker}. Run **Fetch prices** from the sidebar "
                "(this will include your holdings now)."
            )
        else:
            df = add_indicators(df)

            # --- Investability summary card on top of chart
            s = stock_summary.summarize(ticker)
            if s:
                verdict_color = {
                    "STRONG_BUY": ("🟢", "success"),
                    "BUY": ("🟢", "success"),
                    "WAIT": ("⚪", "info"),
                    "AVOID": ("🔴", "error"),
                }
                emoji, banner = verdict_color.get(s.verdict, ("⚪", "info"))
                getattr(st, banner)(
                    f"{emoji} **{s.verdict}** · {s.company} ({s.sector or 'Unknown'}) · "
                    f"₹{s.current_price:,.2f}\n\n_{s.one_liner}_"
                )
                # Fundamental brief
                if s.fundamental_brief:
                    st.caption(s.fundamental_brief)

                sc1, sc2 = st.columns(2)
                if s.why_invest:
                    sc1.markdown("**✓ Why invest**")
                    for r in s.why_invest:
                        sc1.markdown(f"- {r}")
                if s.why_avoid:
                    sc2.markdown("**⚠ Why avoid (or wait)**")
                    for r in s.why_avoid:
                        sc2.markdown(f"- {r}")

            # --- Signal scoreboard: one clear buy/sell tilt from all indicators
            sb = signal_scoreboard(df)
            if sb:
                sb_style = {
                    "STRONG BUY": ("🟢", "success"),
                    "BUY": ("🟢", "success"),
                    "NEUTRAL": ("⚪", "info"),
                    "SELL": ("🔴", "error"),
                    "STRONG SELL": ("🔴", "error"),
                }
                emoji, banner = sb_style.get(sb["tilt"], ("⚪", "info"))
                getattr(st, banner)(
                    f"{emoji} **Signal scoreboard: {sb['tilt']}**  ·  score "
                    f"{sb['score']:+d}/100  ·  {sb['n_bull']} bullish vs "
                    f"{sb['n_bear']} bearish signals"
                )
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    if sb["bull"]:
                        st.markdown("**🟢 Bullish signals**")
                        for r in sb["bull"]:
                            st.markdown(f"- {r}")
                with bcol2:
                    if sb["bear"]:
                        st.markdown("**🔴 Bearish signals**")
                        for r in sb["bear"]:
                            st.markdown(f"- {r}")
                if sb["neutral"]:
                    with st.expander("Neutral / watch"):
                        for r in sb["neutral"]:
                            st.markdown(f"- {r}")

            # --- Trend quality: real vs false uptrend (volume confirmation)
            tq = trend_quality(df)
            if tq:
                tq_style = {
                    "real": ("🟢", "success"),
                    "weak": ("🟡", "warning"),
                    "false": ("🔴", "error"),
                    "no_uptrend": ("⚪", "info"),
                }
                emoji, banner = tq_style.get(tq["verdict"], ("⚪", "info"))
                msg = f"{emoji} **{tq['label']}**"
                if tq["in_uptrend"]:
                    msg += f" · volume-confirmation {tq['score']}/100"
                getattr(st, banner)(msg)
                with st.expander("Why — price vs volume-flow breakdown",
                                 expanded=tq["verdict"] == "false"):
                    for r in tq["reasons"]:
                        st.markdown(f"- {r}")

            # --- Combined price + volume figure (shared x-axis, aligned bars)
            # Row 1 = candles + EMAs + signal markers, Row 2 = volume bars.
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.74, 0.26], vertical_spacing=0.03,
                subplot_titles=("", "Volume (green=up day, red=down day)"),
            )
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["open"], high=df["high"],
                low=df["low"], close=df["close"], name="Price",
            ), row=1, col=1)
            for col, color in [("ema20", "#1f77b4"), ("ema50", "#ff7f0e"),
                               ("ema200", "#888")]:
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines",
                                             name=col.upper(),
                                             line=dict(width=1, color=color)),
                                  row=1, col=1)

            # Bollinger Band envelope (shaded) on row 1
            if "Bollinger Bands" in overlays and "bb_upper" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], mode="lines",
                                         name="BB upper", line=dict(width=1, color="rgba(120,120,200,0.5)")),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], mode="lines",
                                         name="BB lower", fill="tonexty",
                                         fillcolor="rgba(120,120,200,0.08)",
                                         line=dict(width=1, color="rgba(120,120,200,0.5)")),
                              row=1, col=1)
            # Supertrend regime line — green segment when bullish, red when bearish
            if "Supertrend" in overlays and "supertrend" in df.columns:
                st_bull = df["supertrend"].where(df["supertrend_dir"] > 0)
                st_bear = df["supertrend"].where(df["supertrend_dir"] < 0)
                fig.add_trace(go.Scatter(x=df.index, y=st_bull, mode="lines",
                                         name="Supertrend ↑", connectgaps=False,
                                         line=dict(width=2, color="#2ca02c")), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=st_bear, mode="lines",
                                         name="Supertrend ↓", connectgaps=False,
                                         line=dict(width=2, color="#d62728")), row=1, col=1)

            # Volume bars on row 2 (up/down coloured) + 20-day average line
            up_day = (df["close"] >= df["close"].shift(1)).fillna(True)
            fig.add_trace(go.Bar(
                x=df.index, y=df["volume"],
                marker_color=np.where(up_day, "#2ca02c", "#d62728"),
                name="Volume", showlegend=False,
                hovertemplate="Vol %{y:,.0f}<extra></extra>",
            ), row=2, col=1)
            if "vol_avg20" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["vol_avg20"], mode="lines",
                    name="Vol 20d-avg",
                    line=dict(color="#222", width=1.3, dash="dash"),
                    hovertemplate="20d-avg %{y:,.0f}<extra></extra>",
                ), row=2, col=1)

            # --- Signal markers overlay (now carry the *reason* in the hover)
            if show_markers:
                events = chart_signals.events_for_ticker(ticker, lookback=lookback)
                # Filter to events within the displayed window
                first_date = df.index.min()
                events = [e for e in events if pd.Timestamp(e.date) >= first_date]

                def _hover(e, tag):
                    parts = [f"<b>{e.setup}</b> · {pd.Timestamp(e.date):%d %b %Y}"]
                    if e.notes:
                        parts.append(f"<i>{e.notes}</i>")
                    levels = []
                    if e.entry is not None:
                        levels.append(f"entry ₹{e.entry:,.1f}")
                    if e.stoploss is not None:
                        levels.append(f"SL ₹{e.stoploss:,.1f}")
                    if e.target is not None:
                        levels.append(f"tgt ₹{e.target:,.1f}")
                    if e.rr is not None:
                        levels.append(f"R:R {e.rr}")
                    if levels:
                        parts.append(" · ".join(levels))
                    parts.append(tag)
                    return "<br>".join(parts)

                groups = [
                    ("target", "triangle-up", "#2ca02c", 14,
                     lambda e: f"{e.r_multiple:+.1f}R 🎯 (would have hit target)",
                     "Profitable signal"),
                    ("stop", "triangle-down", "#d62728", 14,
                     lambda e: f"{e.r_multiple:+.1f}R ❌ (would have stopped out)",
                     "Stopped-out signal"),
                    ("open", "star", "#ffd700", 18,
                     lambda e: "🔔 active / unresolved",
                     "Active signal"),
                ]
                for outcome, symbol, color, size, tagfn, legend in groups:
                    evs = [e for e in events if e.outcome == outcome]
                    if not evs:
                        continue
                    fig.add_trace(go.Scatter(
                        x=[e.date for e in evs], y=[e.price for e in evs],
                        mode="markers",
                        marker=dict(symbol=symbol, color=color, size=size,
                                    line=dict(color="white", width=1)),
                        text=[_hover(e, tagfn(e)) for e in evs],
                        hoverinfo="text",
                        name=legend,
                    ), row=1, col=1)

            # Hide weekend/holiday gaps so volume bars sit flush under candles
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            fig.update_yaxes(title_text="Price ₹", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_layout(
                height=720,
                xaxis_rangeslider_visible=False,
                xaxis2_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=30, b=10),
                showlegend=True,
                # Vertical legend docked to the right so extra overlays (BB,
                # Supertrend, signal markers) don't pile up above the plot and
                # steal vertical room from the candle pane.
                legend=dict(orientation="v", x=1.01, y=1, xanchor="left"),
                bargap=0,
            )
            st.plotly_chart(fig, width="stretch")

            with st.expander("Indicator snapshot (last bar)"):
                last = df.iloc[-1]
                cols = ["close", "ema20", "ema50", "ema200", "rsi14",
                        "macd", "macd_signal", "atr14", "vol_avg20",
                        "mfi14", "obv", "buy_pressure_20",
                        "adx14", "di_plus", "di_minus", "stoch_k", "stoch_d",
                        "bb_pct", "bb_width", "supertrend", "supertrend_dir", "cci20"]
                st.write({c: round(float(last[c]), 2)
                          for c in cols if c in df.columns and pd.notna(last[c])})

            # --- Volume profile: where shares actually changed hands
            st.markdown("### Volume profile")
            st.caption(
                "Horizontal bars = shares traded at each price level over the "
                "lookback window. High-volume nodes act as natural support/"
                "resistance — that's where the smart money set its cost basis."
            )
            profile = volume_profile(df, bins=24, lookback=60)
            if not profile.empty:
                vp_read = volume_profile_read(df, profile)
                vp_fig = go.Figure()
                vp_fig.add_trace(go.Bar(
                    x=profile["volume"],
                    y=profile["price_mid"],
                    orientation="h",
                    marker_color="#888",
                    width=(profile["price_high"] - profile["price_low"]),
                    name="Volume",
                ))
                # Mark current price + Point of Control + Value Area band
                last_close = float(df["close"].iloc[-1])
                vp_fig.add_hline(y=last_close, line_dash="dash", line_color="red",
                                 annotation_text=f"Price ₹{last_close:.0f}",
                                 annotation_position="top right")
                if vp_read:
                    vp_fig.add_hline(y=vp_read["poc"], line_dash="dot",
                                     line_color="#1f77b4",
                                     annotation_text=f"POC ₹{vp_read['poc']:.0f}",
                                     annotation_position="bottom right")
                    vp_fig.add_hrect(y0=vp_read["value_area_low"],
                                     y1=vp_read["value_area_high"],
                                     fillcolor="#1f77b4", opacity=0.08,
                                     line_width=0,
                                     annotation_text="value area",
                                     annotation_position="top left")
                vp_fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Volume traded",
                    yaxis_title="Price ₹",
                    showlegend=False,
                )
                st.plotly_chart(vp_fig, width="stretch")
                if vp_read:
                    with st.expander("What the volume profile is telling you",
                                     expanded=True):
                        for r in vp_read["reasons"]:
                            st.markdown(f"- {r}")

            # --- Money-flow indicators (OBV + MFI)
            st.markdown("### Money flow (OBV + MFI)")
            st.caption(
                "OBV rising = volume on up-days outweighs down-days "
                "(accumulation). MFI > 80 = overbought, < 20 = oversold."
            )
            mfr = money_flow_read(df)
            mf_fig = go.Figure()
            if "obv" in df.columns:
                mf_fig.add_trace(go.Scatter(
                    x=df.index, y=df["obv"],
                    mode="lines", name="OBV",
                    line=dict(color="#1f77b4"),
                ))
            mf_fig.update_layout(
                height=200,
                margin=dict(l=10, r=10, t=20, b=10),
                title="On-Balance Volume",
            )
            st.plotly_chart(mf_fig, width="stretch")
            if mfr.get("obv"):
                st.markdown(f"**{mfr['obv']}**")
            if mfr.get("obv_detail"):
                st.caption(mfr["obv_detail"])

            if "mfi14" in df.columns:
                mfi_fig = go.Figure()
                mfi_fig.add_trace(go.Scatter(
                    x=df.index, y=df["mfi14"],
                    mode="lines", name="MFI(14)",
                    line=dict(color="#ff7f0e"),
                ))
                mfi_fig.add_hline(y=80, line_dash="dot", line_color="red",
                                  annotation_text="overbought")
                mfi_fig.add_hline(y=20, line_dash="dot", line_color="green",
                                  annotation_text="oversold")
                mfi_fig.add_hline(y=50, line_dash="dot", line_color="#aaa",
                                  annotation_text="neutral")
                mfi_fig.update_layout(
                    height=200, yaxis_range=[0, 100],
                    margin=dict(l=10, r=10, t=20, b=10),
                    title="Money Flow Index",
                )
                st.plotly_chart(mfi_fig, width="stretch")
                if mfr.get("mfi"):
                    st.markdown(f"**{mfr['mfi']}**")
                if mfr.get("mfi_detail"):
                    st.caption(mfr["mfi_detail"])

            # --- Trend strength (ADX/DI) + momentum timing (Stochastic)
            if "adx14" in df.columns:
                st.markdown("### Trend strength — ADX + DI")
                st.caption(
                    "ADX > 25 = a real trend worth trading; below 20 = chop "
                    "(breakouts fail). DI+ above DI− = bulls in control."
                )
                adx_fig = go.Figure()
                adx_fig.add_trace(go.Scatter(x=df.index, y=df["adx14"], mode="lines",
                                             name="ADX", line=dict(color="#111", width=2)))
                adx_fig.add_trace(go.Scatter(x=df.index, y=df["di_plus"], mode="lines",
                                             name="DI+", line=dict(color="#2ca02c", width=1)))
                adx_fig.add_trace(go.Scatter(x=df.index, y=df["di_minus"], mode="lines",
                                             name="DI−", line=dict(color="#d62728", width=1)))
                adx_fig.add_hline(y=25, line_dash="dot", line_color="#888",
                                  annotation_text="trend threshold (25)")
                adx_fig.update_layout(height=220, margin=dict(l=10, r=10, t=20, b=10),
                                      legend=dict(orientation="h", y=1.1))
                st.plotly_chart(adx_fig, width="stretch")

            if "stoch_k" in df.columns:
                st.markdown("### Momentum timing — Stochastic")
                st.caption(
                    "%K crossing above %D below 20 = oversold buy timing; above "
                    "80 = overbought. Best used with the trend (ADX/EMA) as a filter."
                )
                stoch_fig = go.Figure()
                stoch_fig.add_trace(go.Scatter(x=df.index, y=df["stoch_k"], mode="lines",
                                               name="%K", line=dict(color="#1f77b4")))
                stoch_fig.add_trace(go.Scatter(x=df.index, y=df["stoch_d"], mode="lines",
                                               name="%D", line=dict(color="#ff7f0e")))
                stoch_fig.add_hline(y=80, line_dash="dot", line_color="red")
                stoch_fig.add_hline(y=20, line_dash="dot", line_color="green")
                stoch_fig.update_layout(height=200, yaxis_range=[0, 100],
                                        margin=dict(l=10, r=10, t=20, b=10),
                                        legend=dict(orientation="h", y=1.1))
                st.plotly_chart(stoch_fig, width="stretch")

            # --- Methodology: the norms behind every suggestion
            with st.expander("📖 How SwingDesk decides — the norms behind this"):
                st.markdown(glossary.VERDICT_METHODOLOGY)
                st.markdown(glossary.NORMS)
                st.markdown(glossary.SETUPS)

# --- News tab -------------------------------------------------------------------
if _page == "News":
    col1, col2 = st.columns([1, 4])
    with col1:
        sc_tickers = set(get_smallcap_watchlist())
        scope = st.radio("Scope", ["All", "Small caps only"], horizontal=True)
        # Filter dropdown spans the whole universe (watchlist + small caps + holdings)
        universe_news = combined_universe(include_smallcaps=True)
        filter_ticker = st.selectbox("Filter by ticker", ["(all)"] + universe_news)
        limit = st.slider("Items", 20, 500, 100)
    news_df = load_news(
        limit=limit,
        ticker=None if filter_ticker == "(all)" else filter_ticker,
    )
    # "Small caps only" keeps rows that tag at least one small-cap ticker.
    if scope == "Small caps only" and not news_df.empty:
        def _has_smallcap(tk):
            return bool(sc_tickers & set((tk or "").split(",")))
        news_df = news_df[news_df["tickers"].apply(_has_smallcap)]
        if news_df.empty:
            st.info("No small-cap-tagged headlines yet. Click **Fetch news** in the "
                    "sidebar (now includes the small-cap universe) and re-check.")
    if news_df.empty:
        st.info("No news yet. Click **Fetch news** in the sidebar.")
    else:
        sentiment_color = {"bullish": "#1f8a3a", "bearish": "#c0392b", "neutral": "#888"}
        for _, row in news_df.iterrows():
            tickers = row.get("tickers") or ""
            tag = f" — _{tickers}_" if tickers else ""
            sentiment = row.get("sentiment")
            impact = row.get("impact")
            badge = ""
            if sentiment:
                color = sentiment_color.get(sentiment, "#888")
                badge = (f" <span style='background:{color};color:white;"
                         f"padding:2px 6px;border-radius:3px;font-size:0.75em;"
                         f"margin-left:6px'>{sentiment}/{impact or '?'}</span>")
            st.markdown(
                f"**[{row['title']}]({row['link']})**{badge}  \n"
                f"<span style='color:#888;font-size:0.85em'>{row['source']} · "
                f"{row.get('published') or ''}{tag}</span>",
                unsafe_allow_html=True,
            )
            if row.get("rationale"):
                st.caption(f"_{row['rationale']}_")
            elif row.get("summary"):
                st.caption(row["summary"])
            st.divider()

# --- Sector rotation tab --------------------------------------------------------
if _page == "🔀 Sectors":
    from datetime import date as _sec_date

    st.subheader("🔀 Sector & micro-sector rotation")
    st.caption(
        "Which corners of the market are being bought vs sold, the strongest "
        "micro-sectors (industries) inside them, and the leading stocks — plus "
        "what to buy ahead of upcoming bullish calendar events. Relative strength "
        "from local price + fundamentals data."
    )

    SEC_BADGE = {
        sector_rotation.BULLISH: "🟢 Bullish",
        sector_rotation.BEARISH: "🔴 Bearish",
        sector_rotation.NEUTRAL: "⚪ Neutral",
    }

    @st.cache_data(ttl=900, show_spinner="Scanning sectors…")
    def _sector_snapshot():
        return sector_rotation.build_snapshot()

    rc1, rc2 = st.columns([1, 4])
    with rc1:
        if st.button("🔄 Recompute", width="stretch", key="sec_recompute"):
            _sector_snapshot.clear()
    snap = _sector_snapshot()

    if snap.empty:
        st.info(
            "No sector data yet. Run **Refresh fundamentals** (Fundamentals tab) and "
            "fetch prices so stocks have a sector + history."
        )
    else:
        with rc2:
            st.caption(
                f"{len(snap)} stocks · {snap['sector'].nunique()} sectors · "
                f"{snap['industry'].nunique()} micro-sectors. "
                "Strength = breadth (above 50/200-EMA) + median 1m/3m return."
            )

        secs = sector_rotation.rank_groups(snap, by="sector")
        sd = secs.copy()
        sd["bias"] = sd["bias"].map(SEC_BADGE)
        sd = sd.rename(columns={
            "breadth_200": "% > 200EMA", "breadth_50": "% > 50EMA",
            "med_ret_1m": "ret 1m %", "med_ret_3m": "ret 3m %",
        })
        st.markdown("#### Sectors — bullish ➜ bearish")
        st.dataframe(
            sd[["rank", "sector", "bias", "strength", "% > 200EMA", "% > 50EMA",
                "ret 1m %", "ret 3m %", "n"]],
            width="stretch", hide_index=True,
        )

        # --- Drill into a sector's micro-sectors + leading stocks ---------------
        st.divider()
        st.markdown("#### Drill into a sector")
        chosen_sec = st.selectbox("Sector", secs["sector"].tolist(), key="sec_drill")
        sub = snap[snap["sector"] == chosen_sec]

        micro = sector_rotation.rank_groups(sub, by="industry")
        if not micro.empty:
            md = micro.copy()
            md["bias"] = md["bias"].map(SEC_BADGE)
            md = md.rename(columns={"breadth_200": "% > 200EMA", "med_ret_3m": "ret 3m %",
                                    "med_ret_1m": "ret 1m %"})
            st.markdown(f"**Micro-sectors in {chosen_sec}**")
            st.dataframe(
                md[["rank", "industry", "bias", "strength", "% > 200EMA",
                    "ret 1m %", "ret 3m %", "n"]],
                width="stretch", hide_index=True,
            )
            ind_options = ["(whole sector)"] + micro["industry"].tolist()
            chosen_ind = st.selectbox("Focus a micro-sector", ind_options, key="sec_ind")
        else:
            st.caption("Not enough names for a micro-sector breakdown in this sector.")
            chosen_ind = "(whole sector)"

        picks = sector_rotation.top_stocks(
            snap, sector=chosen_sec,
            industry=None if chosen_ind == "(whole sector)" else chosen_ind, n=8,
        )
        label = chosen_sec if chosen_ind == "(whole sector)" else f"{chosen_sec} → {chosen_ind}"
        st.markdown(f"**Leading stocks — {label}**")
        if picks.empty:
            st.caption("No stocks to show.")
        else:
            pd_disp = picks.copy()
            pd_disp["above_200"] = pd_disp["above_200"].map({True: "✓", False: "✗"})
            st.dataframe(
                pd_disp.rename(columns={"above_200": "uptrend", "stock_strength": "strength",
                                        "quality_score": "quality"}),
                width="stretch", hide_index=True,
            )
            st.caption("💡 Open any of these in the **Chart** tab to time an entry.")

        # --- Calendar-driven ideas ---------------------------------------------
        st.divider()
        st.markdown("#### 📅 Calendar-driven ideas — what to buy into upcoming bullish events")
        today = _sec_date.today()
        bull_events = market_calendar.upcoming_bullish(today.year, today, within_days=120)
        if not bull_events:
            st.caption(
                "No bullish calendar events in the next 120 days. Check the **📅 Calendar** "
                "tab for the full year."
            )
        for e in bull_events:
            days = (e.start - today).days
            when = "today" if days <= 0 else f"in {days}d"
            sec_txt = ", ".join(e.sectors) if e.sectors else "broad market (large-cap leaders)"
            star = " ⚠️" if e.approximate else ""
            with st.expander(f"🟢 {e.start:%d %b} ({when}) · {e.name}{star} → {sec_txt}"):
                st.caption(e.note)
                ep = sector_rotation.event_picks(snap, e.sectors, n=6)
                if ep.empty:
                    st.caption("No currently-strong stocks found in those sectors.")
                else:
                    ed = ep.copy()
                    ed["above_200"] = ed["above_200"].map({True: "✓", False: "✗"})
                    st.dataframe(
                        ed[["ticker", "short_name", "sector", "industry", "last",
                            "ret_3m", "above_200", "stock_strength"]].rename(
                            columns={"above_200": "uptrend", "stock_strength": "strength"}),
                        width="stretch", hide_index=True,
                    )
        st.caption(
            "Picks = currently-strong stocks (uptrend + momentum) in the sectors that event "
            "historically favours. ⚠️ = floating event date — confirm it. Not advice; size and "
            "confirm each name on its chart."
        )


# --- Manipulation / unusual-activity tab ----------------------------------------
if _page == "🚨 Manipulation":
    st.subheader("🚨 Unusual-activity scanner")
    st.caption(
        "Screens each stock for an operator footprint — money churning out of "
        "proportion to size, volume eating the float, returns far outside normal, "
        "and thin liquidity that's cheap to push. A high score means *look closer*, "
        "not *guilty*. Computed from price + market-cap/float data already on disk."
    )

    with st.expander("ℹ️ How the scorecard works (and what's not wired yet)"):
        st.markdown(glossary.MANIPULATION)

    mcol1, mcol2, mcol3 = st.columns([1.4, 1, 1])
    with mcol1:
        scope = st.radio(
            "Universe", ["Watchlist", "My holdings", "Watchlist + holdings"],
            horizontal=True, key="manip_scope",
        )
    with mcol2:
        min_risk = st.slider("Min risk score to show", 0, 100, 0, key="manip_min")
    with mcol3:
        st.caption("NSE delivery % + bulk/block deals")
        if st.button("⬇️ Refresh NSE data", width="stretch", key="manip_nse"):
            from swingdesk.ingest import nse as nse_ingest
            univ = sorted(set(get_watchlist()) | set(holdings_tickers()))
            with st.spinner(f"Pulling NSE delivery + deals for {len(univ)} tickers…"):
                res = nse_ingest.ingest(univ, days=20)
            st.success(f"delivery rows: {res['delivery']} · deal rows: {res['deals']}")

    if scope == "Watchlist":
        manip_tickers = get_watchlist()
    elif scope == "My holdings":
        manip_tickers = holdings_tickers()
    else:
        manip_tickers = sorted(set(get_watchlist()) | set(holdings_tickers()))

    if not manip_tickers:
        st.info("No tickers in this universe. Add some via the sidebar watchlist.")
    else:
        # Fundamentals map (market cap / float) — only used for the warning below.
        fund_df = load_fundamentals()
        fund_by_ticker = {r["ticker"]: r for r in fund_df.to_dict("records")} if not fund_df.empty else {}

        cards = _cached_manip_cards(tuple(manip_tickers))

        scored = [c for c in cards if c["risk_score"] is not None]
        scored.sort(key=lambda c: c["risk_score"], reverse=True)
        shown = [c for c in scored if c["risk_score"] >= min_risk]

        if not fund_by_ticker:
            st.warning(
                "No fundamentals on disk — market-cap & float checks are skipped. "
                "Run **Refresh fundamentals** on the Fundamentals tab for the full read."
            )

        if not shown:
            st.info("Nothing above the risk threshold. Lower the slider or refresh data.")
        else:
            tier_emoji = {"High": "🔴", "Elevated": "🟠", "Low": "🟢"}
            # Ranked overview table.
            table = pd.DataFrame([
                {
                    "ticker": c["ticker"],
                    "risk": c["risk_score"],
                    "tier": f"{tier_emoji.get(c['tier'], '')} {c['tier']}",
                    "turnover/mcap": (c["components"].get("turnover_mcap") or {}).get("score"),
                    "vol/float": (c["components"].get("volume_float") or {}).get("score"),
                    "abn.return": (c["components"].get("abnormal_return") or {}).get("score"),
                    "illiquidity": (c["components"].get("amihud") or {}).get("score"),
                    "delivery": (c["components"].get("delivery") or {}).get("score"),
                    "deals": (c["components"].get("deals") or {}).get("score"),
                }
                for c in shown
            ])
            st.markdown(f"**{len(shown)} stocks · ranked by risk score**")
            st.dataframe(table, width="stretch", hide_index=True)

            st.divider()
            st.markdown("#### Per-stock detail")
            for c in shown:
                header = f"{tier_emoji.get(c['tier'], '')} **{c['ticker']}** — risk {c['risk_score']} ({c['tier']})"
                with st.expander(header, expanded=(c["tier"] == "High")):
                    st.caption(c["verdict"])
                    if c["notes"]:
                        for n in c["notes"]:
                            st.markdown(f"- {n}")
                    else:
                        st.caption("No individual flags.")

                    # Component scores as small metrics.
                    labels = {
                        "turnover_mcap": "Turnover vs mcap",
                        "volume_float": "Volume vs float",
                        "abnormal_return": "Abnormal return",
                        "amihud": "Illiquidity",
                        "delivery": "Delivery %",
                        "deals": "Bulk/block",
                    }
                    metric_cols = st.columns(len(labels))
                    for col, (key, label) in zip(metric_cols, labels.items()):
                        comp = c["components"].get(key)
                        col.metric(label, comp["score"] if comp else "—")

                    if c["data_gaps"]:
                        st.caption("Data gaps: " + " · ".join(c["data_gaps"]))


# --- Calendar tab ---------------------------------------------------------------
if _page == "📅 Calendar":
    from datetime import date as _date, timedelta

    st.subheader("📅 Market seasonality & event calendar")
    st.caption(
        "Dates that historically tilt the market bullish or bearish — so you can "
        "position *ahead* of them. Bias is a tendency, not a guarantee."
    )

    BIAS_BADGE = {
        market_calendar.BULLISH: "🟢 Bullish",
        market_calendar.BEARISH: "🔴 Bearish",
        market_calendar.VOLATILE: "🟠 Volatile",
        market_calendar.NEUTRAL: "⚪ Neutral",
    }

    today = _date.today()
    ccol1, ccol2, ccol3 = st.columns([1, 1.4, 1.4])
    with ccol1:
        year = st.number_input("Year", min_value=2020, max_value=2035,
                               value=today.year, step=1, key="cal_year")
        year = int(year)
    with ccol2:
        bias_filter = st.multiselect(
            "Bias", list(BIAS_BADGE), default=list(BIAS_BADGE), key="cal_bias",
            format_func=lambda b: BIAS_BADGE[b],
        )
    with ccol3:
        cats = [market_calendar.CAT_POLICY, market_calendar.CAT_EARNINGS,
                market_calendar.CAT_EXPIRY, market_calendar.CAT_SEASON,
                market_calendar.CAT_FESTIVE, market_calendar.CAT_GLOBAL,
                market_calendar.CAT_HOLIDAY]
        cat_filter = st.multiselect("Category", cats, default=cats, key="cal_cat")

    if not market_calendar.has_curated(year):
        st.warning(
            f"Floating dates (Diwali/Muhurat, RBI MPC, US FOMC, lunar holidays) aren't "
            f"filled in for {year} — only the computable events are shown. Add them in "
            f"`market_calendar.CURATED[{year}]`."
        )

    events = [
        e for e in market_calendar.build_calendar(year)
        if e.bias in bias_filter and e.category in cat_filter
    ]

    def _fmt_window(e):
        if e.is_range:
            return f"{e.start:%d %b} – {e.end:%d %b}"
        return f"{e.start:%a, %d %b}"

    # Upcoming highlights (only meaningful when viewing the current year).
    if year == today.year:
        up = [e for e in events if today <= e.start <= today + timedelta(days=60)]
        if up:
            st.markdown("#### ⏭️ Next 60 days")
            for e in up[:12]:
                days = (e.start - today).days
                when = "today" if days == 0 else f"in {days}d"
                star = " ⚠️" if e.approximate else ""
                st.markdown(
                    f"**{_fmt_window(e)}** ({when}) — {BIAS_BADGE[e.bias]} · "
                    f"**{e.name}**{star}  \n<span style='color:gray'>{e.note}</span>",
                    unsafe_allow_html=True,
                )
            st.divider()

    # Full year, grouped by month.
    st.markdown(f"#### Full year — {year}  ·  {len(events)} events")
    rows = []
    for e in events:
        rows.append({
            "When": _fmt_window(e),
            "Month": f"{e.start.month:02d} · {e.start:%b}",
            "Bias": BIAS_BADGE[e.bias],
            "Event": e.name + (" ⚠️" if e.approximate else ""),
            "Category": e.category,
            "What to do": e.note,
        })
    cal_df = pd.DataFrame(rows)
    if cal_df.empty:
        st.info("No events match the current filters.")
    else:
        st.dataframe(
            cal_df[["When", "Bias", "Event", "Category", "What to do"]],
            width="stretch", hide_index=True,
        )
        st.caption(
            "⚠️ = floating date (lunar festival / announced meeting) — confirm against the "
            "official NSE / RBI / Fed calendar. Legend: 🟢 bullish tendency · 🔴 bearish "
            "tendency · 🟠 two-way volatility · ⚪ informational (e.g. market closed)."
        )


# --- Backtest tab ---------------------------------------------------------------
if _page == "Backtest":
    st.subheader("Walk-forward backtest")
    st.caption("Validates each setup against historical bars. R-multiples — a target hit is +R, a stop is −1R.")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        bt_scope = st.radio("Scope", ["Watchlist", "Single ticker"], horizontal=False)
    with col2:
        bt_max_hold = st.number_input("Max hold (bars)", 5, 60, 20)
    with col3:
        bt_ticker = (st.selectbox("Ticker", get_watchlist())
                     if bt_scope == "Single ticker" else None)
        bt_uptrend = st.checkbox(
            "Long only above 200-EMA (trend filter)", value=True,
            help="Evidence-backed: skipping entries below the 200-day EMA is the "
                 "biggest single expectancy improver (profit factor ~0.83 → ~0.95). "
                 "Exit tweaks barely move it; trailing stops hurt.")

    if st.button("Run backtest", type="primary", width="stretch"):
        from datetime import datetime
        tickers = [bt_ticker] if bt_ticker else get_watchlist()
        with st.spinner(f"Backtesting {len(tickers)} ticker(s)..."):
            trades_df = bt_engine.backtest_universe(
                tickers, max_hold=bt_max_hold, require_uptrend=bt_uptrend)
        if trades_df.empty:
            st.warning("No trades produced — insufficient price history or no setups fired.")
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_backtest_trades(run_id, trades_df)
            _cached_setup_bt_stats.clear()    # signals annotation uses latest run
            st.success(f"run_id={run_id} · {len(trades_df)} trades")

    runs = list_backtest_runs()
    if runs.empty:
        st.info("No backtest runs yet. Click **Run backtest** above.")
    else:
        run_choice = st.selectbox(
            "Show results from",
            runs["run_id"].tolist(),
            format_func=lambda r: f"{r}  ·  {int(runs.loc[runs['run_id']==r,'n_trades'].iloc[0])} trades",
        )
        trades_df = load_backtest_trades(run_choice)

        cost_pct = st.number_input(
            "Round-trip cost % (brokerage + STT + charges + slippage)",
            0.0, 2.0, float(BACKTEST_COST_PCT), 0.05,
            help="Every stat below is reported NET of this cost. ~0.30% is a "
                 "reasonable NSE delivery-swing default (STT alone is ~0.2% "
                 "round trip). Tighter stops cost more in R terms.")

        # --- Summary metrics (NET of cost)
        summary = bt_metrics.summarize(trades_df, cost_pct=cost_pct)
        st.markdown(f"**Strategy summary — NET of {cost_pct:.2f}% round-trip cost** "
                    f"(one row per setup, plus ALL)")
        show_cols = ["setup", "n_trades", "win_rate", "avg_r", "gross_avg_r",
                     "avg_cost_r", "total_r", "expectancy", "profit_factor",
                     "max_drawdown_r", "max_consec_losses", "avg_bars_held"]
        st.dataframe(
            summary[show_cols], width="stretch", hide_index=True,
            column_config={
                "avg_r": st.column_config.NumberColumn(
                    "avg R (net)", format="%.3f", help="Mean R per trade after costs"),
                "gross_avg_r": st.column_config.NumberColumn(
                    "avg R (gross)", format="%.3f", help="Before costs"),
                "avg_cost_r": st.column_config.NumberColumn(
                    "cost (R)", format="%.3f", help="Avg cost per trade in R units"),
                "expectancy": st.column_config.NumberColumn("expectancy (net)", format="%.3f"),
            },
        )

        # --- Edge gating (NET)
        st.markdown(f"**Edge gate** — NET of costs (passes ⇒ tradeable; fails ⇒ "
                    f"keep paper-trading)")
        any_pass = False
        for _, row in summary.iterrows():
            if row["setup"] == "ALL":
                continue
            ok, fails = bt_metrics.gate(row.to_dict())
            if ok:
                any_pass = True
                st.success(f"✅ {row['setup']}  ·  net expectancy={row['expectancy']}R  "
                           f"·  pf={row['profit_factor']}  (gross avg {row['gross_avg_r']}R)")
            else:
                st.error(f"❌ {row['setup']}  ·  net expectancy={row['expectancy']}R  ·  {', '.join(fails)}")
        if not any_pass:
            st.warning("⚠️ No setup passes the edge gate net of costs — these "
                       "signals are screened ideas, not a mechanical system. "
                       "Don't auto-trade them.")

        # --- Equity curve in R (gross vs net)
        if not trades_df.empty:
            curve = trades_df.sort_values("entry_date").copy()
            gross, net, _ = bt_metrics.net_returns(curve, cost_pct)
            x = pd.to_datetime(curve["entry_date"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=gross.cumsum(), mode="lines",
                                     name="Gross", line=dict(color="#bbb", dash="dash")))
            fig.add_trace(go.Scatter(x=x, y=net.cumsum(), mode="lines",
                                     name=f"Net ({cost_pct:.2f}% cost)",
                                     line=dict(color="#1f77b4", width=2)))
            fig.add_hline(y=0, line_dash="dot", line_color="#888")
            fig.update_layout(height=350, title="Equity curve (cumulative R) — gross vs net",
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")

        # --- Trade log
        with st.expander(f"Trade log ({len(trades_df)} trades)"):
            st.dataframe(trades_df.drop(columns=["id", "run_id"], errors="ignore"),
                         width="stretch", hide_index=True)


# --- My Holdings tab ------------------------------------------------------------
if _page == "My Holdings":
    import tempfile
    from swingdesk.storage import load_holdings

    st.subheader("Analyze your Groww holdings")
    st.caption(
        "Upload your Groww portfolio export (CSV or Excel). I'll evaluate each "
        "holding on fundamentals, technicals, news sentiment, and current P&L — "
        "and recommend BUY MORE / HOLD / REDUCE / SELL with a specific reason."
    )

    uploaded = st.file_uploader("Holdings CSV/XLSX", type=["csv", "xlsx", "xls"])
    col_a, col_b = st.columns([3, 1])
    with col_a:
        col_map = st.text_input(
            "Column overrides (optional)",
            placeholder="symbol=Stock,avg_price=Avg Buy Price",
            help="If auto-detection fails, map column names like: symbol=YourCol,qty=YourCol",
        )
    with col_b:
        st.write("")
        if uploaded and st.button("Import + Analyze", type="primary", width="stretch"):
            overrides = {}
            if col_map:
                for kv in col_map.split(","):
                    k, _, v = kv.partition("=")
                    if k and v:
                        overrides[k.strip()] = v.strip()
            suffix = "." + uploaded.name.rsplit(".", 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp.flush()
                try:
                    n = holdings_mod.import_csv(tmp.name, overrides=overrides)
                    st.success(f"Imported {n} holdings")
                except ValueError as e:
                    st.error(str(e))

    df_h = load_holdings()
    if df_h.empty:
        st.info("Upload your Groww holdings CSV above to get started.")
    else:
        # --- Data-health check: flag holdings with no price data (bad symbols
        # from the broker export, or simply not fetched yet).
        held = list(df_h["ticker"])
        no_price = [t for t in held if load_prices(t, days=30).empty]
        if no_price:
            st.warning(
                f"⚠ **{len(no_price)} holding(s) have no price data** — their "
                "charts will be blank and the action plan can't size them: "
                f"{', '.join(no_price)}.\n\n"
                "Usually the broker export wrote a long company name instead of "
                "the NSE symbol. Click below to auto-correct known symbols and "
                "pull their prices."
            )
            hc1, hc2 = st.columns([1, 3])
            with hc1:
                if st.button("🔧 Fix tickers & refetch", type="primary",
                             width="stretch"):
                    with st.spinner("Correcting symbols + downloading prices..."):
                        remap = holdings_mod.remap_existing_tickers()
                        fixed_universe = holdings_tickers()
                        res = prices.ingest(fixed_universe, period="2y")
                        fundamentals_ingest.ingest(fixed_universe, workers=5)
                        _clear_data_caches()
                    if remap["changed"]:
                        st.success(f"Corrected {len(remap['changed'])} symbols: " +
                                   ", ".join(f"{o}→{n}" for o, n in remap["changed"][:10]) +
                                   ("…" if len(remap["changed"]) > 10 else ""))
                    ok = sum(1 for v in res.values() if v > 0)
                    st.success(f"Fetched prices for {ok}/{len(res)} tickers.")
                    if remap["untradable"]:
                        st.info("No equity series available (gold bonds / some "
                                "ETFs) — remove or chart manually: " +
                                ", ".join(remap["untradable"]))
                    st.rerun()
            with hc2:
                st.caption(
                    "Known instruments without a yfinance equity series "
                    "(Sovereign Gold Bonds, some silver/gold ETFs) can't be "
                    "charted and will be skipped."
                )

        ai_on = st.toggle("Include AI thesis (Claude analyzes each holding)",
                          value=False,
                          help="Uses Anthropic API. ~₹0.01 per holding analyzed.")
        _hkey = tuple(sorted(
            (str(r.get("ticker")), float(r.get("qty") or 0), float(r.get("avg_price") or 0))
            for r in df_h.to_dict("records")
        ))
        results = _cached_portfolio_analysis(_hkey, ai_on, df_h)
        summary = holdings_mod.portfolio_summary(results)

        # --- Macro pulse: gives context to the analyses below
        pulse = _cached_market_pulse()
        if pulse:
            st.markdown("### Market pulse")
            mc1, mc2, mc3, mc4 = st.columns(4)
            for col, name in zip([mc1, mc2, mc3, mc4],
                                  ["NIFTY 50", "USD/INR", "Brent Crude", "NASDAQ"]):
                if name in pulse:
                    p = pulse[name]
                    col.metric(name, f"{p['close']:,.2f}",
                               f"{p['chg_1d']:+.2f}% (1d) · {p['chg_1w']:+.2f}% (1w)")
        else:
            st.caption("💡 Run `cli macro` to enable macro context (NIFTY, USD/INR, Crude, etc.)")

        # --- Top-level metrics
        total_invested = df_h["invested"].sum() if "invested" in df_h.columns else 0
        total_value = df_h["current_value"].sum() if "current_value" in df_h.columns else 0
        total_pnl = total_value - total_invested if total_value else 0
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Holdings", summary["n_holdings"])
        c2.metric("Invested", f"₹{total_invested:,.0f}")
        c3.metric("Current value", f"₹{total_value:,.0f}", f"₹{total_pnl:+,.0f}")
        c4.metric("P&L %", f"{total_pnl_pct:+.1f}%")

        # --- Headlines
        st.markdown("### Recommendations")
        h1, h2, h3, h4 = st.columns(4)
        h1.success(f"**BUY MORE** ({len(summary['buy_more'])})\n\n" +
                   (", ".join(summary["buy_more"]) or "—"))
        h2.info(f"**HOLD** ({len(summary['hold'])})\n\n" +
                (", ".join(summary["hold"]) or "—"))
        h3.warning(f"**REDUCE** ({len(summary['reduce'])})\n\n" +
                   (", ".join(summary["reduce"]) or "—"))
        h4.error(f"**SELL** ({len(summary['sell'])})\n\n" +
                 (", ".join(summary["sell"]) or "—"))

        # --- Risk flags
        if summary["concentrated_positions"]:
            st.warning(f"⚠ Concentrated positions (>25% of portfolio): "
                       f"{', '.join(summary['concentrated_positions'])}")
        if summary["concentrated_sectors"]:
            st.warning(f"⚠ Concentrated sectors (>40% of portfolio): "
                       f"{', '.join(summary['concentrated_sectors'])}")

        # --- Action plan in NUMBERS: exactly how many shares / ₹ to trade
        st.markdown("### 📋 Action plan — what to do, in numbers")
        st.caption(
            "Concrete trade for each holding. SELL/REDUCE numbers are a fraction "
            "of your current quantity; BUY MORE is risk-sized to ~"
            f"{int(RISK_PER_TRADE_PCT)}% of the book against the hard stop and "
            "capped at 25% position weight."
        )
        plans = {a.ticker: allocate_mod.action_plan(a, portfolio_value=total_value)
                 for a in results}
        plan_rows = []
        for a in results:
            p = plans[a.ticker]
            plan_rows.append({
                "Action": p.action,
                "Ticker": a.ticker,
                "Hold qty": int(a.qty),
                "Trade qty": f"{p.shares:,}" if p.shares else "—",
                "Trade ₹": f"₹{p.rupees:,.0f}" if p.shares else "—",
                "What to do": p.note,
            })
        plan_df = pd.DataFrame(plan_rows)
        sell_total = sum(plans[a.ticker].rupees for a in results
                         if plans[a.ticker].action in ("SELL", "REDUCE"))
        buy_total = sum(plans[a.ticker].rupees for a in results
                        if plans[a.ticker].action == "BUY MORE")
        pc1, pc2 = st.columns(2)
        pc1.metric("Cash freed up if you act on SELL/REDUCE", f"₹{sell_total:,.0f}")
        pc2.metric("Cash needed for BUY MORE top-ups", f"₹{buy_total:,.0f}")
        st.dataframe(plan_df, width="stretch", hide_index=True)

        # --- Per-holding detail (the full lens breakdown behind each call)
        st.markdown("### Per-holding analysis")
        rows = []
        for a in results:
            rows.append({
                "Action": a.recommendation,
                "Ticker": a.ticker,
                "P&L %": f"{a.pnl_pct:+.1f}%" if a.pnl_pct is not None else "—",
                "Weight": f"{a.portfolio_weight*100:.1f}%" if a.portfolio_weight else "—",
                "Quality": a.quality_score,
                "Technical": a.technical_state,
                "RSI": f"{a.rsi:.0f}" if a.rsi else "—",
                "MFI": f"{a.mfi:.0f}" if a.mfi else "—",
                "BuyPress": f"{a.buy_pressure_20d:.2f}" if a.buy_pressure_20d else "—",
                "News +/-": f"+{a.sentiment_bullish}/-{a.sentiment_bearish}",
                "Hard SL": f"₹{a.initial_stop}" if a.initial_stop else "—",
                "Trail": f"₹{a.trailing_stop}" if a.trailing_stop else "—",
                "Trim @": f"₹{a.book_partial_at}" if a.book_partial_at else "—",
                "Target": f"₹{a.full_target}" if a.full_target else "—",
                "R:R": a.risk_reward,
                "Reasons": " · ".join(a.reasons[:2]),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # --- Early-exit warnings: surface BEFORE the stop hits.
        # Use getattr() so this never crashes if a stale HoldingAnalysis
        # (from before Phase 9) lingers in session state.
        early_exit_holdings = [
            a for a in results
            if getattr(a, "early_exit_action", None) in ("EXIT", "TRIM_50", "TRIM_25")
        ]
        if early_exit_holdings:
            st.markdown("### 🚨 Early-exit warnings")
            st.caption(
                "Stocks showing warning signs **before** the stop level is reached. "
                "Acting on these usually preserves gains better than waiting for the "
                "hard stop to trigger."
            )
            for a in early_exit_holdings:
                action = getattr(a, "early_exit_action", None) or "?"
                action_emoji = {"EXIT": "🔴", "TRIM_50": "🟠", "TRIM_25": "🟡"}.get(
                    action, "⚪")
                with st.expander(f"{action_emoji} {a.ticker}  ·  {action}"):
                    trend_label = getattr(a, "trend_label", None)
                    if trend_label:
                        st.markdown(f"**Trend:** {trend_label}")
                    warnings = getattr(a, "early_warnings", None) or []
                    if warnings:
                        st.markdown("**Warnings:**")
                        for w in warnings:
                            st.markdown(f"- {w}")

        # --- AI theses (one expander per holding)
        if any(getattr(a, "ai_narrative", None) for a in results):
            st.markdown("### AI thesis per holding")
            st.caption("Claude analyzed each position end-to-end. Conviction "
                       "above 70 means multiple lenses align.")
            for a in results:
                narrative = getattr(a, "ai_narrative", None)
                if not narrative:
                    continue
                ai_color = {"BUY_MORE": "🟢", "HOLD": "⚪", "TRIM": "🟡", "EXIT": "🔴"}
                action = getattr(a, "ai_action", "?") or "?"
                conviction = getattr(a, "ai_conviction", "?") or "?"
                emoji = ai_color.get(action, "⚪")
                with st.expander(f"{emoji} {a.ticker}  ·  {action}  ·  "
                                 f"conviction {conviction}/100"):
                    st.markdown(f"**Thesis.** {narrative}")
                    catalyst = getattr(a, "ai_catalyst", None)
                    if catalyst:
                        st.markdown(f"**Catalyst to watch.** {catalyst}")
                    risks = getattr(a, "ai_risks", None) or []
                    if risks:
                        st.markdown("**Top risks:**")
                        for r in risks:
                            st.markdown(f"- {r}")

        # --- Sector breakdown
        with st.expander("Sector concentration"):
            sect_df = pd.DataFrame([
                {"Sector": s, "Weight": f"{w*100:.1f}%"}
                for s, w in sorted(summary["sector_concentration"].items(),
                                   key=lambda x: -x[1])
            ])
            if not sect_df.empty:
                st.dataframe(sect_df, width="stretch", hide_index=True)


# --- Optimize tab ---------------------------------------------------------------
if _page == "Optimize":
    st.subheader("Parameter optimizer")
    st.caption(
        "Grid-searches SL/target/hold combinations per setup. Results ranked by "
        "expectancy. **Warning:** the top row is often overfit — use as a "
        "hypothesis, not a verdict."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        opt_setup = st.selectbox("Setup",
            ["breakout_20d", "pullback_ema20", "volume_thrust", "ema_20_50_cross"])
    with col2:
        st.write("Grid: SL × {1.0, 1.5, 2.0, 2.5} · Target × {2, 3, 4, 5} · Hold × {10, 20, 30}")

    if st.button("Run optimization", type="primary", width="stretch"):
        with st.spinner(f"Searching 48 combinations × {len(get_watchlist())} tickers..."):
            results = opt.optimize(get_watchlist(), opt_setup)
        if results.empty:
            st.warning("No results.")
        else:
            st.session_state["last_optimize"] = (opt_setup, results)

    if "last_optimize" in st.session_state:
        setup_name, results = st.session_state["last_optimize"]
        st.markdown(f"**Top 10 — {setup_name}**")
        st.dataframe(results.head(10), width="stretch", hide_index=True)
        st.markdown("**Bottom 5 (for contrast)**")
        st.dataframe(results.tail(5), width="stretch", hide_index=True)


# --- Portfolio tab --------------------------------------------------------------
if _page == "Portfolio":
    st.subheader("Paper + real portfolio")

    book = st.radio("Book", ["Paper", "Real", "Both"], horizontal=True, key="book")
    is_paper = None if book == "Both" else (book == "Paper")

    # Auto mark-to-market on first visit per session — so positions display
    # real LTP and live unrealized P&L without the user clicking anything.
    if "portfolio_auto_mtm_done" not in st.session_state:
        with st.spinner("Pulling latest prices for open positions..."):
            auto_res = portfolio.mark_to_market(auto_fetch=True, refresh=False)
            _clear_data_caches()
        if auto_res["closed"]:
            st.warning(f"⚠ Auto-closed {auto_res['closed']} position(s) — "
                       "SL or target was hit since last visit.")
        if auto_res.get("skipped"):
            st.info("Couldn't get prices for: " + ", ".join(auto_res["skipped"]))
        st.session_state["portfolio_auto_mtm_done"] = True

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if st.button("🔄 Refresh prices NOW (live)", type="primary", width="stretch",
                     help="Re-pull the latest close from yfinance for every open ticker"):
            with st.spinner("Re-downloading fresh prices..."):
                res = portfolio.mark_to_market(refresh=True)
                _clear_data_caches()
            st.success(
                f"refreshed: {res['fetched']} tickers fetched · "
                f"{res['marked']} marked · {res['closed']} closed"
            )
            if res.get("skipped"):
                st.warning("No price data: " + ", ".join(res["skipped"]))
            st.rerun()
    with col2:
        if st.button("Mark-to-market (use cached prices)", width="stretch"):
            res = portfolio.mark_to_market(refresh=False)
            st.success(
                f"checked={res['checked']} · closed={res['closed']} · "
                f"marked={res['marked']}"
            )
    with col3:
        st.metric("Capital", f"₹{ACCOUNT_CAPITAL:,.0f}")

    # --- Stats cards
    s = pj.stats(is_paper=is_paper)
    if s.n_trades == 0:
        st.info("No positions yet. Open one manually in the form below, or run the daily pipeline.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total trades", s.n_trades, f"{s.n_open} open")
        c2.metric("Win rate", f"{s.win_rate * 100:.1f}%", f"{s.wins}W / {s.losses}L")
        c3.metric("Total P&L", f"₹{s.total_pnl:,.0f}", f"unrealized ₹{s.unrealized_pnl:,.0f}")
        c4.metric("Avg R", f"{s.avg_r}", f"total {s.total_r}R")

        c1, c2, c3 = st.columns(3)
        c1.metric("Best trade", f"₹{s.best_trade:,.0f}")
        c2.metric("Worst trade", f"₹{s.worst_trade:,.0f}")
        c3.metric("Open risk", f"₹{s.open_risk:,.0f}")

        # --- Equity curve
        curve = pj.equity_curve(is_paper=is_paper)
        if not curve.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve["date"], y=curve["equity"], mode="lines+markers",
                                     name="Equity", line=dict(color="#1f77b4")))
            fig.add_hline(y=ACCOUNT_CAPITAL, line_dash="dot", line_color="#888",
                          annotation_text="starting capital")
            fig.update_layout(height=350, title="Equity curve",
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")

        # --- By setup
        by_s = pj.by_setup(is_paper=is_paper)
        if not by_s.empty:
            st.markdown("**Performance by setup**")
            st.dataframe(by_s, width="stretch", hide_index=True)

    # --- Open positions list (with computed live P&L columns)
    open_df = load_positions(status="open", is_paper=is_paper)
    if not open_df.empty:
        st.markdown(f"**Open positions ({len(open_df)})**")
        # Compute LTP-based P&L. If last_price is null (never marked), fall back
        # to entry so the row still renders meaningfully.
        view = open_df.copy()
        view["LTP"] = view["last_price"].fillna(view["entry_price"])
        view["Unrealized ₹"] = ((view["LTP"] - view["entry_price"]) * view["qty"]).round(2)
        view["Unrealized %"] = ((view["LTP"] - view["entry_price"]) / view["entry_price"] * 100).round(2)
        # Distance to stop / target as % — actionable at a glance
        view["To Stop %"] = ((view["stoploss"] - view["LTP"]) / view["LTP"] * 100).round(2)
        view["To Target %"] = ((view["target"] - view["LTP"]) / view["LTP"] * 100).round(2)
        show_cols = ["id", "ticker", "qty", "entry_price", "LTP",
                     "Unrealized ₹", "Unrealized %",
                     "stoploss", "To Stop %", "target", "To Target %",
                     "setup", "entry_date", "last_marked"]
        show_cols = [c for c in show_cols if c in view.columns]
        st.dataframe(view[show_cols], width="stretch", hide_index=True)

        # Aggregate at the top — total unrealized
        total_unrealized = float(view["Unrealized ₹"].sum())
        u1, u2, u3 = st.columns(3)
        u1.metric("Total unrealized P&L", f"₹{total_unrealized:,.0f}")
        invested = float((view["entry_price"] * view["qty"]).sum())
        if invested > 0:
            u2.metric("Invested in open positions", f"₹{invested:,.0f}",
                      f"{total_unrealized/invested*100:+.2f}%")
        u3.metric("Open positions", len(view))

        # Close form
        with st.expander("Close a position"):
            pos_id = st.number_input("Position ID", min_value=1, step=1,
                                     value=int(open_df.iloc[0]["id"]))
            exit_price = st.number_input("Exit price", min_value=0.01, value=100.0, step=0.5)
            reason = st.selectbox("Reason", ["manual", "target", "stoploss", "time_stop"])
            if st.button("Close"):
                res = portfolio.close_position(int(pos_id), exit_price=float(exit_price),
                                               exit_reason=reason)
                st.success(res["status"])
                st.rerun()

    # --- Open new position form
    with st.expander("Open a new position manually"):
        with st.form("open_pos"):
            c1, c2, c3 = st.columns(3)
            f_ticker = c1.selectbox("Ticker", get_watchlist())
            f_entry = c1.number_input("Entry", min_value=0.01, value=100.0, step=0.5)
            f_sl = c2.number_input("Stoploss", min_value=0.01, value=95.0, step=0.5)
            f_tgt = c2.number_input("Target", min_value=0.01, value=110.0, step=0.5)
            f_setup = c3.text_input("Setup", value="manual")
            f_qty = c3.number_input("Qty (0 = auto-size)", min_value=0, value=0, step=1)
            f_paper = st.checkbox("Paper trade", value=True)
            f_notes = st.text_input("Notes", value="")
            submit = st.form_submit_button("Open position")
            if submit:
                res = portfolio.open_position(
                    ticker=f_ticker, entry=f_entry, stoploss=f_sl, target=f_tgt,
                    setup=f_setup, is_paper=f_paper, notes=f_notes or None,
                    qty=int(f_qty) if f_qty > 0 else None,
                )
                if res["status"] == "opened":
                    st.success(f"opened #{res['position']['id']}")
                    st.rerun()
                else:
                    st.error(res.get("reason", res["status"]))

    # --- Closed history
    closed_df = load_positions(status="closed", is_paper=is_paper)
    if not closed_df.empty:
        with st.expander(f"Closed positions ({len(closed_df)})"):
            cols = ["id", "ticker", "setup", "qty", "entry_price", "exit_price",
                    "pnl", "pnl_pct", "r_multiple", "exit_reason",
                    "entry_date", "exit_date"]
            cols = [c for c in cols if c in closed_df.columns]
            st.dataframe(closed_df[cols], width="stretch", hide_index=True)


# --- Reconcile tab --------------------------------------------------------------
if _page == "Reconcile":
    st.subheader("Paper-trade vs Backtest drift")
    st.caption(
        "Once you have 5+ closed paper trades per setup, this surfaces whether "
        "live performance matches your backtest expectations. Drift in either "
        "direction is information."
    )
    min_n = st.slider("Min paper trades to draw a verdict", 3, 30, 5)
    all_runs = st.checkbox("Aggregate all backtest runs (default: latest only)")
    if st.button("Reconcile", width="stretch"):
        df = reconcile_mod.reconcile(latest_run=not all_runs, min_paper_trades=min_n)
        if df.empty:
            st.info("No backtest runs yet — run one in the Backtest tab first.")
        else:
            st.dataframe(df, width="stretch", hide_index=True)
            flagged = df[df["verdict"] == "underperforming"]
            if not flagged.empty:
                st.error(f"⚠ {len(flagged)} setup(s) underperforming backtest — investigate")
            else:
                ok = df[df["verdict"] == "aligned"]
                if not ok.empty:
                    st.success(f"✅ {len(ok)} setup(s) aligned with backtest")


# --- Fundamentals tab -----------------------------------------------------------
if _page == "Fundamentals":
    st.subheader("Fundamental quality ranking")
    st.caption(
        "Each ticker scored 0–100 on a weighted blend of ROE, growth, margins, "
        "valuation, debt, and size. Signals on high-quality stocks get a "
        "composite-score boost."
    )

    from swingdesk.ingest import fundamentals as fund_ingest

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Refresh fundamentals (yfinance)", width="stretch"):
            with st.spinner(f"Pulling fundamentals for {len(get_watchlist())} tickers..."):
                n = fund_ingest.ingest(get_watchlist())
                _clear_data_caches()
            st.success(f"saved {n} tickers")
    with col2:
        min_q = st.slider("Min quality score", 0, 100, 60)

    df_fund = load_fundamentals(min_quality=min_q if min_q > 0 else None)
    if df_fund.empty:
        st.info("No fundamentals yet. Click **Refresh fundamentals** above.")
    else:
        st.markdown(f"**{len(df_fund)} tickers · quality ≥ {min_q}**")
        cols = ["ticker", "short_name", "sector", "quality_score",
                "return_on_equity", "trailing_pe", "debt_to_equity",
                "profit_margin", "earnings_growth", "revenue_growth",
                "market_cap"]
        cols = [c for c in cols if c in df_fund.columns]
        # Format ratios as percentages for readability
        display = df_fund[cols].copy()
        for c in ("return_on_equity", "profit_margin", "earnings_growth", "revenue_growth"):
            if c in display.columns:
                display[c] = display[c].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—")
        if "market_cap" in display.columns:
            display["market_cap"] = display["market_cap"].apply(
                lambda v: f"₹{v/1e7:,.0f} cr" if pd.notna(v) else "—"
            )
        st.dataframe(display, width="stretch", hide_index=True)

        if st.button("Replace watchlist with these names", width="stretch"):
            set_watchlist(df_fund["ticker"].tolist())
            st.success(f"watchlist set to {len(df_fund)} quality names")
            st.rerun()


# --- Raw data tab ---------------------------------------------------------------
if _page == "Raw data":
    st.subheader("Watchlist price coverage")
    st.dataframe(_cached_price_coverage(tuple(get_watchlist())),
                 width="stretch", hide_index=True)


# --- 🌐 Global (US -> India spillover) ------------------------------------------
if _page == "🌐 Global":
    st.subheader("US → India spillover & market regime")
    st.caption(
        "US markets close after India, so last night's Wall Street move is a "
        "real leading signal for today's session. Use this for gap/risk "
        "positioning — single-day direction R² is low by nature."
    )

    reg = _cached_regime()
    if reg is None:
        st.info("No macro data yet. Run **Fetch prices** (sidebar) — it also "
                "pulls S&P/NASDAQ/VIX — then revisit.")
    else:
        banner = {"risk-on": "success", "neutral": "info", "risk-off": "error"}[reg.label]
        getattr(st, banner)(f"**Market regime: {reg.label.upper()}**  ·  score {reg.score:+d}/100")
        with st.expander("Why", expanded=reg.label == "risk-off"):
            for r in reg.reasons:
                st.markdown(f"- {r}")

        out = _cached_outlook()
        if out is not None:
            st.markdown("### Next-session NIFTY outlook")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{out.driver} overnight", f"{out.last_us_move_pct:+.2f}%")
            c2.metric("Expected NIFTY move", f"{out.expected_pct:+.2f}%",
                      help="Point estimate from the lead-lag fit — noisy, not a promise.")
            c3.metric("Likely range (±1σ)", f"{out.low_pct:+.2f}% … {out.high_pct:+.2f}%")
            st.caption(out.narrative)

        betas = _cached_betas()
        if not betas.empty:
            st.markdown("### How strongly each US index leads NIFTY")
            st.dataframe(betas, width="stretch", hide_index=True)

    st.divider()
    st.markdown("### Per-stock drivers")
    st.caption("What actually moves a given holding — index beta, US-tech "
               "linkage, rupee and crude sensitivity.")
    _glob_universe = combined_universe(include_smallcaps=True)
    if _glob_universe:
        sens_pick = st.selectbox("Stock", _glob_universe, key="sens_pick")
        sens = _cached_sensitivities(sens_pick)
        if sens.empty:
            st.info("Not enough price history for this name yet.")
        else:
            st.dataframe(sens, width="stretch", hide_index=True)


# --- 📐 Range (expected move / Monte Carlo) -------------------------------------
if _page == "📐 Range":
    st.subheader("Expected price range")
    st.caption(
        "You can't reliably predict direction, but you can model the "
        "distribution of where price will sit. This is the 'expected move' an "
        "options desk quotes — use it to set realistic targets and stops."
    )
    _rng_universe = combined_universe(include_smallcaps=True)
    if not _rng_universe:
        st.info("Fetch prices first.")
    else:
        rc1, rc2 = st.columns([1, 3])
        with rc1:
            rng_pick = st.selectbox("Stock", _rng_universe, key="rng_pick")
            horizon = st.slider("Horizon (sessions)", 3, 40, 10, key="rng_h")
            mc_method = st.radio("Monte Carlo", ["block", "bootstrap", "gbm"],
                                 help="block = resample contiguous return blocks "
                                      "(keeps volatility clustering — best for "
                                      "swings); bootstrap = iid resample; gbm = "
                                      "textbook normal model.")

        er = _cached_expected_range(rng_pick, horizon)
        if er is None:
            st.warning("Not enough price history (need ~30+ bars).")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Spot", f"₹{er.spot:,.0f}")
            m2.metric(f"±1σ ({horizon}d)", f"±{er.expected_move_pct:.1f}%")
            m3.metric("68% range", f"₹{er.low_68:,.0f}–{er.high_68:,.0f}")
            m4.metric("Annualized vol", f"{er.annualized_vol_pct:.0f}%")
            st.caption(er.narrative)

            mc = _cached_monte_carlo(rng_pick, horizon, mc_method)
            if mc is not None and not mc.fan.empty:
                fan = mc.fan
                fig = go.Figure()
                # Shaded 5–95 and 25–75 bands + median line.
                fig.add_trace(go.Scatter(x=fan.index, y=fan["p95"], mode="lines",
                                         line=dict(width=0), showlegend=False,
                                         hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=fan.index, y=fan["p5"], mode="lines",
                                         fill="tonexty", fillcolor="rgba(31,119,180,0.10)",
                                         line=dict(width=0), name="5–95%"))
                fig.add_trace(go.Scatter(x=fan.index, y=fan["p75"], mode="lines",
                                         line=dict(width=0), showlegend=False,
                                         hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=fan.index, y=fan["p25"], mode="lines",
                                         fill="tonexty", fillcolor="rgba(31,119,180,0.22)",
                                         line=dict(width=0), name="25–75%"))
                fig.add_trace(go.Scatter(x=fan.index, y=fan["p50"], mode="lines",
                                         line=dict(color="#1f77b4", width=2), name="median"))
                fig.add_hline(y=er.spot, line_dash="dash", line_color="#888")
                fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                                  title=f"{rng_pick} — {mc.n_sims:,} simulated paths "
                                        f"({mc_method})",
                                  xaxis_title="sessions ahead", yaxis_title="price ₹")
                st.plotly_chart(fig, width="stretch")
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Median target", f"₹{mc.terminal_p50:,.0f}")
                pc2.metric("Downside (5%)", f"₹{mc.terminal_p5:,.0f}")
                pc3.metric("P(up over horizon)", f"{mc.prob_up*100:.0f}%")

            cone = _cached_vol_cone(rng_pick)
            if not cone.empty:
                with st.expander("Volatility cone — is vol cheap or rich now?"):
                    st.caption("Current realized vol vs this stock's own history. "
                               "Low percentile often precedes a volatility expansion.")
                    st.dataframe(cone, width="stretch", hide_index=True)

            # --- Holding planner: "hold ~N days for ~X%, with ~P% odds"
            st.markdown("### 🎯 Holding planner — hold how long for what gain?")
            st.caption(
                "For each target gain, the odds of the price **touching** it within "
                "~40 sessions (≈2 months) and the typical days-to-target — simulated "
                "from this stock's own recent behaviour. Probabilities, not promises: "
                "use them to set realistic targets and size with the 🛡 Risk tab."
            )
            plan = _cached_holding_plan(rng_pick, 40, mc_method)
            if plan is None:
                st.info("Need ~40+ bars of history for the planner.")
            else:
                pt = plan["table"].copy()
                pt["target_pct"] = pt["target_pct"].map(lambda v: f"+{v:.0f}%")
                st.dataframe(
                    pt, width="stretch", hide_index=True,
                    column_config={
                        "target_pct": st.column_config.TextColumn("Target gain"),
                        "target_price": st.column_config.NumberColumn("Target ₹", format="%.1f"),
                        "prob_hit": st.column_config.ProgressColumn(
                            "Chance of touching (≤40d)", format="%.0f%%",
                            min_value=0.0, max_value=1.0),
                        "median_days_to_hit": st.column_config.NumberColumn(
                            "Typical days to hit",
                            help="Median sessions to reach it, among paths that do"),
                        "vol_implied_days": st.column_config.NumberColumn(
                            "Vol-implied days",
                            help="Sessions at which this gain is a typical (1σ) move — "
                                 "a volatility sanity check"),
                    },
                )
                best = plan["table"].iloc[0]
                if best["median_days_to_hit"]:
                    st.success(
                        f"e.g. a **+{best['target_pct']:.0f}%** move (₹{best['target_price']:,.0f}) "
                        f"has ~**{best['prob_hit']*100:.0f}%** odds of being touched within "
                        f"40 sessions, typically in ~**{best['median_days_to_hit']:.0f} days**."
                    )

            # --- Reward vs risk: P(hit +target before −stop) — the real read
            st.markdown("### ⚖️ Reward vs risk — target before stop?")
            st.caption(
                "Upside alone is half the picture. This simulates which barrier is "
                "touched **first** — your target or your stop — and the resulting "
                "expectancy. A high target % means nothing if the stop gets hit on "
                "the way. (Close-to-close sim, so it slightly under-counts intraday "
                "wicks; pair with the 🛡 Risk tab for sizing.)"
            )
            tcol, scol = st.columns(2)
            tgt_in = tcol.slider("Target gain %", 1.0, 25.0, 8.0, 0.5, key="tvs_tgt")
            stp_in = scol.slider("Stop-loss %", 1.0, 25.0, 4.0, 0.5, key="tvs_stp")
            tvs = _cached_target_vs_stop(rng_pick, tgt_in, stp_in, 40, mc_method)
            if tvs is None:
                st.info("Need ~40+ bars of history.")
            else:
                style = ("success" if "favourable" in tvs.verdict
                         else "info" if tvs.verdict == "marginal" else "error")
                getattr(st, style)(f"**{tvs.verdict.upper()}** · R:R {tvs.rr:.1f}")
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("🎯 Target first", f"{tvs.p_target_first*100:.0f}%",
                          help="P(touch +target before −stop, within 40 sessions)")
                v2.metric("🛑 Stopped first", f"{tvs.p_stop_first*100:.0f}%")
                v3.metric("Win rate (resolved)",
                          f"{tvs.win_rate*100:.0f}%" if pd.notna(tvs.win_rate) else "—",
                          help="Target ÷ (target + stop), ignoring un-resolved paths")
                v4.metric("Expectancy", f"{tvs.expectancy_pct:+.2f}%",
                          f"{tvs.expectancy_r:+.2f}R")
                d1, d2, d3 = st.columns(3)
                d1.metric("Median days→target",
                          f"{tvs.median_days_to_target:.0f}" if pd.notna(tvs.median_days_to_target) else "—")
                d2.metric("Median days→stop",
                          f"{tvs.median_days_to_stop:.0f}" if pd.notna(tvs.median_days_to_stop) else "—")
                d3.metric("Neither in 40d", f"{tvs.p_neither*100:.0f}%")
                st.caption(tvs.narrative)


# --- 🛡 Risk (sizing + portfolio diagnostics) -----------------------------------
if _page == "🛡 Risk":
    st.subheader("Position sizing & portfolio risk")

    st.markdown("### Position-size calculator")
    st.caption("Never risk more than a fixed % of capital on one trade — the "
               "single highest-leverage habit for surviving losing streaks.")
    sc1, sc2, sc3, sc4 = st.columns(4)
    entry_in = sc1.number_input("Entry ₹", min_value=0.0, value=100.0, step=1.0)
    sl_in = sc2.number_input("Stop-loss ₹", min_value=0.0, value=95.0, step=1.0)
    cap_in = sc3.number_input("Capital ₹", min_value=0.0, value=float(ACCOUNT_CAPITAL),
                              step=10000.0)
    risk_in = sc4.number_input("Risk %", min_value=0.1, value=float(RISK_PER_TRADE_PCT),
                               step=0.25)
    ps = risk_mod.position_size(entry_in, sl_in, capital=cap_in, risk_pct=risk_in)
    if ps is None:
        st.warning("Stop-loss must be below entry (long setup).")
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Shares", f"{ps.shares:,}")
        p2.metric("Position value", f"₹{ps.position_value:,.0f}")
        p3.metric("Risk if stopped", f"₹{ps.risk_amount:,.0f}")
        p4.metric("% of capital", f"{ps.pct_of_capital:.0f}%")
        st.caption(ps.note)

    st.divider()
    st.markdown("### Portfolio risk report")
    rep = _cached_risk_report()
    if not rep["ok"]:
        st.info(rep["msg"] + " Import your Groww holdings to see this.")
    else:
        conc = rep["concentration"]
        if conc:
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Holdings", conc.n_holdings)
            cc2.metric("Top name", f"{conc.top_weight_pct:.0f}%",
                       help=conc.top_name)
            cc3.metric("Top-3 weight", f"{conc.top3_weight_pct:.0f}%")
            cc4.metric("Effective bets", f"{conc.effective_names:.1f}",
                       help="1/HHI — how many truly independent positions you hold")
        for s in rep["suggestions"]:
            st.markdown(f"- {s}")
        sectors = rep.get("sectors")
        if sectors is not None and not sectors.empty:
            with st.expander("Sector concentration"):
                st.dataframe(sectors, width="stretch", hide_index=True)
        pairs = rep["correlated_pairs"]
        if not pairs.empty:
            with st.expander("Highly-correlated holding pairs"):
                st.dataframe(pairs, width="stretch", hide_index=True)


# --- 🏆 Rank (cross-sectional factor model) -------------------------------------
if _page == "🏆 Rank":
    st.subheader("Factor ranking")
    st.caption(
        "Ranks your whole watchlist on momentum, low-volatility, quality, value "
        "and trend — the transparent, evidence-based version of 'which stock "
        "will go up'. Favour quintile 1; avoid quintile 5."
    )

    # --- What the factors are, how they're computed, and their weights.
    # Weights are pulled live from the model so this never drifts out of sync.
    _factor_docs = {
        "momentum": ("6-month price momentum",
                     "6-month return, **skipping the most recent month** "
                     "(≈ close₋₂₂ ÷ close₋₁₄₈ − 1). The skip avoids short-term "
                     "mean-reversion.",
                     "Higher = stronger winner. *Jegadeesh–Titman: winners keep winning.*"),
        "low_vol": ("Low volatility",
                    "Negative of 60-day realised daily volatility "
                    "(std of daily returns). Calmer stocks score higher.",
                    "Higher = calmer. *The low-volatility anomaly: steadier names "
                    "deliver better risk-adjusted returns.*"),
        "quality": ("Quality (fundamentals)",
                    "Mean of three **separately z-scored** pieces: ROE, profit "
                    "margin, and −debt/equity (leverage counts against you).",
                    "Higher = more profitable, less levered. Needs fundamentals "
                    "(run the Fundamentals tab)."),
        "value": ("Value (cheapness)",
                  "Negative trailing P/E (only when P/E > 0), z-scored. "
                  "Cheaper earnings multiple scores higher.",
                  "Higher = cheaper. Guards against over-paying for momentum."),
        "trend": ("Long-term trend",
                  "Price relative to its 200-day EMA (close ÷ EMA200 − 1).",
                  "Higher = further above the 200-DMA, i.e. a confirmed long-term "
                  "uptrend / participation filter."),
    }
    _w = factors_mod.DEFAULT_WEIGHTS
    with st.expander("📖 How the ranking works — the 5 factors", expanded=True):
        st.markdown(
            "Each factor is **z-scored across your universe** (winsorised at ±3σ), "
            "so every score says *“how this stock compares to its peers right now.”* "
            "The **composite** is the weighted blend below; if a factor is missing "
            "for a stock (e.g. no fundamentals), its weight is dropped and the rest "
            "re-normalised — the stock isn't penalised. Stocks are then ranked and "
            "split into **quintiles (1 = most attractive, 5 = least).**"
        )
        _rows = [{
            "Factor": _factor_docs[k][0],
            "Weight": f"{_w[k]*100:.0f}%",
            "How it's computed": _factor_docs[k][1],
            "Reading": _factor_docs[k][2],
        } for k in _w]
        st.dataframe(pd.DataFrame(_rows), width="stretch", hide_index=True)
        st.caption(f"Weights sum to {sum(_w.values())*100:.0f}%. "
                   "These are sensible defaults, not gospel — momentum + quality "
                   "carry the most weight by design.")

    wl = tuple(get_watchlist())
    if not wl:
        st.info("Add tickers to your watchlist first.")
    elif st.button("Rank watchlist", type="primary", width="stretch"):
        tbl = _cached_factor_table(wl)
        if tbl.empty:
            st.warning("Need ≥3 names with price history (and ideally "
                       "fundamentals — run the Fundamentals tab).")
        else:
            show = tbl[["rank", "ticker", "quintile", "composite",
                        "momentum_z", "low_vol_z", "quality_z", "value_z", "trend_z"]]
            st.dataframe(
                show, width="stretch", hide_index=True,
                column_config={
                    "quintile": st.column_config.NumberColumn(
                        "quintile", help="1 = most attractive, 5 = least"),
                    "composite": st.column_config.ProgressColumn(
                        "composite", help="Weighted blend of the 5 factor z-scores",
                        format="%.2f",
                        min_value=float(show["composite"].min()),
                        max_value=float(show["composite"].max()),
                    ),
                    "momentum_z": st.column_config.NumberColumn(
                        "Momentum", format="%.2f",
                        help="6-month return (skip last month), z-scored vs peers"),
                    "low_vol_z": st.column_config.NumberColumn(
                        "Low-vol", format="%.2f",
                        help="Inverse 60-day volatility, z-scored (higher = calmer)"),
                    "quality_z": st.column_config.NumberColumn(
                        "Quality", format="%.2f",
                        help="ROE + margin − leverage, z-scored (needs fundamentals)"),
                    "value_z": st.column_config.NumberColumn(
                        "Value", format="%.2f",
                        help="Cheapness on trailing P/E, z-scored"),
                    "trend_z": st.column_config.NumberColumn(
                        "Trend", format="%.2f",
                        help="Price vs 200-day EMA, z-scored"),
                },
            )
            top = tbl.head(5)["ticker"].tolist()
            st.success("Top 5: " + ", ".join(top))
            st.caption("z-scores are cross-sectional (vs the rest of your "
                       "universe). Composite = weighted blend; missing "
                       "fundamentals are skipped, not penalised.")


# --- 🤖 ML (probabilistic direction model) --------------------------------------
if _page == "🤖 ML":
    st.subheader("ML direction model — P(up over N sessions)")
    st.warning(
        "⚠️ Honesty first: daily/weekly equity direction is ~55% predictable at "
        "best. This outputs a calibrated **probability**, validated walk-forward "
        "(train on past → test on future). **Always check it beats the base "
        "rate** below before trusting it, and use it only to *tilt* setups that "
        "already pass your rules — never as a standalone signal."
    )
    ml_wl = tuple(get_watchlist())
    horizon = st.slider("Horizon (sessions)", 5, 30, 10, key="ml_h")

    if not ml_wl:
        st.info("Add tickers to your watchlist first.")
    else:
        eval_col, pred_col, imp_col = st.columns(3)
        with eval_col:
            run_eval = st.button("1 · Validate (walk-forward)", width="stretch",
                                 help="Trains on history, tests out-of-sample. "
                                      "Tells you whether the model has any edge.")
        with pred_col:
            run_pred = st.button("2 · Score stocks now", type="primary",
                                 width="stretch",
                                 help="Train on all history, output P(up) for the "
                                      "latest bar of each watchlist name.")
        with imp_col:
            run_imp = st.button("3 · What drives it", width="stretch",
                                help="Permutation importance — how much each "
                                     "feature actually moves out-of-sample accuracy.")

        if run_eval:
            res = _cached_ml_eval(ml_wl, horizon)
            if res is None:
                st.warning("Not enough labeled history. Fetch ~3y of prices for a "
                           "decent-sized watchlist, then retry.")
            else:
                verdict_style = ("success" if "real edge" in res.verdict
                                 else "info" if "marginal" in res.verdict else "error")
                getattr(st, verdict_style)(f"**Verdict: {res.verdict}**")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("OOS accuracy", f"{res.accuracy*100:.1f}%")
                e2.metric("Base rate", f"{res.base_rate*100:.1f}%",
                          help="Accuracy from always guessing the majority class.")
                e3.metric("Edge vs base", f"{res.edge_vs_baseline*100:+.1f} pp",
                          help="Accuracy minus base rate. >0 = real out-of-sample lift.")
                e4.metric("AUC / Brier", f"{res.auc:.2f} / {res.brier:.3f}",
                          help="AUC>0.5 ranks better than chance; lower Brier = "
                               "better-calibrated probabilities.")
                st.caption(f"{res.n_folds} walk-forward folds · "
                           f"{res.n_test_total:,} out-of-sample predictions.")
                with st.expander("Per-fold detail"):
                    st.dataframe(pd.DataFrame(res.folds), width="stretch",
                                 hide_index=True)

        if run_pred:
            preds = _cached_ml_predict(ml_wl, horizon)
            if preds.empty:
                st.warning("Not enough labeled history to train. Fetch more prices.")
            else:
                st.dataframe(
                    preds, width="stretch", hide_index=True,
                    column_config={
                        "prob_up": st.column_config.ProgressColumn(
                            f"P(up in {horizon}d)", format="%.2f",
                            min_value=0.0, max_value=1.0),
                    },
                )
                st.caption("Run **Validate** first — if the edge vs base is ≤0, "
                           "these probabilities have no demonstrated predictive "
                           "value and should be ignored.")

        if run_imp:
            imp = _cached_ml_importance(ml_wl, horizon)
            if imp.empty:
                st.warning("Not enough labeled history for an importance test.")
            else:
                st.caption("How much shuffling each feature hurts out-of-sample "
                           "accuracy. Near-zero = the model barely uses it. Negative "
                           "= pure noise. This is the honest measure of what matters.")
                st.dataframe(
                    imp, width="stretch", hide_index=True,
                    column_config={
                        "importance": st.column_config.ProgressColumn(
                            "importance", format="%.4f",
                            min_value=float(min(0, imp["importance"].min())),
                            max_value=float(imp["importance"].max())),
                    },
                )


# --- ⚡ Intraday (5/15-min monitoring: VWAP / ORB / RVOL) ------------------------
if _page == "⚡ Intraday":
    st.subheader("Intraday monitor — VWAP · Opening-Range Breakout · RVOL")
    st.info(
        "⚠️ Data is yfinance intraday — **~15 min delayed** and only useful "
        "during/after market hours. This is for monitoring & alerts on a 5–15 "
        "min cadence, **not scalping**. Pairs with the 🌐 Global gap model: a "
        "gap that holds above VWAP is the classic morning continuation."
    )
    ic1, ic2 = st.columns([1, 2])
    with ic1:
        iv = st.radio("Interval", ["5m", "15m"], horizontal=True, key="intra_iv")
    with ic2:
        st.write("")
        if st.button("⬇ Fetch intraday now", type="primary", width="stretch"):
            uni = combined_universe(include_smallcaps=True)
            with st.spinner(f"Pulling {iv} bars for {len(uni)} tickers…"):
                res = intraday_ingest.ingest(uni, interval=iv)
                _cached_intraday_scan.clear()
            ok = sum(1 for v in res.values() if v > 0)
            st.success(f"intraday {iv}: {ok}/{len(res)} tickers updated")

    intra_universe = combined_universe(include_smallcaps=True)
    scan_df = _cached_intraday_scan(tuple(intra_universe), iv)
    if scan_df.empty:
        st.warning("No intraday data yet. Click **Fetch intraday now** "
                   "(works best during/just after NSE hours, 9:15–15:30 IST).")
    else:
        active = scan_df[scan_df["bias"] != "neutral"]
        st.markdown(f"### Setups firing — {len(active)} active / {len(scan_df)} scanned")
        st.dataframe(
            scan_df, width="stretch", hide_index=True,
            column_config={
                "dist_vwap_pct": st.column_config.NumberColumn("vs VWAP %", format="%.2f"),
                "gap_pct": st.column_config.NumberColumn("gap %", format="%.2f"),
                "rvol": st.column_config.NumberColumn("RVOL", format="%.1f×"),
            },
        )

        st.divider()
        pick = st.selectbox("Inspect a stock", scan_df["ticker"].tolist(),
                            key="intra_pick")
        sig = intraday_mod.intraday_signals(pick, interval=iv)
        from swingdesk.storage import load_intraday as _load_intraday
        idf = _load_intraday(pick, interval=iv)
        if sig is None or idf.empty:
            st.info("Not enough intraday data for this name yet.")
        else:
            bias_style = {"long": "success", "short": "error", "neutral": "info"}[sig.bias]
            getattr(st, bias_style)(f"**{sig.setup}**  ·  {sig.narrative}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Last", f"₹{sig.last:,.1f}", f"{sig.dist_vwap_pct:+.2f}% vs VWAP")
            m2.metric("VWAP", f"₹{sig.vwap:,.1f}")
            m3.metric("Gap", f"{sig.gap_pct:+.2f}%")
            m4.metric("RVOL", f"{sig.rvol:.1f}×" if pd.notna(sig.rvol) else "—")

            # Latest-session candles + VWAP + opening-range band.
            day = idf.index.normalize().max()
            sess = idf[idf.index.normalize() == day]
            vw = intraday_mod.vwap(idf).loc[sess.index]
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=sess.index, open=sess["open"], high=sess["high"],
                low=sess["low"], close=sess["close"], name="Price"))
            fig.add_trace(go.Scatter(x=sess.index, y=vw, mode="lines",
                                     name="VWAP", line=dict(color="#1f77b4", width=2)))
            if pd.notna(sig.or_high):
                fig.add_hline(y=sig.or_high, line_dash="dot", line_color="#2ca02c",
                              annotation_text="OR high")
                fig.add_hline(y=sig.or_low, line_dash="dot", line_color="#d62728",
                              annotation_text="OR low")
            fig.update_layout(height=460, xaxis_rangeslider_visible=False,
                              margin=dict(l=10, r=10, t=30, b=10),
                              title=f"{pick} — {iv} (session {day:%d %b})")
            st.plotly_chart(fig, width="stretch")


def _mini_price_chart(ticker: str, lookback: int = 180):
    """Compact daily candles + EMAs + volume for a screened name. None if no data."""
    df = load_prices(ticker)
    if df is None or df.empty:
        return None
    df = add_indicators(df).tail(lookback)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.74, 0.26], vertical_spacing=0.03,
                        subplot_titles=("", "Volume (green=up day, red=down day)"))
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="Price"),
                  row=1, col=1)
    for col, color in [("ema20", "#1f77b4"), ("ema50", "#ff7f0e"), ("ema200", "#888")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines",
                                     name=col.upper(), line=dict(width=1, color=color)),
                          row=1, col=1)
    up_day = (df["close"] >= df["close"].shift(1)).fillna(True)
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", showlegend=False,
                         marker_color=np.where(up_day, "#2ca02c", "#d62728")),
                  row=2, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                      xaxis2_rangeslider_visible=False, bargap=0,
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="v", x=1.01, y=1, xanchor="left"),
                      title=f"{ticker} — daily (last {lookback}d)")
    return fig


# --- 🔎 Screener (factors + liquidity = a few tradeable good stocks) ------------
if _page == "🔎 Screener":
    st.subheader("Screener — good factors AND actually tradeable")
    st.caption(
        "Fuses the factor rank (momentum/quality/value/low-vol/trend) with a "
        "**liquidity** check — ₹ traded/day, **turnover vs market-cap**, and "
        "**volume vs float** + price-impact. A stock that looks great but is "
        "illiquid is flagged, not surfaced as a buy: you can't exit what you "
        "can't trade. Each row carries its reasons."
    )
    sc1, sc2 = st.columns([1, 1])
    with sc1:
        scope = st.radio("Universe", ["Watchlist + small caps", "Watchlist only"],
                         key="scr_scope")
    with sc2:
        min_liq = st.slider("Min liquidity score", 0, 100, 40, 5, key="scr_minliq",
                            help="40 = filter out illiquid/untradeable names")

    uni = (combined_universe(include_smallcaps=True) if scope.startswith("Watchlist +")
           else get_watchlist())
    if not uni:
        st.info("Add tickers / fetch prices first.")
    else:
        # Results are stashed in session_state so the chart picker / hover panel
        # below survive the reruns those widgets trigger (a button is True only on
        # its own click — re-rendering off it alone would wipe the results).
        if st.button(f"Screen {len(uni)} stocks", type="primary", width="stretch"):
            st.session_state["scr_results"] = _cached_screen(tuple(uni), float(min_liq))
        df = st.session_state.get("scr_results")

        if df is None:
            st.caption("Click **Screen** to run the factor + liquidity screen.")
        elif df.empty:
            st.warning("Nothing to screen — need ≥3 names with price history "
                       "(and fundamentals for market-cap/float).")
        else:
            # Which screened names does recent news back? Used to highlight rows,
            # fill the hover tooltips, and flag the chart.
            support = _screen_news_support(tuple(df["ticker"]))
            supported_set = {t for t, i in support.items() if i.get("supported")}

            def _news_cell(t: str) -> str:
                i = support.get(t, {})
                if not i.get("n_total"):
                    return ""
                if i["supported"]:
                    return f"🟢 {i['n_bull']}↑"
                if i["n_bear"] > i["n_bull"]:
                    return f"🔴 {i['n_bear']}↓"
                return f"⚪ {i['n_total']}"

            view = df.copy()
            view["news"] = view["ticker"].map(_news_cell)
            good = view[view["verdict"] == "✅ good & tradeable"]
            illq = view[view["verdict"] == "⚠ good but illiquid"]
            st.success(f"🎯 {len(good)} good & tradeable · ⚠ {len(illq)} good-but-illiquid "
                       f"(flagged) · {len(view)} screened · "
                       f"📰 {len(supported_set)} news-backed.")

            # --- News-backed chips: hover any chip to read its supporting headlines.
            def _esc(s: str) -> str:
                return (str(s).replace("&", "&amp;").replace('"', "&quot;")
                        .replace("<", "&lt;").replace(">", "&gt;"))

            chips = []
            for t in view["ticker"]:
                i = support.get(t, {})
                if not i.get("n_total"):
                    continue
                tip = "&#10;".join(_esc(h) for h in i["headlines"]) or "no headlines"
                if i["supported"]:
                    bg, border, label = "#1f8a3a", "2px solid #0d4d1e", f"🟢 {t} · {i['n_bull']}↑"
                elif i["n_bear"] > i["n_bull"]:
                    bg, border, label = "#c0392b", "1px solid #7d251c", f"🔴 {t} · {i['n_bear']}↓"
                else:
                    bg, border, label = "#777", "1px solid #555", f"⚪ {t} · {i['n_total']}"
                chips.append(
                    f'<span title="{tip}" style="display:inline-block;background:{bg};'
                    f'color:white;padding:3px 9px;border-radius:12px;margin:3px;'
                    f'font-size:0.85em;border:{border};cursor:help">{label}</span>'
                )
            if chips:
                st.markdown("##### 📰 News-backed names — hover a chip to read the headlines")
                st.markdown("<div>" + "".join(chips) + "</div>", unsafe_allow_html=True)

            colcfg = {
                "verdict": st.column_config.TextColumn("Verdict", width="medium"),
                "news": st.column_config.TextColumn("📰 News",
                            help="Fresh (≤14d) headlines tagging this name: 🟢 net-bullish "
                                 "(news backs the trade), 🔴 net-bearish, ⚪ neutral. Hover "
                                 "the chips above to read the actual headlines."),
                "factor_composite": st.column_config.NumberColumn("Factor", format="%.2f"),
                "quintile": st.column_config.NumberColumn("Q", help="Factor quintile (1=best)"),
                "liq_tier": st.column_config.TextColumn("Liquidity"),
                "liq_score": st.column_config.ProgressColumn("Liq score", format="%.0f",
                                                             min_value=0, max_value=100),
                "adv_cr": st.column_config.NumberColumn("₹ ADV (cr)", format="%.1f",
                            help="Avg daily traded value"),
                "turnover_pct": st.column_config.NumberColumn("Turn/mcap %", format="%.3f",
                            help="ADV ÷ market-cap per day"),
                "float_turnover_pct": st.column_config.NumberColumn("Vol/float %", format="%.3f",
                            help="ADV-volume ÷ free-float per day"),
                "amihud": st.column_config.NumberColumn("Impact", format="%.3f",
                            help="Amihud illiquidity — price move per ₹cr traded (lower=better)"),
                "why": st.column_config.TextColumn("Reasons", width="large"),
            }
            show = ["ticker", "verdict", "news", "factor_composite", "quintile", "liq_tier",
                    "liq_score", "adv_cr", "turnover_pct", "float_turnover_pct",
                    "amihud", "why"]

            def _highlight_news(row):
                """Tint rows green where recent news backs the (bullish) factor read."""
                hit = row["ticker"] in supported_set
                return ["background-color: rgba(31,138,58,0.20)" if hit else "" for _ in row]

            def _show_table(sub: pd.DataFrame, n: int | None = None):
                tbl = sub[show].head(n) if n else sub[show]
                st.dataframe(tbl.style.apply(_highlight_news, axis=1),
                             width="stretch", hide_index=True, column_config=colcfg)

            st.markdown("### 🎯 A few good stocks (factor-strong + liquid)")
            st.caption("Rows tinted green are also backed by recent bullish news.")
            if good.empty:
                st.info("No name is both factor-attractive and liquid right now. "
                        "That's a valid answer — don't force a trade.")
            else:
                _show_table(good, 15)

            if not illq.empty:
                st.markdown("### ⚠ Looks good but ILLIQUID — the trap")
                st.caption("Strong factors, but turnover-vs-mcap / volume-vs-float "
                           "are too thin to enter and exit safely. Size tiny or skip.")
                _show_table(illq, 15)

            with st.expander("Full screen (all names, including weak factors)"):
                _show_table(view)

            # --- Open a chart for any screened name -----------------------------
            st.divider()
            st.markdown("### 📈 Open a chart")
            # Order the picker so news-backed names sort to the top.
            ordered = sorted(view["ticker"].tolist(),
                             key=lambda t: (t not in supported_set))
            labels = {t: (f"🟢 {t}" if t in supported_set else t) for t in ordered}
            pick = st.selectbox("Chart a screened name", ["(select)"] + ordered,
                                format_func=lambda t: labels.get(t, t),
                                key="scr_chart_pick")
            if pick and pick != "(select)":
                i = support.get(pick, {})
                if i.get("supported"):
                    st.success(f"🟢 News-supported — {i['n_bull']} bullish vs "
                               f"{i['n_bear']} bearish headline(s) in the last 14 days.")
                elif i.get("n_bear", 0) > i.get("n_bull", 0):
                    st.warning(f"🔴 Recent news leans bearish — {i['n_bear']} bearish "
                               f"vs {i['n_bull']} bullish headline(s). Factor-strong but "
                               "the tape disagrees.")
                fig = _mini_price_chart(pick)
                if fig is None:
                    st.info("No price history for this name.")
                else:
                    st.plotly_chart(fig, width="stretch")
                heads = i.get("headlines") or []
                if heads:
                    with st.expander(f"📰 Recent headlines for {pick}",
                                     expanded=bool(i.get("supported"))):
                        for h in heads:
                            st.markdown(f"- {h}")

# --- 🛠 Execution (execution algos + TCA = how to get in/out, and how you did) --
if _page == "🛠 Execution":
    st.subheader("🛠 Execution — slice an order like a desk, then grade the fill")
    st.caption(
        "The 'algo' in algo trading: you've *picked* the stock elsewhere — this "
        "decides **how to get in and out** without moving the price against you. "
        "Trading all at once pays **market impact**; trading too slowly pays "
        "**timing risk** (price drifts while you wait). An execution algo slices "
        "the order to balance the two. Impact is the **square-root law** "
        "(η·σ·√participation), cross-checked against the **Amihud** impact from "
        "the liquidity engine. You trade by hand on Groww, so the output is an "
        "**order plan you place yourself** — all estimates, not fill guarantees."
    )

    uni = combined_universe(include_smallcaps=True)
    if not uni:
        st.info("Add tickers / fetch prices first.")
    else:
        e1, e2, e3 = st.columns([2, 1, 1])
        with e1:
            etk = st.selectbox("Ticker", uni, key="exec_tk")
        with e2:
            eside = st.radio("Side", ["buy", "sell"], key="exec_side", horizontal=True)
        with e3:
            ealgo = st.selectbox("Algo", ["vwap", "twap", "pov", "is"], key="exec_algo",
                                 format_func=str.upper,
                                 help="VWAP=track volume · TWAP=even slices · "
                                      "POV=% of volume · IS=front-load to cut timing risk")
        e4, e5, e6 = st.columns([1.4, 1, 1])
        with e4:
            by = st.radio("Order size by", ["₹ value", "shares"], key="exec_by",
                          horizontal=True)
            if by == "₹ value":
                eval_rs = st.number_input("Order value ₹", min_value=1000, value=200000,
                                          step=10000, key="exec_val")
                eqty, enot = 0, float(eval_rs)
            else:
                eqty = int(st.number_input("Shares", min_value=1, value=500, step=10,
                                           key="exec_qty"))
                enot = 0.0
        with e5:
            ebkt = st.selectbox("Slice size", [15, 30, 60], index=2, key="exec_bkt",
                                format_func=lambda m: f"{m} min")
        with e6:
            epov = st.slider("POV %", 5, 40, 10, 1, key="exec_pov",
                             help="Used by POV: trade this % of each slice's volume") / 100.0
            eurg = st.slider("IS urgency", 0.0, 1.0, 0.5, 0.1, key="exec_urg",
                             help="Used by IS: 0=even, 1=front-load hard")

        plan = execution_mod.execution_plan(
            etk, eside, qty=eqty or None, notional=enot or None, algo=ealgo,
            bucket_minutes=ebkt, participation=epov, risk_aversion=eurg)

        if plan is None:
            st.warning(f"Can't build a plan for {etk} — no price/spot. Fetch prices first.")
        else:
            for w in plan.warnings:
                st.warning(w)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Arrival ₹", f"{plan.arrival_price:,.2f}",
                      f"{plan.qty:,} sh · ₹{plan.notional:,.0f}")
            m2.metric("Est. cost", f"{plan.est_cost_bps:.0f} bps",
                      f"₹{plan.est_cost_rupees:,.0f}", delta_color="inverse")
            m3.metric("Impact / spread", f"{plan.impact_bps:.0f} / {plan.spread_bps:.0f} bps",
                      help="Square-root-law market impact + assumed half-spread")
            m4.metric("Timing risk", f"{plan.timing_risk_bps:.0f} bps",
                      f"~{plan.avg_participation_pct:.2f}% of vol",
                      help="1σ price drift over the execution horizon — a risk, not a cost")

            for s in plan.summary:
                st.markdown(f"- {s}")
            xcheck = (f"Amihud cross-check ≈ {plan.amihud_bps:.0f} bps"
                      if plan.amihud_bps == plan.amihud_bps else "Amihud cross-check: n/a")
            st.caption(f"{xcheck} · daily σ {plan.daily_vol_pct:.2f}% · "
                       f"volume curve: {'generic U-shape' if plan.used_fallback_curve else 'this name’s intraday history'}.")

            # --- schedule: your slices vs the market's volume shape
            sched = plan.schedule
            cfig = go.Figure()
            cfig.add_trace(go.Bar(x=sched["time"], y=sched["shares"], name="Your shares",
                                  marker_color="#1f77b4",
                                  hovertemplate="%{x}: %{y:,} sh<extra></extra>"))
            if sched["est_mkt_vol"].notna().any():
                cfig.add_trace(go.Scatter(
                    x=sched["time"], y=sched["est_mkt_vol"], name="Est. market volume",
                    mode="lines+markers", line=dict(color="#ff7f0e"), yaxis="y2",
                    hovertemplate="%{x}: ~%{y:,.0f} sh traded<extra></extra>"))
            cfig.update_layout(
                height=340, margin=dict(l=10, r=10, t=30, b=10),
                title=f"{ealgo.upper()} schedule — {plan.qty:,} sh in {len(sched)} slices",
                yaxis=dict(title="Your shares"),
                yaxis2=dict(title="Mkt volume", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", y=1.12), bargap=0.25)
            st.plotly_chart(cfig, width="stretch")

            st.dataframe(
                sched, width="stretch", hide_index=True,
                column_config={
                    "time": st.column_config.TextColumn("Slice (IST)"),
                    "shares": st.column_config.NumberColumn("Shares", format="%d"),
                    "pct_of_order": st.column_config.NumberColumn("% order", format="%.1f"),
                    "cum_shares": st.column_config.NumberColumn("Cum sh", format="%d"),
                    "cum_pct": st.column_config.NumberColumn("Cum %", format="%.0f"),
                    "est_mkt_vol": st.column_config.NumberColumn("Est mkt vol", format="%d",
                                    help="Expected shares the whole market trades in this slice"),
                    "participation_pct": st.column_config.NumberColumn("Your %", format="%.2f",
                                    help="Your shares as % of the slice's volume — keep this low"),
                })

            # --- copy-paste manual order plan
            plan_lines = [
                f"{r.time}  {plan.side.upper():4s} {int(r.shares):>6,} sh   "
                f"(LIMIT near prevailing mid; ref arrival ₹{plan.arrival_price:,.2f})"
                for r in sched.itertuples()]
            st.markdown("**📋 Manual order plan (place on Groww):**")
            st.code("\n".join(plan_lines), language="text")

            # --- algo comparison (pre-trade TCA)
            st.divider()
            st.markdown("### ⚖️ Which algo? — cost vs timing-risk trade-off")
            st.caption(
                "Same order, every algo. **Est cost** (spread+impact) is what you "
                "*expect* to pay; **timing risk** is drift exposure while you work it. "
                "Cheapest expected cost ≠ best if you're in a hurry — that's the trade-off "
                "VWAP/TWAP/POV/IS exist to manage."
            )
            cmp = _cached_algo_compare(etk, eside, int(eqty), float(enot),
                                       int(ebkt), float(epov), float(eurg))
            if not cmp.empty:
                st.dataframe(
                    cmp, width="stretch", hide_index=True,
                    column_config={
                        "algo": st.column_config.TextColumn("Algo"),
                        "slices": st.column_config.NumberColumn("Slices", format="%d"),
                        "horizon_min": st.column_config.NumberColumn("Horizon", format="%d min"),
                        "avg_participation_pct": st.column_config.NumberColumn("Avg % vol", format="%.2f"),
                        "est_cost_bps": st.column_config.NumberColumn("Est cost", format="%.1f bps"),
                        "impact_bps": st.column_config.NumberColumn("Impact", format="%.1f bps"),
                        "timing_risk_bps": st.column_config.NumberColumn("Timing risk", format="%.0f bps"),
                        "est_cost_rupees": st.column_config.NumberColumn("Est ₹", format="%.0f"),
                        "completes": st.column_config.CheckboxColumn("Done in day?"),
                        "note": st.column_config.TextColumn("Note", width="medium"),
                    })

            # --- post-trade TCA
            st.divider()
            st.markdown("### 📊 Post-trade TCA — grade an actual fill")
            st.caption(
                "After you trade, enter your fills to see how you did vs **arrival** "
                "(implementation shortfall — the honest benchmark), **VWAP**, and "
                "**TWAP**. Positive bps = a cost (paid up on a buy / sold low). Add a "
                "`time` (HH:MM) if you want the exact intraday benchmark."
            )
            with st.form("tca_form"):
                fills_default = pd.DataFrame(
                    [{"price": float(plan.arrival_price), "qty": int(plan.qty), "time": ""}])
                fills_edit = st.data_editor(
                    fills_default, num_rows="dynamic", hide_index=True, key="tca_fills",
                    column_config={
                        "price": st.column_config.NumberColumn("Fill ₹", format="%.2f"),
                        "qty": st.column_config.NumberColumn("Qty", format="%d"),
                        "time": st.column_config.TextColumn("Time (HH:MM, optional)"),
                    })
                tca_d1, tca_d2 = st.columns(2)
                with tca_d1:
                    arr_in = st.number_input("Arrival ₹ (0 = auto/day open)", min_value=0.0,
                                             value=float(plan.arrival_price), step=1.0)
                with tca_d2:
                    date_in = st.text_input("Trade date (YYYY-MM-DD, blank = latest)", value="")
                graded = st.form_submit_button("Grade execution")

            if graded:
                fdf = fills_edit.dropna(subset=["price", "qty"])
                fdf = fdf[(fdf["price"] > 0) & (fdf["qty"] > 0)]
                if fdf.empty:
                    st.warning("Enter at least one fill with a price and qty.")
                else:
                    fills = []
                    for r in fdf.itertuples():
                        row = {"price": float(r.price), "qty": float(r.qty)}
                        tstr = str(getattr(r, "time", "") or "").strip()
                        if tstr and date_in.strip():
                            row["time"] = f"{date_in.strip()} {tstr}"
                        fills.append(row)
                    rep = tca_mod.analyze_fills(
                        etk, eside, fills,
                        arrival_price=arr_in or None,
                        date=date_in.strip() or None)
                    if rep is None:
                        st.warning("Couldn't grade — check the inputs.")
                    else:
                        t1, t2, t3, t4 = st.columns(4)
                        t1.metric("Avg fill ₹", f"{rep.avg_fill:,.2f}", f"{rep.qty:,} sh")
                        t2.metric("Impl. shortfall", f"{rep.is_bps:+.0f} bps",
                                  "vs arrival", delta_color="inverse")
                        t3.metric("vs VWAP", f"{rep.vwap_slip_bps:+.0f} bps",
                                  delta_color="inverse")
                        t4.metric("All-in", f"{rep.total_cost_bps:.0f} bps",
                                  f"incl ~{rep.fees_bps:.0f} bps fees", delta_color="inverse")
                        st.caption(f"Arrival ₹{rep.arrival_price:,.2f} · VWAP ₹{rep.benchmark_vwap:,.2f} "
                                   f"· TWAP ₹{rep.benchmark_twap:,.2f} · benchmark basis: {rep.basis}")
                        for v in rep.verdict:
                            st.markdown(f"- {v}")

# --- 📟 Paper Trader (hands-off simulated autotrader — NO real orders) ----------
if _page == "📟 Paper Trader":
    st.subheader("📟 Paper autotrader — hands-off, simulated, no real money")
    st.caption(
        "Runs your signals forward as a **paper** book: each step exits what should "
        "exit, then opens new trades — sized by fixed-fractional risk, **filled at an "
        "execution-cost-adjusted price** (square-root impact + spread, from the "
        "Execution engine), and gated by real risk limits + a drawdown **kill-switch**. "
        "Nothing here places a live order on Groww — it only writes paper positions to "
        "your local DB. Run it daily (button below, the CLI, or the /loop skill)."
    )

    with st.expander("⚙️ Autotrader settings", expanded=False):
        ps1, ps2, ps3, ps4 = st.columns(4)
        with ps1:
            pt_cap = st.number_input("Capital ₹", min_value=10000, value=int(ACCOUNT_CAPITAL),
                                     step=10000, key="pt_cap")
            pt_risk = st.number_input("Risk %/trade", min_value=0.1, max_value=5.0,
                                      value=float(RISK_PER_TRADE_PCT), step=0.1, key="pt_risk")
        with ps2:
            pt_maxpos = st.number_input("Max positions", min_value=1, max_value=30,
                                        value=int(MAX_OPEN_POSITIONS), key="pt_maxpos")
            pt_heat = st.number_input("Max heat %", min_value=1.0, max_value=50.0,
                                      value=6.0, step=0.5, key="pt_heat",
                                      help="Total open risk (Σ (entry−stop)×qty) as % of capital")
        with ps3:
            pt_dd = st.number_input("Kill-switch DD %", min_value=2.0, max_value=50.0,
                                    value=10.0, step=1.0, key="pt_dd",
                                    help="Halt NEW entries when equity drawdown from peak hits this")
            pt_minscore = st.number_input("Min signal score", min_value=0.0, max_value=100.0,
                                          value=60.0, step=1.0, key="pt_minscore")
        with ps4:
            pt_algo = st.selectbox("Fill algo", ["vwap", "twap", "pov", "is"],
                                   format_func=str.upper, key="pt_algo")
            pt_uptrend = st.checkbox("Require uptrend", value=True, key="pt_uptrend",
                                     help="Gate out counter-trend (below-200-EMA) signals")
            pt_flatten = st.checkbox("Flatten on kill-switch", value=False, key="pt_flatten")

    cfg = paper_trader_mod.AutoTraderConfig(
        capital=float(pt_cap), risk_pct=float(pt_risk), max_positions=int(pt_maxpos),
        max_portfolio_heat_pct=float(pt_heat), kill_switch_dd_pct=float(pt_dd),
        min_score=float(pt_minscore), algo=pt_algo, require_uptrend=bool(pt_uptrend),
        flatten_on_kill=bool(pt_flatten))

    state = paper_trader_mod.account_state(cfg)

    # --- kill-switch banner
    if state["halted"]:
        st.error(f"⛔ **Kill-switch ACTIVE** — drawdown {state['drawdown_pct']:.1f}% "
                 f"≥ {cfg.kill_switch_dd_pct:.0f}%. New entries are halted until equity recovers.")
    else:
        st.success(f"🟢 Trading enabled — drawdown {state['drawdown_pct']:.1f}% "
                   f"(halts at {cfg.kill_switch_dd_pct:.0f}%).")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Equity", f"₹{state['equity']:,.0f}",
              f"{(state['equity'] / cfg.capital - 1) * 100:+.1f}% vs capital")
    a2.metric("Realized / Unrealized", f"₹{state['realized_pnl']:,.0f}",
              f"₹{state['unrealized_pnl']:,.0f} open", delta_color="off")
    a3.metric("Open / Free cash", f"{state['n_open']} pos",
              f"₹{state['free_cash']:,.0f} free", delta_color="off")
    a4.metric("Portfolio heat", f"{state['portfolio_heat_pct']:.1f}%",
              f"of {cfg.max_portfolio_heat_pct:.0f}% cap", delta_color="off")

    r1, r2 = st.columns([1, 1])
    with r1:
        run_scope = st.radio("Signal universe", ["Watchlist", "Watchlist + small caps"],
                             horizontal=True, key="pt_scope")
        run_now = st.button("▶ Run one step (scan → exit → enter)", type="primary",
                            width="stretch", key="pt_run")
    with r2:
        st.caption("⚠️ A step writes paper positions to your local DB.")
        flat = st.button("🛑 Flatten all paper positions", width="stretch", key="pt_flat")

    if flat:
        n = paper_trader_mod.flatten_all(reason="manual_flatten")
        st.warning(f"Flattened {n} open paper position(s).")
        state = paper_trader_mod.account_state(cfg)

    if run_now:
        uni = (combined_universe(include_smallcaps=True) if run_scope.endswith("small caps")
               else get_watchlist())
        with st.spinner(f"Scanning {len(uni)} names, then stepping the autotrader…"):
            report = paper_trader_mod.step(cfg, universe=uni)
        st.session_state["pt_last_report"] = report
        state = paper_trader_mod.account_state(cfg)

    report = st.session_state.get("pt_last_report")
    if report is not None:
        st.markdown("#### Last step")
        for n in report.notes:
            st.info(n)
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(f"**🟢 Opened ({len(report.entries)})**")
            if report.entries:
                edf = pd.DataFrame(report.entries)[
                    ["ticker", "setup", "qty", "fill", "signal_entry", "slip_bps", "risk"]]
                st.dataframe(edf, hide_index=True, width="stretch", column_config={
                    "fill": st.column_config.NumberColumn("Fill ₹", format="%.2f"),
                    "signal_entry": st.column_config.NumberColumn("Signal ₹", format="%.2f"),
                    "slip_bps": st.column_config.NumberColumn("Slip", format="%.0f bps"),
                    "risk": st.column_config.NumberColumn("Risk ₹", format="%.0f"),
                })
            else:
                st.caption("Nothing opened this step.")
        with rc2:
            st.markdown(f"**🔴 Closed ({len(report.exits)})**")
            if report.exits:
                st.dataframe(pd.DataFrame(report.exits), hide_index=True, width="stretch")
            else:
                st.caption("Nothing closed this step.")
        if report.rejected:
            with st.expander(f"Rejected signals ({len(report.rejected)})"):
                st.dataframe(pd.DataFrame(report.rejected), hide_index=True, width="stretch")

    # --- equity curve from the run log
    log = paper_trader_mod.run_log(limit=300)
    if not log.empty and log["equity"].notna().any():
        st.markdown("#### Equity curve (autotrader run log)")
        efig = go.Figure()
        efig.add_trace(go.Scatter(y=log["equity"], x=log.index, mode="lines",
                                  name="Equity", line=dict(color="#1f77b4", width=2)))
        if log["peak_equity"].notna().any():
            efig.add_trace(go.Scatter(y=log["peak_equity"], x=log.index, mode="lines",
                                      name="Peak", line=dict(color="#2ca02c", width=1, dash="dot")))
        efig.add_hline(y=cfg.capital, line_dash="dash", line_color="#888",
                       annotation_text="starting capital")
        efig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                           xaxis_title="step #", yaxis_title="₹ equity",
                           legend=dict(orientation="h", y=1.12))
        st.plotly_chart(efig, width="stretch")

    # --- open positions (live)
    open_pos = load_positions(status="open", is_paper=True)
    if not open_pos.empty:
        st.markdown(f"#### Open paper positions ({len(open_pos)})")
        op = open_pos.copy()
        op["pnl_₹"] = (op["last_price"].fillna(op["entry_price"]) - op["entry_price"]) * op["qty"]
        op["pnl_%"] = (op["last_price"].fillna(op["entry_price"]) / op["entry_price"] - 1) * 100
        cols = ["ticker", "setup", "qty", "entry_price", "stoploss", "target",
                "last_price", "pnl_₹", "pnl_%", "entry_date"]
        st.dataframe(op[cols], hide_index=True, width="stretch", column_config={
            "entry_price": st.column_config.NumberColumn("Entry ₹", format="%.2f"),
            "last_price": st.column_config.NumberColumn("Last ₹", format="%.2f"),
            "pnl_₹": st.column_config.NumberColumn("P&L ₹", format="%.0f"),
            "pnl_%": st.column_config.NumberColumn("P&L %", format="%.1f"),
        })

    # --- run-log audit table
    if not log.empty:
        with st.expander("📜 Run log (audit trail)"):
            show_cols = [c for c in ["asof", "equity", "drawdown_pct", "portfolio_heat_pct",
                                     "n_open", "opened", "closed", "halted", "note"]
                         if c in log.columns]
            st.dataframe(log[show_cols].iloc[::-1], hide_index=True, width="stretch")
