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
from swingdesk.analyze import score as scoring
from swingdesk.analyze import smallcaps
from swingdesk.analyze import sentiment as sentiment_mod
from swingdesk.analyze import summary as stock_summary
from swingdesk.analyze.setups import scan_all
from swingdesk.analyze.technicals import (
    add_indicators,
    money_flow_read,
    trend_quality,
    volume_profile,
    volume_profile_read,
)
from swingdesk.backtest import engine as bt_engine
from swingdesk.backtest import metrics as bt_metrics
from swingdesk.config import (
    ACCOUNT_CAPITAL,
    DEFAULT_WATCHLIST,
    MAX_OPEN_POSITIONS,
    RISK_PER_TRADE_PCT,
)
from swingdesk.ingest import fundamentals as fundamentals_ingest
from swingdesk.ingest import macro as macro_ingest
from swingdesk.ingest import news_rss, prices
from swingdesk.notify import telegram
from swingdesk.backtest import optimizer as opt
from swingdesk.portfolio import allocate as allocate_mod
from swingdesk.portfolio import holdings as holdings_mod
from swingdesk.portfolio import journal as pj
from swingdesk.portfolio import positions as portfolio
from swingdesk.portfolio import reconcile as reconcile_mod
from swingdesk.storage import (
    add_to_smallcap_watchlist,
    combined_universe,
    get_smallcap_watchlist,
    get_watchlist,
    holdings_tickers,
    remove_from_smallcap_watchlist,
    set_smallcap_watchlist,
    init_db,
    list_backtest_runs,
    load_backtest_trades,
    load_news,
    load_positions,
    load_prices,
    load_signals,
    save_backtest_trades,
    save_signals,
    seed_watchlist_if_empty,
    set_watchlist,
)

st.set_page_config(page_title="SwingDesk", layout="wide", page_icon=":chart_with_upwards_trend:")

init_db()
seed_watchlist_if_empty(DEFAULT_WATCHLIST)

st.title("SwingDesk — local swing-trading signals (NSE)")
st.caption("Run the daily refresh from the sidebar. All data is stored locally in SQLite.")

# --- Sidebar: actions + watchlist ----------------------------------------------
with st.sidebar:
    st.header("Actions")
    if st.button("Fetch prices", width="stretch"):
        with st.spinner("Downloading OHLCV..."):
            # Watchlist + holdings (+ small caps) so your portfolio always has
            # chartable price data, not just the curated watchlist.
            res = prices.ingest(combined_universe(include_smallcaps=True),
                                period="2y")
        st.success(f"prices: {sum(1 for v in res.values() if v > 0)}/{len(res)} ok")

    if st.button("Fetch news", width="stretch"):
        with st.spinner("Pulling RSS feeds..."):
            # Include small-caps + holdings so headlines naming those names
            # get tagged to the right ticker, not just the main watchlist.
            n = news_rss.ingest(combined_universe(include_smallcaps=True))
        st.success(f"news: {n} new items")

    if st.button("Analyze news (Claude)", width="stretch"):
        with st.spinner("Running sentiment analysis..."):
            n = sentiment_mod.ingest(max_items=200)
        st.success(f"sentiment: {n} headlines classified")

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


(tab_holdings, tab_invest, tab_discover, tab_smallcaps, tab_signals, tab_chart,
 tab_news, tab_backtest, tab_optimize, tab_portfolio, tab_reconcile,
 tab_fundamentals, tab_data) = st.tabs(
    ["My Holdings", "💸 Invest", "Discover", "Small Caps", "Signals", "Chart",
     "News", "Backtest", "Optimize", "Portfolio", "Reconcile", "Fundamentals",
     "Raw data"]
)

# --- Discover tab ---------------------------------------------------------------
with tab_discover:
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
with tab_invest:
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


# --- Small Caps tab -------------------------------------------------------------
with tab_smallcaps:
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
                    ok = sum(1 for v in res.values() if v > 0)
                    st.success(f"prices: {ok}/{len(res)} ok")
            with bb:
                if st.button("📊 Fundamentals", width="stretch",
                              key="sc_wl_fund"):
                    with st.spinner("Fetching fundamentals..."):
                        n = fundamentals_ingest.ingest(sc_wl, workers=5)
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
with tab_signals:
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
        latest_display = latest.copy()
        latest_display["universe"] = latest_display["universe"].map({
            "main": "🟦 main", "smallcap": "🟧 smallcap"
        }).fillna(latest_display["universe"])
        st.dataframe(
            latest_display[
                ["generated_at", "universe", "ticker", "setup", "direction",
                 "entry", "stoploss", "target", "rr", "score", "notes"]
            ],
            width="stretch",
            hide_index=True,
        )

# --- Chart tab ------------------------------------------------------------------
with tab_chart:
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
            display = {t: f"★ {t}" if t in held_set else t for t in options}
            ticker = st.selectbox("Ticker", options,
                                   format_func=lambda t: display[t])
            lookback = st.slider("Lookback (days)", 60, 500, 180)
            show_markers = st.checkbox("Show signal labels", value=True,
                                        help="Plot historical entry signals as markers")

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
                legend=dict(orientation="h", y=1.02, x=0),
                bargap=0,
            )
            st.plotly_chart(fig, width="stretch")

            with st.expander("Indicator snapshot (last bar)"):
                last = df.iloc[-1]
                cols = ["close", "ema20", "ema50", "ema200", "rsi14",
                        "macd", "macd_signal", "atr14", "vol_avg20",
                        "mfi14", "obv", "buy_pressure_20"]
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

            # --- Methodology: the norms behind every suggestion
            with st.expander("📖 How SwingDesk decides — the norms behind this"):
                st.markdown(glossary.VERDICT_METHODOLOGY)
                st.markdown(glossary.NORMS)
                st.markdown(glossary.SETUPS)

# --- News tab -------------------------------------------------------------------
with tab_news:
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

# --- Backtest tab ---------------------------------------------------------------
with tab_backtest:
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

    if st.button("Run backtest", type="primary", width="stretch"):
        from datetime import datetime
        tickers = [bt_ticker] if bt_ticker else get_watchlist()
        with st.spinner(f"Backtesting {len(tickers)} ticker(s)..."):
            trades_df = bt_engine.backtest_universe(tickers, max_hold=bt_max_hold)
        if trades_df.empty:
            st.warning("No trades produced — insufficient price history or no setups fired.")
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_backtest_trades(run_id, trades_df)
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

        # --- Summary metrics
        summary = bt_metrics.summarize(trades_df)
        st.markdown("**Strategy summary** (one row per setup, plus ALL)")
        show_cols = ["setup", "n_trades", "win_rate", "avg_r", "total_r",
                     "expectancy", "profit_factor", "max_drawdown_r",
                     "max_consec_losses", "avg_bars_held"]
        st.dataframe(summary[show_cols], width="stretch", hide_index=True)

        # --- Edge gating
        st.markdown("**Edge gate** (passes ⇒ tradeable; fails ⇒ keep paper-trading)")
        for _, row in summary.iterrows():
            if row["setup"] == "ALL":
                continue
            ok, fails = bt_metrics.gate(row.to_dict())
            if ok:
                st.success(f"✅ {row['setup']}  ·  expectancy={row['expectancy']}R  ·  pf={row['profit_factor']}")
            else:
                st.error(f"❌ {row['setup']}  ·  {', '.join(fails)}")

        # --- Equity curve in R
        if not trades_df.empty:
            curve = trades_df.sort_values("entry_date").copy()
            curve["cum_r"] = curve["r"].cumsum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(curve["entry_date"]),
                y=curve["cum_r"],
                mode="lines+markers",
                name="Cumulative R",
                line=dict(color="#1f77b4"),
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="#888")
            fig.update_layout(
                height=350,
                title="Equity curve (cumulative R)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, width="stretch")

        # --- Trade log
        with st.expander(f"Trade log ({len(trades_df)} trades)"):
            st.dataframe(trades_df.drop(columns=["id", "run_id"], errors="ignore"),
                         width="stretch", hide_index=True)


# --- My Holdings tab ------------------------------------------------------------
with tab_holdings:
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
        with st.spinner("Running per-holding analysis..."):
            results = holdings_mod.analyze_portfolio(df_h, use_ai_thesis=ai_on)
        summary = holdings_mod.portfolio_summary(results)

        # --- Macro pulse: gives context to the analyses below
        pulse = macro_ingest.market_pulse()
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
                "Trade qty": p.shares if p.shares else "—",
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
with tab_optimize:
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
with tab_portfolio:
    st.subheader("Paper + real portfolio")

    book = st.radio("Book", ["Paper", "Real", "Both"], horizontal=True, key="book")
    is_paper = None if book == "Both" else (book == "Paper")

    # Auto mark-to-market on first visit per session — so positions display
    # real LTP and live unrealized P&L without the user clicking anything.
    if "portfolio_auto_mtm_done" not in st.session_state:
        with st.spinner("Pulling latest prices for open positions..."):
            auto_res = portfolio.mark_to_market(auto_fetch=True, refresh=False)
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
with tab_reconcile:
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
with tab_fundamentals:
    st.subheader("Fundamental quality ranking")
    st.caption(
        "Each ticker scored 0–100 on a weighted blend of ROE, growth, margins, "
        "valuation, debt, and size. Signals on high-quality stocks get a "
        "composite-score boost."
    )

    from swingdesk.ingest import fundamentals as fund_ingest
    from swingdesk.storage import load_fundamentals

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Refresh fundamentals (yfinance)", width="stretch"):
            with st.spinner(f"Pulling fundamentals for {len(get_watchlist())} tickers..."):
                n = fund_ingest.ingest(get_watchlist())
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
with tab_data:
    st.subheader("Watchlist price coverage")
    rows = []
    for t in get_watchlist():
        df = load_prices(t)
        if df.empty:
            rows.append({"ticker": t, "bars": 0, "first": None, "last": None})
        else:
            rows.append({
                "ticker": t,
                "bars": len(df),
                "first": df.index.min().date(),
                "last": df.index.max().date(),
            })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
