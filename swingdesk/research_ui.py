"""Research page, imported only when selected by the Streamlit router."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from swingdesk import storage
from swingdesk.analyze import research


def _save(kind, name, payload):
    try:
        research.save_item(kind, name, payload)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.success(f"Saved {name.strip()} locally.")
    st.rerun()


def render():
    st.subheader("Research workspace")
    st.caption("User-defined research using stored data. Refresh prices and fundamentals with the sidebar actions. Returns count stored observations, which may skip trading sessions. Window start dates and price_as_of show the actual periods.")
    mode = st.radio("Research tool", ["Watchlists", "Saved screens", "Compare", "Heatmap", "Alerts"], horizontal=True)
    lists = research.saved_items("watchlist")
    options = ["Current watchlist"] + [f"Saved: {name}" for name in lists]
    selected = st.selectbox("Research universe", options)
    tickers = storage.get_watchlist() if selected == options[0] else lists[selected[7:]]["tickers"]

    if mode == "Watchlists":
        st.caption("Save independent lists. Names are case-sensitive; saving an existing name replaces that list. Use NSE .NS or BSE .BO symbols.")
        with st.form("research_watchlist"):
            name = st.text_input("Watchlist name", value="" if selected == options[0] else selected[7:])
            raw = st.text_area("Tickers, separated by commas or new lines", value=", ".join(tickers))
            if st.form_submit_button("Save watchlist"):
                _save("watchlist", name, {"tickers": raw.replace("\n", ",").split(",")})
        if selected != options[0]:
            left, right = st.columns(2)
            if left.button("Use as current watchlist"):
                storage.set_watchlist(tickers)
                st.rerun()
            if right.button("Delete selected saved watchlist"):
                research.delete_item("watchlist", selected[7:])
                st.rerun()
        st.dataframe(research.snapshot(tickers), hide_index=True, width="stretch")
        return

    if mode == "Saved screens":
        saved = research.saved_items("screen")
        choice = st.selectbox("Screen template", ["New screen", *[f"Saved: {name}" for name in saved]])
        chosen = choice[7:] if choice != "New screen" else None
        template = saved.get(chosen, {"rules": [], "tickers": tickers})
        st.caption("All conditions must match. Missing values are excluded by active filters. A saved screen retains its own ticker universe; saving the same name replaces it.")
        use_saved = chosen is not None and st.checkbox("Use this screen's saved universe", value=True)
        universe = template["tickers"] if use_saved else tickers
        # A keyed editor is recreated when selecting a different saved screen.
        edited = st.data_editor(
            pd.DataFrame(template["rules"], columns=["metric", "op", "value"]),
            num_rows="dynamic", hide_index=True, key=f"rules_{choice}",
            column_config={
                "metric": st.column_config.SelectboxColumn("Metric", options=list(research.METRICS), required=True),
                "op": st.column_config.SelectboxColumn("Comparison", options=list(research.OPS), required=True),
                "value": st.column_config.NumberColumn("Threshold", required=True),
            }, width="stretch",
        )
        with st.expander("Metric units"):
            st.table(pd.DataFrame(research.METRICS.items(), columns=["Metric", "Meaning"]))
        rules = edited.to_dict("records")
        try:
            results = research.filter_snapshot(research.snapshot(universe), rules)
        except ValueError as exc:
            st.info(str(exc))
            return
        st.write(f"{len(results)} of {len(universe)} symbols match your criteria")
        sort = st.selectbox("Sort results by", list(research.METRICS), format_func=research.METRICS.get)
        ascending = st.checkbox("Ascending order")
        results = results.sort_values(sort, ascending=ascending, na_position="last")
        st.dataframe(results, hide_index=True, width="stretch")
        st.download_button("Export results as CSV", results.to_csv(index=False), "swingdesk_screen.csv", "text/csv")
        name = st.text_input("Save screen as", value=chosen or "")
        if st.button("Save screen"):
            _save("screen", name, {"rules": rules, "tickers": universe})
        if chosen is not None and st.button("Delete selected screen"):
            research.delete_item("screen", chosen)
            st.rerun()
        return

    if mode == "Compare":
        chosen = st.multiselect("Compare up to six symbols", tickers, default=tickers[:min(3, len(tickers))], max_selections=6)
        sessions = st.select_slider("Stored observations", options=[21, 63, 126, 252], value=126)
        if not chosen:
            st.info("Select symbols to compare.")
            return
        st.dataframe(research.snapshot(chosen).set_index("ticker").T, width="stretch")
        chart = research.comparison_series(chosen, sessions)
        if chart.empty:
            st.info("Comparison needs at least two common dates of positive closes for every selected symbol.")
        else:
            st.plotly_chart(px.line(chart, labels={"value": "Rebased close (start = 100)", "variable": "Ticker"}), width="stretch")
            st.caption(f"Common observations: {chart.index[0].date()} to {chart.index[-1].date()}. Missing sessions are not filled. Stored close performance excludes cash dividends; past performance does not indicate future returns.")
        return

    if mode == "Heatmap":
        frame = research.snapshot(tickers)
        metric = st.selectbox("Return period", ["change_1d", "change_5d", "change_21d"], format_func=research.METRICS.get)
        available = frame.dropna(subset=[metric]).copy()
        st.caption("Equal-size stock tiles grouped by sector. Colour shows returns across stored observations; actual date windows may differ by symbol.")
        if available.empty:
            st.info("No return history in this universe. Refresh prices first.")
        else:
            available["tile_size"] = 1
            st.plotly_chart(px.treemap(available, path=["sector", "ticker"], values="tile_size",
                                      color=metric, color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                                      hover_data=[f"{metric}_from", "price_as_of", metric]), width="stretch")
        st.dataframe(frame[["ticker", "sector", f"{metric}_from", "price_as_of", metric]], hide_index=True, width="stretch")
        return

    st.caption("Alerts are checked only when you press Check alerts, against stored daily bars. No background monitoring or messages are sent. Data older than four calendar days is marked stale. Saving an existing alert name replaces its rule.")
    with st.form("research_alert"):
        name = st.text_input("Alert name")
        metric = st.selectbox("Alert metric", research.PRICE_METRICS, format_func=research.METRICS.get)
        op = st.selectbox("Condition", list(research.OPS))
        value = st.number_input("Threshold", value=0.0)
        if st.form_submit_button("Save alert for this universe"):
            _save("alert", name, {"tickers": tickers, "rules": [{"metric": metric, "op": op, "value": value}]})
    alerts = research.saved_items("alert")
    if alerts:
        st.dataframe(pd.DataFrame([{"name": name, "symbols": ", ".join(item["tickers"]),
                                    "conditions": str(item["rules"])} for name, item in alerts.items()]), hide_index=True)
        selected_alert = st.selectbox("Saved alert", list(alerts))
        if st.button("Delete selected alert"):
            research.delete_item("alert", selected_alert)
            st.rerun()
        if st.button("Check alerts", type="primary"):
            st.dataframe(research.evaluate_alerts(), hide_index=True, width="stretch")
    else:
        st.info("Save a condition to create your first alert.")
