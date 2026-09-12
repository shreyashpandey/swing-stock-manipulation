"""Small metric views shared by dashboard routes."""
import math

import streamlit as st


def format_metric(value, spec=".2f", suffix=""):
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{value:{spec}}{suffix}"


def render_backtest_metrics(summary):
    first, second, third = st.columns(3)
    first.metric("ROC-AUC / Average precision",
                 f"{format_metric(summary.get('roc_auc'))} / {format_metric(summary.get('pr_auc'))}")
    second.metric("Precision@5 / Recall@5",
                  f"{format_metric(summary.get('precision_at_5'))} / {format_metric(summary.get('recall_at_5'))}")
    third.metric("Avg Return Top-5",
                 format_metric(summary.get("avg_return_top_5_pct"), "+.2f", "%"))
    st.caption("Metrics cover candidates passing the scanner threshold. — means unavailable: "
               "there may be no candidates, no positive outcomes, or only one outcome class. "
               "Portfolio Sharpe and drawdown are unavailable because this backtest does not "
               "model daily equity, overlapping positions or idle cash.")


def render_market_pulse(pulse):
    for column, name in zip(st.columns(3), ("NIFTY 50", "NIFTY BANK", "INDIA VIX")):
        item = pulse.get(name, {})
        change = item.get("chg_1d")
        column.metric(name, format_metric(item.get("close")),
                      delta=None if change is None else format_metric(change, "+.2f", "%"))
    if not pulse:
        st.caption("Refresh macro data to populate the market pulse.")
