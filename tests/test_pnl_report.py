"""Realized round-trip reconstruction, charge-aware net P&L, and performance."""
from __future__ import annotations

import pandas as pd

from swingdesk.analyze import pnl_report as pr


def _trades(rows):
    """rows: list of (ticker, side, qty, price, 'YYYY-MM-DD')."""
    df = pd.DataFrame(rows, columns=["ticker", "side", "qty", "price", "trade_date"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["segment"] = "delivery"
    df["exchange"] = "NSE"
    return df


def test_simple_round_trip_net_below_gross():
    rt = pr.realized_roundtrips(_trades([
        ("X.NS", "buy", 100, 100, "2025-01-01"),
        ("X.NS", "sell", 100, 110, "2025-01-15"),
    ]))
    assert len(rt) == 1
    row = rt.iloc[0]
    assert row["gross_pnl"] == 1000.0          # (110-100)*100
    assert row["charges"] > 0
    assert row["net_pnl"] < row["gross_pnl"]   # charges bite
    assert row["holding_days"] == 14
    assert row["bucket"] == "swing"
    assert row["gain_type"] == "STCG"


def test_fifo_partial_matching_across_lots():
    rt = pr.realized_roundtrips(_trades([
        ("X.NS", "buy", 100, 100, "2025-01-01"),
        ("X.NS", "buy", 50, 120, "2025-01-05"),
        ("X.NS", "sell", 120, 130, "2025-02-01"),   # consumes lot1 (100) + 20 of lot2
    ]))
    assert len(rt) == 2
    # FIFO: first round-trip is the ₹100 lot, second the ₹120 lot.
    assert rt.iloc[0]["buy_price"] == 100.0 and rt.iloc[0]["qty"] == 100
    assert rt.iloc[1]["buy_price"] == 120.0 and rt.iloc[1]["qty"] == 20
    assert round(rt["gross_pnl"].sum(), 2) == round((130 - 100) * 100 + (130 - 120) * 20, 2)


def test_long_term_bucket_and_ltcg():
    rt = pr.realized_roundtrips(_trades([
        ("Y.NS", "buy", 10, 1000, "2024-01-01"),
        ("Y.NS", "sell", 10, 1500, "2025-01-10"),   # 375 days
    ]))
    row = rt.iloc[0]
    assert row["bucket"] == "long_term"
    assert row["gain_type"] == "LTCG"


def test_performance_metrics_and_bucket_filter():
    rt = pr.realized_roundtrips(_trades([
        ("A.NS", "buy", 100, 100, "2025-01-01"),
        ("A.NS", "sell", 100, 120, "2025-01-20"),   # +2000 gross, swing, win
        ("B.NS", "buy", 100, 200, "2025-02-01"),
        ("B.NS", "sell", 100, 190, "2025-02-10"),   # -1000 gross, swing, loss
    ]))
    perf = pr.performance(rt, bucket="swing")
    assert perf["n_trades"] == 2
    assert perf["win_rate"] == 50.0
    assert perf["net_pnl"] < perf["gross_pnl"]          # charges
    assert perf["profit_factor"] is not None and perf["profit_factor"] > 0
    assert perf["charges_pct_of_turnover"] > 0
    assert "by_month" in perf and perf["tax"]["total_tax"] >= 0
    # A bucket with no trades returns an empty-ish summary.
    assert pr.performance(rt, bucket="long_term")["n_trades"] == 0


def test_tax_uses_stcg_rate_on_net_short_term_gains():
    rt = pr.realized_roundtrips(_trades([
        ("A.NS", "buy", 100, 100, "2025-01-01"),
        ("A.NS", "sell", 100, 200, "2025-03-01"),   # big STCG gain
    ]))
    tax = pr.performance(rt, bucket=None)["tax"]
    # STCG tax = 20% of net short-term gain; no LTCG here.
    assert tax["ltcg_tax"] == 0.0
    assert tax["stcg_tax"] > 0


def test_empty_trades_returns_empty_frame():
    rt = pr.realized_roundtrips(pd.DataFrame(columns=["ticker", "side", "qty", "price", "trade_date"]))
    assert rt.empty
    assert pr.performance(rt)["n_trades"] == 0


def test_benchmark_alpha_graceful_without_macro(tmp_db):
    rt = pr.realized_roundtrips(_trades([
        ("A.NS", "buy", 100, 100, "2025-01-01"),
        ("A.NS", "sell", 100, 120, "2025-01-20"),
    ]))
    bm = pr.benchmark_alpha(rt)            # tmp_db has no macro rows
    assert bm["realized_return_pct"] is not None
    assert bm["benchmark_return_pct"] is None and bm["alpha_pct"] is None
