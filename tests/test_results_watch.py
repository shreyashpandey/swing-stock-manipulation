from __future__ import annotations

from datetime import date

import pandas as pd

from swingdesk.analyze import results_watch
from swingdesk.analyze.institutional import BrokerageAction, InstitutionalFlow


def _prices(start: float, end: float) -> pd.DataFrame:
    return pd.DataFrame({"close": [start, end]})


def test_results_watch_ranks_nearest_result_and_flags_accumulation():
    earnings = pd.DataFrame([
        {"ticker": "AAA.NS", "next_earnings": "2026-07-20"},
        {"ticker": "BBB.NS", "next_earnings": "2026-08-10"},
    ])
    fundamentals = pd.DataFrame([
        {"ticker": "AAA.NS", "revenue_growth": 0.12, "earnings_growth": 0.15,
         "profit_margin": 0.18, "operating_margin": 0.21},
        {"ticker": "BBB.NS", "revenue_growth": 0.02, "earnings_growth": -0.05,
         "profit_margin": 0.04, "operating_margin": 0.05},
    ])
    flows = [
        InstitutionalFlow(
            ticker="AAA.NS", security="Aaa", n_deals=1, buy_qty=100, sell_qty=0,
            net_qty=100, buy_value=1_000_000_000, sell_value=0, net_value=1_000_000_000,
            net_side="BUY", clients=[("MORGAN STANLEY", "BUY")],
            marquee=["MORGAN STANLEY"], latest_date="2026-07-18"),
    ]
    actions = [
        BrokerageAction("AAA.NS", "JEFFERIES", "RAISES TARGET", "bullish",
                        "Jefferies raises target", "2026-07-18", None),
    ]

    out = results_watch.build_results_watch(
        ["BBB.NS", "AAA.NS"],
        earnings,
        fundamentals,
        lambda ticker: _prices(100, 104),
        flows=flows,
        actions=actions,
        today=date(2026, 7, 18),
    )

    assert out.iloc[0]["ticker"] == "AAA.NS"
    assert out.iloc[0]["days_left"] == 2
    assert out.iloc[0]["window"] == "0-3d"
    assert out.iloc[0]["broker_tone"] == "Positive"
    assert out.iloc[0]["institutional_flow"] == "BUY"
    assert out.iloc[0]["marquee"] == "MORGAN STANLEY"
    assert out.iloc[0]["readiness"] == "Institutional accumulation"


def test_results_watch_high_gap_risk_when_near_and_run_up_with_negative_growth():
    earnings = pd.DataFrame([
        {"ticker": "WEAK.NS", "next_earnings": "2026-07-19"},
    ])
    fundamentals = pd.DataFrame([
        {"ticker": "WEAK.NS", "revenue_growth": -0.10, "earnings_growth": -0.25,
         "profit_margin": 0.02, "operating_margin": 0.03},
    ])
    flows = [
        InstitutionalFlow(
            ticker="WEAK.NS", security="Weak", n_deals=1, buy_qty=0, sell_qty=100,
            net_qty=-100, buy_value=0, sell_value=500_000_000, net_value=-500_000_000,
            net_side="SELL", clients=[("SOME FUND", "SELL")], marquee=[],
            latest_date="2026-07-18"),
    ]
    actions = [
        BrokerageAction("WEAK.NS", "MORGAN STANLEY", "DOWNGRADE", "bearish",
                        "Morgan Stanley downgrades Weak", "2026-07-18", None),
    ]

    out = results_watch.build_results_watch(
        ["WEAK.NS"],
        earnings,
        fundamentals,
        lambda _ticker: _prices(100, 113),
        flows=flows,
        actions=actions,
        today=date(2026, 7, 18),
    )

    row = out.iloc[0]
    assert row["gap_risk"] == "High"
    assert row["readiness"] == "High gap risk"
    assert row["pre_result_move_pct"] == 13.0


def test_results_watch_can_keep_unknown_dates_for_watchlist_visibility():
    out = results_watch.build_results_watch(
        ["MYSTERY.NS"],
        pd.DataFrame(columns=["ticker", "next_earnings"]),
        pd.DataFrame(columns=["ticker"]),
        lambda _ticker: pd.DataFrame(),
        today=date(2026, 7, 18),
    )

    assert len(out) == 1
    assert out.iloc[0]["window"] == "Unknown"
    assert out.iloc[0]["readiness"] == "Watchlist only"
