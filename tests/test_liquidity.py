"""Tests for the liquidity/tradeability profile and the combined screener."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk.analyze import liquidity as liq
from swingdesk.analyze import screener
from swingdesk.storage import upsert_fundamentals, upsert_prices


def _seed_stock(ticker, price, daily_volume, n=120, seed=1):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.001, 0.02, n)
    close = price * np.cumprod(1 + rets)
    vol = np.full(n, float(daily_volume)) * rng.uniform(0.8, 1.2, n)
    idx = pd.date_range("2024-06-01", periods=n, freq="B")
    upsert_prices(ticker, pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": vol}, index=idx))


def _fund(ticker, mcap, float_sh, shares_out):
    upsert_fundamentals([{"ticker": ticker, "market_cap": mcap,
                          "float_shares": float_sh, "shares_outstanding": shares_out}])


def test_liquid_vs_illiquid_scoring(tmp_db):
    # LIQUID: ₹500 price × ~2,000,000 sh/day = ~₹100 cr/day, decent float churn.
    _seed_stock("LIQ.NS", 500, 2_000_000, seed=1)
    _fund("LIQ.NS", mcap=2.0e11, float_sh=2.0e8, shares_out=4.0e8)
    # ILLIQUID: ₹500 × ~3,000 sh/day = ~₹0.15 cr/day.
    _seed_stock("ILQ.NS", 500, 3_000, seed=2)
    _fund("ILQ.NS", mcap=2.0e10, float_sh=4.0e7, shares_out=8.0e7)

    lq = liq.liquidity_profile("LIQ.NS")
    il = liq.liquidity_profile("ILQ.NS")
    assert lq.adv_value_cr > 50 and il.adv_value_cr < 1
    assert lq.score > il.score
    assert lq.tier == "liquid"
    assert il.tier in ("illiquid", "untradeable")
    assert any("illiquid" in r.lower() for r in il.reasons)


def test_turnover_and_float_metrics(tmp_db):
    _seed_stock("X.NS", 100, 1_000_000, seed=3)   # ~₹10 cr/day
    _fund("X.NS", mcap=1.0e11, float_sh=5.0e8, shares_out=1.0e9)
    p = liq.liquidity_profile("X.NS")
    # turnover ≈ ADV(₹10cr) / mcap(₹10000cr) ≈ 0.1%/day
    assert 0.05 < p.turnover_pct < 0.2
    # float turnover ≈ 1,000,000 / 5e8 = 0.2%/day
    assert 0.1 < p.float_turnover_pct < 0.4
    assert p.float_basis == "float"


def test_float_fallback_to_shares_out(tmp_db):
    _seed_stock("NF.NS", 100, 500_000, seed=4)
    _fund("NF.NS", mcap=5.0e10, float_sh=None, shares_out=3.0e8)
    p = liq.liquidity_profile("NF.NS")
    assert p.float_basis == "shares_out"
    assert np.isfinite(p.float_turnover_pct)


def test_missing_data_returns_none(tmp_db):
    _seed_stock("SHORT.NS", 100, 10_000, n=10)
    assert liq.liquidity_profile("SHORT.NS") is None


# ---- combined screener ----------------------------------------------------
def test_screen_flags_good_but_illiquid(tmp_db):
    # AAA & CCC both have strong (clean, low-noise) uptrends -> good factors;
    # BBB is flat -> weak. AAA is liquid, CCC is illiquid. The screener must
    # call AAA '✅ good & tradeable' and CCC '⚠ good but illiquid'.
    specs = [("AAA.NS", 0.005, 2_000_000, 1),
             ("BBB.NS", -0.001, 800_000, 2),
             ("CCC.NS", 0.0048, 2_000, 3)]
    for t, drift, vol, seed in specs:
        rng = np.random.default_rng(seed)
        n = 300
        rets = rng.normal(drift, 0.01, n)
        close = 500 * np.cumprod(1 + rets)
        v = np.full(n, float(vol)) * rng.uniform(0.8, 1.2, n)
        idx = pd.date_range("2023-06-01", periods=n, freq="B")
        upsert_prices(t, pd.DataFrame({"open": close, "high": close*1.01,
            "low": close*0.99, "close": close, "volume": v}, index=idx))
        _fund(t, mcap=2.0e11, float_sh=2.0e8, shares_out=4.0e8)

    df = screener.screen(["AAA.NS", "BBB.NS", "CCC.NS"])
    assert not df.empty
    row = df.set_index("ticker")
    # The illiquid-but-good name is flagged, never surfaced as tradeable.
    assert row.loc["CCC.NS", "verdict"] == "⚠ good but illiquid"
    assert row.loc["CCC.NS", "liq_tier"] in ("illiquid", "untradeable")
    # The liquid + good name is the top pick.
    assert row.loc["AAA.NS", "verdict"] == "✅ good & tradeable"
    assert df.iloc[0]["ticker"] == "AAA.NS"
    assert "illiquid" in row.loc["CCC.NS", "why"].lower()


def test_screen_empty(tmp_db):
    assert screener.screen([]).empty
