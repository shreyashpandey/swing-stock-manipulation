"""market_metrics is the single source of truth both liquidity and manipulation
delegate to. These tests pin (a) that today's and the 60-day-average turnover are
distinct, reconciled by today_vs_avg_value_mult, and (b) that the average block
reproduces liquidity's numbers exactly."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk.analyze import liquidity as liq
from swingdesk.analyze import market_metrics as mm
from swingdesk.storage import upsert_fundamentals, upsert_prices


def _seed(ticker, price, daily_vol, n=120, last_vol_mult=1.0, seed=1):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, n)
    close = price * np.cumprod(1 + rets)
    vol = np.full(n, float(daily_vol)) * rng.uniform(0.9, 1.1, n)
    vol[-1] *= last_vol_mult                        # bump the final bar's volume
    idx = pd.date_range("2024-06-01", periods=n, freq="B")
    upsert_prices(ticker, pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": vol}, index=idx))


def _fund(ticker, mcap, float_sh=2.0e8, shares_out=4.0e8):
    upsert_fundamentals([{"ticker": ticker, "market_cap": mcap,
                          "float_shares": float_sh, "shares_outstanding": shares_out}])


def test_today_vs_avg_turnover_differ_on_a_spike(tmp_db):
    # A 5× volume blast today: today's turnover should sit well above the
    # 60-day average, and today_vs_avg_value_mult should flag the spike.
    _seed("SPK.NS", 500, 1_000_000, last_vol_mult=5.0, seed=2)
    _fund("SPK.NS", mcap=2.0e11)
    m = mm.for_ticker("SPK.NS")
    assert m is not None
    assert m.today_turnover_pct > m.avg_turnover_pct          # today is hotter than usual
    assert m.today_vs_avg_value_mult > 3                       # ~5× a normal day
    # Both are real percentages of the same market cap — just different windows.
    assert m.today_turnover_pct > 0 and m.avg_turnover_pct > 0


def test_average_block_matches_liquidity_exactly(tmp_db):
    # The delegation must not change liquidity's numbers.
    _seed("X.NS", 100, 1_000_000, seed=3)
    _fund("X.NS", mcap=1.0e11, float_sh=5.0e8, shares_out=1.0e9)
    p = liq.liquidity_profile("X.NS")
    m = mm.for_ticker("X.NS", allow_derive=False)
    assert round(p.turnover_pct, 3) == round(m.avg_turnover_pct, 3)
    assert round(p.float_turnover_pct, 3) == round(m.avg_float_turnover_pct, 3)
    assert round(p.adv_value_cr, 2) == round(m.adv_value_cr, 2)
    assert p.float_basis == m.float_basis


def test_float_basis_fallback_and_derive(tmp_db):
    # No free float, no shares out, but a market cap + price → derive (when asked).
    _seed("DRV.NS", 100, 500_000, seed=4)
    _fund("DRV.NS", mcap=5.0e10, float_sh=None, shares_out=None)
    no_derive = mm.for_ticker("DRV.NS", allow_derive=False)
    assert no_derive.float_basis == "n/a"
    assert no_derive.avg_float_turnover_pct is None
    derived = mm.for_ticker("DRV.NS", allow_derive=True)
    assert derived.float_basis == "derived"
    assert derived.avg_float_turnover_pct is not None


def test_compute_returns_none_on_empty():
    assert mm.compute(pd.DataFrame(), {"market_cap": 1e9}) is None
