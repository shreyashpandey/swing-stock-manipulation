"""Tests for the per-signal fused analysis (Rank · Range · Risk · Global)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk.analyze import signal_analysis as sa
from swingdesk.storage import upsert_macro, upsert_prices


def _close_frame(rets, start=100.0):
    idx = pd.date_range("2024-01-01", periods=len(rets), freq="B")
    close = start * np.cumprod(1 + rets)
    return pd.DataFrame({"close": close, "volume": np.full(len(rets), 1e6)}, index=idx)


def _seed_stock(ticker, drift, vol, seed, n=300, start=500.0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start * np.cumprod(1 + rets)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    upsert_prices(ticker, pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full(n, 5e5)}, index=idx))


def _seed_macro(n=300):
    rng = np.random.default_rng(1)
    upsert_macro("^NSEI", _close_frame(rng.normal(0, 0.01, n), 22000))
    upsert_macro("^IXIC", _close_frame(rng.normal(0, 0.012, n), 14000))


def _signals():
    return [
        {"ticker": "AAA.NS", "setup": "breakout_20d", "direction": "long",
         "entry": 500.0, "stoploss": 485.0, "target": 530.0, "rr": 2.0},
        {"ticker": "BBB.NS", "setup": "pullback_ema20", "direction": "long",
         "entry": 500.0, "stoploss": 490.0, "target": 525.0, "rr": 2.5},
        {"ticker": "CCC.NS", "setup": "macd_cross", "direction": "long",
         "entry": 500.0, "stoploss": 480.0, "target": 540.0, "rr": 2.0},
    ]


def _seed_all(tmp_db):
    _seed_macro()
    _seed_stock("AAA.NS", 0.003, 0.015, 1)
    _seed_stock("BBB.NS", 0.0, 0.02, 2)
    _seed_stock("CCC.NS", -0.001, 0.03, 3)


def test_analyze_signals_columns_and_one_row_each(tmp_db):
    _seed_all(tmp_db)
    df = sa.analyze_signals(_signals(), capital=100_000, risk_pct=1.0, mc_sims=1500)
    assert len(df) == 3
    for c in ["ticker", "setup", "rank_q", "rank_score", "p_target_first",
              "exp_R", "exp_move_10d_pct", "shares", "pct_cap", "risk_amt",
              "beta_nifty", "beta_nasdaq"]:
        assert c in df.columns


def test_risk_sizing_is_sensible(tmp_db):
    _seed_all(tmp_db)
    df = sa.analyze_signals(_signals(), capital=100_000, risk_pct=1.0).set_index("ticker")
    # AAA: 1% risk wants 66 sh (₹33k), but the 25%-of-capital cap binds -> 50 sh.
    assert df.loc["AAA.NS", "shares"] == 50
    assert df.loc["AAA.NS", "pct_cap"] == 25.0
    assert abs(df.loc["AAA.NS", "risk_amt"] - 50 * 15) < 1e-6   # capped risk
    assert (df["pct_cap"] > 0).all()


def test_range_and_global_populated(tmp_db):
    _seed_all(tmp_db)
    df = sa.analyze_signals(_signals(), mc_sims=1500).set_index("ticker")
    # Barrier probabilities are valid fractions; betas computed vs NIFTY.
    assert df["p_target_first"].dropna().between(0, 1).all()
    assert df["beta_nifty"].notna().any()


def test_empty_signals(tmp_db):
    assert sa.analyze_signals([]).empty


def test_sorted_by_expectancy(tmp_db):
    _seed_all(tmp_db)
    df = sa.analyze_signals(_signals(), mc_sims=1500)
    er = df["exp_R"].dropna()
    # Output is sorted by exp_R descending (NaNs last).
    assert list(er) == sorted(er, reverse=True)
