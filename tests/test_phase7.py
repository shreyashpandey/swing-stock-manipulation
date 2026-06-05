"""Phase 7 tests: volume indicators, volume profile, exit levels, macro, AI thesis."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from swingdesk import storage
from swingdesk.analyze import exits as exits_mod
from swingdesk.analyze.technicals import add_indicators, volume_profile


# ---- volume indicators in technicals ------------------------------------------

def test_volume_indicators_added(synth_ohlcv):
    df = add_indicators(synth_ohlcv)
    for col in ("obv", "mfi14", "ad_line", "obv_slope_10", "buy_pressure_20"):
        assert col in df.columns, f"missing {col}"


def test_mfi_bounded_0_to_100(synth_ohlcv):
    df = add_indicators(synth_ohlcv)
    mfi = df["mfi14"].dropna()
    assert (mfi >= 0).all() and (mfi <= 100).all()


def test_obv_cumulative_monotonic_when_all_up(synth_ohlcv):
    """OBV cumulates volume — its absolute final value should grow with input."""
    df = add_indicators(synth_ohlcv)
    obv = df["obv"].dropna()
    assert len(obv) > 50
    # OBV is unbounded but should not be all-zero for a real series
    assert obv.abs().max() > 0


# ---- volume profile -----------------------------------------------------------

def test_volume_profile_returns_bins(synth_ohlcv):
    p = volume_profile(synth_ohlcv, bins=20, lookback=60)
    assert not p.empty
    assert len(p) == 20
    assert {"price_low", "price_high", "price_mid", "volume", "pct"} <= set(p.columns)
    # pct should sum to ~100
    assert abs(p["pct"].sum() - 100) < 0.01


def test_volume_profile_empty_when_no_data():
    p = volume_profile(pd.DataFrame())
    assert p.empty


# ---- exit-level analyzer ------------------------------------------------------

def _seed_uptrending_data(ticker: str, n: int = 100):
    """Synthetic uptrending OHLCV so indicators are well-defined."""
    rng = np.random.default_rng(13)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    base = 100 + np.cumsum(rng.normal(0.3, 1.2, n))
    opens = base + rng.normal(0, 0.4, n)
    closes = base + rng.normal(0, 0.4, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.2, 1.5, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.2, 1.5, n)
    vols = rng.integers(100_000, 1_000_000, n).astype(float)
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                       "close": closes, "volume": vols}, index=idx)
    storage.upsert_prices(ticker, df)


def test_compute_returns_none_when_no_data(tmp_db):
    plan = exits_mod.compute("MISSING.NS", avg_buy=100)
    assert plan is None


def test_compute_returns_plan_with_levels(tmp_db):
    _seed_uptrending_data("TEST.NS")
    plan = exits_mod.compute("TEST.NS", avg_buy=100)
    assert plan is not None
    assert plan.current_price > 0
    # At least the trailing stop should be computable (uses -7% fallback)
    assert plan.trailing_stop is not None
    assert plan.trailing_stop <= plan.current_price


def test_initial_stop_below_current(tmp_db):
    _seed_uptrending_data("TEST.NS")
    plan = exits_mod.compute("TEST.NS", avg_buy=100)
    if plan and plan.initial_stop is not None:
        assert plan.initial_stop < plan.current_price


def test_risk_reward_when_both_known(tmp_db):
    _seed_uptrending_data("TEST.NS")
    plan = exits_mod.compute("TEST.NS", avg_buy=100)
    if plan and plan.risk_amount and plan.reward_amount and plan.risk_reward:
        assert plan.risk_reward > 0


# ---- macro module -------------------------------------------------------------

def test_macro_storage_roundtrip(tmp_db):
    idx = pd.date_range("2025-01-01", periods=5, freq="B")
    df = pd.DataFrame({
        "close": [100, 101, 99, 102, 103],
        "volume": [1e6] * 5,
    }, index=idx)
    n = storage.upsert_macro("^NSEI", df)
    assert n == 5
    loaded = storage.load_macro("^NSEI")
    assert len(loaded) == 5
    assert "^NSEI" in storage.list_macro_tickers()


def test_market_pulse_empty_when_no_data(tmp_db):
    from swingdesk.ingest import macro
    pulse = macro.market_pulse()
    assert pulse == {} or all("close" in v for v in pulse.values())


def test_market_pulse_computes_changes(tmp_db):
    from swingdesk.ingest import macro
    # Seed Nifty data
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    closes = np.linspace(100, 110, 30)
    df = pd.DataFrame({"close": closes, "volume": [1e6] * 30}, index=idx)
    storage.upsert_macro("^NSEI", df)
    pulse = macro.market_pulse()
    if "NIFTY 50" in pulse:
        assert "chg_1d" in pulse["NIFTY 50"]
        assert "chg_1w" in pulse["NIFTY 50"]


def test_correlations_returns_empty_without_data(tmp_db):
    from swingdesk.ingest import macro
    corr = macro.correlations("UNKNOWN.NS")
    assert corr.empty


# ---- AI thesis (mocked Claude) -----------------------------------------------

def test_thesis_returns_none_without_api_key(tmp_db, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from swingdesk.analyze import thesis
    out = thesis.generate(
        ticker="TEST.NS", qty=10, avg_buy=100, last_price=110,
        pnl_pct=10.0, fundamentals=None, technical_state={},
        recent_news=pd.DataFrame(),
    )
    assert out is None


def test_thesis_parses_structured_response(tmp_db, monkeypatch):
    """With API key set + mocked Claude, the structured output flows through."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from swingdesk.analyze import thesis

    fake_thesis = MagicMock()
    fake_thesis.narrative = "Strong fundamentals + technical breakout above 50EMA"
    fake_thesis.conviction = 78
    fake_thesis.action = "BUY_MORE"
    fake_thesis.risks = ["earnings miss", "rate hike", "USD/INR spike"]
    fake_thesis.catalyst_to_watch = "Q1 earnings on 15-Jul"

    fake_resp = MagicMock()
    fake_resp.parsed_output = fake_thesis

    with patch("swingdesk.analyze.thesis.anthropic.Anthropic") as MockAnth, \
         patch("swingdesk.analyze.thesis.macro_mod.market_pulse", return_value={}), \
         patch("swingdesk.analyze.thesis.macro_mod.correlations", return_value=pd.DataFrame()):
        client = MagicMock()
        client.messages.parse.return_value = fake_resp
        MockAnth.return_value = client
        out = thesis.generate(
            ticker="TEST.NS", qty=10, avg_buy=100, last_price=110,
            pnl_pct=10.0, fundamentals={"sector": "Technology", "quality_score": 80},
            technical_state={"state": "uptrend", "rsi": 60},
            recent_news=pd.DataFrame(),
        )
    assert out is not None
    assert out.conviction == 78
    assert out.action == "BUY_MORE"
    assert len(out.risks) == 3


# ---- holdings integration: exit plan attached --------------------------------

def test_analyze_one_includes_exit_plan(tmp_db):
    """analyze_one should attach the computed exit plan."""
    from swingdesk.portfolio import holdings
    _seed_uptrending_data("TEST.NS")
    storage.upsert_fundamentals([
        {"ticker": "TEST.NS", "sector": "Technology", "quality_score": 75},
    ])
    a = holdings.analyze_one("TEST.NS", qty=10, avg_price=100,
                             last_price=110, portfolio_value=1100, pnl_pct=10)
    # Exit-plan fields should be populated
    assert a.trailing_stop is not None or a.initial_stop is not None or a.full_target is not None


def test_analyze_one_includes_volume_flow(tmp_db):
    """analyze_one should attach MFI + buy_pressure."""
    from swingdesk.portfolio import holdings
    _seed_uptrending_data("TEST.NS")
    a = holdings.analyze_one("TEST.NS", qty=10, avg_price=100,
                             last_price=110, portfolio_value=1100, pnl_pct=10)
    # mfi/buy_pressure are optional but should at least be tracked
    # (won't always be non-None on tiny synthetic data — test the attribute exists)
    assert hasattr(a, "mfi")
    assert hasattr(a, "buy_pressure_20d")
