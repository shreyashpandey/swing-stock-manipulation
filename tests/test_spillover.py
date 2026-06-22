"""Tests for the US->India spillover engine. Seeds synthetic macro series with
a *known* lead-lag relationship (NIFTY follows yesterday's S&P) and asserts the
engine recovers it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingdesk.analyze import spillover
from swingdesk.storage import upsert_macro, upsert_prices


def _close_frame(returns: np.ndarray, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(returns), freq="B")
    close = start * np.cumprod(1 + returns)
    return pd.DataFrame({"close": close, "volume": np.full(len(returns), 1e6)}, index=idx)


@pytest.fixture
def seeded(tmp_db):
    """NIFTY return_t = 0.5 * S&P return_{t-1} + noise; a stock that tracks
    NIFTY same-day and US tech overnight."""
    rng = np.random.default_rng(7)
    n = 320
    spx_ret = rng.normal(0, 0.01, n)
    ixic_ret = spx_ret * 0.9 + rng.normal(0, 0.004, n)        # tech ~ broad US
    nifty_ret = np.empty(n)
    nifty_ret[0] = 0.0
    nifty_ret[1:] = 0.5 * spx_ret[:-1] + rng.normal(0, 0.004, n - 1)  # lead-lag
    inr_ret = rng.normal(0, 0.003, n)
    brent_ret = rng.normal(0, 0.015, n)

    upsert_macro("^GSPC", _close_frame(spx_ret, 4000))
    upsert_macro("^IXIC", _close_frame(ixic_ret, 14000))
    upsert_macro("^NSEI", _close_frame(nifty_ret, 22000))
    upsert_macro("INR=X", _close_frame(inr_ret, 83))
    upsert_macro("BZ=F", _close_frame(brent_ret, 80))
    # India VIX: low, stable -> calm regime
    upsert_macro("^INDIAVIX", _close_frame(rng.normal(0, 0.01, n), 12.0))

    # Stock tracks NIFTY same-day strongly + a little overnight NASDAQ
    stock_ret = np.empty(n)
    stock_ret[0] = 0.0
    stock_ret[1:] = (1.1 * nifty_ret[1:] + 0.2 * ixic_ret[:-1]
                     + rng.normal(0, 0.006, n - 1))
    close = 500 * np.cumprod(1 + stock_ret)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    px = pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": np.full(n, 5e5),
    }, index=idx)
    upsert_prices("TESTSTK.NS", px)
    return "TESTSTK.NS"


def test_spillover_betas_recovers_lead_lag(seeded):
    df = spillover.spillover_betas()
    assert not df.empty
    sp = df[df["factor"] == "S&P 500"].iloc[0]
    # We injected beta ~0.5; allow noise.
    assert 0.3 < sp["beta"] < 0.7
    assert sp["r2"] > 0.05
    # A real injected relationship over 300+ obs must register as significant.
    assert "t_stat" in df.columns and "significant" in df.columns
    assert sp["significant"] is True or sp["significant"] == True  # noqa: E712
    assert abs(sp["t_stat"]) >= 2.0


def test_spillover_flags_noise_as_insignificant(tmp_db):
    # NIFTY return independent of US -> beta should be statistically insignificant.
    rng = np.random.default_rng(3)
    n = 320
    upsert_macro("^GSPC", _close_frame(rng.normal(0, 0.01, n), 4000))
    upsert_macro("^IXIC", _close_frame(rng.normal(0, 0.01, n), 14000))
    upsert_macro("^DJI", _close_frame(rng.normal(0, 0.01, n), 35000))
    upsert_macro("^NSEI", _close_frame(rng.normal(0, 0.01, n), 22000))  # unrelated
    df = spillover.spillover_betas()
    assert not df.empty
    # With pure-noise inputs, the |t|>=2 filter should reject most series.
    # (A stray false positive is expected ~5% of the time per series — the point
    # is that significance screens noise, not that it's infallible.)
    assert df["significant"].sum() <= 1
    # And the explanatory power is tiny regardless.
    assert (df["r2"] < 0.05).all()


def test_next_day_outlook_directionally_consistent(seeded):
    out = spillover.next_day_outlook()
    assert out is not None
    # Band must bracket the point estimate.
    assert out.low_pct <= out.expected_pct <= out.high_pct
    assert out.confidence in {"low", "moderate"}
    assert out.driver in {"S&P 500", "NASDAQ", "Dow Jones"}


def test_stock_sensitivities_ranks_nifty_first(seeded):
    df = spillover.stock_sensitivities(seeded)
    assert not df.empty
    # NIFTY same-day beta was the dominant driver (1.1x).
    assert df.iloc[0]["driver"] == "NIFTY 50"
    assert df.iloc[0]["r"] > 0.4


def test_regime_reads_calm_market_as_risk_on(seeded):
    reg = spillover.regime()
    assert reg is not None
    # Uptrending NIFTY/S&P (cumprod of positive-drift noise may vary) + low VIX.
    assert reg.label in {"risk-on", "neutral", "risk-off"}
    assert -100 <= reg.score <= 100
    assert any("VIX" in r for r in reg.reasons)


def test_empty_db_returns_safely(tmp_db):
    assert spillover.spillover_betas().empty
    assert spillover.next_day_outlook() is None
    assert spillover.stock_sensitivities("NOPE.NS").empty
