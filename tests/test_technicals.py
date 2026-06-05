from __future__ import annotations

import pandas as pd

from swingdesk.analyze.technicals import add_indicators


def test_indicators_added(synth_ohlcv):
    out = add_indicators(synth_ohlcv)
    for col in ["ema20", "ema50", "ema200", "rsi14", "atr14",
                "vol_avg20", "high20", "low20", "high55",
                "macd", "macd_signal", "macd_hist"]:
        assert col in out.columns, f"missing indicator: {col}"


def test_indicators_handle_empty():
    out = add_indicators(pd.DataFrame())
    assert out.empty


def test_indicators_have_warmup_nans(synth_ohlcv):
    """First N bars should be NaN for long-window indicators."""
    out = add_indicators(synth_ohlcv)
    assert pd.isna(out["ema200"].iloc[0])
    assert pd.notna(out["ema20"].iloc[-1])
    assert pd.notna(out["ema200"].iloc[-1])


def test_rsi_bounded(synth_ohlcv):
    out = add_indicators(synth_ohlcv)
    rsi = out["rsi14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()
