from __future__ import annotations

import pandas as pd

from swingdesk.analyze.technicals import add_indicators, signal_scoreboard


def test_indicators_added(synth_ohlcv):
    out = add_indicators(synth_ohlcv)
    for col in ["ema20", "ema50", "ema200", "rsi14", "atr14",
                "vol_avg20", "high20", "low20", "high55",
                "macd", "macd_signal", "macd_hist"]:
        assert col in out.columns, f"missing indicator: {col}"


def test_new_indicators_added(synth_ohlcv):
    out = add_indicators(synth_ohlcv)
    for col in ["bb_lower", "bb_mid", "bb_upper", "bb_width", "bb_pct",
                "bb_width_min60", "adx14", "di_plus", "di_minus",
                "stoch_k", "stoch_d", "supertrend", "supertrend_dir", "cci20"]:
        assert col in out.columns, f"missing indicator: {col}"


def test_new_indicators_bounded(synth_ohlcv):
    out = add_indicators(synth_ohlcv)
    adx = out["adx14"].dropna()
    assert (adx >= 0).all() and (adx <= 100).all()
    for c in ("stoch_k", "stoch_d"):
        s = out[c].dropna()
        assert (s >= 0).all() and (s <= 100).all()
    assert set(out["supertrend_dir"].dropna().unique()) <= {1.0, -1.0}


def test_signal_scoreboard_shape(synth_ohlcv):
    sb = signal_scoreboard(add_indicators(synth_ohlcv))
    assert sb is not None
    assert sb["tilt"] in ("STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL")
    assert -100 <= sb["score"] <= 100
    assert isinstance(sb["bull"], list) and isinstance(sb["bear"], list)


def test_signal_scoreboard_short_history():
    assert signal_scoreboard(add_indicators(pd.DataFrame())) is None


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
