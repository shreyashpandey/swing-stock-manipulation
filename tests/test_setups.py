"""Tests for each setup detector.

Strategy: hand-craft small DataFrames that *should* and *should not* trigger
each setup, then run it through `add_indicators` + the detector and assert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingdesk.analyze import setups
from swingdesk.analyze.technicals import add_indicators


def _frame(closes, vols=None, opens=None, highs=None, lows=None) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.asarray(closes, dtype=float)
    opens = np.asarray(opens if opens is not None else closes - 0.1, dtype=float)
    highs = np.asarray(highs if highs is not None else np.maximum(opens, closes) + 0.5, dtype=float)
    lows = np.asarray(lows if lows is not None else np.minimum(opens, closes) - 0.5, dtype=float)
    vols = np.asarray(vols if vols is not None else np.full(n, 500_000.0), dtype=float)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


# ---- breakout_20d --------------------------------------------------------------

def test_breakout_triggers():
    # 80 bars uptrend, then a fresh 20d-high close on a volume surge.
    n = 80
    closes = np.linspace(100, 140, n).tolist()
    closes[-1] = closes[-2] + 5  # decisive new high
    vols = [500_000] * n
    vols[-1] = 2_000_000  # ~4x avg
    df = _frame(closes, vols=vols)
    df = add_indicators(df)
    sig = setups.detect_breakout(df, "X.NS")
    assert sig is not None
    assert sig["setup"] == "breakout_20d"
    assert sig["entry"] > sig["stoploss"]
    assert sig["target"] > sig["entry"]
    assert sig["rr"] > 1.0


def test_breakout_skips_low_volume():
    n = 80
    closes = np.linspace(100, 140, n).tolist()
    closes[-1] = closes[-2] + 5
    df = _frame(closes, vols=[500_000] * n)  # flat volume — no surge
    df = add_indicators(df)
    assert setups.detect_breakout(df, "X.NS") is None


def test_breakout_skips_downtrend():
    n = 80
    closes = np.linspace(140, 100, n).tolist()  # downtrend, below 50EMA
    df = _frame(closes, vols=[500_000] * (n - 1) + [2_000_000])
    df = add_indicators(df)
    assert setups.detect_breakout(df, "X.NS") is None


def test_breakout_needs_enough_history():
    df = _frame([100] * 20)
    df = add_indicators(df)
    assert setups.detect_breakout(df, "X.NS") is None


# ---- ema_20_50_cross -----------------------------------------------------------

def test_ema_cross_triggers():
    # Long flat-low base, then sharp rise — guarantees 20EMA crosses above 50EMA.
    n = 120
    closes = [100.0] * 60 + list(np.linspace(100, 150, 60))
    df = _frame(closes)
    df = add_indicators(df)
    # Walk forward to find the cross day, then trim — detector looks at last bar.
    crossed = None
    for i in range(60, n):
        if pd.notna(df["ema20"].iloc[i]) and pd.notna(df["ema50"].iloc[i]):
            if df["ema20"].iloc[i-1] <= df["ema50"].iloc[i-1] and df["ema20"].iloc[i] > df["ema50"].iloc[i]:
                crossed = i
                break
    assert crossed is not None, "test fixture failed to produce a cross"
    sig = setups.detect_ma_cross(df.iloc[: crossed + 1], "X.NS")
    assert sig is not None
    assert sig["setup"] == "ema_20_50_cross"


def test_ema_cross_no_signal_in_steady_trend():
    df = _frame(np.linspace(100, 200, 120))  # already trending, no fresh cross at end
    df = add_indicators(df)
    assert setups.detect_ma_cross(df, "X.NS") is None


# ---- volume_thrust -------------------------------------------------------------

def test_volume_thrust_triggers():
    n = 40
    closes = [100.0] * (n - 1) + [102.0]  # up day at the end
    opens = [100.0] * n
    vols = [500_000] * (n - 1) + [2_000_000]  # 4x
    df = _frame(closes, vols=vols, opens=opens)
    df = add_indicators(df)
    sig = setups.detect_volume_thrust(df, "X.NS")
    assert sig is not None
    assert sig["setup"] == "volume_thrust"


def test_volume_thrust_skips_down_day():
    n = 40
    closes = [100.0] * (n - 1) + [99.0]  # down day
    vols = [500_000] * (n - 1) + [3_000_000]
    df = _frame(closes, vols=vols, opens=[100.0] * n)
    df = add_indicators(df)
    assert setups.detect_volume_thrust(df, "X.NS") is None


# ---- pullback_ema20 ------------------------------------------------------------

def test_pullback_returns_dict_or_none(synth_ohlcv):
    """Pullback is harder to hand-craft; just confirm it never crashes and shape is right."""
    df = add_indicators(synth_ohlcv)
    out = setups.detect_pullback_to_ema(df, "X.NS")
    assert out is None or {"entry", "stoploss", "target", "rr", "setup"} <= out.keys()


# ---- new strategy detectors ----------------------------------------------------

def _scan_history(detector, df):
    """Run one detector bar-by-bar (like chart_signals) and return every signal
    it produced — for cross/flip detectors whose trigger lands on a given bar."""
    out = []
    for i in range(60, len(df)):
        sig = detector(df.iloc[: i + 1], "X.NS")
        if sig:
            out.append(sig)
    return out


def test_detectors_registry_well_formed():
    assert len(setups.DETECTORS) == 10
    names = {fn.__name__ for fn in setups.DETECTORS}
    assert len(names) == 10
    assert all(callable(fn) for fn in setups.DETECTORS)


def test_golden_cross_triggers():
    # Flat well past bar 210 (so the detector's 210-bar warmup is satisfied),
    # THEN a ramp so the 50-EMA crosses the 200-EMA after warmup completes.
    n = 280
    closes = ([100.0] * 216) + np.linspace(100, 175, n - 216).tolist()
    df = add_indicators(_frame(closes))
    sigs = _scan_history(setups.detect_golden_cross, df)
    assert sigs, "golden cross should fire on a flat→ramp series"
    s = sigs[0]
    assert s["setup"] == "golden_cross"
    assert s["target"] > s["entry"] > s["stoploss"]
    assert s["rr"] > 1.0


def test_new_detectors_return_valid_shape():
    # Whatever fires on a long trending series must carry a complete bracket.
    n = 260
    closes = np.linspace(100, 180, n).tolist()
    vols = [500_000.0] * n
    vols[-1] = 1_500_000.0
    df = add_indicators(_frame(closes, vols=vols))
    required = {"ticker", "setup", "entry", "stoploss", "target", "rr", "score", "notes"}
    for fn in (setups.detect_macd_cross, setups.detect_supertrend_flip,
               setups.detect_bollinger_breakout, setups.detect_rsi_reversal,
               setups.detect_adx_trend):
        sig = fn(df, "X.NS")
        if sig:
            assert required <= sig.keys()
            assert sig["entry"] > sig["stoploss"]
            assert sig["target"] > sig["entry"]


# ---- scan_ticker integration ---------------------------------------------------

def test_scan_ticker_runs(tmp_db, synth_ohlcv):
    from swingdesk import storage
    storage.upsert_prices("SYNTH.NS", synth_ohlcv)
    sigs = setups.scan_ticker("SYNTH.NS")
    assert isinstance(sigs, list)
    for s in sigs:
        assert {"ticker", "setup", "entry", "stoploss", "target", "rr", "score"} <= s.keys()


def test_scan_all_persists(tmp_db, synth_ohlcv):
    from swingdesk import storage
    storage.upsert_prices("A.NS", synth_ohlcv)
    storage.upsert_prices("B.NS", synth_ohlcv)
    setups.scan_all(["A.NS", "B.NS"], persist=True)
    df = storage.load_signals(limit=50)
    # Synthetic data may or may not produce signals — main assertion: no crash & shape ok.
    assert isinstance(df, pd.DataFrame)


def test_uptrend_gate_blocks_counter_trend(tmp_db):
    """A clear downtrend (price well below its 200-EMA) emits no signals with the
    trend gate on, but the gate can be disabled."""
    from swingdesk import storage
    n = 260
    closes = np.linspace(300, 120, n)            # persistent downtrend
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "open": closes + 0.1, "high": closes + 1.0,
        "low": closes - 1.0, "close": closes,
        "volume": np.full(n, 8e5),
    }, index=idx)
    storage.upsert_prices("DOWN.NS", df)
    assert setups.scan_ticker("DOWN.NS", require_uptrend=True) == []
    # With the gate off, detectors are free to fire (or not) — just no gating.
    gated_off = setups.scan_ticker("DOWN.NS", require_uptrend=False)
    assert isinstance(gated_off, list)
