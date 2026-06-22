"""Tests for intraday VWAP / ORB / relative-volume signal detection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk.analyze import intraday
from swingdesk.storage import load_intraday, upsert_intraday


def _session_bars(day: str, closes, vols):
    idx = pd.date_range(f"{day} 09:15", periods=len(closes), freq="5min")
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": closes,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": np.asarray(vols, dtype=float),
    }, index=idx)


def _seed(ticker, sessions):
    df = pd.concat([_session_bars(d, c, v) for d, c, v in sessions])
    upsert_intraday(ticker, df, "5m")


def test_upsert_and_load_roundtrip(tmp_db):
    _seed("X.NS", [("2026-06-10", [100, 101, 102], [1e5, 1e5, 1e5])])
    df = load_intraday("X.NS", "5m")
    assert len(df) == 3
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_vwap_resets_each_session(tmp_db):
    _seed("X.NS", [
        ("2026-06-10", [100, 100, 100], [1e5, 1e5, 1e5]),
        ("2026-06-11", [200, 200, 200], [1e5, 1e5, 1e5]),
    ])
    df = load_intraday("X.NS", "5m")
    vw = intraday.vwap(df)
    # First bar of each session: VWAP == that bar's typical price (~its close).
    day2 = df[df.index.normalize() == pd.Timestamp("2026-06-11")]
    assert abs(vw.loc[day2.index[0]] - 200) < 1.0   # not dragged by day-1's 100s


def test_opening_range(tmp_db):
    sess = _session_bars("2026-06-10", [100, 102, 101, 105, 106], [1e5] * 5)
    orr = intraday.opening_range(sess, or_bars=3)
    assert orr["or_high"] == 102 + 0.5     # max high of first 3 bars
    assert orr["or_low"] == 100 - 0.5      # min low of first 3 bars


def test_orb_long_detected(tmp_db):
    # Latest session: first 3 bars range ~100, then breaks out to 106 on volume.
    _seed("BREAK.NS", [
        ("2026-06-10", [100] * 6, [1e5] * 6),                       # prior day
        ("2026-06-11", [100, 100.5, 101, 103, 105, 106],
         [1e5, 1e5, 1e5, 3e5, 4e5, 5e5]),                          # breakout day
    ])
    sig = intraday.intraday_signals("BREAK.NS", or_bars=3)
    assert sig is not None
    assert sig.bias == "long"
    assert "ORB long" in sig.setup
    assert sig.last > sig.or_high
    assert sig.last >= sig.vwap


def test_orb_short_detected(tmp_db):
    _seed("DROP.NS", [
        ("2026-06-10", [100] * 6, [1e5] * 6),
        ("2026-06-11", [100, 99.5, 99, 97, 95, 94],
         [1e5, 1e5, 1e5, 3e5, 4e5, 5e5]),
    ])
    sig = intraday.intraday_signals("DROP.NS", or_bars=3)
    assert sig is not None
    assert sig.bias == "short"
    assert "ORB short" in sig.setup
    assert sig.last < sig.or_low


def test_gap_and_rvol_computed(tmp_db):
    _seed("GAP.NS", [
        ("2026-06-10", [100] * 6, [1e5] * 6),
        ("2026-06-11", [105, 105, 105, 105, 105, 110],
         [1e5, 1e5, 1e5, 1e5, 1e5, 6e5]),    # gap up ~5%, volume spike on last bar
    ])
    sig = intraday.intraday_signals("GAP.NS", or_bars=2)
    assert sig is not None
    assert sig.gap_pct > 4.0          # 105 open vs 100 prior close
    assert sig.rvol > 1.0             # last bar 6e5 vs median 1e5


def test_scan_orders_active_first(tmp_db):
    _seed("ACT.NS", [
        ("2026-06-10", [100] * 6, [1e5] * 6),
        ("2026-06-11", [100, 100.5, 101, 103, 105, 106],
         [1e5, 1e5, 1e5, 3e5, 4e5, 5e5]),
    ])
    _seed("FLAT.NS", [
        ("2026-06-10", [100] * 6, [1e5] * 6),
        ("2026-06-11", [100, 100, 100, 100, 100, 100], [1e5] * 6),
    ])
    df = intraday.scan(["FLAT.NS", "ACT.NS"])
    assert not df.empty
    # The active (long) setup should sort above the neutral one.
    assert df.iloc[0]["ticker"] == "ACT.NS"
    assert df.iloc[0]["bias"] == "long"


def test_no_data_returns_none(tmp_db):
    assert intraday.intraday_signals("NOPE.NS") is None
    assert intraday.scan(["NOPE.NS"]).empty
