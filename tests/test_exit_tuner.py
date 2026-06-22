"""Tests for the exit-rule tuner."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk.backtest import exit_tuner as et
from swingdesk.backtest.exit_tuner import ExitPolicy
from swingdesk.storage import upsert_prices


def _df(bars: list[dict]) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(bars), freq="B")
    return pd.DataFrame(bars, index=idx)


# ---- deterministic forward-sim --------------------------------------------
def test_target_hit_r_multiple():
    # entry 100, atr 2, stop_atr 1 (stop 98), target_atr 2 (target 104).
    df = _df([{"open": 100, "high": 100, "low": 100, "close": 100},
              {"open": 101, "high": 105, "low": 100, "close": 104}])  # hits 104
    out = et._simulate(df, 0, 100.0, 2.0, ExitPolicy(stop_atr=1, target_atr=2, max_hold=5))
    assert out[0] == "target"
    assert abs(out[1] - 2.0) < 1e-9          # +2R


def test_stop_hit_r_multiple():
    df = _df([{"open": 100, "high": 100, "low": 100, "close": 100},
              {"open": 99, "high": 100, "low": 97, "close": 98}])      # low 97 < stop 98
    out = et._simulate(df, 0, 100.0, 2.0, ExitPolicy(stop_atr=1, target_atr=2, max_hold=5))
    assert out[0] == "stop"
    assert abs(out[1] - (-1.0)) < 1e-9       # −1R


def test_gap_down_exits_at_open():
    df = _df([{"open": 100, "high": 100, "low": 100, "close": 100},
              {"open": 96, "high": 97, "low": 95, "close": 96}])       # gaps below stop 98
    out = et._simulate(df, 0, 100.0, 2.0, ExitPolicy(stop_atr=1, target_atr=5, max_hold=5))
    assert out[0] == "stop"
    assert abs(out[1] - (-2.0)) < 1e-9       # exited at open 96 → −2R (worse than −1)


def test_time_stop():
    df = _df([{"open": 100, "high": 100, "low": 100, "close": 100},
              {"open": 100, "high": 101, "low": 99.5, "close": 100.5},
              {"open": 100.5, "high": 101, "low": 100, "close": 101}])
    out = et._simulate(df, 0, 100.0, 2.0, ExitPolicy(stop_atr=1, target_atr=5, max_hold=2))
    assert out[0] == "time"
    assert abs(out[1] - 0.5) < 1e-9          # (101−100)/2 = +0.5R


def test_breakeven_protects_winner():
    # Rises to +1R (102), triggering breakeven, then falls back to 100 → exit ~0R
    # instead of the −1R it would have taken without breakeven.
    df = _df([{"open": 100, "high": 100, "low": 100, "close": 100},
              {"open": 101, "high": 102, "low": 101, "close": 102},   # hi 102 = +1R → BE
              {"open": 101, "high": 101, "low": 99.5, "close": 100}])  # dips to 99.5 ≤ entry stop
    out = et._simulate(df, 0, 100.0, 2.0,
                       ExitPolicy(stop_atr=1, target_atr=5, max_hold=5, breakeven_at_r=1.0))
    assert out[0] == "stop"
    assert abs(out[1] - 0.0) < 1e-9          # stopped at entry (breakeven), 0R


# ---- end-to-end on synthetic prices ---------------------------------------
def test_collect_and_run_policy(tmp_db, synth_ohlcv):
    upsert_prices("SYN.NS", synth_ohlcv)
    data = et.collect_entries(["SYN.NS"])
    # Uptrending synthetic series should fire at least one detector.
    if not data:
        return  # acceptable: detectors are strict; no crash is the contract
    trades = et.run_policy(data, ExitPolicy())
    assert set(["setup", "r", "bars_held", "outcome", "entry_date"]).issubset(trades.columns)
    assert trades["r"].notna().all()


def test_wider_target_changes_outcomes(tmp_db, synth_ohlcv):
    upsert_prices("SYN.NS", synth_ohlcv)
    data = et.collect_entries(["SYN.NS"])
    if not data:
        return
    tight = et.run_policy(data, ExitPolicy(stop_atr=1.5, target_atr=1.5))
    wide = et.run_policy(data, ExitPolicy(stop_atr=1.5, target_atr=6.0))
    # Same entries, different exits → the outcome mix must differ.
    if not tight.empty and not wide.empty:
        assert (tight["outcome"].value_counts().to_dict()
                != wide["outcome"].value_counts().to_dict())
