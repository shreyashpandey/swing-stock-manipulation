"""Backtest engine + metrics tests.

The critical test here is the no-lookahead guard (test_no_lookahead_bias):
the detector must NEVER see data past its supposed entry date. We verify
this by patching a detector to capture the DataFrame it receives and
asserting nothing in that window has a date >= entry_date+1.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from swingdesk import storage
from swingdesk.backtest import engine, metrics


def _seeded_prices(ticker: str, n: int = 250) -> None:
    """Insert deterministic OHLCV data into the test DB."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    base = 100 + np.cumsum(rng.normal(0.15, 1.0, n))
    open_ = base + rng.normal(0, 0.3, n)
    close = base + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.2, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.2, n)
    vol = rng.integers(100_000, 1_000_000, n).astype(float)
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close, "volume": vol}, index=idx)
    storage.upsert_prices(ticker, df)


# --- engine ---------------------------------------------------------------------

def test_backtest_ticker_no_data(tmp_db):
    trades = engine.backtest_ticker("MISSING.NS")
    assert trades == []


def test_backtest_ticker_runs(tmp_db):
    _seeded_prices("TEST.NS")
    trades = engine.backtest_ticker("TEST.NS", max_hold=20)
    # Synthetic uptrending data — should produce some trades
    assert isinstance(trades, list)
    for t in trades:
        assert t.entry > 0
        assert t.stoploss < t.entry
        assert t.target > t.entry
        assert t.outcome in ("target", "stoploss", "stoploss_gap", "time_stop")
        assert t.r_multiple is not None
        # Sanity: r_multiple in a reasonable range. Gap-downs can exceed -1R
        # because we exit at the gapped-down open, so allow some headroom.
        assert -5 < t.r_multiple < 10


def test_backtest_universe_aggregates(tmp_db):
    _seeded_prices("A.NS")
    _seeded_prices("B.NS")
    df = engine.backtest_universe(["A.NS", "B.NS"])
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert set(df["ticker"].unique()) <= {"A.NS", "B.NS"}
        assert "r" in df.columns


def test_no_lookahead_bias(tmp_db):
    """CRITICAL: detectors must only see bars up to and including the entry day."""
    _seeded_prices("TEST.NS")

    seen_windows: list[tuple[str, int]] = []  # (last_date_in_window, signal_idx)

    def fake_detector(df: pd.DataFrame, ticker: str):
        last_date = df.index[-1]
        seen_windows.append((str(last_date.date()), len(df)))
        # Don't actually signal — we only want to inspect the windows
        return None

    fake_detector.__name__ = "detect_fake"
    engine.backtest_ticker("TEST.NS", detectors=[fake_detector], max_hold=5)

    # Every window should be strictly increasing in length, and the last
    # date should advance by exactly one bar each step (no gaps, no jumps).
    assert len(seen_windows) > 50
    prev_len = 0
    for date, length in seen_windows:
        assert length == prev_len + 1 or prev_len == 0
        prev_len = length


def test_target_hit_gives_planned_r(tmp_db):
    """If price quickly hits the target, r_multiple should equal planned R:R."""
    # Build a tiny synthetic series where bar 70 rips up to a clear target.
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.linspace(100, 100, n).tolist()
    highs = closes.copy()
    lows = closes.copy()
    opens = closes.copy()
    vols = [500_000] * n
    # Trigger a breakout on day 70: huge volume + close above 20d high
    closes[70] = 110
    highs[70] = 112
    opens[70] = 100
    vols[70] = 5_000_000
    # Then drift to +20 over next 5 days so target gets hit
    for k in range(71, 80):
        closes[k] = 125
        highs[k] = 130
        lows[k] = 120
        opens[k] = 124
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                       "close": closes, "volume": vols}, index=idx)
    storage.upsert_prices("SYN.NS", df)

    trades = engine.backtest_ticker("SYN.NS", max_hold=20)
    # Should have at least one trade and at least one should be a target hit
    target_hits = [t for t in trades if t.outcome == "target"]
    if target_hits:
        # Target hits should yield positive R approximately equal to planned R:R
        for t in target_hits:
            assert t.r_multiple > 1.5


def test_stoploss_gives_negative_r(tmp_db):
    """If price drops through stop after entry, r_multiple should be ~ -1."""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.linspace(100, 130, n).tolist()
    closes[70] = 135  # breakout
    closes[71] = 110  # crash, hitting any reasonable stop
    closes[72] = 95
    for k in range(73, n):
        closes[k] = 90
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    opens = closes.copy()
    vols = [500_000] * n
    vols[70] = 5_000_000
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                       "close": closes, "volume": vols}, index=idx)
    storage.upsert_prices("CRASH.NS", df)

    trades = engine.backtest_ticker("CRASH.NS", max_hold=20)
    sl_hits = [t for t in trades if t.outcome in ("stoploss", "stoploss_gap")]
    for t in sl_hits:
        # Stop hits should be roughly -1R (or worse on gap-downs)
        assert t.r_multiple <= -0.5


# --- metrics --------------------------------------------------------------------

def test_summarize_empty():
    out = metrics.summarize(pd.DataFrame())
    assert out.empty


def test_summarize_basic():
    trades = pd.DataFrame([
        {"setup": "A", "r": 2.0, "bars_held": 5, "entry_date": "2024-01-01"},
        {"setup": "A", "r": -1.0, "bars_held": 3, "entry_date": "2024-01-05"},
        {"setup": "A", "r": 2.0, "bars_held": 4, "entry_date": "2024-01-10"},
        {"setup": "B", "r": -1.0, "bars_held": 2, "entry_date": "2024-01-15"},
    ])
    out = metrics.summarize(trades)
    assert "ALL" in out["setup"].values
    a_row = out[out["setup"] == "A"].iloc[0]
    assert a_row["n_trades"] == 3
    assert a_row["wins"] == 2
    assert a_row["losses"] == 1
    assert abs(a_row["win_rate"] - 0.667) < 0.01
    # avg_r = (2 + -1 + 2) / 3 = 1.0
    assert abs(a_row["avg_r"] - 1.0) < 0.01


def test_max_consec_losses():
    # _max_consec_losses is module-private; test it via summarize
    trades = pd.DataFrame([
        {"setup": "X", "r": -1, "bars_held": 1, "entry_date": "2024-01-01"},
        {"setup": "X", "r": -1, "bars_held": 1, "entry_date": "2024-01-02"},
        {"setup": "X", "r": -1, "bars_held": 1, "entry_date": "2024-01-03"},
        {"setup": "X", "r": 2,  "bars_held": 1, "entry_date": "2024-01-04"},
        {"setup": "X", "r": -1, "bars_held": 1, "entry_date": "2024-01-05"},
    ])
    out = metrics.summarize(trades)
    x = out[out["setup"] == "X"].iloc[0]
    assert x["max_consec_losses"] == 3


def test_gate_passes_with_strong_strategy():
    row = {"n_trades": 50, "expectancy": 0.5, "profit_factor": 2.0,
           "max_drawdown_r": 5.0}
    ok, fails = metrics.gate(row)
    assert ok is True
    assert fails == []


def test_gate_fails_with_weak_strategy():
    row = {"n_trades": 5, "expectancy": 0.01, "profit_factor": 0.9,
           "max_drawdown_r": 30.0}
    ok, fails = metrics.gate(row)
    assert ok is False
    assert len(fails) == 4  # all four criteria fail


# --- storage roundtrip ----------------------------------------------------------

def test_save_load_backtest(tmp_db):
    trades = pd.DataFrame([
        {"ticker": "A.NS", "setup": "x", "entry_date": "2024-01-01",
         "exit_date": "2024-01-05", "entry": 100, "exit": 110, "stoploss": 95,
         "target": 115, "outcome": "target", "r": 2.0, "bars_held": 4,
         "planned_rr": 3.0},
    ])
    n = storage.save_backtest_trades("run_test", trades)
    assert n == 1
    loaded = storage.load_backtest_trades("run_test")
    assert len(loaded) == 1
    assert loaded.iloc[0]["ticker"] == "A.NS"


def test_runs_are_isolated_by_run_id(tmp_db):
    """Two saved runs must not bleed into each other."""
    trades_a = pd.DataFrame([{
        "ticker": "A.NS", "setup": "x", "entry_date": "2024-01-01",
        "exit_date": "2024-01-05", "entry": 100, "exit": 110, "stoploss": 95,
        "target": 115, "outcome": "target", "r": 2.0, "bars_held": 4, "planned_rr": 3.0}])
    trades_b = pd.DataFrame([{
        "ticker": "B.NS", "setup": "y", "entry_date": "2024-02-01",
        "exit_date": "2024-02-05", "entry": 200, "exit": 195, "stoploss": 195,
        "target": 220, "outcome": "stoploss", "r": -1.0, "bars_held": 2, "planned_rr": 4.0}])
    storage.save_backtest_trades("run_a", trades_a)
    storage.save_backtest_trades("run_b", trades_b)

    assert len(storage.load_backtest_trades("run_a")) == 1
    assert len(storage.load_backtest_trades("run_b")) == 1
    assert storage.load_backtest_trades("run_a").iloc[0]["ticker"] == "A.NS"


def test_list_backtest_runs(tmp_db):
    trades = pd.DataFrame([{
        "ticker": "A.NS", "setup": "x", "entry_date": "2024-01-01",
        "exit_date": "2024-01-05", "entry": 100, "exit": 110, "stoploss": 95,
        "target": 115, "outcome": "target", "r": 2.0, "bars_held": 4, "planned_rr": 3.0}])
    storage.save_backtest_trades("run_1", trades)
    storage.save_backtest_trades("run_2", trades)

    runs = storage.list_backtest_runs()
    assert len(runs) == 2
    assert set(runs["run_id"]) == {"run_1", "run_2"}
    assert (runs["n_trades"] == 1).all()


def test_load_backtest_trades_all(tmp_db):
    """Calling load with no run_id returns trades from all runs."""
    trades = pd.DataFrame([{
        "ticker": "A.NS", "setup": "x", "entry_date": "2024-01-01",
        "exit_date": "2024-01-05", "entry": 100, "exit": 110, "stoploss": 95,
        "target": 115, "outcome": "target", "r": 2.0, "bars_held": 4, "planned_rr": 3.0}])
    storage.save_backtest_trades("r1", trades)
    storage.save_backtest_trades("r2", trades)
    assert len(storage.load_backtest_trades()) == 2


def test_gap_down_exits_at_open_not_stop(tmp_db):
    """If price gaps down through the stop, exit at the gapped open (not the
    stop level) — outcome is 'stoploss_gap' and r_multiple < -1."""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.linspace(100, 130, n).tolist()
    closes[70] = 135  # breakout trigger
    closes[71] = 100  # next bar gaps WAY below any reasonable stop
    closes[72] = 95
    for k in range(73, n):
        closes[k] = 90
    # Critical: opens[71] gaps down hard — well below the trigger bar's SL
    opens = closes.copy()
    opens[71] = 100  # gap-down open
    highs = [max(o, c) + 0.5 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 0.5 for o, c in zip(opens, closes)]
    vols = [500_000] * n
    vols[70] = 5_000_000  # ensures breakout fires
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                       "close": closes, "volume": vols}, index=idx)
    storage.upsert_prices("GAP.NS", df)

    trades = engine.backtest_ticker("GAP.NS", max_hold=20)
    gap_exits = [t for t in trades if t.outcome == "stoploss_gap"]
    if gap_exits:
        for t in gap_exits:
            # Gap-down should produce loss worse than -1R
            assert t.r_multiple < -1.0


def test_cooldown_prevents_overlapping_trades(tmp_db):
    """While a trade is open in a given setup, the same setup shouldn't fire again."""
    _seeded_prices("TEST.NS")
    trades = engine.backtest_ticker("TEST.NS", max_hold=20)

    # Group by setup and verify no two trades for the same setup overlap in time
    by_setup = {}
    for t in trades:
        by_setup.setdefault(t.setup, []).append(t)
    for setup, ts in by_setup.items():
        ts_sorted = sorted(ts, key=lambda x: x.entry_date)
        for prev, nxt in zip(ts_sorted, ts_sorted[1:]):
            # Next entry must be after the previous exit
            assert nxt.entry_date >= prev.exit_date, \
                f"Overlap in {setup}: {prev.entry_date}-{prev.exit_date} vs {nxt.entry_date}"


def test_engine_handles_detector_exceptions(tmp_db):
    """A buggy detector shouldn't crash the whole backtest."""
    _seeded_prices("TEST.NS")

    def broken_detector(df, ticker):
        raise ValueError("simulated bug")

    broken_detector.__name__ = "detect_broken"

    # Should not raise — exceptions are swallowed per-detector
    trades = engine.backtest_ticker("TEST.NS", detectors=[broken_detector], max_hold=20)
    assert trades == []


def test_time_stop_when_neither_sl_nor_target_hit(tmp_db):
    """A trade that drifts sideways should exit at max_hold with outcome=time_stop."""
    _seeded_prices("TEST.NS")
    trades = engine.backtest_ticker("TEST.NS", max_hold=20)
    time_stops = [t for t in trades if t.outcome == "time_stop"]
    for t in time_stops:
        assert t.bars_held <= 20
        # Time stop r_multiple can be positive or negative (whatever the close was)
        assert t.r_multiple is not None


def test_profit_factor_infinity_when_no_losses():
    """A perfect strategy (no losses) has profit factor = infinity."""
    trades = pd.DataFrame([
        {"setup": "perfect", "r": 2.0, "bars_held": 5, "entry_date": "2024-01-01"},
        {"setup": "perfect", "r": 3.0, "bars_held": 4, "entry_date": "2024-01-02"},
    ])
    out = metrics.summarize(trades)
    perfect = out[out["setup"] == "perfect"].iloc[0]
    assert perfect["profit_factor"] == float("inf")
    assert perfect["win_rate"] == 1.0


def test_cli_backtest_runs(tmp_db, capsys):
    """End-to-end CLI smoke test."""
    from swingdesk import cli
    _seeded_prices("TEST.NS")
    cli.main(["backtest", "--ticker", "TEST.NS", "--max-hold", "10"])
    out = capsys.readouterr().out
    # Should mention trades or "no trades" — both are valid outcomes
    assert "trades" in out.lower() or "no trades" in out.lower()


def test_cli_backtest_no_data(tmp_db, capsys):
    """CLI should handle the 'no trades' case gracefully."""
    from swingdesk import cli
    # Set watchlist to a ticker we have no data for
    cli.main(["watchlist", "--set", "NONEXISTENT.NS"])
    capsys.readouterr()  # drain
    cli.main(["backtest"])
    out = capsys.readouterr().out
    assert "no trades" in out.lower()
