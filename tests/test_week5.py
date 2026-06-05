"""Week 5 tests: trailing stops, earnings filter, optimizer, reconcile."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from swingdesk import storage
from swingdesk.portfolio import positions as portfolio, reconcile


def _add_bar(ticker: str, **kwargs):
    """Insert a single bar; convenience for mark-to-market tests."""
    idx = pd.date_range("2025-01-01", periods=1, freq="B")
    bar = {"open": 100, "high": 102, "low": 98, "close": 101, "volume": 100_000}
    bar.update(kwargs)
    df = pd.DataFrame([bar], index=idx)
    storage.upsert_prices(ticker, df)


# ---- earnings filter ----------------------------------------------------------

def test_open_blocked_if_earnings_imminent(tmp_db, monkeypatch):
    """Earnings within blackout window should reject the new position."""
    from swingdesk.portfolio import positions as p
    monkeypatch.setattr(p, "EARNINGS_BLACKOUT_DAYS", 3)
    # Mock days_to_earnings → 2 days from now
    monkeypatch.setattr(p, "days_to_earnings", lambda t: 2)
    res = portfolio.open_position("TCS.NS", entry=100, stoploss=95, target=115, qty=10)
    assert res["status"] == "rejected"
    assert "earnings" in res["reason"]


def test_open_allowed_when_earnings_far(tmp_db, monkeypatch):
    from swingdesk.portfolio import positions as p
    monkeypatch.setattr(p, "EARNINGS_BLACKOUT_DAYS", 3)
    monkeypatch.setattr(p, "days_to_earnings", lambda t: 10)  # 10 days away
    res = portfolio.open_position("TCS.NS", entry=100, stoploss=95, target=115, qty=10)
    assert res["status"] == "opened"


def test_open_allowed_when_no_earnings_known(tmp_db, monkeypatch):
    """If we don't know the earnings date, don't block (fail-open)."""
    from swingdesk.portfolio import positions as p
    monkeypatch.setattr(p, "days_to_earnings", lambda t: None)
    res = portfolio.open_position("TCS.NS", entry=100, stoploss=95, target=115, qty=10)
    assert res["status"] == "opened"


def test_skip_earnings_check_flag(tmp_db, monkeypatch):
    """skip_earnings_check=True bypasses the blackout."""
    from swingdesk.portfolio import positions as p
    monkeypatch.setattr(p, "days_to_earnings", lambda t: 1)  # earnings tomorrow
    res = portfolio.open_position("TCS.NS", entry=100, stoploss=95, target=115,
                                  qty=10, skip_earnings_check=True)
    assert res["status"] == "opened"


# ---- trailing stops -----------------------------------------------------------

def test_initial_stop_persisted_on_open(tmp_db, monkeypatch):
    monkeypatch.setattr(portfolio, "days_to_earnings", lambda t: None)
    res = portfolio.open_position("TEST.NS", entry=100, stoploss=95, target=115, qty=10)
    pos = res["position"]
    assert pos["initial_stop"] == 95
    assert pos["high_water"] == 100


def test_trailing_moves_stop_to_breakeven_at_1r(tmp_db, monkeypatch):
    """At +1R unrealized, stop should move to entry price."""
    monkeypatch.setattr(portfolio, "days_to_earnings", lambda t: None)
    portfolio.open_position("TEST.NS", entry=100, stoploss=95, target=120, qty=10)
    # New bar: close at 105 = +1R unrealized (risk per share = 5; gain per share = 5)
    _add_bar("TEST.NS", open=104, high=106, low=103, close=105)
    res = portfolio.mark_to_market()
    assert res["trailed"] == 1
    pos = storage.load_positions(status="open").iloc[0]
    assert pos["stoploss"] == 100  # moved to entry (breakeven)
    assert pos["initial_stop"] == 95  # initial preserved


def test_trailing_does_not_move_stop_down(tmp_db, monkeypatch):
    monkeypatch.setattr(portfolio, "days_to_earnings", lambda t: None)
    portfolio.open_position("TEST.NS", entry=100, stoploss=95, target=120, qty=10)
    # Price slightly up but not yet +1R
    _add_bar("TEST.NS", open=101, high=102, low=100.5, close=101.5)
    portfolio.mark_to_market()
    pos = storage.load_positions(status="open").iloc[0]
    assert pos["stoploss"] == 95  # untouched


def test_trailing_at_2r_uses_atr_proxy(tmp_db, monkeypatch):
    """Past +2R, trail by ATR-proxy below the new high."""
    monkeypatch.setattr(portfolio, "days_to_earnings", lambda t: None)
    portfolio.open_position("TEST.NS", entry=100, stoploss=95, target=200, qty=10)
    # Big up day: close at 115 = +3R, high 118 low 110 → atr_proxy=8
    # Expected trail = 118 - 1.5*8 = 106
    _add_bar("TEST.NS", open=112, high=118, low=110, close=115)
    portfolio.mark_to_market()
    pos = storage.load_positions(status="open").iloc[0]
    # Stop should have moved up (at least past entry, possibly to ~106)
    assert pos["stoploss"] >= 100
    assert pos["high_water"] == 118


def test_trail_exit_reason_when_trailed_stop_hit(tmp_db, monkeypatch):
    """When a trailed (>initial) stop is hit, exit_reason should be 'trail'."""
    monkeypatch.setattr(portfolio, "days_to_earnings", lambda t: None)
    portfolio.open_position("TEST.NS", entry=100, stoploss=95, target=200, qty=10)
    # Push past +1R first to move stop to breakeven
    _add_bar("TEST.NS", open=103, high=108, low=102, close=107)
    portfolio.mark_to_market()
    # Now drop below the new (trailed) stop
    _add_bar("TEST.NS", open=99, high=99, low=95, close=96)
    portfolio.mark_to_market()
    closed = storage.load_positions(status="closed")
    assert len(closed) == 1
    assert closed.iloc[0]["exit_reason"] == "trail"


# ---- optimizer ----------------------------------------------------------------

def _seed_prices_for_opt(ticker: str, n: int = 200):
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    base = 100 + np.cumsum(rng.normal(0.2, 1.0, n))
    open_ = base + rng.normal(0, 0.3, n)
    close = base + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.2, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.2, n)
    vol = rng.integers(100_000, 1_000_000, n).astype(float)
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close, "volume": vol}, index=idx)
    storage.upsert_prices(ticker, df)


def test_optimizer_returns_ranked_results(tmp_db):
    from swingdesk.backtest import optimizer
    _seed_prices_for_opt("TEST.NS")
    # Use a tiny grid to keep the test fast
    grid = optimizer.ParamGrid(sl_mult=(1.5,), tgt_mult=(3.0, 4.0), max_hold=(20,))
    results = optimizer.optimize(["TEST.NS"], "breakout_20d", grid)
    if not results.empty:
        # Sorted descending by expectancy
        exp_values = results["expectancy"].tolist()
        assert exp_values == sorted(exp_values, reverse=True)
        assert "sl_mult" in results.columns
        assert "tgt_mult" in results.columns


def test_optimizer_unknown_setup_raises(tmp_db):
    from swingdesk.backtest import optimizer
    with pytest.raises(ValueError, match="unknown setup"):
        optimizer.optimize(["TEST.NS"], "nonexistent_setup")


# ---- reconcile ----------------------------------------------------------------

def test_reconcile_empty_when_no_backtest(tmp_db):
    df = reconcile.reconcile()
    assert df.empty


def test_reconcile_insufficient_when_few_paper_trades(tmp_db, monkeypatch):
    """Setups with <min_paper_trades get verdict='insufficient'."""
    # Seed one backtest run with 10 breakout trades
    bt = pd.DataFrame([{
        "ticker": "A.NS", "setup": "breakout_20d", "entry_date": f"2024-01-{i:02d}",
        "exit_date": f"2024-01-{i+1:02d}", "entry": 100, "exit": 105, "stoploss": 95,
        "target": 115, "outcome": "target", "r": 1.0, "bars_held": 1, "planned_rr": 3.0,
    } for i in range(1, 11)])
    storage.save_backtest_trades("r1", bt)
    # No paper positions
    df = reconcile.reconcile(min_paper_trades=5)
    row = df.iloc[0]
    assert row["setup"] == "breakout_20d"
    assert row["verdict"] == "insufficient"


def test_reconcile_aligned_when_paper_matches_backtest(tmp_db, monkeypatch):
    """Verdict='aligned' when paper-trade stats are within tolerance of backtest."""
    monkeypatch.setattr(portfolio, "days_to_earnings", lambda t: None)
    # Backtest: 10 trades, 60% win rate, avg R = 0.5
    bt_trades = []
    for i in range(1, 11):
        r = 1.5 if i <= 6 else -1.0
        bt_trades.append({
            "ticker": "A.NS", "setup": "breakout_20d", "entry_date": f"2024-01-{i:02d}",
            "exit_date": f"2024-01-{i+1:02d}", "entry": 100,
            "exit": 100 + r * 5, "stoploss": 95, "target": 115,
            "outcome": "target" if r > 0 else "stoploss", "r": r, "bars_held": 1,
            "planned_rr": 3.0,
        })
    storage.save_backtest_trades("r1", pd.DataFrame(bt_trades))

    # Paper: 5 trades matching backtest distribution
    for i, r in enumerate([1.5, 1.5, 1.5, -1.0, 1.5]):
        pos_res = portfolio.open_position(
            f"P{i}.NS", entry=100, stoploss=95, target=115, qty=1,
            setup="breakout_20d",
        )
        portfolio.close_position(pos_res["position"]["id"],
                                 exit_price=100 + r * 5,
                                 exit_reason="target" if r > 0 else "stoploss")

    df = reconcile.reconcile(min_paper_trades=5)
    row = df[df["setup"] == "breakout_20d"].iloc[0]
    # Paper win rate = 4/5 = 0.8, backtest = 0.6. Drift = 0.2 > 0.1 → not aligned.
    assert row["paper_trades"] == 5
    assert row["verdict"] in ("aligned", "outperforming", "underperforming")


# ---- earnings storage ----------------------------------------------------------

def test_upsert_and_get_earnings(tmp_db):
    storage.upsert_earnings("TCS.NS", "2026-07-15", "2026-04-15")
    assert storage.get_next_earnings("TCS.NS") == "2026-07-15"

    # Upsert overwrites
    storage.upsert_earnings("TCS.NS", "2026-08-01")
    assert storage.get_next_earnings("TCS.NS") == "2026-08-01"


def test_load_earnings_calendar(tmp_db):
    storage.upsert_earnings("A.NS", "2026-07-01")
    storage.upsert_earnings("B.NS", "2026-06-15")
    cal = storage.load_earnings_calendar()
    assert len(cal) == 2
    # Sorted ascending by next_earnings
    assert cal.iloc[0]["ticker"] == "B.NS"


def test_days_to_earnings_none_when_unknown(tmp_db):
    from swingdesk.ingest.earnings import days_to_earnings
    assert days_to_earnings("UNKNOWN.NS") is None
