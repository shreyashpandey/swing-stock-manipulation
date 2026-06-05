"""Portfolio module tests: sizing, lifecycle, mark-to-market, journal, importer."""
from __future__ import annotations

import io
import math
from datetime import datetime

import pandas as pd
import pytest

from swingdesk import storage
from swingdesk.portfolio import import_groww, journal as pj, positions as portfolio


# ---- position sizing ----------------------------------------------------------

def test_sizing_basic():
    # Capital 100k, risk 1% = 1000. SL is 5 below entry → 1000/5 = 200 shares.
    r = portfolio.size_position(entry=100, stoploss=95, capital=100_000, risk_pct=1.0)
    assert r.qty == 200
    assert abs(r.risk_amount - 1000) < 0.01
    assert r.notional == 20_000
    assert not r.rejected


def test_sizing_rejected_when_sl_above_entry():
    r = portfolio.size_position(entry=100, stoploss=105, capital=100_000, risk_pct=1.0)
    assert r.rejected
    assert r.qty == 0
    assert "invalid SL" in r.reason


def test_sizing_caps_at_available_capital():
    # Tiny risk per share but expensive stock — notional should cap at capital.
    r = portfolio.size_position(entry=10_000, stoploss=9_999, capital=100_000, risk_pct=1.0)
    # Risk amount = 1000, risk per share = 1, qty would be 1000.
    # But notional = 1000 * 10_000 = 10M >> 100k capital → must cap.
    assert r.qty <= 10  # capital / entry
    assert r.notional <= 100_000


def test_sizing_rejected_when_risk_too_small_for_one_share():
    # 100 capital * 0.01 = 1 of risk; if risk_per_share is 5, qty = 0.
    r = portfolio.size_position(entry=100, stoploss=95, capital=100, risk_pct=1.0)
    assert r.rejected
    assert "too small" in r.reason


# ---- open / close lifecycle ---------------------------------------------------

def test_open_position_creates_row(tmp_db):
    res = portfolio.open_position("TEST.NS", entry=100, stoploss=95, target=115,
                                  setup="manual", is_paper=True)
    assert res["status"] == "opened"
    pos = res["position"]
    assert pos["ticker"] == "TEST.NS"
    assert pos["status"] == "open"
    assert pos["qty"] > 0
    assert pos["is_paper"] == 1


def test_open_position_with_explicit_qty(tmp_db):
    res = portfolio.open_position("TEST.NS", entry=100, stoploss=95, target=115,
                                  qty=50, is_paper=True)
    assert res["status"] == "opened"
    assert res["position"]["qty"] == 50


def test_open_rejects_duplicate_on_same_ticker(tmp_db):
    portfolio.open_position("TEST.NS", entry=100, stoploss=95, target=115, qty=10)
    res2 = portfolio.open_position("TEST.NS", entry=110, stoploss=105, target=120, qty=10)
    assert res2["status"] == "rejected"
    assert "already have" in res2["reason"]


def test_open_respects_max_concurrent_cap(tmp_db, monkeypatch):
    from swingdesk.portfolio import positions as p
    monkeypatch.setattr(p, "MAX_OPEN_POSITIONS", 2)
    portfolio.open_position("A.NS", entry=100, stoploss=95, target=115, qty=1)
    portfolio.open_position("B.NS", entry=100, stoploss=95, target=115, qty=1)
    res = portfolio.open_position("C.NS", entry=100, stoploss=95, target=115, qty=1)
    assert res["status"] == "rejected"
    assert "max positions" in res["reason"]


def test_close_position_records_pnl(tmp_db):
    res = portfolio.open_position("TEST.NS", entry=100, stoploss=95, target=115, qty=10)
    pos_id = res["position"]["id"]
    closed = portfolio.close_position(pos_id, exit_price=110, exit_reason="target")
    assert closed["status"] == "closed"
    p = closed["position"]
    assert p["status"] == "closed"
    assert p["exit_price"] == 110
    # P&L = (110 - 100) * 10 = 100
    assert p["pnl"] == 100
    # R-multiple = (110-100) / (100-95) = 2.0
    assert abs(p["r_multiple"] - 2.0) < 0.01


def test_close_nonexistent_returns_not_found(tmp_db):
    res = portfolio.close_position(99999, exit_price=100)
    assert res["status"] == "not_found"


def test_close_idempotent(tmp_db):
    """Closing an already-closed position should not crash or re-record."""
    res = portfolio.open_position("TEST.NS", entry=100, stoploss=95, target=115, qty=10)
    pos_id = res["position"]["id"]
    portfolio.close_position(pos_id, exit_price=110)
    res2 = portfolio.close_position(pos_id, exit_price=120)
    assert res2["status"] == "already_closed"


# ---- mark-to-market -----------------------------------------------------------

def _add_price_bars(ticker: str, last_bar: dict):
    """Insert a single-bar OHLCV row so mark_to_market has data to read."""
    import pandas as pd
    idx = pd.date_range("2025-01-01", periods=1, freq="B")
    df = pd.DataFrame([last_bar], index=idx)
    storage.upsert_prices(ticker, df)


def test_mark_to_market_closes_on_target_hit(tmp_db):
    portfolio.open_position("TEST.NS", entry=100, stoploss=95, target=115, qty=10)
    _add_price_bars("TEST.NS", {"open": 110, "high": 120, "low": 109, "close": 118, "volume": 100000})
    res = portfolio.mark_to_market()
    assert res["closed"] == 1
    closed_pos = storage.load_positions(status="closed")
    assert len(closed_pos) == 1
    assert closed_pos.iloc[0]["exit_reason"] == "target"
    assert closed_pos.iloc[0]["exit_price"] == 115  # exits at target, not at high


def test_mark_to_market_closes_on_stoploss(tmp_db):
    portfolio.open_position("TEST.NS", entry=100, stoploss=95, target=115, qty=10)
    _add_price_bars("TEST.NS", {"open": 98, "high": 99, "low": 94, "close": 96, "volume": 100000})
    res = portfolio.mark_to_market()
    assert res["closed"] == 1
    closed = storage.load_positions(status="closed").iloc[0]
    assert closed["exit_reason"] == "stoploss"
    assert closed["exit_price"] == 95  # exits at stoploss level


def test_mark_to_market_gap_down_exits_at_open(tmp_db):
    portfolio.open_position("TEST.NS", entry=100, stoploss=95, target=115, qty=10)
    # Gap-down: opens at 90, well below stop
    _add_price_bars("TEST.NS", {"open": 90, "high": 91, "low": 85, "close": 87, "volume": 100000})
    res = portfolio.mark_to_market()
    assert res["closed"] == 1
    closed = storage.load_positions(status="closed").iloc[0]
    assert closed["exit_reason"] == "stoploss_gap"
    assert closed["exit_price"] == 90  # the gapped open


def test_mark_to_market_updates_last_price_when_no_exit(tmp_db):
    portfolio.open_position("TEST.NS", entry=100, stoploss=95, target=115, qty=10)
    _add_price_bars("TEST.NS", {"open": 102, "high": 103, "low": 101, "close": 102.5, "volume": 100000})
    res = portfolio.mark_to_market()
    assert res["closed"] == 0
    assert res["marked"] == 1
    open_pos = storage.load_positions(status="open").iloc[0]
    assert open_pos["last_price"] == 102.5


# ---- auto paper-trade ---------------------------------------------------------

def test_auto_paper_trade_opens_high_score_signals(tmp_db):
    signals = [
        {"ticker": "A.NS", "setup": "breakout", "entry": 100, "stoploss": 95,
         "target": 115, "composite_score": 75.0, "notes": "n"},
        {"ticker": "B.NS", "setup": "breakout", "entry": 100, "stoploss": 95,
         "target": 115, "composite_score": 55.0, "notes": "n"},  # below threshold
    ]
    res = portfolio.auto_paper_trade(signals, min_composite=60.0)
    assert res["opened"] == 1
    assert res["skipped"] == 1


def test_auto_paper_trade_skips_duplicates(tmp_db):
    portfolio.open_position("A.NS", entry=100, stoploss=95, target=115, qty=10)
    signals = [
        {"ticker": "A.NS", "setup": "x", "entry": 100, "stoploss": 95,
         "target": 115, "composite_score": 90.0, "notes": ""},
    ]
    res = portfolio.auto_paper_trade(signals)
    assert res["opened"] == 0
    assert res["skipped"] == 1


# ---- journal / stats ---------------------------------------------------------

def test_stats_empty_returns_zero(tmp_db):
    s = pj.stats()
    assert s.n_trades == 0
    assert s.total_pnl == 0


def test_stats_one_winning_trade(tmp_db):
    res = portfolio.open_position("A.NS", entry=100, stoploss=95, target=115, qty=10)
    portfolio.close_position(res["position"]["id"], exit_price=115, exit_reason="target")
    s = pj.stats()
    assert s.n_trades == 1
    assert s.wins == 1
    assert s.losses == 0
    assert s.total_pnl == 150  # (115-100)*10
    assert s.win_rate == 1.0
    assert abs(s.avg_r - 3.0) < 0.01


def test_stats_mixed_trades(tmp_db):
    r1 = portfolio.open_position("A.NS", entry=100, stoploss=95, target=115, qty=10)
    portfolio.close_position(r1["position"]["id"], exit_price=115)
    r2 = portfolio.open_position("B.NS", entry=100, stoploss=95, target=115, qty=10)
    portfolio.close_position(r2["position"]["id"], exit_price=95)
    s = pj.stats()
    assert s.n_trades == 2
    assert s.wins == 1 and s.losses == 1
    assert s.win_rate == 0.5
    assert s.total_pnl == (150 + -50)  # 100


def test_equity_curve(tmp_db):
    r1 = portfolio.open_position("A.NS", entry=100, stoploss=95, target=115, qty=10)
    portfolio.close_position(r1["position"]["id"], exit_price=115)
    curve = pj.equity_curve()
    assert len(curve) == 1
    assert curve.iloc[0]["pnl"] == 150
    # equity = starting_capital + cum_pnl
    from swingdesk.config import ACCOUNT_CAPITAL
    assert curve.iloc[0]["equity"] == ACCOUNT_CAPITAL + 150


def test_by_setup_breakdown(tmp_db):
    r1 = portfolio.open_position("A.NS", entry=100, stoploss=95, target=115, qty=10,
                                 setup="breakout")
    portfolio.close_position(r1["position"]["id"], exit_price=115)
    r2 = portfolio.open_position("B.NS", entry=100, stoploss=95, target=115, qty=10,
                                 setup="pullback")
    portfolio.close_position(r2["position"]["id"], exit_price=90)
    by_s = pj.by_setup()
    assert len(by_s) == 2
    breakout = by_s[by_s["setup"] == "breakout"].iloc[0]
    assert breakout["wins"] == 1
    assert breakout["total_pnl"] == 150


# ---- broker CSV importer ------------------------------------------------------

GROWW_CSV = """Symbol,Trade Type,Quantity,Price,Date
RELIANCE,buy,10,2500,2025-01-15
RELIANCE,sell,10,2600,2025-01-20
TCS,buy,5,4000,2025-01-22
INFY,buy,8,1500,2025-02-01
INFY,sell,4,1600,2025-02-10
"""


def test_parse_csv_normalizes_columns(tmp_path):
    csv_path = tmp_path / "trades.csv"
    csv_path.write_text(GROWW_CSV)
    df = import_groww.parse_csv(csv_path)
    assert set(df.columns) >= {"symbol", "side", "qty", "price", "date"}
    assert (df["symbol"] == "RELIANCE.NS").any()
    assert (df["symbol"] == "TCS.NS").any()


def test_parse_csv_strips_series_suffix(tmp_path):
    csv_path = tmp_path / "trades.csv"
    csv_path.write_text("Symbol,Trade Type,Quantity,Price,Date\nRELIANCE-EQ,buy,10,2500,2025-01-15\n")
    df = import_groww.parse_csv(csv_path)
    assert df.iloc[0]["symbol"] == "RELIANCE.NS"


def test_parse_csv_missing_columns_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("Foo,Bar\n1,2\n")
    with pytest.raises(ValueError, match="missing required columns"):
        import_groww.parse_csv(csv_path)


def test_parse_csv_override_mapping(tmp_path):
    """Custom column names work via override map."""
    csv_path = tmp_path / "weird.csv"
    csv_path.write_text("Instrument,Action,Shares,Rate,Timestamp\nTCS,BUY,5,4000,2025-01-22\n")
    # 'Action' should auto-detect as 'side' via alias; 'Instrument' as 'symbol'.
    df = import_groww.parse_csv(csv_path)
    assert df.iloc[0]["symbol"] == "TCS.NS"
    assert df.iloc[0]["side"] == "buy"


def test_import_trades_fifo_matching(tmp_db, tmp_path):
    csv_path = tmp_path / "trades.csv"
    csv_path.write_text(GROWW_CSV)
    res = import_groww.import_trades(csv_path, is_paper=False)
    assert res["buys"] == 3
    assert res["sells"] == 2

    # RELIANCE: full buy + full sell → closed
    rel = storage.load_positions()
    rel = rel[rel["ticker"] == "RELIANCE.NS"]
    assert (rel["status"] == "closed").all()
    # P&L = (2600 - 2500) * 10 = 1000
    assert rel.iloc[0]["pnl"] == 1000

    # INFY: bought 8, sold 4 → one closed position of 4, one open of 4
    infy = storage.load_positions()
    infy = infy[infy["ticker"] == "INFY.NS"]
    assert len(infy) == 2
    assert (infy["status"] == "closed").sum() == 1
    assert (infy["status"] == "open").sum() == 1

    # TCS: buy only, no sell → still open
    tcs = storage.load_positions()
    tcs = tcs[tcs["ticker"] == "TCS.NS"]
    assert len(tcs) == 1
    assert tcs.iloc[0]["status"] == "open"


def test_import_warns_on_unmatched_sell(tmp_db, tmp_path, capsys):
    """Selling more than was bought logs a warning rather than crashing."""
    csv = "Symbol,Trade Type,Quantity,Price,Date\nTCS,sell,5,4000,2025-01-22\n"
    csv_path = tmp_path / "weird.csv"
    csv_path.write_text(csv)
    res = import_groww.import_trades(csv_path)
    assert res["sells"] == 1
    assert res["matched"] == 0  # nothing to match against
