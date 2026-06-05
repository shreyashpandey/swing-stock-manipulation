"""Tests for the auto-fetch behavior in open_position + mark_to_market.

The bug we're guarding against: paper positions opened on tickers with no
local price history would show last_price == entry_price forever, so
unrealized P&L always read zero. The fix is to auto-fetch via yfinance
when no data exists yet.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from swingdesk import storage
from swingdesk.portfolio import positions as portfolio


def _fake_yf_df(n: int = 30, close_value: float = 110.0) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": [close_value] * n,
        "High": [close_value + 1] * n,
        "Low": [close_value - 1] * n,
        "Close": [close_value] * n,
        "Volume": [100_000.0] * n,
    }, index=idx)


def test_open_position_auto_fetches_when_no_local_data(tmp_db, monkeypatch):
    """If we try to open a position on a ticker we've never seen, the open
    should trigger a price fetch so last_price reflects reality, not entry."""
    monkeypatch.setattr(portfolio, "days_to_earnings", lambda t: None)

    # Mock fetch_one so we don't hit the network
    fake_df = _fake_yf_df(close_value=110.0)
    with patch("swingdesk.ingest.prices.fetch_one", return_value=fake_df):
        res = portfolio.open_position("UNKNOWN.NS",
                                       entry=100, stoploss=95, target=115, qty=10)
    assert res["status"] == "opened"
    pos = res["position"]
    # last_price should be the FETCHED close, not the entry
    assert pos["last_price"] == 110.0


def test_open_position_falls_back_to_entry_if_fetch_returns_empty(tmp_db, monkeypatch):
    """When yfinance returns nothing, we still open the position cleanly,
    falling back to entry as the seed last_price."""
    monkeypatch.setattr(portfolio, "days_to_earnings", lambda t: None)
    with patch("swingdesk.ingest.prices.fetch_one", return_value=pd.DataFrame()):
        res = portfolio.open_position("MYSTERY.NS",
                                       entry=100, stoploss=95, target=115, qty=10)
    assert res["status"] == "opened"
    pos = res["position"]
    # Fallback: last_price seeded to entry
    assert pos["last_price"] == 100.0


def test_open_position_uses_existing_data_when_available(tmp_db, monkeypatch):
    """If we already have prices stored, open_position should use them and
    NOT call fetch_one."""
    monkeypatch.setattr(portfolio, "days_to_earnings", lambda t: None)
    # Seed local data
    idx = pd.date_range("2025-01-01", periods=10, freq="B")
    storage.upsert_prices("HAVE.NS", pd.DataFrame({
        "open": [100] * 10, "high": [105] * 10, "low": [98] * 10,
        "close": [103] * 10, "volume": [100_000.0] * 10,
    }, index=idx))

    fetch_called = {"n": 0}
    def _spy(*a, **kw):
        fetch_called["n"] += 1
        return _fake_yf_df(close_value=999)  # would be wrong
    with patch("swingdesk.ingest.prices.fetch_one", side_effect=_spy):
        res = portfolio.open_position("HAVE.NS",
                                       entry=100, stoploss=95, target=115, qty=10)
    assert res["status"] == "opened"
    # last_price should reflect existing data (103), not the spy's 999
    assert res["position"]["last_price"] == 103.0
    # fetch_one should NOT have been called
    assert fetch_called["n"] == 0


def test_mark_to_market_auto_fetch_when_position_has_no_data(tmp_db, monkeypatch):
    """MTM should auto-fetch for tickers we've never priced — so a paper
    position opened on a wildcard ticker still gets marked."""
    monkeypatch.setattr(portfolio, "days_to_earnings", lambda t: None)

    # Open a position WITHOUT seeding prices first (the open call will fetch
    # automatically, so the next MTM has data); but force the position to
    # have NULL last_price + clear local data to recreate the "stale" case.
    with patch("swingdesk.ingest.prices.fetch_one",
               return_value=_fake_yf_df(close_value=120.0)):
        res = portfolio.open_position("BLANK.NS",
                                       entry=100, stoploss=95, target=130, qty=10)
    pos_id = res["position"]["id"]

    # Wipe the prices to simulate a fresh DB / stale data
    with storage.connect() as con:
        con.execute("DELETE FROM prices WHERE ticker=?", ("BLANK.NS",))

    # MTM should re-fetch
    with patch("swingdesk.ingest.prices.ingest") as mock_ingest:
        def _fake_ingest(tickers, period="6mo", workers=6):
            # Simulate ingest behavior: store data
            for tk in tickers:
                storage.upsert_prices(tk, pd.DataFrame({
                    "open": [115], "high": [125], "low": [114],
                    "close": [120], "volume": [100_000.0],
                }, index=pd.date_range("2025-06-01", periods=1, freq="B")))
        mock_ingest.side_effect = _fake_ingest
        res2 = portfolio.mark_to_market(auto_fetch=True)
    # Should record a fetched count
    assert res2.get("fetched", 0) >= 1
