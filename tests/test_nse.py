from __future__ import annotations

import pandas as pd

from swingdesk.ingest import nse


def test_symbol_ticker_roundtrip():
    assert nse.to_symbol("RELIANCE.NS") == "RELIANCE"
    assert nse.to_symbol("INFY") == "INFY"
    assert nse.to_symbol("TATAMOTORS.BO") is None  # BSE not on NSE
    assert nse.to_ticker("RELIANCE") == "RELIANCE.NS"


def test_num_coerces_nse_dashes():
    assert nse._num("1,234.5") == 1234.5
    assert nse._num("-") is None
    assert nse._num(" 12.0 ") == 12.0
    assert nse._num(None) is None


def test_parse_deals_filters_and_maps():
    raw = pd.DataFrame({
        "Date": ["09-JUN-2026", "09-JUN-2026"],
        "Symbol": ["ACCENTMIC", "RELIANCE"],
        "Security Name": ["Accent Microcell", "Reliance"],
        "Client Name": ["H. AMIN HUF", "SOME FUND"],
        "Buy/Sell": ["BUY", "SELL"],
        "Quantity Traded": ["125000", "50000"],
        "Trade Price / Wght. Avg. Price": ["479.66", "2900.0"],
        "Remarks": ["-", "-"],
    })
    rows = nse._parse_deals(raw, "bulk", wanted={"ACCENTMIC"})
    assert len(rows) == 1
    r = rows[0]
    assert r["ticker"] == "ACCENTMIC.NS"
    assert r["date"] == "2026-06-09"
    assert r["side"] == "BUY"
    assert r["qty"] == 125000.0
    assert r["price"] == 479.66
    assert r["deal_type"] == "bulk"


def test_parse_deals_unfiltered_keeps_all():
    raw = pd.DataFrame({
        "Date": ["09-JUN-2026"],
        "Symbol": ["XYZ"],
        "Security Name": ["X"],
        "Client Name": ["C"],
        "Buy/Sell": ["BUY"],
        "Quantity Traded": ["1"],
        "Trade Price / Wght. Avg. Price": ["1.0"],
    })
    assert len(nse._parse_deals(raw, "block", wanted=None)) == 1


def test_delivery_and_deals_storage_roundtrip(tmp_db):
    from swingdesk import storage

    storage.upsert_delivery([
        {"ticker": "ABC.NS", "date": "2026-06-09", "traded_qty": 1000,
         "deliv_qty": 200, "deliv_pct": 20.0},
        {"ticker": "ABC.NS", "date": "2026-06-06", "traded_qty": 800,
         "deliv_qty": 480, "deliv_pct": 60.0},
    ])
    d = storage.load_delivery("ABC.NS")
    assert len(d) == 2
    assert d["deliv_pct"].iloc[-1] == 20.0  # ordered by date ascending

    storage.upsert_deals([
        {"deal_type": "bulk", "date": "2026-06-09", "ticker": "ABC.NS",
         "security": "ABC Ltd", "client": "OP LLP", "side": "BUY",
         "qty": 1e5, "price": 100.0},
    ])
    deals = storage.load_deals("ABC.NS")
    assert len(deals) == 1
    assert deals["client"].iloc[0] == "OP LLP"
