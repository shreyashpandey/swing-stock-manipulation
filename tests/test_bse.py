from __future__ import annotations

from swingdesk.ingest import bse


def test_records_to_rows_maps_side_and_exchange():
    records = [{
        "DEAL_DATE": "2026-08-12T00:00:00",
        "SCRIP_CODE": 540565,
        "scripname": "INDIGRID",
        "CLIENT_NAME": "GOVERNMENT OF SINGAPORE",
        "TRANSACTION_TYPE": "S",
        "QUANTITY": 28735632.0,
        "PRICE": 174.03,
    }]
    rows = bse._records_to_rows(records, "bulk", wanted=None, known_nse_symbols={"INDIGRID"})
    assert len(rows) == 1
    row = rows[0]
    assert row["exchange"] == "BSE"
    assert row["deal_type"] == "bulk"
    assert row["ticker"] == "INDIGRID.NS"
    assert row["side"] == "SELL"
    assert row["client"] == "GOVERNMENT OF SINGAPORE"


def test_records_to_rows_filters_symbols():
    records = [{
        "DEAL_DATE": "2026-08-12T00:00:00",
        "SCRIP_CODE": 540565,
        "scripname": "INDIGRID",
        "CLIENT_NAME": "A",
        "TRANSACTION_TYPE": "P",
        "QUANTITY": 1.0,
        "PRICE": 1.0,
    }]
    rows = bse._records_to_rows(records, "bulk", wanted={"OTHER"}, known_nse_symbols=set())
    assert rows == []
