from __future__ import annotations

import pandas as pd
import pytest

from swingdesk import storage


def test_init_creates_tables(tmp_db):
    import sqlite3

    with sqlite3.connect(tmp_db) as con:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert {"prices", "news", "signals", "watchlist"} <= tables


def test_prices_upsert_and_load(tmp_db, synth_ohlcv):
    n = storage.upsert_prices("TEST.NS", synth_ohlcv)
    assert n == len(synth_ohlcv)
    df = storage.load_prices("TEST.NS")
    assert len(df) == len(synth_ohlcv)
    assert set(["open", "high", "low", "close", "volume"]) <= set(df.columns)


def test_prices_upsert_is_idempotent(tmp_db, synth_ohlcv):
    storage.upsert_prices("TEST.NS", synth_ohlcv)
    storage.upsert_prices("TEST.NS", synth_ohlcv)  # again
    df = storage.load_prices("TEST.NS")
    assert len(df) == len(synth_ohlcv)  # not doubled


def test_news_insert_dedupes_on_link(tmp_db):
    items = [
        {"source": "A", "title": "T1", "link": "http://x/1", "published": None,
         "summary": "", "tickers": ["RELIANCE.NS"]},
        {"source": "A", "title": "T1-dup", "link": "http://x/1", "published": None,
         "summary": "", "tickers": []},
        {"source": "A", "title": "T2", "link": "http://x/2", "published": None,
         "summary": "", "tickers": []},
    ]
    storage.insert_news(items)
    storage.insert_news(items)  # again — should be deduped
    df = storage.load_news(limit=10)
    assert len(df) == 2


def test_news_filter_by_ticker(tmp_db):
    storage.insert_news([
        {"source": "A", "title": "x", "link": "l1", "published": None,
         "summary": "", "tickers": ["RELIANCE.NS"]},
        {"source": "A", "title": "y", "link": "l2", "published": None,
         "summary": "", "tickers": ["TCS.NS"]},
    ])
    rel = storage.load_news(limit=10, ticker="RELIANCE.NS")
    assert len(rel) == 1
    assert "RELIANCE" in rel.iloc[0]["tickers"]


def test_signals_save_and_load(tmp_db):
    storage.save_signals([
        {"ticker": "TEST.NS", "setup": "breakout_20d", "direction": "long",
         "entry": 100.0, "stoploss": 95.0, "target": 115.0, "rr": 3.0,
         "score": 70.0, "notes": "test"},
    ])
    df = storage.load_signals(limit=10)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "TEST.NS"
    assert df.iloc[0]["rr"] == 3.0


def test_watchlist_set_get_seed(tmp_db):
    assert storage.get_watchlist() == []
    storage.set_watchlist(["A.NS", "B.NS"])
    assert storage.get_watchlist() == ["A.NS", "B.NS"]
    # seed should NOT overwrite when non-empty
    storage.seed_watchlist_if_empty(["C.NS"])
    assert storage.get_watchlist() == ["A.NS", "B.NS"]
    storage.set_watchlist([])
    storage.seed_watchlist_if_empty(["C.NS"])
    assert storage.get_watchlist() == ["C.NS"]


def test_load_prices_days_limit(tmp_db, synth_ohlcv):
    storage.upsert_prices("TEST.NS", synth_ohlcv)
    df = storage.load_prices("TEST.NS", days=30)
    assert len(df) == 30
