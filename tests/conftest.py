"""Shared fixtures. Redirects the SQLite DB to a tmp file so tests don't touch
real data, and mocks yfinance to keep tests offline + fast."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Autouse fixture: replace network calls with empty returns so no test
    ever hits the wire. Specific tests that need to assert fetch behaviour
    can still patch with their own mock — this only kicks in as a default.

    Without this guard, tests that call open_position / mark_to_market on
    a synthetic ticker would wait for yfinance to return 404 — making the
    suite take >80s when it should take ~10s."""
    import swingdesk.ingest.prices as prices_mod

    def _empty_fetch(*args, **kwargs):
        return pd.DataFrame()

    def _empty_ingest(tickers, *args, **kwargs):
        return {t: 0 for t in tickers}

    # Patch the LIVE objects (not via monkeypatch on module to avoid stale refs)
    monkeypatch.setattr(prices_mod, "fetch_one", _empty_fetch, raising=False)
    monkeypatch.setattr(prices_mod, "ingest", _empty_ingest, raising=False)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch) -> Path:
    """Point swingdesk.storage at a fresh DB for the test, then init schema."""
    from swingdesk import storage

    db = tmp_path / "test.sqlite"
    monkeypatch.setattr(storage, "DB_PATH", db)
    storage.init_db(db)
    return db


@pytest.fixture
def synth_ohlcv() -> pd.DataFrame:
    """200 bars of synthetic upward-drifting OHLCV with realistic intra-bar structure."""
    rng = np.random.default_rng(42)
    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    base = 100 + np.cumsum(rng.normal(0.2, 1.0, n))  # mild uptrend
    open_ = base + rng.normal(0, 0.3, n)
    close = base + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.0, n)
    vol = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )
