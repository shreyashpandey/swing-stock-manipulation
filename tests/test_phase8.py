"""Phase 8 tests: discovery scanner, chart signal markers, investability summary."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingdesk import storage
from swingdesk.analyze import chart_signals, discovery, summary as summary_mod


def _seed(ticker: str, n: int = 200, trend: float = 0.25):
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    base = 100 + np.cumsum(rng.normal(trend, 1.0, n))
    opens = base + rng.normal(0, 0.3, n)
    closes = base + rng.normal(0, 0.3, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.1, 1.2, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.1, 1.2, n)
    vols = rng.integers(100_000, 1_000_000, n).astype(float)
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                       "close": closes, "volume": vols}, index=idx)
    storage.upsert_prices(ticker, df)


# ---- storage helpers ---------------------------------------------------------

def test_holdings_tickers_returns_distinct(tmp_db):
    storage.replace_holdings([
        {"ticker": "A.NS", "qty": 10, "avg_price": 100},
        {"ticker": "B.NS", "qty": 5, "avg_price": 200},
        {"ticker": "A.NS", "qty": 3, "avg_price": 110},  # duplicate ticker
    ])
    assert set(storage.holdings_tickers()) == {"A.NS", "B.NS"}


def test_combined_universe_dedupes(tmp_db):
    storage.set_watchlist(["A.NS", "B.NS"])
    storage.replace_holdings([
        {"ticker": "B.NS", "qty": 10, "avg_price": 100},   # overlap
        {"ticker": "C.NS", "qty": 5, "avg_price": 200},    # holding-only
    ])
    universe = storage.combined_universe()
    assert universe == ["A.NS", "B.NS", "C.NS"]


# ---- discovery scanner -------------------------------------------------------

def test_discovery_excludes_held_and_watchlist(tmp_db):
    _seed("FOO.NS")
    _seed("BAR.NS")
    storage.replace_holdings([{"ticker": "FOO.NS", "qty": 1, "avg_price": 1}])
    storage.set_watchlist(["BAR.NS"])
    # Both seeded tickers should be excluded
    opps = discovery.scan(universe=["FOO.NS", "BAR.NS"])
    assert opps == []


def test_discovery_ranks_by_composite_score(tmp_db):
    _seed("A.NS")
    _seed("B.NS")
    opps = discovery.scan(universe=["A.NS", "B.NS"],
                           exclude_held=False, exclude_watchlist=False)
    if len(opps) >= 2:
        scores = [o.composite_score for o in opps]
        assert scores == sorted(scores, reverse=True)


def test_discovery_handles_no_data():
    opps = discovery.scan(universe=["NONEXISTENT.NS"])
    assert opps == []


def test_discovery_universe_size():
    """Sanity check that the curated universe is meaningfully large."""
    assert len(discovery.DISCOVERY_UNIVERSE) >= 100


# ---- chart signal markers ----------------------------------------------------

def test_events_for_ticker_returns_list(tmp_db):
    _seed("EVT.NS")
    events = chart_signals.events_for_ticker("EVT.NS", lookback=200)
    assert isinstance(events, list)
    for e in events:
        assert e.setup
        assert e.direction == "buy"
        assert e.outcome in ("target", "stop", "open")
        assert isinstance(e.date, pd.Timestamp)


def test_events_empty_when_no_data():
    assert chart_signals.events_for_ticker("MISSING.NS") == []


def test_events_for_ticker_respects_lookback(tmp_db):
    _seed("EVT.NS")
    short_events = chart_signals.events_for_ticker("EVT.NS", lookback=80)
    long_events = chart_signals.events_for_ticker("EVT.NS", lookback=200)
    assert len(long_events) >= len(short_events)


# ---- investability summary ---------------------------------------------------

def test_summary_returns_none_without_data():
    s = summary_mod.summarize("UNKNOWN.NS")
    assert s is None


def test_summary_builds_fields(tmp_db):
    _seed("SUM.NS")
    storage.upsert_fundamentals([{
        "ticker": "SUM.NS", "short_name": "SUM CORP",
        "sector": "Technology", "market_cap": 1e11,
        "trailing_pe": 18, "return_on_equity": 0.20,
        "debt_to_equity": 0.3, "profit_margin": 0.15,
        "earnings_growth": 0.12, "revenue_growth": 0.10,
        "quality_score": 75,
    }])
    s = summary_mod.summarize("SUM.NS")
    assert s is not None
    assert s.ticker == "SUM.NS"
    assert s.quality_score == 75
    assert s.roe_pct == 20  # 0.20 * 100
    assert s.verdict in ("STRONG_BUY", "BUY", "WAIT", "AVOID")
    assert s.one_liner
    assert s.fundamental_brief
    # Quality 75 + trend (synthetic data trends up) → at least 1 reason
    assert len(s.why_invest) >= 1


def test_summary_verdict_avoid_for_weak_fundamentals(tmp_db):
    _seed("WEAK.NS")
    storage.upsert_fundamentals([{
        "ticker": "WEAK.NS", "short_name": "WEAK CORP",
        "sector": "Industrials", "market_cap": 1e10,
        "trailing_pe": 80, "return_on_equity": 0.02,
        "debt_to_equity": 3.0, "profit_margin": 0.01,
        "earnings_growth": -0.10, "revenue_growth": -0.05,
        "quality_score": 30,
    }])
    s = summary_mod.summarize("WEAK.NS")
    assert s is not None
    assert s.verdict == "AVOID"
    assert any("weak fundamentals" in r.lower() or "quality" in r.lower()
               for r in s.why_avoid)


def test_summary_fundamental_brief_without_fundamentals(tmp_db):
    """Even without fundamentals, summary should produce a verdict."""
    _seed("NOFUND.NS")
    s = summary_mod.summarize("NOFUND.NS")
    assert s is not None
    assert s.verdict == "WAIT"
    assert "run" in s.one_liner.lower() or "data" in s.one_liner.lower()
