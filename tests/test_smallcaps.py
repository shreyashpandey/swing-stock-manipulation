"""Tests for the small-cap scanner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingdesk import storage
from swingdesk.analyze import discovery, smallcaps


def _seed(ticker: str, n: int = 200):
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    base = 100 + np.cumsum(rng.normal(0.25, 1.0, n))
    opens = base + rng.normal(0, 0.3, n)
    closes = base + rng.normal(0, 0.3, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.1, 1.2, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.1, 1.2, n)
    vols = rng.integers(100_000, 1_000_000, n).astype(float)
    storage.upsert_prices(ticker, pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    }, index=idx))


def test_smallcap_universe_is_nontrivial():
    """The curated list should have meaningful breadth."""
    assert len(smallcaps.SMALLCAP_UNIVERSE) >= 50
    # All tickers should use the .NS suffix
    assert all(t.endswith(".NS") for t in smallcaps.SMALLCAP_UNIVERSE)


def test_smallcap_universe_distinct_from_discovery():
    """We want smallcaps SEPARATE from the large/mid-cap discovery list.
    Some overlap is fine but they shouldn't be identical sets."""
    sc = set(smallcaps.SMALLCAP_UNIVERSE)
    dc = set(discovery.DISCOVERY_UNIVERSE)
    overlap = sc & dc
    # ≤ 20% overlap — anything more would defeat the purpose
    assert len(overlap) / max(len(sc), 1) <= 0.20


def test_scan_excludes_held_and_watchlist(tmp_db):
    """Same exclusion rules as the main discovery scanner."""
    # Pick the first two universe tickers and place them in holdings/watchlist
    held, wl = smallcaps.SMALLCAP_UNIVERSE[0], smallcaps.SMALLCAP_UNIVERSE[1]
    storage.replace_holdings([{"ticker": held, "qty": 1, "avg_price": 1}])
    storage.set_watchlist([wl])
    # Seed both so the scanner would otherwise rank them
    _seed(held)
    _seed(wl)
    opps = smallcaps.scan()
    tickers = {o.ticker for o in opps}
    assert held not in tickers
    assert wl not in tickers


def test_scan_returns_empty_when_no_data(tmp_db):
    """Without any seeded price data, the scan returns an empty list."""
    opps = smallcaps.scan()
    assert opps == []


def test_scan_ranks_by_composite_score(tmp_db):
    """Two seeded smallcaps should come back sorted by composite descending."""
    _seed(smallcaps.SMALLCAP_UNIVERSE[0])
    _seed(smallcaps.SMALLCAP_UNIVERSE[1])
    opps = smallcaps.scan(exclude_held=False, exclude_watchlist=False)
    if len(opps) >= 2:
        scores = [o.composite_score for o in opps]
        assert scores == sorted(scores, reverse=True)


def test_high_conviction_filter_works(tmp_db):
    """high_conviction() returns only conviction=high above min_score."""
    opps = [
        discovery.Opportunity(ticker="X.NS", company="X", sector="Tech",
                              price=100, quality_score=80,
                              technical_state="uptrend", conviction="high",
                              composite_score=80.0),
        discovery.Opportunity(ticker="Y.NS", company="Y", sector="Tech",
                              price=100, quality_score=80,
                              technical_state="uptrend", conviction="medium",
                              composite_score=90.0),
    ]
    hc = smallcaps.high_conviction(opps, min_score=70.0)
    assert len(hc) == 1
    assert hc[0].ticker == "X.NS"


def test_liquidity_filter_drops_thin_names(tmp_db):
    """A name whose volume_x_avg < min_volume_x should be filtered out."""
    # Build a price frame with collapsing volume so vol_x at the end is low
    ticker = smallcaps.SMALLCAP_UNIVERSE[0]
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    base = 100 + np.cumsum(rng.normal(0.2, 1.0, 200))
    opens = base + rng.normal(0, 0.3, 200)
    closes = base + rng.normal(0, 0.3, 200)
    highs = np.maximum(opens, closes) + rng.uniform(0.1, 1.0, 200)
    lows = np.minimum(opens, closes) - rng.uniform(0.1, 1.0, 200)
    vols = np.full(200, 1_000_000.0)
    # Last bar's volume is way below recent average → vol_x < 0.5
    vols[-1] = 50_000
    storage.upsert_prices(ticker, pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    }, index=idx))

    opps = smallcaps.scan(exclude_held=False, exclude_watchlist=False,
                          min_volume_x=0.5)
    # The thin-volume ticker should NOT appear
    assert ticker not in {o.ticker for o in opps}


def test_smallcap_accessor():
    """Public accessor returns a copy of the universe."""
    u = smallcaps.smallcap_universe()
    assert isinstance(u, list)
    assert len(u) == len(smallcaps.SMALLCAP_UNIVERSE)
    # And it's a copy — mutating it shouldn't affect the module
    u.append("X.NS")
    assert "X.NS" not in smallcaps.SMALLCAP_UNIVERSE


# ---- news ticker-matcher picks up small-cap aliases ----------------------------

def test_news_matches_smallcap_company_names():
    """News headlines that say 'Astra Microwave' or 'ZEN Technologies' should
    tag the right small-cap ticker — without that the news layer is useless
    for small-cap analysis."""
    from swingdesk.ingest.news_rss import _match_tickers
    wl = ["ASTRAMICRO.NS", "ZENTEC.NS", "MTARTECH.NS", "PARAS.NS",
          "ANGELONE.NS", "LAURUSLABS.NS"]
    cases = [
        ("Astra Microwave wins ₹120 cr defense order",      "ASTRAMICRO.NS"),
        ("ZEN Technologies posts strong Q4 results",        "ZENTEC.NS"),
        ("MTAR Tech bags new ISRO contract",                "MTARTECH.NS"),
        ("Paras Defence raises ₹500 cr in QIP",             "PARAS.NS"),
        ("Angel One adds 1 million clients in May",         "ANGELONE.NS"),
        ("Laurus Labs Q3 profit beats estimates",           "LAURUSLABS.NS"),
    ]
    for headline, expected_ticker in cases:
        hits = _match_tickers(headline, wl)
        assert expected_ticker in hits, f"Expected {expected_ticker} for: {headline}"


def test_combined_universe_includes_smallcaps_when_requested(tmp_db):
    """combined_universe(include_smallcaps=True) should fold in the
    small-cap pool — used by `cli news` to tag small-cap headlines."""
    from swingdesk import storage
    storage.set_watchlist(["TCS.NS"])
    u = storage.combined_universe(include_smallcaps=True)
    assert "TCS.NS" in u
    # At least one well-known small cap should be in the result
    assert any(t in u for t in
               ("ASTRAMICRO.NS", "ZENTEC.NS", "ANGELONE.NS", "MTARTECH.NS"))


def test_combined_universe_default_excludes_smallcaps(tmp_db):
    from swingdesk import storage
    storage.set_watchlist(["TCS.NS"])
    u = storage.combined_universe()  # default: no small caps
    assert "TCS.NS" in u
    assert "ASTRAMICRO.NS" not in u
    assert "ZENTEC.NS" not in u


def test_combined_universe_dedupes(tmp_db):
    """If a small-cap is also in the watchlist, it shouldn't appear twice."""
    from swingdesk import storage
    storage.set_watchlist(["ASTRAMICRO.NS"])
    u = storage.combined_universe(include_smallcaps=True)
    assert u.count("ASTRAMICRO.NS") == 1
