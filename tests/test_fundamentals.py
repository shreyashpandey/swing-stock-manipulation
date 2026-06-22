"""Tests for quality scoring + fundamentals storage."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from swingdesk import storage
from swingdesk.analyze import quality


# ---- quality score components --------------------------------------------------

def test_score_roe_curve():
    # Values along the piecewise curve
    assert quality._score_roe(0) == 0
    assert quality._score_roe(0.05) == 20
    assert quality._score_roe(0.12) == 50
    assert quality._score_roe(0.18) == 75
    assert quality._score_roe(0.25) == 90
    assert quality._score_roe(0.35) == 100   # caps at 100
    assert quality._score_roe(-0.10) == 0    # negative ROE = 0
    assert quality._score_roe(None) is None


def test_score_growth_uses_average():
    # 10% earnings + 10% revenue = avg 10% → 60
    assert quality._score_growth(0.10, 0.10) == 60
    # Single value works
    assert quality._score_growth(0.10, None) == 60
    # Negative growth handled
    assert quality._score_growth(-0.20, -0.20) == 0
    # Big growth caps at 100
    assert quality._score_growth(0.30, 0.30) == 100
    assert quality._score_growth(None, None) is None


def test_score_margins():
    assert quality._score_margins(0) == 0
    assert quality._score_margins(0.10) == 55
    assert quality._score_margins(0.30) == 100  # caps
    assert quality._score_margins(-0.05) == 0
    assert quality._score_margins(None) is None


def test_score_valuation_peaks_in_middle():
    # Sub-15 P/E = sweet spot
    assert quality._score_valuation(15) == 90
    # Very high P/E penalised
    assert quality._score_valuation(60) == 25
    # Very low P/E not max (could be value trap)
    assert quality._score_valuation(5) == 70
    assert quality._score_valuation(None) is None
    assert quality._score_valuation(0) is None


def test_score_debt_excluded_for_financials():
    # Banks get None for debt (excluded from composite)
    assert quality._score_debt(2.0, "Financial Services") is None
    # Non-financials get a real score
    assert quality._score_debt(0.1, "Technology") == 100
    assert quality._score_debt(2.0, "Industrials") == 20  # D/E at edge of pain
    assert quality._score_debt(3.0, "Industrials") == 5   # max-debt floor


def test_score_size():
    # Various market caps in INR
    assert quality._score_size(1e9) == 20    # ₹100 cr → small
    assert quality._score_size(1e11) == 70   # ₹10,000 cr → mid
    assert quality._score_size(1e13) == 95   # ₹10 lakh cr → mega
    assert quality._score_size(None) is None


# ---- composite quality --------------------------------------------------------

def test_quality_score_blue_chip():
    """A textbook strong company should score 80+."""
    f = {
        "ticker": "TCS.NS", "sector": "Technology",
        "return_on_equity": 0.48,     # great
        "earnings_growth": 0.12, "revenue_growth": 0.10,
        "profit_margin": 0.18,
        "trailing_pe": 17,
        "debt_to_equity": 0.10,
        "market_cap": 8e12,           # mega-cap
    }
    s = quality.score(f)
    assert s is not None
    assert s >= 80, f"expected blue-chip score >= 80, got {s}"


def test_quality_score_weak_company():
    """A weak company should score below 40."""
    f = {
        "ticker": "WEAK.NS", "sector": "Industrials",
        "return_on_equity": 0.02,     # poor
        "earnings_growth": -0.10, "revenue_growth": -0.05,
        "profit_margin": 0.01,
        "trailing_pe": 80,            # expensive
        "debt_to_equity": 2.5,        # leveraged
        "market_cap": 5e9,            # small cap
    }
    s = quality.score(f)
    assert s is not None
    assert s <= 40, f"expected weak score <= 40, got {s}"


def test_quality_score_none_when_insufficient_data():
    f = {"ticker": "MYSTERY.NS", "return_on_equity": 0.15}  # only one metric
    assert quality.score(f) is None


def test_quality_score_handles_bank_without_de():
    """Financials with missing D/E should still score (D/E is excluded)."""
    f = {
        "ticker": "HDFCBANK.NS", "sector": "Financial Services",
        "return_on_equity": 0.18,
        "earnings_growth": 0.10, "revenue_growth": 0.08,
        "profit_margin": 0.27,
        "trailing_pe": 17,
        "debt_to_equity": None,
        "market_cap": 1e13,
    }
    s = quality.score(f)
    assert s is not None
    assert s >= 70


# ---- passes_quality_bar -------------------------------------------------------

def test_quality_bar_pass():
    f = {
        "return_on_equity": 0.20, "earnings_growth": 0.15,
        "revenue_growth": 0.12, "profit_margin": 0.15,
        "trailing_pe": 18, "debt_to_equity": 0.3,
        "market_cap": 5e11, "sector": "Technology",
    }
    ok, fails = quality.passes_quality_bar(f, min_score=60)
    assert ok is True
    assert fails == []


def test_quality_bar_fails_on_low_score():
    f = {"return_on_equity": 0.02, "profit_margin": 0.01,
         "trailing_pe": 80, "debt_to_equity": 3.0,
         "market_cap": 5e9, "earnings_growth": -0.1,
         "revenue_growth": -0.1, "sector": "Industrials"}
    ok, fails = quality.passes_quality_bar(f, min_score=60)
    assert ok is False
    assert any("quality" in r for r in fails)


def test_quality_bar_hard_filter_on_low_roe():
    f = {"return_on_equity": 0.01, "profit_margin": 0.20,
         "trailing_pe": 20, "debt_to_equity": 0.2,
         "market_cap": 1e12, "earnings_growth": 0.20,
         "revenue_growth": 0.20, "sector": "Technology"}
    ok, fails = quality.passes_quality_bar(f, hard_filters=True, min_score=0)
    # Even with high quality_score, ROE < 5% must fail the hard filter
    assert any("ROE" in r for r in fails) or ok is False


# ---- storage roundtrip --------------------------------------------------------

def test_upsert_and_load_fundamentals(tmp_db):
    rows = [
        {"ticker": "A.NS", "short_name": "Aco", "sector": "Tech",
         "industry": None, "market_cap": 1e12, "trailing_pe": 20,
         "forward_pe": None, "price_to_book": 5, "return_on_equity": 0.18,
         "debt_to_equity": 0.3, "profit_margin": 0.15, "operating_margin": 0.18,
         "earnings_growth": 0.12, "revenue_growth": 0.10, "current_ratio": 2.0,
         "dividend_yield": 0.02, "beta": 1.1, "quality_score": 75.0},
        {"ticker": "B.NS", "short_name": "Bco", "sector": "Bank",
         "industry": None, "market_cap": 5e11, "trailing_pe": 15,
         "forward_pe": None, "price_to_book": 2, "return_on_equity": 0.15,
         "debt_to_equity": None, "profit_margin": 0.22, "operating_margin": 0.25,
         "earnings_growth": 0.08, "revenue_growth": 0.05, "current_ratio": None,
         "dividend_yield": 0.01, "beta": 0.9, "quality_score": 70.0},
    ]
    storage.upsert_fundamentals(rows)
    df = storage.load_fundamentals()
    assert len(df) == 2

    # Sorted by quality DESC
    assert df.iloc[0]["ticker"] == "A.NS"

    # Single-ticker retrieval
    f = storage.get_fundamentals("B.NS")
    assert f is not None
    assert f["short_name"] == "Bco"


def test_load_fundamentals_min_quality_filter(tmp_db):
    storage.upsert_fundamentals([
        {"ticker": "HIGH.NS", "quality_score": 85},
        {"ticker": "MID.NS", "quality_score": 60},
        {"ticker": "LOW.NS", "quality_score": 40},
    ])
    df = storage.load_fundamentals(min_quality=70)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "HIGH.NS"


# ---- score.enrich uses quality when available --------------------------------

def test_enrich_uses_quality_score(tmp_db):
    """enrich() should give a quality-aware composite when fundamentals exist."""
    from swingdesk.analyze import score
    storage.upsert_fundamentals([
        {"ticker": "Q.NS", "quality_score": 90},  # very strong
    ])
    signals = [{"ticker": "Q.NS", "setup": "x", "score": 70.0, "notes": ""}]
    enriched = score.enrich(signals)
    e = enriched[0]
    assert "quality_score" in e
    assert e["quality_score"] == 90.0
    # composite = 0.55*70 + 0.20*50 + 0.25*90 = 38.5 + 10 + 22.5 = 71.0
    assert abs(e["composite_score"] - 71.0) < 0.5


def test_enrich_falls_back_when_no_quality(tmp_db):
    """When fundamentals are missing, composite uses just tech + sentiment."""
    from swingdesk.analyze import score
    signals = [{"ticker": "NOFUND.NS", "setup": "x", "score": 70.0, "notes": ""}]
    enriched = score.enrich(signals)
    e = enriched[0]
    # Without quality: composite = 0.7*70 + 0.3*50 = 49 + 15 = 64
    assert abs(e["composite_score"] - 64.0) < 0.5
    assert "quality_score" not in e


# ---- ingest with mocked yfinance ---------------------------------------------

def test_ingest_handles_yfinance_payload(tmp_db):
    from swingdesk.ingest import fundamentals as fi
    fake_info = {
        "shortName": "TEST CORP",
        "sector": "Technology",
        "industry": "Software",
        "marketCap": 1e12,
        "trailingPE": 25,
        "forwardPE": 20,
        "priceToBook": 5,
        "returnOnEquity": 0.20,
        "debtToEquity": 30.0,  # yfinance often returns as percentage
        "profitMargins": 0.15,
        "operatingMargins": 0.18,
        "earningsGrowth": 0.12,
        "revenueGrowth": 0.10,
        "currentRatio": 2.0,
        "dividendYield": 0.015,
        "beta": 1.1,
    }
    # yfinance is lazily imported inside fetch_one, so patch the real module.
    with patch("yfinance.Ticker") as MockT:
        MockT.return_value.info = fake_info
        n = fi.ingest(["TEST.NS"], workers=1)
    assert n == 1
    f = storage.get_fundamentals("TEST.NS")
    assert f["short_name"] == "TEST CORP"
    # D/E percentage form (30) should have been normalised to 0.3
    assert abs(f["debt_to_equity"] - 0.3) < 0.01
    # Quality score was computed
    assert f["quality_score"] is not None
