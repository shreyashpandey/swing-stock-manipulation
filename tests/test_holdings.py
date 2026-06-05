"""Tests for holdings import + analysis."""
from __future__ import annotations

import io

import pandas as pd
import pytest

from swingdesk import storage
from swingdesk.portfolio import holdings


# ---- parser -------------------------------------------------------------------

GROWW_CSV = """Stock Name,Symbol,Quantity,Average Buy Price,Current Price,Total Invested,Current Value,P&L,P&L %
Reliance Industries,RELIANCE,10,2500,2600,25000,26000,1000,4.0
Tata Consultancy Services,TCS,5,3800,4000,19000,20000,1000,5.26
HDFC Bank,HDFCBANK,15,1500,1450,22500,21750,-750,-3.33
"""


def test_parse_groww_format(tmp_path):
    p = tmp_path / "holdings.csv"
    p.write_text(GROWW_CSV)
    df = holdings.parse(p)
    assert len(df) == 3
    # Symbols got .NS suffix
    assert "RELIANCE.NS" in df["symbol"].values
    assert "TCS.NS" in df["symbol"].values
    # Derived/parsed numerics
    assert (df["qty"] > 0).all()
    assert (df["avg_price"] > 0).all()


def test_parse_missing_required_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Foo,Bar\n1,2\n")
    with pytest.raises(ValueError, match="missing required columns"):
        holdings.parse(p)


def test_parse_derives_invested_when_missing(tmp_path):
    """If CSV doesn't have 'Total Invested', it's derived as qty * avg_price."""
    p = tmp_path / "thin.csv"
    p.write_text("Symbol,Quantity,Average Buy Price\nRELIANCE,10,2500\n")
    df = holdings.parse(p)
    assert df.iloc[0]["invested"] == 25000


def test_parse_handles_excel_extension(tmp_path):
    """Excel files route through pd.read_excel."""
    # Build a tiny xlsx
    p = tmp_path / "h.xlsx"
    pd.DataFrame({
        "Symbol": ["TCS"],
        "Quantity": [10],
        "Average Buy Price": [3800],
    }).to_excel(p, index=False)
    df = holdings.parse(p)
    assert df.iloc[0]["symbol"] == "TCS.NS"


def test_parse_column_overrides(tmp_path):
    p = tmp_path / "weird.csv"
    p.write_text("Stock,Shares,Buy\nRELIANCE,10,2500\n")
    df = holdings.parse(p, overrides={"symbol": "Stock", "qty": "Shares",
                                       "avg_price": "Buy"})
    assert df.iloc[0]["symbol"] == "RELIANCE.NS"


def test_parse_skips_groww_preamble(tmp_path):
    """Groww's real export has a metadata preamble before the table.
    The parser must auto-detect the real header row."""
    p = tmp_path / "groww_real.csv"
    p.write_text(
        "Name,Shreyash Pandey,,,,,,\n"
        "PAN,ABCDE1234F,,,,,,\n"
        "Generated on,01-Jun-2026,,,,,,\n"
        "\n"
        "Stock Name,ISIN,Quantity,Average Buy Price,Current Price,Total Invested,Current Value,P&L\n"
        "Reliance Industries,INE002A01018,10,2500,2600,25000,26000,1000\n"
        "Tata Consultancy Services,INE467B01029,5,3800,4000,19000,20000,1000\n"
    )
    df = holdings.parse(p)
    assert len(df) == 2
    # Company names should be resolved to NSE tickers via the name lookup
    assert "RELIANCE.NS" in df["symbol"].values
    assert "TCS.NS" in df["symbol"].values
    # qty and avg_price should be properly numeric
    assert df["qty"].sum() == 15
    assert df["avg_price"].min() > 0


def test_normalize_symbol_resolves_company_names():
    """Company names → NSE tickers via the resolver."""
    assert holdings._normalize_symbol("Reliance Industries Limited") == "RELIANCE.NS"
    assert holdings._normalize_symbol("Tata Consultancy Services") == "TCS.NS"
    assert holdings._normalize_symbol("HDFC Bank Ltd") == "HDFCBANK.NS"
    assert holdings._normalize_symbol("State Bank of India") == "SBIN.NS"
    assert holdings._normalize_symbol("Bharat Electronics Ltd") == "BEL.NS"


def test_normalize_symbol_passes_through_real_tickers():
    """Existing tickers must not be mangled by the resolver."""
    assert holdings._normalize_symbol("RELIANCE") == "RELIANCE.NS"
    assert holdings._normalize_symbol("RELIANCE.NS") == "RELIANCE.NS"
    assert holdings._normalize_symbol("RELIANCE-EQ") == "RELIANCE.NS"
    assert holdings._normalize_symbol("M&M") == "M&M.NS"


def test_parse_handles_blank_lines_in_preamble(tmp_path):
    """Empty lines mixed into the preamble shouldn't confuse the detector."""
    p = tmp_path / "messy.csv"
    p.write_text(
        "Holdings Report,,,\n"
        "\n"
        "Account,12345,,\n"
        "\n"
        "\n"
        "Stocks,Quantity,Average Buy Price,Current Price\n"
        "RELIANCE,10,2500,2600\n"
    )
    df = holdings.parse(p)
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "RELIANCE.NS"


def test_parse_strips_eq_suffix(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("Symbol,Quantity,Average Buy Price\nRELIANCE-EQ,10,2500\n")
    df = holdings.parse(p)
    assert df.iloc[0]["symbol"] == "RELIANCE.NS"


# ---- import_csv writes to DB --------------------------------------------------

def test_import_csv_writes_holdings_table(tmp_db, tmp_path):
    p = tmp_path / "h.csv"
    p.write_text(GROWW_CSV)
    n = holdings.import_csv(p)
    assert n == 3
    df = storage.load_holdings()
    assert len(df) == 3
    assert set(df["ticker"]) == {"RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"}


def test_reimport_replaces_holdings(tmp_db, tmp_path):
    """Holdings are snapshots — second import wipes the first."""
    p1 = tmp_path / "h1.csv"
    p1.write_text(GROWW_CSV)
    holdings.import_csv(p1)

    p2 = tmp_path / "h2.csv"
    p2.write_text("Symbol,Quantity,Average Buy Price\nINFY,8,1500\n")
    holdings.import_csv(p2)

    df = storage.load_holdings()
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "INFY.NS"


# ---- analysis lenses ----------------------------------------------------------

def test_quality_verdict_classifies():
    assert holdings._quality_verdict(80) == "strong"
    assert holdings._quality_verdict(65) == "ok"
    assert holdings._quality_verdict(40) == "weak"
    assert holdings._quality_verdict(None) == "unknown"


def test_technical_state_unknown_when_no_data():
    state, *_ = holdings._technical_state(pd.DataFrame())
    assert state == "unknown"


# ---- recommendation engine ----------------------------------------------------

def test_recommend_strong_buy():
    """High quality + uptrend + bullish news + active setup → BUY_MORE."""
    a = holdings.HoldingAnalysis(
        ticker="TCS.NS", qty=10, avg_price=4000, last_price=4100,
        pnl_pct=2.5, portfolio_weight=0.10,
        quality_score=88, quality_verdict="strong",
        technical_state="uptrend", rsi=58, above_50ema=True, above_200ema=True,
        sentiment_bullish=2, sentiment_bearish=0,
        active_setup="breakout_20d",
    )
    rec, reasons = holdings._recommend(a)
    assert rec == "BUY_MORE"
    assert any("uptrend" in r for r in reasons)


def test_recommend_clear_sell():
    """Weak fundamentals + broken technicals + bearish news → SELL."""
    a = holdings.HoldingAnalysis(
        ticker="BAD.NS", qty=10, avg_price=100, last_price=80,
        pnl_pct=-20, portfolio_weight=0.05,
        quality_score=35, quality_verdict="weak",
        technical_state="broken", rsi=28, above_50ema=False, above_200ema=False,
        sentiment_bullish=0, sentiment_bearish=3,
        active_setup=None,
    )
    rec, _ = holdings._recommend(a)
    assert rec == "SELL"


def test_recommend_reduce_when_oversized():
    """Otherwise-ok stock that's grown to >35% of portfolio → REDUCE."""
    a = holdings.HoldingAnalysis(
        ticker="WINNER.NS", qty=100, avg_price=100, last_price=300,
        pnl_pct=200, portfolio_weight=0.45,
        quality_score=70, quality_verdict="ok",
        technical_state="weakening", rsi=55, above_50ema=False, above_200ema=True,
        sentiment_bullish=0, sentiment_bearish=0,
    )
    rec, reasons = holdings._recommend(a)
    assert rec in ("REDUCE", "SELL")
    assert any("oversized" in r for r in reasons)


def test_recommend_hold_when_neutral():
    """Mid-grade everything → HOLD."""
    a = holdings.HoldingAnalysis(
        ticker="MID.NS", qty=10, avg_price=100, last_price=102,
        pnl_pct=2, portfolio_weight=0.10,
        quality_score=65, quality_verdict="ok",
        technical_state="ok", rsi=52, above_50ema=True, above_200ema=True,
        sentiment_bullish=0, sentiment_bearish=0,
    )
    rec, _ = holdings._recommend(a)
    assert rec == "HOLD"


def test_recommend_no_data_returns_no_data():
    """If we genuinely have nothing → NO_DATA."""
    a = holdings.HoldingAnalysis(
        ticker="OBSCURE.NS", qty=10, avg_price=100, last_price=None,
        pnl_pct=None, portfolio_weight=None,
        quality_score=None, quality_verdict="unknown",
        technical_state="unknown", rsi=None,
        above_50ema=None, above_200ema=None,
    )
    rec, _ = holdings._recommend(a)
    assert rec == "NO_DATA"


# ---- portfolio_summary --------------------------------------------------------

def test_portfolio_summary_aggregates_recommendations(tmp_db):
    storage.upsert_fundamentals([
        {"ticker": "A.NS", "sector": "Technology", "quality_score": 88},
        {"ticker": "B.NS", "sector": "Technology", "quality_score": 80},
        {"ticker": "C.NS", "sector": "Industrials", "quality_score": 40},
    ])
    analyses = [
        holdings.HoldingAnalysis(ticker="A.NS", qty=10, avg_price=100,
                                 last_price=100, pnl_pct=0, portfolio_weight=0.40,
                                 quality_score=88, quality_verdict="strong",
                                 technical_state="uptrend", rsi=55,
                                 above_50ema=True, above_200ema=True,
                                 recommendation="BUY_MORE"),
        holdings.HoldingAnalysis(ticker="B.NS", qty=10, avg_price=100,
                                 last_price=100, pnl_pct=0, portfolio_weight=0.30,
                                 quality_score=80, quality_verdict="strong",
                                 technical_state="ok", rsi=50,
                                 above_50ema=True, above_200ema=True,
                                 recommendation="HOLD"),
        holdings.HoldingAnalysis(ticker="C.NS", qty=10, avg_price=100,
                                 last_price=100, pnl_pct=0, portfolio_weight=0.30,
                                 quality_score=40, quality_verdict="weak",
                                 technical_state="broken", rsi=25,
                                 above_50ema=False, above_200ema=False,
                                 recommendation="SELL"),
    ]
    summary = holdings.portfolio_summary(analyses)
    assert summary["n_holdings"] == 3
    assert "A.NS" in summary["buy_more"]
    assert "B.NS" in summary["hold"]
    assert "C.NS" in summary["sell"]
    # A.NS is 40% — concentrated
    assert "A.NS" in summary["concentrated_positions"]
    # Tech = 70%, > 40% → concentrated sector
    assert "Technology" in summary["concentrated_sectors"]


# ---- analyze_one integration --------------------------------------------------

def test_analyze_one_uses_all_lenses(tmp_db):
    """Smoke test that analyze_one assembles results from all 4 modules."""
    storage.upsert_fundamentals([
        {"ticker": "TEST.NS", "sector": "Technology", "quality_score": 75},
    ])
    a = holdings.analyze_one(
        ticker="TEST.NS", qty=10, avg_price=100, last_price=102,
        portfolio_value=10000, pnl_pct=2.0,
    )
    assert a.ticker == "TEST.NS"
    assert a.quality_score == 75
    assert a.quality_verdict == "strong"
    # No price data → technical_state should be "unknown"
    assert a.technical_state == "unknown"
    # No news → both 0
    assert a.sentiment_bullish == 0
    assert a.sentiment_bearish == 0
    # Recommendation produced
    assert a.recommendation in ("BUY_MORE", "HOLD", "REDUCE", "SELL", "NO_DATA")
