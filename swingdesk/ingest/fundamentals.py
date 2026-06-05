"""Pull fundamental ratios per ticker via yfinance.

What yfinance gives us (reliably, for NSE large/mid caps):
    market_cap, trailing_pe, forward_pe, price_to_book
    return_on_equity, profit_margin, operating_margin
    earnings_growth (YoY), revenue_growth (YoY)
    debt_to_equity (sometimes None, especially for banks)
    current_ratio, dividend_yield, beta

What it does NOT give (you'd need Screener.in scraping for these):
    promoter holding, pledged %, governance flags
    5-year CAGRs (only YoY)
    quarterly trend
    sector-relative valuations

For banks/financials, debt/equity is structurally high (regulatory leverage).
We exclude D/E from the quality score for Financial-Services sector tickers.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
from rich.console import Console

from swingdesk.storage import upsert_fundamentals

console = Console()


def _safe(v):
    """yfinance sometimes returns None or NaN-like values — normalize to None."""
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN check
            return None
        return f
    except (TypeError, ValueError):
        return None


def _normalize_d_to_e(raw, sector: str | None) -> float | None:
    """yfinance D/E can be either a percentage (313.4 = 313%) or a fraction (1.5).
    For Indian large caps it's usually the percentage form. Convert to fraction.
    Banks/NBFCs get a free pass (D/E doesn't mean the same thing for them)."""
    if raw is None:
        return None
    if raw > 10:  # almost certainly percentage form
        return raw / 100
    return raw


def fetch_one(ticker: str) -> dict | None:
    try:
        info = yf.Ticker(ticker).info
        if not info or not info.get("shortName"):
            return None
        sector = info.get("sector")
        return {
            "ticker": ticker,
            "short_name": info.get("shortName"),
            "sector": sector,
            "industry": info.get("industry"),
            "market_cap": _safe(info.get("marketCap")),
            "trailing_pe": _safe(info.get("trailingPE")),
            "forward_pe": _safe(info.get("forwardPE")),
            "price_to_book": _safe(info.get("priceToBook")),
            "return_on_equity": _safe(info.get("returnOnEquity")),
            "debt_to_equity": _normalize_d_to_e(_safe(info.get("debtToEquity")), sector),
            "profit_margin": _safe(info.get("profitMargins")),
            "operating_margin": _safe(info.get("operatingMargins")),
            "earnings_growth": _safe(info.get("earningsGrowth")),
            "revenue_growth": _safe(info.get("revenueGrowth")),
            "current_ratio": _safe(info.get("currentRatio")),
            "dividend_yield": _safe(info.get("dividendYield")),
            "beta": _safe(info.get("beta")),
        }
    except Exception as e:
        console.print(f"[red]fundamentals failed for {ticker}: {e}[/red]")
        return None


def ingest(tickers: list[str], workers: int = 5) -> int:
    """Fetch fundamentals for many tickers in parallel. Returns count saved."""
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                rows.append(row)
                console.print(
                    f"  {row['ticker']:>15}  ROE={row['return_on_equity']!s:<6}  "
                    f"P/E={row['trailing_pe']!s:<6}  sector={row['sector']}"
                )
            time.sleep(0.05)
    # Quality score is computed AFTER we have the row — call analyze.quality.score
    from swingdesk.analyze.quality import score as quality_score
    for r in rows:
        r["quality_score"] = quality_score(r)
    upsert_fundamentals(rows)
    console.print(f"[green]saved fundamentals for {len(rows)} tickers[/green]")
    return len(rows)
