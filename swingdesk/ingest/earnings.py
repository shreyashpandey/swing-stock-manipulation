"""Pull next/last earnings dates per ticker via yfinance.

yfinance exposes earnings_dates as a DataFrame indexed by date. We pick the
nearest future date as `next_earnings` and the most recent past date as
`last_earnings`. Used by the position-open guard to block trades within
N days of an earnings event (huge price-gap risk).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from rich.console import Console

from swingdesk.storage import upsert_earnings

console = Console()


def fetch_one(ticker: str) -> tuple[str | None, str | None]:
    """Return (next_earnings_iso, last_earnings_iso). Either may be None."""
    try:
        t = yf.Ticker(ticker)
        df = t.earnings_dates
        if df is None or df.empty:
            return None, None
        # Index is a DatetimeIndex (timezone-aware). Compare to now in UTC.
        now = pd.Timestamp.now(tz="UTC")
        idx = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        future = idx[idx >= now]
        past = idx[idx < now]
        next_e = future.min().strftime("%Y-%m-%d") if len(future) else None
        last_e = past.max().strftime("%Y-%m-%d") if len(past) else None
        return next_e, last_e
    except Exception as e:
        console.print(f"[red]earnings fetch failed for {ticker}: {e}[/red]")
        return None, None


def ingest(tickers: list[str]) -> int:
    """Refresh earnings_calendar for each ticker. Returns count updated."""
    n = 0
    for t in tickers:
        nxt, last = fetch_one(t)
        upsert_earnings(t, nxt, last)
        if nxt:
            console.print(f"  earnings: {t:>15}  next={nxt}  last={last or '-'}")
            n += 1
    console.print(f"[green]{n} tickers have upcoming earnings dates[/green]")
    return n


def days_to_earnings(ticker: str) -> int | None:
    """Return integer days until next earnings, or None if unknown / past."""
    from swingdesk.storage import get_next_earnings
    iso = get_next_earnings(ticker)
    if not iso:
        return None
    try:
        e = datetime.fromisoformat(iso).date()
        today = datetime.now().date()
        delta = (e - today).days
        return delta if delta >= 0 else None
    except Exception:
        return None
