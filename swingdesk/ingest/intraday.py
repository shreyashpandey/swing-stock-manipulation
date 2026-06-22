"""Intraday OHLCV ingest via yfinance.

Honest about the source: yfinance intraday is **delayed (~15 min)** and rate-
limited, and only goes back a limited window (1m: ~7d, 5m/15m: ~60d). That is
fine for monitoring, opening-range-breakout and VWAP setups on a 15-30 minute
cadence — NOT for scalping. For real-time you'd swap this for a broker API
(Groww/Kite/Dhan), keeping the same storage shape so the UI never changes.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from rich.console import Console

from swingdesk.storage import upsert_intraday

console = Console()

# yfinance caps how far back each interval can go. Keep requests within limits.
PERIOD_FOR = {"5m": "5d", "15m": "30d", "1h": "60d"}


def fetch_one(ticker: str, interval: str = "5m",
              period: str | None = None) -> pd.DataFrame:
    """Fetch intraday OHLCV for one ticker. Returns empty on failure."""
    period = period or PERIOD_FOR.get(interval, "5d")
    import yfinance as yf  # lazy: keep heavy import off app startup
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        console.print(f"[red]intraday fetch failed for {ticker}: {e}[/red]")
        return pd.DataFrame()


def ingest(tickers: list[str], interval: str = "5m",
           workers: int = 6) -> dict[str, int]:
    """Fetch + persist intraday bars for a list of tickers."""
    results: dict[str, int] = {}

    def _work(t: str) -> tuple[str, int]:
        df = fetch_one(t, interval=interval)
        if df.empty:
            return t, 0
        return t, upsert_intraday(t, df, interval)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_work, t): t for t in tickers}
        for fut in as_completed(futures):
            t, n = fut.result()
            results[t] = n
            console.print(f"  intraday {interval}: {t:>15} -> {n:>5} bars")
            time.sleep(0.05)
    return results
