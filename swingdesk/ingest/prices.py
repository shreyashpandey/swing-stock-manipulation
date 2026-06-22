from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from rich.console import Console

from swingdesk.storage import upsert_prices

console = Console()


def fetch_one(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV for a single ticker via yfinance."""
    import yfinance as yf  # lazy: keep heavy import off app startup
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # yfinance sometimes returns MultiIndex columns when threads>1 or multi-ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        console.print(f"[red]price fetch failed for {ticker}: {e}[/red]")
        return pd.DataFrame()


def ingest(tickers: list[str], period: str = "2y", workers: int = 6) -> dict[str, int]:
    """Fetch and persist OHLCV for a list of tickers. Returns rows-written counts."""
    results: dict[str, int] = {}

    def _work(t: str) -> tuple[str, int]:
        df = fetch_one(t, period=period)
        if df.empty:
            return t, 0
        n = upsert_prices(t, df)
        return t, n

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_work, t): t for t in tickers}
        for fut in as_completed(futures):
            t, n = fut.result()
            results[t] = n
            console.print(f"  prices: {t:>15} -> {n:>5} rows")
            time.sleep(0.05)  # gentle rate limit
    return results
