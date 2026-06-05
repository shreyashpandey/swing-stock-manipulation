"""Pull macro indicators that drive Indian stock prices.

Indian factors:
    NIFTY 50          — broad market direction
    NIFTY BANK        — financial-sector pulse
    NIFTY IT          — IT services pulse (USD-INR sensitive)
    NIFTY AUTO        — auto demand
    INDIA VIX         — fear gauge

International factors:
    USD/INR           — rupee weakness helps IT exports, hurts importers
    Brent crude       — oil price; hurts importers (RELIANCE, BPCL), helps ONGC
    S&P 500           — global risk-on/off
    NASDAQ            — US tech sentiment (correlates with INFY, TCS)
    Dow Jones         — broad US market
    Gold              — safe-haven proxy

Per-stock correlation: for each holding, we compute Pearson correlation of
60-day returns against each macro indicator. Highest |r| factors are
flagged as "primary drivers" so the user understands WHY a stock is moving.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf
from rich.console import Console

from swingdesk.storage import load_macro, load_prices, upsert_macro

console = Console()

# (yfinance_ticker, display_name, category)
MACRO_TICKERS = [
    # Indian indices
    ("^NSEI",       "NIFTY 50",       "indian"),
    ("^NSEBANK",    "NIFTY BANK",     "indian"),
    ("^CNXIT",      "NIFTY IT",       "indian"),
    ("^CNXAUTO",    "NIFTY AUTO",     "indian"),
    ("^INDIAVIX",   "INDIA VIX",      "indian"),
    # FX / commodities
    ("INR=X",       "USD/INR",        "fx"),
    ("BZ=F",        "Brent Crude",    "commodity"),
    ("GC=F",        "Gold",           "commodity"),
    # US markets
    ("^GSPC",       "S&P 500",        "us"),
    ("^IXIC",       "NASDAQ",         "us"),
    ("^DJI",        "Dow Jones",      "us"),
]


def ingest(period: str = "1y") -> dict[str, int]:
    """Fetch all macro tickers. Returns rows-written per ticker."""
    out = {}
    for tk, name, _ in MACRO_TICKERS:
        try:
            df = yf.download(tk, period=period, interval="1d",
                             progress=False, auto_adjust=False, threads=False)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            n = upsert_macro(tk, df)
            out[tk] = n
            console.print(f"  macro: {name:>20} ({tk:>10}) -> {n:>4} bars")
        except Exception as e:
            console.print(f"[red]macro fetch failed for {tk}: {e}[/red]")
    console.print(f"[green]macro: {len(out)} indicators[/green]")
    return out


def _returns(close: pd.Series, days: int = 60) -> pd.Series:
    """Daily log returns over the last `days` bars."""
    s = close.tail(days + 1)
    return s.pct_change().dropna()


def correlations(ticker: str, days: int = 60) -> pd.DataFrame:
    """Pearson correlation between a stock's returns and each macro's returns.

    Returns a DataFrame ordered by |r| descending. Used to identify the
    primary drivers behind a stock's moves.
    """
    stock_df = load_prices(ticker)
    if stock_df.empty or len(stock_df) < days:
        return pd.DataFrame()

    stock_returns = _returns(stock_df["close"], days=days)

    rows = []
    for tk, name, cat in MACRO_TICKERS:
        macro_df = load_macro(tk)
        if macro_df.empty:
            continue
        macro_returns = _returns(macro_df["close"], days=days)
        # Align on dates
        joined = pd.concat([stock_returns, macro_returns], axis=1).dropna()
        if len(joined) < 20:
            continue
        r = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        if pd.isna(r):
            continue
        rows.append({
            "factor": name, "ticker": tk, "category": cat,
            "correlation": round(r, 3),
            "abs_corr": round(abs(r), 3),
            "n_obs": len(joined),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def market_pulse() -> dict:
    """Quick read of the overall market state — used for narrative context."""
    pulse = {}
    for tk, name, cat in MACRO_TICKERS:
        df = load_macro(tk, days=30)
        if df.empty or len(df) < 5:
            continue
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest
        wk_ago = df.iloc[-5] if len(df) >= 5 else prev
        chg_1d = (latest["close"] - prev["close"]) / prev["close"] * 100
        chg_1w = (latest["close"] - wk_ago["close"]) / wk_ago["close"] * 100
        pulse[name] = {
            "close": round(float(latest["close"]), 2),
            "chg_1d": round(chg_1d, 2),
            "chg_1w": round(chg_1w, 2),
            "category": cat,
        }
    return pulse
