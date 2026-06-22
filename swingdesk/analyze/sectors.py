"""Sector & micro-sector (industry) rotation — which corners of the market are
being bought, which are being sold, and the strongest stocks inside each.

The engine is relative-strength: for every tracked stock it takes a cheap
snapshot (1/3/6-month return, position vs the 50- and 200-EMA), then rolls those
up by **sector** (the 11 broad GICS buckets yfinance tags) and by **industry**
(the micro-sectors — "Banks - Regional", "Aerospace & Defense", "Specialty
Chemicals", …). Each group gets a 0-100 strength score blending *breadth* (how
many of its stocks are above their long trend) with *momentum* (median return),
and a Bullish / Neutral / Bearish label.

`top_stocks()` then surfaces the leaders inside any group, and
`event_picks()` answers the calendar question directly: for an upcoming
bullish, sector-tilted event, which currently-strong stocks sit in the sectors
that event historically favours.

All data is local (prices + fundamentals). A snapshot of ~250 names builds in a
second or two; the app caches it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk.storage import connect, load_fundamentals, load_prices

# Strength-score blend (group level).
W_BREADTH_200 = 0.35     # share above the 200-EMA — structural trend
W_BREADTH_50 = 0.20      # share above the 50-EMA — near-term trend
W_RET_3M = 0.30          # median 3-month return — momentum
W_RET_1M = 0.15          # median 1-month return — recent momentum

BULLISH = "Bullish"
BEARISH = "Bearish"
NEUTRAL = "Neutral"
BULL_CUT = 58.0          # strength >= this -> Bullish
BEAR_CUT = 42.0          # strength <= this -> Bearish

MIN_GROUP_SIZE = 3       # don't label a sector/industry off fewer than this many names


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _ret_to_score(r: float) -> float:
    """Map a percentage return to 0-100 (-15% -> 0, +30% -> 100)."""
    if r != r:  # NaN
        return 50.0
    return _clamp((r + 15.0) / 45.0 * 100.0)


def _ticker_snapshot(ticker: str) -> dict | None:
    """Cheap per-ticker read: returns + trend position. None if too little data."""
    df = load_prices(ticker)
    if df is None or df.empty or len(df) < 60:
        return None
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    if last <= 0:
        return None

    def ret(n: int) -> float:
        return (last / float(close.iloc[-1 - n]) - 1) * 100 if len(close) > n else float("nan")

    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = (float(close.ewm(span=200, adjust=False).mean().iloc[-1])
              if len(close) >= 150 else float("nan"))
    ret_1m, ret_3m, ret_6m = ret(21), ret(63), ret(126)

    above_50 = last > ema50
    above_200 = (last > ema200) if ema200 == ema200 else None

    # Per-stock strength (used to rank within a group).
    strength = (
        0.35 * (100 if above_200 else 0 if above_200 is not None else 50)
        + 0.15 * (100 if above_50 else 0)
        + 0.30 * _ret_to_score(ret_3m)
        + 0.20 * _ret_to_score(ret_1m)
    )
    return {
        "ticker": ticker,
        "last": round(last, 2),
        "ret_1m": round(ret_1m, 1) if ret_1m == ret_1m else None,
        "ret_3m": round(ret_3m, 1) if ret_3m == ret_3m else None,
        "ret_6m": round(ret_6m, 1) if ret_6m == ret_6m else None,
        "above_50": bool(above_50),
        "above_200": above_200,
        "stock_strength": round(strength, 1),
    }


def _priced_tickers() -> list[str]:
    with connect() as con:
        df = pd.read_sql_query("SELECT DISTINCT ticker FROM prices", con)
    return df["ticker"].tolist()


def build_snapshot(tickers: list[str] | None = None) -> pd.DataFrame:
    """One row per tradable stock: sector/industry + return/trend snapshot +
    quality. The base table every other function in this module groups over."""
    fund = load_fundamentals()
    if fund.empty:
        return pd.DataFrame()
    fund = fund[fund["sector"].notna()].copy()

    universe = tickers if tickers is not None else _priced_tickers()
    fund = fund[fund["ticker"].isin(universe)]

    rows = []
    for r in fund.itertuples():
        snap = _ticker_snapshot(r.ticker)
        if not snap:
            continue
        snap["sector"] = r.sector
        snap["industry"] = getattr(r, "industry", None)
        snap["short_name"] = getattr(r, "short_name", None)
        snap["quality_score"] = getattr(r, "quality_score", None)
        rows.append(snap)
    return pd.DataFrame(rows)


def _bias(strength: float) -> str:
    if strength >= BULL_CUT:
        return BULLISH
    if strength <= BEAR_CUT:
        return BEARISH
    return NEUTRAL


def rank_groups(snap: pd.DataFrame, by: str = "sector") -> pd.DataFrame:
    """Aggregate the snapshot into a ranked group table (sector or industry).

    Columns: <by>, n, breadth_200, breadth_50, med_ret_1m, med_ret_3m,
    strength, bias — sorted strongest first."""
    if snap.empty or by not in snap.columns:
        return pd.DataFrame()

    out = []
    for name, g in snap.groupby(by):
        n = len(g)
        if n < MIN_GROUP_SIZE:
            continue
        has200 = g["above_200"].dropna()
        breadth_200 = 100 * has200.mean() if len(has200) else 50.0
        breadth_50 = 100 * g["above_50"].mean()
        med_1m = float(g["ret_1m"].median())
        med_3m = float(g["ret_3m"].median())
        strength = (
            W_BREADTH_200 * breadth_200
            + W_BREADTH_50 * breadth_50
            + W_RET_3M * _ret_to_score(med_3m)
            + W_RET_1M * _ret_to_score(med_1m)
        )
        out.append({
            by: name,
            "n": n,
            "breadth_200": round(breadth_200),
            "breadth_50": round(breadth_50),
            "med_ret_1m": round(med_1m, 1),
            "med_ret_3m": round(med_3m, 1),
            "strength": round(strength, 1),
            "bias": _bias(strength),
        })
    df = pd.DataFrame(out).sort_values("strength", ascending=False).reset_index(drop=True)
    if not df.empty:
        df.insert(0, "rank", df.index + 1)
    return df


def top_stocks(snap: pd.DataFrame, *, sector: str | None = None,
               industry: str | None = None, n: int = 5,
               trending_only: bool = False) -> pd.DataFrame:
    """Strongest stocks inside a sector and/or industry, ranked by stock_strength.

    `trending_only` keeps only names above their 200-EMA (above_200 True)."""
    if snap.empty:
        return pd.DataFrame()
    g = snap
    if sector:
        g = g[g["sector"] == sector]
    if industry:
        g = g[g["industry"] == industry]
    if trending_only:
        g = g[g["above_200"] == True]  # noqa: E712 (pandas mask, not identity)
    cols = ["ticker", "short_name", "sector", "industry", "last",
            "ret_1m", "ret_3m", "ret_6m", "above_200", "quality_score", "stock_strength"]
    cols = [c for c in cols if c in g.columns]
    return g.sort_values("stock_strength", ascending=False)[cols].head(n).reset_index(drop=True)


def event_picks(snap: pd.DataFrame, sectors: list[str], n: int = 5) -> pd.DataFrame:
    """Top currently-strong stocks across an event's favoured sectors. With no
    sectors (a broad bullish event), returns market-wide leaders."""
    if snap.empty:
        return pd.DataFrame()
    g = snap[snap["sector"].isin(sectors)] if sectors else snap
    # Prefer names in genuine uptrends.
    trending = g[g["above_200"] == True]  # noqa: E712
    g = trending if len(trending) >= n else g
    return top_stocks(g, n=n)
