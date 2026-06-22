"""Intraday setup detection — VWAP, Opening-Range Breakout (ORB), relative
volume and gap context.

Designed for a 5/15-minute monitoring cadence (yfinance is ~15 min delayed), so
these are *positioning/alert* reads, not scalping triggers. They pair with the
US→India overnight-gap model: a gap-and-go that holds above VWAP is the classic
morning continuation, and the spillover engine tells you the gap was coming.

All numpy/pandas. VWAP resets each session; the opening range is the first few
bars of the latest session.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from swingdesk.storage import load_intraday


def _session_key(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.normalize().values


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP (resets each trading day)."""
    if df.empty:
        return pd.Series(dtype=float)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    day = pd.Series(_session_key(df.index), index=df.index)
    cum_pv = pv.groupby(day).cumsum()
    cum_v = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return cum_pv / cum_v


def opening_range(session: pd.DataFrame, or_bars: int = 3) -> dict | None:
    """High/low of the first `or_bars` bars of a single session."""
    if session.empty or len(session) < 1:
        return None
    head = session.iloc[:or_bars]
    return {"or_high": float(head["high"].max()),
            "or_low": float(head["low"].min()),
            "or_bars": len(head)}


def relative_volume(df: pd.DataFrame) -> float:
    """Latest bar's volume vs the median bar volume — >1.5 = unusual activity."""
    v = df["volume"].dropna()
    if len(v) < 10:
        return float("nan")
    med = float(v.median())
    return float(v.iloc[-1] / med) if med > 0 else float("nan")


@dataclass
class IntradaySignal:
    ticker: str
    asof: str
    last: float
    vwap: float
    dist_vwap_pct: float        # last vs VWAP, %
    or_high: float
    or_low: float
    setup: str                  # ORB long/short, VWAP reclaim/reject, inside range
    bias: str                   # "long" | "short" | "neutral"
    gap_pct: float              # latest session open vs prior session close
    rvol: float
    narrative: str


def _sessions(df: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(pd.unique(df.index.normalize()))


def intraday_signals(ticker: str, interval: str = "5m",
                     or_bars: int = 3) -> IntradaySignal | None:
    """Read the current intraday state for one ticker's latest session."""
    df = load_intraday(ticker, interval=interval)
    if df.empty or len(df) < 5:
        return None
    df = df.sort_index()
    vw = vwap(df)
    days = _sessions(df)
    last_day = days[-1]
    session = df[df.index.normalize() == last_day]
    if session.empty:
        return None

    orr = opening_range(session, or_bars=or_bars)
    last = float(session["close"].iloc[-1])
    last_vwap = float(vw.loc[session.index[-1]])
    dist_vwap = (last / last_vwap - 1) * 100 if last_vwap else 0.0

    # Gap: latest session's first open vs prior session's last close.
    gap_pct = 0.0
    if len(days) >= 2:
        prev = df[df.index.normalize() == days[-2]]
        if not prev.empty:
            prev_close = float(prev["close"].iloc[-1])
            sess_open = float(session["open"].iloc[0])
            if prev_close:
                gap_pct = (sess_open / prev_close - 1) * 100

    rvol = relative_volume(df)
    above_vwap = last >= last_vwap

    setup, bias = "inside range — no setup", "neutral"
    if orr:
        if last > orr["or_high"] and above_vwap:
            setup, bias = "ORB long (broke opening-range high, above VWAP)", "long"
        elif last < orr["or_low"] and not above_vwap:
            setup, bias = "ORB short (broke opening-range low, below VWAP)", "short"
        elif above_vwap and dist_vwap < 0.3:
            setup, bias = "VWAP reclaim (holding above)", "long"
        elif not above_vwap and dist_vwap > -0.3:
            setup, bias = "VWAP rejection (capped below)", "short"

    vstr = "above" if above_vwap else "below"
    narrative = (
        f"{ticker}: ₹{last:,.1f}, {vstr} VWAP ({dist_vwap:+.2f}%). "
        f"Opening range ₹{orr['or_low']:,.1f}–₹{orr['or_high']:,.1f}. "
        f"Gap {gap_pct:+.2f}% · RVOL {rvol:.1f}×. → {setup}."
    ) if orr else f"{ticker}: ₹{last:,.1f}, {vstr} VWAP."

    return IntradaySignal(
        ticker=ticker, asof=session.index[-1].strftime("%Y-%m-%d %H:%M"),
        last=round(last, 2), vwap=round(last_vwap, 2),
        dist_vwap_pct=round(dist_vwap, 2),
        or_high=round(orr["or_high"], 2) if orr else float("nan"),
        or_low=round(orr["or_low"], 2) if orr else float("nan"),
        setup=setup, bias=bias, gap_pct=round(gap_pct, 2),
        rvol=round(rvol, 2) if np.isfinite(rvol) else float("nan"),
        narrative=narrative,
    )


def scan(tickers: list[str], interval: str = "5m") -> pd.DataFrame:
    """Intraday setup scan across tickers. Active (long/short) setups first."""
    rows = []
    for t in tickers:
        sig = intraday_signals(t, interval=interval)
        if sig is None:
            continue
        rows.append({
            "ticker": sig.ticker, "setup": sig.setup, "bias": sig.bias,
            "last": sig.last, "vwap": sig.vwap, "dist_vwap_pct": sig.dist_vwap_pct,
            "gap_pct": sig.gap_pct, "rvol": sig.rvol, "asof": sig.asof,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Active setups (long/short) on top, then by relative volume.
    df["_active"] = df["bias"] != "neutral"
    return df.sort_values(["_active", "rvol"], ascending=[False, False]).drop(
        columns="_active").reset_index(drop=True)
