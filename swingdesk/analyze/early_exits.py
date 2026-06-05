"""Early-exit warning system — gets you out BEFORE the stop hits.

Six independent warning checks. Each holding is run through all of them
on every mark-to-market or `cli holdings --analyze`. Warnings are
aggregated into a single severity (LOW / MEDIUM / HIGH / CRITICAL) plus
a suggested action.

The philosophy: a hard stop is a *backstop*, not a strategy. By the time
price actually hits your stop, you've already given up the trend break,
volume divergence, sector breakdown, or bearish news cluster that was
shouting at you for days. These checks aim to surface those signals
early — usually in time to trim 30-50% and protect gains.

Warnings:
  TREND_FALSE       — trend_quality says distribution masquerading as uptrend
  TREND_BROKEN      — closed below 20-EMA for ≥3 days
  VOLUME_DIVERGE    — price up but OBV down (smart money exiting)
  BEARISH_CLUSTER   — ≥3 bearish news in last 3 days, no bullish offset
  SECTOR_WEAK       — relevant sector index below 50-EMA, trending down
  EARNINGS_IMMINENT — earnings within 5 days (binary-event risk)
  MFI_OVERHEATED    — MFI > 80 + price overextended (>20% above 50-EMA)
  VIX_SPIKE         — India VIX up >15% over 5 days (market-wide stress)

Each warning carries a severity (1-4) and a one-line reason. The total
severity decides the action:
   ≥ 8  CRITICAL   — exit immediately
   ≥ 5  HIGH       — trim 50%, tighten stop to breakeven
   ≥ 3  MEDIUM     — trim 25%, watch closely
   ≥ 1  LOW        — monitoring only, no action yet
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from swingdesk.analyze.technicals import _slope_dir, add_indicators, trend_quality
from swingdesk.ingest.earnings import days_to_earnings
from swingdesk.storage import (
    get_fundamentals,
    load_macro,
    load_prices,
    recent_sentiment_for_ticker,
)


@dataclass
class Warning:
    kind: str            # "TREND_FALSE" | ...
    severity: int        # 1 (low) - 4 (critical)
    reason: str


@dataclass
class ExitRead:
    ticker: str
    severity_total: int
    action: str          # "EXIT" | "TRIM_50" | "TRIM_25" | "WATCH" | "NONE"
    warnings: list[Warning] = field(default_factory=list)


SECTOR_INDEX_MAP = {
    "Technology": "^CNXIT",
    "Financial Services": "^NSEBANK",
    "Industrials": None,           # no clean Nifty sector index
    "Consumer Cyclical": "^CNXAUTO",
    "Consumer Defensive": None,
    "Energy": None,
    "Communication Services": None,
    "Basic Materials": None,
    "Utilities": None,
    "Real Estate": None,
    "Healthcare": None,
}


def _check_trend_false(df: pd.DataFrame) -> Warning | None:
    tq = trend_quality(df)
    if tq is None:
        return None
    if tq["verdict"] == "false":
        return Warning(
            kind="TREND_FALSE",
            severity=3,
            reason=f"Distribution masquerading as uptrend ({tq['score']}/100 confirmation)",
        )
    return None


def _check_trend_broken(df: pd.DataFrame) -> Warning | None:
    """Closed below 20-EMA for ≥3 consecutive days."""
    if "ema20" not in df.columns or len(df) < 4:
        return None
    last_3 = df.tail(3)
    below = (last_3["close"] < last_3["ema20"]).all()
    if below:
        days = len(df.tail(10).query("close < ema20"))
        return Warning(
            kind="TREND_BROKEN",
            severity=2,
            reason=f"Closed below 20-EMA for the last 3 days ({days}/10 recent bars under)",
        )
    return None


def _check_volume_divergence(df: pd.DataFrame, window: int = 20) -> Warning | None:
    """Price grinding higher but OBV slipping = distribution."""
    if "obv" not in df.columns or len(df) <= window:
        return None
    price_dir = _slope_dir(df["close"], window)
    obv_dir = _slope_dir(df["obv"], window)
    if price_dir > 0 and obv_dir < 0:
        return Warning(
            kind="VOLUME_DIVERGE",
            severity=3,
            reason="Price rising but OBV falling — smart money distributing into strength",
        )
    return None


def _check_bearish_news(ticker: str) -> Warning | None:
    """3+ bearish news in last 3 days with no bullish offset."""
    sent = recent_sentiment_for_ticker(ticker, days=3)
    if sent.empty:
        return None
    bear = int((sent["sentiment"] == "bearish").sum())
    bull = int((sent["sentiment"] == "bullish").sum())
    high_impact_bear = sent[(sent["sentiment"] == "bearish") & (sent["impact"] == "high")]
    if bear >= 3 and bull == 0:
        sev = 4 if len(high_impact_bear) >= 1 else 3
        return Warning(
            kind="BEARISH_CLUSTER",
            severity=sev,
            reason=f"{bear} bearish news in 3 days, no bullish offset"
                   + (" — at least one high-impact" if len(high_impact_bear) else ""),
        )
    return None


def _check_sector_weakness(ticker: str) -> Warning | None:
    """Relevant sector index below 50-EMA AND trending down."""
    fund = get_fundamentals(ticker) or {}
    sector = fund.get("sector")
    idx_ticker = SECTOR_INDEX_MAP.get(sector)
    if not idx_ticker:
        return None
    df = load_macro(idx_ticker, days=120)
    if df.empty or len(df) < 60:
        return None
    df = add_indicators(df.rename(columns={"close": "close", "volume": "volume"}).assign(
        open=df["close"], high=df["close"], low=df["close"]
    ))
    if "ema50" not in df.columns:
        return None
    last = df.iloc[-1]
    if pd.isna(last.get("ema50")):
        return None
    if last["close"] < last["ema50"] and _slope_dir(df["close"], 20) < 0:
        change = (last["close"] - df["close"].iloc[-20]) / df["close"].iloc[-20] * 100
        return Warning(
            kind="SECTOR_WEAK",
            severity=2,
            reason=f"{sector} sector index below 50-EMA and trending down ({change:+.1f}% in 20d)",
        )
    return None


def _check_earnings_imminent(ticker: str) -> Warning | None:
    """Earnings within 5 days = binary event risk; trim ahead."""
    dte = days_to_earnings(ticker)
    if dte is None:
        return None
    if dte <= 2:
        return Warning(kind="EARNINGS_IMMINENT", severity=3,
                       reason=f"Earnings in {dte} day(s) — binary event imminent")
    if dte <= 5:
        return Warning(kind="EARNINGS_IMMINENT", severity=2,
                       reason=f"Earnings in {dte} days — consider trimming")
    return None


def _check_mfi_overheated(df: pd.DataFrame) -> Warning | None:
    """MFI > 80 + price extended >20% above 50-EMA."""
    if "mfi14" not in df.columns or "ema50" not in df.columns:
        return None
    last = df.iloc[-1]
    if pd.isna(last.get("mfi14")) or pd.isna(last.get("ema50")):
        return None
    mfi = float(last["mfi14"])
    pct_above_50 = (float(last["close"]) - float(last["ema50"])) / float(last["ema50"]) * 100
    if mfi > 80 and pct_above_50 > 20:
        return Warning(
            kind="MFI_OVERHEATED",
            severity=2,
            reason=f"MFI {mfi:.0f} overbought + price {pct_above_50:.0f}% above 50-EMA",
        )
    return None


def _check_vix_spike() -> Warning | None:
    """India VIX up >15% over last 5 trading days — market-wide stress."""
    vix = load_macro("^INDIAVIX", days=10)
    if vix.empty or len(vix) < 6:
        return None
    latest = float(vix["close"].iloc[-1])
    five_ago = float(vix["close"].iloc[-6])
    if five_ago > 0 and (latest - five_ago) / five_ago > 0.15:
        return Warning(
            kind="VIX_SPIKE",
            severity=2,
            reason=f"India VIX up {(latest-five_ago)/five_ago*100:.0f}% in 5 days (={latest:.1f}) — market-wide stress",
        )
    return None


def evaluate(ticker: str) -> ExitRead:
    """Run all checks for one holding and aggregate to a single action."""
    warnings: list[Warning] = []

    # Indicators we need are computed once
    df = load_prices(ticker)
    if not df.empty and len(df) >= 60:
        df = add_indicators(df)
        for check in (_check_trend_false, _check_trend_broken,
                      _check_volume_divergence, _check_mfi_overheated):
            w = check(df)
            if w:
                warnings.append(w)

    # Ticker-level checks (independent of price frame)
    for ticker_check in (_check_bearish_news, _check_sector_weakness,
                        _check_earnings_imminent):
        w = ticker_check(ticker)
        if w:
            warnings.append(w)

    # Market-wide (computed once but attached per holding)
    vix_w = _check_vix_spike()
    if vix_w:
        warnings.append(vix_w)

    sev = sum(w.severity for w in warnings)
    if sev >= 8:
        action = "EXIT"
    elif sev >= 5:
        action = "TRIM_50"
    elif sev >= 3:
        action = "TRIM_25"
    elif sev >= 1:
        action = "WATCH"
    else:
        action = "NONE"

    return ExitRead(ticker=ticker, severity_total=sev, action=action, warnings=warnings)
