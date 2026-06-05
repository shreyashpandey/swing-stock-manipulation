"""Per-stock investability summary — concise reason + fundamental snapshot.

Rule-based (no API needed) — produces a 1-3 line summary explaining why this
stock looks investable (or doesn't) right now, plus key fundamental ratios
in human-readable form.

Combines: fundamentals (ROE, P/E, growth) + technicals (state, RSI, volume)
+ recent sentiment + active setups.

Used by the Streamlit Chart tab as the "investability card" next to the chart.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from swingdesk.analyze.setups import scan_ticker
from swingdesk.analyze.technicals import add_indicators
from swingdesk.storage import (
    get_fundamentals,
    load_prices,
    recent_sentiment_for_ticker,
)


@dataclass
class StockSummary:
    ticker: str
    company: str
    sector: str | None
    current_price: float
    # Fundamentals (formatted strings ready to display)
    market_cap_cr: float | None
    pe: float | None
    roe_pct: float | None
    debt_to_equity: float | None
    earnings_growth_pct: float | None
    revenue_growth_pct: float | None
    quality_score: float | None
    # Technical state
    technical_state: str
    rsi: float | None
    momentum_20d_pct: float | None
    above_50ema: bool | None
    above_200ema: bool | None
    # Volume flow
    mfi: float | None
    volume_x_avg: float | None
    # Sentiment
    bullish_news_count: int
    bearish_news_count: int
    # Active setup
    active_setup: str | None
    # The headline output
    verdict: str               # "STRONG_BUY" | "BUY" | "WAIT" | "AVOID"
    one_liner: str             # short reason
    why_invest: list[str]      # 2-4 bullets, positive case
    why_avoid: list[str]       # 1-3 bullets, risk case
    fundamental_brief: str     # human-readable paragraph


def _format_fundamentals(f: dict | None) -> str:
    if not f:
        return "Fundamentals not yet ingested. Run `cli fundamentals` to pull from yfinance."
    bits: list[str] = []
    mc = f.get("market_cap")
    if mc:
        bits.append(f"market cap ₹{mc/1e7:,.0f} cr")
    pe = f.get("trailing_pe")
    if pe:
        bits.append(f"P/E {pe:.1f}")
    roe = f.get("return_on_equity")
    if roe is not None:
        bits.append(f"ROE {roe*100:.1f}%")
    pm = f.get("profit_margin")
    if pm is not None:
        bits.append(f"margin {pm*100:.1f}%")
    eg = f.get("earnings_growth")
    if eg is not None:
        bits.append(f"earnings {eg*100:+.0f}% YoY")
    de = f.get("debt_to_equity")
    if de is not None:
        bits.append(f"D/E {de:.2f}")
    sector = f.get("sector", "?")
    name = f.get("short_name", "?")
    return f"{name} ({sector}): " + " · ".join(bits)


def _verdict_logic(s: StockSummary) -> tuple[str, str]:
    """Synthesize verdict + one-liner from all the lenses."""
    q = s.quality_score
    state = s.technical_state
    bullish = s.bullish_news_count
    bearish = s.bearish_news_count
    setup = s.active_setup
    rsi = s.rsi
    momentum = s.momentum_20d_pct or 0

    # STRONG BUY: best of all worlds
    if q and q >= 75 and state == "uptrend" and bullish > bearish and setup:
        return ("STRONG_BUY",
                f"Strong fundamentals + uptrend + bullish news + fresh signal ({setup}). "
                "Multiple lenses align.")

    # BUY: solid quality + trending + no major negatives
    if q and q >= 65 and state in ("uptrend", "weakening") and bearish <= 1:
        if setup:
            return ("BUY",
                    f"Good quality (Q={q:.0f}) in trend with fresh {setup} signal.")
        return ("BUY", f"Good quality (Q={q:.0f}), trend intact, sentiment ok.")

    # AVOID: weak fundamentals or broken technicals + bearish
    if q is not None and q < 50:
        return ("AVOID", f"Weak fundamentals (Q={q:.0f}). Better businesses elsewhere.")
    if state == "broken" and bearish >= 2:
        return ("AVOID", "Below both 50+200 EMA with bearish news flow. Wait for clarity.")
    if rsi and rsi > 80 and momentum > 30:
        return ("AVOID", f"Overextended (RSI {rsi:.0f}, +{momentum:.0f}% in 20d). "
                "Wait for pullback.")

    # WAIT: ambiguous
    if not q:
        return ("WAIT", "Insufficient data — run `cli fundamentals` first.")
    if state == "broken":
        return ("WAIT", "Technicals broken. Wait for reclaim of 50-EMA.")
    return ("WAIT", "No clear edge in either direction. Hold tight.")


def summarize(ticker: str) -> StockSummary | None:
    """Build a full summary for one ticker."""
    df = load_prices(ticker)
    if df.empty or len(df) < 30:
        return None
    df = add_indicators(df)
    last = df.iloc[-1]
    close = float(last["close"])

    fund = get_fundamentals(ticker) or {}
    sent_df = recent_sentiment_for_ticker(ticker, days=7)
    bull = int((sent_df["sentiment"] == "bullish").sum()) if not sent_df.empty else 0
    bear = int((sent_df["sentiment"] == "bearish").sum()) if not sent_df.empty else 0

    # Technicals
    above_50 = bool(close > last["ema50"]) if pd.notna(last.get("ema50")) else None
    above_200 = bool(close > last["ema200"]) if pd.notna(last.get("ema200")) else None
    if above_50 and above_200:
        state = "uptrend"
    elif above_200:
        state = "weakening"
    elif above_50 is False and above_200 is False:
        state = "broken"
    else:
        state = "unknown"
    rsi = float(last["rsi14"]) if pd.notna(last.get("rsi14")) else None
    momentum = ((close - df["close"].iloc[-20]) / df["close"].iloc[-20] * 100
                if len(df) >= 20 else None)
    mfi = float(last["mfi14"]) if pd.notna(last.get("mfi14")) else None
    vol_x = ((float(last["volume"]) / float(last["vol_avg20"]))
             if pd.notna(last.get("vol_avg20")) else None)

    setup = None
    sigs = scan_ticker(ticker)
    if sigs:
        setup = sigs[0]["setup"]

    s = StockSummary(
        ticker=ticker,
        company=fund.get("short_name") or ticker.replace(".NS", ""),
        sector=fund.get("sector"),
        current_price=close,
        market_cap_cr=(fund["market_cap"] / 1e7) if fund.get("market_cap") else None,
        pe=fund.get("trailing_pe"),
        roe_pct=(fund["return_on_equity"] * 100) if fund.get("return_on_equity") is not None else None,
        debt_to_equity=fund.get("debt_to_equity"),
        earnings_growth_pct=(fund["earnings_growth"] * 100) if fund.get("earnings_growth") is not None else None,
        revenue_growth_pct=(fund["revenue_growth"] * 100) if fund.get("revenue_growth") is not None else None,
        quality_score=fund.get("quality_score"),
        technical_state=state, rsi=rsi, momentum_20d_pct=momentum,
        above_50ema=above_50, above_200ema=above_200,
        mfi=mfi, volume_x_avg=vol_x,
        bullish_news_count=bull, bearish_news_count=bear,
        active_setup=setup,
        verdict="WAIT", one_liner="",
        why_invest=[], why_avoid=[],
        fundamental_brief=_format_fundamentals(fund),
    )
    s.verdict, s.one_liner = _verdict_logic(s)

    # Why-invest bullets — fundamentals
    if s.quality_score and s.quality_score >= 65:
        s.why_invest.append(f"Quality score {s.quality_score:.0f}/100 — fundamentals pass the bar")
    if s.roe_pct and s.roe_pct >= 15:
        s.why_invest.append(f"ROE {s.roe_pct:.1f}% — efficient capital use")
    if s.earnings_growth_pct and s.earnings_growth_pct >= 10:
        s.why_invest.append(f"Earnings growing {s.earnings_growth_pct:+.0f}% YoY")
    if s.revenue_growth_pct and s.revenue_growth_pct >= 10:
        s.why_invest.append(f"Revenue growing {s.revenue_growth_pct:+.0f}% YoY")
    if s.debt_to_equity is not None and s.debt_to_equity < 0.5:
        s.why_invest.append(f"Low debt (D/E {s.debt_to_equity:.2f}) — balance-sheet headroom")
    if s.pe and 0 < s.pe < 25:
        s.why_invest.append(f"Reasonable valuation (P/E {s.pe:.1f})")
    # Why-invest bullets — technicals / trend
    if state == "uptrend":
        s.why_invest.append("Trading above both 50 and 200 EMA — in clear uptrend")
    elif above_200:
        s.why_invest.append("Holding above the 200 EMA — long-term trend still up")
    if momentum is not None and momentum >= 5:
        s.why_invest.append(f"Up {momentum:+.0f}% over the last 20 sessions — momentum positive")
    if rsi is not None and 45 <= rsi <= 65:
        s.why_invest.append(f"RSI {rsi:.0f} — healthy, room to run before overbought")
    # Why-invest bullets — volume / money flow
    if vol_x is not None and vol_x >= 1.5:
        s.why_invest.append(f"Today's volume {vol_x:.1f}× the 20-day average — conviction behind the move")
    if mfi is not None and 50 <= mfi <= 80:
        s.why_invest.append(f"Money Flow Index {mfi:.0f} — accumulation, not yet overbought")
    # Why-invest bullets — signal & sentiment
    if setup:
        notes = sigs[0].get("notes") if sigs else None
        s.why_invest.append(f"Fresh {setup} signal firing today" + (f" ({notes})" if notes else ""))
    if bull > bear and bull >= 2:
        s.why_invest.append(f"News sentiment positive: {bull} bullish vs {bear} bearish (last 7d)")

    # Why-avoid bullets — fundamentals
    if s.quality_score is not None and s.quality_score < 50:
        s.why_avoid.append(f"Quality score only {s.quality_score:.0f} — weak fundamentals")
    if s.earnings_growth_pct is not None and s.earnings_growth_pct < 0:
        s.why_avoid.append(f"Earnings shrinking {s.earnings_growth_pct:+.0f}% YoY")
    if s.pe and s.pe > 60:
        s.why_avoid.append(f"Rich valuation (P/E {s.pe:.0f}) — priced for perfection")
    if s.debt_to_equity and s.debt_to_equity > 2 and (s.sector or "") != "Financial Services":
        s.why_avoid.append(f"High debt (D/E {s.debt_to_equity:.1f}) outside financials")
    # Why-avoid bullets — technicals / trend
    if state == "broken":
        s.why_avoid.append("Below both 50 and 200 EMA — trend broken")
    elif state == "weakening":
        s.why_avoid.append("Below the 50 EMA — short-term trend weakening")
    if momentum is not None and momentum <= -5:
        s.why_avoid.append(f"Down {momentum:+.0f}% over the last 20 sessions — momentum negative")
    if rsi and rsi > 80:
        s.why_avoid.append(f"Overbought (RSI {rsi:.0f}) — wait for pullback")
    elif rsi is not None and rsi < 30:
        s.why_avoid.append(f"Oversold (RSI {rsi:.0f}) — falling knife until it bases")
    # Why-avoid bullets — volume / money flow / sentiment
    if mfi is not None and mfi < 20:
        s.why_avoid.append(f"Money Flow Index {mfi:.0f} — heavy distribution")
    if vol_x is not None and vol_x < 0.5 and momentum is not None and momentum > 0:
        s.why_avoid.append(f"Rally on thin volume ({vol_x:.1f}× avg) — not yet convincing")
    if bear > bull and bear >= 2:
        s.why_avoid.append(f"Bearish news flow: {bear} bearish vs {bull} bullish")

    return s
