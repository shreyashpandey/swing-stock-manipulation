"""AI-generated investment thesis per holding.

For each holding, Claude receives a STRUCTURED bundle containing:
    - Company / sector / current price / P&L
    - Key fundamentals (ROE, P/E, debt, growth)
    - Technical state (above/below key EMAs, RSI, volume flow)
    - Recent classified news (top 5 by impact)
    - Macro context (NIFTY trend, USD/INR, sector index move)
    - Most-correlated macro factors

And returns a structured thesis:
    - 2-3 sentence narrative
    - Conviction score (0-100)
    - Top 3 risks
    - Recommended action (BUY_MORE / HOLD / TRIM / EXIT)
    - Specific catalyst to watch

Uses structured outputs (Pydantic + messages.parse) so we never have to
parse free-form text. Prompt-cached system prompt + bundle schema makes
repeated calls cheap.
"""
from __future__ import annotations

import os
from typing import Literal

import anthropic
import pandas as pd
from pydantic import BaseModel, Field
from rich.console import Console

from swingdesk.config import CLAUDE_MODEL
from swingdesk.ingest import macro as macro_mod

console = Console()

Action = Literal["BUY_MORE", "HOLD", "TRIM", "EXIT"]


class StockThesis(BaseModel):
    narrative: str = Field(description="2-3 sentence investment thesis")
    conviction: int = Field(ge=0, le=100, description="Conviction 0-100")
    action: Action = Field(description="Recommended action")
    risks: list[str] = Field(description="Top 3 risks, one phrase each")
    catalyst_to_watch: str = Field(description="Single most important upcoming catalyst")


SYSTEM_PROMPT = """You are a senior equity analyst at an Indian asset manager.
A retail trader will share a single stock holding with you, including:
- Their position (avg buy price, current price, P&L %)
- Fundamentals (ROE, P/E, growth, debt)
- Technicals (price vs 50/200 EMA, RSI, volume-flow indicators)
- Recent news (last 7 days, sentiment-classified)
- Macro context (relevant Nifty index, currency, commodity moves)

Your job is to give a **specific, decision-grade thesis**. Not generic advice.
Specific to this stock, at this price, in this regime.

Rules:
1. Cite concrete numbers from the data provided
2. Action options: BUY_MORE | HOLD | TRIM | EXIT
3. Conviction 0-100 — high only when MULTIPLE lenses align
4. Risks should be unique to this stock — not generic ("market volatility")
5. Catalyst should be a specific upcoming event or level
6. Never recommend trading on rumors or vague sentiment
7. If fundamentals are weak (ROE < 10%), bias toward EXIT regardless of price action
8. If technicals are broken (below both 50 + 200 EMA), require a strong bullish
   fundamental + news case before suggesting BUY_MORE; otherwise TRIM/EXIT

Be terse. Indian retail audience — avoid jargon. Mention amounts in ₹."""


def _build_bundle(ticker: str, qty: float, avg_buy: float | None,
                  last_price: float | None, pnl_pct: float | None,
                  fundamentals: dict | None, technical_state: dict,
                  recent_news: pd.DataFrame, market_pulse: dict,
                  correlations: pd.DataFrame) -> str:
    """Render the per-stock context as a compact JSON-ish text block."""
    lines = [f"## {ticker}"]
    if fundamentals and fundamentals.get("short_name"):
        lines.append(f"Company: {fundamentals['short_name']} ({fundamentals.get('sector', 'Unknown')})")

    # Position
    lines.append(f"\nPOSITION: qty={qty}, avg={avg_buy or '?'}, current={last_price or '?'}, "
                 f"P&L={pnl_pct or '?'}%")

    # Fundamentals
    if fundamentals:
        f = fundamentals
        lines.append(
            f"\nFUNDAMENTALS:"
            f"\n  ROE: {f.get('return_on_equity', '?')!r}"
            f"\n  P/E: {f.get('trailing_pe', '?')!r}"
            f"\n  Debt/Equity: {f.get('debt_to_equity', '?')!r}"
            f"\n  Profit margin: {f.get('profit_margin', '?')!r}"
            f"\n  Earnings growth (YoY): {f.get('earnings_growth', '?')!r}"
            f"\n  Revenue growth (YoY): {f.get('revenue_growth', '?')!r}"
            f"\n  Quality score (0-100): {f.get('quality_score', '?')!r}"
        )

    # Technicals
    t = technical_state
    lines.append(
        f"\nTECHNICALS:"
        f"\n  Trend: {t.get('state', '?')}"
        f"\n  RSI(14): {t.get('rsi', '?')}"
        f"\n  Above 50-EMA: {t.get('above_50ema')}"
        f"\n  Above 200-EMA: {t.get('above_200ema')}"
        f"\n  MFI(14): {t.get('mfi', '?')}"
        f"\n  Buying pressure (20d): {t.get('buy_pressure', '?')}"
        f"\n  Active setup: {t.get('active_setup', 'none')}"
    )

    # Recent news (top 5 by impact)
    if not recent_news.empty:
        lines.append("\nRECENT NEWS (last 7 days):")
        for _, row in recent_news.head(5).iterrows():
            lines.append(f"  - [{row['sentiment']}/{row['impact']}/{row['event_type']}] "
                         f"{row['title'][:100]}")

    # Macro context
    if market_pulse:
        nifty = market_pulse.get("NIFTY 50", {})
        usdinr = market_pulse.get("USD/INR", {})
        brent = market_pulse.get("Brent Crude", {})
        lines.append(
            f"\nMACRO CONTEXT:"
            f"\n  NIFTY 50: 1d {nifty.get('chg_1d', '?')}%, 1w {nifty.get('chg_1w', '?')}%"
            f"\n  USD/INR:  1d {usdinr.get('chg_1d', '?')}%, 1w {usdinr.get('chg_1w', '?')}%"
            f"\n  Brent:    1d {brent.get('chg_1d', '?')}%, 1w {brent.get('chg_1w', '?')}%"
        )

    # Top correlations
    if not correlations.empty:
        top = correlations.head(3)
        lines.append("\nTOP 3 CORRELATED MACRO FACTORS (60d, |r|>):")
        for _, row in top.iterrows():
            sign = "+" if row["correlation"] > 0 else "−"
            lines.append(f"  {sign} {row['factor']}: r={row['correlation']:+.2f}")

    return "\n".join(lines)


def generate(ticker: str, qty: float, avg_buy: float | None,
             last_price: float | None, pnl_pct: float | None,
             fundamentals: dict | None,
             technical_state: dict,
             recent_news: pd.DataFrame) -> StockThesis | None:
    """Call Claude for a thesis. Returns None if API key not set."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    pulse = macro_mod.market_pulse()
    corr = macro_mod.correlations(ticker)
    bundle = _build_bundle(
        ticker=ticker, qty=qty, avg_buy=avg_buy, last_price=last_price,
        pnl_pct=pnl_pct, fundamentals=fundamentals,
        technical_state=technical_state, recent_news=recent_news,
        market_pulse=pulse, correlations=corr,
    )

    client = anthropic.Anthropic(timeout=60.0, max_retries=2)
    try:
        resp = client.messages.parse(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # rubric is stable
            }],
            messages=[{
                "role": "user",
                "content": f"Generate a thesis for this holding:\n\n{bundle}",
            }],
            output_format=StockThesis,
        )
        if resp.parsed_output is None:
            return None
        return resp.parsed_output
    except anthropic.APIStatusError as e:
        console.print(f"[red]thesis API error for {ticker}: {e.status_code}[/red]")
        return None
    except Exception as e:
        console.print(f"[yellow]thesis failed for {ticker}: {e.__class__.__name__}[/yellow]")
        return None
