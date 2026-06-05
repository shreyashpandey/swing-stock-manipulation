"""Generate historical signal events for chart annotation.

For any ticker, walks the entire price history and records every bar where
a setup detector fired. Returns a flat list of (date, price, setup,
direction, was_profitable) tuples so the Streamlit chart can plot them
as triangle markers with labels.

This is essentially a stripped-down backtest that keeps only entry events
+ outcome (target vs stop) — not exit-bar tracking. Much cheaper than the
full backtest.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from swingdesk.analyze.setups import DETECTORS
from swingdesk.analyze.technicals import add_indicators
from swingdesk.storage import load_prices


@dataclass
class ChartEvent:
    date: pd.Timestamp
    price: float
    setup: str
    direction: str           # "buy" | "sell"
    outcome: str | None      # "target" | "stop" | "open" — what happened next
    r_multiple: float | None # realized R if closed; None if still hypothetical
    notes: str = ""          # human-readable reason the setup fired
    entry: float | None = None
    stoploss: float | None = None
    target: float | None = None
    rr: float | None = None


def _simulate_outcome(df: pd.DataFrame, start_idx: int, sig: dict,
                      max_hold: int = 20) -> tuple[str, float | None]:
    """Walk forward to determine whether this signal would have hit target,
    stop, or timed out. Returns (outcome, r_multiple)."""
    entry = sig["entry"]
    sl = sig["stoploss"]
    target = sig["target"]
    risk = entry - sl
    if risk <= 0:
        return "open", None

    last_idx = min(start_idx + max_hold, len(df) - 1)
    for j in range(start_idx + 1, last_idx + 1):
        bar = df.iloc[j]
        if bar["open"] <= sl:
            return "stop", (float(bar["open"]) - entry) / risk
        if bar["low"] <= sl:
            return "stop", -1.0
        if bar["high"] >= target:
            return "target", (target - entry) / risk
    # Timed out — use last close
    final_close = float(df.iloc[last_idx]["close"])
    return "open", (final_close - entry) / risk


def events_for_ticker(ticker: str, lookback: int = 250,
                      include_outcome: bool = True) -> list[ChartEvent]:
    """Find every historical setup signal on this ticker and tag with outcome."""
    df = load_prices(ticker, days=lookback)
    if df.empty or len(df) < 60:
        return []
    df = add_indicators(df)

    events: list[ChartEvent] = []
    cooldown_until: dict[str, int] = {}

    for i in range(60, len(df)):
        window = df.iloc[: i + 1]
        for fn in DETECTORS:
            try:
                sig = fn(window, ticker)
            except Exception:
                continue
            if not sig:
                continue
            setup_key = sig["setup"]
            if i < cooldown_until.get(setup_key, 0):
                continue

            outcome, r_mult = "open", None
            if include_outcome:
                outcome, r_mult = _simulate_outcome(df, i, sig)

            events.append(ChartEvent(
                date=window.index[-1],
                price=float(sig["entry"]),
                setup=setup_key,
                direction="buy",
                outcome=outcome,
                r_multiple=round(r_mult, 2) if r_mult is not None else None,
                notes=sig.get("notes", ""),
                entry=sig.get("entry"),
                stoploss=sig.get("stoploss"),
                target=sig.get("target"),
                rr=sig.get("rr"),
            ))
            # Cooldown so we don't get back-to-back same-setup signals
            cooldown_until[setup_key] = i + 15
    return events
