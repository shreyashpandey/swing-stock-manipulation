"""Walk-forward backtest engine.

Design — guarding against lookahead bias:

  1. We compute all indicators ONCE on the full price series. Every indicator
     in technicals.py is causal (EMA, rolling max, etc.), so the value at
     index i depends only on bars 0..i. This is safe.

  2. For each candidate "today" bar i, we slice the DataFrame to df.iloc[:i+1]
     and pass it to the detector. Detectors look at `.iloc[-1]` for "today's
     bar" and `.iloc[-2]` for "yesterday's bar" — they cannot see the future.

  3. When a setup fires we simulate the trade forward bar-by-bar from i+1
     using OHLC: if low <= stoploss → SL hit (exit at SL); else if
     high >= target → target hit (exit at target). Pessimistic ordering
     (SL checked first) when both are touched in the same bar.

  4. If neither SL nor target is hit within `max_hold` bars, we exit at the
     close of the last bar in the window (time stop).

Reported as **R-multiples**: (exit - entry) / (entry - stoploss). A target
hit is normally +R_planned (the original R:R), a stop is -1.0R, a time stop
is whatever fraction it actually finished at.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
from rich.console import Console

from swingdesk.analyze.setups import DETECTORS
from swingdesk.analyze.technicals import add_indicators
from swingdesk.storage import load_prices

console = Console()


@dataclass
class Trade:
    ticker: str
    setup: str
    entry_date: str
    entry: float
    stoploss: float
    target: float
    planned_rr: float

    exit_date: str | None = None
    exit_price: float | None = None
    outcome: str | None = None  # "target" | "stoploss" | "time_stop"
    r_multiple: float | None = None
    bars_held: int | None = None

    def close(self, exit_date: str, exit_price: float, outcome: str, bars_held: int) -> None:
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.outcome = outcome
        self.bars_held = bars_held
        risk = self.entry - self.stoploss
        self.r_multiple = (exit_price - self.entry) / risk if risk > 0 else 0.0


def _simulate_forward(df: pd.DataFrame, start_idx: int, sig: dict,
                      max_hold: int) -> tuple[str, float, str, int]:
    """Walk forward from start_idx+1; return (exit_date, exit_price, outcome, bars_held).

    Pessimistic intraday ordering: if a bar's low touches SL AND high touches
    target on the same day, we assume SL was hit first. This penalises rather
    than rewards ambiguity.
    """
    entry = sig["entry"]
    sl = sig["stoploss"]
    target = sig["target"]

    last_idx = min(start_idx + max_hold, len(df) - 1)
    for j in range(start_idx + 1, last_idx + 1):
        bar = df.iloc[j]
        date = bar.name.strftime("%Y-%m-%d") if hasattr(bar.name, "strftime") else str(bar.name)
        # Gap-down through stop: exit at the open, not the stop level (more honest).
        if bar["open"] <= sl:
            return date, float(bar["open"]), "stoploss_gap", j - start_idx
        if bar["low"] <= sl:
            return date, float(sl), "stoploss", j - start_idx
        if bar["high"] >= target:
            return date, float(target), "target", j - start_idx

    # Time stop — exit at last close in window
    bar = df.iloc[last_idx]
    date = bar.name.strftime("%Y-%m-%d") if hasattr(bar.name, "strftime") else str(bar.name)
    return date, float(bar["close"]), "time_stop", last_idx - start_idx


def backtest_ticker(
    ticker: str,
    *,
    max_hold: int = 20,
    warmup: int = 60,
    detectors: list[Callable] | None = None,
    require_uptrend: bool = False,
) -> list[Trade]:
    """Run all detectors over the full history of one ticker. Returns closed trades.

    require_uptrend: if True, only take a (long) entry when the close is above
    the 200-day EMA. Evidence (exit_tuner sweep, 2026-06) shows this trend gate
    is the single biggest expectancy improver — it lifts the whole book's profit
    factor from ~0.83 to ~0.95 — far more than any exit tweak.
    """
    df = load_prices(ticker)
    if df.empty or len(df) < warmup + 5:
        return []
    df = add_indicators(df)

    use_detectors = detectors or DETECTORS
    trades: list[Trade] = []

    # Cooldown per (ticker, setup): don't re-enter while a prior trade in the
    # same setup is still open. Simpler than full position tracking.
    in_trade_until: dict[str, int] = {}

    for i in range(warmup, len(df) - 1):
        window = df.iloc[: i + 1]
        # Trend gate: skip everything below the 200-EMA (don't fight the tide).
        if require_uptrend:
            bar = df.iloc[i]
            e200 = bar.get("ema200")
            if e200 is None or pd.isna(e200) or bar["close"] <= e200:
                continue
        for fn in use_detectors:
            try:
                sig = fn(window, ticker)
            except Exception:
                continue
            if not sig:
                continue
            # Cooldown is keyed off sig["setup"] (the canonical name in the
            # signal dict) — same key for both read and write to ensure the
            # check actually fires.
            setup_key = sig["setup"]
            if i < in_trade_until.get(setup_key, 0):
                continue

            entry_date = window.index[-1]
            entry_date_s = entry_date.strftime("%Y-%m-%d") if hasattr(entry_date, "strftime") else str(entry_date)

            t = Trade(
                ticker=ticker,
                setup=setup_key,
                entry_date=entry_date_s,
                entry=float(sig["entry"]),
                stoploss=float(sig["stoploss"]),
                target=float(sig["target"]),
                planned_rr=float(sig.get("rr") or 0.0),
            )
            exit_date, exit_price, outcome, bars = _simulate_forward(df, i, sig, max_hold)
            t.close(exit_date, exit_price, outcome, bars)
            trades.append(t)
            in_trade_until[setup_key] = i + bars

    return trades


def backtest_universe(
    tickers: list[str],
    *,
    max_hold: int = 20,
    warmup: int = 60,
    require_uptrend: bool = False,
) -> pd.DataFrame:
    """Backtest every ticker; return a flat DataFrame of all trades."""
    all_trades: list[Trade] = []
    for t in tickers:
        trades = backtest_ticker(t, max_hold=max_hold, warmup=warmup,
                                 require_uptrend=require_uptrend)
        if trades:
            console.print(f"  {t:>15}: {len(trades)} trades")
        all_trades.extend(trades)

    if not all_trades:
        return pd.DataFrame()

    rows = [
        {
            "ticker": t.ticker,
            "setup": t.setup,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "entry": t.entry,
            "exit": t.exit_price,
            "stoploss": t.stoploss,
            "target": t.target,
            "outcome": t.outcome,
            "r": round(t.r_multiple, 3) if t.r_multiple is not None else None,
            "bars_held": t.bars_held,
            "planned_rr": t.planned_rr,
        }
        for t in all_trades
    ]
    return pd.DataFrame(rows)
