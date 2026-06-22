"""Exit-rule tuner — keep the entries, sweep the exits.

The walk-forward backtest showed most setups are negative-expectancy. The
hypothesis: the *entries* may be fine but the *exits* aren't (fixed ATR targets
too far, stops too tight, no trailing/breakeven — the engine never trails even
though the live config defines TRAIL params).

This module decouples the two: it collects every detector entry ONCE (the
expensive part), then re-simulates them forward under any number of parametric
ExitPolicies so we can grid-search what actually pays. R-multiple convention
matches the engine: 1R = the initial stop distance.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from swingdesk.analyze.setups import DETECTORS
from swingdesk.analyze.technicals import add_indicators
from swingdesk.storage import load_prices


@dataclass(frozen=True)
class ExitPolicy:
    stop_atr: float = 1.5          # initial stop = entry − stop_atr × ATR
    target_atr: float = 3.0        # fixed target = entry + target_atr × ATR
    max_hold: int = 20             # time stop (bars)
    breakeven_at_r: float | None = None   # move stop→entry once +R reached
    trail_atr: float | None = None        # trail stop at high − trail_atr × ATR

    def label(self) -> str:
        s = f"s{self.stop_atr}/t{self.target_atr}/h{self.max_hold}"
        if self.breakeven_at_r:
            s += f"/be{self.breakeven_at_r}"
        if self.trail_atr:
            s += f"/tr{self.trail_atr}"
        return s


def collect_entries(tickers: list[str], warmup: int = 60) -> list[tuple]:
    """Run every detector over full history once. Returns a list of
    (df_with_indicators, [(idx, setup, entry, atr), ...]) per ticker."""
    data = []
    for t in tickers:
        df = load_prices(t)
        if df.empty or len(df) < warmup + 10:
            continue
        df = add_indicators(df)
        entries = []
        for i in range(warmup, len(df) - 1):
            window = df.iloc[: i + 1]
            for fn in DETECTORS:
                try:
                    sig = fn(window, t)
                except Exception:
                    continue
                if not sig:
                    continue
                atr = df.iloc[i]["atr14"]
                if pd.isna(atr) or atr <= 0:
                    continue
                entries.append((i, sig["setup"], float(df.iloc[i]["close"]), float(atr)))
        if entries:
            data.append((df, entries))
    return data


def _simulate(df: pd.DataFrame, idx: int, entry: float, atr: float,
              p: ExitPolicy) -> tuple[str, float, int] | None:
    """Forward-simulate one entry under policy `p`. Pessimistic intraday order
    (stop checked before target). Trailing/breakeven use the prior bar's high,
    so there's no intrabar lookahead."""
    risk = p.stop_atr * atr
    if risk <= 0:
        return None
    stop = entry - risk
    target = entry + p.target_atr * atr
    be_done = False
    last_idx = min(idx + p.max_hold, len(df) - 1)
    for j in range(idx + 1, last_idx + 1):
        bar = df.iloc[j]
        op, hi, lo = float(bar["open"]), float(bar["high"]), float(bar["low"])
        # gap-down through stop → exit at the open (honest)
        if op <= stop:
            return "stop", (op - entry) / risk, j - idx
        if lo <= stop:
            return "stop", (stop - entry) / risk, j - idx
        if hi >= target:
            return "target", (target - entry) / risk, j - idx
        # Adjust stop for the NEXT bar using this bar's high.
        if p.breakeven_at_r and not be_done and hi >= entry + p.breakeven_at_r * risk:
            stop = max(stop, entry)
            be_done = True
        if p.trail_atr and hi > entry:
            stop = max(stop, hi - p.trail_atr * atr)
    bar = df.iloc[last_idx]
    return "time", (float(bar["close"]) - entry) / risk, last_idx - idx


def run_policy(data: list[tuple], p: ExitPolicy) -> pd.DataFrame:
    """Simulate all collected entries under one policy. Per-(ticker,setup)
    cooldown: no re-entry while a prior same-setup trade is still open."""
    rows = []
    for df, entries in data:
        in_trade_until: dict[str, int] = {}
        for idx, setup, entry, atr in entries:
            if idx < in_trade_until.get(setup, 0):
                continue
            res = _simulate(df, idx, entry, atr, p)
            if res is None:
                continue
            outcome, r, bars = res
            edate = df.index[idx]
            rows.append({
                "setup": setup,
                "entry_date": edate.strftime("%Y-%m-%d") if hasattr(edate, "strftime") else str(edate),
                "r": round(r, 3), "bars_held": bars, "outcome": outcome,
            })
            in_trade_until[setup] = idx + bars
    return pd.DataFrame(rows)
