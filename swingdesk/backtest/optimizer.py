"""Grid-search parameter optimizer for setup detectors.

Each detector has implicit "magic numbers" for SL multiplier, target multiplier,
and volume threshold. The current values were picked by intuition. This module
runs a backtest with each combination on the grid, then ranks by expectancy.

NOTE: This is exploratory tooling. Grid search WILL overfit if you take the
single best params and trust them blindly. Use the results as hypotheses to
test, not as ground truth. The reliable next step is out-of-sample testing —
fit on year N-2/N-1, validate on year N.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd
from rich.console import Console

from swingdesk.analyze.setups import (
    detect_breakout,
    detect_ma_cross,
    detect_pullback_to_ema,
    detect_volume_thrust,
)
from swingdesk.backtest import engine
from swingdesk.backtest.metrics import summarize

console = Console()


@dataclass
class ParamGrid:
    """Per-setup parameter spaces to search over."""
    sl_mult: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)
    tgt_mult: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)
    max_hold: tuple[int, ...] = (10, 20, 30)


# We can't actually patch the detectors' constants in-place without modifying
# the source. Instead we generate parametrised wrapper detectors that override
# the SL/target multipliers via a closure.

def _make_breakout(sl_mult: float, tgt_mult: float) -> Callable:
    def detect(df: pd.DataFrame, ticker: str):
        sig = detect_breakout(df, ticker)
        if not sig:
            return None
        atr = df["atr14"].iloc[-1]
        if pd.isna(atr):
            return None
        entry = sig["entry"]
        sig["stoploss"] = round(entry - sl_mult * atr, 2)
        sig["target"] = round(entry + tgt_mult * atr, 2)
        sig["rr"] = round((sig["target"] - entry) / max(entry - sig["stoploss"], 1e-6), 2)
        return sig
    detect.__name__ = f"detect_breakout_sl{sl_mult}_tgt{tgt_mult}"
    return detect


def _make_pullback(sl_mult: float, tgt_mult: float) -> Callable:
    def detect(df: pd.DataFrame, ticker: str):
        sig = detect_pullback_to_ema(df, ticker)
        if not sig:
            return None
        atr = df["atr14"].iloc[-1]
        if pd.isna(atr):
            return None
        entry = sig["entry"]
        low = df["low"].iloc[-1]
        sig["stoploss"] = round(low - (sl_mult - 1.0) * atr, 2) if sl_mult >= 1 else round(entry - sl_mult * atr, 2)
        sig["target"] = round(entry + tgt_mult * atr, 2)
        sig["rr"] = round((sig["target"] - entry) / max(entry - sig["stoploss"], 1e-6), 2)
        return sig
    detect.__name__ = f"detect_pullback_sl{sl_mult}_tgt{tgt_mult}"
    return detect


def _make_volume_thrust(sl_mult: float, tgt_mult: float) -> Callable:
    def detect(df: pd.DataFrame, ticker: str):
        sig = detect_volume_thrust(df, ticker)
        if not sig:
            return None
        atr = df["atr14"].iloc[-1]
        if pd.isna(atr):
            return None
        entry = sig["entry"]
        sig["stoploss"] = round(entry - sl_mult * atr, 2)
        sig["target"] = round(entry + tgt_mult * atr, 2)
        sig["rr"] = round((sig["target"] - entry) / max(entry - sig["stoploss"], 1e-6), 2)
        return sig
    detect.__name__ = f"detect_volume_thrust_sl{sl_mult}_tgt{tgt_mult}"
    return detect


def _make_ma_cross(sl_mult: float, tgt_mult: float) -> Callable:
    def detect(df: pd.DataFrame, ticker: str):
        sig = detect_ma_cross(df, ticker)
        if not sig:
            return None
        atr = df["atr14"].iloc[-1]
        if pd.isna(atr):
            return None
        entry = sig["entry"]
        sig["stoploss"] = round(entry - sl_mult * atr, 2)
        sig["target"] = round(entry + tgt_mult * atr, 2)
        sig["rr"] = round((sig["target"] - entry) / max(entry - sig["stoploss"], 1e-6), 2)
        return sig
    detect.__name__ = f"detect_ma_cross_sl{sl_mult}_tgt{tgt_mult}"
    return detect


SETUP_FACTORIES = {
    "breakout_20d": _make_breakout,
    "pullback_ema20": _make_pullback,
    "volume_thrust": _make_volume_thrust,
    "ema_20_50_cross": _make_ma_cross,
}


def optimize(tickers: list[str], setup: str, grid: ParamGrid | None = None) -> pd.DataFrame:
    """Grid-search over (sl_mult, tgt_mult, max_hold) for one setup.

    Returns a DataFrame with one row per combination, sorted by expectancy.
    """
    if setup not in SETUP_FACTORIES:
        raise ValueError(f"unknown setup: {setup}. Known: {list(SETUP_FACTORIES)}")
    factory = SETUP_FACTORIES[setup]
    grid = grid or ParamGrid()

    results = []
    combos = list(product(grid.sl_mult, tgt_mult := grid.tgt_mult, grid.max_hold))
    console.print(f"[bold]Optimizing {setup}: {len(combos)} combinations × {len(tickers)} tickers[/bold]")

    for sl, tgt, hold in combos:
        # Construct the parametrised detector once for the whole batch
        det = factory(sl, tgt)
        all_trades = []
        for tk in tickers:
            trades = engine.backtest_ticker(
                tk, max_hold=hold, detectors=[det], warmup=60,
            )
            all_trades.extend(trades)
        if not all_trades:
            continue
        df = pd.DataFrame([{
            "setup": setup,
            "r": t.r_multiple,
            "bars_held": t.bars_held,
            "entry_date": t.entry_date,
        } for t in all_trades])
        stats = summarize(df)
        row = stats[stats["setup"] == setup].iloc[0].to_dict() if not stats.empty else {}
        row.update({"sl_mult": sl, "tgt_mult": tgt, "max_hold": hold})
        results.append(row)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    cols = ["sl_mult", "tgt_mult", "max_hold", "n_trades", "win_rate",
            "avg_r", "total_r", "expectancy", "profit_factor",
            "max_drawdown_r", "max_consec_losses"]
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values("expectancy", ascending=False).reset_index(drop=True)
