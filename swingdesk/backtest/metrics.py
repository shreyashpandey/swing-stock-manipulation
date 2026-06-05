"""Performance metrics on a flat DataFrame of trades."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class StrategyStats:
    setup: str
    n_trades: int
    wins: int
    losses: int
    win_rate: float            # 0..1
    avg_r: float               # mean R per trade
    total_r: float             # sum of R
    expectancy: float          # win_rate*avg_win + (1-win_rate)*avg_loss
    avg_win_r: float
    avg_loss_r: float
    profit_factor: float       # gross_wins / abs(gross_losses)
    max_consec_losses: int
    max_drawdown_r: float      # max equity-curve drawdown in R units
    avg_bars_held: float


def _max_consec_losses(rs: pd.Series) -> int:
    streak = best = 0
    for r in rs:
        if r < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def _max_drawdown_r(rs: pd.Series) -> float:
    """Max peak-to-trough decline of the cumulative R-curve (chronological)."""
    if rs.empty:
        return 0.0
    cum = rs.cumsum()
    peak = cum.cummax()
    dd = peak - cum
    return float(dd.max())


def _stats_for_subset(name: str, trades: pd.DataFrame) -> StrategyStats:
    rs = trades["r"].dropna()
    if rs.empty:
        return StrategyStats(name, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    win_rate = len(wins) / len(rs)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 1e-9 else float("inf")

    return StrategyStats(
        setup=name,
        n_trades=len(rs),
        wins=len(wins),
        losses=len(losses),
        win_rate=round(float(win_rate), 3),
        avg_r=round(float(rs.mean()), 3),
        total_r=round(float(rs.sum()), 2),
        expectancy=round(float(expectancy), 3),
        avg_win_r=round(float(avg_win), 3),
        avg_loss_r=round(float(avg_loss), 3),
        profit_factor=round(float(profit_factor), 2) if profit_factor != float("inf") else float("inf"),
        max_consec_losses=int(_max_consec_losses(rs)),
        max_drawdown_r=round(float(_max_drawdown_r(rs)), 2),
        avg_bars_held=round(float(trades["bars_held"].mean()), 1) if "bars_held" in trades else 0.0,
    )


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per setup + an 'all' row."""
    if trades.empty:
        return pd.DataFrame()

    # Trades are in arbitrary order; sort chronologically for drawdown calc
    df = trades.sort_values("entry_date") if "entry_date" in trades else trades.copy()

    rows = [asdict(_stats_for_subset("ALL", df))]
    for setup, sub in df.groupby("setup"):
        rows.append(asdict(_stats_for_subset(setup, sub.sort_values("entry_date") if "entry_date" in sub else sub)))

    out = pd.DataFrame(rows)
    return out


# --- gating rules used to decide whether a strategy is "live-tradeable" -----

EDGE_CRITERIA = {
    "min_trades": 20,
    "min_expectancy": 0.1,    # at least +0.1R per trade in expectation
    "min_profit_factor": 1.3,
    "max_drawdown_r": 15.0,   # in R units — accept up to 15R DD
}


def gate(stats_row: dict) -> tuple[bool, list[str]]:
    """Return (passes, list_of_failures)."""
    fails = []
    if stats_row["n_trades"] < EDGE_CRITERIA["min_trades"]:
        fails.append(f"n_trades<{EDGE_CRITERIA['min_trades']}")
    if stats_row["expectancy"] < EDGE_CRITERIA["min_expectancy"]:
        fails.append(f"expectancy<{EDGE_CRITERIA['min_expectancy']}")
    pf = stats_row["profit_factor"]
    pf_val = pf if pf != float("inf") else 999
    if pf_val < EDGE_CRITERIA["min_profit_factor"]:
        fails.append(f"profit_factor<{EDGE_CRITERIA['min_profit_factor']}")
    if stats_row["max_drawdown_r"] > EDGE_CRITERIA["max_drawdown_r"]:
        fails.append(f"drawdown>{EDGE_CRITERIA['max_drawdown_r']}R")
    return (len(fails) == 0, fails)
