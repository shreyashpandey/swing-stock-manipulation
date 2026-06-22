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
    win_rate: float            # 0..1 (NET of costs)
    avg_r: float               # mean R per trade (NET)
    total_r: float             # sum of R (NET)
    expectancy: float          # win_rate*avg_win + (1-win_rate)*avg_loss (NET)
    avg_win_r: float
    avg_loss_r: float
    profit_factor: float       # gross_wins / abs(gross_losses) (NET)
    max_consec_losses: int
    max_drawdown_r: float      # max equity-curve drawdown in R units (NET)
    avg_bars_held: float
    gross_avg_r: float = 0.0   # mean R BEFORE costs (for the haircut comparison)
    avg_cost_r: float = 0.0    # mean per-trade cost in R units


def net_returns(trades: pd.DataFrame, cost_pct: float):
    """Per-trade gross R, net R and cost-in-R for a round-trip cost of
    `cost_pct`% of notional. Cost in R = cost_pct% × entry ÷ (entry − stop),
    i.e. tighter stops cost more R. Falls back to zero cost when stop levels
    aren't available."""
    gross = trades["r"].astype(float)
    if cost_pct <= 0 or "entry" not in trades or "stoploss" not in trades:
        cost = pd.Series(0.0, index=trades.index)
    else:
        risk = trades["entry"].astype(float) - trades["stoploss"].astype(float)
        cost = (cost_pct / 100.0) * trades["entry"].astype(float) / risk.where(risk > 0)
        cost = cost.fillna(0.0).clip(lower=0.0)
    return gross, gross - cost, cost


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


def _stats_for_subset(name: str, trades: pd.DataFrame, cost_pct: float = 0.0) -> StrategyStats:
    gross_all, net_all, cost_all = net_returns(trades, cost_pct)
    rs = net_all.dropna()                       # NET R drives all headline stats
    if rs.empty:
        return StrategyStats(name, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
    gross = gross_all.loc[rs.index]
    cost = cost_all.loc[rs.index]

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
        gross_avg_r=round(float(gross.mean()), 3),
        avg_cost_r=round(float(cost.mean()), 3),
    )


def summarize(trades: pd.DataFrame, cost_pct: float = 0.0) -> pd.DataFrame:
    """One row per setup + an 'ALL' row. All headline stats are NET of a
    `cost_pct`% round-trip cost (0 = gross). `gross_avg_r`/`avg_cost_r` columns
    expose the haircut."""
    if trades.empty:
        return pd.DataFrame()

    # Trades are in arbitrary order; sort chronologically for drawdown calc
    df = trades.sort_values("entry_date") if "entry_date" in trades else trades.copy()

    rows = [asdict(_stats_for_subset("ALL", df, cost_pct))]
    for setup, sub in df.groupby("setup"):
        sub = sub.sort_values("entry_date") if "entry_date" in sub else sub
        rows.append(asdict(_stats_for_subset(setup, sub, cost_pct)))

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
