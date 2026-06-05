"""Portfolio analytics: equity curve, P&L stats, by-setup breakdown."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from swingdesk.config import ACCOUNT_CAPITAL
from swingdesk.storage import load_positions


@dataclass
class PortfolioStats:
    n_trades: int
    n_open: int
    n_closed: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    best_trade: float
    worst_trade: float
    avg_r: float
    total_r: float
    avg_holding_days: float
    open_risk: float       # ₹ at risk on open positions
    unrealized_pnl: float  # marked-to-market gain/loss on open positions


def _holding_days(entry: str, exit: str | None) -> int:
    try:
        e = pd.to_datetime(entry)
        x = pd.to_datetime(exit) if exit else pd.Timestamp.utcnow()
        return max(0, (x - e).days)
    except Exception:
        return 0


def stats(is_paper: bool | None = None) -> PortfolioStats:
    df = load_positions(is_paper=is_paper)
    if df.empty:
        return PortfolioStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    closed = df[df["status"] == "closed"]
    open_df = df[df["status"] == "open"]

    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] <= 0]

    avg_hold = 0.0
    if not closed.empty:
        avg_hold = float(closed.apply(
            lambda r: _holding_days(r["entry_date"], r["exit_date"]), axis=1
        ).mean())

    # Unrealized P&L = (last_price - entry_price) * qty for each open position
    unrealized = 0.0
    if not open_df.empty:
        for _, p in open_df.iterrows():
            lp = p.get("last_price") or p["entry_price"]
            unrealized += (lp - p["entry_price"]) * p["qty"]

    open_risk = 0.0
    if not open_df.empty:
        for _, p in open_df.iterrows():
            if p["stoploss"]:
                open_risk += (p["entry_price"] - p["stoploss"]) * p["qty"]

    return PortfolioStats(
        n_trades=len(df),
        n_open=len(open_df),
        n_closed=len(closed),
        wins=len(wins),
        losses=len(losses),
        win_rate=round(len(wins) / len(closed), 3) if len(closed) else 0.0,
        total_pnl=round(float(closed["pnl"].sum()) if not closed.empty else 0.0, 2),
        avg_pnl=round(float(closed["pnl"].mean()) if not closed.empty else 0.0, 2),
        best_trade=round(float(closed["pnl"].max()) if not closed.empty else 0.0, 2),
        worst_trade=round(float(closed["pnl"].min()) if not closed.empty else 0.0, 2),
        avg_r=round(float(closed["r_multiple"].dropna().mean()) if not closed.empty else 0.0, 3),
        total_r=round(float(closed["r_multiple"].dropna().sum()) if not closed.empty else 0.0, 2),
        avg_holding_days=round(avg_hold, 1),
        open_risk=round(float(open_risk), 2),
        unrealized_pnl=round(float(unrealized), 2),
    )


def equity_curve(*, is_paper: bool | None = None,
                 starting_capital: float | None = None) -> pd.DataFrame:
    """Time-series of cumulative P&L for closed positions, ordered by exit date."""
    df = load_positions(is_paper=is_paper)
    if df.empty:
        return pd.DataFrame(columns=["date", "pnl", "cum_pnl", "equity"])
    closed = df[df["status"] == "closed"].copy()
    if closed.empty:
        return pd.DataFrame(columns=["date", "pnl", "cum_pnl", "equity"])
    closed["date"] = pd.to_datetime(closed["exit_date"])
    closed = closed.sort_values("date")
    closed["cum_pnl"] = closed["pnl"].cumsum()
    cap = starting_capital if starting_capital is not None else ACCOUNT_CAPITAL
    closed["equity"] = cap + closed["cum_pnl"]
    return closed[["date", "ticker", "setup", "pnl", "r_multiple", "cum_pnl", "equity"]]


def by_setup(is_paper: bool | None = None) -> pd.DataFrame:
    """One row per setup with aggregated stats."""
    df = load_positions(is_paper=is_paper)
    if df.empty:
        return pd.DataFrame()
    closed = df[df["status"] == "closed"]
    if closed.empty:
        return pd.DataFrame()
    grouped = closed.groupby("setup").agg(
        n_trades=("id", "count"),
        wins=("pnl", lambda s: int((s > 0).sum())),
        win_rate=("pnl", lambda s: round((s > 0).mean(), 3)),
        total_pnl=("pnl", lambda s: round(s.sum(), 2)),
        avg_pnl=("pnl", lambda s: round(s.mean(), 2)),
        avg_r=("r_multiple", lambda s: round(s.dropna().mean() if not s.dropna().empty else 0.0, 3)),
    ).reset_index()
    return grouped.sort_values("total_pnl", ascending=False)
