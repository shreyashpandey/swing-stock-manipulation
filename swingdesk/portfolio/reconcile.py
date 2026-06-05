"""Compare live paper-trade performance against the most recent backtest expectation.

If paper results drift significantly from what the backtest predicted, that's a
signal something has changed — regime shift, slippage you didn't model, broken
data feed, or your own behavior (e.g. closing positions early). Surface those
drifts loudly.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from swingdesk.storage import (
    list_backtest_runs,
    load_backtest_trades,
    load_positions,
)


@dataclass
class DriftRow:
    setup: str
    paper_trades: int
    paper_win_rate: float
    paper_avg_r: float
    backtest_win_rate: float
    backtest_avg_r: float
    win_rate_drift: float   # paper - backtest, in fractional points
    avg_r_drift: float
    verdict: str            # "aligned" | "outperforming" | "underperforming" | "insufficient"


# Tolerance bands (drift must exceed these to flag)
WIN_RATE_TOLERANCE = 0.10   # 10 pp
AVG_R_TOLERANCE = 0.15      # 0.15 R


def _stats(trades: pd.DataFrame, setup: str) -> tuple[int, float, float]:
    sub = trades[trades["setup"] == setup]
    if sub.empty or "r" not in sub.columns:
        return 0, 0.0, 0.0
    rs = sub["r"].dropna()
    if rs.empty:
        return 0, 0.0, 0.0
    win_rate = float((rs > 0).mean())
    avg_r = float(rs.mean())
    return len(rs), win_rate, avg_r


def _paper_stats(setup: str, min_trades: int = 5) -> tuple[int, float, float]:
    df = load_positions(is_paper=True)
    closed = df[(df["status"] == "closed") & (df["setup"] == setup)]
    if len(closed) < min_trades:
        return len(closed), 0.0, 0.0
    rs = closed["r_multiple"].dropna()
    if rs.empty:
        return 0, 0.0, 0.0
    return len(rs), float((rs > 0).mean()), float(rs.mean())


def reconcile(*, latest_run: bool = True, min_paper_trades: int = 5) -> pd.DataFrame:
    """Build a drift report across setups.

    By default reads against the most recent backtest run. Set
    `latest_run=False` to aggregate ALL historical backtest trades.
    """
    runs = list_backtest_runs()
    if runs.empty:
        return pd.DataFrame()
    run_id = runs.iloc[0]["run_id"] if latest_run else None
    bt = load_backtest_trades(run_id)
    if bt.empty:
        return pd.DataFrame()

    rows = []
    for setup in bt["setup"].unique():
        bt_n, bt_wr, bt_r = _stats(bt, setup)
        pp_n, pp_wr, pp_r = _paper_stats(setup, min_trades=min_paper_trades)
        if pp_n < min_paper_trades:
            verdict = "insufficient"
        else:
            wr_drift = pp_wr - bt_wr
            r_drift = pp_r - bt_r
            if abs(wr_drift) < WIN_RATE_TOLERANCE and abs(r_drift) < AVG_R_TOLERANCE:
                verdict = "aligned"
            elif wr_drift > 0 and r_drift > 0:
                verdict = "outperforming"
            else:
                verdict = "underperforming"

        rows.append({
            "setup": setup,
            "paper_trades": pp_n,
            "backtest_trades": bt_n,
            "paper_win_rate": round(pp_wr, 3),
            "backtest_win_rate": round(bt_wr, 3),
            "win_rate_drift": round(pp_wr - bt_wr, 3) if pp_n else None,
            "paper_avg_r": round(pp_r, 3),
            "backtest_avg_r": round(bt_r, 3),
            "avg_r_drift": round(pp_r - bt_r, 3) if pp_n else None,
            "verdict": verdict,
        })

    return pd.DataFrame(rows).sort_values("setup")
