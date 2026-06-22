"""Transaction Cost Analysis (TCA) — grade your execution the way a desk does.

Two questions, two functions:

  • PRE-TRADE  — `compare_algos()`: before you trade, which algo is cheapest for
    THIS order? Runs every execution algo on the same parent order and lays the
    cost/impact/timing-risk trade-off side by side. A small order in a liquid
    name → they're all cheap, use VWAP. A large order in a thin name → POV/IS
    matter, and the table shows why.

  • POST-TRADE — `analyze_fills()`: after you trade, how well did you actually do?
    Benchmarks your average fill against the three references the desks use:
      – Arrival price (Implementation Shortfall): the price when you decided.
        This is the honest one — it counts the drift while you were working it.
      – Interval/day VWAP: did you beat the volume-weighted average?
      – TWAP: did you beat a naive even-execution?
    Slippage is signed so positive = cost (paid up on a buy / sold low on a sell).

If we have intraday bars for the trade date the benchmarks are exact; otherwise
we fall back to a daily-OHLC approximation and say so (`basis`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swingdesk.analyze import execution as exec_mod
from swingdesk.analyze.intraday import vwap as session_vwap
from swingdesk.config import BACKTEST_COST_PCT
from swingdesk.storage import load_intraday, load_prices

# One-way fee/tax/slippage proxy (bps). BACKTEST_COST_PCT is a round-trip %.
DEFAULT_FEES_BPS = BACKTEST_COST_PCT * 100 / 2


# --------------------------------------------------------------------------- #
# Pre-trade: compare execution algos for one order
# --------------------------------------------------------------------------- #
def compare_algos(ticker: str, side: str = "buy", *, qty: int | None = None,
                  notional: float | None = None, bucket_minutes: int = 30,
                  participation: float = 0.10,
                  risk_aversion: float = 0.5) -> pd.DataFrame:
    """Run every algo on the same order; return a cheapest-first comparison."""
    rows = []
    for algo in exec_mod.ALGOS:
        plan = exec_mod.execution_plan(
            ticker, side, qty=qty, notional=notional, algo=algo,
            bucket_minutes=bucket_minutes, participation=participation,
            risk_aversion=risk_aversion)
        if plan is None:
            continue
        note = ""
        if not plan.completes_in_session:
            note = f"⚠ {plan.unfilled_shares:,} left over"
        elif plan.warnings:
            note = plan.warnings[0][:60]
        rows.append({
            "algo": algo.upper(),
            "slices": len(plan.schedule),
            "horizon_min": plan.horizon_minutes,
            "avg_participation_pct": plan.avg_participation_pct,
            "est_cost_bps": plan.est_cost_bps,
            "impact_bps": plan.impact_bps,
            "timing_risk_bps": plan.timing_risk_bps,
            "est_cost_rupees": plan.est_cost_rupees,
            "completes": plan.completes_in_session,
            "note": note,
        })
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values("est_cost_bps", na_position="last")
            .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Post-trade: grade actual fills
# --------------------------------------------------------------------------- #
@dataclass
class TCAReport:
    ticker: str
    side: str
    date: str
    qty: int
    avg_fill: float
    arrival_price: float
    benchmark_vwap: float
    benchmark_twap: float
    is_bps: float              # Implementation Shortfall vs arrival (the honest one)
    vwap_slip_bps: float
    twap_slip_bps: float
    fees_bps: float
    total_cost_bps: float      # IS + fees, all-in vs arrival
    notional: float
    basis: str                 # "intraday" | "daily-approx"
    verdict: list[str] = field(default_factory=list)


def _fills_frame(fills) -> pd.DataFrame:
    """Coerce fills (list of dicts or DataFrame) → frame with price, qty, time."""
    df = pd.DataFrame(fills).copy() if not isinstance(fills, pd.DataFrame) else fills.copy()
    if "price" not in df.columns or "qty" not in df.columns:
        raise ValueError("each fill needs 'price' and 'qty'")
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def _slip_bps(side: str, fill: float, bench: float) -> float:
    """Signed cost in bps: positive = worse than benchmark (paid up / sold low)."""
    if not (np.isfinite(fill) and np.isfinite(bench) and bench > 0):
        return float("nan")
    sign = 1.0 if side == "buy" else -1.0
    return round(sign * (fill - bench) / bench * 1e4, 1)


def analyze_fills(ticker: str, side: str, fills, *, arrival_price: float | None = None,
                  date: str | None = None, fees_bps: float = DEFAULT_FEES_BPS,
                  interval: str = "5m") -> TCAReport | None:
    """Grade actual fills against arrival / VWAP / TWAP benchmarks.

    `fills`: list of {price, qty, [time]} (or a DataFrame). `arrival_price`: the
    price when you decided to trade (defaults to the session/day open). `date`:
    trade date 'YYYY-MM-DD' if fills carry no timestamps."""
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    f = _fills_frame(fills)
    if f.empty:
        return None
    qty = float(f["qty"].sum())
    if qty <= 0:
        return None
    avg_fill = float((f["price"] * f["qty"]).sum() / qty)

    # Determine the trade date.
    if date is not None:
        day = pd.Timestamp(date).normalize()
    elif "time" in f.columns and f["time"].notna().any():
        day = f["time"].dropna().iloc[0].normalize()
    else:
        px = load_prices(ticker, days=5)
        day = px.index.max().normalize() if px is not None and not px.empty else None

    bench_vwap = bench_twap = arrival = float("nan")
    basis = "daily-approx"

    # Preferred: intraday bars for the trade date → exact VWAP/TWAP/arrival.
    idf = load_intraday(ticker, interval=interval)
    if day is not None and idf is not None and not idf.empty:
        sess = idf[idf.index.normalize() == day]
        if not sess.empty:
            basis = "intraday"
            vw = session_vwap(sess)
            # Interval window = first→last fill time if available, else full session.
            window = sess
            if "time" in f.columns and f["time"].notna().any():
                lo, hi = f["time"].min(), f["time"].max()
                w = sess[(sess.index >= lo) & (sess.index <= hi)]
                if not w.empty:
                    window = w
            wv = session_vwap(window)
            bench_vwap = float(wv.iloc[-1]) if not wv.empty else float(vw.iloc[-1])
            tp = (window["high"] + window["low"] + window["close"]) / 3.0
            bench_twap = float(tp.mean())
            arrival = float(sess["open"].iloc[0])

    # Fallback: daily OHLC approximation.
    if basis == "daily-approx" and day is not None:
        px = load_prices(ticker)
        if px is not None and not px.empty:
            row = px[px.index.normalize() == day]
            row = row.iloc[-1] if not row.empty else px.iloc[-1]
            arrival = float(row["open"])
            bench_vwap = float((row["high"] + row["low"] + row["close"]) / 3.0)  # VWAP proxy
            bench_twap = float((row["open"] + row["high"] + row["low"] + row["close"]) / 4.0)

    if arrival_price is not None:
        arrival = float(arrival_price)

    is_bps = _slip_bps(side, avg_fill, arrival)
    vwap_slip = _slip_bps(side, avg_fill, bench_vwap)
    twap_slip = _slip_bps(side, avg_fill, bench_twap)
    total_cost = round((is_bps if np.isfinite(is_bps) else 0.0) + fees_bps, 1)

    verdict = []
    if np.isfinite(vwap_slip):
        verdict.append(f"{'Beat' if vwap_slip < 0 else 'Lagged'} VWAP by "
                       f"{abs(vwap_slip):.0f} bps.")
    if np.isfinite(is_bps):
        verdict.append(f"Implementation shortfall {is_bps:+.0f} bps vs arrival "
                       f"(₹{abs(is_bps)/1e4*avg_fill*qty:,.0f}).")
    verdict.append(f"All-in ≈ {total_cost:.0f} bps incl. ~{fees_bps:.0f} bps fees/taxes.")
    if basis == "daily-approx":
        verdict.append("⚠ No intraday bars for that date — benchmarks use a daily-OHLC "
                       "approximation (less precise).")

    return TCAReport(
        ticker=ticker, side=side,
        date=str(day.date()) if day is not None else "n/a",
        qty=int(qty), avg_fill=round(avg_fill, 2),
        arrival_price=round(arrival, 2) if np.isfinite(arrival) else float("nan"),
        benchmark_vwap=round(bench_vwap, 2) if np.isfinite(bench_vwap) else float("nan"),
        benchmark_twap=round(bench_twap, 2) if np.isfinite(bench_twap) else float("nan"),
        is_bps=is_bps, vwap_slip_bps=vwap_slip, twap_slip_bps=twap_slip,
        fees_bps=round(fees_bps, 1), total_cost_bps=total_cost,
        notional=round(avg_fill * qty, 2), basis=basis, verdict=verdict,
    )
