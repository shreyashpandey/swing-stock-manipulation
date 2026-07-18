"""Realized P&L + swing-trading performance, net of real charges.

Turns a Groww tradebook (raw executions in the `trades` table) — or a Groww
Tax-P&L export (already-matched lots) — into FIFO round trips, applies the
:mod:`swingdesk.analyze.charges` cost model (or the export's real charge figures),
classifies each as a short-/long-term gain, and aggregates the swing-trading
performance the user actually earned: win rate, profit factor, expectancy,
holding period, the cost drag charges impose, the indicative tax bill, and a
rough return vs NIFTY.

Net P&L is always computed the same way regardless of source —
``gross = (sell − buy) × qty`` from the prices, minus charges (the export's real
total when present, else the model). We deliberately don't trust a broker
"realized P&L" column, whose gross/net convention varies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk.analyze import charges as charges_mod
from swingdesk.config import (
    LTCG_EXEMPTION,
    LTCG_TAX_PCT,
    STCG_TAX_PCT,
)
from swingdesk.storage import load_macro, load_trades

ROUNDTRIP_COLS = [
    "ticker", "buy_date", "sell_date", "qty", "buy_price", "sell_price",
    "holding_days", "gross_pnl", "charges", "net_pnl", "net_pnl_pct",
    "bucket", "gain_type", "tax_rate_pct", "tax_est", "segment", "exchange",
]

# Holding-period buckets (calendar days). "swing" is the headline the user cares about.
SWING_MAX_DAYS = 40        # ~2 trading months
POSITIONAL_MAX_DAYS = 364  # below the 365-day LTCG line


def _bucket(holding_days: int) -> str:
    if holding_days <= 0:
        return "intraday"
    if holding_days <= SWING_MAX_DAYS:
        return "swing"
    if holding_days <= POSITIONAL_MAX_DAYS:
        return "positional"
    return "long_term"


def _make_roundtrip(lot: dict, sell_price: float, sell_date, qty: float,
                    actual_charges: float | None) -> dict:
    """Build one matched round-trip row (a buy lot consumed by a sell)."""
    buy_price = float(lot["price"])
    gross = (float(sell_price) - buy_price) * qty
    segment = lot.get("segment") or "delivery"
    exchange = lot.get("exchange") or "NSE"
    # Same-day in-and-out is intraday for charge purposes even if tagged delivery.
    holding_days = int((pd.Timestamp(sell_date) - pd.Timestamp(lot["date"])).days)
    seg_for_charge = "intraday" if holding_days <= 0 and segment != "delivery" else segment
    charge_total = (float(actual_charges) if actual_charges is not None
                    else charges_mod.round_trip_charges(
                        buy_price, sell_price, qty, segment=seg_for_charge, exchange=exchange)["total"])
    net = gross - charge_total
    cls = charges_mod.classify_gain(lot["date"], sell_date, net)
    cost_basis = buy_price * qty
    return {
        "ticker": lot["ticker"], "buy_date": pd.Timestamp(lot["date"]),
        "sell_date": pd.Timestamp(sell_date), "qty": qty,
        "buy_price": round(buy_price, 2), "sell_price": round(float(sell_price), 2),
        "holding_days": cls["holding_days"], "gross_pnl": round(gross, 2),
        "charges": round(charge_total, 2), "net_pnl": round(net, 2),
        "net_pnl_pct": round(net / cost_basis * 100, 2) if cost_basis else 0.0,
        "bucket": _bucket(cls["holding_days"]), "gain_type": cls["gain_type"],
        "tax_rate_pct": cls["tax_rate_pct"], "tax_est": cls["tax_est"],
        "segment": segment, "exchange": exchange,
    }


def _roundtrips_fifo(trades_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    df = trades_df.sort_values("trade_date")
    for ticker, g in df.groupby("ticker"):
        queue: list[dict] = []
        for _, t in g.iterrows():
            side = str(t["side"]).lower().strip()
            qty, price = float(t["qty"]), float(t["price"])
            seg = t.get("segment") if "segment" in g.columns else "delivery"
            exch = t.get("exchange") if "exchange" in g.columns else "NSE"
            actual = (float(t["charges"]) if "charges" in g.columns and pd.notna(t.get("charges"))
                      else None)
            if side == "buy":
                queue.append({"ticker": ticker, "qty": qty, "price": price,
                              "date": pd.Timestamp(t["trade_date"]), "segment": seg,
                              "exchange": exch})
            elif side == "sell":
                remaining = qty
                while remaining > 1e-9 and queue:
                    lot = queue[0]
                    consumed = min(lot["qty"], remaining)
                    # Allocate a buy-side actual charge proportionally (rare); the
                    # sell's actual charge applies to the consumed share.
                    rows.append(_make_roundtrip(lot, price, t["trade_date"], consumed, actual))
                    lot["qty"] -= consumed
                    remaining -= consumed
                    if lot["qty"] <= 1e-9:
                        queue.pop(0)
    return rows


def _roundtrips_from_tax_pnl(tax_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    has_charges = "charges" in tax_df.columns
    for _, r in tax_df.iterrows():
        lot = {"ticker": r["symbol"], "price": float(r["buy_price"]),
               "date": pd.Timestamp(r["buy_date"]), "segment": "delivery", "exchange": "NSE"}
        actual = float(r["charges"]) if has_charges and pd.notna(r.get("charges")) else None
        rows.append(_make_roundtrip(lot, float(r["sell_price"]), r["sell_date"],
                                    float(r["qty"]), actual))
    return rows


def realized_roundtrips(trades_df: pd.DataFrame | None = None,
                        tax_pnl_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Matched round trips with charge-aware net P&L. Prefers a Tax-P&L export
    (already matched) when provided; otherwise FIFO-matches the tradebook."""
    if tax_pnl_df is not None and not tax_pnl_df.empty:
        rows = _roundtrips_from_tax_pnl(tax_pnl_df)
    else:
        if trades_df is None:
            trades_df = load_trades()
        if trades_df is None or trades_df.empty:
            return pd.DataFrame(columns=ROUNDTRIP_COLS)
        rows = _roundtrips_fifo(trades_df)
    if not rows:
        return pd.DataFrame(columns=ROUNDTRIP_COLS)
    return pd.DataFrame(rows, columns=ROUNDTRIP_COLS).sort_values("sell_date").reset_index(drop=True)


def _tax_estimate(rt: pd.DataFrame) -> dict:
    """Indicative capital-gains tax: STCG net gains @ rate; LTCG net gains above
    the annual ₹1.25L exemption @ rate. Gains and losses offset within a class."""
    stcg = float(rt.loc[rt["gain_type"] == "STCG", "net_pnl"].sum())
    ltcg = float(rt.loc[rt["gain_type"] == "LTCG", "net_pnl"].sum())
    stcg_tax = max(0.0, stcg) * STCG_TAX_PCT / 100.0
    ltcg_tax = max(0.0, ltcg - LTCG_EXEMPTION) * LTCG_TAX_PCT / 100.0
    return {
        "stcg_net": round(stcg, 2), "ltcg_net": round(ltcg, 2),
        "stcg_tax": round(stcg_tax, 2), "ltcg_tax": round(ltcg_tax, 2),
        "total_tax": round(stcg_tax + ltcg_tax, 2),
    }


def performance(roundtrips: pd.DataFrame, bucket: str | None = "swing") -> dict:
    """Aggregate performance for a holding-period bucket (default 'swing').
    Pass ``bucket=None`` for all realized trades."""
    rt = roundtrips if bucket is None else roundtrips[roundtrips["bucket"] == bucket]
    rt = rt.copy()
    if rt.empty:
        return {"bucket": bucket or "all", "n_trades": 0}

    net = rt["net_pnl"]
    wins, losses = net[net > 0], net[net < 0]
    gross_profit, gross_loss = float(wins.sum()), float(abs(losses.sum()))
    turnover = float((rt["buy_price"] * rt["qty"]).sum() + (rt["sell_price"] * rt["qty"]).sum())
    total_charges = float(rt["charges"].sum())
    gross_pnl = float(rt["gross_pnl"].sum())

    by_month = (rt.assign(month=rt["sell_date"].dt.to_period("M").astype(str))
                  .groupby("month")["net_pnl"].sum().round(2))

    best = rt.loc[net.idxmax()] if len(rt) else None
    worst = rt.loc[net.idxmin()] if len(rt) else None

    return {
        "bucket": bucket or "all",
        "n_trades": int(len(rt)),
        "win_rate": round(len(wins) / len(rt) * 100, 1),
        "gross_pnl": round(gross_pnl, 2),
        "total_charges": round(total_charges, 2),
        "net_pnl": round(float(net.sum()), 2),
        "charges_pct_of_turnover": round(total_charges / turnover * 100, 3) if turnover else 0.0,
        "charges_pct_of_gross": round(total_charges / gross_pnl * 100, 1) if gross_pnl > 0 else None,
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else None,
        "expectancy": round(float(net.mean()), 2),
        "avg_holding_days": round(float(rt["holding_days"].mean()), 1),
        "best": {"ticker": best["ticker"], "net_pnl": float(best["net_pnl"])} if best is not None else None,
        "worst": {"ticker": worst["ticker"], "net_pnl": float(worst["net_pnl"])} if worst is not None else None,
        "by_month": by_month,
        "tax": _tax_estimate(rt),
    }


def benchmark_alpha(roundtrips: pd.DataFrame, benchmark: str = "NIFTY.NS") -> dict | None:
    """Rough realized return-on-capital-deployed vs NIFTY buy-and-hold over the
    same window. NOT time-weighted or annualized — capital is recycled across
    trades — so read it as a sanity check, not a precise alpha."""
    if roundtrips.empty:
        return None
    deployed = float((roundtrips["buy_price"] * roundtrips["qty"]).sum())
    net = float(roundtrips["net_pnl"].sum())
    realized_return = net / deployed * 100 if deployed else 0.0
    start, end = roundtrips["buy_date"].min(), roundtrips["sell_date"].max()

    bench_return = None
    try:
        bench = load_macro(benchmark)
        if bench is not None and not bench.empty:
            b = bench[(bench.index >= start) & (bench.index <= end)]
            if len(b) >= 2:
                bench_return = (float(b["close"].iloc[-1]) / float(b["close"].iloc[0]) - 1) * 100
    except Exception:
        bench_return = None

    return {
        "deployed": round(deployed, 2),
        "net_pnl": round(net, 2),
        "realized_return_pct": round(realized_return, 2),
        "benchmark": benchmark,
        "benchmark_return_pct": round(bench_return, 2) if bench_return is not None else None,
        "alpha_pct": round(realized_return - bench_return, 2) if bench_return is not None else None,
        "start": start, "end": end,
    }
