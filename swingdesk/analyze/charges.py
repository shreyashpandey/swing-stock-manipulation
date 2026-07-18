"""Indian equity charge model (Groww discount-broker rate card) + capital-gains
classification.

Pure functions; every rate lives in :mod:`swingdesk.config` so the numbers are
auditable and overridable. Modelled charges are **indicative** — Groww's
brokerage and India's capital-gains rates have changed over time, so for exact
figures prefer the real charge columns from a Groww contract-note / Tax-P&L
export. This module is the single place that turns "₹X traded" into the actual
brokerage + STT + exchange + GST + stamp + DP cost a retail delivery trade pays.

Percentages in config are "percent of turnover" (0.10 == 0.10%).
"""
from __future__ import annotations

import pandas as pd

from swingdesk.config import (
    DP_CHARGE_PER_SELL,
    EXCHANGE_TXN_PCT_BSE,
    EXCHANGE_TXN_PCT_NSE,
    GROWW_BROKERAGE_CAP,
    GROWW_BROKERAGE_PCT_DELIVERY,
    GROWW_BROKERAGE_PCT_INTRADAY,
    GST_PCT,
    LTCG_HOLDING_DAYS,
    LTCG_TAX_PCT,
    SEBI_TXN_PCT,
    STAMP_PCT_DELIVERY,
    STAMP_PCT_INTRADAY,
    STCG_TAX_PCT,
    STT_PCT_DELIVERY,
    STT_PCT_INTRADAY,
)

# The itemised line items, in display order. `total` is the sum.
CHARGE_KEYS = ["brokerage", "stt", "exchange_txn", "sebi", "stamp_duty", "gst", "dp_charge", "total"]


def _r2(x: float) -> float:
    return round(float(x), 2)


def leg_charges(side: str, price: float, qty: float, *,
                segment: str = "delivery", exchange: str = "NSE") -> dict:
    """Itemised charges for ONE executed leg (a buy or a sell).

    ``segment`` is "delivery" | "intraday"; ``exchange`` is "NSE" | "BSE".
    Returns a dict with the keys in :data:`CHARGE_KEYS` (rupees)."""
    side = str(side).lower().strip()
    qty = abs(float(qty or 0))
    turnover = float(price or 0) * qty
    delivery = str(segment).lower().strip() != "intraday"
    is_sell = side in ("sell", "s")

    if turnover <= 0:
        return {k: 0.0 for k in CHARGE_KEYS}

    # Brokerage: lower of the per-order cap and the percentage of turnover.
    brok_pct = GROWW_BROKERAGE_PCT_DELIVERY if delivery else GROWW_BROKERAGE_PCT_INTRADAY
    brokerage = min(GROWW_BROKERAGE_CAP, turnover * brok_pct / 100.0)

    # STT: delivery taxes both legs; intraday taxes only the sell.
    if delivery:
        stt = turnover * STT_PCT_DELIVERY / 100.0
    else:
        stt = turnover * STT_PCT_INTRADAY / 100.0 if is_sell else 0.0

    ex_pct = EXCHANGE_TXN_PCT_BSE if str(exchange).upper().startswith("BSE") else EXCHANGE_TXN_PCT_NSE
    exchange_txn = turnover * ex_pct / 100.0
    sebi = turnover * SEBI_TXN_PCT / 100.0

    # Stamp duty: buy leg only.
    stamp = (turnover * (STAMP_PCT_DELIVERY if delivery else STAMP_PCT_INTRADAY) / 100.0
             if side in ("buy", "b") else 0.0)

    # GST: 18% on brokerage + exchange txn + SEBI fee.
    gst = (brokerage + exchange_txn + sebi) * GST_PCT / 100.0

    # DP charge: levied on a delivery SELL (per scrip), already incl. its GST.
    dp = DP_CHARGE_PER_SELL if (delivery and is_sell) else 0.0

    total = brokerage + stt + exchange_txn + sebi + stamp + gst + dp
    return {
        "brokerage": _r2(brokerage), "stt": _r2(stt), "exchange_txn": _r2(exchange_txn),
        "sebi": _r2(sebi), "stamp_duty": _r2(stamp), "gst": _r2(gst),
        "dp_charge": _r2(dp), "total": _r2(total),
    }


def merge_charges(*dicts: dict) -> dict:
    """Sum several charge dicts line-item by line-item."""
    return {k: _r2(sum(d.get(k, 0.0) for d in dicts)) for k in CHARGE_KEYS}


def round_trip_charges(buy_price: float, sell_price: float, qty: float, *,
                       segment: str = "delivery", exchange: str = "NSE") -> dict:
    """Total charges for a full buy→sell round trip."""
    return merge_charges(
        leg_charges("buy", buy_price, qty, segment=segment, exchange=exchange),
        leg_charges("sell", sell_price, qty, segment=segment, exchange=exchange),
    )


def exit_charges(price: float, qty: float, *,
                 segment: str = "delivery", exchange: str = "NSE") -> dict:
    """Sell-side-only charges — 'what it would cost to exit this position today'."""
    return leg_charges("sell", price, qty, segment=segment, exchange=exchange)


def classify_gain(buy_date, sell_date, net_gain: float) -> dict:
    """Short- vs long-term classification for listed equity + an indicative
    per-trade tax figure.

    ``tax_est`` here is a naive per-trade number that does NOT apply the annual
    ₹1.25L LTCG exemption — that exemption is portfolio-wide, so
    :func:`swingdesk.analyze.pnl_report.performance` recomputes the real tax at
    the aggregate level. Use this for per-row display + bucketing only."""
    bd, sd = pd.Timestamp(buy_date), pd.Timestamp(sell_date)
    days = int((sd - bd).days)
    # Listed equity is long-term only when held > 12 months (i.e. > 365 days);
    # a holding of exactly 365 days is still short-term.
    is_ltcg = days > LTCG_HOLDING_DAYS
    rate = LTCG_TAX_PCT if is_ltcg else STCG_TAX_PCT
    tax = max(0.0, float(net_gain)) * rate / 100.0
    return {
        "holding_days": days,
        "gain_type": "LTCG" if is_ltcg else "STCG",
        "tax_rate_pct": rate,
        "tax_est": _r2(tax),
    }
