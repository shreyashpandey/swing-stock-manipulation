"""Single source of truth for share-structure / turnover / liquidity metrics.

The Manipulation tab and the Screener (liquidity) tab historically computed
turnover-vs-market-cap two different ways — Manipulation off **today's**
``close × volume`` and the Screener off the **60-day average** — so the same
stock showed two different turnover numbers. They are genuinely different reads
(spike-detection vs typical tradeability), but they were computed in two places
with no shared definition.

This module is that shared definition. It is **pure** (no storage imports at
module scope; numpy/pandas only) so both :mod:`liquidity` and
:mod:`manipulation` can import *down* into it without a cycle and delegate their
arithmetic here. Both windows live on one :class:`MarketMetrics` object:

  * ``today_*``  — the last bar (what Manipulation surfaces).
  * ``adv_*`` / ``avg_*`` — the trailing ``adv_window`` average (what Liquidity
    surfaces).
  * ``today_vs_avg_value_mult`` — the reconciliation: "today is N× the usual
    ₹/day", which makes the two numbers obviously consistent rather than
    contradictory.

The formulas here reproduce the existing per-module arithmetic exactly, so the
delegating modules keep their public behaviour (and their tests) unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Canonical windows. ADV_WINDOW matches liquidity's default lookback; BASE_WINDOW
# matches manipulation's WINDOW (the "prior 20 sessions" baseline, last bar
# excluded).
ADV_WINDOW = 60
BASE_WINDOW = 20


@dataclass
class MarketMetrics:
    ticker: str
    spot: float
    market_cap: float | None
    float_shares: float | None
    float_basis: str                      # "float" | "shares_out" | "derived" | "n/a"
    # --- TODAY (last bar) — Manipulation's numbers ---
    today_value_cr: float
    today_turnover_pct: float | None      # today_value / mcap * 100
    today_float_turnover_pct: float | None
    today_vol_mult: float | None          # today_vol / mean(prior BASE_WINDOW vols)
    today_value_spike_mult: float | None  # today_value / median(prior BASE_WINDOW values)
    # --- AVERAGE (trailing adv_window) — Liquidity / Screener's numbers ---
    adv_value_cr: float                   # mean(close*vol) over adv_window / 1e7
    adv_volume: float
    avg_turnover_pct: float | None        # adv_value / mcap * 100
    avg_float_turnover_pct: float | None  # adv_volume / float * 100
    amihud: float                         # avg |ret| per ₹1 cr traded, over adv_window
    median_value_cr: float
    zero_vol_days: int
    # --- reconciliation ---
    today_vs_avg_value_mult: float | None  # today_value / adv_value
    n_bars: int


def _resolve_float(fund: dict | None, spot: float | None,
                   *, allow_derive: bool = False) -> tuple[float | None, str]:
    """Resolve the share count to measure volume against, with a documented
    fallback so a read is never silently optimistic.

    Order: free float → shares outstanding → (only if ``allow_derive``) derived
    ``market_cap / price``. Returns ``(shares, basis)`` where basis is one of
    ``float`` / ``shares_out`` / ``derived`` / ``n/a``.

    With ``allow_derive=False`` this reproduces :mod:`liquidity`'s exact
    fallback (which never derives), keeping that module's behaviour identical.
    """
    fund = fund or {}
    float_sh = fund.get("float_shares")
    basis = "float"
    if not float_sh or (isinstance(float_sh, (int, float)) and not np.isfinite(float_sh)):
        float_sh = fund.get("shares_outstanding")
        basis = "shares_out" if float_sh else "n/a"
    if (not float_sh) and allow_derive:
        mcap = fund.get("market_cap")
        if mcap and np.isfinite(mcap) and spot and spot > 0:
            float_sh = mcap / spot
            basis = "derived"
    return (float(float_sh) if float_sh else None), basis


def _today_turnover(df: pd.DataFrame, market_cap: float | None,
                    base_window: int = BASE_WINDOW
                    ) -> tuple[float, float, float, float]:
    """The four numbers :func:`manipulation.turnover_vs_marketcap` needs, in one
    place: ``(today_value, turnover_ratio, median_value, spike_mult)``.

    ``today_value`` is the last bar's ``close × volume``. ``turnover_ratio`` is
    that over market cap (NaN if no mcap). ``median_value`` / ``spike_mult`` use
    the prior ``base_window`` bars **excluding the last** (so a spike never
    contaminates the norm it is measured against) — matching manipulation's
    ``_baseline`` semantics exactly. The caller is responsible for the history
    gate; this stays pure so it works on any in-memory frame.
    """
    traded_value = df["close"] * df["volume"]
    today_value = float(traded_value.iloc[-1])
    turnover_ratio = today_value / market_cap if market_cap else float("nan")

    base = traded_value.iloc[:-1].tail(base_window).dropna()
    median_value = float(base.median()) if len(base) else float("nan")
    spike_mult = (today_value / median_value
                  if median_value and median_value > 0 else float("nan"))
    return today_value, turnover_ratio, median_value, spike_mult


def compute(df: pd.DataFrame, fund: dict | None, *,
            adv_window: int = ADV_WINDOW, base_window: int = BASE_WINDOW,
            allow_derive: bool = False, ticker: str = "") -> MarketMetrics | None:
    """Compute every share-structure metric for a price frame in one pass.

    ``df`` is OHLCV (DatetimeIndex, columns incl. ``close``/``volume``); ``fund``
    is a fundamentals row (``market_cap`` / ``float_shares`` /
    ``shares_outstanding``). The trailing-average block reproduces
    :func:`liquidity.liquidity_profile`'s arithmetic; the today/spike block
    reproduces :func:`manipulation.turnover_vs_marketcap`'s. Returns ``None``
    only on an unusably short frame (callers apply their own stricter gates).
    """
    if df is None or df.empty or len(df) < 2:
        return None
    fund = fund or {}
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    spot = float(close.iloc[-1])

    mcap = fund.get("market_cap")
    market_cap = float(mcap) if mcap and np.isfinite(mcap) else None
    float_sh, float_basis = _resolve_float(fund, spot, allow_derive=allow_derive)

    # --- TODAY (last bar) + spike vs prior baseline ---
    today_value, turnover_ratio, median_value, spike_mult = _today_turnover(
        df, market_cap, base_window)
    today_vol = float(volume.iloc[-1])
    vol_base = volume.iloc[:-1].tail(base_window).dropna()
    mean_base = float(vol_base.mean()) if len(vol_base) else float("nan")
    today_vol_mult = (today_vol / mean_base
                      if mean_base and mean_base > 0 else None)
    today_turnover_pct = (turnover_ratio * 100
                          if market_cap and turnover_ratio == turnover_ratio else None)
    today_float_turnover_pct = (today_vol / float_sh * 100) if float_sh else None
    today_value_spike_mult = spike_mult if spike_mult == spike_mult else None

    # --- AVERAGE (trailing adv_window) — mirrors liquidity exactly ---
    vol_w = volume.tail(adv_window)
    cl_w = close.tail(adv_window)
    tv_w = cl_w * vol_w
    adv_value = float(tv_w.mean())
    adv_volume = float(vol_w.mean())
    zero_vol_days = int((vol_w <= 0).sum())

    ret = close.pct_change().tail(adv_window).abs()
    tv_cr = (tv_w / 1e7).replace(0, np.nan)
    amihud_series = (ret / tv_cr).replace([np.inf, -np.inf], np.nan).dropna()
    amihud = float(amihud_series.mean()) if len(amihud_series) else float("nan")

    avg_turnover_pct = (adv_value / market_cap * 100) if market_cap else None
    avg_float_turnover_pct = (adv_volume / float_sh * 100) if float_sh else None
    today_vs_avg_value_mult = (today_value / adv_value
                               if adv_value and adv_value > 0 else None)

    return MarketMetrics(
        ticker=ticker, spot=spot, market_cap=market_cap,
        float_shares=float_sh, float_basis=float_basis,
        today_value_cr=today_value / 1e7,
        today_turnover_pct=today_turnover_pct,
        today_float_turnover_pct=today_float_turnover_pct,
        today_vol_mult=today_vol_mult,
        today_value_spike_mult=today_value_spike_mult,
        adv_value_cr=adv_value / 1e7, adv_volume=adv_volume,
        avg_turnover_pct=avg_turnover_pct,
        avg_float_turnover_pct=avg_float_turnover_pct,
        amihud=amihud,
        median_value_cr=(median_value / 1e7) if median_value == median_value else float("nan"),
        zero_vol_days=zero_vol_days,
        today_vs_avg_value_mult=today_vs_avg_value_mult,
        n_bars=len(df),
    )


def for_ticker(ticker: str, fund: dict | None = None,
               adv_window: int = ADV_WINDOW,
               allow_derive: bool = True) -> MarketMetrics | None:
    """Storage-backed convenience: load prices + fundamentals and compute.

    This is the only function here that touches storage (imported lazily to keep
    the module import-light and the pure helpers test-friendly).
    """
    from swingdesk.storage import get_fundamentals, load_prices

    df = load_prices(ticker, days=adv_window + 5)
    if df is None or df.empty:
        return None
    if fund is None:
        fund = get_fundamentals(ticker) or {}
    return compute(df, fund, adv_window=adv_window,
                   allow_derive=allow_derive, ticker=ticker)
