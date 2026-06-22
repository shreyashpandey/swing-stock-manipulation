"""Liquidity / tradeability profile — the missing dimension behind "this stock
looks great on the chart but I can't get in or out."

A setup can be perfect on momentum/quality/trend and still be a trap if the
stock barely trades. We measure the three things that matter for a retail swing
trader actually being able to enter and EXIT at a fair price:

  1. ADV (avg daily traded VALUE, ₹ crore) — can you build/exit a position
     without being most of the day's volume?
  2. Turnover vs market-cap (ADV ÷ mcap) — how actively the company trades
     relative to its size; tiny = neglected / pump-prone.
  3. Volume vs float (ADV-volume ÷ free-float shares) — the real exit gauge:
     what fraction of the *tradeable* shares change hands daily.
  Plus Amihud illiquidity (price impact per ₹ traded) and zero-volume days.

Everything is causal (trailing window) and returns a 0–100 score, a tier, and
plain-English reasons so the user knows WHY a name is or isn't tradeable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swingdesk.analyze import market_metrics
from swingdesk.storage import load_prices

# Thresholds tuned for Indian retail swing trading (₹). Adjust to taste.
ADV_CR_LIQUID = 10.0      # ≥ ₹10 cr/day traded value = comfortably liquid
ADV_CR_OK = 3.0           # ₹3–10 cr = fine for modest size
ADV_CR_THIN = 1.0         # ₹1–3 cr = thin; < ₹1 cr = illiquid
FLOAT_TO_HIGH = 1.0       # ≥1.0% of float traded/day = healthy churn
FLOAT_TO_OK = 0.3         # 0.3–1.0% ok; < 0.1% = very thin
TURN_OK = 0.3             # ADV/mcap ≥0.3%/day ok; < 0.05% = neglected


@dataclass
class LiquidityProfile:
    ticker: str
    adv_value_cr: float        # avg daily traded value, ₹ crore
    adv_volume: float          # avg daily shares
    turnover_pct: float        # ADV ÷ market-cap, % per day
    float_turnover_pct: float  # ADV-volume ÷ float, % per day
    amihud: float              # price impact: avg |ret| per ₹ crore traded
    zero_vol_days: int
    float_basis: str           # "float" | "shares_out" | "n/a"
    score: float               # 0–100 (higher = more liquid)
    tier: str                  # liquid | moderate | illiquid | untradeable
    reasons: list[str] = field(default_factory=list)


def _scale(x: float, lo: float, hi: float) -> float:
    """Linear 0–100 between lo and hi, clamped."""
    if not np.isfinite(x):
        return 0.0
    return float(np.clip((x - lo) / (hi - lo) * 100, 0, 100))


def liquidity_profile(ticker: str, fund: dict | None = None,
                      lookback: int = 60) -> LiquidityProfile | None:
    px = load_prices(ticker, days=lookback + 5)
    if px.empty or len(px) < 20:
        return None

    if fund is None:                       # not supplied → fetch the row ourselves
        from swingdesk.storage import get_fundamentals
        fund = get_fundamentals(ticker) or {}

    # Shared source of truth: the trailing-average block (ADV, turnover-vs-mcap,
    # float turnover, Amihud, zero-vol days, float basis) is computed once in
    # market_metrics and reused by both this module and the manipulation tab.
    # allow_derive=False keeps the historic behaviour (this module never derives
    # a share count from market_cap/price).
    mm = market_metrics.compute(px, fund, adv_window=lookback, allow_derive=False)
    adv_value_cr = mm.adv_value_cr
    adv_volume = mm.adv_volume
    zero_vol_days = mm.zero_vol_days
    amihud = mm.amihud
    turnover_pct = mm.avg_turnover_pct if mm.avg_turnover_pct is not None else float("nan")
    float_turnover_pct = (mm.avg_float_turnover_pct
                          if mm.avg_float_turnover_pct is not None else float("nan"))
    basis = mm.float_basis

    # --- transparent 0–100 score
    s_value = _scale(np.log10(max(adv_value_cr, 1e-3)),
                     np.log10(ADV_CR_THIN), np.log10(ADV_CR_LIQUID))   # ₹1cr→0, ₹10cr→100
    s_float = _scale(float_turnover_pct, FLOAT_TO_OK / 3, FLOAT_TO_HIGH) if np.isfinite(float_turnover_pct) else 50.0
    s_turn = _scale(turnover_pct, 0.05, TURN_OK) if np.isfinite(turnover_pct) else 50.0
    score = 0.45 * s_value + 0.30 * s_float + 0.25 * s_turn - 6 * zero_vol_days
    score = float(np.clip(score, 0, 100))

    tier = ("liquid" if score >= 65 else "moderate" if score >= 40
            else "illiquid" if score >= 20 else "untradeable")

    # --- reasons
    reasons = []
    reasons.append(f"₹{adv_value_cr:.1f} cr/day traded — " +
                   ("comfortably liquid" if adv_value_cr >= ADV_CR_LIQUID
                    else "fine for modest size" if adv_value_cr >= ADV_CR_OK
                    else "thin" if adv_value_cr >= ADV_CR_THIN else "illiquid (hard to exit)"))
    if np.isfinite(float_turnover_pct):
        reasons.append(f"{float_turnover_pct:.2f}% of {('float' if basis=='float' else 'shares')}"
                       f" traded/day — " +
                       ("healthy churn" if float_turnover_pct >= FLOAT_TO_HIGH
                        else "ok" if float_turnover_pct >= FLOAT_TO_OK else "very thin float turnover"))
    if np.isfinite(turnover_pct):
        reasons.append(f"turnover {turnover_pct:.2f}%/day vs mcap — " +
                       ("active" if turnover_pct >= TURN_OK else "neglected" if turnover_pct < 0.05 else "moderate"))
    if zero_vol_days:
        reasons.append(f"⚠ {zero_vol_days} zero-volume day(s) in {lookback} — gaps/no fills")
    if tier in ("illiquid", "untradeable"):
        reasons.append("⚠ illiquidity risk: slippage on entry/exit, prone to manipulation — size tiny or skip")

    return LiquidityProfile(
        ticker=ticker, adv_value_cr=round(adv_value_cr, 2), adv_volume=round(adv_volume, 0),
        turnover_pct=round(turnover_pct, 3) if np.isfinite(turnover_pct) else float("nan"),
        float_turnover_pct=round(float_turnover_pct, 3) if np.isfinite(float_turnover_pct) else float("nan"),
        amihud=round(amihud, 4) if np.isfinite(amihud) else float("nan"),
        zero_vol_days=zero_vol_days, float_basis=basis,
        score=round(score, 1), tier=tier, reasons=reasons,
    )
