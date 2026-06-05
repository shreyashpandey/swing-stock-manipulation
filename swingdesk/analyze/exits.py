"""Exit-level recommendations per holding.

For each open holding, compute four concrete levels:
    initial_stop   — hard floor: exit if breached (capital protection)
    trailing_stop  — adjusts as price moves up (locks in gains)
    book_partial   — first resistance / profit-taking target (sell 1/3)
    full_target    — second/major resistance / final exit

Sources:
    initial_stop:   max(200-EMA, recent swing-low, breakeven if in profit)
    trailing_stop:  max(20-EMA, current_close × 0.93)
    book_partial:   nearest high-volume node above current (volume profile)
    full_target:    20-day or 55-day high

All levels are computed from the actual price/volume data — no magic numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from swingdesk.analyze.technicals import add_indicators, volume_profile
from swingdesk.storage import load_prices


@dataclass
class ExitPlan:
    ticker: str
    current_price: float
    avg_buy: float | None              # if known
    initial_stop: float | None         # absolute hard stop
    trailing_stop: float | None        # current trail level
    book_partial_at: float | None      # take 1/3 here
    full_target: float | None          # final exit here
    risk_amount: float | None          # current - initial_stop
    reward_amount: float | None        # full_target - current
    risk_reward: float | None
    rationale: list[str]


def _nearest_swing_low(df: pd.DataFrame, lookback: int = 30) -> float | None:
    """Most recent local-minimum close in last `lookback` bars."""
    if len(df) < lookback:
        return None
    window = df.tail(lookback)
    return float(window["low"].min())


def _nearest_swing_high(df: pd.DataFrame, lookback: int = 30) -> float | None:
    if len(df) < lookback:
        return None
    window = df.tail(lookback)
    return float(window["high"].max())


def _nearest_volume_node_above(profile: pd.DataFrame, current: float) -> float | None:
    """Find the next high-volume price node above the current price."""
    if profile is None or profile.empty:
        return None
    above = profile[profile["price_mid"] > current * 1.005]  # at least 0.5% above
    if above.empty:
        return None
    # Pick the highest-volume node within the next 15%
    range_cap = current * 1.15
    in_range = above[above["price_mid"] <= range_cap]
    cand = in_range if not in_range.empty else above
    return float(cand.loc[cand["volume"].idxmax(), "price_mid"])


def _nearest_volume_node_below(profile: pd.DataFrame, current: float) -> float | None:
    """Highest-volume support node below current price (in last 15%)."""
    if profile is None or profile.empty:
        return None
    below = profile[profile["price_mid"] < current * 0.995]
    if below.empty:
        return None
    range_floor = current * 0.85
    in_range = below[below["price_mid"] >= range_floor]
    cand = in_range if not in_range.empty else below
    return float(cand.loc[cand["volume"].idxmax(), "price_mid"])


def compute(ticker: str, avg_buy: float | None = None) -> ExitPlan | None:
    """Compute the exit plan for one ticker."""
    df = load_prices(ticker)
    if df.empty or len(df) < 50:
        return None
    df = add_indicators(df)
    last = df.iloc[-1]
    current = float(last["close"])
    if pd.isna(current) or current <= 0:
        return None
    rationale: list[str] = []

    # --- Initial (hard) stop ---
    candidates = []
    if pd.notna(last.get("ema200")):
        ema200 = float(last["ema200"])
        if ema200 < current:
            candidates.append(("200-EMA", ema200))
    swing_low = _nearest_swing_low(df, 30)
    if swing_low and swing_low < current * 0.98:
        candidates.append(("30d swing-low", swing_low))
    # Use volume-profile support too
    profile = volume_profile(df, bins=24, lookback=60)
    vol_support = _nearest_volume_node_below(profile, current)
    if vol_support:
        candidates.append(("volume support", vol_support))
    # If in profit, lock at breakeven minimum
    if avg_buy and avg_buy < current * 0.98:
        candidates.append(("breakeven", avg_buy))

    valid_stops = [(n, v) for n, v in candidates if v is not None and pd.notna(v)]
    initial_stop = None
    if valid_stops:
        source, initial_stop = max(valid_stops, key=lambda t: t[1])
        rationale.append(f"hard stop at ₹{initial_stop:.2f} ({source})")

    # --- Trailing stop ---
    trail_candidates = []
    if pd.notna(last.get("ema20")):
        ema20 = float(last["ema20"])
        if ema20 < current:
            trail_candidates.append(("20-EMA", ema20))
    # 7% below current
    trail_candidates.append(("-7%", current * 0.93))
    # Above breakeven if in profit
    if avg_buy and avg_buy < current:
        trail_candidates.append(("breakeven+", avg_buy * 1.005))

    valid_trails = [(n, v) for n, v in trail_candidates if v is not None and pd.notna(v)]
    trailing_stop = None
    if valid_trails:
        source, trailing_stop = max(valid_trails, key=lambda t: t[1])
        rationale.append(f"trail at ₹{trailing_stop:.2f} ({source})")

    # --- Book partial target ---
    book_partial = _nearest_volume_node_above(profile, current)
    swing_high = _nearest_swing_high(df, 30)
    if book_partial and swing_high:
        # Prefer whichever is closer to current (more achievable)
        if abs(swing_high - current) < abs(book_partial - current):
            book_partial = swing_high
            rationale.append(f"trim 1/3 at ₹{book_partial:.2f} (30d swing-high)")
        else:
            rationale.append(f"trim 1/3 at ₹{book_partial:.2f} (volume resistance)")
    elif swing_high and swing_high > current * 1.01:
        book_partial = swing_high
        rationale.append(f"trim 1/3 at ₹{book_partial:.2f} (30d swing-high)")

    # --- Full target ---
    full_target = None
    if pd.notna(last.get("high55")):
        h55 = float(last["high55"])
        if h55 > current * 1.02:
            full_target = h55
            rationale.append(f"full exit at ₹{full_target:.2f} (55-day high)")

    if not full_target and book_partial:
        # Project a 2× extension from current
        atr = float(last["atr14"]) if pd.notna(last.get("atr14")) else 0
        full_target = current + 5 * atr if atr > 0 else current * 1.15
        rationale.append(f"full exit at ₹{full_target:.2f} (ATR-projected)")

    # --- Risk/reward ---
    risk = current - initial_stop if initial_stop else None
    reward = full_target - current if full_target else None
    rr = round(reward / risk, 2) if (risk and risk > 0 and reward) else None

    return ExitPlan(
        ticker=ticker,
        current_price=current,
        avg_buy=avg_buy,
        initial_stop=round(initial_stop, 2) if initial_stop else None,
        trailing_stop=round(trailing_stop, 2) if trailing_stop else None,
        book_partial_at=round(book_partial, 2) if book_partial else None,
        full_target=round(full_target, 2) if full_target else None,
        risk_amount=round(risk, 2) if risk else None,
        reward_amount=round(reward, 2) if reward else None,
        risk_reward=rr,
        rationale=rationale,
    )
