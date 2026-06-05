"""Turn recommendations into concrete *numbers* — how many shares / how many
rupees to buy, hold, reduce or sell — and a fresh-money distribution engine.

Two public entry points:

  action_plan(analysis, portfolio_value)  -> ActionPlan
      For an existing holding: given the BUY_MORE/HOLD/REDUCE/SELL verdict and
      its exit levels, compute the exact share count and ₹ amount to trade,
      sized by risk (never risking more than RISK_PER_TRADE_PCT of the book on
      a top-up) and capped by a concentration limit.

  allocate(amount, opportunities, ...)     -> AllocationResult
      For new money: distribute a rupee amount across the best-ranked ideas,
      conviction-weighted, capped per position, floored to whole shares — with
      per-name entry / stop / target and the risk each slice carries.

All sizing is rule-based and transparent; the UI shows the why next to the how.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swingdesk.config import (
    MAX_OPEN_POSITIONS,
    RISK_PER_TRADE_PCT,
)
from swingdesk.analyze.setups import scan_ticker
from swingdesk.analyze.technicals import add_indicators
from swingdesk.storage import load_prices

# How aggressively to weight an idea by its conviction band.
CONVICTION_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}
# No single fresh position may exceed this share of the deployed amount.
DEFAULT_MAX_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# Existing-holding action numbers
# ---------------------------------------------------------------------------
@dataclass
class ActionPlan:
    ticker: str
    action: str               # BUY MORE | HOLD | REDUCE | SELL | NO DATA
    shares: int               # how many shares to trade (0 for HOLD)
    rupees: float             # ₹ value of that trade (at last price)
    last_price: float | None
    note: str                 # plain-English instruction with the number in it


def _levels(ticker: str) -> tuple[float, float, float, str | None] | None:
    """(entry, stop, target, setup) for a ticker.

    Uses a live setup's pre-computed levels when one is firing; otherwise falls
    back to an ATR-based bracket off the last close (stop 1.5×ATR, target 3×ATR).
    """
    sigs = scan_ticker(ticker)
    if sigs:
        s = sigs[0]
        if s.get("entry") and s.get("stoploss") and s.get("target"):
            return float(s["entry"]), float(s["stoploss"]), float(s["target"]), s["setup"]
    df = load_prices(ticker)
    if df.empty or len(df) < 50:
        return None
    df = add_indicators(df)
    last = df.iloc[-1]
    close = float(last["close"])
    atr = float(last["atr14"]) if "atr14" in df.columns and not math.isnan(last["atr14"]) else None
    if not atr or atr <= 0:
        return None
    return close, round(close - 1.5 * atr, 2), round(close + 3.0 * atr, 2), None


def action_plan(a, portfolio_value: float | None = None,
                max_weight: float = DEFAULT_MAX_WEIGHT) -> ActionPlan:
    """Translate a HoldingAnalysis into a concrete share/₹ instruction."""
    last = a.last_price
    qty = float(a.qty or 0)
    rec = a.recommendation
    ee = getattr(a, "early_exit_action", None)

    def _plan(action, shares, note):
        shares = max(0, int(shares))
        rupees = round(shares * last, 0) if last else 0.0
        return ActionPlan(a.ticker, action, shares, rupees, last, note)

    if rec == "NO_DATA" or not last:
        return _plan("NO DATA", 0, "Not enough data to size a trade.")

    # --- Sells / trims: fraction of the existing position ---
    if rec == "SELL" or ee == "EXIT":
        return _plan("SELL", round(qty),
                     f"Exit fully — sell all {int(qty)} shares (≈₹{qty*last:,.0f}). "
                     f"{a.reasons[0] if a.reasons else ''}".strip())
    if ee == "TRIM_50" or (rec == "REDUCE" and (a.pnl_pct or 0) >= 0):
        sh = round(qty * 0.5)
        return _plan("REDUCE", sh,
                     f"Trim half — sell {sh} of {int(qty)} shares (≈₹{sh*last:,.0f}), "
                     "let the rest run with a trailing stop.")
    if ee == "TRIM_25":
        sh = round(qty * 0.25)
        return _plan("REDUCE", sh,
                     f"Trim a quarter — sell {sh} of {int(qty)} shares (≈₹{sh*last:,.0f}).")
    if rec == "REDUCE":
        sh = round(qty * (0.5 if (a.portfolio_weight or 0) > max_weight else 0.33))
        return _plan("REDUCE", sh,
                     f"Reduce — sell {sh} of {int(qty)} shares (≈₹{sh*last:,.0f}) "
                     "to cut risk / concentration.")

    # --- Buy more: size the top-up by risk, capped by concentration ---
    if rec == "BUY_MORE":
        stop = a.initial_stop
        add_by_risk = None
        if portfolio_value and stop and last > stop:
            risk_budget = portfolio_value * (RISK_PER_TRADE_PCT / 100.0)
            add_by_risk = int(risk_budget / (last - stop))
        # Concentration cap: don't let the position exceed max_weight of the book.
        add_by_cap = None
        if portfolio_value:
            cap_value = max_weight * portfolio_value
            current_value = qty * last
            add_by_cap = max(0, int((cap_value - current_value) / last))
        candidates = [x for x in (add_by_risk, add_by_cap) if x is not None]
        add = min(candidates) if candidates else max(1, int(qty * 0.25))
        if add <= 0:
            return _plan("HOLD", 0,
                         "Conviction is high, but the position is already at its "
                         "size cap — add only on a fresh pullback.")
        return _plan("BUY MORE", add,
                     f"Add {add} shares (≈₹{add*last:,.0f}), risk-sized to ~"
                     f"{RISK_PER_TRADE_PCT:.0f}% of the book against the ₹{stop} stop."
                     if stop else f"Add {add} shares (≈₹{add*last:,.0f}).")

    # --- Hold ---
    note = "Hold — no action."
    if a.trailing_stop:
        note = f"Hold — ride it; trail the stop at ₹{a.trailing_stop}."
    return _plan("HOLD", 0, note)


# ---------------------------------------------------------------------------
# Fresh-money distribution
# ---------------------------------------------------------------------------
@dataclass
class Allocation:
    ticker: str
    company: str
    sector: str | None
    conviction: str
    price: float
    shares: int
    rupees: float
    weight_pct: float            # share of the *amount* this slice represents
    entry: float | None
    stoploss: float | None
    target: float | None
    risk_rupees: float | None    # (entry-stop) × shares — ₹ at risk if stopped
    rr: float | None             # reward:risk on the bracket
    setup: str | None
    reasons: list[str] = field(default_factory=list)


@dataclass
class AllocationResult:
    amount: float
    deployed: float
    leftover: float
    n_positions: int
    allocations: list[Allocation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


_CONV_RANK = {"high": 3, "medium": 2, "low": 1}


def allocate(amount: float, opportunities: list,
             max_positions: int | None = None,
             max_weight: float = DEFAULT_MAX_WEIGHT,
             min_conviction: str = "medium") -> AllocationResult:
    """Distribute `amount` across the best opportunities.

    Strategy: keep ideas at/above `min_conviction`, weight each by its
    conviction band × composite score, cap any single name at `max_weight`,
    take the top `max_positions`, then floor each slice to whole shares.
    """
    max_positions = max_positions or MAX_OPEN_POSITIONS
    notes: list[str] = []

    min_rank = _CONV_RANK.get(min_conviction, 2)
    pool = [o for o in opportunities
            if _CONV_RANK.get(getattr(o, "conviction", "low"), 1) >= min_rank
            and getattr(o, "price", 0)]
    if not pool:
        notes.append(f"No ideas at '{min_conviction}'+ conviction — lower the bar "
                     "or scan a wider universe.")
        return AllocationResult(amount, 0.0, amount, 0, [], notes)

    # Best first, then take the top N we're willing to hold concurrently.
    pool.sort(key=lambda o: (_CONV_RANK.get(o.conviction, 1), o.composite_score),
              reverse=True)
    chosen = pool[:max_positions]
    if len(pool) > max_positions:
        notes.append(f"Showing the top {max_positions} of {len(pool)} qualifying "
                     f"ideas (MAX_OPEN_POSITIONS={max_positions}).")

    # Raw weights = conviction × score, then normalise and apply the per-name cap.
    raw = {o.ticker: CONVICTION_WEIGHT.get(o.conviction, 1.0) * max(o.composite_score, 1.0)
           for o in chosen}
    total_raw = sum(raw.values()) or 1.0
    weights = {t: w / total_raw for t, w in raw.items()}
    # Cap & redistribute once (simple, good enough for a handful of names).
    capped = {t: min(w, max_weight) for t, w in weights.items()}
    spill = 1.0 - sum(capped.values())
    uncapped = [t for t in capped if weights[t] < max_weight]
    if spill > 1e-6 and uncapped:
        share = sum(weights[t] for t in uncapped) or 1.0
        for t in uncapped:
            capped[t] = min(max_weight, capped[t] + spill * (weights[t] / share))

    allocs: list[Allocation] = []
    deployed = 0.0
    for o in chosen:
        target_rupees = capped[o.ticker] * amount
        shares = int(target_rupees // o.price)
        if shares <= 0:
            continue
        rupees = round(shares * o.price, 0)
        deployed += rupees
        lv = _levels(o.ticker)
        entry = stop = tgt = setup = None
        risk_rupees = rr = None
        if lv:
            entry, stop, tgt, setup = lv
            if entry and stop and entry > stop:
                risk_rupees = round((entry - stop) * shares, 0)
                if tgt:
                    rr = round((tgt - entry) / (entry - stop), 2)
        allocs.append(Allocation(
            ticker=o.ticker, company=o.company, sector=o.sector,
            conviction=o.conviction, price=o.price, shares=shares, rupees=rupees,
            weight_pct=round(rupees / amount * 100, 1) if amount else 0.0,
            entry=entry, stoploss=stop, target=tgt,
            risk_rupees=risk_rupees, rr=rr, setup=setup,
            reasons=list(getattr(o, "reasons", []))[:3],
        ))

    leftover = round(amount - deployed, 0)
    total_risk = sum(a.risk_rupees for a in allocs if a.risk_rupees)
    if total_risk:
        notes.append(f"Total capital at risk if every stop triggers: "
                     f"₹{total_risk:,.0f} ({total_risk/amount*100:.1f}% of the amount).")
    if leftover > 0:
        notes.append(f"₹{leftover:,.0f} left as cash (whole-share rounding) — "
                     "hold it for the next pullback or to add to a winner.")
    return AllocationResult(amount, round(deployed, 0), leftover, len(allocs),
                            allocs, notes)
