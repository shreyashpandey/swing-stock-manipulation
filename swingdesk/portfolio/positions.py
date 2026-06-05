"""Position lifecycle: sizing, opening, marking-to-market, closing.

Position sizing uses fixed fractional risk:
    qty = floor(risk_amount / (entry - stoploss))
where risk_amount = capital * (risk_per_trade_pct / 100).

Mark-to-market walks every open position against the latest OHLC bar for
its ticker. SL/target hits are detected by comparing the bar's low/high
to the position's stop/target (long-side logic). When triggered we close
the position at the SL/target price.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from rich.console import Console

from swingdesk.config import (
    ACCOUNT_CAPITAL,
    EARNINGS_BLACKOUT_DAYS,
    MAX_OPEN_POSITIONS,
    RISK_PER_TRADE_PCT,
    TRAIL_ATR_MULT,
    TRAIL_BREAKEVEN_R,
)
from swingdesk.ingest.earnings import days_to_earnings
from swingdesk.storage import (
    get_position,
    insert_position,
    load_positions,
    load_prices,
    open_positions_for_ticker,
    update_position,
)

console = Console()


@dataclass
class SizingResult:
    qty: int
    risk_amount: float
    notional: float
    rejected: bool = False
    reason: str = ""


def size_position(entry: float, stoploss: float, *,
                  capital: float | None = None,
                  risk_pct: float | None = None) -> SizingResult:
    """Compute share quantity for fixed fractional risk."""
    cap = capital if capital is not None else ACCOUNT_CAPITAL
    rpct = risk_pct if risk_pct is not None else RISK_PER_TRADE_PCT
    risk_per_share = entry - stoploss
    if risk_per_share <= 0:
        return SizingResult(qty=0, risk_amount=0, notional=0,
                            rejected=True, reason="invalid SL (>= entry)")
    risk_amount = cap * (rpct / 100.0)
    qty = math.floor(risk_amount / risk_per_share)
    if qty <= 0:
        return SizingResult(qty=0, risk_amount=risk_amount, notional=0,
                            rejected=True, reason="risk too small for one share")
    notional = qty * entry
    if notional > cap:
        # Cap notional at available capital
        qty = math.floor(cap / entry)
        notional = qty * entry
        if qty <= 0:
            return SizingResult(qty=0, risk_amount=risk_amount, notional=0,
                                rejected=True, reason="entry > capital")
    return SizingResult(qty=qty, risk_amount=risk_amount, notional=notional)


def open_position(ticker: str, entry: float, stoploss: float, target: float,
                  *, setup: str | None = None, signal_id: int | None = None,
                  is_paper: bool = True, notes: str | None = None,
                  qty: int | None = None, entry_date: str | None = None,
                  skip_earnings_check: bool = False) -> dict:
    """Open a new position. If `qty` is None, size automatically."""
    # Earnings blackout — block new trades too close to a results day
    if not skip_earnings_check:
        dte = days_to_earnings(ticker)
        if dte is not None and dte <= EARNINGS_BLACKOUT_DAYS:
            return {"status": "rejected",
                    "reason": f"earnings in {dte}d (blackout = {EARNINGS_BLACKOUT_DAYS}d)"}

    # Refuse duplicate open positions on same ticker
    existing = open_positions_for_ticker(ticker)
    if not existing.empty:
        return {"status": "rejected", "reason": f"already have {len(existing)} open in {ticker}"}

    # Enforce max concurrent positions cap
    open_count = len(load_positions(status="open", is_paper=is_paper))
    if open_count >= MAX_OPEN_POSITIONS:
        return {"status": "rejected", "reason": f"max positions reached ({open_count})"}

    if qty is None:
        sized = size_position(entry, stoploss)
        if sized.rejected:
            return {"status": "rejected", "reason": sized.reason}
        qty = sized.qty

    pos_id = insert_position({
        "ticker": ticker,
        "setup": setup,
        "side": "long",
        "qty": qty,
        "entry_price": entry,
        "entry_date": entry_date or datetime.now().strftime("%Y-%m-%d"),
        "stoploss": stoploss,
        "target": target,
        "is_paper": int(is_paper),
        "signal_id": signal_id,
        "notes": notes,
    })
    # Seed last_price from real market data so the position shows live P&L
    # immediately. If this ticker has never been ingested, fetch it now —
    # paper trading is meaningless if positions show only the entry price.
    seed_df = load_prices(ticker)
    if seed_df.empty:
        try:
            from swingdesk.ingest import prices as _prices
            from swingdesk.storage import upsert_prices
            fresh = _prices.fetch_one(ticker, period="6mo")
            if fresh is not None and not fresh.empty:
                upsert_prices(ticker, fresh)
                seed_df = load_prices(ticker)
                console.print(f"  [dim]auto-fetched {len(seed_df)} bars for {ticker}[/dim]")
        except Exception as e:
            console.print(f"[yellow]could not auto-fetch prices for {ticker}: {e}[/yellow]")
    seed_price = entry
    if not seed_df.empty and pd.notna(seed_df.iloc[-1]["close"]):
        seed_price = float(seed_df.iloc[-1]["close"])
    update_position(pos_id, initial_stop=stoploss, high_water=entry,
                    last_price=seed_price)
    pos = get_position(pos_id)
    console.print(
        f"[green]opened[/green] {'paper' if is_paper else 'REAL'} #{pos_id} "
        f"{ticker} qty={qty} entry={entry} sl={stoploss} tgt={target}"
    )
    return {"status": "opened", "position": pos}


def close_position(pos_id: int, *, exit_price: float, exit_reason: str = "manual",
                   exit_date: str | None = None) -> dict:
    pos = get_position(pos_id)
    if not pos:
        return {"status": "not_found"}
    if pos["status"] != "open":
        return {"status": "already_closed", "position": pos}

    qty, entry, sl = pos["qty"], pos["entry_price"], pos["stoploss"]
    pnl = (exit_price - entry) * qty
    pnl_pct = (exit_price - entry) / entry * 100 if entry else 0
    risk_per_share = entry - sl if sl else None
    r_mult = (exit_price - entry) / risk_per_share if risk_per_share and risk_per_share > 0 else None

    update_position(
        pos_id,
        status="closed",
        exit_price=exit_price,
        exit_date=exit_date or datetime.now().strftime("%Y-%m-%d"),
        exit_reason=exit_reason,
        pnl=round(pnl, 2),
        pnl_pct=round(pnl_pct, 3),
        r_multiple=round(r_mult, 3) if r_mult is not None else None,
    )
    console.print(
        f"[yellow]closed[/yellow] #{pos_id} {pos['ticker']} {exit_reason}  "
        f"P&L=₹{round(pnl, 2)} ({round(pnl_pct, 2)}%)  R={round(r_mult or 0, 2)}"
    )
    return {"status": "closed", "position": get_position(pos_id)}


def _trailing_stop(p: dict, last_bar) -> tuple[float, bool]:
    """Compute the new stop based on trailing rules.

    Returns (new_stop, changed). Trailing rules:
      - When unrealized R reaches TRAIL_BREAKEVEN_R (default +1R), move stop
        to the entry price (lock in breakeven).
      - Beyond +TRAIL_BREAKEVEN_R, trail at high_water minus
        TRAIL_ATR_MULT * ATR_proxy where ATR_proxy ≈ (high - low) of recent bar.
      - The stop only moves UP, never down (we use max(old, new)).
    """
    entry = p["entry_price"]
    init_stop = p.get("initial_stop") or p["stoploss"]
    current_stop = p["stoploss"]
    high_water = p.get("high_water") or entry
    new_high = max(high_water, float(last_bar["high"]))

    risk = entry - init_stop
    if risk <= 0:
        return current_stop, False
    unrealized_r = (float(last_bar["close"]) - entry) / risk

    new_stop = current_stop
    if unrealized_r >= TRAIL_BREAKEVEN_R:
        # Move stop to at least breakeven
        new_stop = max(new_stop, entry)
    if unrealized_r >= 2 * TRAIL_BREAKEVEN_R:
        # Past +2R, also trail by ATR-proxy off the new high
        atr_proxy = float(last_bar["high"]) - float(last_bar["low"])
        trail_level = new_high - TRAIL_ATR_MULT * atr_proxy
        new_stop = max(new_stop, trail_level)

    return new_stop, new_stop > current_stop or new_high > high_water


def mark_to_market(*, auto_fetch: bool = True, refresh: bool = False) -> dict:
    """Update last_price on every open position. Auto-close if SL or target hit.
    Also applies trailing-stop policy. Returns counts.

    auto_fetch: when a position's ticker has no local price history, download it
                on demand so a paper trade on an un-watchlisted ticker still gets
                marked instead of being silently skipped.
    refresh:    re-download the latest prices for every open ticker before
                marking, even when local data already exists (pulls freshest close).
    """
    open_pos = load_positions(status="open")
    if open_pos.empty:
        return {"checked": 0, "closed": 0, "marked": 0, "trailed": 0,
                "fetched": 0, "skipped": []}

    closed = marked = trailed = fetched = 0
    skipped: list[str] = []
    for _, p in open_pos.iterrows():
        ticker = p["ticker"]
        df = load_prices(ticker)
        # Top up on-demand when prices are missing (or a refresh was requested),
        # so the position is marked against real data rather than skipped.
        if auto_fetch and (df.empty or refresh):
            from swingdesk.ingest import prices as _prices  # lazy: avoid import cycle
            _prices.ingest([ticker], period="6mo")
            df = load_prices(ticker)
            fetched += 1
        if df.empty:
            skipped.append(ticker)  # no data anywhere — surface it, don't hide it
            continue
        last_bar = df.iloc[-1]
        last_date = (last_bar.name.strftime("%Y-%m-%d")
                     if hasattr(last_bar.name, "strftime") else str(last_bar.name))

        # Was the stop already trailed above its initial level?
        is_trailed = p["stoploss"] > (p.get("initial_stop") or p["stoploss"])
        # Pessimistic intraday ordering: SL first, then target.
        # We check against the CURRENT (possibly trailed) stop.
        if last_bar["open"] <= p["stoploss"]:
            close_position(int(p["id"]),
                           exit_price=float(last_bar["open"]),
                           exit_reason="trail" if is_trailed else "stoploss_gap",
                           exit_date=last_date)
            closed += 1
            continue
        if last_bar["low"] <= p["stoploss"]:
            close_position(int(p["id"]),
                           exit_price=float(p["stoploss"]),
                           exit_reason="trail" if is_trailed else "stoploss",
                           exit_date=last_date)
            closed += 1
            continue
        if last_bar["high"] >= p["target"]:
            close_position(int(p["id"]),
                           exit_price=float(p["target"]),
                           exit_reason="target",
                           exit_date=last_date)
            closed += 1
            continue

        # Position still open — update last_price, high_water, and trailing stop
        new_stop, changed = _trailing_stop(p.to_dict(), last_bar)
        new_high = max(p.get("high_water") or p["entry_price"], float(last_bar["high"]))
        updates = {
            "last_price": float(last_bar["close"]),
            "last_marked": last_date,
            "high_water": new_high,
        }
        if new_stop > p["stoploss"]:
            updates["stoploss"] = new_stop
            trailed += 1
            console.print(
                f"  [cyan]trail[/cyan] #{int(p['id'])} {p['ticker']}: "
                f"stop {p['stoploss']:.2f} → {new_stop:.2f}"
            )
        update_position(int(p["id"]), **updates)
        marked += 1

    return {"checked": len(open_pos), "closed": closed, "marked": marked,
            "trailed": trailed, "fetched": fetched, "skipped": skipped}


# --- paper-trading integration --------------------------------------------------

def auto_paper_trade(signals: list[dict], *, min_composite: float = 60.0) -> dict:
    """Open paper positions for any signal that:
       1. Has composite score >= threshold
       2. Doesn't already have an open position
       3. Hasn't pushed us past MAX_OPEN_POSITIONS

    Returns counts. Called from the daily run.
    """
    opened = skipped = 0
    reasons: list[str] = []
    for sig in signals:
        composite = float(sig.get("composite_score") or sig.get("score") or 0)
        if composite < min_composite:
            skipped += 1
            reasons.append(f"{sig['ticker']}: score {composite} < {min_composite}")
            continue
        result = open_position(
            ticker=sig["ticker"],
            entry=float(sig["entry"]),
            stoploss=float(sig["stoploss"]),
            target=float(sig["target"]),
            setup=sig.get("setup"),
            signal_id=sig.get("id"),
            is_paper=True,
            notes=sig.get("notes"),
        )
        if result["status"] == "opened":
            opened += 1
        else:
            skipped += 1
            reasons.append(f"{sig['ticker']}: {result.get('reason', result['status'])}")

    return {"opened": opened, "skipped": skipped, "reasons": reasons}
