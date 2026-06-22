"""Paper autotrader — a hands-off *simulated* trader you can run forward.

This is the loop the execution + TCA work was building toward, but with NO real
money and NO broker: each `step()` evaluates one bar/day, exits what should
exit, then opens new paper positions from the signal scan — gated by real
risk limits and filled at an **execution-cost-adjusted** price (so the paper P&L
reflects what you'd actually get, not a frictionless fantasy).

What it adds on top of the existing position primitives (size/open/close/
mark-to-market in positions.py):

  • Execution-aware entries — the fill price is bumped by the estimated cost of
    actually working the order (square-root impact + spread from execution.py),
    using the chosen algo. Buys pay up; the slippage is recorded on the trade.
  • Portfolio-heat cap — total open risk (Σ (entry−stop)×qty) is capped at a %
    of capital, so the book can't quietly stack into one big correlated bet.
  • Free-cash check — can't buy more than the uninvested cash on hand.
  • Kill-switch — when equity drawdown from its peak crosses a threshold, NEW
    entries halt (optionally flatten everything). Auto-clears if equity recovers.
  • Run log — every step is persisted (equity, drawdown, heat, what opened/closed)
    so you get an equity curve and an audit trail.

Run it daily (a button in the app, the CLI, or the /loop skill). Nothing here
places a live order — it only writes paper positions to the local DB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from swingdesk.analyze import execution as exec_mod
from swingdesk.config import (
    ACCOUNT_CAPITAL,
    KILL_SWITCH_DD_PCT,
    MAX_OPEN_POSITIONS,
    MAX_PORTFOLIO_HEAT_PCT,
    RISK_PER_TRADE_PCT,
)
from swingdesk.portfolio import positions as portfolio
from swingdesk.storage import (
    load_autotrader_log,
    load_positions,
    log_autotrader_step,
    open_positions_for_ticker,
)


@dataclass
class AutoTraderConfig:
    capital: float = ACCOUNT_CAPITAL
    risk_pct: float = RISK_PER_TRADE_PCT
    max_positions: int = MAX_OPEN_POSITIONS
    max_portfolio_heat_pct: float = MAX_PORTFOLIO_HEAT_PCT
    kill_switch_dd_pct: float = KILL_SWITCH_DD_PCT
    min_score: float = 60.0            # signal-quality gate (setup `score`)
    algo: str = "vwap"                 # execution algo used to price entry fills
    bucket_minutes: int = 30
    require_uptrend: bool = True        # gate counter-trend signals (matches live scan)
    flatten_on_kill: bool = False       # close everything when the kill-switch trips
    force_halt: bool = False            # manual override — block new entries


@dataclass
class StepReport:
    asof: str
    equity: float
    free_cash: float
    realized_pnl: float
    unrealized_pnl: float
    peak_equity: float
    drawdown_pct: float
    portfolio_heat_pct: float
    halted: bool
    n_open: int
    exits: list[dict] = field(default_factory=list)
    entries: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Account state
# --------------------------------------------------------------------------- #
def _open_risk(open_pos) -> float:
    """Total ₹ at risk across open positions = Σ max(entry − current stop, 0)×qty.
    A trailed stop above entry contributes 0 (the trade can't lose from here)."""
    total = 0.0
    for _, p in open_pos.iterrows():
        stop = p["stoploss"] if p["stoploss"] is not None else p["entry_price"]
        total += max(float(p["entry_price"]) - float(stop), 0.0) * float(p["qty"])
    return total


def account_state(cfg: AutoTraderConfig | None = None) -> dict:
    """Mark-to-market snapshot of the paper book: realized/unrealized P&L,
    equity, free cash, peak/drawdown, open risk and heat."""
    cfg = cfg or AutoTraderConfig()
    open_pos = load_positions(status="open", is_paper=True)
    closed = load_positions(status="closed", is_paper=True)

    realized = float(closed["pnl"].fillna(0).sum()) if not closed.empty else 0.0
    invested = unrealized = 0.0
    for _, p in open_pos.iterrows():
        entry, qty = float(p["entry_price"]), float(p["qty"])
        last = float(p["last_price"]) if p.get("last_price") is not None else entry
        invested += entry * qty
        unrealized += (last - entry) * qty

    equity = cfg.capital + realized + unrealized
    free_cash = cfg.capital + realized - invested
    open_risk = _open_risk(open_pos)
    heat_pct = open_risk / cfg.capital * 100 if cfg.capital else 0.0

    # Peak from the run log (plus the current equity), for the kill-switch.
    log = load_autotrader_log(limit=10_000)
    hist_peak = float(log["equity"].max()) if not log.empty and log["equity"].notna().any() else cfg.capital
    peak = max(hist_peak, equity, cfg.capital)
    dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0.0
    halted = dd_pct >= cfg.kill_switch_dd_pct or cfg.force_halt

    return {
        "capital": cfg.capital, "realized_pnl": round(realized, 2),
        "unrealized_pnl": round(unrealized, 2), "invested": round(invested, 2),
        "equity": round(equity, 2), "free_cash": round(free_cash, 2),
        "peak_equity": round(peak, 2), "drawdown_pct": round(dd_pct, 2),
        "open_risk": round(open_risk, 2), "portfolio_heat_pct": round(heat_pct, 2),
        "n_open": int(len(open_pos)), "halted": bool(halted),
    }


# --------------------------------------------------------------------------- #
# Entry with execution-cost-aware fill + risk gates
# --------------------------------------------------------------------------- #
def _try_enter(sig: dict, cfg: AutoTraderConfig, state: dict) -> dict:
    ticker = sig["ticker"]
    entry, stop, target = float(sig["entry"]), float(sig["stoploss"]), float(sig["target"])
    score = float(sig.get("composite_score") or sig.get("score") or 0)

    def reject(reason: str) -> dict:
        return {"status": "rejected", "ticker": ticker, "reason": reason}

    if score < cfg.min_score:
        return reject(f"score {score:.0f} < {cfg.min_score:.0f}")
    if not open_positions_for_ticker(ticker).empty:
        return reject("already open")

    sized = portfolio.size_position(entry, stop, capital=cfg.capital, risk_pct=cfg.risk_pct)
    if sized.rejected:
        return reject(sized.reason)
    qty = sized.qty

    # Execution-cost-aware fill: bump the buy price by the estimated cost of
    # actually working this order with the chosen algo. Risk/cash gates below
    # are measured at this REAL fill, so the realized book never breaches them.
    plan = exec_mod.execution_plan(ticker, "buy", qty=qty, algo=cfg.algo,
                                   bucket_minutes=cfg.bucket_minutes)
    slip_bps = plan.est_cost_bps if plan is not None else exec_mod.HALF_SPREAD_BPS
    fill = round(entry * (1 + slip_bps / 1e4), 2)

    # Free-cash check — shrink to fit if needed.
    if qty * fill > state["free_cash"]:
        qty = int(state["free_cash"] // fill)
        if qty <= 0:
            return reject(f"no free cash (₹{state['free_cash']:,.0f})")

    # Portfolio-heat cap (book-level open risk, measured at the fill).
    new_risk = max(fill - stop, 0.0) * qty
    heat_budget = cfg.capital * cfg.max_portfolio_heat_pct / 100.0
    if state["open_risk"] + new_risk > heat_budget:
        return reject(f"portfolio heat would exceed {cfg.max_portfolio_heat_pct:.1f}% "
                      f"(open risk ₹{state['open_risk']:,.0f} + ₹{new_risk:,.0f})")

    note = (f"auto · {sig.get('setup', '?')} · {cfg.algo.upper()} fill "
            f"+{slip_bps:.0f}bps (signal entry ₹{entry:,.2f})")
    res = portfolio.open_position(
        ticker, entry=fill, stoploss=stop, target=target, setup=sig.get("setup"),
        signal_id=sig.get("id"), is_paper=True, notes=note, qty=qty)
    if res["status"] != "opened":
        return reject(res.get("reason", res["status"]))
    return {"status": "opened", "ticker": ticker, "qty": qty, "fill": fill,
            "signal_entry": entry, "slip_bps": round(float(slip_bps), 1),
            "risk": round(new_risk, 0), "setup": sig.get("setup")}


def flatten_all(*, reason: str = "manual_flatten") -> int:
    """Close every open paper position at its latest marked price. Returns count."""
    portfolio.mark_to_market(auto_fetch=True, refresh=False)
    open_pos = load_positions(status="open", is_paper=True)
    n = 0
    for _, p in open_pos.iterrows():
        last = float(p["last_price"]) if p.get("last_price") is not None else float(p["entry_price"])
        portfolio.close_position(int(p["id"]), exit_price=last, exit_reason=reason)
        n += 1
    return n


# --------------------------------------------------------------------------- #
# The step: exits → risk check → entries → log
# --------------------------------------------------------------------------- #
def step(cfg: AutoTraderConfig | None = None, *, signals: list[dict] | None = None,
         universe: list[str] | None = None, refresh: bool = False,
         persist_log: bool = True) -> StepReport:
    """Evaluate one bar/day: process exits, then open new paper trades from
    `signals` (or a fresh scan of `universe`) under the risk limits + kill-switch."""
    cfg = cfg or AutoTraderConfig()
    notes: list[str] = []

    # --- 1. exits (reuse mark-to-market: stops/targets/gaps/trailing)
    before = set(load_positions(status="open", is_paper=True)["id"].tolist())
    portfolio.mark_to_market(auto_fetch=True, refresh=refresh)
    exits = []
    closed = load_positions(status="closed", is_paper=True)
    for _, p in closed.iterrows():
        if int(p["id"]) in before:
            exits.append({
                "ticker": p["ticker"], "exit_reason": p["exit_reason"],
                "exit_price": float(p["exit_price"]) if p["exit_price"] is not None else None,
                "pnl": round(float(p["pnl"]), 2) if p["pnl"] is not None else None,
                "r_multiple": round(float(p["r_multiple"]), 2) if p["r_multiple"] is not None else None,
            })

    # --- 2. state + kill-switch
    state = account_state(cfg)
    entries, rejected = [], []
    if state["halted"]:
        why = ("manual force-halt" if cfg.force_halt
               else f"drawdown {state['drawdown_pct']:.1f}% ≥ {cfg.kill_switch_dd_pct:.0f}%")
        notes.append(f"⛔ kill-switch active ({why}) — no new entries.")
        if cfg.flatten_on_kill and state["n_open"] > 0:
            n = flatten_all(reason="kill_switch")
            notes.append(f"flattened {n} open position(s) on kill-switch.")
            state = account_state(cfg)
    else:
        # --- 3. entries
        if signals is None:
            from swingdesk.analyze.setups import scan_all
            from swingdesk.storage import combined_universe, get_watchlist
            uni = universe or combined_universe(include_smallcaps=False) or get_watchlist()
            signals = scan_all(uni, persist=False, require_uptrend=cfg.require_uptrend)
        ranked = sorted(signals, key=lambda s: float(s.get("composite_score")
                                                       or s.get("score") or 0), reverse=True)
        for sig in ranked:
            if state["n_open"] >= cfg.max_positions:
                rejected.append({"ticker": sig.get("ticker"), "reason": "max positions reached"})
                continue
            res = _try_enter(sig, cfg, state)
            if res["status"] == "opened":
                entries.append(res)
                # Incrementally update gates so the next entry sees this trade.
                state["n_open"] += 1
                state["open_risk"] += res["risk"]
                state["free_cash"] -= res["qty"] * res["fill"]
            else:
                rejected.append({"ticker": res["ticker"], "reason": res["reason"]})

    # --- 4. recompute state for an accurate log row, then persist
    final = account_state(cfg)
    asof = datetime.now().strftime("%Y-%m-%d")
    if not notes:
        notes.append(f"{len(entries)} opened · {len(exits)} closed · "
                     f"{final['n_open']} open · heat {final['portfolio_heat_pct']:.1f}%.")
    if persist_log:
        log_autotrader_step({
            "asof": asof, "equity": final["equity"], "free_cash": final["free_cash"],
            "realized_pnl": final["realized_pnl"], "unrealized_pnl": final["unrealized_pnl"],
            "peak_equity": final["peak_equity"], "drawdown_pct": final["drawdown_pct"],
            "portfolio_heat_pct": final["portfolio_heat_pct"], "n_open": final["n_open"],
            "opened": len(entries), "closed": len(exits),
            "halted": int(final["halted"]), "note": " ".join(notes)[:500],
        })

    return StepReport(
        asof=asof, equity=final["equity"], free_cash=final["free_cash"],
        realized_pnl=final["realized_pnl"], unrealized_pnl=final["unrealized_pnl"],
        peak_equity=final["peak_equity"], drawdown_pct=final["drawdown_pct"],
        portfolio_heat_pct=final["portfolio_heat_pct"], halted=final["halted"],
        n_open=final["n_open"], exits=exits, entries=entries,
        rejected=rejected, notes=notes,
    )


def run_log(limit: int = 200):
    """The persisted run log (oldest→newest) for the equity curve & audit."""
    df = load_autotrader_log(limit=limit)
    return df.iloc[::-1].reset_index(drop=True) if not df.empty else df
