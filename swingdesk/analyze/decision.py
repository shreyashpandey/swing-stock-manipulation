"""Unified decision engine — one verdict per ticker, from every lens we have.

This is the single place that fuses the app's separate analytics into one
answer to "should I put money here, and is *today* the day?":

  * **Rank**  — cross-sectional factor quintile/composite (factors.py)
  * **Trend** — signal scoreboard tilt + volume-confirmed trend quality (technicals.py)
  * **Odds**  — target-vs-stop expectancy and the expected range (expected_range.py)
  * **Quality / sentiment** — fundamentals verdict + recent news (summary.py, storage)
  * **Macro** — risk-on/off regime + sector rotation (spillover.py, sectors.py)
  * **Vetoes** — manipulation risk and illiquidity can only *lower* the verdict
    (manipulation.py, liquidity.py)

It produces a `Decision` with an `action` (STRONG_BUY…AVOID), a 0-100
`conviction`, and a SEPARATE `timing` read (enter today / wait for pullback /
wait for confirmation / don't enter) — because a great business can still be a
bad entry *today*. Given a planned ₹ amount it also sizes the position to your
risk (risk.py) and, for names you already hold, advises HOLD / ADD / TRIM / EXIT.

The shared turnover / liquidity numbers come from :mod:`market_metrics`, so this
engine, the Manipulation tab and the Screener all read the same figures.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swingdesk.analyze import expected_range as er_mod
from swingdesk.analyze import factors as factors_mod
from swingdesk.analyze import liquidity as liq_mod
from swingdesk.analyze import manipulation as manip_mod
from swingdesk.analyze import market_metrics
from swingdesk.analyze import risk as risk_mod
from swingdesk.analyze import sectors as sectors_mod
from swingdesk.analyze import spillover as spillover_mod
from swingdesk.analyze import summary as summary_mod
from swingdesk.analyze.setups import scan_ticker
from swingdesk.analyze.technicals import add_indicators, signal_scoreboard, trend_quality
from swingdesk.config import ACCOUNT_CAPITAL, RISK_PER_TRADE_PCT
from swingdesk.storage import (
    get_fundamentals,
    load_holdings,
    load_prices,
    recent_sentiment_for_ticker,
)

# --- Conviction blend weights (sum ≈ 1.0; renormalised over available signals) -
W_TILT = 0.24      # technical scoreboard tilt
W_FACTOR = 0.20    # cross-sectional factor rank
W_TREND = 0.16     # volume-confirmed trend quality
W_EXPECT = 0.16    # target-vs-stop expectancy
W_QUALITY = 0.10   # fundamental quality score
W_SENT = 0.06      # recent news sentiment
W_MACRO = 0.08     # regime + sector rotation
W_ML = 0.12        # ML P(up) — only when use_ml and a prediction is supplied

# Action bands on conviction.
TIER_STRONG = 72
TIER_BUY = 58
TIER_ACCUM = 45
TIER_WAIT = 30

# Timing thresholds.
RSI_HOT = 75.0          # overbought → wait for a pullback
EMA_EXT = 0.08          # >8% above the 20-EMA → extended
SPIKE_TODAY = 5.0       # today's ₹ value ≥5× its usual → don't chase
RETURN_Z_HOT = 4.0      # a 4σ daily move today → don't chase
HIGH_BETA = 1.2         # "high beta" for the risk-off timing gate

# Low→high so we can clamp an action to a ceiling by index.
ACTION_ORDER = ["AVOID", "WAIT", "ACCUMULATE", "BUY", "STRONG_BUY"]
_SUMMARY_MAP = {"STRONG_BUY": "STRONG_BUY", "BUY": "BUY", "WAIT": "WAIT", "AVOID": "AVOID"}


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


@dataclass
class PlanSizing:
    planned_amount: float
    shares_by_amount: int          # whole shares the ₹ amount buys at entry
    risk_sized_shares: int | None  # shares that keep a stop-out ≤ risk_pct of capital
    suggested_shares: int          # the smaller of the two
    suggested_value: float
    pct_of_capital: float
    risk_amount: float             # ₹ lost if the stop is hit, at suggested size
    risk_pct_of_capital: float
    over_risk: bool                # the ₹ amount risks more than risk_pct at the stop
    over_concentration: bool       # adds a concentration flag the book didn't have
    concentration_flag: str | None
    advice: str


@dataclass
class HoldingAdvice:
    qty: float
    avg_price: float
    last_price: float
    unrealized_pct: float
    action: str                    # HOLD | ADD | TRIM | EXIT
    reason: str


@dataclass
class Decision:
    ticker: str
    spot: float
    action: str
    conviction: float
    timing: str
    timing_reason: str
    # shared market-structure numbers (single source of truth)
    today_turnover_pct: float | None
    avg_turnover_pct: float | None
    today_vs_avg_value_mult: float | None
    adv_value_cr: float | None
    # contributing sub-signals
    factor_quintile: int | None
    factor_composite: float | None
    tech_tilt: str | None
    tech_score: int | None
    trend_verdict: str | None
    expectancy_r: float | None
    p_target_first: float | None
    quality_score: float | None
    sentiment_net: int
    regime_label: str
    sector_bias: str | None
    manip_tier: str
    manip_score: float | None
    liq_tier: str
    liq_score: float | None
    ml_prob_up: float | None
    # gates / narrative
    gates_passed: bool
    veto_reason: str | None
    pros: list[str]
    cons: list[str]
    # trade levels
    entry: float | None
    stoploss: float | None
    target: float | None
    rr: float | None
    level_source: str | None
    # sizing / holding
    plan: PlanSizing | None
    holding: HoldingAdvice | None
    data_gaps: list[str] = field(default_factory=list)


def _insufficient(ticker: str, df) -> Decision:
    n = 0 if df is None else len(df)
    return Decision(
        ticker=ticker, spot=float(df["close"].iloc[-1]) if n else 0.0,
        action="WAIT", conviction=0.0, timing="WAIT_CONFIRMATION",
        timing_reason="Not enough price history to assess.",
        today_turnover_pct=None, avg_turnover_pct=None,
        today_vs_avg_value_mult=None, adv_value_cr=None,
        factor_quintile=None, factor_composite=None, tech_tilt=None, tech_score=None,
        trend_verdict=None, expectancy_r=None, p_target_first=None, quality_score=None,
        sentiment_net=0, regime_label="n/a", sector_bias=None,
        manip_tier="n/a", manip_score=None, liq_tier="n/a", liq_score=None,
        ml_prob_up=None, gates_passed=False, veto_reason="insufficient history",
        pros=[], cons=[], entry=None, stoploss=None, target=None, rr=None,
        level_source=None, plan=None, holding=None,
        data_gaps=[f"need ≥ 60 daily bars, have {n}"],
    )


def _beta_nifty(ticker: str) -> float | None:
    """NIFTY beta from the spillover regression — only needed for the risk-off
    timing gate, so callers compute it lazily."""
    try:
        sens = spillover_mod.stock_sensitivities(ticker)
        if sens is None or sens.empty:
            return None
        row = sens[sens["driver"].astype(str).str.contains("NIFTY", case=False)]
        return float(row["beta"].iloc[0]) if not row.empty else None
    except Exception:
        return None


def decide(ticker: str, *, planned_amount: float | None = None,
           capital: float = ACCOUNT_CAPITAL, risk_pct: float = RISK_PER_TRADE_PCT,
           held: dict | None = None, regime_obj=None, sector_bias_map: dict | None = None,
           factor_row: dict | None = None, run_montecarlo: bool = True,
           mc_sims: int = 2000, use_ml: bool = False, ml_prob: float | None = None,
           fund: dict | None = None) -> Decision | None:
    """Fuse every lens into one verdict for ``ticker``.

    Cross-sectional / global inputs (``factor_row``, ``regime_obj``,
    ``sector_bias_map``, ``ml_prob``) are passed in by :func:`decide_universe`
    so they're computed once for the whole scan; a standalone call computes the
    cheap ones itself and tolerates the absence of the rest.
    """
    df_raw = load_prices(ticker)
    if df_raw is None or df_raw.empty or len(df_raw) < 60:
        return _insufficient(ticker, df_raw)
    df = add_indicators(df_raw)
    last = df.iloc[-1]
    spot = float(last["close"])
    if fund is None:
        fund = get_fundamentals(ticker) or {}

    data_gaps: list[str] = []
    pros: list[str] = []
    cons: list[str] = []

    # --- shared market-structure numbers (single source of truth) ---
    mm = market_metrics.compute(df_raw, fund, allow_derive=True, ticker=ticker)

    # --- liquidity (veto candidate) ---
    liq = liq_mod.liquidity_profile(ticker, fund)
    liq_tier = liq.tier if liq else "n/a"
    liq_score = liq.score if liq else None

    # --- manipulation (veto candidate) ---
    manip = manip_mod.scorecard(ticker, df_raw, fund)
    manip_tier = manip.get("tier", "n/a")
    manip_score = manip.get("risk_score")
    abn = (manip.get("components") or {}).get("abnormal_return") or {}
    return_z = abn.get("return_z")
    if manip.get("data_gaps"):
        # keep the market-cap gap (it affects the turnover read) but not the noise
        data_gaps += [g for g in manip["data_gaps"] if "market_cap" in g]

    # --- technical tilt + trend quality ---
    sb = signal_scoreboard(df)
    tq = trend_quality(df)

    # --- fundamentals summary (verdict base + quality + sentiment counts) ---
    summ = summary_mod.summarize(ticker)
    quality_score = summ.quality_score if summ else fund.get("quality_score")

    # --- factor rank ---
    if factor_row is None:
        factor_row = _single_factor_row(ticker)
    quintile = factor_row.get("quintile") if factor_row else None
    composite = factor_row.get("composite") if factor_row else None

    # --- recent news sentiment ---
    try:
        sent = recent_sentiment_for_ticker(ticker, days=7)
        bull = int((sent["sentiment"] == "bullish").sum()) if not sent.empty else 0
        bear = int((sent["sentiment"] == "bearish").sum()) if not sent.empty else 0
    except Exception:
        bull = bear = 0
    sentiment_net = bull - bear

    # --- macro: regime + sector bias ---
    regime_obj = regime_obj or spillover_mod.regime()
    regime_label = regime_obj.label if regime_obj else "n/a"
    regime_score = regime_obj.score if regime_obj else 0
    sector = fund.get("sector")
    sector_bias = sector_bias_map.get(sector) if (sector_bias_map and sector) else None

    # --- trade levels: prefer a fresh setup, else derive from expected range ---
    sigs = scan_ticker(ticker)
    er = _safe(er_mod.expected_range, ticker)
    entry = stoploss = target = rr = None
    level_source = None
    if sigs:
        top = sigs[0]
        entry, stoploss, target, rr = top.get("entry"), top.get("stoploss"), top.get("target"), top.get("rr")
        level_source = top.get("setup")
    elif er is not None and er.expected_move_pct:
        entry = spot
        stoploss = round(spot * (1 - er.expected_move_pct / 100), 2)
        target = round(er.high_68, 2)
        rr = round((target - entry) / (entry - stoploss), 2) if entry > stoploss else None
        level_source = "expected_range"

    # --- odds: target-vs-stop expectancy, tied to the actual levels when known ---
    expectancy_r = p_target_first = None
    if run_montecarlo:
        tpct, spct = 8.0, 4.0
        if entry and target and stoploss and entry > 0 and target > entry > stoploss:
            tpct = (target - entry) / entry * 100
            spct = (entry - stoploss) / entry * 100
        tvs = _safe(er_mod.target_vs_stop, ticker, target_pct=round(tpct, 2),
                    stop_pct=round(spct, 2), n_sims=mc_sims)
        if tvs is not None:
            expectancy_r = tvs.expectancy_r
            p_target_first = tvs.p_target_first

    # ---------- conviction blend (renormalised over available signals) ----------
    parts: list[tuple[float, float]] = []
    if sb is not None:
        parts.append((W_TILT, (sb["score"] + 100) / 2))
    if quintile in (1, 2, 3, 4, 5):
        parts.append((W_FACTOR, {1: 90, 2: 70, 3: 50, 4: 30, 5: 10}[int(quintile)]))
    if tq is not None:
        parts.append((W_TREND, tq["score"] if tq["verdict"] != "no_uptrend" else 30))
    if expectancy_r is not None:
        parts.append((W_EXPECT, _clip(50 + expectancy_r * 25)))
    if quality_score is not None:
        parts.append((W_QUALITY, float(quality_score)))
    parts.append((W_SENT, _clip(50 + sentiment_net * 10)))
    macro_val = (regime_score + 100) / 2 + {"Bullish": 15, "Bearish": -15}.get(sector_bias or "", 0)
    parts.append((W_MACRO, _clip(macro_val)))
    if use_ml and ml_prob is not None:
        parts.append((W_ML, _clip(ml_prob * 100)))

    blend = sum(w * v for w, v in parts) / sum(w for w, _ in parts) if parts else 0.0

    # ---------- vetoes / gates (can only lower the action and conviction) -------
    penalty = 1.0
    ceiling = "STRONG_BUY"
    veto_reason = None

    def _cap(action_ceiling: str) -> str:
        return ACTION_ORDER[min(ACTION_ORDER.index(ceiling), ACTION_ORDER.index(action_ceiling))]

    if liq_tier == "untradeable":
        ceiling = _cap("AVOID")
        veto_reason = "untradeable — too illiquid to enter or exit"
        cons.append("⚠ untradeable liquidity — slippage/exit risk, avoid")
    elif liq_tier == "illiquid":
        ceiling = _cap("WAIT")
        cons.append("⚠ illiquid — size tiny or skip")

    if manip_tier == "High":
        penalty *= 0.5
        ceiling = _cap("WAIT")
        if veto_reason is None:
            veto_reason = "high manipulation-risk footprint"
        cons.append(f"⚠ high unusual-activity risk (score {manip_score}) — investigate before trading")
    elif manip_tier == "Elevated":
        penalty *= 0.85
        cons.append(f"some unusual activity (manip score {manip_score}) — check the flagged metrics")

    if tq is not None and tq["verdict"] == "false":
        ceiling = _cap("WAIT")
        cons.append("trend looks like distribution (false uptrend) — don't add")

    conviction = round(_clip(blend * penalty), 0)
    if liq_tier == "untradeable":
        conviction = min(conviction, 20)
    gates_passed = veto_reason is None

    # ---------- raw action from conviction, then clamp to the ceiling ----------
    if conviction >= TIER_STRONG and gates_passed and sb is not None and sb["tilt"] in ("BUY", "STRONG BUY"):
        raw = "STRONG_BUY"
    elif conviction >= TIER_BUY:
        raw = "BUY"
    elif conviction >= TIER_ACCUM and tq is not None and tq["verdict"] in ("real", "weak"):
        raw = "ACCUMULATE"
    elif conviction >= TIER_WAIT:
        raw = "WAIT"
    else:
        raw = "AVOID"
    action = ACTION_ORDER[min(ACTION_ORDER.index(raw), ACTION_ORDER.index(ceiling))]

    # Cross-check against the rule-based summary verdict; if we're ≥2 notches more
    # bullish, step down one and say why (keeps the two surfaces from clashing).
    if summ is not None and _SUMMARY_MAP.get(summ.verdict) in ACTION_ORDER:
        si = ACTION_ORDER.index(_SUMMARY_MAP[summ.verdict])
        di = ACTION_ORDER.index(action)
        if di - si >= 2:
            action = ACTION_ORDER[di - 1]
            cons.append(f"summary view is more cautious: {summ.one_liner}")

    # ---------- pros (positive case) ----------
    if sb is not None and sb["score"] >= 20:
        pros.append(f"technical tilt {sb['tilt']} (score {sb['score']:+d})")
    if quintile in (1, 2):
        pros.append(f"top-quintile factor rank (Q{int(quintile)}, composite {composite})")
    if tq is not None and tq["verdict"] == "real":
        pros.append("volume-confirmed uptrend")
    if expectancy_r is not None and expectancy_r > 0.1:
        pros.append(f"favourable odds: +{expectancy_r:.2f}R expectancy, P(target first) {p_target_first:.0%}")
    if quality_score is not None and quality_score >= 65:
        pros.append(f"quality {quality_score:.0f}/100")
    if sentiment_net >= 2:
        pros.append(f"news flow positive ({bull} bullish vs {bear} bearish, 7d)")
    if sector_bias == "Bullish":
        pros.append(f"{sector} sector in a bullish rotation")
    if not pros:
        pros.append("no standout positive — middling on most lenses")

    # ---------- today-timing read (independent of the action) ----------
    timing, timing_reason = _timing(
        action=action, last=last, spot=spot, er=er, mm=mm,
        liq_tier=liq_tier, manip_tier=manip_tier, return_z=return_z,
        regime_label=regime_label, sb=sb, has_setup=bool(sigs),
        ticker=ticker,
    )

    # ---------- plan sizing ("invest ₹X today?") ----------
    plan = None
    if planned_amount and entry and stoploss and entry > stoploss:
        plan = _size_plan(ticker, planned_amount, entry, stoploss, capital, risk_pct)

    # ---------- existing-holding advice ----------
    holding = None
    if held:
        holding = _holding_advice(held, spot, last, tq, sb, manip_tier, timing, conviction)

    return Decision(
        ticker=ticker, spot=round(spot, 2), action=action, conviction=conviction,
        timing=timing, timing_reason=timing_reason,
        today_turnover_pct=round(mm.today_turnover_pct, 3) if mm and mm.today_turnover_pct is not None else None,
        avg_turnover_pct=round(mm.avg_turnover_pct, 3) if mm and mm.avg_turnover_pct is not None else None,
        today_vs_avg_value_mult=round(mm.today_vs_avg_value_mult, 2) if mm and mm.today_vs_avg_value_mult is not None else None,
        adv_value_cr=round(mm.adv_value_cr, 2) if mm else None,
        factor_quintile=int(quintile) if quintile in (1, 2, 3, 4, 5) else None,
        factor_composite=composite, tech_tilt=sb["tilt"] if sb else None,
        tech_score=sb["score"] if sb else None,
        trend_verdict=tq["verdict"] if tq else None,
        expectancy_r=round(expectancy_r, 2) if expectancy_r is not None else None,
        p_target_first=round(p_target_first, 3) if p_target_first is not None else None,
        quality_score=quality_score, sentiment_net=sentiment_net,
        regime_label=regime_label, sector_bias=sector_bias,
        manip_tier=manip_tier, manip_score=manip_score,
        liq_tier=liq_tier, liq_score=liq_score,
        ml_prob_up=round(ml_prob, 3) if (use_ml and ml_prob is not None) else None,
        gates_passed=gates_passed, veto_reason=veto_reason, pros=pros, cons=cons,
        entry=round(entry, 2) if entry else None,
        stoploss=round(stoploss, 2) if stoploss else None,
        target=round(target, 2) if target else None,
        rr=rr, level_source=level_source,
        plan=plan, holding=holding, data_gaps=data_gaps,
    )


def _timing(*, action, last, spot, er, mm, liq_tier, manip_tier, return_z,
            regime_label, sb, has_setup, ticker) -> tuple[str, str]:
    """The 'is today the day?' read — deliberately separate from the action so a
    STRONG_BUY can still say 'wait for a pullback today'."""
    if action == "AVOID":
        return "DONT_ENTER_TODAY", "Engine says avoid — not an entry."

    rsi = float(last["rsi14"]) if pd.notna(last.get("rsi14")) else None
    ema20 = float(last["ema20"]) if pd.notna(last.get("ema20")) else None
    spike = mm.today_value_spike_mult if mm else None
    high_68 = er.high_68 if er is not None else None

    # 1) Don't chase — illiquid, an operator-footprint tape today, or risk-off + high beta.
    if liq_tier in ("illiquid", "untradeable"):
        return "DONT_ENTER_TODAY", "Illiquid — entry/exit slippage risk; not worth chasing."
    if manip_tier == "High" or (spike is not None and spike >= SPIKE_TODAY) or \
            (return_z is not None and abs(return_z) >= RETURN_Z_HOT):
        return ("DONT_ENTER_TODAY",
                "Today's tape looks like a spike / operator footprint — don't chase, let it settle.")
    if regime_label == "risk-off":
        beta = _beta_nifty(ticker)
        if beta is not None and beta >= HIGH_BETA:
            return ("DONT_ENTER_TODAY",
                    f"Market is risk-off and this is high-beta (β≈{beta:.1f}) — wait for the index to steady.")

    # 2) Overextended — better R on a pullback.
    if rsi is not None and rsi > RSI_HOT:
        return "WAIT_PULLBACK", f"Overbought (RSI {rsi:.0f}) — better entry on a pullback."
    if high_68 is not None and spot >= high_68:
        return "WAIT_PULLBACK", "At the top of its expected range — wait for a dip toward fair value."
    if ema20 and (spot - ema20) / ema20 >= EMA_EXT:
        return "WAIT_PULLBACK", f"{(spot-ema20)/ema20*100:.0f}% above the 20-EMA — extended; buy nearer ₹{ema20:.0f}."

    # 3) Enter today — fresh trigger, healthy momentum, not extended, not risk-off.
    if has_setup and sb is not None and sb["tilt"] in ("BUY", "STRONG BUY") and \
            rsi is not None and 45 <= rsi <= 70 and regime_label != "risk-off":
        return "ENTER_TODAY", "Fresh setup firing, momentum healthy and not extended — today works."

    # 4) Default — no fresh trigger.
    return "WAIT_CONFIRMATION", "No fresh trigger today — wait for a setup or a confirming close."


def _size_plan(ticker, planned_amount, entry, stoploss, capital, risk_pct) -> PlanSizing:
    ps = risk_mod.position_size(entry, stoploss, capital, risk_pct, max_position_pct=25.0)
    shares_by_amount = int(planned_amount // entry) if entry > 0 else 0
    risk_sized_shares = ps.shares if ps else None
    if risk_sized_shares is not None:
        suggested = min(shares_by_amount, risk_sized_shares)
    else:
        suggested = shares_by_amount
    over_risk = risk_sized_shares is not None and shares_by_amount > risk_sized_shares
    suggested_value = suggested * entry
    risk_amount = suggested * (entry - stoploss)

    # Concentration cross-check vs the current book (read-only).
    over_conc, conc_flag = _concentration_delta(ticker, suggested_value)

    if suggested <= 0:
        advice = "stop too wide / amount too small — no whole-share position fits your risk budget"
    elif over_risk:
        advice = (f"trim to {risk_sized_shares} shares (₹{risk_sized_shares*entry:,.0f}) to keep a "
                  f"stop-out ≤ {risk_pct:.0f}% of capital")
    elif over_conc:
        advice = f"size ok on risk, but {conc_flag}"
    else:
        advice = "size ok — risk and concentration within limits"

    return PlanSizing(
        planned_amount=round(planned_amount, 2), shares_by_amount=shares_by_amount,
        risk_sized_shares=risk_sized_shares, suggested_shares=suggested,
        suggested_value=round(suggested_value, 2),
        pct_of_capital=round(suggested_value / capital * 100, 1) if capital else 0.0,
        risk_amount=round(risk_amount, 2),
        risk_pct_of_capital=round(risk_amount / capital * 100, 2) if capital else 0.0,
        over_risk=over_risk, over_concentration=over_conc,
        concentration_flag=conc_flag, advice=advice,
    )


def _concentration_delta(ticker: str, add_value: float) -> tuple[bool, str | None]:
    """Does adding ``add_value`` of ``ticker`` introduce a concentration flag the
    book doesn't already have? Holdings are read-only here."""
    try:
        base = load_holdings()
    except Exception:
        return False, None
    base_flags = set(risk_mod.concentration(base).flags) if base is not None and not base.empty \
        and "current_value" in base.columns else set()
    add_row = pd.DataFrame([{"ticker": ticker, "current_value": add_value}])
    hypo = pd.concat([base[["ticker", "current_value"]] if not base.empty and "current_value" in base.columns
                      else pd.DataFrame(columns=["ticker", "current_value"]), add_row],
                     ignore_index=True)
    hc = risk_mod.concentration(hypo)
    if hc is None:
        return False, None
    new = [f for f in hc.flags if f not in base_flags]
    return (bool(new), new[0] if new else None)


def _holding_advice(held, spot, last, tq, sb, manip_tier, timing, conviction) -> HoldingAdvice:
    qty = float(held.get("qty") or 0)
    avg = float(held.get("avg_price") or 0)
    lastp = float(held.get("last_price") or spot)
    unreal = (lastp - avg) / avg * 100 if avg > 0 else 0.0

    ema200 = float(last["ema200"]) if pd.notna(last.get("ema200")) else None
    broken = ema200 is not None and spot < ema200
    tilt = sb["tilt"] if sb else "NEUTRAL"

    if (broken and tilt in ("SELL", "STRONG SELL")) or manip_tier == "High" or \
            (tq is not None and tq["verdict"] == "false"):
        action, reason = "EXIT", ("trend broken with selling pressure" if broken
                                  else "distribution / unusual-activity risk into your position — protect capital")
    elif timing == "WAIT_PULLBACK" and unreal > 5:
        action, reason = "TRIM", f"extended and you're +{unreal:.0f}% — book partial, ride the rest"
    elif manip_tier == "Elevated":
        action, reason = "TRIM", "elevated unusual-activity — lighten and tighten the stop"
    elif timing == "ENTER_TODAY" and conviction >= TIER_BUY:
        action, reason = "ADD", "still firing and not extended — room to add if concentration allows"
    else:
        action, reason = "HOLD", "trend intact, nothing actionable — let it work"

    return HoldingAdvice(qty=qty, avg_price=round(avg, 2), last_price=round(lastp, 2),
                         unrealized_pct=round(unreal, 1), action=action, reason=reason)


def _single_factor_row(ticker: str) -> dict | None:
    """Best-effort single-ticker factor row (cross-sectional rank needs ≥3 names,
    so this is usually None for a lone ticker — weights renormalise)."""
    try:
        tab = factors_mod.factor_table([ticker])
        if tab is None or tab.empty:
            return None
        return tab.to_dict("records")[0]
    except Exception:
        return None


def _safe(fn, *args, **kwargs):
    """Call a fan-out analytic, swallowing failures (missing data, too few bars)
    so one weak input never sinks the whole decision."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def decide_universe(tickers: list[str], *, capital: float = ACCOUNT_CAPITAL,
                    risk_pct: float = RISK_PER_TRADE_PCT, run_montecarlo: bool = True,
                    mc_sims: int = 2000, use_ml: bool = False,
                    plans: dict | None = None) -> list[Decision]:
    """Decide over a whole universe, computing the cross-sectional / global inputs
    ONCE (factor table, regime, sector rotation, holdings, optional ML) and
    threading them into each :func:`decide`. ``plans`` maps ticker→planned ₹."""
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return []

    factor_map: dict = {}
    ftab = _safe(factors_mod.factor_table, tickers)
    if ftab is not None and not ftab.empty:
        factor_map = {r["ticker"]: r for r in ftab.to_dict("records")}

    regime_obj = _safe(spillover_mod.regime)

    sector_bias_map: dict = {}
    snap = _safe(sectors_mod.build_snapshot, tickers)
    if snap is not None and not snap.empty:
        groups = _safe(sectors_mod.rank_groups, snap, by="sector")
        if groups is not None and not groups.empty and "sector" in groups.columns:
            sector_bias_map = dict(zip(groups["sector"], groups["bias"]))

    held_map: dict = {}
    try:
        hold = load_holdings()
        if hold is not None and not hold.empty:
            for r in hold.to_dict("records"):
                held_map[r["ticker"]] = {"qty": r.get("qty"), "avg_price": r.get("avg_price"),
                                         "last_price": r.get("last_price")}
    except Exception:
        pass

    ml_map: dict = {}
    if use_ml:
        from swingdesk.analyze import ml_direction
        pred = _safe(ml_direction.train_and_predict, tickers)
        if pred is not None and not pred.empty:
            ml_map = {r["ticker"]: r.get("prob_up") for r in pred.to_dict("records")}

    plans = plans or {}
    out: list[Decision] = []
    for t in tickers:
        d = _safe(decide, t, planned_amount=plans.get(t), capital=capital, risk_pct=risk_pct,
                  held=held_map.get(t), regime_obj=regime_obj, sector_bias_map=sector_bias_map,
                  factor_row=factor_map.get(t), run_montecarlo=run_montecarlo, mc_sims=mc_sims,
                  use_ml=use_ml, ml_prob=ml_map.get(t))
        if d is not None:
            out.append(d)
    out.sort(key=lambda d: (d.conviction if d.conviction is not None else -1), reverse=True)
    return out
