"""Market-flow aware strategy selection and confluence ranking.

Traders who look "accurate" usually are not predicting one stock in isolation.
They are matching playbook to tape:

* trend-up / risk-on: breakouts and trend-following have room to run
* range / chop: pullbacks and mean-reversion entries behave better
* volatile / risk-off: reduce size; only the cleanest momentum or defensive reads

This module adds that missing layer over the existing SwingDesk parts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from swingdesk.analyze import board as board_mod
from swingdesk.analyze import global_impact, intraday as intraday_mod
from swingdesk.analyze import liquidity as liquidity_mod
from swingdesk.analyze import manipulation, market_metrics, sudden_move
from swingdesk.backtest import engine, metrics
from swingdesk.config import BACKTEST_COST_PCT
from swingdesk.storage import combined_universe, get_fundamentals, load_macro, load_prices


FLOW_COLS = ["date", "flow", "score", "nifty_trend", "volatility", "reason"]
STRATEGY_FLOW_COLS = [
    "flow", "family", "setup", "n_trades", "win_rate", "avg_r", "expectancy",
    "profit_factor", "max_drawdown_r", "works", "reason",
]
CONFLUENCE_COLS = [
    "ticker", "confluence_score", "market_flow", "preferred_playbook",
    "explosive_score", "target_move_pct", "hist_1d_hit_rate_pct",
    "max_1d_high_move_pct", "entry_trigger", "invalid_if",
    "live_confirmation_score", "breakout_confirmed", "vwap_hold", "rvol",
    "catalyst_score", "order_value_mcap_pct", "order_value_spike_mult",
    "volume_float_pct", "liquidity_tier", "liquidity_score", "amihud",
    "manip_tier",
    "radar_score", "board_conviction", "strategy_fit", "risk_penalty",
    "readiness", "category", "tags", "why_10_15_possible", "reasons", "risks",
]
EXPLOSIVE_BT_COLS = [
    "ticker", "date", "explosive_score", "radar_score", "plausibility_score",
    "next_high_ret_pct", "next_close_ret_pct", "hit", "playbook",
]


SETUP_FAMILY = {
    "breakout_20d": "momentum_breakout",
    "volume_thrust": "momentum_breakout",
    "bollinger_breakout": "momentum_breakout",
    "adx_trend": "trend_following",
    "ema_20_50_cross": "trend_following",
    "macd_cross": "trend_following",
    "supertrend_flip": "trend_following",
    "golden_cross": "trend_following",
    "pullback_ema20": "pullback_continuation",
    "rsi_reversal": "pullback_continuation",
}

FLOW_FIT = {
    "risk_on_trend": {
        "momentum_breakout": 95,
        "trend_following": 90,
        "pullback_continuation": 72,
        "mean_reversion": 40,
    },
    "steady_uptrend": {
        "pullback_continuation": 88,
        "trend_following": 82,
        "momentum_breakout": 75,
        "mean_reversion": 45,
    },
    "range_chop": {
        "pullback_continuation": 70,
        "mean_reversion": 78,
        "momentum_breakout": 42,
        "trend_following": 45,
    },
    "volatile": {
        "momentum_breakout": 58,
        "pullback_continuation": 45,
        "trend_following": 40,
        "mean_reversion": 35,
    },
    "risk_off": {
        "momentum_breakout": 30,
        "trend_following": 25,
        "pullback_continuation": 28,
        "mean_reversion": 32,
    },
    "unknown": {
        "momentum_breakout": 50,
        "trend_following": 50,
        "pullback_continuation": 50,
        "mean_reversion": 50,
    },
}

PLAYBOOKS = {
    "momentum_breakout": {
        "name": "Momentum Breakout",
        "entry_trigger": "Prior-day high break with RVOL >= 2x and price holding above VWAP",
        "invalid_if": "Breakout fades below VWAP/opening range or volume dries below 1.5x",
        "works_when": "Risk-on trend, sector tailwind, compressed base resolving upward",
    },
    "trend_following": {
        "name": "Trend Follow-Through",
        "entry_trigger": "Trend signal fires, first pullback/reclaim holds 20EMA/VWAP",
        "invalid_if": "Close slips below 20EMA/VWAP with weak breadth or market turns risk-off",
        "works_when": "Steady uptrend with low/moderate volatility",
    },
    "pullback_continuation": {
        "name": "Pullback Continuation",
        "entry_trigger": "Dip into 20EMA/support reverses with bullish candle and improving volume",
        "invalid_if": "Support breaks or bounce lacks volume confirmation",
        "works_when": "Range/chop or steady uptrend, not panic/risk-off tape",
    },
    "mean_reversion": {
        "name": "Mean Reversion",
        "entry_trigger": "Oversold reversal from support with market stabilizing",
        "invalid_if": "New low after bounce attempt or broad market volatility expands",
        "works_when": "Range-bound market with clear support/resistance",
    },
}


@dataclass
class FlowState:
    flow: str
    score: int
    nifty_trend: str
    volatility: str
    reason: str


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


def setup_family(setup: str | None) -> str:
    return SETUP_FAMILY.get(setup or "", "momentum_breakout")


def strategy_fit_score(flow: str, setup_or_family: str | None) -> int:
    fam = setup_or_family if setup_or_family in next(iter(FLOW_FIT.values())) else setup_family(setup_or_family)
    return int(FLOW_FIT.get(flow, FLOW_FIT["unknown"]).get(fam, 50))


def playbook_details(playbook: str) -> dict:
    return PLAYBOOKS.get(playbook, PLAYBOOKS["momentum_breakout"])


def one_day_explosive_profile(ticker: str, *, target_move_pct: float = 10.0,
                              lookback: int = 252) -> dict:
    """Can this stock realistically print a target one-day high move?

    This is a plausibility profile, not a forecast. It looks at the stock's own
    recent high-from-previous-close distribution and returns how often/large
    those one-day expansions have been.
    """
    df = load_prices(ticker)
    if df is None or df.empty or len(df) < 40:
        return {
            "target_move_pct": target_move_pct,
            "hist_1d_hit_rate_pct": 0.0,
            "max_1d_high_move_pct": 0.0,
            "p95_1d_high_move_pct": 0.0,
            "plausibility_score": 0.0,
            "reason": "not enough price history",
        }
    d = df.tail(lookback + 1).dropna(subset=["close", "high", "volume"]).copy()
    prev_close = d["close"].shift(1)
    high_move = (d["high"] / prev_close - 1) * 100
    high_move = high_move.replace([np.inf, -np.inf], np.nan).dropna()
    if high_move.empty:
        return {
            "target_move_pct": target_move_pct,
            "hist_1d_hit_rate_pct": 0.0,
            "max_1d_high_move_pct": 0.0,
            "p95_1d_high_move_pct": 0.0,
            "plausibility_score": 0.0,
            "reason": "no valid high/close moves",
        }
    hit_rate = float((high_move >= target_move_pct).mean() * 100)
    max_move = float(high_move.max())
    p95 = float(high_move.quantile(0.95))
    # Plausibility rises if the stock has actually made these moves or routinely
    # prints high-tail expansions near the target.
    score = _clip(0.45 * min(hit_rate * 12.0, 100.0) +
                  0.35 * min(max_move / target_move_pct * 100.0, 100.0) +
                  0.20 * min(p95 / target_move_pct * 100.0, 100.0))
    reason = (
        f"{hit_rate:.1f}% of last {len(high_move)} sessions reached +{target_move_pct:.0f}% intraday; "
        f"max +{max_move:.1f}%, p95 +{p95:.1f}%"
    )
    return {
        "target_move_pct": target_move_pct,
        "hist_1d_hit_rate_pct": round(hit_rate, 1),
        "max_1d_high_move_pct": round(max_move, 1),
        "p95_1d_high_move_pct": round(p95, 1),
        "plausibility_score": round(score, 1),
        "reason": reason,
    }


def live_explosive_checks(ticker: str, *, target_move_pct: float = 10.0) -> dict:
    """First-class checklist for the 'is today becoming a +10/+15% day?' question.

    This makes the previously scattered factors visible:
    catalyst, RVOL, breakout, VWAP, order-value-vs-market-cap, volume-vs-float,
    liquidity/illiquidity, and manipulation footprint.
    """
    df = load_prices(ticker)
    fund = get_fundamentals(ticker) or {}
    out = {
        "live_confirmation_score": 0.0,
        "breakout_confirmed": False,
        "vwap_hold": False,
        "rvol": None,
        "catalyst_score": 0.0,
        "order_value_mcap_pct": None,
        "order_value_spike_mult": None,
        "volume_float_pct": None,
        "liquidity_tier": "n/a",
        "liquidity_score": None,
        "amihud": None,
        "manip_tier": "n/a",
        "checklist": [],
        "warnings": [],
    }
    if df is None or df.empty or len(df) < 25:
        out["warnings"].append("not enough price history for live checks")
        return out

    df = df.dropna(subset=["close", "high", "volume"]).copy()
    if len(df) < 25:
        out["warnings"].append("not enough valid OHLCV bars")
        return out

    # Catalyst: recent Indian tagged news + mapped global pressure.
    sent_score, sent_reasons = sudden_move._sentiment_pressure(ticker)
    glob_score, glob_reasons = global_impact.ticker_impact_score(ticker)
    catalyst_raw = sent_score + glob_score
    catalyst_score = _clip((catalyst_raw + 8.0) / 32.0 * 100)
    out["catalyst_score"] = round(catalyst_score, 1)

    # Market-structure metrics: order value vs market cap, value spike, volume
    # vs float, ADV, Amihud. This is exactly the family of checks the user named.
    mm = market_metrics.compute(df, fund, allow_derive=True, ticker=ticker)
    if mm:
        out["order_value_mcap_pct"] = (
            round(mm.today_turnover_pct, 3) if mm.today_turnover_pct is not None else None)
        out["order_value_spike_mult"] = (
            round(mm.today_value_spike_mult, 2) if mm.today_value_spike_mult is not None else None)
        out["volume_float_pct"] = (
            round(mm.today_float_turnover_pct, 3) if mm.today_float_turnover_pct is not None else None)
        out["amihud"] = round(mm.amihud, 4) if mm.amihud == mm.amihud else None

    liq = liquidity_mod.liquidity_profile(ticker, fund)
    if liq:
        out["liquidity_tier"] = liq.tier
        out["liquidity_score"] = liq.score
        if liq.tier in ("illiquid", "untradeable"):
            out["warnings"].append(f"{liq.tier} liquidity")

    try:
        manip = manipulation.scorecard(ticker, df, fund)
        out["manip_tier"] = manip.get("tier", "n/a")
        if out["manip_tier"] in ("High", "Elevated"):
            out["warnings"].append(f"{out['manip_tier']} unusual-activity footprint")
    except Exception:
        pass

    # Daily fallback checks.
    prior_high = float(df["high"].iloc[:-1].tail(20).max())
    last_close = float(df["close"].iloc[-1])
    daily_breakout = last_close > prior_high if prior_high > 0 else False
    base_vol = float(df["volume"].iloc[-21:-1].mean())
    daily_rvol = float(df["volume"].iloc[-1] / base_vol) if base_vol > 0 else None
    out["rvol"] = round(daily_rvol, 2) if daily_rvol is not None else None
    out["breakout_confirmed"] = bool(daily_breakout)

    # Intraday overrides if 5m bars exist: stronger than EOD fallback because it
    # can verify VWAP and opening-range behavior.
    try:
        intra = intraday_mod.intraday_signals(ticker)
    except Exception:
        intra = None
    if intra is not None:
        out["rvol"] = intra.rvol if intra.rvol == intra.rvol else out["rvol"]
        out["vwap_hold"] = intra.dist_vwap_pct > 0
        out["breakout_confirmed"] = bool(intra.bias == "long" or intra.last > prior_high)

    # Confirmation score: each component is observable and separately shown.
    score = 0.0
    if out["catalyst_score"] >= 60:
        score += 18
        out["checklist"].append("catalyst present")
    rvol = out["rvol"] or 0.0
    if rvol >= 2.5:
        score += 20
        out["checklist"].append(f"RVOL {rvol:.1f}x")
    elif rvol >= 1.5:
        score += 10
        out["checklist"].append(f"RVOL building {rvol:.1f}x")
    if out["breakout_confirmed"]:
        score += 18
        out["checklist"].append("breakout confirmed")
    if out["vwap_hold"]:
        score += 14
        out["checklist"].append("VWAP hold")
    if (out["order_value_spike_mult"] or 0) >= 3:
        score += 12
        out["checklist"].append(f"value spike {out['order_value_spike_mult']}x")
    if (out["volume_float_pct"] or 0) >= 2.0:
        score += 10
        out["checklist"].append(f"{out['volume_float_pct']}% float traded")
    if out["liquidity_tier"] in ("liquid", "moderate"):
        score += 8
        out["checklist"].append(out["liquidity_tier"])
    if out["manip_tier"] == "High":
        score -= 18
    elif out["manip_tier"] == "Elevated":
        score -= 8
    out["live_confirmation_score"] = round(_clip(score), 1)
    return out


def explosive_score(*, radar_score: float, confluence_score: float,
                    strategy_fit: float, plausibility_score: float,
                    live_confirmation_score: float = 0.0,
                    risk_penalty: float) -> float:
    """Single score for '10-15% one-day move candidate'."""
    return round(_clip(
        0.28 * radar_score +
        0.20 * confluence_score +
        0.15 * strategy_fit +
        0.17 * plausibility_score +
        0.20 * live_confirmation_score -
        0.75 * risk_penalty
    ), 1)


def explosive_move_backtest(tickers: list[str], *, target_move_pct: float = 10.0,
                            min_explosive_score: float = 70.0,
                            lookback: int = 120) -> tuple[pd.DataFrame, dict]:
    """Historical test for the concrete question: did a high-scoring setup
    reach +target_move_pct intraday on the next session?"""
    flow = current_flow()
    rows: list[dict] = []
    for t in list(dict.fromkeys(tickers)):
        full = load_prices(t)
        if full is None or full.empty or len(full) < lookback + 2:
            continue
        playbook = _preferred_playbook(flow.flow)
        fit = strategy_fit_score(flow.flow, playbook)
        for i in range(lookback, len(full) - 1):
            hist = full.iloc[:i + 1]
            rr = sudden_move.score_frame(
                t, hist, include_manipulation=False, include_intraday=False)
            if rr is None:
                continue
            # Historical plausibility using only bars up to today.
            prev_close = hist["close"].shift(1)
            hm = ((hist["high"] / prev_close - 1) * 100).replace([np.inf, -np.inf], np.nan).dropna()
            hit_rate = float((hm.tail(lookback) >= target_move_pct).mean() * 100) if not hm.empty else 0.0
            max_move = float(hm.tail(lookback).max()) if not hm.empty else 0.0
            p95 = float(hm.tail(lookback).quantile(0.95)) if not hm.empty else 0.0
            plaus = _clip(0.45 * min(hit_rate * 12.0, 100.0) +
                          0.35 * min(max_move / target_move_pct * 100.0, 100.0) +
                          0.20 * min(p95 / target_move_pct * 100.0, 100.0))
            ex = explosive_score(
                radar_score=rr.radar_score,
                confluence_score=rr.radar_score,
                strategy_fit=fit,
                plausibility_score=plaus,
                risk_penalty=0.0,
            )
            if ex < min_explosive_score:
                continue
            entry = float(full["close"].iloc[i])
            nxt = full.iloc[i + 1]
            high_ret = (float(nxt["high"]) / entry - 1) * 100
            close_ret = (float(nxt["close"]) / entry - 1) * 100
            rows.append({
                "ticker": t,
                "date": full.index[i].strftime("%Y-%m-%d") if hasattr(full.index[i], "strftime") else str(full.index[i]),
                "explosive_score": ex,
                "radar_score": rr.radar_score,
                "plausibility_score": round(plaus, 1),
                "next_high_ret_pct": round(high_ret, 2),
                "next_close_ret_pct": round(close_ret, 2),
                "hit": high_ret >= target_move_pct,
                "playbook": playbook,
            })
    trades = pd.DataFrame(rows, columns=EXPLOSIVE_BT_COLS)
    if trades.empty:
        return trades, {"n": 0, "hit_rate_pct": 0.0, "avg_next_high_ret_pct": 0.0}
    summary = {
        "n": int(len(trades)),
        "target_move_pct": target_move_pct,
        "min_explosive_score": min_explosive_score,
        "hit_rate_pct": round(float(trades["hit"].mean() * 100), 1),
        "avg_next_high_ret_pct": round(float(trades["next_high_ret_pct"].mean()), 2),
        "avg_next_close_ret_pct": round(float(trades["next_close_ret_pct"].mean()), 2),
        "median_explosive_score": round(float(trades["explosive_score"].median()), 1),
    }
    return trades, summary


def _classify_row(close: pd.Series, vix: pd.Series | None = None) -> FlowState:
    close = close.dropna().astype(float)
    if len(close) < 60:
        return FlowState("unknown", 0, "insufficient", "unknown", "Need at least 60 NIFTY bars.")
    last = float(close.iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close) >= 200 else ema50
    ret20 = last / float(close.iloc[-21]) - 1 if len(close) > 21 else 0.0
    ret60 = last / float(close.iloc[-61]) - 1 if len(close) > 61 else ret20
    daily = close.pct_change().dropna()
    vol20 = float(daily.tail(20).std() * np.sqrt(252) * 100) if len(daily) >= 20 else 0.0

    trend_score = 0
    reasons: list[str] = []
    if last > ema20 > ema50 and last > ema200:
        trend_score += 45
        reasons.append("NIFTY stacked above 20/50/200 EMA")
        trend = "up"
    elif last > ema50:
        trend_score += 25
        reasons.append("NIFTY above 50 EMA")
        trend = "mild_up"
    elif last < ema50 and last < ema200:
        trend_score -= 40
        reasons.append("NIFTY below 50/200 EMA")
        trend = "down"
    else:
        reasons.append("NIFTY between moving averages")
        trend = "chop"
    if ret20 > 0.03:
        trend_score += 20
        reasons.append(f"20d momentum {ret20*100:+.1f}%")
    elif ret20 < -0.03:
        trend_score -= 20
        reasons.append(f"20d momentum {ret20*100:+.1f}%")
    if ret60 > 0.06:
        trend_score += 10
    elif ret60 < -0.06:
        trend_score -= 10

    vol_label = "calm"
    vix_last = None
    if vix is not None and not vix.dropna().empty:
        vix_last = float(vix.dropna().iloc[-1])
        if vix_last >= 20:
            trend_score -= 30
            vol_label = "fear"
            reasons.append(f"India VIX elevated ({vix_last:.1f})")
        elif vix_last <= 13:
            trend_score += 10
            vol_label = "calm"
            reasons.append(f"India VIX calm ({vix_last:.1f})")
        else:
            vol_label = "normal"
    elif vol20 >= 28:
        trend_score -= 20
        vol_label = "high"
        reasons.append(f"realized volatility high ({vol20:.0f}% annualized)")
    elif vol20 <= 15:
        trend_score += 5
        vol_label = "calm"

    if trend_score <= -35:
        flow = "risk_off"
    elif vol_label in ("fear", "high") and abs(ret20) > 0.025:
        flow = "volatile"
    elif trend_score >= 60:
        flow = "risk_on_trend"
    elif trend_score >= 25:
        flow = "steady_uptrend"
    else:
        flow = "range_chop"
    return FlowState(
        flow=flow, score=int(max(-100, min(100, trend_score))),
        nifty_trend=trend, volatility=vol_label,
        reason="; ".join(reasons),
    )


def current_flow() -> FlowState:
    nifty = load_macro("^NSEI")
    vix = load_macro("^INDIAVIX")
    if nifty.empty:
        return FlowState("unknown", 0, "missing", "unknown", "No NIFTY macro data. Run `macro` first.")
    vix_s = vix["close"] if not vix.empty else None
    return _classify_row(nifty["close"], vix_s)


def historical_flows() -> pd.DataFrame:
    nifty = load_macro("^NSEI")
    if nifty.empty:
        return pd.DataFrame(columns=FLOW_COLS)
    vix = load_macro("^INDIAVIX")
    close = nifty["close"].astype(float)
    vix_close = vix["close"].astype(float) if not vix.empty else None
    rows = []
    for i in range(60, len(close)):
        date = close.index[i]
        vix_slice = None
        if vix_close is not None:
            vix_slice = vix_close[vix_close.index <= date]
        st = _classify_row(close.iloc[:i + 1], vix_slice)
        rows.append({
            "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
            "flow": st.flow, "score": st.score, "nifty_trend": st.nifty_trend,
            "volatility": st.volatility, "reason": st.reason,
        })
    return pd.DataFrame(rows, columns=FLOW_COLS)


def strategy_flow_backtest(tickers: list[str] | None = None, *, max_hold: int = 20,
                           min_trades: int = 5,
                           cost_pct: float = BACKTEST_COST_PCT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest existing setup detectors and summarize edge by market flow."""
    if tickers is None:
        tickers = combined_universe(include_smallcaps=False, include_discovery=False)
    trades = engine.backtest_universe(list(tickers), max_hold=max_hold, require_uptrend=False)
    flows = historical_flows()
    if trades.empty or flows.empty:
        return pd.DataFrame(columns=STRATEGY_FLOW_COLS), trades

    t = trades.copy()
    t["date"] = pd.to_datetime(t["entry_date"]).dt.strftime("%Y-%m-%d")
    f = flows[["date", "flow"]]
    t = t.merge(f, on="date", how="left")
    t["flow"] = t["flow"].fillna("unknown")
    t["family"] = t["setup"].map(lambda s: setup_family(str(s)))

    rows = []
    for (flow, family, setup), sub in t.groupby(["flow", "family", "setup"]):
        summ = metrics.summarize(sub, cost_pct=cost_pct)
        if summ.empty:
            continue
        row = summ[summ["setup"] == setup].iloc[0].to_dict()
        n = int(row["n_trades"])
        exp = float(row["expectancy"])
        pf = float(row["profit_factor"]) if row["profit_factor"] != float("inf") else 999.0
        works = n >= min_trades and exp > 0 and pf >= 1.05
        reason = "works in this flow" if works else (
            f"needs more samples ({n})" if n < min_trades else "historically weak in this flow")
        rows.append({
            "flow": flow, "family": family, "setup": setup, "n_trades": n,
            "win_rate": row["win_rate"], "avg_r": row["avg_r"],
            "expectancy": row["expectancy"], "profit_factor": row["profit_factor"],
            "max_drawdown_r": row["max_drawdown_r"], "works": works, "reason": reason,
        })
    out = pd.DataFrame(rows, columns=STRATEGY_FLOW_COLS)
    if not out.empty:
        out = out.sort_values(["flow", "expectancy", "profit_factor"],
                              ascending=[True, False, False]).reset_index(drop=True)
    return out, t


def _preferred_playbook(flow: str) -> str:
    fits = FLOW_FIT.get(flow, FLOW_FIT["unknown"])
    return max(fits, key=fits.get)


def confluence_board(tickers: list[str] | None = None, *, limit: int = 30,
                     include_intraday: bool = False,
                     target_move_pct: float = 10.0) -> pd.DataFrame:
    """Fuse radar + board + market-flow playbook fit into one ranked table."""
    if tickers is None:
        tickers = combined_universe(include_smallcaps=True, include_discovery=False)
    tickers = list(dict.fromkeys(tickers))
    flow = current_flow()
    radar = sudden_move.scan(tickers, include_intraday=include_intraday,
                             include_smallcaps=False, include_discovery=False, limit=max(limit, len(tickers)))
    board = board_mod.build_board(tickers, run_montecarlo=False, top_us=min(40, len(tickers)))

    rmap = {r["ticker"]: r for r in radar.to_dict("records")} if not radar.empty else {}
    bmap = {r["ticker"]: r for r in board.to_dict("records")} if not board.empty else {}
    rows = []
    for t in tickers:
        rr = rmap.get(t, {})
        br = bmap.get(t, {})
        setups = [x.strip() for x in str(br.get("setups") or "").split(",") if x.strip()]
        setup = setups[0] if setups else None
        playbook = setup_family(setup) if setup else _preferred_playbook(flow.flow)
        fit = strategy_fit_score(flow.flow, playbook)
        radar_score = float(rr.get("radar_score") or 0.0)
        conviction = float(br.get("conviction") or 0.0)
        gates = float(br.get("n_gates") or 0.0) / max(float(board_mod.N_GATES), 1.0) * 100.0
        risk_penalty = 0.0
        if br.get("manip_tier") in ("High", "Elevated"):
            risk_penalty += 18.0
        if br.get("liq_tier") in ("illiquid", "untradeable"):
            risk_penalty += 18.0
        if flow.flow == "risk_off":
            risk_penalty += 10.0
        score = _clip(0.34 * radar_score + 0.24 * conviction + 0.18 * gates + 0.24 * fit - risk_penalty)
        profile = one_day_explosive_profile(t, target_move_pct=target_move_pct)
        checks = live_explosive_checks(t, target_move_pct=target_move_pct)
        ex_score = explosive_score(
            radar_score=radar_score,
            confluence_score=score,
            strategy_fit=fit,
            plausibility_score=float(profile["plausibility_score"]),
            live_confirmation_score=float(checks["live_confirmation_score"]),
            risk_penalty=risk_penalty,
        )
        details = playbook_details(playbook)
        why = []
        if profile["plausibility_score"] >= 50:
            why.append(profile["reason"])
        if radar_score >= 60:
            why.append(f"radar {radar_score:.0f}: {rr.get('reasons') or 'setup pressure present'}")
        if fit >= 70:
            why.append(f"{details['name']} fits {flow.flow}")
        if checks["checklist"]:
            why.append("live: " + ", ".join(checks["checklist"][:4]))
        rows.append({
            "ticker": t,
            "confluence_score": round(score, 1),
            "market_flow": flow.flow,
            "preferred_playbook": playbook,
            "explosive_score": ex_score,
            "target_move_pct": target_move_pct,
            "hist_1d_hit_rate_pct": profile["hist_1d_hit_rate_pct"],
            "max_1d_high_move_pct": profile["max_1d_high_move_pct"],
            "entry_trigger": details["entry_trigger"],
            "invalid_if": details["invalid_if"],
            "live_confirmation_score": checks["live_confirmation_score"],
            "breakout_confirmed": checks["breakout_confirmed"],
            "vwap_hold": checks["vwap_hold"],
            "rvol": checks["rvol"],
            "catalyst_score": checks["catalyst_score"],
            "order_value_mcap_pct": checks["order_value_mcap_pct"],
            "order_value_spike_mult": checks["order_value_spike_mult"],
            "volume_float_pct": checks["volume_float_pct"],
            "liquidity_tier": checks["liquidity_tier"],
            "liquidity_score": checks["liquidity_score"],
            "amihud": checks["amihud"],
            "manip_tier": checks["manip_tier"],
            "radar_score": radar_score,
            "board_conviction": conviction,
            "strategy_fit": fit,
            "risk_penalty": round(risk_penalty, 1),
            "readiness": rr.get("readiness") or "",
            "category": br.get("category") or "",
            "tags": br.get("tags") or "",
            "why_10_15_possible": " · ".join(why) if why else profile["reason"],
            "reasons": rr.get("reasons") or br.get("tags") or "",
            "risks": " · ".join([x for x in [rr.get("risks"), *checks["warnings"]] if x and x != "—"])
                     or br.get("manip_tier") or "",
        })
    out = pd.DataFrame(rows, columns=CONFLUENCE_COLS)
    return out.sort_values(["explosive_score", "confluence_score"],
                           ascending=False).head(limit).reset_index(drop=True)
