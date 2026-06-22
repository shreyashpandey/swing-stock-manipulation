from __future__ import annotations

import math

import pandas as pd
from rich.console import Console

from swingdesk.storage import load_prices, save_signals
from swingdesk.analyze.technicals import add_indicators

console = Console()

# --- individual setup detectors --------------------------------------------------
# Each detector takes the indicator-augmented DataFrame and returns a dict
# (the signal) or None.

def _round(x, n=2):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return round(float(x), n)
    except Exception:
        return None


def detect_breakout(df: pd.DataFrame, ticker: str) -> dict | None:
    """20-day high breakout with volume confirmation, above 50EMA trend filter."""
    if len(df) < 60:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if any(pd.isna(last[c]) for c in ("high20", "vol_avg20", "ema50", "atr14")):
        return None
    breakout = last["close"] > prev["high20"]  # closes above prior 20d high
    vol_ok = last["volume"] > 1.5 * last["vol_avg20"]
    trend_ok = last["close"] > last["ema50"]
    if not (breakout and vol_ok and trend_ok):
        return None
    atr = last["atr14"]
    entry = last["close"]
    stoploss = entry - 1.5 * atr
    target = entry + 3.0 * atr
    rr = (target - entry) / max(entry - stoploss, 1e-6)
    score = 60 + min(40, (last["volume"] / last["vol_avg20"] - 1) * 20)
    return {
        "ticker": ticker,
        "setup": "breakout_20d",
        "direction": "long",
        "entry": _round(entry),
        "stoploss": _round(stoploss),
        "target": _round(target),
        "rr": _round(rr, 2),
        "score": _round(score, 1),
        "notes": (f"close {entry:.2f} > 20d-high {prev['high20']:.2f}; "
                  f"vol {last['volume']/last['vol_avg20']:.1f}x avg; above 50EMA"),
    }


def detect_pullback_to_ema(df: pd.DataFrame, ticker: str) -> dict | None:
    """Uptrend pullback: price tags 20EMA from above with RSI rebounding."""
    if len(df) < 60:
        return None
    last = df.iloc[-1]
    if any(pd.isna(last[c]) for c in ("ema20", "ema50", "rsi14", "atr14")):
        return None
    uptrend = last["ema20"] > last["ema50"] and last["close"] > last["ema50"]
    near_ema = abs(last["low"] - last["ema20"]) / last["close"] < 0.015  # within 1.5%
    bounced = last["close"] > last["open"] and last["rsi14"] > 40 and last["rsi14"] < 60
    if not (uptrend and near_ema and bounced):
        return None
    atr = last["atr14"]
    entry = last["close"]
    stoploss = last["low"] - 0.3 * atr
    target = entry + 2.5 * atr
    rr = (target - entry) / max(entry - stoploss, 1e-6)
    return {
        "ticker": ticker,
        "setup": "pullback_ema20",
        "direction": "long",
        "entry": _round(entry),
        "stoploss": _round(stoploss),
        "target": _round(target),
        "rr": _round(rr, 2),
        "score": _round(55 + (last["rsi14"] - 40), 1),
        "notes": f"pullback to 20EMA in uptrend; RSI {last['rsi14']:.0f}, bullish bar",
    }


def detect_ma_cross(df: pd.DataFrame, ticker: str) -> dict | None:
    """20EMA crossing above 50EMA — classic positional trend-start signal."""
    if len(df) < 60:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if any(pd.isna(x) for x in (last["ema20"], last["ema50"], prev["ema20"], prev["ema50"])):
        return None
    cross = prev["ema20"] <= prev["ema50"] and last["ema20"] > last["ema50"]
    if not cross:
        return None
    atr = last["atr14"]
    entry = last["close"]
    stoploss = entry - 2.0 * atr
    target = entry + 5.0 * atr
    rr = (target - entry) / max(entry - stoploss, 1e-6)
    return {
        "ticker": ticker,
        "setup": "ema_20_50_cross",
        "direction": "long",
        "entry": _round(entry),
        "stoploss": _round(stoploss),
        "target": _round(target),
        "rr": _round(rr, 2),
        "score": 65.0,
        "notes": "20EMA crossed above 50EMA (positional trend start)",
    }


def detect_volume_thrust(df: pd.DataFrame, ticker: str) -> dict | None:
    """Up-day on >2.5x average volume — often precedes continuation."""
    if len(df) < 30:
        return None
    last = df.iloc[-1]
    if any(pd.isna(x) for x in (last["vol_avg20"], last["atr14"])):
        return None
    big_vol = last["volume"] > 2.5 * last["vol_avg20"]
    up_day = last["close"] > last["open"] and last["close"] > df.iloc[-2]["close"]
    if not (big_vol and up_day):
        return None
    atr = last["atr14"]
    entry = last["close"]
    stoploss = entry - 1.2 * atr
    target = entry + 2.5 * atr
    rr = (target - entry) / max(entry - stoploss, 1e-6)
    return {
        "ticker": ticker,
        "setup": "volume_thrust",
        "direction": "long",
        "entry": _round(entry),
        "stoploss": _round(stoploss),
        "target": _round(target),
        "rr": _round(rr, 2),
        "score": _round(50 + (last["volume"] / last["vol_avg20"] - 2.5) * 10, 1),
        "notes": f"up-day on {last['volume']/last['vol_avg20']:.1f}x avg volume",
    }


def detect_macd_cross(df: pd.DataFrame, ticker: str) -> dict | None:
    """MACD line crosses above its signal line while above the 50-EMA — a clean,
    widely-followed momentum buy trigger."""
    if len(df) < 60:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    cols = ("macd", "macd_signal", "ema50", "atr14")
    if any(pd.isna(last.get(c)) for c in cols) or pd.isna(prev.get("macd")):
        return None
    cross_up = prev["macd"] <= prev["macd_signal"] and last["macd"] > last["macd_signal"]
    trend_ok = last["close"] > last["ema50"]
    below_zero = last["macd"] < 0          # cross from below 0 = earlier, stronger
    if not (cross_up and trend_ok):
        return None
    atr = last["atr14"]
    entry = last["close"]
    stoploss = entry - 1.5 * atr
    target = entry + 3.0 * atr
    rr = (target - entry) / max(entry - stoploss, 1e-6)
    return {
        "ticker": ticker, "setup": "macd_cross", "direction": "long",
        "entry": _round(entry), "stoploss": _round(stoploss), "target": _round(target),
        "rr": _round(rr, 2), "score": _round(60 + (8 if below_zero else 0), 1),
        "notes": (f"MACD crossed above signal{' from below zero' if below_zero else ''}; "
                  "above 50EMA"),
    }


def detect_supertrend_flip(df: pd.DataFrame, ticker: str) -> dict | None:
    """Supertrend flips from bearish to bullish with the long-term trend intact
    — an explicit regime-change buy signal; the Supertrend line is the stop."""
    if len(df) < 60 or "supertrend_dir" not in df.columns:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    if any(pd.isna(last.get(c)) for c in ("supertrend_dir", "supertrend", "ema200", "atr14")):
        return None
    flipped_up = float(prev.get("supertrend_dir", 0) or 0) < 0 and float(last["supertrend_dir"]) > 0
    trend_ok = last["close"] > last["ema200"]
    if not (flipped_up and trend_ok):
        return None
    atr = last["atr14"]
    entry = last["close"]
    # Stop just under the supertrend line (its built-in trailing stop), bounded by ATR.
    stoploss = min(float(last["supertrend"]), entry - 0.5 * atr)
    target = entry + 3.0 * atr
    rr = (target - entry) / max(entry - stoploss, 1e-6)
    if rr <= 0:
        return None
    return {
        "ticker": ticker, "setup": "supertrend_flip", "direction": "long",
        "entry": _round(entry), "stoploss": _round(stoploss), "target": _round(target),
        "rr": _round(rr, 2), "score": 62.0,
        "notes": f"Supertrend flipped bullish at ₹{last['supertrend']:.0f}; above 200EMA",
    }


def detect_bollinger_breakout(df: pd.DataFrame, ticker: str) -> dict | None:
    """Bollinger 'squeeze' (volatility coil at a 60-day low) resolving with a
    close above the upper band on expanding volume — a classic expansion buy."""
    if len(df) < 60:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    cols = ("bb_upper", "bb_width", "bb_width_min60", "vol_avg20", "atr14", "ema50")
    if any(pd.isna(last.get(c)) for c in cols):
        return None
    was_squeezed = prev["bb_width"] <= prev.get("bb_width_min60", prev["bb_width"]) * 1.15
    breakout = last["close"] > last["bb_upper"]
    vol_ok = last["volume"] > 1.3 * last["vol_avg20"]
    trend_ok = last["close"] > last["ema50"]
    if not (was_squeezed and breakout and vol_ok and trend_ok):
        return None
    atr = last["atr14"]
    entry = last["close"]
    stoploss = float(last["bb_mid"]) if pd.notna(last.get("bb_mid")) else entry - 1.5 * atr
    target = entry + 3.0 * atr
    rr = (target - entry) / max(entry - stoploss, 1e-6)
    if rr <= 0:
        return None
    return {
        "ticker": ticker, "setup": "bollinger_breakout", "direction": "long",
        "entry": _round(entry), "stoploss": _round(stoploss), "target": _round(target),
        "rr": _round(rr, 2), "score": _round(58 + (last["volume"]/last["vol_avg20"] - 1.3) * 8, 1),
        "notes": (f"squeeze breakout: close above upper band on "
                  f"{last['volume']/last['vol_avg20']:.1f}x volume"),
    }


def detect_rsi_reversal(df: pd.DataFrame, ticker: str) -> dict | None:
    """Oversold bounce inside a longer-term uptrend: RSI was < 35 and turns back
    up above 40 while price holds above the 200-EMA. Buy the dip, not the crash."""
    if len(df) < 60:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    if any(pd.isna(last.get(c)) for c in ("rsi14", "ema200", "atr14")) or pd.isna(prev.get("rsi14")):
        return None
    was_oversold = prev["rsi14"] < 35
    turning_up = last["rsi14"] > prev["rsi14"] and last["rsi14"] > 40
    uptrend = last["close"] > last["ema200"]
    bullish_bar = last["close"] > last["open"]
    if not (was_oversold and turning_up and uptrend and bullish_bar):
        return None
    atr = last["atr14"]
    entry = last["close"]
    stoploss = last["low"] - 0.5 * atr
    target = entry + 2.5 * atr
    rr = (target - entry) / max(entry - stoploss, 1e-6)
    if rr <= 0:
        return None
    return {
        "ticker": ticker, "setup": "rsi_reversal", "direction": "long",
        "entry": _round(entry), "stoploss": _round(stoploss), "target": _round(target),
        "rr": _round(rr, 2), "score": _round(55 + (last["rsi14"] - 40), 1),
        "notes": f"oversold bounce (RSI {prev['rsi14']:.0f}→{last['rsi14']:.0f}) above 200EMA",
    }


def detect_golden_cross(df: pd.DataFrame, ticker: str) -> dict | None:
    """50-EMA crosses above the 200-EMA — the classic long-horizon 'golden cross'
    that often marks the start of a multi-month positional uptrend."""
    if len(df) < 210:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    if any(pd.isna(x) for x in (last.get("ema50"), last.get("ema200"),
                                prev.get("ema50"), prev.get("ema200"), last.get("atr14"))):
        return None
    cross = prev["ema50"] <= prev["ema200"] and last["ema50"] > last["ema200"]
    if not cross:
        return None
    atr = last["atr14"]
    entry = last["close"]
    stoploss = entry - 2.5 * atr
    target = entry + 6.0 * atr
    rr = (target - entry) / max(entry - stoploss, 1e-6)
    return {
        "ticker": ticker, "setup": "golden_cross", "direction": "long",
        "entry": _round(entry), "stoploss": _round(stoploss), "target": _round(target),
        "rr": _round(rr, 2), "score": 68.0,
        "notes": "50EMA crossed above 200EMA (golden cross — positional trend start)",
    }


def detect_adx_trend(df: pd.DataFrame, ticker: str) -> dict | None:
    """Trend-strength entry: ADX rising through 25 with DI+ above DI- and price
    above the 20-EMA — enter only when the market is genuinely trending up."""
    if len(df) < 60 or "adx14" not in df.columns:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    cols = ("adx14", "di_plus", "di_minus", "ema20", "atr14")
    if any(pd.isna(last.get(c)) for c in cols) or pd.isna(prev.get("adx14")):
        return None
    strong = last["adx14"] >= 25
    rising = last["adx14"] > prev["adx14"]
    bull_dir = last["di_plus"] > last["di_minus"]
    trend_ok = last["close"] > last["ema20"]
    if not (strong and rising and bull_dir and trend_ok):
        return None
    atr = last["atr14"]
    entry = last["close"]
    stoploss = entry - 2.0 * atr
    target = entry + 4.0 * atr
    rr = (target - entry) / max(entry - stoploss, 1e-6)
    return {
        "ticker": ticker, "setup": "adx_trend", "direction": "long",
        "entry": _round(entry), "stoploss": _round(stoploss), "target": _round(target),
        "rr": _round(rr, 2), "score": _round(58 + min(20, last["adx14"] - 25), 1),
        "notes": f"ADX {last['adx14']:.0f} rising, DI+ > DI−, above 20EMA — trending up",
    }


DETECTORS = [
    detect_breakout, detect_pullback_to_ema, detect_ma_cross, detect_volume_thrust,
    detect_macd_cross, detect_supertrend_flip, detect_bollinger_breakout,
    detect_rsi_reversal, detect_golden_cross, detect_adx_trend,
]


def scan_ticker(ticker: str, require_uptrend: bool = True) -> list[dict]:
    """Detect setups for one ticker. With require_uptrend (default), no signals
    are emitted while price is below its 200-day EMA — a counter-trend long is
    the lowest-quality entry. Backtest evidence (exit_tuner sweep, 2026-06): this
    trend gate is the single biggest expectancy improver (profit factor
    0.83→0.95). It does NOT make the setups net-profitable on its own — treat
    signals as screened ideas, not a mechanical system."""
    df = load_prices(ticker)
    if df.empty:
        return []
    df = add_indicators(df)
    if require_uptrend:
        last = df.iloc[-1]
        e200 = last.get("ema200")
        # Only gate when EMA200 is computable; too-short history isn't penalised.
        if e200 is not None and not pd.isna(e200) and last["close"] <= e200:
            return []
    signals = []
    for fn in DETECTORS:
        try:
            sig = fn(df, ticker)
            if sig:
                signals.append(sig)
        except Exception as e:
            console.print(f"[red]{fn.__name__} failed on {ticker}: {e}[/red]")
    return signals


def scan_all(tickers: list[str], persist: bool = True,
             universe: str = "main", require_uptrend: bool = True) -> list[dict]:
    """Scan a list of tickers for setups. `universe` labels persisted signals
    so the Signals tab can filter (main vs smallcap). `require_uptrend` gates
    out counter-trend (below-200-EMA) entries."""
    all_signals: list[dict] = []
    for t in tickers:
        sigs = scan_ticker(t, require_uptrend=require_uptrend)
        if sigs:
            for s in sigs:
                s["universe"] = universe
                console.print(
                    f"  [green]signal[/green] {s['ticker']:>15}  "
                    f"{s['setup']:<18}  entry={s['entry']:>8}  "
                    f"sl={s['stoploss']:>8}  tgt={s['target']:>8}  R:R={s['rr']}  "
                    f"[dim]{universe}[/dim]"
                )
            all_signals.extend(sigs)
    if persist and all_signals:
        save_signals(all_signals, universe=universe)
    return all_signals
