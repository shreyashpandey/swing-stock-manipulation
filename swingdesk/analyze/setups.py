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


DETECTORS = [detect_breakout, detect_pullback_to_ema, detect_ma_cross, detect_volume_thrust]


def scan_ticker(ticker: str) -> list[dict]:
    df = load_prices(ticker)
    if df.empty:
        return []
    df = add_indicators(df)
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
             universe: str = "main") -> list[dict]:
    """Scan a list of tickers for setups. `universe` labels persisted signals
    so the Signals tab can filter (main vs smallcap)."""
    all_signals: list[dict] = []
    for t in tickers:
        sigs = scan_ticker(t)
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
