"""Sudden Move Radar and backtest.

This does not predict with certainty. It ranks stocks where the setup pressure
for an outsized move is unusually high, then lets intraday confirmation improve
the read when available.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from swingdesk.analyze import global_impact, manipulation, market_metrics
from swingdesk.analyze.technicals import add_indicators, signal_scoreboard, trend_quality
from swingdesk.storage import (
    combined_universe,
    get_fundamentals,
    load_intraday,
    load_prices,
    recent_sentiment_for_ticker,
)

MIN_BARS = 90

RADAR_COLS = [
    "ticker", "radar_score", "readiness", "directional_bias", "compression",
    "accumulation", "prebreakout", "relative_strength", "float_spark",
    "catalyst", "global_score", "intraday_confirm", "manip_penalty",
    "last", "range_20d_pct", "near_20d_high_pct", "volume_mult",
    "adv_value_cr", "float_turnover_pct", "reasons", "risks",
]


@dataclass
class RadarRow:
    ticker: str
    radar_score: float
    readiness: str
    directional_bias: str
    compression: float
    accumulation: float
    prebreakout: float
    relative_strength: float
    float_spark: float
    catalyst: float
    global_score: float
    intraday_confirm: float
    manip_penalty: float
    last: float
    range_20d_pct: float | None
    near_20d_high_pct: float | None
    volume_mult: float | None
    adv_value_cr: float | None
    float_turnover_pct: float | None
    reasons: list[str]
    risks: list[str]

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["reasons"] = " · ".join(self.reasons) if self.reasons else "—"
        d["risks"] = " · ".join(self.risks) if self.risks else "—"
        return d


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if x != x:
        return 0.0
    return float(max(lo, min(hi, x)))


def _ret(close: pd.Series, n: int) -> float | None:
    if len(close) <= n or close.iloc[-1 - n] <= 0:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - n] - 1)


def _compression(df: pd.DataFrame) -> tuple[float, float | None]:
    last = float(df["close"].iloc[-1])
    hi20 = float(df["high"].tail(20).max())
    lo20 = float(df["low"].tail(20).min())
    range_pct = (hi20 / lo20 - 1) * 100 if lo20 > 0 else None
    # 3-12% 20d range is the useful "coil" band for swing moves.
    range_score = _clip((12.0 - (range_pct or 99.0)) / 9.0 * 100)

    ret = df["close"].pct_change().dropna()
    vol20 = float(ret.tail(20).std()) if len(ret) >= 20 else float("nan")
    vol60 = float(ret.tail(60).std()) if len(ret) >= 60 else float("nan")
    vol_score = _clip((1.0 - (vol20 / vol60 if vol60 > 0 else 1.0)) * 160)
    return round(0.7 * range_score + 0.3 * vol_score, 1), round(range_pct, 2) if range_pct else None


def _accumulation(df: pd.DataFrame) -> tuple[float, float | None]:
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    direction = np.sign(close.diff().fillna(0.0))
    signed_vol = direction * vol
    flow20 = float(signed_vol.tail(20).sum())
    abs20 = float(vol.tail(20).sum())
    flow_score = _clip((flow20 / abs20 + 0.2) / 0.7 * 100) if abs20 > 0 else 0.0

    base = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float("nan")
    volume_mult = float(vol.iloc[-1] / base) if base and base > 0 else None
    vol_score = _clip(((volume_mult or 1.0) - 0.8) / 1.7 * 100)

    r20 = _ret(close, 20) or 0.0
    quiet_gain_score = _clip((0.12 - abs(r20)) / 0.12 * 100)
    score = 0.45 * flow_score + 0.30 * vol_score + 0.25 * quiet_gain_score
    return round(score, 1), round(volume_mult, 2) if volume_mult else None


def _prebreakout(df: pd.DataFrame) -> tuple[float, float | None]:
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    hi20 = float(df["high"].tail(20).max())
    if hi20 <= 0:
        return 0.0, None
    dist = (hi20 / last - 1) * 100
    near_score = _clip((5.0 - dist) / 5.0 * 100)
    trend = _ret(close, 60) or 0.0
    trend_score = _clip((trend + 0.05) / 0.25 * 100)
    return round(0.65 * near_score + 0.35 * trend_score, 1), round(dist, 2)


def _relative_strength(df: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> float:
    own = _ret(df["close"], 20)
    if own is None:
        return 0.0
    bench_ret = 0.0
    if benchmark is not None and not benchmark.empty and len(benchmark) > 20:
        bench_ret = _ret(benchmark["close"], 20) or 0.0
    spread = own - bench_ret
    return round(_clip((spread + 0.04) / 0.16 * 100), 1)


def _float_spark(df: pd.DataFrame, fund: dict | None) -> tuple[float, float | None, float | None]:
    mm = market_metrics.compute(df, fund or {}, allow_derive=True)
    if mm is None:
        return 0.0, None, None
    # Sweet spot: enough value to trade, but not so huge that a surprise move is
    # hard. Penalise extremely thin names through liquidity/manipulation risks.
    adv = mm.adv_value_cr
    tradeable = _clip((adv - 2.0) / 28.0 * 100) if adv < 30 else _clip((180.0 - adv) / 150.0 * 100)
    float_turn = mm.avg_float_turnover_pct or 0.0
    float_score = _clip(float_turn / 1.5 * 100)
    return round(0.65 * tradeable + 0.35 * float_score, 1), round(adv, 2), (
        round(float_turn, 2) if mm.avg_float_turnover_pct is not None else None)


def _sentiment_pressure(ticker: str, days: int = 3) -> tuple[float, list[str]]:
    df = recent_sentiment_for_ticker(ticker, days=days)
    if df.empty:
        return 0.0, []
    raw = 0.0
    reasons: list[str] = []
    for _, r in df.iterrows():
        imp = {"high": 12.0, "medium": 7.0, "low": 3.0}.get(r.get("impact"), 0.0)
        if r.get("sentiment") == "bullish":
            raw += imp
        elif r.get("sentiment") == "bearish":
            raw -= imp * 1.2
        if len(reasons) < 2:
            reasons.append(str(r.get("title") or "")[:80])
    return raw, reasons


def _intraday_confirmation(ticker: str) -> tuple[float, list[str]]:
    try:
        from swingdesk.analyze.intraday import intraday_signals
        sig = intraday_signals(ticker)
    except Exception:
        return 0.0, []
    if sig is None:
        return 0.0, []
    score = 0.0
    reasons: list[str] = []
    if sig.bias == "long":
        score += 12.0
        reasons.append(sig.setup)
    if sig.rvol == sig.rvol and sig.rvol >= 1.8:
        score += min(10.0, (sig.rvol - 1.8) * 4.0)
        reasons.append(f"intraday RVOL {sig.rvol:.1f}x")
    if sig.dist_vwap_pct > 0:
        score += min(6.0, sig.dist_vwap_pct * 2.0)
        reasons.append("above VWAP")
    return round(_clip(score, 0, 25), 1), reasons


def score_frame(ticker: str, df: pd.DataFrame, *, fund: dict | None = None,
                benchmark: pd.DataFrame | None = None, sentiment_score: float = 0.0,
                sentiment_reasons: list[str] | None = None,
                global_score: float = 0.0, global_reasons: list[str] | None = None,
                include_manipulation: bool = True,
                include_intraday: bool = False) -> RadarRow | None:
    """Score one ticker from an OHLCV frame. Pure enough for backtests: callers
    can inject historical sentiment/global scores or leave them zero."""
    if df is None or df.empty or len(df) < MIN_BARS:
        return None
    df = df.dropna(subset=["close", "volume"]).copy()
    if len(df) < MIN_BARS:
        return None

    fund = fund or {}
    comp, range_pct = _compression(df)
    accum, volume_mult = _accumulation(df)
    pre, near_high = _prebreakout(df)
    rel = _relative_strength(df, benchmark)
    spark, adv_cr, float_turn = _float_spark(df, fund)

    catalyst_raw = sentiment_score + global_score
    catalyst = _clip((catalyst_raw + 8.0) / 32.0 * 100)

    intraday, intraday_reasons = _intraday_confirmation(ticker) if include_intraday else (0.0, [])

    manip_penalty = 0.0
    risks: list[str] = []
    if include_manipulation:
        try:
            manip = manipulation.scorecard(ticker, df, fund)
            if manip.get("tier") == "High":
                manip_penalty = 25.0
                risks.append("high unusual-activity risk")
            elif manip.get("tier") == "Elevated":
                manip_penalty = 12.0
                risks.append("elevated unusual-activity risk")
        except Exception:
            pass
    if adv_cr is not None and adv_cr < 2.0:
        risks.append("very thin average traded value")

    score = (
        0.20 * comp +
        0.20 * accum +
        0.18 * pre +
        0.14 * rel +
        0.12 * spark +
        0.11 * catalyst +
        intraday -
        manip_penalty
    )
    score = round(_clip(score), 1)
    readiness = "High" if score >= 75 else "Watch" if score >= 60 else "Developing" if score >= 45 else "Low"

    reasons: list[str] = []
    if comp >= 65:
        reasons.append("compressed 20d range")
    if accum >= 65:
        reasons.append("quiet accumulation")
    if pre >= 65:
        reasons.append("near breakout zone")
    if rel >= 65:
        reasons.append("relative strength")
    if spark >= 65:
        reasons.append("float/liquidity can move")
    for r in (sentiment_reasons or [])[:2]:
        reasons.append(f"news: {r}")
    for r in (global_reasons or [])[:2]:
        reasons.append(f"global: {r}")
    reasons.extend(intraday_reasons[:3])

    return RadarRow(
        ticker=ticker, radar_score=score, readiness=readiness,
        directional_bias="upside" if score >= 45 else "neutral",
        compression=comp, accumulation=accum, prebreakout=pre,
        relative_strength=rel, float_spark=spark, catalyst=round(catalyst, 1),
        global_score=global_score, intraday_confirm=intraday,
        manip_penalty=manip_penalty, last=round(float(df["close"].iloc[-1]), 2),
        range_20d_pct=range_pct, near_20d_high_pct=near_high,
        volume_mult=volume_mult, adv_value_cr=adv_cr,
        float_turnover_pct=float_turn, reasons=reasons, risks=risks,
    )


def scan(tickers: list[str] | None = None, *, include_intraday: bool = False,
         include_smallcaps: bool = True, include_discovery: bool = False,
         limit: int = 50) -> pd.DataFrame:
    """Live sudden-move radar over a universe."""
    if tickers is None:
        tickers = combined_universe(include_smallcaps=include_smallcaps,
                                    include_discovery=include_discovery)
    rows: list[dict] = []
    for t in list(dict.fromkeys(tickers)):
        df = load_prices(t)
        if df is None or df.empty:
            continue
        fund = get_fundamentals(t) or {}
        sent, sent_reasons = _sentiment_pressure(t)
        glob, glob_reasons = global_impact.ticker_impact_score(t)
        row = score_frame(
            t, df, fund=fund, sentiment_score=sent, sentiment_reasons=sent_reasons,
            global_score=glob, global_reasons=glob_reasons,
            include_intraday=include_intraday, include_manipulation=True,
        )
        if row:
            rows.append(row.as_dict())
    if not rows:
        return pd.DataFrame(columns=RADAR_COLS)
    out = pd.DataFrame(rows, columns=RADAR_COLS)
    return out.sort_values("radar_score", ascending=False).head(limit).reset_index(drop=True)


def backtest(tickers: list[str], *, min_score: float = 70.0, move_threshold_pct: float = 3.0,
             horizon: int = 1, lookback: int = MIN_BARS) -> tuple[pd.DataFrame, dict]:
    """Historical test: when radar score exceeds `min_score`, did the next
    `horizon` session(s) print an outsized high/close move?"""
    rows: list[dict] = []
    for t in list(dict.fromkeys(tickers)):
        full = load_prices(t)
        if full is None or full.empty or len(full) < lookback + horizon + 5:
            continue
        fund = get_fundamentals(t) or {}
        for i in range(lookback, len(full) - horizon):
            hist = full.iloc[:i + 1]
            r = score_frame(t, hist, fund=fund, include_manipulation=False,
                            include_intraday=False)
            if r is None or r.radar_score < min_score:
                continue
            entry = float(full["close"].iloc[i])
            future = full.iloc[i + 1:i + 1 + horizon]
            high_ret = (float(future["high"].max()) / entry - 1) * 100
            close_ret = (float(future["close"].iloc[-1]) / entry - 1) * 100
            rows.append({
                "ticker": t,
                "date": full.index[i].strftime("%Y-%m-%d"),
                "radar_score": r.radar_score,
                "next_high_ret_pct": round(high_ret, 2),
                "next_close_ret_pct": round(close_ret, 2),
                "hit_high": high_ret >= move_threshold_pct,
                "hit_close": close_ret >= move_threshold_pct,
                "readiness": r.readiness,
            })
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, {
            "n": 0, "hit_rate_high": 0.0, "hit_rate_close": 0.0,
            "avg_next_high_ret_pct": 0.0, "avg_next_close_ret_pct": 0.0,
        }
    summary = {
        "n": int(len(trades)),
        "hit_rate_high": round(float(trades["hit_high"].mean() * 100), 1),
        "hit_rate_close": round(float(trades["hit_close"].mean() * 100), 1),
        "avg_next_high_ret_pct": round(float(trades["next_high_ret_pct"].mean()), 2),
        "avg_next_close_ret_pct": round(float(trades["next_close_ret_pct"].mean()), 2),
        "median_score": round(float(trades["radar_score"].median()), 1),
    }
    return trades, summary

