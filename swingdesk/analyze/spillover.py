"""US -> India spillover engine.

The single biggest driver of an Indian swing trader's overnight risk is what
happened on Wall Street while they slept. US cash markets close ~02:00 IST,
*after* the Indian session has ended, so a given day's US move is already known
before India opens the next morning. That makes US returns a genuine *leading*
signal for the next Indian session — unlike most "predictors", the information
ordering is real, not look-ahead.

What this module computes (all with numpy/pandas only — no statsmodels):

  1. spillover_betas()     — how strongly each US index leads NIFTY (lagged OLS).
  2. next_day_outlook()    — given last night's US move, the expected next-day
                             NIFTY return and a 1-sigma band. Honest about R^2.
  3. stock_sensitivities() — per-stock beta to NIFTY (market), NASDAQ (overnight
                             tech), USD/INR and Brent — so you know WHAT moves it.
  4. regime()              — risk-on / neutral / risk-off read from India VIX,
                             NIFTY trend and the US trend.

Caveat we surface everywhere: single-day direction R^2 is low (markets are
mostly noise day to day). These outputs are for *positioning and risk*, not a
crystal ball. A high beta to NASDAQ tells you how exposed you are, not that the
trade will win.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from swingdesk.storage import load_macro, load_prices

# Macro tickers grouped by how they relate to an Indian stock's next session.
US_INDICES = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones"}
NIFTY = "^NSEI"
USDINR = "INR=X"
BRENT = "BZ=F"
INDIA_VIX = "^INDIAVIX"


# --------------------------------------------------------------------------- #
# numeric helpers
# --------------------------------------------------------------------------- #
def _ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple univariate OLS y = b0 + b1*x. Returns slope, intercept, r, r2,
    residual std and n. Robust to tiny samples (returns NaNs)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 15 or np.var(x) < 1e-18:
        return {"beta": float("nan"), "intercept": float("nan"),
                "r": float("nan"), "r2": float("nan"),
                "resid_std": float("nan"), "t_stat": float("nan"), "n": n}
    beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    intercept = float(np.mean(y) - beta * np.mean(x))
    r = float(np.corrcoef(x, y)[0, 1])
    resid = y - (intercept + beta * x)
    # ddof=2: two estimated params (slope + intercept)
    resid_std = float(np.std(resid, ddof=2)) if n > 2 else float("nan")
    # t-stat of the slope: beta / SE(beta), SE = resid_std / sqrt(Sxx).
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    se_beta = resid_std / np.sqrt(sxx) if sxx > 0 and np.isfinite(resid_std) else float("nan")
    t_stat = beta / se_beta if se_beta and np.isfinite(se_beta) and se_beta > 0 else float("nan")
    return {"beta": beta, "intercept": intercept, "r": r,
            "r2": r * r, "resid_std": resid_std, "t_stat": t_stat, "n": n}


def _returns(close: pd.Series) -> pd.Series:
    return close.astype(float).pct_change()


def _macro_returns(ticker: str, lookback: int | None = None) -> pd.Series:
    df = load_macro(ticker)
    if df.empty:
        return pd.Series(dtype=float)
    s = _returns(df["close"])
    return s.tail(lookback) if lookback else s


# --------------------------------------------------------------------------- #
# 1. how strongly US markets lead NIFTY
# --------------------------------------------------------------------------- #
def spillover_betas(target: str = NIFTY, lookback: int = 250) -> pd.DataFrame:
    """Lagged OLS of `target` daily returns on each US index's *prior-day*
    return (the overnight handoff). Higher beta/r2 = stronger spillover.

    Columns: factor, beta, r, r2, n, read.
    """
    tgt = _macro_returns(target)
    if tgt.empty:
        return pd.DataFrame()

    rows = []
    for tk, name in US_INDICES.items():
        us = _macro_returns(tk)
        if us.empty:
            continue
        # Align on common dates, then lag the US series by one session so we
        # regress today's NIFTY move on *yesterday's* (already-known) US move.
        joined = pd.concat([tgt, us.shift(1)], axis=1).dropna()
        joined = joined.tail(lookback)
        if len(joined) < 15:
            continue
        fit = _ols(joined.iloc[:, 1].values, joined.iloc[:, 0].values)
        sig = bool(np.isfinite(fit["t_stat"]) and abs(fit["t_stat"]) >= 2.0)
        rows.append({
            "factor": name,
            "beta": round(fit["beta"], 3),
            "r": round(fit["r"], 3),
            "r2": round(fit["r2"], 3),
            "t_stat": round(fit["t_stat"], 2) if np.isfinite(fit["t_stat"]) else None,
            "significant": sig,
            "n": fit["n"],
            "read": _spillover_read(fit["beta"], fit["r2"], sig),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


def _spillover_read(beta: float, r2: float, significant: bool = True) -> str:
    if not np.isfinite(beta) or not np.isfinite(r2):
        return "insufficient data"
    if not significant:
        return "not statistically significant — treat as noise (|t|<2)"
    strength = ("strong" if r2 >= 0.15 else "moderate" if r2 >= 0.05 else "weak")
    direction = "follows" if beta > 0 else "inverse to"
    return f"{strength} — NIFTY {direction} it (~{beta:.2f}x of the overnight move)"


# --------------------------------------------------------------------------- #
# 2. next-day outlook from last night's US close
# --------------------------------------------------------------------------- #
@dataclass
class NextDayOutlook:
    driver: str
    last_us_move_pct: float       # the overnight US move feeding the forecast
    expected_pct: float           # expected next-session NIFTY return
    low_pct: float                # expected_pct - 1 resid sigma
    high_pct: float               # expected_pct + 1 resid sigma
    r2: float
    n: int
    confidence: str               # low/moderate — single-day R^2 is always modest
    narrative: str


def next_day_outlook(lookback: int = 250) -> NextDayOutlook | None:
    """Project the next Indian session's NIFTY return from the most recent
    (already-closed) US move, using the strongest US leader.

    Deliberately conservative: we report a 1-sigma band, not a point bet, and
    label confidence by R^2 because day-ahead direction is mostly noise.
    """
    betas = spillover_betas(lookback=lookback)
    if betas.empty:
        return None
    best = betas.iloc[0]
    driver_tk = {v: k for k, v in US_INDICES.items()}[best["factor"]]

    tgt = _macro_returns(NIFTY)
    us = _macro_returns(driver_tk)
    joined = pd.concat([tgt, us.shift(1)], axis=1).dropna().tail(lookback)
    if len(joined) < 15:
        return None
    fit = _ols(joined.iloc[:, 1].values, joined.iloc[:, 0].values)

    last_us = float(us.dropna().iloc[-1]) if not us.dropna().empty else 0.0
    expected = fit["intercept"] + fit["beta"] * last_us
    sigma = fit["resid_std"] if np.isfinite(fit["resid_std"]) else 0.0

    significant = np.isfinite(fit["t_stat"]) and abs(fit["t_stat"]) >= 2.0
    conf = "moderate" if (fit["r2"] >= 0.10 and significant) else "low"
    exp_pct = expected * 100
    last_pct = last_us * 100
    band = sigma * 100
    direction = "higher" if expected > 0 else "lower"
    narrative = (
        f"{best['factor']} moved {last_pct:+.2f}% overnight. History says NIFTY "
        f"tends to open ~{exp_pct:+.2f}% the next session (±{band:.2f}% noise). "
        f"Bias: {direction}. Confidence {conf} (R²={fit['r2']:.2f}) — use for "
        f"gap/risk positioning, not as a standalone trade."
    )
    return NextDayOutlook(
        driver=best["factor"],
        last_us_move_pct=round(last_pct, 3),
        expected_pct=round(exp_pct, 3),
        low_pct=round(exp_pct - band, 3),
        high_pct=round(exp_pct + band, 3),
        r2=round(fit["r2"], 3),
        n=fit["n"],
        confidence=conf,
        narrative=narrative,
    )


# --------------------------------------------------------------------------- #
# 3. per-stock sensitivities
# --------------------------------------------------------------------------- #
def stock_sensitivities(ticker: str, lookback: int = 120) -> pd.DataFrame:
    """Beta of one stock's daily return to its key drivers.

    - NIFTY (same-day): classic market beta — how much index risk you carry.
    - NASDAQ (overnight, lagged): US-tech sensitivity — high for IT names.
    - USD/INR (same-day): rupee sensitivity — exporters +, importers -.
    - Brent (same-day): oil sensitivity — OMCs/airlines -, upstream +.

    Columns: driver, beta, r, lag, read — sorted by |r| (explanatory power).
    """
    px = load_prices(ticker)
    if px.empty or len(px) < 30:
        return pd.DataFrame()
    stock = _returns(px["close"]).tail(lookback)

    specs = [
        (NIFTY, "NIFTY 50", 0, "market"),
        ("^IXIC", "NASDAQ (overnight)", 1, "us_tech"),
        (USDINR, "USD/INR", 0, "rupee"),
        (BRENT, "Brent crude", 0, "oil"),
    ]
    rows = []
    for tk, name, lag, kind in specs:
        drv = _macro_returns(tk)
        if drv.empty:
            continue
        x = drv.shift(lag) if lag else drv
        joined = pd.concat([stock, x], axis=1).dropna().tail(lookback)
        if len(joined) < 20:
            continue
        fit = _ols(joined.iloc[:, 1].values, joined.iloc[:, 0].values)
        rows.append({
            "driver": name,
            "beta": round(fit["beta"], 3),
            "r": round(fit["r"], 3),
            "lag": lag,
            "read": _sensitivity_read(name, kind, fit["beta"], fit["r"]),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["_abs"] = out["r"].abs()
    return out.sort_values("_abs", ascending=False).drop(columns="_abs").reset_index(drop=True)


def _sensitivity_read(name: str, kind: str, beta: float, r: float) -> str:
    if not np.isfinite(beta):
        return "n/a"
    mag = abs(r)
    strength = "strongly" if mag >= 0.5 else "moderately" if mag >= 0.3 else "weakly"
    if kind == "market":
        return f"{strength} tracks the index (β={beta:.2f})"
    if kind == "us_tech":
        if mag < 0.2:
            return "little US-tech linkage"
        return f"{strength} moves with US tech overnight (β={beta:.2f})"
    if kind == "rupee":
        side = "gains when rupee weakens (exporter-like)" if beta > 0 else "hurt by rupee weakness (importer-like)"
        return f"{strength}: {side}"
    if kind == "oil":
        side = "rises with crude (upstream-like)" if beta > 0 else "pressured by higher crude (consumer-like)"
        return f"{strength}: {side}"
    return f"β={beta:.2f}"


# --------------------------------------------------------------------------- #
# 4. risk-on / risk-off regime
# --------------------------------------------------------------------------- #
@dataclass
class Regime:
    label: str           # "risk-on" | "neutral" | "risk-off"
    score: int           # -100..+100 (positive = risk-on)
    reasons: list[str]


def _trend_signal(close: pd.Series) -> tuple[int, str]:
    """+1 above both 50/200 EMAs, -1 below both, else 0. Plus a one-liner."""
    if close.empty or len(close) < 60:
        return 0, "insufficient history"
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1] if len(close) >= 200 else ema50
    last = float(close.iloc[-1])
    if last > ema50 and last > ema200:
        return 1, "above 50 & 200 EMA (uptrend)"
    if last < ema50 and last < ema200:
        return -1, "below 50 & 200 EMA (downtrend)"
    return 0, "between moving averages (chop)"


def regime() -> Regime | None:
    """Composite risk read for positioning. Drivers: NIFTY trend, US (S&P)
    trend, and India VIX level + 1-week change."""
    nifty = load_macro(NIFTY)
    spx = load_macro("^GSPC")
    vix = load_macro(INDIA_VIX, days=15)
    if nifty.empty and spx.empty:
        return None

    score = 0
    reasons = []

    nt, ntxt = _trend_signal(nifty["close"]) if not nifty.empty else (0, "no NIFTY data")
    score += nt * 30
    reasons.append(f"NIFTY: {ntxt}")

    ut, utxt = _trend_signal(spx["close"]) if not spx.empty else (0, "no S&P data")
    score += ut * 20
    reasons.append(f"S&P 500: {utxt}")

    if not vix.empty and len(vix) >= 6:
        lvl = float(vix["close"].iloc[-1])
        wk_ago = float(vix["close"].iloc[-6])
        chg = (lvl - wk_ago) / wk_ago * 100 if wk_ago else 0.0
        # India VIX: <13 calm, >20 fearful. Rising VIX = risk-off.
        if lvl < 13:
            score += 25; reasons.append(f"India VIX low ({lvl:.1f}) — calm")
        elif lvl > 20:
            score -= 35; reasons.append(f"India VIX elevated ({lvl:.1f}) — fear")
        else:
            reasons.append(f"India VIX moderate ({lvl:.1f})")
        if chg > 15:
            score -= 20; reasons.append(f"VIX spiking (+{chg:.0f}% w/w) — caution")
        elif chg < -15:
            score += 10; reasons.append(f"VIX cooling ({chg:.0f}% w/w)")

    score = int(max(-100, min(100, score)))
    label = "risk-on" if score >= 25 else "risk-off" if score <= -25 else "neutral"
    return Regime(label=label, score=score, reasons=reasons)
