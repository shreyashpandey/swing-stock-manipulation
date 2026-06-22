"""Cross-sectional factor ranking — the honest version of "which stock will go
up".

Instead of forecasting one stock in isolation (hard, low-signal), we rank the
WHOLE universe on factors that have decades of out-of-sample evidence behind
them, then favour the top of the ranking. This is the core of how quant shops
like AQR actually invest — no black box, fully transparent.

Factors (each z-scored across the universe, then weighted):

  momentum  — 6-month return skipping the last month (Jegadeesh-Titman; the
              skip avoids short-term reversal). Winners keep winning.
  low_vol   — inverse of recent realized volatility. Low-vol stocks have
              historically delivered better risk-adjusted returns (the
              low-volatility anomaly).
  quality   — high ROE + high margins + low leverage (from fundamentals).
  value     — cheap on trailing P/E.
  trend     — price above its 200-day EMA (regime/participation filter).

composite = weighted sum of the available factor z-scores. Stocks are ranked
and bucketed into quintiles; quintile 1 = most attractive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk.storage import get_watchlist, load_fundamentals, load_prices

DEFAULT_WEIGHTS = {
    "momentum": 0.35,
    "low_vol": 0.15,
    "quality": 0.25,
    "value": 0.15,
    "trend": 0.10,
}


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score, NaN-safe. Zero std -> all zeros."""
    s = s.astype(float)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).clip(-3, 3)        # winsorize extreme z at ±3


def _momentum(close: pd.Series, lookback: int = 126, skip: int = 21) -> float:
    if len(close) < lookback + skip + 1:
        return float("nan")
    recent = float(close.iloc[-skip - 1])
    past = float(close.iloc[-lookback - skip - 1])
    return recent / past - 1 if past > 0 else float("nan")


def _realized_vol(close: pd.Series, window: int = 60) -> float:
    r = close.pct_change().dropna().tail(window)
    if len(r) < 20:
        return float("nan")
    return float(r.std(ddof=1))


def _trend(close: pd.Series) -> float:
    if len(close) < 200:
        return float("nan")
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
    return float(close.iloc[-1]) / float(ema200) - 1 if ema200 > 0 else float("nan")


def _raw_factors(tickers: list[str], fund: pd.DataFrame) -> pd.DataFrame:
    """Collect raw (pre-z-score) factor inputs for each ticker."""
    fmap = {r["ticker"]: r for r in fund.to_dict("records")} if not fund.empty else {}
    rows = []
    for t in tickers:
        px = load_prices(t)
        if px.empty or len(px) < 60:
            continue
        close = px["close"].astype(float)
        f = fmap.get(t, {})
        pe = f.get("trailing_pe")
        # Carry quality sub-components RAW and on their native scales — they are
        # z-scored *separately* later, so debt (scale ~0-300) can't swamp ROE
        # (scale ~0-0.4) the way a naive sum would.
        rows.append({
            "ticker": t,
            "momentum": _momentum(close),
            "low_vol": -_realized_vol(close),         # negate: low vol -> high score
            "trend": _trend(close),
            "roe": f.get("return_on_equity"),
            "margin": f.get("profit_margin"),
            "d2e": f.get("debt_to_equity"),
            "value": -pe if (pe is not None and pe > 0) else np.nan,  # cheap -> high
        })
    return pd.DataFrame(rows)


def factor_table(tickers: list[str] | None = None,
                 weights: dict | None = None) -> pd.DataFrame:
    """Rank a universe by composite factor score.

    Returns one row per ticker with each factor's z-score, the weighted
    composite, an integer rank (1 = best) and a quintile bucket. Factors with
    no data for a stock simply don't contribute (weights renormalised per row).
    """
    tickers = tickers if tickers is not None else get_watchlist()
    if not tickers:
        return pd.DataFrame()
    weights = weights or DEFAULT_WEIGHTS
    raw = _raw_factors(tickers, load_fundamentals()).set_index("ticker")
    if raw.empty or len(raw) < 3:
        return pd.DataFrame()

    # --- Build the five factor z-scores. Quality is the mean of its z-scored
    # sub-components (ROE, margin, -debt) so each contributes equally regardless
    # of native scale; debt enters negatively (less leverage = higher quality).
    quality_parts = pd.concat([
        _zscore(raw["roe"]),
        _zscore(raw["margin"]),
        -_zscore(raw["d2e"]),
    ], axis=1)
    # Only average where the underlying value actually existed (not z=0 fill).
    quality_present = raw[["roe", "margin", "d2e"]].notna()
    quality_z = quality_parts.where(quality_present.values).mean(axis=1)

    z = pd.DataFrame({
        "momentum_z": _zscore(raw["momentum"]),
        "low_vol_z": _zscore(raw["low_vol"]),
        "quality_z": quality_z,
        "value_z": _zscore(raw["value"].where(raw["value"].notna())),
        "trend_z": _zscore(raw["trend"]),
    })

    # Per-row weighted average over the factors that are present, renormalising
    # weights so a missing fundamental doesn't zero a stock out.
    factor_cols = list(weights.keys())
    w = pd.Series(weights)
    present = pd.DataFrame({
        "momentum": raw["momentum"].notna(),
        "low_vol": raw["low_vol"].notna(),
        "quality": quality_present.any(axis=1),
        "value": raw["value"].notna(),
        "trend": raw["trend"].notna(),
    })[factor_cols]
    z = z[[f"{c}_z" for c in factor_cols]]      # align column order to weights
    wmat = present.mul(w, axis=1)
    wsum = wmat.sum(axis=1).replace(0, np.nan)
    composite = (z.fillna(0).values * wmat.values).sum(axis=1) / wsum.values

    out = z.copy()
    out["composite"] = np.round(composite, 3)
    out = out.reset_index().dropna(subset=["composite"])
    out = out.sort_values("composite", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    out["quintile"] = pd.qcut(out["composite"].rank(method="first"),
                              q=min(5, len(out)), labels=False)
    # qcut labels 0..4 ascending; flip so quintile 1 = best.
    nq = out["quintile"].max() + 1
    out["quintile"] = (nq - out["quintile"]).astype(int)
    return out
