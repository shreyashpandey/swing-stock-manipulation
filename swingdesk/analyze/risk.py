"""Risk & position-sizing layer.

The reason most retail portfolios underperform is rarely bad stock picks — it's
sizing and hidden concentration. Three concrete tools here:

  1. position_size()        — classic fixed-risk sizing: never lose more than
                              `risk_pct` of capital if the stop is hit.
  2. vol_target_size()      — size so every position contributes roughly the
                              SAME daily rupee volatility, so one wild small-cap
                              doesn't dominate your P&L (what risk-parity desks do).
  3. portfolio diagnostics  — concentration (HHI + top weights), correlated
                              clusters (are your "8 stocks" really 1 Nasdaq bet?),
                              and a composite risk report tied to the macro regime.

All numpy/pandas; reads live holdings + prices from storage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swingdesk.analyze.expected_range import ewma_vol
from swingdesk.config import ACCOUNT_CAPITAL, RISK_PER_TRADE_PCT
from swingdesk.storage import load_fundamentals, load_holdings, load_prices


# --------------------------------------------------------------------------- #
# 1. fixed-risk position sizing
# --------------------------------------------------------------------------- #
@dataclass
class PositionSize:
    shares: int
    risk_amount: float            # rupee loss if stop hit
    position_value: float
    pct_of_capital: float
    risk_per_share: float
    note: str


def position_size(entry: float, stoploss: float,
                  capital: float = ACCOUNT_CAPITAL,
                  risk_pct: float = RISK_PER_TRADE_PCT,
                  max_position_pct: float = 25.0) -> PositionSize | None:
    """Shares to buy so a stop-out costs exactly `risk_pct`% of capital.

    Caps the position at `max_position_pct`% of capital regardless (so a tight
    stop doesn't blow the whole account into one name)."""
    if entry <= 0 or stoploss <= 0 or entry <= stoploss:
        return None
    risk_per_share = entry - stoploss
    risk_budget = capital * risk_pct / 100.0
    shares = int(risk_budget // risk_per_share)
    note = "sized by stop distance"

    # Concentration cap
    max_value = capital * max_position_pct / 100.0
    if shares * entry > max_value:
        shares = int(max_value // entry)
        note = f"capped at {max_position_pct:.0f}% of capital"
    if shares <= 0:
        return PositionSize(0, 0.0, 0.0, 0.0, round(risk_per_share, 2),
                            "stop too wide for risk budget — skip or widen capital")
    pos_value = shares * entry
    return PositionSize(
        shares=shares,
        risk_amount=round(shares * risk_per_share, 2),
        position_value=round(pos_value, 2),
        pct_of_capital=round(pos_value / capital * 100, 1),
        risk_per_share=round(risk_per_share, 2),
        note=note,
    )


# --------------------------------------------------------------------------- #
# 2. volatility-target sizing
# --------------------------------------------------------------------------- #
def vol_target_size(ticker: str, capital: float = ACCOUNT_CAPITAL,
                    target_daily_vol_pct: float = 0.4,
                    lookback: int = 120) -> dict | None:
    """Size so the position's expected daily move ≈ target_daily_vol_pct% of
    capital. Calm stocks get a bigger allocation, jumpy ones smaller — equal
    *risk* contribution rather than equal rupees."""
    px = load_prices(ticker, days=lookback + 5)
    if px.empty or len(px) < 30:
        return None
    close = px["close"].astype(float)
    spot = float(close.iloc[-1])
    dvol = ewma_vol(close.pct_change().dropna().tail(lookback))
    if not np.isfinite(dvol) or dvol <= 0:
        return None
    target_rupees = capital * target_daily_vol_pct / 100.0
    shares = int(target_rupees // (dvol * spot))
    pos_value = shares * spot
    return {
        "ticker": ticker,
        "shares": shares,
        "spot": round(spot, 2),
        "daily_vol_pct": round(dvol * 100, 2),
        "position_value": round(pos_value, 2),
        "pct_of_capital": round(pos_value / capital * 100, 1),
        "expected_daily_pnl": round(pos_value * dvol, 2),
    }


# --------------------------------------------------------------------------- #
# 3. portfolio concentration
# --------------------------------------------------------------------------- #
@dataclass
class Concentration:
    n_holdings: int
    total_value: float
    top_name: str
    top_weight_pct: float
    top3_weight_pct: float
    hhi: float                     # Herfindahl index, 0..1 (1 = all in one name)
    effective_names: float         # 1/HHI — how many "real" independent bets
    flags: list[str] = field(default_factory=list)


def concentration(holdings: pd.DataFrame | None = None) -> Concentration | None:
    """Concentration read on current holdings by market value."""
    h = holdings if holdings is not None else load_holdings()
    if h.empty or "current_value" not in h.columns:
        return None
    h = h.dropna(subset=["current_value"])
    h = h[h["current_value"] > 0]
    if h.empty:
        return None
    total = float(h["current_value"].sum())
    w = (h["current_value"] / total).sort_values(ascending=False)
    hhi = float((w ** 2).sum())
    top3 = float(w.head(3).sum())
    flags = []
    if w.iloc[0] > 0.25:
        flags.append(f"{h.loc[w.index[0], 'ticker']} "
                     f"is {w.iloc[0]*100:.0f}% of the book — single-name risk")
    if hhi > 0.20:
        flags.append(f"HHI {hhi:.2f} — portfolio is concentrated (≈{1/hhi:.1f} real bets)")
    if top3 > 0.60:
        flags.append(f"top 3 names = {top3*100:.0f}% of capital")
    return Concentration(
        n_holdings=len(h),
        total_value=round(total, 2),
        top_name=str(h.loc[w.index[0], "ticker"]),
        top_weight_pct=round(float(w.iloc[0]) * 100, 1),
        top3_weight_pct=round(top3 * 100, 1),
        hhi=round(hhi, 3),
        effective_names=round(1 / hhi, 1) if hhi > 0 else float("nan"),
        flags=flags,
    )


# --------------------------------------------------------------------------- #
# 4. correlation between holdings
# --------------------------------------------------------------------------- #
def _returns_matrix(tickers: list[str], lookback: int = 90) -> pd.DataFrame:
    series = {}
    for t in tickers:
        px = load_prices(t, days=lookback + 5)
        if px.empty or len(px) < 30:
            continue
        series[t] = px["close"].astype(float).pct_change()
    if len(series) < 2:
        return pd.DataFrame()
    return pd.DataFrame(series).dropna(how="all").tail(lookback)


def portfolio_correlation(tickers: list[str], lookback: int = 90) -> pd.DataFrame:
    """Pairwise return-correlation matrix of the given tickers."""
    rets = _returns_matrix(tickers, lookback)
    if rets.empty:
        return pd.DataFrame()
    return rets.corr().round(2)


def correlated_pairs(tickers: list[str], threshold: float = 0.7,
                     lookback: int = 90) -> pd.DataFrame:
    """Highly-correlated holding pairs — hidden concentration. These move
    together, so holding both is closer to a double bet than diversification."""
    corr = portfolio_correlation(tickers, lookback)
    if corr.empty:
        return pd.DataFrame()
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({"a": cols[i], "b": cols[j], "corr": float(r)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("corr", ascending=False, key=abs).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. sector concentration
# --------------------------------------------------------------------------- #
def sector_concentration(holdings: pd.DataFrame | None = None,
                         flag_pct: float = 40.0) -> pd.DataFrame:
    """Portfolio weight by sector (from fundamentals). The classic hidden bet
    is 'I hold 8 stocks' that are really 5 IT names = one Nasdaq trade. Returns
    sector, weight_pct, n_names, flagged — sorted by weight."""
    h = holdings if holdings is not None else load_holdings()
    if h.empty or "current_value" not in h.columns:
        return pd.DataFrame()
    h = h.dropna(subset=["current_value"])
    h = h[h["current_value"] > 0]
    if h.empty:
        return pd.DataFrame()
    fund = load_fundamentals()
    sector_map = ({r["ticker"]: (r.get("sector") or "Unknown")
                   for r in fund.to_dict("records")} if not fund.empty else {})
    h = h.copy()
    h["sector"] = h["ticker"].map(sector_map).fillna("Unknown")
    total = float(h["current_value"].sum())
    grp = h.groupby("sector").agg(
        value=("current_value", "sum"), n_names=("ticker", "nunique")).reset_index()
    grp["weight_pct"] = (grp["value"] / total * 100).round(1)
    grp["flagged"] = grp["weight_pct"] >= flag_pct
    return grp[["sector", "weight_pct", "n_names", "flagged"]].sort_values(
        "weight_pct", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6. composite portfolio risk report
# --------------------------------------------------------------------------- #
def portfolio_risk_report(lookback: int = 90, corr_threshold: float = 0.7) -> dict:
    """One-call summary: concentration + correlated clusters + suggestions.
    Ties the diagnostics together for the dashboard."""
    h = load_holdings()
    if h.empty:
        return {"ok": False, "msg": "No holdings imported yet."}
    tickers = [t for t in h["ticker"].dropna().unique().tolist()]
    conc = concentration(h)
    pairs = correlated_pairs(tickers, threshold=corr_threshold, lookback=lookback)
    sectors = sector_concentration(h)

    suggestions = []
    if conc and conc.flags:
        suggestions.extend(conc.flags)
    if not sectors.empty:
        for _, row in sectors[sectors["flagged"]].iterrows():
            suggestions.append(
                f"{row['weight_pct']:.0f}% of the book is in {row['sector']} "
                f"({int(row['n_names'])} name(s)) — sector-concentration risk."
            )
    if not pairs.empty:
        ex = pairs.iloc[0]
        suggestions.append(
            f"{ex['a']} & {ex['b']} move together (r={ex['corr']:.2f}) — "
            f"{len(pairs)} highly-correlated pair(s); trim to truly diversify."
        )
    if conc and conc.effective_names < 4:
        suggestions.append(
            f"Only ~{conc.effective_names:.1f} independent bets — add uncorrelated "
            f"names or sectors to lower portfolio swings."
        )
    if not suggestions:
        suggestions.append("Concentration and correlation look healthy.")
    return {
        "ok": True,
        "concentration": conc,
        "sectors": sectors,
        "correlated_pairs": pairs,
        "suggestions": suggestions,
    }
