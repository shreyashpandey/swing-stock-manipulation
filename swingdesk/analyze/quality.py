"""Compute a 0-100 fundamental quality score from yfinance metrics.

Scoring philosophy:
  - High ROE = efficient use of capital (most important single metric)
  - Strong profit margins = real economic earnings (not just revenue)
  - Earnings + revenue growth = compounding business
  - Reasonable valuation = pay sane prices (P/E vs typical thresholds)
  - Low debt = survivable in downturns

Each metric is scored independently in [0, 100], then weighted-averaged.
The asymmetry — generous to growth, harsh to red flags — matches how
discretionary investors actually look at companies.

Banks/NBFCs get the D/E component dropped (they're structurally leveraged).
"""
from __future__ import annotations


# Component weights (must sum to 1.0)
WEIGHTS = {
    "roe": 0.25,
    "growth": 0.20,        # avg of earnings + revenue growth
    "margins": 0.15,
    "valuation": 0.15,
    "debt": 0.15,
    "size": 0.10,          # liquidity proxy — bigger = better
}

FINANCIAL_SECTORS = {"Financial Services", "Financial", "Banks"}


def _clip(x: float, lo: float = 0, hi: float = 100) -> float:
    return max(lo, min(hi, x))


def _score_roe(roe: float | None) -> float | None:
    if roe is None:
        return None
    # Map: 5% → 20, 12% → 50, 18% → 75, 25% → 90, 30%+ → 100
    pct = roe * 100
    if pct <= 0:
        return 0
    if pct >= 30:
        return 100
    # Piecewise-linear interpolation
    breaks = [(0, 0), (5, 20), (12, 50), (18, 75), (25, 90), (30, 100)]
    for (x1, y1), (x2, y2) in zip(breaks, breaks[1:]):
        if x1 <= pct <= x2:
            return y1 + (y2 - y1) * (pct - x1) / (x2 - x1)
    return 100


def _score_growth(earnings_growth: float | None, revenue_growth: float | None) -> float | None:
    """Score on average of YoY earnings + revenue growth."""
    vals = [v for v in (earnings_growth, revenue_growth) if v is not None]
    if not vals:
        return None
    avg = sum(vals) / len(vals)
    pct = avg * 100
    if pct >= 25:
        return 100
    if pct <= -10:
        return 0
    # Map: -10% → 0, 0% → 30, 10% → 60, 15% → 75, 25% → 100
    breaks = [(-10, 0), (0, 30), (10, 60), (15, 75), (25, 100)]
    for (x1, y1), (x2, y2) in zip(breaks, breaks[1:]):
        if x1 <= pct <= x2:
            return y1 + (y2 - y1) * (pct - x1) / (x2 - x1)
    return 50


def _score_margins(profit_margin: float | None) -> float | None:
    if profit_margin is None:
        return None
    pct = profit_margin * 100
    if pct <= 0:
        return 0
    if pct >= 25:
        return 100
    # 5% → 30, 10% → 55, 15% → 75, 20% → 90
    breaks = [(0, 0), (5, 30), (10, 55), (15, 75), (20, 90), (25, 100)]
    for (x1, y1), (x2, y2) in zip(breaks, breaks[1:]):
        if x1 <= pct <= x2:
            return y1 + (y2 - y1) * (pct - x1) / (x2 - x1)
    return 50


def _score_valuation(pe: float | None) -> float | None:
    """Lower P/E is better, but not always. We penalise very high P/E but
    don't max-score ultra-low P/E (could be a value trap)."""
    if pe is None or pe <= 0:
        return None
    # Map: P/E 5 → 70, 15 → 90, 25 → 75, 40 → 50, 60 → 25, 100+ → 5
    if pe <= 5:
        return 70
    if pe <= 15:
        return 70 + (pe - 5) * 2  # → 90
    if pe <= 25:
        return 90 - (pe - 15) * 1.5  # → 75
    if pe <= 40:
        return 75 - (pe - 25) * (25 / 15)  # → 50
    if pe <= 60:
        return 50 - (pe - 40) * (25 / 20)  # → 25
    if pe <= 100:
        return 25 - (pe - 60) * (20 / 40)  # → 5
    return 5


def _score_debt(d_to_e: float | None, sector: str | None) -> float | None:
    """Score debt/equity, with a free pass for financials."""
    if sector in FINANCIAL_SECTORS:
        return None  # excluded from financials
    if d_to_e is None:
        return None
    if d_to_e <= 0.2:
        return 100
    if d_to_e <= 0.5:
        return 90 - (d_to_e - 0.2) * (10 / 0.3)
    if d_to_e <= 1.0:
        return 80 - (d_to_e - 0.5) * (30 / 0.5)
    if d_to_e <= 2.0:
        return 50 - (d_to_e - 1.0) * (30 / 1.0)
    if d_to_e <= 3.0:
        return 20 - (d_to_e - 2.0) * (15 / 1.0)
    return 5


def _score_size(market_cap: float | None) -> float | None:
    """Market cap as a liquidity / business-maturity proxy."""
    if market_cap is None:
        return None
    cr = market_cap / 1e7  # convert to ₹ crores
    if cr < 500:
        return 20       # small-cap, illiquid
    if cr < 5000:
        return 50       # small-mid
    if cr < 20000:
        return 70       # mid-cap
    if cr < 100000:
        return 85       # large-cap
    return 95           # mega-cap


def score(f: dict) -> float | None:
    """Compute the composite quality score for one fundamentals row.
    Returns None if too many key metrics are missing."""
    components = {
        "roe":       _score_roe(f.get("return_on_equity")),
        "growth":    _score_growth(f.get("earnings_growth"), f.get("revenue_growth")),
        "margins":   _score_margins(f.get("profit_margin")),
        "valuation": _score_valuation(f.get("trailing_pe")),
        "debt":      _score_debt(f.get("debt_to_equity"), f.get("sector")),
        "size":      _score_size(f.get("market_cap")),
    }
    # Drop None components and re-weight the available ones
    available = {k: (v, WEIGHTS[k]) for k, v in components.items() if v is not None}
    if len(available) < 3:
        return None
    total_w = sum(w for _, w in available.values())
    weighted = sum(v * w for v, w in available.values())
    return round(weighted / total_w, 1)


def passes_quality_bar(f: dict, min_score: float = 60.0,
                      hard_filters: bool = True) -> tuple[bool, list[str]]:
    """Apply both the composite-score threshold AND hard guardrails.
    Returns (passes, reasons_failed)."""
    fails = []
    s = score(f)
    if s is None:
        fails.append("insufficient data")
    elif s < min_score:
        fails.append(f"quality {s} < {min_score}")
    if hard_filters:
        roe = f.get("return_on_equity")
        if roe is not None and roe < 0.05:
            fails.append("ROE < 5%")
        pm = f.get("profit_margin")
        if pm is not None and pm < 0.02:
            fails.append("profit margin < 2%")
        mc = f.get("market_cap")
        if mc is not None and mc < 5e9:
            fails.append("market cap < ₹500 cr")
    return (len(fails) == 0, fails)
