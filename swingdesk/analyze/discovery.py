"""Discovery scanner — find tradeable opportunities OUTSIDE the current watchlist.

We maintain a curated universe of ~150 NSE names spanning Nifty 100 +
Nifty Next 50 + quality mid-caps + thematic plays (defense, capital goods,
PSU, IT, banks, autos, FMCG, healthcare, real estate).

For each ticker we score:
  - quality (fundamentals)
  - trend (above 50/200 EMA, RSI not extended)
  - momentum (rate of change, volume confirmation)
  - active setup (any of the 4 detectors firing today)

The output is a ranked list of "investable now" stocks that the user
DOESN'T already own and that AREN'T in their watchlist — i.e. truly new
ideas. Costs nothing extra at runtime since all data is local.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from rich.console import Console

from swingdesk.analyze.setups import scan_ticker
from swingdesk.analyze.technicals import add_indicators, trend_quality
from swingdesk.storage import (
    get_fundamentals,
    get_watchlist,
    holdings_tickers,
    load_prices,
)

console = Console()

# Curated discovery universe — ~140 NSE names beyond pure Nifty-50.
# Organised by sector for context; the scanner treats them as one pool.
DISCOVERY_UNIVERSE: list[str] = [
    # IT services
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "OFSS.NS",
    "KPITTECH.NS", "TATATECH.NS",
    # Banks / financials
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "AUBANK.NS",
    "IDFCFIRSTB.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "FEDERALBNK.NS",
    "BAJAJHLDNG.NS", "HDFCAMC.NS", "POLICYBZR.NS", "MCX.NS", "BSE.NS",
    "CDSL.NS", "KFINTECH.NS",
    # Insurance
    "SBILIFE.NS", "HDFCLIFE.NS", "ICICIPRULI.NS", "ICICIGI.NS", "MAXFIN.NS",
    # Auto / auto ancillary
    "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "EICHERMOT.NS", "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BHARATFORG.NS", "MOTHERSON.NS",
    "BOSCHLTD.NS", "TIINDIA.NS",
    # Capital goods / defense / industrials
    "LT.NS", "SIEMENS.NS", "ABB.NS", "CUMMINSIND.NS", "THERMAX.NS",
    "BEL.NS", "HAL.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "BEML.NS",
    "CGPOWER.NS", "POLYCAB.NS", "HAVELLS.NS", "DIXON.NS", "KAYNES.NS",
    "AMBER.NS", "PRAJIND.NS",
    # PSU / power / utilities
    "ONGC.NS", "COALINDIA.NS", "NTPC.NS", "POWERGRID.NS", "BPCL.NS",
    "IOC.NS", "HINDPETRO.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "JSWENERGY.NS",
    "NHPC.NS", "SJVN.NS", "RECLTD.NS", "PFC.NS", "IREDA.NS",
    "IRCTC.NS", "IRFC.NS", "RVNL.NS", "CONCOR.NS",
    # FMCG / consumer
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
    "MARICO.NS", "COLPAL.NS", "GODREJCP.NS", "VBL.NS", "TATACONSUM.NS",
    "EMAMI.NS", "BAJAJCON.NS",
    # Healthcare / pharma
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    "MAXHEALTH.NS", "FORTIS.NS", "LUPIN.NS", "ALKEM.NS", "TORNTPHARM.NS",
    "GLAND.NS", "ZYDUSLIFE.NS", "AUROPHARMA.NS",
    # Metals / commodities
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "JINDALSTEL.NS",
    # Cement / building materials
    "ULTRACEMCO.NS", "GRASIM.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS",
    # Real estate
    "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "PRESTIGE.NS",
    # Specialty chemicals / paints
    "ASIANPAINT.NS", "PIDILITIND.NS", "SRF.NS", "ATUL.NS", "AARTIIND.NS",
    "NAVINFLUOR.NS", "BERGEPAINT.NS",
    # Retail / consumer discretionary
    "TRENT.NS", "TITAN.NS", "ABFRL.NS", "DMART.NS", "JUBLFOOD.NS",
    # Adani group + reliance + telcos
    "ADANIENT.NS", "ADANIPORTS.NS", "RELIANCE.NS", "BHARTIARTL.NS",
    # Misc strong
    "LICI.NS", "LICHSGFIN.NS",
]


@dataclass
class Opportunity:
    ticker: str
    company: str
    sector: str | None
    price: float
    quality_score: float | None
    technical_state: str          # "uptrend" | "weakening" | "broken" | "unknown"
    trend_verdict: str | None = None    # "real" | "weak" | "false" | "no_uptrend"
    trend_label: str | None = None      # human-readable
    rsi: float | None = None
    above_50ema: bool | None = None
    above_200ema: bool | None = None
    momentum_20d_pct: float | None = None
    volume_x_avg: float | None = None
    active_setup: str | None = None
    composite_score: float = 0.0
    conviction: str = "low"             # "high" | "medium" | "low" — alignment score
    reasons: list[str] = field(default_factory=list)


def _state_score(state: str) -> float:
    return {"uptrend": 30, "weakening": 10, "broken": 0, "unknown": 5}.get(state, 5)


def _rank_one(ticker: str) -> Opportunity | None:
    df = load_prices(ticker)
    if df.empty or len(df) < 60:
        return None
    df = add_indicators(df)
    last = df.iloc[-1]
    close = float(last["close"])

    # Trend state
    above_50 = bool(close > last["ema50"]) if pd.notna(last.get("ema50")) else None
    above_200 = bool(close > last["ema200"]) if pd.notna(last.get("ema200")) else None
    if above_50 and above_200:
        state = "uptrend"
    elif above_200:
        state = "weakening"
    elif above_50 is False and above_200 is False:
        state = "broken"
    else:
        state = "unknown"

    rsi = float(last["rsi14"]) if pd.notna(last.get("rsi14")) else None
    momentum = ((close - df["close"].iloc[-20]) / df["close"].iloc[-20] * 100
                if len(df) >= 20 else None)
    vol_x = ((float(last["volume"]) / float(last["vol_avg20"]))
             if pd.notna(last.get("vol_avg20")) else None)

    fund = get_fundamentals(ticker) or {}
    qscore = fund.get("quality_score")

    # Fresh setup firing today?
    sigs = scan_ticker(ticker)
    setup = sigs[0]["setup"] if sigs else None

    # Trend-quality read: is the price advance volume-confirmed?
    tq = trend_quality(df) or {}
    trend_verdict = tq.get("verdict")        # "real" | "weak" | "false" | "no_uptrend"
    trend_label = tq.get("label")

    # --- Composite (0-100), higher = better fresh investment idea
    score = 0.0
    reasons: list[str] = []

    score += _state_score(state)
    if state == "uptrend":
        reasons.append("uptrend (above 50+200 EMA)")

    # Volume-confirmed trend matters a LOT — boost real, penalise false hard
    if trend_verdict == "real":
        score += 20
        reasons.append(f"✓ {trend_label}")
    elif trend_verdict == "weak":
        score += 5
        reasons.append(f"⚠ {trend_label}")
    elif trend_verdict == "false":
        score -= 20
        reasons.append(f"✗ {trend_label}")
    # "no_uptrend" gets no adjustment — already reflected in state_score

    if qscore is not None:
        score += min(35, qscore * 0.35)
        if qscore >= 75:
            reasons.append(f"strong fundamentals (Q={qscore:.0f})")
        elif qscore < 50:
            reasons.append(f"weak fundamentals (Q={qscore:.0f})")

    if momentum is not None:
        if momentum > 5:
            score += 10
            reasons.append(f"+{momentum:.1f}% in 20d")
        elif momentum < -10:
            score -= 5

    if rsi is not None:
        if 50 <= rsi <= 65:
            score += 8
        elif rsi > 75:
            score -= 5
            reasons.append(f"overbought (RSI {rsi:.0f})")

    if vol_x and vol_x > 1.5:
        score += 5
        reasons.append(f"{vol_x:.1f}x avg volume today")

    if setup:
        score += 15
        reasons.append(f"signal firing: {setup}")

    # --- Conviction: how many of the 4 high-quality criteria align?
    # Required all 4 for HIGH:
    #   quality ≥ 70, state="uptrend", trend_verdict="real", momentum > 0
    aligned = sum([
        bool(qscore and qscore >= 70),
        state == "uptrend",
        trend_verdict == "real",
        bool(momentum and momentum > 0),
        bool(rsi and 50 <= rsi <= 70),
    ])
    if aligned >= 4:
        conviction = "high"
    elif aligned >= 3:
        conviction = "medium"
    else:
        conviction = "low"

    return Opportunity(
        ticker=ticker,
        company=fund.get("short_name") or ticker.replace(".NS", ""),
        sector=fund.get("sector"),
        price=round(close, 2),
        quality_score=qscore,
        technical_state=state,
        trend_verdict=trend_verdict,
        trend_label=trend_label,
        rsi=round(rsi, 1) if rsi else None,
        above_50ema=above_50,
        above_200ema=above_200,
        momentum_20d_pct=round(momentum, 2) if momentum is not None else None,
        volume_x_avg=round(vol_x, 2) if vol_x else None,
        active_setup=setup,
        composite_score=round(max(0, min(100, score)), 1),
        conviction=conviction,
        reasons=reasons[:5],
    )


def scan(universe: list[str] | None = None,
         exclude_held: bool = True,
         exclude_watchlist: bool = True) -> list[Opportunity]:
    """Rank the discovery universe by investability composite score."""
    pool = universe or DISCOVERY_UNIVERSE
    held = set(holdings_tickers()) if exclude_held else set()
    wl = set(get_watchlist()) if exclude_watchlist else set()
    skip = held | wl

    results: list[Opportunity] = []
    for tk in pool:
        if tk in skip:
            continue
        opp = _rank_one(tk)
        if opp:
            results.append(opp)

    results.sort(key=lambda o: o.composite_score, reverse=True)
    return results


def discovery_universe() -> list[str]:
    """Public accessor for the curated discovery list."""
    return list(DISCOVERY_UNIVERSE)


def high_conviction(opps: list[Opportunity] | None = None,
                    min_score: float = 70.0) -> list[Opportunity]:
    """Return only the names where MULTIPLE lenses align cleanly.

    These are stocks where:
      - Quality ≥ 70 (solid business)
      - Currently in uptrend (above both 50 + 200 EMA)
      - Trend is volume-confirmed (real, not distribution)
      - Positive 20-day momentum
      - Composite rank score ≥ `min_score`

    Designed to be a "invest without thinking too hard" list — when everything
    aligns, the prior probability is meaningfully better. Still requires you
    to size and time the entry sensibly.
    """
    pool = opps if opps is not None else scan()
    return [o for o in pool
            if o.conviction == "high" and o.composite_score >= min_score]
