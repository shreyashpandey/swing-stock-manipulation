"""Execution algorithms — the 'algo' in algo trading, retail edition.

The bulge-bracket execution desks (JPMorgan, Morgan Stanley) don't *pick* stocks
with their algos — they pick stocks elsewhere and use algos to get IN and OUT of
a position without moving the price against themselves. Trading a big order all
at once pays **market impact**; trading it too slowly pays **timing risk** (the
price drifts away while you wait). An execution algo slices a parent order into
child orders over time to balance the two. That logic is pure math on data we
already have — no exchange membership, no colocation, no Jane-Street latency.

You trade manually on Groww, so the output here is an **order plan you place by
hand**: a slicing schedule (when, how many shares, what % of expected volume),
an estimate of what it should cost, and plain-English steps. Four standard algos:

  • TWAP  — equal slices over a time window. Dumb, predictable, ignores volume.
  • VWAP  — slice along the day's volume curve (trade more when the market does),
            so your average price tracks the volume-weighted average. The desk
            default for 'just get it done at a fair price'.
  • POV   — Percent-Of-Volume: trade a fixed % of each bucket's expected volume.
            Adapts to how much is actually trading; may not finish in a day.
  • IS    — Implementation Shortfall: front-load the schedule to cut timing risk
            when you're in a hurry (urgency dialled by `risk_aversion`).

Impact is estimated with the industry-standard **square-root law**
(impact ≈ η·σ·√participation) and cross-checked against the **Amihud** impact we
already compute in liquidity.py. Everything is a *model estimate*, clearly
labelled — not a promise of fill price.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swingdesk.analyze import liquidity as liq_mod
from swingdesk.analyze.expected_range import daily_vol
from swingdesk.config import ACCOUNT_CAPITAL
from swingdesk.storage import load_intraday, load_prices

# NSE continuous session: 09:15 – 15:30 IST = 375 minutes.
SESSION_OPEN_MIN = 9 * 60 + 15
SESSION_CLOSE_MIN = 15 * 60 + 30
SESSION_MINUTES = SESSION_CLOSE_MIN - SESSION_OPEN_MIN

# Square-root impact-model calibration. η is dimensionless (~0.5–1 empirically);
# half-spread is the round-trip cost we can't see without an order book.
IMPACT_ETA = 0.8
HALF_SPREAD_BPS = 3.0

ALGOS = ("vwap", "twap", "pov", "is")


@dataclass
class ExecutionPlan:
    ticker: str
    side: str                      # "buy" | "sell"
    algo: str
    qty: int
    arrival_price: float           # spot at decision time — the IS benchmark
    notional: float                # qty * arrival_price
    schedule: pd.DataFrame         # one row per time bucket
    horizon_minutes: int           # how long the schedule spans
    avg_participation_pct: float   # order qty ÷ expected volume over the horizon
    completes_in_session: bool
    unfilled_shares: int
    # cost model (all in basis points unless noted)
    est_cost_bps: float            # expected cost vs arrival = spread + impact
    impact_bps: float
    spread_bps: float
    timing_risk_bps: float         # 1σ price drift over the horizon (a *risk*, not a cost)
    amihud_bps: float              # independent cross-check from liquidity.py
    est_cost_rupees: float
    # context
    adv_value_cr: float
    adv_volume: float
    liquidity_tier: str
    daily_vol_pct: float
    used_fallback_curve: bool
    summary: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Intraday volume curve — the backbone of VWAP / POV scheduling
# --------------------------------------------------------------------------- #
def _session_buckets(bucket_minutes: int) -> list[tuple[str, int, int, float]]:
    """List of (label, start_min, end_min, center_fraction) tiling the session."""
    out = []
    m = SESSION_OPEN_MIN
    while m < SESSION_CLOSE_MIN:
        end = min(m + bucket_minutes, SESSION_CLOSE_MIN)
        label = f"{m // 60:02d}:{m % 60:02d}"
        center = ((m + end) / 2 - SESSION_OPEN_MIN) / SESSION_MINUTES
        out.append((label, m, end, center))
        m = end
    return out


def _fallback_curve(buckets: list[tuple[str, int, int, float]]) -> pd.Series:
    """Generic U-shaped intraday volume profile (heavy at open & close) for names
    with no intraday history. Clearly an approximation, but better than uniform."""
    w = np.array([1.0 + 1.6 * np.exp(-c / 0.15) + 1.1 * np.exp(-(1 - c) / 0.12)
                  for _, _, _, c in buckets])
    w = w / w.sum()
    return pd.Series(w, index=[b[0] for b in buckets])


def intraday_volume_curve(ticker: str, interval: str = "5m",
                          bucket_minutes: int = 30,
                          lookback_days: int = 20) -> tuple[pd.Series, bool]:
    """Average share of daily volume traded in each intraday bucket (sums to 1).

    Returns (curve, used_fallback). Falls back to a U-shape if there's no
    intraday history for the name."""
    buckets = _session_buckets(bucket_minutes)
    labels = [b[0] for b in buckets]
    df = load_intraday(ticker, interval=interval, days=lookback_days)
    if df is None or df.empty or "volume" not in df.columns:
        return _fallback_curve(buckets), True

    df = df.copy()
    mod = df.index.hour * 60 + df.index.minute
    bidx = np.clip((mod - SESSION_OPEN_MIN) // bucket_minutes, 0, len(buckets) - 1)
    grp = pd.DataFrame({"_day": df.index.normalize(), "_b": bidx,
                        "volume": df["volume"].astype(float).values})
    bucket_vol = grp.groupby(["_day", "_b"])["volume"].sum()
    day_tot = grp.groupby("_day")["volume"].sum().replace(0, np.nan)
    frac = bucket_vol / day_tot.reindex(bucket_vol.index.get_level_values("_day")).values
    avg = frac.groupby(level="_b").mean()

    curve = pd.Series(0.0, index=range(len(buckets)))
    curve.update(avg)
    if not np.isfinite(curve.sum()) or curve.sum() <= 0:
        return _fallback_curve(buckets), True
    curve = curve / curve.sum()
    return pd.Series(curve.values, index=labels), False


# --------------------------------------------------------------------------- #
# Schedule builders
# --------------------------------------------------------------------------- #
def _round_to_total(weighted: np.ndarray, total: int) -> np.ndarray:
    """Round a float allocation to integers that sum exactly to `total`
    (largest-remainder method)."""
    floored = np.floor(weighted).astype(int)
    floored = np.clip(floored, 0, None)
    remainder = int(total - floored.sum())
    if remainder > 0:
        order = np.argsort(weighted - np.floor(weighted))[::-1]
        for i in range(remainder):
            floored[order[i % len(order)]] += 1
    return floored


def _schedule_from_shares(labels: list[str], shares: np.ndarray,
                          curve: pd.Series, adv_shares: float,
                          qty: int) -> pd.DataFrame:
    est_vol = curve.values * adv_shares if adv_shares and np.isfinite(adv_shares) else np.full(len(labels), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        part = np.where(est_vol > 0, shares / est_vol * 100, np.nan)
    cum = np.cumsum(shares)
    df = pd.DataFrame({
        "time": labels,
        "shares": shares.astype(int),
        "pct_of_order": np.round(shares / qty * 100, 1),
        "cum_shares": cum.astype(int),
        "cum_pct": np.round(cum / qty * 100, 1),
        "est_mkt_vol": np.round(est_vol, 0),
        "participation_pct": np.round(part, 2),
    })
    return df[df["shares"] > 0].reset_index(drop=True)


def _weights_for(algo: str, curve: pd.Series, risk_aversion: float) -> np.ndarray:
    n = len(curve)
    if algo == "twap":
        return np.full(n, 1.0 / n)
    if algo == "vwap":
        return curve.values.copy()
    if algo == "is":
        # Front-load: urgency κ grows with risk_aversion. κ=0 → uniform (TWAP).
        centers = np.linspace(0, 1, n)
        kappa = float(np.clip(risk_aversion, 0, 1)) * 6.0
        w = np.exp(-kappa * centers)
        return w / w.sum()
    raise ValueError(f"no weight scheme for algo={algo!r}")


def _pov_shares(curve: pd.Series, qty: int, adv_shares: float,
                participation: float) -> tuple[np.ndarray, int]:
    """Trade `participation` of each bucket's expected volume until filled.
    Returns (shares per bucket, unfilled shares)."""
    cap = curve.values * adv_shares * participation
    shares = np.zeros(len(curve), dtype=int)
    remaining = qty
    for i in range(len(curve)):
        take = int(min(np.floor(cap[i]), remaining))
        take = max(take, 0)
        shares[i] = take
        remaining -= take
        if remaining <= 0:
            break
    return shares, max(remaining, 0)


# --------------------------------------------------------------------------- #
# Cost model — square-root impact + timing risk, Amihud cross-check
# --------------------------------------------------------------------------- #
def _cost_estimate(sched: pd.DataFrame, qty: int, spot: float, adv_shares: float,
                   sigma_daily: float, bucket_minutes: int, amihud: float,
                   eta: float = IMPACT_ETA,
                   half_spread_bps: float = HALF_SPREAD_BPS) -> dict:
    """Schedule-aware cost: impact = η·σ·Σwᵢ√pᵢ (share-weighted per-bucket
    participation), timing risk from the share-weighted *execution time* (so a
    front-loaded schedule is correctly rewarded with lower drift exposure)."""
    order_value_cr = qty * spot / 1e7
    shares = sched["shares"].to_numpy(dtype=float)
    est_vol = sched["est_mkt_vol"].to_numpy(dtype=float)
    w = shares / qty                                    # weight per bucket, sums to 1
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(est_vol > 0, shares / est_vol, np.nan)   # participation fraction

    # --- market impact (square-root law)
    valid = np.isfinite(p)
    if valid.any() and np.isfinite(sigma_daily):
        wv = w[valid]
        wv = wv / wv.sum() if wv.sum() > 0 else wv
        impact_bps = 1e4 * eta * sigma_daily * float(np.sum(wv * np.sqrt(p[valid])))
    elif adv_shares and np.isfinite(adv_shares) and np.isfinite(sigma_daily):
        impact_bps = 1e4 * eta * sigma_daily * np.sqrt(qty / adv_shares)
    else:
        impact_bps = float("nan")

    # --- timing risk from share-weighted minutes-into-the-session
    starts = np.array([int(t[:2]) * 60 + int(t[3:5]) - SESSION_OPEN_MIN
                       for t in sched["time"]], dtype=float)
    centers = np.minimum(starts + bucket_minutes / 2.0, SESSION_MINUTES)
    eff_min = float(np.sum(w * centers))
    timing_risk_bps = (1e4 * sigma_daily * np.sqrt(max(eff_min, 0.0) / SESSION_MINUTES)
                       if np.isfinite(sigma_daily) else float("nan"))

    spread_bps = half_spread_bps
    total_bps = spread_bps + (impact_bps if np.isfinite(impact_bps) else 0.0)
    amihud_bps = (amihud * order_value_cr * 1e4
                  if amihud is not None and np.isfinite(amihud) else float("nan"))
    return {
        "impact_bps": round(impact_bps, 1) if np.isfinite(impact_bps) else float("nan"),
        "spread_bps": round(spread_bps, 1),
        "timing_risk_bps": round(timing_risk_bps, 1) if np.isfinite(timing_risk_bps) else float("nan"),
        "est_cost_bps": round(total_bps, 1),
        "amihud_bps": round(amihud_bps, 1) if np.isfinite(amihud_bps) else float("nan"),
        "est_cost_rupees": round(total_bps / 1e4 * qty * spot, 0),
    }


# --------------------------------------------------------------------------- #
# Top-level: build an execution plan
# --------------------------------------------------------------------------- #
def execution_plan(ticker: str, side: str = "buy", *, qty: int | None = None,
                   notional: float | None = None, algo: str = "vwap",
                   bucket_minutes: int = 30, participation: float = 0.10,
                   risk_aversion: float = 0.5, interval: str = "5m",
                   lookback_days: int = 20) -> ExecutionPlan | None:
    """Slice a parent order into a child-order schedule with a cost estimate.

    Give either `qty` (shares) or `notional` (₹). `algo` ∈ {vwap,twap,pov,is}.
    `participation` is the POV rate (0.10 = 10% of volume). `risk_aversion`
    (0–1) sets IS urgency. Returns None if the name has no price/spot."""
    side = side.lower()
    algo = algo.lower()
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    if algo not in ALGOS:
        raise ValueError(f"algo must be one of {ALGOS}")

    px = load_prices(ticker, days=10)
    if px is None or px.empty:
        return None
    spot = float(px["close"].iloc[-1])
    if spot <= 0:
        return None

    if qty is None:
        if notional is None:
            raise ValueError("pass either qty or notional")
        qty = int(notional // spot)
    qty = int(qty)
    if qty <= 0:
        return None

    liq = liq_mod.liquidity_profile(ticker)
    adv_shares = float(liq.adv_volume) if liq else float("nan")
    adv_cr = float(liq.adv_value_cr) if liq else float("nan")
    amihud = float(liq.amihud) if liq and np.isfinite(liq.amihud) else float("nan")
    tier = liq.tier if liq else "n/a"
    sigma = daily_vol(ticker) or float("nan")

    curve, used_fallback = intraday_volume_curve(
        ticker, interval=interval, bucket_minutes=bucket_minutes, lookback_days=lookback_days)
    labels = list(curve.index)

    warnings: list[str] = []
    unfilled = 0
    completes = True

    if algo == "pov":
        if not (adv_shares and np.isfinite(adv_shares) and adv_shares > 0):
            warnings.append("No ADV available — POV needs volume data; using VWAP shape instead.")
            shares = _round_to_total(curve.values * qty, qty)
        else:
            shares, unfilled = _pov_shares(curve, qty, adv_shares, participation)
            completes = unfilled <= 0
            if not completes:
                warnings.append(
                    f"At {participation*100:.0f}% participation this won't finish in one "
                    f"session — {unfilled:,} shares left over. Spread across days or raise "
                    "the participation rate (more impact).")
    else:
        weights = _weights_for(algo, curve, risk_aversion)
        shares = _round_to_total(weights * qty, qty)

    sched = _schedule_from_shares(labels, shares, curve, adv_shares, qty)
    if sched.empty:
        return None

    filled = int(sched["shares"].sum())
    horizon_min = len(sched) * bucket_minutes
    # Average participation over the horizon = order filled ÷ expected volume there.
    exp_vol = float(sched["est_mkt_vol"].sum()) if sched["est_mkt_vol"].notna().any() else float("nan")
    avg_part = (filled / exp_vol * 100) if exp_vol and np.isfinite(exp_vol) and exp_vol > 0 else float("nan")

    cost = _cost_estimate(sched, filled, spot, adv_shares, sigma, bucket_minutes, amihud)

    # --- warnings & summary
    if np.isfinite(adv_shares) and adv_shares > 0 and qty / adv_shares > 0.20:
        warnings.append(
            f"Order is {qty / adv_shares * 100:.0f}% of ADV — large; impact will bite. "
            "Consider splitting across multiple days.")
    if tier in ("illiquid", "untradeable"):
        warnings.append(f"⚠ {ticker} is {tier} — execution costs/slippage are unreliable; size tiny or skip.")
    if used_fallback:
        warnings.append("No intraday history for this name — schedule uses a generic U-shaped volume curve.")

    verb = "Buy" if side == "buy" else "Sell"
    summary = [
        f"{verb} {filled:,} shares of {ticker} (~₹{filled * spot:,.0f}) via {algo.upper()}.",
        f"Arrival price ₹{spot:,.2f} · ADV ₹{adv_cr:.1f} cr · liquidity: {tier}.",
        f"Work the order in {len(sched)} slices over ~{horizon_min} min "
        f"(avg ~{avg_part:.1f}% of volume)." if np.isfinite(avg_part)
        else f"Work the order in {len(sched)} slices over ~{horizon_min} min.",
        f"Estimated cost ≈ {cost['est_cost_bps']:.0f} bps (₹{cost['est_cost_rupees']:,.0f}) vs arrival; "
        f"timing risk ≈ {cost['timing_risk_bps']:.0f} bps over the horizon.",
        "Place each slice as a LIMIT order near mid; skip a slice if the bid/ask spread blows out.",
    ]

    return ExecutionPlan(
        ticker=ticker, side=side, algo=algo, qty=filled, arrival_price=round(spot, 2),
        notional=round(filled * spot, 2), schedule=sched, horizon_minutes=horizon_min,
        avg_participation_pct=round(avg_part, 1) if np.isfinite(avg_part) else float("nan"),
        completes_in_session=completes, unfilled_shares=int(unfilled),
        est_cost_bps=cost["est_cost_bps"], impact_bps=cost["impact_bps"],
        spread_bps=cost["spread_bps"], timing_risk_bps=cost["timing_risk_bps"],
        amihud_bps=cost["amihud_bps"], est_cost_rupees=cost["est_cost_rupees"],
        adv_value_cr=round(adv_cr, 2) if np.isfinite(adv_cr) else float("nan"),
        adv_volume=round(adv_shares, 0) if np.isfinite(adv_shares) else float("nan"),
        liquidity_tier=tier, daily_vol_pct=round(sigma * 100, 2) if np.isfinite(sigma) else float("nan"),
        used_fallback_curve=used_fallback, summary=summary, warnings=warnings,
    )
