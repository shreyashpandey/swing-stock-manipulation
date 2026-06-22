"""Forward expected-range modelling — the honest answer to "where will the
price go?".

You cannot reliably predict direction, but you *can* model the **distribution**
of where price is likely to sit over your holding horizon. That's exactly what
options desks quote when they price a straddle: an expected move.

Three lenses, all numpy-only (no GARCH/scipy dependency):

  1. expected_range()  — lognormal sigma-bands. Volatility is estimated with
                         EWMA (RiskMetrics, lambda=0.94) which weights recent
                         days more, then scaled by sqrt(horizon). Reports the
                         ~68% (1 sigma) and ~95% (2 sigma) price envelope.
  2. monte_carlo()     — simulate thousands of forward paths (bootstrap from
                         the stock's own recent returns, or GBM) and read the
                         percentile fan. Captures fat tails the normal bands miss.
  3. vol_cone()        — is volatility currently high or low *for this stock*?
                         Current realized vol vs its own historical percentiles.

Convention: returns are simple daily pct-change; price bands use the lognormal
form spot * exp(+/- z * sigma * sqrt(h)) so they can never go negative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swingdesk.storage import load_prices

TRADING_DAYS = 252
EWMA_LAMBDA = 0.94          # RiskMetrics standard
Z_68 = 1.0
Z_95 = 1.96


def _daily_returns(close: pd.Series) -> pd.Series:
    return close.astype(float).pct_change().dropna()


def _log_returns(close: pd.Series) -> pd.Series:
    """Log returns — the right scale for the lognormal sigma-bands below."""
    c = close.astype(float)
    return np.log(c / c.shift(1)).dropna()


def ewma_vol(returns: pd.Series, lam: float = EWMA_LAMBDA) -> float:
    """Latest EWMA daily volatility (std of returns). Recent days weighted more
    heavily, so the estimate adapts to volatility regime shifts."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return float("nan")
    # var_t = lam*var_{t-1} + (1-lam)*r_{t-1}^2, seeded with sample variance.
    var = np.var(r[: min(20, len(r))])
    for x in r:
        var = lam * var + (1 - lam) * x * x
    return float(np.sqrt(var))


def daily_vol(ticker: str, lookback: int = 250) -> float | None:
    """EWMA daily (log-return) volatility for a ticker, or None if too little
    history. Cheap building block for vol-implied time-to-target estimates."""
    px = load_prices(ticker, days=lookback + 5)
    if px.empty or len(px) < 40:
        return None
    dv = ewma_vol(_log_returns(px["close"]).tail(lookback))
    return float(dv) if np.isfinite(dv) and dv > 0 else None


def vol_implied_days(daily_vol_frac: float, entry: float, target: float) -> float | None:
    """Sessions at which the entry→target move equals a typical (1σ) move:
    ≈ (|ln(target/entry)| / daily_vol)². Works for longs and shorts."""
    if not daily_vol_frac or daily_vol_frac <= 0 or entry <= 0 or target <= 0:
        return None
    move_log = abs(np.log(target / entry))
    if move_log <= 0:
        return None
    return float((move_log / daily_vol_frac) ** 2)


def realized_vol(returns: pd.Series, window: int) -> float:
    """Simple trailing realized daily vol over `window` bars."""
    r = returns.dropna().tail(window)
    if len(r) < max(5, window // 2):
        return float("nan")
    return float(r.std(ddof=1))


# --------------------------------------------------------------------------- #
# 1. lognormal expected-range bands
# --------------------------------------------------------------------------- #
@dataclass
class ExpectedRange:
    ticker: str
    spot: float
    horizon_days: int
    daily_vol_pct: float       # EWMA daily vol, %
    horizon_vol_pct: float     # vol scaled to the horizon, %
    expected_move_pct: float   # the +/- 1 sigma move as a %
    low_68: float
    high_68: float
    low_95: float
    high_95: float
    annualized_vol_pct: float
    narrative: str


def expected_range(ticker: str, horizon_days: int = 10,
                   lookback: int = 120) -> ExpectedRange | None:
    """Lognormal sigma-band forecast over `horizon_days` sessions.

    Drift is intentionally set to zero (the median outcome = today's price):
    estimated drift over a few weeks is mostly noise and would bias the range.
    Use this to set targets/stops that respect the stock's actual volatility,
    not round numbers.
    """
    px = load_prices(ticker, days=lookback + 5)
    if px.empty or len(px) < 30:
        return None
    close = px["close"].astype(float)
    spot = float(close.iloc[-1])
    # Log-return vol: consistent with the lognormal band form spot*exp(±z*sigma).
    dvol = ewma_vol(_log_returns(close).tail(lookback))
    if not np.isfinite(dvol) or dvol <= 0:
        return None

    hvol = dvol * np.sqrt(horizon_days)        # sqrt-time scaling
    band68 = Z_68 * hvol
    band95 = Z_95 * hvol
    er = ExpectedRange(
        ticker=ticker,
        spot=round(spot, 2),
        horizon_days=horizon_days,
        daily_vol_pct=round(dvol * 100, 2),
        horizon_vol_pct=round(hvol * 100, 2),
        expected_move_pct=round(band68 * 100, 2),
        low_68=round(spot * np.exp(-band68), 2),
        high_68=round(spot * np.exp(band68), 2),
        low_95=round(spot * np.exp(-band95), 2),
        high_95=round(spot * np.exp(band95), 2),
        annualized_vol_pct=round(dvol * np.sqrt(TRADING_DAYS) * 100, 1),
        narrative="",
    )
    er.narrative = (
        f"Over the next {horizon_days} sessions, {ticker} is ~68% likely to sit "
        f"between ₹{er.low_68:,.0f} and ₹{er.high_68:,.0f} (±{er.expected_move_pct:.1f}%), "
        f"and ~95% likely within ₹{er.low_95:,.0f}–₹{er.high_95:,.0f}. "
        f"Annualized vol {er.annualized_vol_pct:.0f}%."
    )
    return er


# --------------------------------------------------------------------------- #
# 2. Monte Carlo fan
# --------------------------------------------------------------------------- #
def _simulate_paths(spot: float, rets: np.ndarray, horizon_days: int,
                    n_sims: int, method: str, block: int,
                    seed: int | None) -> np.ndarray:
    """Generate an (horizon_days x n_sims) array of simulated price paths.
    Shared by monte_carlo() and the holding planner."""
    rng = np.random.default_rng(seed)
    if method == "gbm":
        mu, sigma = float(np.mean(rets)), float(np.std(rets, ddof=1))
        shocks = rng.normal(mu - 0.5 * sigma**2, sigma, size=(horizon_days, n_sims))
        return spot * np.exp(np.cumsum(shocks, axis=0))
    if method == "block" and len(rets) > block >= 2:
        n_blocks = int(np.ceil(horizon_days / block))
        max_start = len(rets) - block
        starts = rng.integers(0, max_start + 1, size=(n_sims, n_blocks))
        idx = starts[:, :, None] + np.arange(block)[None, None, :]
        idx = idx.reshape(n_sims, n_blocks * block)[:, :horizon_days]
        return spot * np.cumprod(1 + rets[idx].T, axis=0)
    draws = rng.choice(rets, size=(horizon_days, n_sims), replace=True)
    return spot * np.cumprod(1 + draws, axis=0)


@dataclass
class MonteCarlo:
    ticker: str
    spot: float
    horizon_days: int
    method: str
    n_sims: int
    terminal_p5: float
    terminal_p25: float
    terminal_p50: float
    terminal_p75: float
    terminal_p95: float
    prob_up: float                       # P(terminal > spot)
    fan: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


def monte_carlo(ticker: str, horizon_days: int = 10, n_sims: int = 5000,
                method: str = "block", lookback: int = 250,
                block: int = 5, seed: int | None = 42) -> MonteCarlo | None:
    """Simulate `n_sims` forward price paths.

    method="block" (default): block bootstrap — resample *contiguous blocks* of
        the stock's recent returns (length `block`). This preserves volatility
        clustering and short-run autocorrelation, so multi-day ranges aren't
        understated the way plain iid resampling does. Best for swing horizons.
    method="bootstrap": iid resample of single daily returns (keeps fat tails
        but assumes day-to-day independence).
    method="gbm": geometric Brownian motion with sample mean/vol (smooth,
        thin-tailed; textbook baseline).

    Returns terminal-price percentiles + a per-day percentile `fan` DataFrame
    for plotting (columns p5/p25/p50/p75/p95, indexed by day 1..horizon).
    """
    px = load_prices(ticker, days=lookback + 5)
    if px.empty or len(px) < 30:
        return None
    close = px["close"].astype(float)
    spot = float(close.iloc[-1])
    rets = _daily_returns(close).tail(lookback).values
    if len(rets) < 20:
        return None

    paths = _simulate_paths(spot, rets, horizon_days, n_sims, method, block, seed)

    qs = [5, 25, 50, 75, 95]
    fan = pd.DataFrame(
        {f"p{q}": np.percentile(paths, q, axis=1) for q in qs},
        index=pd.RangeIndex(1, horizon_days + 1, name="day"),
    )
    terminal = paths[-1, :]
    return MonteCarlo(
        ticker=ticker, spot=round(spot, 2), horizon_days=horizon_days,
        method=method, n_sims=n_sims,
        terminal_p5=round(float(np.percentile(terminal, 5)), 2),
        terminal_p25=round(float(np.percentile(terminal, 25)), 2),
        terminal_p50=round(float(np.percentile(terminal, 50)), 2),
        terminal_p75=round(float(np.percentile(terminal, 75)), 2),
        terminal_p95=round(float(np.percentile(terminal, 95)), 2),
        prob_up=round(float((terminal > spot).mean()), 3),
        fan=fan,
    )


# --------------------------------------------------------------------------- #
# 3. volatility cone — is vol cheap or rich right now?
# --------------------------------------------------------------------------- #
def vol_cone(ticker: str, windows: tuple[int, ...] = (10, 20, 30, 60),
             lookback: int = 252) -> pd.DataFrame:
    """For each window, current realized vol vs its own historical distribution.

    A high percentile means the stock is unusually volatile right now (ranges
    will be wide, options expensive); a low percentile means it's compressed
    (often precedes expansion). Columns: window, current_vol_pct, pctile,
    median_vol_pct, read.
    """
    px = load_prices(ticker)
    if px.empty or len(px) < max(windows) + 30:
        return pd.DataFrame()
    rets = _daily_returns(px["close"])

    rows = []
    for w in windows:
        roll = rets.rolling(w).std(ddof=1).dropna()
        hist = roll.tail(lookback)
        if len(hist) < 20:
            continue
        cur = float(roll.iloc[-1])
        pctile = float((hist < cur).mean() * 100)
        rows.append({
            "window": w,
            "current_vol_pct": round(cur * 100, 2),
            "annualized_pct": round(cur * np.sqrt(TRADING_DAYS) * 100, 1),
            "pctile": round(pctile, 0),
            "median_vol_pct": round(float(hist.median()) * 100, 2),
            "read": ("compressed — expansion likely" if pctile < 25
                     else "elevated — wide ranges/risk" if pctile > 75
                     else "normal"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. holding planner — "hold ~N days for a ~X% gain, with ~P% odds"
# --------------------------------------------------------------------------- #
def holding_plan(ticker: str, targets_pct: tuple[float, ...] = (3, 5, 8, 10, 15),
                 max_horizon: int = 40, n_sims: int = 6000,
                 method: str = "block", lookback: int = 250,
                 block: int = 5, seed: int | None = 42) -> dict | None:
    """How long to hold for a given % gain, and the odds of getting there.

    For each target gain we Monte-Carlo the stock's own recent return behaviour
    forward `max_horizon` sessions and read off:
      * prob_hit       — share of paths that TOUCH the target within the horizon
      * median_days    — typical sessions-to-target among the paths that hit it
      * vol_implied_days — sessions at which the target equals a 1σ move
                           (≈ (target / daily_vol)² ); a quick volatility sanity check

    This is probabilistic, NOT a promise: it says "given how this stock has been
    moving, a +X% gain is a ~P% bet over ~N sessions" — use it to set realistic
    targets and holding periods, sized with the 🛡 Risk tools.
    """
    px = load_prices(ticker, days=lookback + 5)
    if px.empty or len(px) < 40:
        return None
    close = px["close"].astype(float)
    spot = float(close.iloc[-1])
    rets = _daily_returns(close).tail(lookback).values
    if len(rets) < 30:
        return None
    dvol = ewma_vol(_log_returns(close).tail(lookback))
    if not np.isfinite(dvol) or dvol <= 0:
        return None

    paths = _simulate_paths(spot, rets, max_horizon, n_sims, method, block, seed)
    rows = []
    for tgt in targets_pct:
        tp = spot * (1 + tgt / 100.0)
        hit = paths >= tp                       # (horizon, n_sims)
        touched = hit.any(axis=0)
        prob = float(touched.mean())
        # first-touch day per path (1-based); median among those that hit.
        first_day = np.where(touched, hit.argmax(axis=0) + 1, np.nan)
        median_days = float(np.nanmedian(first_day[touched])) if touched.any() else float("nan")
        # vol-implied sessions for the target to be a 1σ (lognormal) move.
        tgt_log = np.log1p(tgt / 100.0)
        vol_implied = (tgt_log / dvol) ** 2
        rows.append({
            "target_pct": tgt,
            "target_price": round(tp, 2),
            "prob_hit": round(prob, 3),
            "median_days_to_hit": round(median_days, 1) if np.isfinite(median_days) else None,
            "vol_implied_days": round(float(vol_implied), 1),
        })
    return {
        "ticker": ticker, "spot": round(spot, 2),
        "daily_vol_pct": round(dvol * 100, 2),
        "max_horizon": max_horizon, "method": method, "n_sims": n_sims,
        "table": pd.DataFrame(rows),
    }


# --------------------------------------------------------------------------- #
# 5. target-vs-stop — reward:risk via first-passage simulation
# --------------------------------------------------------------------------- #
@dataclass
class TargetVsStop:
    ticker: str
    spot: float
    target_pct: float
    stop_pct: float
    rr: float                    # reward:risk = target/stop
    max_horizon: int
    p_target_first: float        # P(touch +target before -stop, within horizon)
    p_stop_first: float          # P(touch -stop before +target)
    p_neither: float             # P(neither barrier touched in the window)
    win_rate: float              # target / (target + stop), of resolved paths
    median_days_to_target: float
    median_days_to_stop: float
    expectancy_pct: float        # mean outcome % (target win, stop loss, else terminal)
    expectancy_r: float          # expectancy in R units (1R = the stop distance)
    verdict: str
    narrative: str


def target_vs_stop(ticker: str, target_pct: float = 8.0, stop_pct: float = 4.0,
                   max_horizon: int = 40, n_sims: int = 6000, method: str = "block",
                   lookback: int = 250, block: int = 5,
                   seed: int | None = 42) -> TargetVsStop | None:
    """Probability of hitting a +target% before a -stop% (and the expectancy).

    Simulates the stock's own recent behaviour forward and, per path, finds which
    barrier is touched FIRST. Honest caveat: paths are close-to-close, so this
    can't see intraday wicks — it slightly *under*-counts barrier touches vs real
    OHLC. Expectancy assumes you exit at target/stop, else at the horizon close.
    """
    if target_pct <= 0 or stop_pct <= 0:
        return None
    px = load_prices(ticker, days=lookback + 5)
    if px.empty or len(px) < 40:
        return None
    close = px["close"].astype(float)
    spot = float(close.iloc[-1])
    rets = _daily_returns(close).tail(lookback).values
    if len(rets) < 30:
        return None

    paths = _simulate_paths(spot, rets, max_horizon, n_sims, method, block, seed)
    tp = spot * (1 + target_pct / 100.0)
    sl = spot * (1 - stop_pct / 100.0)
    big = max_horizon + 1
    up_hit, dn_hit = paths >= tp, paths <= sl
    first_up = np.where(up_hit.any(axis=0), up_hit.argmax(axis=0) + 1, big)
    first_dn = np.where(dn_hit.any(axis=0), dn_hit.argmax(axis=0) + 1, big)

    target_first = (first_up < first_dn) & (first_up <= max_horizon)
    stop_first = (first_dn < first_up) & (first_dn <= max_horizon)
    neither = (first_up > max_horizon) & (first_dn > max_horizon)

    p_t = float(target_first.mean())
    p_s = float(stop_first.mean())
    p_n = float(neither.mean())
    resolved = p_t + p_s
    win_rate = p_t / resolved if resolved > 0 else float("nan")

    term_pct = (paths[-1] / spot - 1) * 100
    outcome = np.where(target_first, target_pct,
                       np.where(stop_first, -stop_pct, term_pct))
    exp_pct = float(outcome.mean())
    exp_r = exp_pct / stop_pct                       # 1R = the stop distance
    rr = target_pct / stop_pct

    md_t = float(np.median(first_up[target_first])) if target_first.any() else float("nan")
    md_s = float(np.median(first_dn[stop_first])) if stop_first.any() else float("nan")

    verdict = ("favourable — positive edge" if exp_r > 0.1 and p_t > p_s
               else "marginal" if exp_r > 0 else "unfavourable — negative edge")
    narrative = (
        f"+{target_pct:.0f}% target vs −{stop_pct:.0f}% stop (R:R {rr:.1f}): "
        f"~{p_t*100:.0f}% hit target first, ~{p_s*100:.0f}% stopped out, "
        f"~{p_n*100:.0f}% neither within {max_horizon} sessions. "
        f"Expectancy ≈ {exp_pct:+.2f}% ({exp_r:+.2f}R). → {verdict}."
    )
    return TargetVsStop(
        ticker=ticker, spot=round(spot, 2), target_pct=target_pct, stop_pct=stop_pct,
        rr=round(rr, 2), max_horizon=max_horizon,
        p_target_first=round(p_t, 3), p_stop_first=round(p_s, 3), p_neither=round(p_n, 3),
        win_rate=round(win_rate, 3) if np.isfinite(win_rate) else float("nan"),
        median_days_to_target=round(md_t, 1) if np.isfinite(md_t) else float("nan"),
        median_days_to_stop=round(md_s, 1) if np.isfinite(md_s) else float("nan"),
        expectancy_pct=round(exp_pct, 2), expectancy_r=round(exp_r, 2),
        verdict=verdict, narrative=narrative,
    )
