"""The unified decision engine: veto order, conviction ordering, the today-timing
read, plan sizing and holding advice."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk.analyze import decision
from swingdesk.storage import replace_holdings, upsert_fundamentals, upsert_prices


def _seed(ticker, price, daily_vol, drift, n=320, quality=70, sector="IT", seed=1):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.012, n)
    close = price * np.cumprod(1 + rets)
    vol = np.full(n, float(daily_vol)) * rng.uniform(0.9, 1.1, n)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    upsert_prices(ticker, pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": vol}, index=idx))
    upsert_fundamentals([{"ticker": ticker, "market_cap": 2.0e11, "float_shares": 2.0e8,
                          "shares_outstanding": 4.0e8, "quality_score": quality,
                          "sector": sector, "return_on_equity": 0.18, "trailing_pe": 22,
                          "earnings_growth": 0.15, "revenue_growth": 0.12,
                          "debt_to_equity": 0.3}])


# --- integration: vetoes, conviction ordering, plan sizing -------------------
def test_illiquid_is_vetoed_to_avoid(tmp_db):
    # Strong uptrend but only ~₹0.1 cr/day traded → untradeable liquidity veto.
    _seed("ILQ.NS", 500, 2_000, drift=0.004, seed=5)
    d = decision.decide("ILQ.NS", run_montecarlo=False)
    assert d.action == "AVOID"
    assert d.liq_tier in ("illiquid", "untradeable")
    assert d.timing == "DONT_ENTER_TODAY"
    if d.liq_tier == "untradeable":
        assert d.conviction <= 20
        assert d.veto_reason is not None


def test_good_outranks_weak(tmp_db):
    _seed("GOOD.NS", 500, 2_000_000, drift=0.004, quality=80, seed=1)
    _seed("MEH.NS", 300, 1_500_000, drift=0.0, quality=60, seed=2)
    _seed("WEAK.NS", 200, 1_200_000, drift=-0.003, quality=45, seed=3)
    ds = {d.ticker: d for d in decision.decide_universe(
        ["GOOD.NS", "MEH.NS", "WEAK.NS"], run_montecarlo=False)}
    assert ds["GOOD.NS"].conviction > ds["WEAK.NS"].conviction
    # The strong, liquid, quality name should be a buy-ish action; the downtrend isn't.
    assert ds["GOOD.NS"].action in ("ACCUMULATE", "BUY", "STRONG_BUY")
    assert decision.ACTION_ORDER.index(ds["GOOD.NS"].action) >= \
        decision.ACTION_ORDER.index(ds["WEAK.NS"].action)


def test_turnover_numbers_present_and_single_source(tmp_db):
    _seed("SRC.NS", 500, 2_000_000, drift=0.003, seed=7)
    d = decision.decide("SRC.NS", run_montecarlo=False)
    # Both windows surfaced on the one decision object (the single source of truth).
    assert d.today_turnover_pct is not None
    assert d.avg_turnover_pct is not None
    assert d.today_vs_avg_value_mult is not None


def test_plan_trims_when_amount_over_risk(tmp_db):
    _seed("PLN.NS", 500, 2_000_000, drift=0.004, seed=1)
    d = decision.decide("PLN.NS", planned_amount=300000, capital=100000,
                        risk_pct=1.0, run_montecarlo=False)
    assert d.plan is not None
    if d.entry:                                   # levels resolved → sizing applies
        assert d.plan.over_risk is True
        assert d.plan.suggested_shares == d.plan.risk_sized_shares


def test_holding_advice_attached_for_owned_name(tmp_db):
    _seed("HLD.NS", 500, 2_000_000, drift=0.004, seed=1)
    replace_holdings([{"ticker": "HLD.NS", "qty": 100, "avg_price": 400,
                       "last_price": 520, "current_value": 52000}])
    ds = {d.ticker: d for d in decision.decide_universe(["HLD.NS"], run_montecarlo=False)}
    h = ds["HLD.NS"].holding
    assert h is not None
    assert h.action in ("HOLD", "ADD", "TRIM", "EXIT")
    assert h.unrealized_pct > 0                   # bought at 400, marked at 520


def test_short_history_returns_wait(tmp_db):
    _seed("TINY.NS", 100, 100_000, drift=0.001, n=30, seed=1)
    d = decision.decide("TINY.NS", run_montecarlo=False)
    assert d.action == "WAIT"
    assert d.veto_reason == "insufficient history"


# --- unit: the today-timing decision tree (deterministic, no fan-out) --------
def _last(rsi, ema20):
    return pd.Series({"rsi14": rsi, "ema20": ema20, "ema200": 50.0})


def _kw(**over):
    base = dict(action="BUY", last=_last(55, 100), spot=102, er=None, mm=None,
                liq_tier="liquid", manip_tier="Low", return_z=None,
                regime_label="neutral", sb={"tilt": "BUY", "score": 30},
                has_setup=True, ticker="X.NS")
    base.update(over)
    return base


def test_timing_enter_today():
    t, _ = decision._timing(**_kw())
    assert t == "ENTER_TODAY"


def test_timing_wait_pullback_when_overbought():
    t, _ = decision._timing(**_kw(last=_last(82, 100), spot=102))
    assert t == "WAIT_PULLBACK"


def test_timing_wait_pullback_when_extended_above_ema20():
    t, _ = decision._timing(**_kw(last=_last(60, 100), spot=110))   # +10% over EMA20
    assert t == "WAIT_PULLBACK"


def test_timing_dont_enter_on_illiquid():
    t, _ = decision._timing(**_kw(liq_tier="illiquid"))
    assert t == "DONT_ENTER_TODAY"


def test_timing_dont_enter_on_abnormal_day():
    t, _ = decision._timing(**_kw(return_z=4.5))
    assert t == "DONT_ENTER_TODAY"


def test_timing_avoid_action_forces_dont_enter():
    t, _ = decision._timing(**_kw(action="AVOID"))
    assert t == "DONT_ENTER_TODAY"
