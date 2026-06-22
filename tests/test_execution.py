"""Execution-algorithm tests — schedules sum to the order, algos differ sensibly,
the volume curve normalises, and the cost model is monotone in size."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingdesk.analyze import execution as ex


# --------------------------------------------------------------------------- #
# Pure helpers (no DB)
# --------------------------------------------------------------------------- #
def test_session_buckets_tile_the_session():
    b = ex._session_buckets(30)
    assert b[0][0] == "09:15"
    assert b[0][1] == ex.SESSION_OPEN_MIN
    assert b[-1][2] == ex.SESSION_CLOSE_MIN
    # buckets are contiguous and cover the whole session
    assert sum(end - start for _, start, end, _ in b) == ex.SESSION_MINUTES


def test_fallback_curve_normalises_and_is_u_shaped():
    buckets = ex._session_buckets(30)
    curve = ex._fallback_curve(buckets)
    assert pytest.approx(curve.sum(), abs=1e-9) == 1.0
    # open and close buckets carry more than the quietest midday bucket
    assert curve.iloc[0] > curve.min()
    assert curve.iloc[-1] > curve.min()


def test_round_to_total_preserves_sum():
    w = np.array([0.1, 0.2, 0.25, 0.45]) * 97  # 97 shares, fractional
    shares = ex._round_to_total(w, 97)
    assert shares.sum() == 97
    assert (shares >= 0).all()


def test_weights_twap_uniform_vwap_tracks_curve():
    curve = pd.Series([0.4, 0.1, 0.1, 0.4], index=["a", "b", "c", "d"])
    twap = ex._weights_for("twap", curve, 0.5)
    assert np.allclose(twap, 0.25)
    vwap = ex._weights_for("vwap", curve, 0.5)
    assert np.allclose(vwap, curve.values)


def test_is_weights_frontload_more_with_risk_aversion():
    curve = pd.Series(np.ones(6) / 6, index=range(6))
    calm = ex._weights_for("is", curve, 0.0)        # → uniform
    urgent = ex._weights_for("is", curve, 1.0)      # → front-loaded
    assert np.allclose(calm, 1 / 6)
    assert urgent[0] > urgent[-1]
    # urgent puts more in the first half than the calm/uniform schedule
    assert urgent[:3].sum() > calm[:3].sum()


def test_pov_caps_at_participation_and_reports_leftover():
    curve = pd.Series([0.5, 0.5], index=["a", "b"])
    # ADV 1000 sh, 10% participation → 50 sh capacity per bucket = 100 total.
    shares, unfilled = ex._pov_shares(curve, qty=300, adv_shares=1000, participation=0.10)
    assert shares.sum() == 100
    assert unfilled == 200


def test_cost_estimate_monotone_in_participation():
    # Same order, same time — but a thinner bucket = higher participation = more impact.
    low_part = pd.DataFrame({"time": ["09:15"], "shares": [10.0], "est_mkt_vol": [1e6]})
    high_part = pd.DataFrame({"time": ["09:15"], "shares": [10.0], "est_mkt_vol": [50.0]})
    cheap = ex._cost_estimate(low_part, qty=10, spot=100, adv_shares=1e6,
                              sigma_daily=0.02, bucket_minutes=375, amihud=0.001)
    pricey = ex._cost_estimate(high_part, qty=10, spot=100, adv_shares=1e6,
                               sigma_daily=0.02, bucket_minutes=375, amihud=0.001)
    assert pricey["impact_bps"] > cheap["impact_bps"]
    assert pricey["est_cost_bps"] > cheap["est_cost_bps"]


def test_cost_estimate_frontloading_cuts_timing_risk():
    # Two-bucket order: same shares, but front-loaded (more early) vs back-loaded.
    front = pd.DataFrame({"time": ["09:15", "14:15"], "shares": [90.0, 10.0],
                          "est_mkt_vol": [1e6, 1e6]})
    back = pd.DataFrame({"time": ["09:15", "14:15"], "shares": [10.0, 90.0],
                         "est_mkt_vol": [1e6, 1e6]})
    f = ex._cost_estimate(front, qty=100, spot=100, adv_shares=1e6,
                          sigma_daily=0.02, bucket_minutes=60, amihud=0.001)
    b = ex._cost_estimate(back, qty=100, spot=100, adv_shares=1e6,
                          sigma_daily=0.02, bucket_minutes=60, amihud=0.001)
    assert f["timing_risk_bps"] < b["timing_risk_bps"]


# --------------------------------------------------------------------------- #
# End-to-end against the local DB (skips cleanly if no priced names)
# --------------------------------------------------------------------------- #
def _a_priced_ticker():
    from swingdesk.storage import connect
    try:
        with connect() as con:
            df = pd.read_sql_query("SELECT DISTINCT ticker FROM prices LIMIT 1", con)
        return df["ticker"].iloc[0] if not df.empty else None
    except Exception:
        return None


def test_execution_plan_schedule_sums_to_order():
    t = _a_priced_ticker()
    if t is None:
        pytest.skip("no priced tickers in local DB")
    plan = ex.execution_plan(t, "buy", qty=500, algo="vwap", bucket_minutes=60)
    if plan is None:
        pytest.skip(f"no spot for {t}")
    assert plan.schedule["shares"].sum() == plan.qty
    assert plan.qty == 500
    assert plan.arrival_price > 0
    assert pytest.approx(plan.schedule["cum_pct"].iloc[-1], abs=0.5) == 100.0


def test_execution_plan_notional_and_algos_run():
    t = _a_priced_ticker()
    if t is None:
        pytest.skip("no priced tickers in local DB")
    for algo in ex.ALGOS:
        plan = ex.execution_plan(t, "sell", notional=200000, algo=algo, bucket_minutes=60)
        if plan is None:
            pytest.skip(f"no spot for {t}")
        assert plan.qty > 0
        assert not plan.schedule.empty
        assert np.isfinite(plan.est_cost_bps)
