"""Tests for the forward expected-range / Monte Carlo / vol-cone module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingdesk.analyze import expected_range as er
from swingdesk.storage import upsert_prices


def _seed_stock(ticker: str, n: int = 300, daily_vol: float = 0.02, seed: int = 3):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, daily_vol, n)
    close = 1000 * np.cumprod(1 + rets)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    px = pd.DataFrame({
        "open": close * 0.999, "high": close * 1.012,
        "low": close * 0.988, "close": close, "volume": np.full(n, 4e5),
    }, index=idx)
    upsert_prices(ticker, px)


def test_expected_range_bands_nested_and_centered(tmp_db):
    _seed_stock("RNG.NS")
    out = er.expected_range("RNG.NS", horizon_days=10)
    assert out is not None
    # 95% band must be wider than 68% band, both bracketing spot.
    assert out.low_95 < out.low_68 < out.spot < out.high_68 < out.high_95
    # 10-day vol ~ daily*sqrt(10); daily ~2% -> ~6.3%. Loose bounds for noise.
    assert 3.0 < out.expected_move_pct < 12.0


def test_expected_range_scales_with_horizon(tmp_db):
    _seed_stock("RNG.NS")
    short = er.expected_range("RNG.NS", horizon_days=5)
    long = er.expected_range("RNG.NS", horizon_days=20)
    # Longer horizon -> wider expected move (sqrt-time).
    assert long.expected_move_pct > short.expected_move_pct


def test_higher_vol_stock_has_wider_range(tmp_db):
    _seed_stock("CALM.NS", daily_vol=0.01, seed=1)
    _seed_stock("WILD.NS", daily_vol=0.04, seed=2)
    calm = er.expected_range("CALM.NS", horizon_days=10)
    wild = er.expected_range("WILD.NS", horizon_days=10)
    assert wild.expected_move_pct > calm.expected_move_pct * 1.5


def test_monte_carlo_fan_grows_and_brackets(tmp_db):
    _seed_stock("RNG.NS")
    mc = er.monte_carlo("RNG.NS", horizon_days=15, n_sims=3000, method="bootstrap")
    assert mc is not None
    assert mc.terminal_p5 < mc.terminal_p50 < mc.terminal_p95
    # Fan widens with time: day-15 spread > day-1 spread.
    spread = mc.fan["p95"] - mc.fan["p5"]
    assert spread.iloc[-1] > spread.iloc[0]
    assert 0.0 <= mc.prob_up <= 1.0


def test_monte_carlo_gbm_method_runs(tmp_db):
    _seed_stock("RNG.NS")
    mc = er.monte_carlo("RNG.NS", horizon_days=10, n_sims=2000, method="gbm")
    assert mc is not None and mc.method == "gbm"
    assert mc.terminal_p25 < mc.terminal_p75


def test_block_bootstrap_default_runs_and_brackets(tmp_db):
    _seed_stock("RNG.NS")
    mc = er.monte_carlo("RNG.NS", horizon_days=15, n_sims=3000)  # default=block
    assert mc is not None and mc.method == "block"
    assert mc.terminal_p5 < mc.terminal_p50 < mc.terminal_p95
    # Fan still widens with time.
    spread = mc.fan["p95"] - mc.fan["p5"]
    assert spread.iloc[-1] > spread.iloc[0]


def test_block_bootstrap_has_fatter_tails_than_gbm(tmp_db):
    # Volatility-clustering series (alternating calm/wild 20-day regimes).
    # Resampling-based methods preserve the fat tails of the real distribution;
    # GBM forces Gaussian terminals. Tail ratio (p95-p5)/(p75-p25) ~2.44 for a
    # normal — block bootstrap should exceed that and beat GBM.
    rng = np.random.default_rng(11)
    n = 400
    rets = np.array([rng.normal(0, 0.05 if (i // 20) % 2 else 0.005)
                     for i in range(n)])
    close = 1000 * np.cumprod(1 + rets)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    upsert_prices("CLUST.NS", pd.DataFrame({
        "open": close, "high": close * 1.02, "low": close * 0.98,
        "close": close, "volume": np.full(n, 4e5)}, index=idx))

    def tail_ratio(mc):
        return (mc.terminal_p95 - mc.terminal_p5) / (mc.terminal_p75 - mc.terminal_p25)

    block = er.monte_carlo("CLUST.NS", horizon_days=20, n_sims=8000, method="block")
    gbm = er.monte_carlo("CLUST.NS", horizon_days=20, n_sims=8000, method="gbm")
    assert tail_ratio(block) > tail_ratio(gbm)
    assert tail_ratio(block) > 2.44     # fatter than Gaussian


def test_vol_cone_returns_percentiles(tmp_db):
    _seed_stock("RNG.NS", n=320)
    cone = er.vol_cone("RNG.NS")
    assert not cone.empty
    assert {"window", "current_vol_pct", "pctile", "read"}.issubset(cone.columns)
    assert cone["pctile"].between(0, 100).all()


def test_holding_plan_structure_and_monotonicity(tmp_db):
    _seed_stock("RNG.NS", daily_vol=0.02)
    plan = er.holding_plan("RNG.NS", targets_pct=(3, 5, 10), max_horizon=40)
    assert plan is not None
    tbl = plan["table"]
    assert list(tbl["target_pct"]) == [3, 5, 10]
    # Bigger gains are LESS likely to be touched within the same horizon...
    assert tbl["prob_hit"].is_monotonic_decreasing
    # ...and take longer (vol-implied sessions grow with the target).
    assert tbl["vol_implied_days"].is_monotonic_increasing
    assert tbl["prob_hit"].between(0, 1).all()


def test_holding_plan_higher_vol_hits_faster(tmp_db):
    _seed_stock("CALM.NS", daily_vol=0.01, seed=1)
    _seed_stock("WILD.NS", daily_vol=0.04, seed=2)
    calm = er.holding_plan("CALM.NS", targets_pct=(5,), max_horizon=40)
    wild = er.holding_plan("WILD.NS", targets_pct=(5,), max_horizon=40)
    # A +5% target is far more reachable for the volatile stock.
    assert wild["table"].iloc[0]["prob_hit"] > calm["table"].iloc[0]["prob_hit"]
    # And the vol-implied time-to-target is shorter for the volatile stock.
    assert wild["table"].iloc[0]["vol_implied_days"] < calm["table"].iloc[0]["vol_implied_days"]


def test_target_vs_stop_probabilities_sum_to_one(tmp_db):
    _seed_stock("RNG.NS", daily_vol=0.02)
    tvs = er.target_vs_stop("RNG.NS", target_pct=8, stop_pct=4, max_horizon=40)
    assert tvs is not None
    total = tvs.p_target_first + tvs.p_stop_first + tvs.p_neither
    assert abs(total - 1.0) < 1e-6
    assert tvs.rr == 2.0
    for p in (tvs.p_target_first, tvs.p_stop_first, tvs.p_neither):
        assert 0.0 <= p <= 1.0


def test_tight_stop_wide_target_gets_stopped_more(tmp_db):
    _seed_stock("RNG.NS", daily_vol=0.02)
    # +10% target with a tight -2% stop -> stop hit first far more often.
    tvs = er.target_vs_stop("RNG.NS", target_pct=10, stop_pct=2, max_horizon=40)
    assert tvs.p_stop_first > tvs.p_target_first
    # Mirror image: easy +2% target, far -10% stop -> target first dominates.
    tvs2 = er.target_vs_stop("RNG.NS", target_pct=2, stop_pct=10, max_horizon=40)
    assert tvs2.p_target_first > tvs2.p_stop_first


def test_symmetric_barriers_both_resolve(tmp_db):
    # Equal target/stop over a long horizon: nearly all paths resolve to one
    # barrier, and BOTH sides get a meaningful share. (We don't assert ~50/50 —
    # mean-zero arithmetic returns carry negative geometric drift / volatility
    # drag, which legitimately tilts first-touch toward the downside stop.)
    _seed_stock("FLAT.NS", daily_vol=0.02, seed=7)
    tvs = er.target_vs_stop("FLAT.NS", target_pct=5, stop_pct=5, max_horizon=60)
    assert tvs.p_neither < 0.15                      # long horizon -> mostly resolved
    assert tvs.p_target_first > 0.1 and tvs.p_stop_first > 0.1
    assert tvs.verdict in {"favourable — positive edge", "marginal",
                           "unfavourable — negative edge"}


def test_daily_vol_and_vol_implied_days(tmp_db):
    _seed_stock("CALM.NS", daily_vol=0.01, seed=1)
    _seed_stock("WILD.NS", daily_vol=0.04, seed=2)
    dv_calm = er.daily_vol("CALM.NS")
    dv_wild = er.daily_vol("WILD.NS")
    assert dv_calm and dv_wild and dv_wild > dv_calm
    # Same +5% target: the calmer stock needs MORE sessions to make it a 1σ move.
    d_calm = er.vol_implied_days(dv_calm, 100, 105)
    d_wild = er.vol_implied_days(dv_wild, 100, 105)
    assert d_calm > d_wild > 0
    # Works for shorts (target below entry) and rejects degenerate inputs.
    assert er.vol_implied_days(dv_calm, 100, 95) > 0
    assert er.vol_implied_days(dv_calm, 100, 100) is None
    assert er.vol_implied_days(None, 100, 105) is None
    assert er.daily_vol("SHORT.NS") is None


def test_insufficient_data_returns_none(tmp_db):
    _seed_stock("SHORT.NS", n=10)
    assert er.expected_range("SHORT.NS") is None
    assert er.monte_carlo("SHORT.NS") is None
    assert er.vol_cone("SHORT.NS").empty
    assert er.holding_plan("SHORT.NS") is None
    assert er.target_vs_stop("SHORT.NS") is None
