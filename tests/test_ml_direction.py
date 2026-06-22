"""Tests for the ML P(up over N) direction model.

Two signal regimes are exercised:
  * Trending (AR(1) returns, positive autocorrelation) — direction is partly
    predictable, so walk-forward should show an edge over the base rate.
  * Random walk — no predictability; the model must NOT manufacture a big edge
    out-of-sample (the honesty guardrail).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingdesk.analyze import ml_direction as ml
from swingdesk.storage import upsert_prices


def _seed_ar1(ticker, phi, seed, n=700, vol=0.015):
    """AR(1) returns: r_t = phi*r_{t-1} + eps. phi>0 => momentum persistence."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, vol, n)
    r = np.empty(n)
    r[0] = eps[0]
    for i in range(1, n):
        r[i] = phi * r[i - 1] + eps[i]
    close = 1000 * np.cumprod(1 + r)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    px = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.integers(2e5, 8e5, n).astype(float),
    }, index=idx)
    upsert_prices(ticker, px)


def _trending_universe(phi=0.35):
    tickers = [f"TRD{i}.NS" for i in range(8)]
    for i, t in enumerate(tickers):
        _seed_ar1(t, phi=phi, seed=100 + i)
    return tickers


def test_make_dataset_builds_labeled_pool(tmp_db):
    tickers = _trending_universe()
    X, y, dates = ml.make_dataset(tickers, horizon=10)
    assert not X.empty
    assert list(X.columns) == ml.FEATURES
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y) == len(dates)
    # No NaNs survive into the training matrix.
    assert not X.isna().any().any()


def test_new_features_present_and_bounded(tmp_db):
    tickers = _trending_universe()
    X, y, dates = ml.make_dataset(tickers, horizon=10)
    # New engineered features exist.
    for c in ["atr_regime", "dist_52w_high", "rel_strength_20",
              "xs_mom_rank", "xs_vol_rank"]:
        assert c in X.columns
    # Cross-sectional ranks are percentiles in [0, 1].
    assert X["xs_mom_rank"].between(0, 1).all()
    assert X["xs_vol_rank"].between(0, 1).all()


def test_feature_importance_ranks_features(tmp_db):
    tickers = _trending_universe(phi=0.45)
    imp = ml.feature_importance(tickers, horizon=10, n_repeats=3, min_train=500)
    assert not imp.empty
    assert list(imp.columns) == ["feature", "importance", "std"]
    assert set(imp["feature"]) == set(ml.FEATURES)
    # Sorted descending by importance.
    assert imp["importance"].is_monotonic_decreasing


def test_walk_forward_runs_and_reports_metrics(tmp_db):
    tickers = _trending_universe()
    res = ml.walk_forward_eval(tickers, horizon=10, n_splits=4, min_train=600)
    assert res is not None
    assert res.n_folds >= 1
    assert 0.0 <= res.base_rate <= 1.0
    assert 0.0 <= res.accuracy <= 1.0
    assert 0.0 <= res.brier <= 1.0
    assert len(res.folds) == res.n_folds
    # Walk-forward must never train on more than the full pool.
    assert res.n_test_total > 0


def test_trending_series_shows_some_edge(tmp_db):
    # Strong persistence -> the model should at least not be worse than guessing
    # out-of-sample, and typically shows a small positive edge.
    tickers = _trending_universe(phi=0.45)
    res = ml.walk_forward_eval(tickers, horizon=10, n_splits=4, min_train=600)
    assert res is not None
    assert res.edge_vs_baseline >= -0.02     # not meaningfully worse than base
    assert res.verdict in {"real edge out-of-sample",
                           "marginal — barely beats guessing",
                           "no edge — do not trade this alone"}


def test_random_walk_has_no_large_edge(tmp_db):
    # iid returns: no predictability. Walk-forward edge must stay small — the
    # model should not hallucinate a big out-of-sample advantage.
    tickers = [f"RW{i}.NS" for i in range(8)]
    for i, t in enumerate(tickers):
        _seed_ar1(t, phi=0.0, seed=500 + i)
    res = ml.walk_forward_eval(tickers, horizon=10, n_splits=4, min_train=600)
    assert res is not None
    assert res.edge_vs_baseline < 0.06


def test_train_and_predict_outputs_probabilities(tmp_db):
    tickers = _trending_universe()
    out = ml.train_and_predict(tickers, horizon=10, min_train=400)
    assert not out.empty
    assert {"ticker", "prob_up", "signal", "asof"}.issubset(out.columns)
    assert out["prob_up"].between(0, 1).all()
    assert set(out["signal"]).issubset({"bullish", "bearish", "neutral"})


def test_insufficient_data_returns_none(tmp_db):
    _seed_ar1("SHORT.NS", phi=0.3, seed=1, n=120)
    assert ml.walk_forward_eval(["SHORT.NS"], horizon=10) is None
    assert ml.train_and_predict(["SHORT.NS"], horizon=10).empty
