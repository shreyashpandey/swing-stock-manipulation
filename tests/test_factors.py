"""Tests for cross-sectional factor ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk.analyze import factors
from swingdesk.storage import upsert_fundamentals, upsert_prices


def _seed(ticker, drift, vol, seed, n=300, start=500.0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start * np.cumprod(1 + rets)
    idx = pd.date_range("2023-06-01", periods=n, freq="B")
    px = pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": np.full(n, 4e5),
    }, index=idx)
    upsert_prices(ticker, px)


def _fund(ticker, roe, pe, d2e, margin):
    upsert_fundamentals([{
        "ticker": ticker, "return_on_equity": roe, "trailing_pe": pe,
        "debt_to_equity": d2e, "profit_margin": margin,
    }])


def test_factor_table_ranks_winner_first(tmp_db):
    # WINNER: strong positive momentum, low vol, great fundamentals.
    _seed("WINNER.NS", drift=0.004, vol=0.012, seed=1)
    _fund("WINNER.NS", roe=0.30, pe=15, d2e=20, margin=0.25)
    # LAGGARD: negative momentum, high vol, poor fundamentals.
    _seed("LAGGARD.NS", drift=-0.002, vol=0.035, seed=2)
    _fund("LAGGARD.NS", roe=0.05, pe=60, d2e=200, margin=0.03)
    # MIDDLE: flat.
    _seed("MIDDLE.NS", drift=0.0008, vol=0.02, seed=3)
    _fund("MIDDLE.NS", roe=0.15, pe=30, d2e=80, margin=0.12)

    tbl = factors.factor_table(["WINNER.NS", "LAGGARD.NS", "MIDDLE.NS"])
    assert not tbl.empty
    assert tbl.iloc[0]["ticker"] == "WINNER.NS"
    assert tbl.iloc[-1]["ticker"] == "LAGGARD.NS"
    assert tbl.iloc[0]["rank"] == 1
    # Winner should land in the top quintile (bucket 1).
    assert tbl.iloc[0]["quintile"] == 1


def test_factor_columns_present(tmp_db):
    for i, t in enumerate(["A.NS", "B.NS", "C.NS", "D.NS"]):
        _seed(t, drift=0.001 * i, vol=0.02, seed=10 + i)
        _fund(t, roe=0.1 + 0.05 * i, pe=20 + i, d2e=50, margin=0.1)
    tbl = factors.factor_table(["A.NS", "B.NS", "C.NS", "D.NS"])
    for c in ["momentum_z", "low_vol_z", "quality_z", "value_z", "trend_z",
              "composite", "rank", "quintile"]:
        assert c in tbl.columns


def test_missing_fundamentals_still_ranks(tmp_db):
    # No fundamentals at all — should still rank on price factors alone.
    # Low vol so trailing-window momentum is signal- not noise-driven.
    _seed("X.NS", drift=0.004, vol=0.005, seed=1)    # clean strong uptrend
    _seed("Y.NS", drift=-0.003, vol=0.02, seed=2)    # clear downtrend
    _seed("Z.NS", drift=0.0, vol=0.02, seed=3)       # flat
    tbl = factors.factor_table(["X.NS", "Y.NS", "Z.NS"])
    assert not tbl.empty
    assert tbl.iloc[0]["ticker"] == "X.NS"
    assert tbl.iloc[-1]["ticker"] == "Y.NS"


def test_quality_not_dominated_by_debt_scale(tmp_db):
    # All three share identical price paths, so ONLY fundamentals differ.
    # GOOD: high ROE + margin, modest debt. POOR: low ROE/margin, modest debt.
    # The fix z-scores each sub-component, so ROE/margin must move quality_z —
    # not be swamped by debt's larger native scale.
    for t in ["GOOD.NS", "POOR.NS", "MID.NS"]:
        _seed(t, drift=0.001, vol=0.02, seed=42)     # same seed -> same prices
    _fund("GOOD.NS", roe=0.35, pe=25, d2e=40, margin=0.28)
    _fund("POOR.NS", roe=0.04, pe=25, d2e=40, margin=0.02)
    _fund("MID.NS", roe=0.15, pe=25, d2e=40, margin=0.12)
    tbl = factors.factor_table(["GOOD.NS", "POOR.NS", "MID.NS"]).set_index("ticker")
    # Equal debt + identical prices/value -> ranking driven purely by ROE+margin.
    assert tbl.loc["GOOD.NS", "quality_z"] > tbl.loc["MID.NS", "quality_z"]
    assert tbl.loc["MID.NS", "quality_z"] > tbl.loc["POOR.NS", "quality_z"]
    assert tbl.loc["GOOD.NS", "composite"] > tbl.loc["POOR.NS", "composite"]


def test_too_few_names_returns_empty(tmp_db):
    _seed("ONLY.NS", drift=0.001, vol=0.02, seed=1)
    assert factors.factor_table(["ONLY.NS"]).empty


def test_empty_universe(tmp_db):
    assert factors.factor_table([]).empty
