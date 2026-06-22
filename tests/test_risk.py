"""Tests for the risk & position-sizing layer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk.analyze import risk
from swingdesk.storage import replace_holdings, upsert_prices


def _seed_stock(ticker, rets, start=500.0):
    close = start * np.cumprod(1 + rets)
    idx = pd.date_range("2024-01-01", periods=len(rets), freq="B")
    px = pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": np.full(len(rets), 4e5),
    }, index=idx)
    upsert_prices(ticker, px)


# ---- fixed-risk sizing -----------------------------------------------------
def test_position_size_risk_budget(tmp_db):
    ps = risk.position_size(entry=100, stoploss=95, capital=100_000, risk_pct=1.0)
    assert ps is not None
    # 1% of 100k = ₹1000 risk; ₹5/share -> 200 shares; loss = exactly ₹1000.
    assert ps.shares == 200
    assert ps.risk_amount == 1000.0
    assert ps.pct_of_capital == 20.0


def test_position_size_concentration_cap(tmp_db):
    # Tight ₹1 stop would buy 1000 shares = 100% of capital; must cap at 25%.
    ps = risk.position_size(entry=100, stoploss=99, capital=100_000, risk_pct=1.0,
                            max_position_pct=25.0)
    assert ps.shares == 250
    assert "capped" in ps.note
    assert ps.pct_of_capital == 25.0


def test_position_size_invalid(tmp_db):
    assert risk.position_size(entry=100, stoploss=100) is None
    assert risk.position_size(entry=100, stoploss=110) is None


# ---- vol targeting ---------------------------------------------------------
def test_vol_target_sizing_smaller_for_volatile(tmp_db):
    rng = np.random.default_rng(1)
    _seed_stock("CALM.NS", rng.normal(0, 0.01, 200))
    _seed_stock("WILD.NS", rng.normal(0, 0.04, 200))
    calm = risk.vol_target_size("CALM.NS", capital=100_000)
    wild = risk.vol_target_size("WILD.NS", capital=100_000)
    assert calm and wild
    # Same risk budget -> calmer stock gets a larger rupee allocation.
    assert calm["position_value"] > wild["position_value"]


# ---- concentration ---------------------------------------------------------
def _holdings(values):
    rows = [{"ticker": t, "qty": 1, "avg_price": 100, "last_price": v,
             "invested": v, "current_value": v, "pnl": 0, "pnl_pct": 0}
            for t, v in values.items()]
    replace_holdings(rows)


def test_concentration_flags_single_name(tmp_db):
    _holdings({"BIG.NS": 70_000, "A.NS": 10_000, "B.NS": 10_000, "C.NS": 10_000})
    c = risk.concentration()
    assert c is not None
    assert c.top_name == "BIG.NS"
    assert c.top_weight_pct == 70.0
    assert c.hhi > 0.2
    assert any("single-name" in f for f in c.flags)


def test_concentration_balanced_is_clean(tmp_db):
    _holdings({f"S{i}.NS": 10_000 for i in range(10)})
    c = risk.concentration()
    assert c.top_weight_pct == 10.0
    assert c.effective_names > 8
    assert c.flags == []


# ---- correlation -----------------------------------------------------------
def test_correlated_pairs_detected(tmp_db):
    rng = np.random.default_rng(5)
    base = rng.normal(0, 0.02, 200)
    _seed_stock("TWIN1.NS", base + rng.normal(0, 0.001, 200))   # ~identical
    _seed_stock("TWIN2.NS", base + rng.normal(0, 0.001, 200))
    _seed_stock("INDEP.NS", rng.normal(0, 0.02, 200))           # unrelated
    pairs = risk.correlated_pairs(["TWIN1.NS", "TWIN2.NS", "INDEP.NS"], threshold=0.7)
    assert not pairs.empty
    top = pairs.iloc[0]
    assert {top["a"], top["b"]} == {"TWIN1.NS", "TWIN2.NS"}
    assert top["corr"] > 0.7


def test_portfolio_risk_report(tmp_db):
    rng = np.random.default_rng(9)
    base = rng.normal(0, 0.02, 200)
    _seed_stock("TWIN1.NS", base + rng.normal(0, 0.001, 200))
    _seed_stock("TWIN2.NS", base + rng.normal(0, 0.001, 200))
    _holdings({"TWIN1.NS": 50_000, "TWIN2.NS": 50_000})
    rep = risk.portfolio_risk_report()
    assert rep["ok"] is True
    assert rep["concentration"] is not None
    assert len(rep["suggestions"]) >= 1


def test_sector_concentration_flags_dominant_sector(tmp_db):
    from swingdesk.storage import upsert_fundamentals
    # 4 IT names + 1 bank: IT should dominate and be flagged.
    for t in ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"]:
        upsert_fundamentals([{"ticker": t, "sector": "Technology"}])
    upsert_fundamentals([{"ticker": "HDFCBANK.NS", "sector": "Financial Services"}])
    _holdings({"TCS.NS": 20_000, "INFY.NS": 20_000, "WIPRO.NS": 20_000,
               "HCLTECH.NS": 20_000, "HDFCBANK.NS": 20_000})
    sc = risk.sector_concentration()
    assert not sc.empty
    top = sc.iloc[0]
    assert top["sector"] == "Technology"
    assert top["weight_pct"] == 80.0
    assert top["n_names"] == 4
    assert bool(top["flagged"]) is True


def test_sector_concentration_unknown_when_no_fundamentals(tmp_db):
    _holdings({"A.NS": 50_000, "B.NS": 50_000})
    sc = risk.sector_concentration()
    assert not sc.empty
    assert set(sc["sector"]) == {"Unknown"}


def test_report_includes_sectors(tmp_db):
    from swingdesk.storage import upsert_fundamentals
    for t in ["TCS.NS", "INFY.NS"]:
        upsert_fundamentals([{"ticker": t, "sector": "Technology"}])
    _holdings({"TCS.NS": 60_000, "INFY.NS": 40_000})
    rep = risk.portfolio_risk_report()
    assert rep["ok"] is True
    assert rep["sectors"] is not None and not rep["sectors"].empty
    # 100% Technology must produce a sector-concentration suggestion.
    assert any("Technology" in s for s in rep["suggestions"])


def test_report_no_holdings(tmp_db):
    rep = risk.portfolio_risk_report()
    assert rep["ok"] is False
