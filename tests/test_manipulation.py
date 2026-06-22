from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk.analyze import manipulation as m


def _pump_ohlcv() -> pd.DataFrame:
    """A thin small-cap with a clustered run-up and a volume explosion at the
    end — the kind of footprint the scanner should light up on."""
    n = 60
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.linspace(100, 110, n)
    # Three near-circuit up-days + a volume blast in the final sessions.
    close[-4:] = [115.0, 126.0, 138.0, 152.0]
    vol = np.full(n, 50_000.0)
    vol[-3:] = [400_000.0, 900_000.0, 1_500_000.0]
    open_ = np.r_[close[0], close[:-1]]  # opens at prior close (no gap by default)
    df = pd.DataFrame(
        {"open": open_, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": vol},
        index=idx,
    )
    return df


def test_detectors_return_scores(synth_ohlcv):
    fund = {"market_cap": 1e11, "shares_outstanding": 1e9, "float_shares": 5e8}
    for fn, args in [
        (m.turnover_vs_marketcap, (synth_ohlcv, fund["market_cap"])),
        (m.volume_float_spike, (synth_ohlcv, fund["float_shares"], fund["shares_outstanding"])),
        (m.abnormal_return, (synth_ohlcv,)),
        (m.amihud_illiquidity, (synth_ohlcv,)),
    ]:
        res = fn(*args)
        assert res is not None
        assert 0 <= res["score"] <= 100
        assert isinstance(res["notes"], list) and res["notes"]


def test_short_history_returns_none():
    tiny = pd.DataFrame(
        {"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [1, 2]}
    )
    assert m.turnover_vs_marketcap(tiny, 1e9) is None
    assert m.volume_float_spike(tiny) is None
    assert m.abnormal_return(tiny) is None
    assert m.amihud_illiquidity(tiny) is None


def test_scorecard_flags_pump_higher_than_normal():
    pump = _pump_ohlcv()
    fund = {"market_cap": 5e9, "shares_outstanding": 5e7, "float_shares": 2e7}
    pump_card = m.scorecard("PUMP.NS", pump, fund)

    rng = np.random.default_rng(7)
    n = 60
    quiet = pd.DataFrame({
        "open": 100 + rng.normal(0, 0.2, n),
        "high": 101 + rng.normal(0, 0.2, n),
        "low": 99 + rng.normal(0, 0.2, n),
        "close": 100 + rng.normal(0, 0.2, n),
        "volume": rng.integers(900_000, 1_100_000, n).astype(float),
    }, index=pd.date_range("2025-01-01", periods=n, freq="B"))
    quiet_card = m.scorecard("QUIET.NS", quiet, {"market_cap": 5e12, "float_shares": 1e9})

    assert pump_card["risk_score"] > quiet_card["risk_score"]
    assert pump_card["tier"] in ("Elevated", "High")
    # The clustered run-up should surface a warning note.
    assert any(n.startswith("⚠") for n in pump_card["notes"])


def test_scorecard_tolerates_missing_fundamentals(synth_ohlcv):
    card = m.scorecard("X.NS", synth_ohlcv, None)
    assert card["risk_score"] is not None          # still scores on volume/return/illiquidity
    assert card["components"]["turnover_mcap"] is None
    assert any("market_cap" in g for g in card["data_gaps"])
    assert any("NSE" in g for g in card["data_gaps"])


def test_float_basis_falls_back_to_shares_outstanding(synth_ohlcv):
    res = m.volume_float_spike(synth_ohlcv, float_shares=None, shares_outstanding=1e9)
    assert res["float_basis"] == "shares outstanding"
    res2 = m.volume_float_spike(synth_ohlcv, float_shares=5e8, shares_outstanding=1e9)
    assert res2["float_basis"] == "free float"


def test_nse_hooks_documented():
    assert set(m.NSE_HOOKS) >= {"delivery_spike", "bulk_block_deals", "promoter_pledge"}


def test_delivery_and_deals_none_on_empty_injection():
    assert m.delivery_spike("X.NS", delivery_df=pd.DataFrame()) is None
    assert m.bulk_block_deals("X.NS", deals_df=pd.DataFrame()) is None


def _delivery_series(values, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.DataFrame({"deliv_pct": values}, index=idx)


def test_delivery_spike_flags_run_up_on_falling_delivery():
    pump = _pump_ohlcv()  # price ramps hard at the end
    # Delivery was healthy ~55%, then collapses to ~12% during the run-up.
    deliv = _delivery_series([55, 54, 56, 53, 55, 52, 54, 30, 18, 12])
    res = m.delivery_spike("PUMP.NS", df=pump, delivery_df=deliv)
    assert res is not None
    assert res["score"] > 50
    assert any(n.startswith("⚠") for n in res["notes"])


def test_delivery_spike_calm_when_delivery_healthy():
    pump = _pump_ohlcv()
    deliv = _delivery_series([55, 54, 56, 53, 55, 52, 54, 58, 56, 57])
    res = m.delivery_spike("OK.NS", df=pump, delivery_df=deliv)
    assert res is not None
    assert res["score"] < 30


def test_bulk_block_deals_flags_repeated_party():
    deals = pd.DataFrame({
        "deal_type": ["bulk", "bulk", "block", "bulk"],
        "client": ["OPERATOR LLP", "OPERATOR LLP", "OPERATOR LLP", "SOMEONE ELSE"],
        "side": ["BUY", "BUY", "BUY", "SELL"],
        "qty": [1e5, 2e5, 5e5, 1e5],
    })
    res = m.bulk_block_deals("X.NS", deals_df=deals)
    assert res is not None
    assert res["max_same_party"] == 3
    assert res["n_deals"] == 4
    assert any("OPERATOR LLP" in n for n in res["notes"])
