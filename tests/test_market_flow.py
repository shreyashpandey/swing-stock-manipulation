from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk import storage
from swingdesk.analyze import market_flow


def _macro(close, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(close), freq="B")
    return pd.DataFrame({"close": close, "volume": np.ones(len(close))}, index=idx)


def test_classify_row_risk_on_trend():
    close = pd.Series(np.linspace(100, 145, 260))
    state = market_flow._classify_row(close, pd.Series(np.full(260, 11.0)))
    assert state.flow == "risk_on_trend"
    assert state.score > 0
    assert state.nifty_trend == "up"


def test_classify_row_risk_off():
    close = pd.Series(np.linspace(145, 100, 260))
    state = market_flow._classify_row(close, pd.Series(np.full(260, 24.0)))
    assert state.flow == "risk_off"
    assert state.score < 0


def test_strategy_fit_maps_setups_to_market_flow():
    assert market_flow.setup_family("breakout_20d") == "momentum_breakout"
    assert market_flow.setup_family("pullback_ema20") == "pullback_continuation"
    assert market_flow.strategy_fit_score("risk_on_trend", "breakout_20d") > \
        market_flow.strategy_fit_score("risk_on_trend", "rsi_reversal")
    assert market_flow.strategy_fit_score("range_chop", "pullback_ema20") > \
        market_flow.strategy_fit_score("range_chop", "breakout_20d")


def test_current_and_historical_flow_from_macro(tmp_db):
    storage.upsert_macro("^NSEI", _macro(np.linspace(100, 145, 260)))
    storage.upsert_macro("^INDIAVIX", _macro(np.full(260, 11.0)))
    state = market_flow.current_flow()
    assert state.flow == "risk_on_trend"
    hist = market_flow.historical_flows()
    assert not hist.empty
    assert set(["date", "flow", "score"]).issubset(hist.columns)


def test_confluence_board_fuses_radar_and_board(monkeypatch):
    monkeypatch.setattr(
        market_flow, "current_flow",
        lambda: market_flow.FlowState("risk_on_trend", 75, "up", "calm", "test"),
    )
    monkeypatch.setattr(
        market_flow.sudden_move,
        "scan",
        lambda *a, **k: pd.DataFrame([
            {"ticker": "A.NS", "radar_score": 80, "readiness": "High",
             "reasons": "compressed", "risks": "—"},
            {"ticker": "B.NS", "radar_score": 40, "readiness": "Low",
             "reasons": "—", "risks": "—"},
        ]),
    )
    monkeypatch.setattr(
        market_flow.board_mod,
        "build_board",
        lambda *a, **k: pd.DataFrame([
            {"ticker": "A.NS", "conviction": 78, "n_gates": 7, "setups": "breakout_20d",
             "manip_tier": "Low", "liq_tier": "liquid", "category": "x", "tags": "Bull"},
            {"ticker": "B.NS", "conviction": 42, "n_gates": 3, "setups": "pullback_ema20",
             "manip_tier": "Elevated", "liq_tier": "liquid", "category": "y", "tags": "Warn"},
        ]),
    )
    monkeypatch.setattr(
        market_flow,
        "one_day_explosive_profile",
        lambda t, target_move_pct=10.0: {
            "target_move_pct": target_move_pct,
            "hist_1d_hit_rate_pct": 5.0 if t == "A.NS" else 0.0,
            "max_1d_high_move_pct": 14.0 if t == "A.NS" else 3.0,
            "p95_1d_high_move_pct": 8.0 if t == "A.NS" else 2.0,
            "plausibility_score": 80.0 if t == "A.NS" else 10.0,
            "reason": "test profile",
        },
    )
    df = market_flow.confluence_board(["A.NS", "B.NS"], limit=2)
    assert list(df.columns) == market_flow.CONFLUENCE_COLS
    assert df.iloc[0]["ticker"] == "A.NS"
    assert df.iloc[0]["preferred_playbook"] == "momentum_breakout"
    assert df.iloc[0]["explosive_score"] > df.iloc[1]["explosive_score"]
    assert "entry_trigger" in df.columns
    assert df.iloc[0]["confluence_score"] > df.iloc[1]["confluence_score"]


def test_one_day_explosive_profile_uses_stock_history(tmp_db):
    idx = pd.date_range("2025-01-01", periods=80, freq="B")
    close = np.linspace(100, 120, 80)
    high = close * 1.02
    high[30] = close[29] * 1.12
    high[55] = close[54] * 1.15
    df = pd.DataFrame({
        "open": close * 0.99,
        "high": high,
        "low": close * 0.98,
        "close": close,
        "volume": np.full(80, 500_000.0),
    }, index=idx)
    storage.upsert_prices("MOVE.NS", df)
    profile = market_flow.one_day_explosive_profile("MOVE.NS", target_move_pct=10.0)
    assert profile["hist_1d_hit_rate_pct"] > 0
    assert profile["max_1d_high_move_pct"] >= 10
    assert profile["plausibility_score"] > 0


def test_playbook_details_are_actionable():
    pb = market_flow.playbook_details("momentum_breakout")
    assert "entry_trigger" in pb
    assert "invalid_if" in pb
    assert "RVOL" in pb["entry_trigger"]


def test_live_explosive_checks_exposes_market_structure(tmp_db):
    idx = pd.date_range("2025-01-01", periods=90, freq="B")
    close = np.linspace(100, 115, 90)
    volume = np.full(90, 500_000.0)
    volume[-1] = 2_000_000.0
    df = pd.DataFrame({
        "open": close * 0.995,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": volume,
    }, index=idx)
    # Force the final close above prior highs so breakout_confirmed can be true.
    df.iloc[-1, df.columns.get_loc("close")] = float(df["high"].iloc[:-1].max() * 1.02)
    df.iloc[-1, df.columns.get_loc("high")] = float(df["close"].iloc[-1] * 1.01)
    storage.upsert_prices("LIVE.NS", df)
    storage.upsert_fundamentals([
        {"ticker": "LIVE.NS", "market_cap": 5e9, "float_shares": 2e7,
         "shares_outstanding": 5e7, "quality_score": 70, "sector": "IT"},
    ])
    checks = market_flow.live_explosive_checks("LIVE.NS")
    assert checks["order_value_mcap_pct"] is not None
    assert checks["order_value_spike_mult"] is not None
    assert checks["volume_float_pct"] is not None
    assert checks["liquidity_tier"] in ("liquid", "moderate", "illiquid", "untradeable")
    assert checks["breakout_confirmed"] is True
    assert checks["live_confirmation_score"] > 0
