"""Paper-autotrader tests — account math, execution-cost fills, risk gates
(max-positions, portfolio-heat, free-cash), and the drawdown kill-switch.

All tests run against the isolated tmp_db fixture and pass `signals` explicitly,
so nothing scans the network or touches the real paper book."""
from __future__ import annotations

import pytest

from swingdesk.portfolio import paper_trader as pt
from swingdesk.portfolio import positions as portfolio
from swingdesk.storage import (
    load_autotrader_log,
    load_positions,
    log_autotrader_step,
    upsert_prices,
)


def _seed(ticker: str, synth_ohlcv):
    upsert_prices(ticker, synth_ohlcv)


def _sig(ticker: str, *, entry=100.0, stop=90.0, target=130.0, score=65.0,
         setup="breakout_20d") -> dict:
    return {"ticker": ticker, "setup": setup, "entry": entry, "stoploss": stop,
            "target": target, "score": score, "rr": 3.0}


def test_account_state_empty_book(tmp_db):
    s = pt.account_state(pt.AutoTraderConfig(capital=100_000))
    assert s["equity"] == 100_000
    assert s["n_open"] == 0
    assert s["realized_pnl"] == 0
    assert s["portfolio_heat_pct"] == 0
    assert not s["halted"]


def test_step_opens_and_caps_at_max_positions(tmp_db, synth_ohlcv):
    cfg = pt.AutoTraderConfig(capital=1_000_000, risk_pct=0.5, max_positions=2,
                              max_portfolio_heat_pct=99.0)
    tickers = ["AAA.NS", "BBB.NS", "CCC.NS"]
    for t in tickers:
        _seed(t, synth_ohlcv)
    rep = pt.step(cfg, signals=[_sig(t) for t in tickers])
    assert len(rep.entries) == 2
    assert rep.n_open == 2
    assert any(r["reason"] == "max positions reached" for r in rep.rejected)
    # a run-log row was written
    assert not load_autotrader_log().empty


def test_execution_cost_bumps_the_buy_fill(tmp_db, synth_ohlcv):
    _seed("AAA.NS", synth_ohlcv)
    cfg = pt.AutoTraderConfig(capital=1_000_000, risk_pct=0.5)
    rep = pt.step(cfg, signals=[_sig("AAA.NS", entry=100.0, stop=90.0)])
    assert len(rep.entries) == 1
    assert rep.entries[0]["slip_bps"] >= 0
    pos = load_positions(status="open", is_paper=True).iloc[0]
    # buy fill is worse (higher) than the raw signal entry — that's the cost
    assert float(pos["entry_price"]) > 100.0


def test_portfolio_heat_cap_blocks_extra_entries(tmp_db, synth_ohlcv):
    # risk/trade ≈ (fill-90)*qty; qty=floor(5000/10)=500 → ≈₹5025 incl. fill slip.
    # Heat cap 1.8% (₹18k) fits 3 trades (≈₹15.1k); the 4th (≈₹20.1k) is rejected.
    cfg = pt.AutoTraderConfig(capital=1_000_000, risk_pct=0.5, max_positions=8,
                              max_portfolio_heat_pct=1.8)
    tickers = ["AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS"]
    for t in tickers:
        _seed(t, synth_ohlcv)
    rep = pt.step(cfg, signals=[_sig(t, entry=100.0, stop=90.0) for t in tickers])
    assert len(rep.entries) == 3
    assert any("heat" in r["reason"] for r in rep.rejected)
    assert rep.portfolio_heat_pct <= 1.8 + 1e-6   # realized heat never breaches the cap


def test_score_gate_rejects_weak_signals(tmp_db, synth_ohlcv):
    _seed("AAA.NS", synth_ohlcv)
    cfg = pt.AutoTraderConfig(capital=1_000_000, min_score=60.0)
    rep = pt.step(cfg, signals=[_sig("AAA.NS", score=45.0)])
    assert not rep.entries
    assert any("score" in r["reason"] for r in rep.rejected)


def test_force_halt_blocks_all_entries(tmp_db, synth_ohlcv):
    _seed("AAA.NS", synth_ohlcv)
    cfg = pt.AutoTraderConfig(force_halt=True)
    rep = pt.step(cfg, signals=[_sig("AAA.NS")])
    assert rep.halted
    assert not rep.entries
    assert any("kill-switch" in n for n in rep.notes)


def test_drawdown_trips_kill_switch(tmp_db, synth_ohlcv):
    cfg = pt.AutoTraderConfig(capital=100_000, kill_switch_dd_pct=10.0)
    # Establish a peak in the run log...
    log_autotrader_step({"asof": "2025-01-01", "equity": 100_000,
                         "peak_equity": 100_000})
    # ...then realize an 11% loss.
    _seed("AAA.NS", synth_ohlcv)
    portfolio.open_position("AAA.NS", entry=100.0, stoploss=90.0, target=130.0,
                            is_paper=True, qty=500, skip_earnings_check=True)
    pid = int(load_positions(status="open", is_paper=True).iloc[0]["id"])
    portfolio.close_position(pid, exit_price=78.0, exit_reason="manual")

    s = pt.account_state(cfg)
    assert s["realized_pnl"] == pytest.approx(-11_000.0)
    assert s["drawdown_pct"] >= 10.0
    assert s["halted"]

    _seed("BBB.NS", synth_ohlcv)
    rep = pt.step(cfg, signals=[_sig("BBB.NS")])
    assert rep.halted
    assert not rep.entries


def test_flatten_all_closes_open_positions(tmp_db, synth_ohlcv):
    # Unreachable stop/target so mark-to-market never auto-closes them first.
    cfg = pt.AutoTraderConfig(capital=1_000_000, risk_pct=0.5)
    for t in ["AAA.NS", "BBB.NS"]:
        _seed(t, synth_ohlcv)
    pt.step(cfg, signals=[_sig(t, entry=100.0, stop=1.0, target=10_000.0)
                          for t in ["AAA.NS", "BBB.NS"]])
    assert pt.account_state(cfg)["n_open"] == 2
    n = pt.flatten_all()
    assert n == 2
    assert load_positions(status="open", is_paper=True).empty
