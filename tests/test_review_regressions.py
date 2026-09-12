import sqlite3

import numpy as np
import pandas as pd
import pytest

from swingdesk import storage
from swingdesk.analyze import research, sudden_move


@pytest.mark.parametrize("legacy", [False, "without_exchange", "with_exchange"])
def test_exchange_migration_preserves_rows_and_reingestion(tmp_path, monkeypatch, legacy):
    db = tmp_path / "deals.sqlite"
    monkeypatch.setattr(storage, "DB_PATH", db)
    if legacy:
        ddl = next(s.strip() for s in storage.SCHEMA.split(";")
                   if s.strip().startswith("CREATE TABLE IF NOT EXISTS deals ("))
        if legacy == "without_exchange":
            ddl = ddl.replace("exchange    TEXT DEFAULT 'NSE',", "")
        ddl = ddl.replace("PRIMARY KEY (exchange,", "PRIMARY KEY (")
        with sqlite3.connect(db) as con:
            con.execute(ddl)
            con.execute("INSERT INTO deals (deal_type,date,ticker,security,client,side,qty,price) "
                        "VALUES ('bulk','2026-09-01','TEST.NS',"
                        "'TEST','CLIENT','BUY',100,50)")
    storage.init_db()
    if legacy:
        assert storage.load_deals().exchange.tolist() == ["NSE"]
    row = dict(deal_type="bulk", date="2026-09-01", ticker="TEST.NS",
               security="TEST", client="CLIENT", side="BUY", qty=100, price=50)
    storage.upsert_deals([dict(row, exchange="NSE"), dict(row, exchange="BSE")])
    storage.upsert_deals([dict(row, exchange="BSE", price=51)])
    storage.init_db()
    data = storage.load_deals().set_index("exchange")
    assert len(data) == 2
    assert data.loc["NSE", "price"] == 50
    assert data.loc["BSE", "price"] == 51
    assert len(storage.existing_deal_keys()) == 2


def test_tied_score_metrics_are_permutation_invariant():
    for labels in ([0, 1], [1, 0]):
        assert sudden_move._binary_auc(np.array(labels), np.ones(2)) == .5
        assert sudden_move._average_precision(np.array(labels), np.ones(2)) == .5
    # Mixed tied/untied thresholds: AP = (2/3 * 2/3) + (3/4 * 1/3).
    labels = np.array([1, 0, 1, 1])
    scores = np.array([2, 2, 2, 1])
    assert sudden_move._average_precision(labels, scores) == pytest.approx(25 / 36)
    assert sudden_move._binary_auc(labels, scores) == pytest.approx(1 / 3)


@pytest.mark.parametrize("returns", [[], [-10], [-10, -20], [10, -20, 30]])
def test_signal_outcomes_do_not_claim_portfolio_statistics(returns):
    trades = pd.DataFrame({"date": pd.date_range("2026-09-01", periods=len(returns)),
                           "score": [50] * len(returns),
                           "hit": [r > 0 for r in returns], "ret": returns})
    metrics = sudden_move.ranking_metrics(trades, score_col="score", hit_col="hit",
                                         return_col="ret", ks=(2,))
    assert metrics["sharpe_top_2"] is None
    assert metrics["max_drawdown_top_2_pct"] is None
    assert "daily equity" in metrics["portfolio_metrics_reason"]
    assert "sharpe_top_5" not in metrics
    if returns:
        assert metrics["avg_return_top_2_pct"] == pytest.approx(np.mean(returns))


def test_constant_anomaly_baseline_distinguishes_noise_and_departure():
    history = pd.Series([1.] * 60)
    assert sudden_move._robust_z(1., history) == 0
    assert sudden_move._robust_z(1. + 1e-10, history) == 0
    assert sudden_move._robust_z(3., history) == 4
    assert sudden_move._robust_z(0., history) == -4


def test_research_windows_expose_actual_dates(tmp_db):
    frame = pd.DataFrame({"open": [100, 110], "high": [100, 110],
                          "low": [100, 110], "close": [100, 110], "volume": [10, 10]},
                         index=pd.to_datetime(["2026-09-01", "2026-09-11"]))
    storage.upsert_prices("TEST.NS", frame)
    row = research.snapshot(["TEST.NS"]).iloc[0]
    assert row.change_1d == pytest.approx(10)
    assert row.change_1d_from == "2026-09-01"
    assert row.price_as_of == "2026-09-11"
    assert "stored observation" in research.METRICS["change_1d"]
    assert row.change_5d_from is None
    with pytest.raises(ValueError):
        research.validate_rules([{"metric": "close", "op": ">=", "value": True}])


@pytest.mark.parametrize("populated", [False, True])
def test_metric_views_preserve_missing_and_zero(populated):
    from streamlit.testing.v1 import AppTest
    pulse = {"NIFTY 50": {"close": 25000, "chg_1d": 0}} if populated else {}
    summary = {"roc_auc": 0, "pr_auc": .5} if populated else {}
    app = AppTest.from_string(
        "from swingdesk.metric_ui import render_market_pulse, render_backtest_metrics\n"
        f"render_market_pulse({pulse!r})\nrender_backtest_metrics({summary!r})")
    app.run()
    assert not app.exception
    assert app.metric[0].value == ("25000.00" if populated else "—")
    assert app.metric[1].value == "—"
    assert app.metric[3].value == ("0.00 / 0.50" if populated else "— / —")
    assert app.metric[5].value == "—"
