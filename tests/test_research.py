import numpy as np
import pandas as pd
import pytest

from swingdesk import storage
from swingdesk.analyze import research


def prices(ticker, closes, start="2026-09-01", volumes=None):
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes,
                       "close": closes, "volume": volumes if volumes is not None else 100},
                      index=pd.date_range(start, periods=len(closes)))
    storage.upsert_prices(ticker, df)


def test_saved_lists_persist_and_do_not_replace_active_watchlist(tmp_db):
    storage.set_watchlist(["OLD.NS"])
    research.save_item("watchlist", " My list ", {"tickers": [" abc.ns ", "ABC.NS", "XYZ.BO"]})
    storage.init_db()
    assert research.saved_items("watchlist")["My list"]["tickers"] == ["ABC.NS", "XYZ.BO"]
    assert storage.get_watchlist() == ["OLD.NS"]
    research.save_item("watchlist", "My list", {"tickers": ["NEW.NS"]})
    assert research.saved_items("watchlist")["My list"]["tickers"] == ["NEW.NS"]
    research.delete_item("watchlist", "My list")
    assert research.saved_items("watchlist") == {}


@pytest.mark.parametrize("rule", [
    {"metric": "__import__", "op": ">=", "value": 1},
    {"metric": "close", "op": "eval", "value": 1},
    {"metric": "close", "op": ">=", "value": float("nan")},
    {"metric": "close", "op": ">=", "value": None},
])
def test_invalid_rules_rejected(tmp_db, rule):
    with pytest.raises(ValueError):
        research.save_item("screen", "invalid", {"tickers": ["ABC.NS"], "rules": [rule]})
    assert research.saved_items("screen") == {}


def test_snapshot_units_and_prior_volume_baseline(tmp_db):
    prices("ABC.NS", list(range(100, 122)), volumes=[100] * 21 + [500])
    storage.upsert_fundamentals([{"ticker": "ABC.NS", "market_cap": 2e9,
                                 "return_on_equity": .15, "revenue_growth": .2}])
    frame = research.snapshot(["ABC.NS", "MISSING.NS"]).set_index("ticker")
    row = frame.loc["ABC.NS"]
    assert row.change_1d == pytest.approx((121 / 120 - 1) * 100)
    assert row.change_21d == pytest.approx(21)
    assert row.rvol == 5
    assert row.market_cap_cr == 200
    assert row.roe_pct == 15
    assert row.revenue_growth_pct == 20
    assert pd.isna(frame.loc["MISSING.NS", "close"])


def test_filters_and_saved_screen_roundtrip(tmp_db):
    rules = [{"metric": "close", "op": ">=", "value": 100},
             {"metric": "trailing_pe", "op": "<=", "value": 20}]
    research.save_item("screen", "Value", {"tickers": ["A", "B"], "rules": rules})
    frame = pd.DataFrame({"ticker": ["A", "B", "C"], "close": [110, 80, 120], "trailing_pe": [15, 10, np.nan]})
    result = research.filter_snapshot(frame, research.saved_items("screen")["Value"]["rules"])
    assert result.ticker.tolist() == ["A"]
    assert len(research.filter_snapshot(frame, [])) == 3


def test_comparison_uses_common_dates_without_filling(tmp_db):
    prices("A", [100, 200, 300])
    prices("B", [20, 40], start="2026-09-02")
    result = research.comparison_series(["A", "B"])
    assert result.index[0] == pd.Timestamp("2026-09-02")
    assert result["A"].tolist() == [100, 150]
    assert result["B"].tolist() == [100, 200]
    assert research.comparison_series(["A", "UNKNOWN"]).empty


def test_alerts_distinguish_stale_missing_and_matches(tmp_db):
    prices("FRESH", [100, 120], start="2026-09-10")
    prices("OLD", [100, 120], start="2026-08-01")
    prices("BELOW", [80, 90], start="2026-09-10")
    research.save_item("alert", "Price", {"tickers": ["FRESH", "OLD", "BELOW", "MISSING"],
                                          "rules": [{"metric": "close", "op": ">=", "value": 100}]})
    result = research.evaluate_alerts(today="2026-09-11").set_index("ticker")
    assert result.status.to_dict() == {"FRESH": "Condition met", "OLD": "Stale data",
                                      "BELOW": "Not met", "MISSING": "Missing data"}


def test_research_ui_modes(tmp_db):
    from streamlit.testing.v1 import AppTest
    storage.set_watchlist(["A", "B"])
    prices("A", list(range(100, 125)))
    prices("B", list(range(200, 225)))
    app = AppTest.from_string("from swingdesk.research_ui import render\nrender()")
    app.run()
    assert not app.exception
    for mode in ["Saved screens", "Compare", "Heatmap", "Alerts"]:
        app.radio[0].set_value(mode).run()
        assert not app.exception, mode


def test_research_ui_saves_watchlist_and_alert(tmp_db):
    from streamlit.testing.v1 import AppTest
    storage.set_watchlist(["ABC.NS"])
    app = AppTest.from_string("from swingdesk.research_ui import render\nrender()")
    app.run()
    app.text_input[0].set_value("My research")
    app.text_area[0].set_value("ABC.NS, XYZ.BO")
    app.button[0].click().run()
    assert not app.exception
    assert research.saved_items("watchlist")["My research"]["tickers"] == ["ABC.NS", "XYZ.BO"]
    app.radio[0].set_value("Alerts").run()
    app.text_input[0].set_value("Close threshold")
    app.number_input[0].set_value(100)
    app.button[0].click().run()
    assert not app.exception
    assert research.saved_items("alert")["Close threshold"]["rules"][0]["value"] == 100
    app.button[-1].click().run()
    assert not app.exception
