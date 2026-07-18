from __future__ import annotations

from swingdesk.analyze import global_impact
from swingdesk import storage


def test_map_headline_maps_crude_to_indian_sectors():
    item = global_impact.map_headline(
        "TestFeed",
        "Brent crude oil prices surge as OPEC cuts supply",
        "http://example.com/oil",
        ticker_resolver=lambda sectors: ["ONGC.NS", "BPCL.NS"] if "Oil & Gas" in sectors else [],
    )
    assert item is not None
    assert item.cue == "crude_oil"
    assert item.direction == "positive"
    assert "Oil & Gas" in item.sectors
    assert "Aviation" in item.sectors
    assert item.tickers == ["ONGC.NS", "BPCL.NS"]
    assert item.impact_score > 50


def test_map_headline_ignores_non_market_noise():
    assert global_impact.map_headline(
        "TestFeed", "Celebrity launches a new cooking show", "http://example.com/x",
    ) is None


def test_global_news_storage_roundtrip(tmp_db):
    storage.insert_global_news([{
        "source": "Test",
        "title": "Nasdaq rally boosts AI technology stocks",
        "link": "http://example.com/tech",
        "published": "2026-07-07T09:00:00",
        "summary": "",
        "cue": "us_tech",
        "direction": "positive",
        "impact_score": 72,
        "sectors": ["IT"],
        "tickers": ["TCS.NS", "INFY.NS"],
        "rationale": "test",
    }])
    df = storage.load_global_news(ticker="TCS.NS")
    assert len(df) == 1
    assert df.iloc[0]["cue"] == "us_tech"
    score, reasons = global_impact.ticker_impact_score("TCS.NS", days=30)
    assert score > 0
    assert reasons and "us_tech" in reasons[0]

