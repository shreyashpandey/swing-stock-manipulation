from __future__ import annotations

import pandas as pd

from swingdesk.analyze import catalyst


def _news(rows):
    return pd.DataFrame(
        rows, columns=["tickers", "sentiment", "impact", "event_type", "title"])


def test_aggregate_ranks_bullish_with_confirming_move():
    news = _news([
        ("WABAG.NS", "bullish", "high", "order_win", "VA Tech Wabag bags Rs 700 cr order"),
        ("WABAG.NS", "bullish", "medium", "broker_action", "Brokerage positive on Wabag"),
        ("XYZ.NS", "bearish", "high", "regulatory", "XYZ faces probe"),
    ])
    prices = {"WABAG.NS": (6.0, 3.0), "XYZ.NS": (-4.0, 2.0)}
    hits = catalyst.aggregate_catalysts(news, lambda t: prices.get(t, (None, None)),
                                        universe={"XYZ.NS"})
    by = {h.ticker: h for h in hits}
    assert by["WABAG.NS"].direction == "bullish"
    assert by["WABAG.NS"].news_count == 2
    assert by["WABAG.NS"].high_impact == 1
    assert by["WABAG.NS"].in_universe is False          # NEW idea (outside watchlist)
    assert by["XYZ.NS"].direction == "bearish"
    assert by["XYZ.NS"].in_universe is True
    # Wabag outranks XYZ: higher pressure (20+10) plus a confirming up-move.
    assert hits[0].ticker == "WABAG.NS"


def test_confirming_move_boosts_over_news_alone():
    news = _news([
        ("A.NS", "bullish", "high", "order_win", "A wins big order"),
        ("B.NS", "bullish", "high", "order_win", "B wins big order"),
    ])
    # Same news pressure; only A has a confirming price/volume reaction.
    prices = {"A.NS": (8.0, 4.0), "B.NS": (0.1, 1.0)}
    hits = catalyst.aggregate_catalysts(news, lambda t: prices.get(t, (None, None)))
    assert hits[0].ticker == "A.NS"
    assert hits[0].strength > hits[1].strength


def test_min_news_filter():
    news = _news([("AAA.NS", "bullish", "low", "other", "x")])
    assert catalyst.aggregate_catalysts(news, lambda t: (None, None), min_news=2) == []


def test_multi_ticker_headline_explodes():
    news = _news([("TCS.NS,INFY.NS", "bullish", "high", "sector", "IT pack rallies")])
    hits = catalyst.aggregate_catalysts(news, lambda t: (None, None))
    assert {h.ticker for h in hits} == {"TCS.NS", "INFY.NS"}


def test_empty_news():
    assert catalyst.aggregate_catalysts(pd.DataFrame(), lambda t: (None, None)) == []
