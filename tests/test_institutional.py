from __future__ import annotations

import pandas as pd

from swingdesk.analyze import institutional as inst


def _deals(rows):
    return pd.DataFrame(
        rows, columns=["date", "ticker", "security", "client", "side", "qty", "price"])


def test_aggregate_flow_net_and_marquee():
    deals = _deals([
        ("2026-06-20", "WABAG.NS", "VA Tech Wabag", "MORGAN STANLEY ASIA SINGAPORE PTE", "BUY", 100000, 1500),
        ("2026-06-20", "WABAG.NS", "VA Tech Wabag", "SOME HNI", "SELL", 40000, 1500),
        ("2026-06-19", "AAA.NS", "Aaa Ltd", "RANDOM TRADER", "BUY", 5000, 100),
    ])
    flows = inst.aggregate_flow(deals)
    by = {f.ticker: f for f in flows}
    assert by["WABAG.NS"].net_qty == 60000
    assert by["WABAG.NS"].net_side == "BUY"
    assert "MORGAN STANLEY" in by["WABAG.NS"].marquee
    assert by["AAA.NS"].marquee == []
    # Marquee names rank first.
    assert flows[0].ticker == "WABAG.NS"


def test_only_marquee_filter():
    deals = _deals([("2026-06-20", "AAA.NS", "Aaa", "RANDOM TRADER", "BUY", 1, 1)])
    assert inst.aggregate_flow(deals, only_marquee=True) == []


def test_marquee_no_false_match_inside_word():
    # 'GIC' must not match inside 'LOGIC SOLUTIONS'.
    deals = _deals([("2026-06-20", "ZZZ.NS", "Zzz", "LOGIC SOLUTIONS LLP", "BUY", 1, 1)])
    assert inst.aggregate_flow(deals)[0].marquee == []


def test_extract_brokerage_action_bullish():
    out = inst.extract_brokerage_action(
        "Jefferies initiates Buy on Aegis Logistics, target Rs 1000")
    assert out is not None
    broker, _action, sentiment = out
    assert broker == "JEFFERIES"
    assert sentiment == "bullish"


def test_extract_brokerage_action_bearish():
    out = inst.extract_brokerage_action("Morgan Stanley downgrades stock to Underweight")
    assert out is not None
    assert out[0] == "MORGAN STANLEY"
    assert out[2] == "bearish"


def test_extract_brokerage_action_none():
    assert inst.extract_brokerage_action("Nifty ends higher led by IT stocks") is None
    assert inst.extract_brokerage_action("") is None
    # Broker named but no directional call -> not an action.
    assert inst.extract_brokerage_action("Jefferies hosts an investor conference") is None


def test_extract_broadened_keywords():
    # "sees upside", "positive catalyst" — real calls the narrow list used to miss.
    up = inst.extract_brokerage_action("Reliance soars, CLSA sees highest upside at Rs 1800")
    assert up is not None and up[0] == "CLSA" and up[2] == "bullish"
    pos = inst.extract_brokerage_action("Citi highlights multiple triggers, positive on Cipla")
    assert pos is not None and pos[2] == "bullish"


def test_extract_goldman_short_name():
    out = inst.extract_brokerage_action("Goldman turns bullish on Reliance, sees upside")
    assert out is not None and out[0] == "GOLDMAN"


def test_extract_ignores_buy_sell_hold_boilerplate_in_summary():
    # The real call (bullish) is in the title; the summary's "buy, sell or hold"
    # boilerplate must not flip it bearish.
    out = inst.extract_brokerage_action(
        "Bajaj Finance: check why JPMorgan is bullish",
        "Should you buy, sell or hold the stock? Here is what brokerages say.")
    assert out is not None and out[0] == "JPMORGAN" and out[2] == "bullish"


def test_extract_bearish_wins_over_soft_bull_word():
    # "Cautious / Neutral" is the actual call; must not be flipped bullish by a
    # stray bull-ish word elsewhere in the headline.
    out = inst.extract_brokerage_action(
        "Motilal Oswal Cautious on Nykaa valuations; maintains Neutral despite upside")
    assert out is not None and out[2] == "bearish"
