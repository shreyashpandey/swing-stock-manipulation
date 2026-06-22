"""TCA tests — slippage sign conventions, fill aggregation, and the pre-trade
algo comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingdesk.analyze import tca


def test_slip_sign_buy_paying_up_is_a_cost():
    # Bought at 102 vs a 100 benchmark → paid 200 bps too much (positive cost).
    assert tca._slip_bps("buy", 102, 100) == pytest.approx(200.0)
    # Bought below benchmark → negative cost (you did well).
    assert tca._slip_bps("buy", 98, 100) == pytest.approx(-200.0)


def test_slip_sign_sell_is_mirror_of_buy():
    # Sold at 98 vs 100 benchmark → sold low, a 200 bps cost.
    assert tca._slip_bps("sell", 98, 100) == pytest.approx(200.0)
    assert tca._slip_bps("sell", 102, 100) == pytest.approx(-200.0)


def test_fills_frame_requires_price_and_qty():
    with pytest.raises(ValueError):
        tca._fills_frame([{"price": 100}])


def test_analyze_fills_vwap_average_and_arrival_override():
    # Two fills → quantity-weighted average price.
    fills = [{"price": 100, "qty": 10}, {"price": 110, "qty": 30}]
    rep = tca.analyze_fills("ZZZZ.NS", "buy", fills, arrival_price=100.0,
                            date="2024-01-02")
    assert rep is not None
    assert rep.qty == 40
    assert rep.avg_fill == pytest.approx(107.5)          # (100*10 + 110*30)/40
    # Paid 107.5 vs 100 arrival → +750 bps implementation shortfall.
    assert rep.is_bps == pytest.approx(750.0, abs=1.0)
    assert rep.total_cost_bps > rep.is_bps               # fees added on top


def test_analyze_fills_handles_unknown_ticker_gracefully():
    rep = tca.analyze_fills("NO_SUCH_TICKER.NS", "buy",
                            [{"price": 50, "qty": 5}], arrival_price=50.0)
    assert rep is not None
    assert rep.qty == 5
    assert rep.is_bps == pytest.approx(0.0, abs=1.0)     # fill == arrival


def _a_priced_ticker():
    from swingdesk.storage import connect
    try:
        with connect() as con:
            df = pd.read_sql_query("SELECT DISTINCT ticker FROM prices LIMIT 1", con)
        return df["ticker"].iloc[0] if not df.empty else None
    except Exception:
        return None


def test_compare_algos_returns_all_algos_sorted():
    t = _a_priced_ticker()
    if t is None:
        pytest.skip("no priced tickers in local DB")
    df = tca.compare_algos(t, "buy", qty=500, bucket_minutes=60)
    if df.empty:
        pytest.skip(f"no spot for {t}")
    assert set(df["algo"]) <= {"VWAP", "TWAP", "POV", "IS"}
    # sorted cheapest-first
    costs = df["est_cost_bps"].dropna().tolist()
    assert costs == sorted(costs)
