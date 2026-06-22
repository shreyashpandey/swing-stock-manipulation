from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk.analyze import sectors


def _make_snapshot():
    """Hand-built snapshot: a strong sector, a weak one, with two industries each."""
    rows = []
    # Strong sector "Alpha" — all above trend, good momentum.
    for i in range(4):
        rows.append({"ticker": f"A{i}.NS", "short_name": f"A{i}", "sector": "Alpha",
                     "industry": "AlphaOne" if i < 2 else "AlphaTwo", "last": 100,
                     "ret_1m": 5 + i, "ret_3m": 15 + i, "ret_6m": 25,
                     "above_50": True, "above_200": True,
                     "quality_score": 70, "stock_strength": 80 + i})
    # Weak sector "Beta" — below trend, negative momentum.
    for i in range(4):
        rows.append({"ticker": f"B{i}.NS", "short_name": f"B{i}", "sector": "Beta",
                     "industry": "BetaOne", "last": 50,
                     "ret_1m": -4, "ret_3m": -12, "ret_6m": -20,
                     "above_50": False, "above_200": False,
                     "quality_score": 40, "stock_strength": 20 + i})
    return pd.DataFrame(rows)


def test_rank_groups_orders_strong_first_and_labels_bias():
    snap = _make_snapshot()
    g = sectors.rank_groups(snap, by="sector")
    assert list(g["sector"]) == ["Alpha", "Beta"]
    assert g.iloc[0]["bias"] == sectors.BULLISH
    assert g.iloc[1]["bias"] == sectors.BEARISH
    assert g.iloc[0]["rank"] == 1


def test_rank_groups_skips_tiny_groups():
    snap = _make_snapshot()
    # add a 1-stock "Gamma" sector -> should be dropped (< MIN_GROUP_SIZE)
    snap = pd.concat([snap, pd.DataFrame([{
        "ticker": "G0.NS", "short_name": "G0", "sector": "Gamma", "industry": "G",
        "last": 10, "ret_1m": 1, "ret_3m": 1, "ret_6m": 1,
        "above_50": True, "above_200": True, "quality_score": 50, "stock_strength": 50,
    }])], ignore_index=True)
    g = sectors.rank_groups(snap, by="sector")
    assert "Gamma" not in set(g["sector"])


def test_top_stocks_sorts_by_strength_and_filters():
    snap = _make_snapshot()
    top = sectors.top_stocks(snap, sector="Alpha", n=2)
    assert len(top) == 2
    assert top.iloc[0]["stock_strength"] >= top.iloc[1]["stock_strength"]
    assert set(top["sector"]) == {"Alpha"}

    one = sectors.top_stocks(snap, sector="Alpha", industry="AlphaOne", n=5)
    assert set(one["industry"]) == {"AlphaOne"}
    assert len(one) == 2


def test_event_picks_restricts_to_event_sectors():
    snap = _make_snapshot()
    picks = sectors.event_picks(snap, ["Alpha"], n=3)
    assert not picks.empty
    assert set(picks["sector"]) == {"Alpha"}


def test_event_picks_broad_when_no_sectors():
    snap = _make_snapshot()
    picks = sectors.event_picks(snap, [], n=4)
    # broad: should pull the market leaders (the Alpha names rank top)
    assert picks.iloc[0]["sector"] == "Alpha"


def test_ret_to_score_monotonic_and_bounded():
    assert sectors._ret_to_score(-50) == 0
    assert sectors._ret_to_score(100) == 100
    assert 0 < sectors._ret_to_score(0) < 100
    assert sectors._ret_to_score(10) > sectors._ret_to_score(-10)


def test_empty_snapshot_safe():
    empty = pd.DataFrame()
    assert sectors.rank_groups(empty, by="sector").empty
    assert sectors.top_stocks(empty, sector="X").empty
    assert sectors.event_picks(empty, ["X"]).empty
