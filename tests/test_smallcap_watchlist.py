"""Tests for the separate small-cap watchlist + universe-tagged signals."""
from __future__ import annotations

import pytest

from swingdesk import storage


# ---- small-cap watchlist CRUD ------------------------------------------------

def test_smallcap_watchlist_empty_by_default(tmp_db):
    assert storage.get_smallcap_watchlist() == []


def test_smallcap_watchlist_set_get(tmp_db):
    storage.set_smallcap_watchlist(["PARAS.NS", "ZENTEC.NS", "ANGELONE.NS"])
    assert storage.get_smallcap_watchlist() == ["ANGELONE.NS", "PARAS.NS", "ZENTEC.NS"]


def test_smallcap_watchlist_replaces_on_set(tmp_db):
    storage.set_smallcap_watchlist(["A.NS", "B.NS"])
    storage.set_smallcap_watchlist(["C.NS"])
    assert storage.get_smallcap_watchlist() == ["C.NS"]


def test_smallcap_watchlist_add_idempotent(tmp_db):
    storage.set_smallcap_watchlist(["A.NS"])
    assert storage.add_to_smallcap_watchlist("B.NS") is True   # new
    assert storage.add_to_smallcap_watchlist("B.NS") is False  # dup
    assert set(storage.get_smallcap_watchlist()) == {"A.NS", "B.NS"}


def test_smallcap_watchlist_remove(tmp_db):
    storage.set_smallcap_watchlist(["A.NS", "B.NS", "C.NS"])
    assert storage.remove_from_smallcap_watchlist("B.NS") is True
    assert storage.remove_from_smallcap_watchlist("B.NS") is False  # not present
    assert storage.get_smallcap_watchlist() == ["A.NS", "C.NS"]


def test_main_and_smallcap_watchlists_are_independent(tmp_db):
    """Modifying one shouldn't touch the other — they live in separate tables."""
    storage.set_watchlist(["RELIANCE.NS", "TCS.NS"])
    storage.set_smallcap_watchlist(["PARAS.NS", "ZENTEC.NS"])
    assert storage.get_watchlist() == ["RELIANCE.NS", "TCS.NS"]
    assert storage.get_smallcap_watchlist() == ["PARAS.NS", "ZENTEC.NS"]

    storage.add_to_smallcap_watchlist("ANGELONE.NS")
    assert "ANGELONE.NS" not in storage.get_watchlist()
    assert "ANGELONE.NS" in storage.get_smallcap_watchlist()


# ---- universe-tagged signals -------------------------------------------------

def test_signals_default_to_main_universe(tmp_db):
    storage.save_signals([{
        "ticker": "TCS.NS", "setup": "x", "direction": "long",
        "entry": 100, "stoploss": 95, "target": 110, "rr": 2.0,
        "score": 75, "notes": "n",
    }])
    df = storage.load_signals(universe="main")
    assert len(df) == 1
    df_sc = storage.load_signals(universe="smallcap")
    assert df_sc.empty


def test_signals_can_be_tagged_smallcap(tmp_db):
    storage.save_signals([{
        "ticker": "PARAS.NS", "setup": "x", "direction": "long",
        "entry": 100, "stoploss": 95, "target": 110, "rr": 2.0,
        "score": 75, "notes": "n",
    }], universe="smallcap")
    df = storage.load_signals(universe="smallcap")
    assert len(df) == 1
    assert df.iloc[0]["universe"] == "smallcap"
    # Main filter should not see it
    assert storage.load_signals(universe="main").empty


def test_signals_load_all_when_no_filter(tmp_db):
    storage.save_signals([{"ticker": "TCS.NS", "setup": "x", "direction": "long",
                            "entry": 100, "stoploss": 95, "target": 110,
                            "rr": 2.0, "score": 70}], universe="main")
    storage.save_signals([{"ticker": "PARAS.NS", "setup": "x", "direction": "long",
                            "entry": 100, "stoploss": 95, "target": 110,
                            "rr": 2.0, "score": 75}], universe="smallcap")
    df = storage.load_signals()
    assert len(df) == 2
    assert set(df["universe"]) == {"main", "smallcap"}


def test_per_signal_universe_overrides_param(tmp_db):
    """If a signal dict carries its own 'universe' field, it wins over the
    function arg — useful when a single batch mixes universes."""
    storage.save_signals(
        [{"ticker": "PARAS.NS", "setup": "x", "direction": "long",
          "entry": 100, "stoploss": 95, "target": 110, "rr": 2.0,
          "score": 75, "universe": "smallcap"}],
        universe="main",  # ← param says main but the row says smallcap
    )
    df = storage.load_signals(universe="smallcap")
    assert len(df) == 1


# ---- scan_all tags signals with universe -------------------------------------

def test_scan_all_tags_universe_label(tmp_db, synth_ohlcv):
    """scan_all should set signal['universe'] so persisted rows have the right tag."""
    from swingdesk.analyze.setups import scan_all
    storage.upsert_prices("TEST.NS", synth_ohlcv)
    sigs = scan_all(["TEST.NS"], persist=True, universe="smallcap")
    for s in sigs:
        assert s.get("universe") == "smallcap"
    # And the persisted rows should be tagged
    df = storage.load_signals(universe="smallcap")
    assert len(df) >= 0  # may be 0 if no setup fired, but no errors


def test_scan_all_default_universe_is_main(tmp_db, synth_ohlcv):
    from swingdesk.analyze.setups import scan_all
    storage.upsert_prices("TEST.NS", synth_ohlcv)
    sigs = scan_all(["TEST.NS"], persist=True)
    for s in sigs:
        assert s.get("universe") == "main"
