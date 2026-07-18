from __future__ import annotations

import pandas as pd

from swingdesk.ingest import news_rss


WATCH = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS", "SBIN.NS"]


def test_matches_exact_symbol():
    hits = news_rss._match_tickers("RELIANCE Q4 results beat estimates", WATCH)
    assert "RELIANCE.NS" in hits


def test_matches_alias():
    hits = news_rss._match_tickers("Infosys announces buyback", WATCH)
    assert "INFY.NS" in hits


def test_matches_hdfc_bank_alias():
    hits = news_rss._match_tickers("HDFC Bank reports higher NIM", WATCH)
    assert "HDFCBANK.NS" in hits


def test_no_match_when_unrelated():
    hits = news_rss._match_tickers("Government revises GDP forecast", WATCH)
    assert hits == []


def test_no_false_match_inside_word():
    # 'ITC' should not match when embedded inside another word like 'WITCH'
    hits = news_rss._match_tickers("A bewitching tale about WITCH hunts", WATCH)
    assert "ITC.NS" not in hits


def test_empty_input():
    assert news_rss._match_tickers("", WATCH) == []
    assert news_rss._match_tickers(None, WATCH) == []


def test_multiple_tickers_in_one_headline():
    hits = news_rss._match_tickers("TCS and Infosys both gain on US deal", WATCH)
    assert set(hits) >= {"TCS.NS", "INFY.NS"}


def test_normalize_date_handles_missing():
    assert news_rss._normalize_date({}) is None


def test_normalize_date_rfc822():
    out = news_rss._normalize_date({"published": "Tue, 27 May 2025 10:30:00 +0530"})
    assert out is not None
    assert "2025-05-27" in out


# --- Universe-wide tagging via fundamentals-derived aliases ----------------------
# These cover the exact gap that made Bharat Forge / VA Tech Wabag / NOCIL /
# Kirloskar / Aegis invisible to the scanner: company-name headlines for stocks
# outside the hand-written alias map.

def test_normalize_company_name_strips_corporate_suffixes():
    assert news_rss._normalize_company_name("Bharat Forge Ltd") == "BHARAT FORGE"
    assert news_rss._normalize_company_name("VA Tech Wabag Limited") == "VA TECH WABAG"
    assert news_rss._normalize_company_name("NOCIL Limited") == "NOCIL"
    assert news_rss._normalize_company_name("Kirloskar Industries Limited") == "KIRLOSKAR INDUSTRIES"
    assert news_rss._normalize_company_name("Aegis Logistics Ltd") == "AEGIS LOGISTICS"


def test_normalize_company_name_rejects_too_short():
    assert news_rss._normalize_company_name("Ltd") is None
    assert news_rss._normalize_company_name("") is None
    assert news_rss._normalize_company_name(None) is None


def test_build_alias_map_derives_and_scopes(monkeypatch):
    df = pd.DataFrame([
        {"ticker": "BHARATFORG.NS", "short_name": "Bharat Forge Ltd"},
        {"ticker": "WABAG.NS", "short_name": "VA Tech Wabag Limited"},
        {"ticker": "OUTSIDE.NS", "short_name": "Outside Universe Ltd"},
    ])
    monkeypatch.setattr(news_rss, "load_fundamentals", lambda *a, **k: df)
    amap = news_rss.build_alias_map({"BHARATFORG.NS", "WABAG.NS"})
    assert amap["BHARATFORG.NS"] == ["BHARAT FORGE"]
    assert amap["WABAG.NS"] == ["VA TECH WABAG"]
    assert "OUTSIDE.NS" not in amap          # out-of-universe tickers excluded


def test_match_via_derived_alias_tags_stock_outside_curated_map():
    uni = ["BHARATFORG.NS", "WABAG.NS"]
    amap = {"BHARATFORG.NS": ["BHARAT FORGE"], "WABAG.NS": ["VA TECH WABAG"]}
    hits = news_rss._match_tickers(
        "VA Tech Wabag bags Rs 500 crore water treatment order", uni, amap)
    assert hits == ["WABAG.NS"]


def test_derived_alias_respects_universe_gate():
    # Even with an alias entry, a ticker NOT in the universe is never returned.
    amap = {"BHARATFORG.NS": ["BHARAT FORGE"]}
    hits = news_rss._match_tickers("Bharat Forge wins defence order", [], amap)
    assert hits == []


def test_build_alias_map_empty_fundamentals(monkeypatch):
    monkeypatch.setattr(news_rss, "load_fundamentals", lambda *a, **k: pd.DataFrame())
    assert news_rss.build_alias_map({"BHARATFORG.NS"}) == {}
