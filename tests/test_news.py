from __future__ import annotations

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
