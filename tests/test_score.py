from __future__ import annotations

from datetime import datetime, timedelta

from swingdesk import storage
from swingdesk.analyze import score


def _insert_analyzed(tmp_db, ticker, items):
    """Insert pre-classified news rows for `ticker`. Items: list of (sentiment, impact).
    Uses a recent date (yesterday) so the 7-day filter in recent_sentiment_for_ticker
    always picks them up regardless of when the test runs."""
    recent_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    rows = []
    for i, (s, imp) in enumerate(items):
        rows.append({
            "source": "test",
            "title": f"headline-{i}",
            "link": f"http://x/{ticker}/{i}",
            "published": recent_date,
            "summary": "",
            "tickers": [ticker],
        })
    storage.insert_news(rows)
    # Now fetch them back and assign sentiment
    with storage.connect() as con:
        ids = [r[0] for r in con.execute(
            "SELECT id FROM news WHERE tickers=? ORDER BY id", (ticker,)).fetchall()]
    payload = [
        {"id": rid, "sentiment": s, "impact": imp, "event_type": "earnings", "rationale": "x"}
        for rid, (s, imp) in zip(ids, items)
    ]
    storage.update_news_sentiment(payload)


def test_neutral_when_no_news(tmp_db):
    s = score.sentiment_score("UNKNOWN.NS")
    assert s.score == 50.0
    assert s.raw == 0.0
    assert s.bullish == s.bearish == 0


def test_strongly_bullish(tmp_db):
    _insert_analyzed(tmp_db, "TEST.NS", [
        ("bullish", "high"),
        ("bullish", "high"),
        ("bullish", "medium"),
    ])
    s = score.sentiment_score("TEST.NS")
    assert s.bullish == 3
    assert s.score > 50  # bullish should push above neutral
    assert s.raw == 20 + 20 + 10


def test_strongly_bearish_dominates(tmp_db):
    _insert_analyzed(tmp_db, "TEST.NS", [
        ("bullish", "low"),
        ("bearish", "high"),  # asymmetric: -25 outweighs +3
    ])
    s = score.sentiment_score("TEST.NS")
    assert s.score < 50  # bearish dominates


def test_score_clipped(tmp_db):
    # 10 high-bearish items → raw = -250, but clipped to -50 → score=0
    _insert_analyzed(tmp_db, "TEST.NS", [("bearish", "high")] * 10)
    s = score.sentiment_score("TEST.NS")
    assert s.score == 0.0
    assert s.raw == -250


def test_enrich_adds_composite(tmp_db):
    _insert_analyzed(tmp_db, "TEST.NS", [("bullish", "high")])
    signals = [{"ticker": "TEST.NS", "setup": "breakout_20d", "score": 80.0,
                "entry": 100, "stoploss": 95, "target": 110, "rr": 2.0, "notes": "n"}]
    enriched = score.enrich(signals)
    assert len(enriched) == 1
    e = enriched[0]
    assert "composite_score" in e
    assert "sentiment_score" in e
    # technical=80, sentiment=70 (50 + 20), composite = 0.7*80 + 0.3*70 = 56 + 21 = 77
    assert abs(e["composite_score"] - 77.0) < 0.5


def test_enrich_sorts_by_composite(tmp_db):
    _insert_analyzed(tmp_db, "A.NS", [("bullish", "high"), ("bullish", "high")])
    _insert_analyzed(tmp_db, "B.NS", [("bearish", "high")])
    signals = [
        {"ticker": "B.NS", "setup": "x", "score": 60.0, "notes": ""},
        {"ticker": "A.NS", "setup": "x", "score": 60.0, "notes": ""},
    ]
    enriched = score.enrich(signals)
    assert enriched[0]["ticker"] == "A.NS"  # bullish should rank first
