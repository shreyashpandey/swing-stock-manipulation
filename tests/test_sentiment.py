"""Sentiment tests — mock the Anthropic client so no network calls happen."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from swingdesk import storage
from swingdesk.analyze import sentiment


def _seed_unanalyzed_news(n: int = 3) -> list[int]:
    rows = [
        {"source": "test", "title": f"Headline {i}", "link": f"http://x/{i}",
         "published": "2026-05-26T10:00:00", "summary": "", "tickers": ["RELIANCE.NS"]}
        for i in range(n)
    ]
    storage.insert_news(rows)
    with storage.connect() as con:
        return [r[0] for r in con.execute("SELECT id FROM news ORDER BY id").fetchall()]


def _mock_parse_response(ids: list[int]):
    """Build a fake `messages.parse()` return value."""
    items = []
    sentiments = ["bullish", "bearish", "neutral"]
    for i, rid in enumerate(ids):
        item = MagicMock()
        item.model_dump.return_value = {
            "id": rid,
            "sentiment": sentiments[i % 3],
            "impact": "medium",
            "event_type": "earnings",
            "rationale": "test rationale",
        }
        items.append(item)
    parsed = MagicMock()
    parsed.items = items

    resp = MagicMock()
    resp.parsed_output = parsed
    resp.usage = MagicMock(
        cache_read_input_tokens=0,
        input_tokens=100,
        cache_creation_input_tokens=500,
    )
    return resp


def test_ingest_writes_sentiment(tmp_db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    ids = _seed_unanalyzed_news(3)

    with patch("swingdesk.analyze.sentiment.anthropic.Anthropic") as MockAnth:
        client = MagicMock()
        client.messages.parse.return_value = _mock_parse_response(ids)
        MockAnth.return_value = client
        n = sentiment.ingest(max_items=10)

    assert n == 3
    # Verify they are no longer "unanalyzed"
    remaining = storage.load_unanalyzed_news(limit=10)
    assert len(remaining) == 0


def test_ingest_skips_without_api_key(tmp_db, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _seed_unanalyzed_news(3)
    n = sentiment.ingest()
    assert n == 0


def test_ingest_handles_no_unanalyzed(tmp_db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # No news in DB — should return 0 without calling Anthropic
    with patch("swingdesk.analyze.sentiment.anthropic.Anthropic") as MockAnth:
        n = sentiment.ingest()
        MockAnth.assert_not_called()
    assert n == 0


def test_ingest_drops_invalid_ids(tmp_db, monkeypatch):
    """If the model returns an id outside the batch, that row is dropped."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    ids = _seed_unanalyzed_news(2)

    with patch("swingdesk.analyze.sentiment.anthropic.Anthropic") as MockAnth:
        client = MagicMock()
        # Return one valid id and one phantom id
        client.messages.parse.return_value = _mock_parse_response([ids[0], 99999])
        MockAnth.return_value = client
        n = sentiment.ingest(max_items=10)

    # Only the valid row should be updated
    assert n == 1


def test_format_batch_is_compact():
    import pandas as pd
    df = pd.DataFrame([
        {"id": 1, "title": "TCS Q4 beat", "summary": "Revenue up 8%",
         "tickers": "TCS.NS"},
        {"id": 2, "title": "Macro news", "summary": "", "tickers": ""},
    ])
    out = sentiment._format_batch(df)
    assert "id=1" in out
    assert "id=2" in out
    assert "TCS Q4 beat" in out
    assert "[TCS.NS]" in out
