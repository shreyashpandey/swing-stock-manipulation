from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from swingdesk.notify import telegram


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(telegram, "TELEGRAM_TOKEN", "test-token")
    monkeypatch.setattr(telegram, "TELEGRAM_CHAT_ID", "12345")


def test_not_configured_returns_false(monkeypatch):
    monkeypatch.setattr(telegram, "TELEGRAM_TOKEN", "")
    monkeypatch.setattr(telegram, "TELEGRAM_CHAT_ID", "")
    assert telegram.is_configured() is False
    assert telegram.send_message("hello") is False


def test_send_message_posts_to_api(configured):
    with patch("swingdesk.notify.telegram.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        ok = telegram.send_message("hi")
    assert ok is True
    call = mock_post.call_args
    assert "/bot test-token /sendMessage".replace(" ", "") in call.args[0]
    assert call.kwargs["json"]["text"] == "hi"
    assert call.kwargs["json"]["chat_id"] == "12345"


def test_send_message_returns_false_on_non_200(configured):
    with patch("swingdesk.notify.telegram.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=400, text="bad")
        ok = telegram.send_message("hi")
    assert ok is False


def test_send_message_handles_exception(configured):
    import requests as r
    with patch("swingdesk.notify.telegram.requests.post",
               side_effect=r.ConnectionError("net down")):
        ok = telegram.send_message("hi")
    assert ok is False


def test_format_signal_contains_key_fields():
    sig = {
        "ticker": "RELIANCE.NS",
        "setup": "breakout_20d",
        "entry": 2500.0,
        "stoploss": 2450.0,
        "target": 2620.0,
        "rr": 2.4,
        "composite_score": 78.5,
        "notes": "good vol",
    }
    text = telegram.format_signal(sig)
    assert "RELIANCE" in text
    assert "breakout_20d" in text
    assert "2500" in text
    assert "78.5" in text


def test_send_signals_empty_list(configured):
    n = telegram.send_signals([])
    assert n == 0


def test_send_signals_caps_at_10(configured):
    sigs = [{"ticker": f"T{i}.NS", "setup": "x", "entry": 100, "stoploss": 95,
             "target": 110, "rr": 2, "score": 70, "notes": ""} for i in range(20)]
    with patch("swingdesk.notify.telegram.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        telegram.send_signals(sigs)
        body = mock_post.call_args.kwargs["json"]["text"]
        # First 10 tickers in, 11+ excluded
        assert "T0" in body and "T9" in body
        assert "T15" not in body
