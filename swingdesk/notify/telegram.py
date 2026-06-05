"""Telegram push notifications for new signals.

Setup:
1. Talk to @BotFather on Telegram, run /newbot, save the token as TELEGRAM_TOKEN.
2. Start a chat with your new bot, send it any message.
3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates to find your chat id.
4. Put both in .env as TELEGRAM_TOKEN and TELEGRAM_CHAT_ID.
"""
from __future__ import annotations

import requests
from rich.console import Console

from swingdesk.config import TELEGRAM_CHAT_ID, TELEGRAM_TOKEN

console = Console()
TIMEOUT = 10


def is_configured() -> bool:
    return bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)


def send_message(text: str, *, parse_mode: str = "Markdown") -> bool:
    if not is_configured():
        console.print("[yellow]telegram: not configured (set TELEGRAM_TOKEN + TELEGRAM_CHAT_ID)[/yellow]")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            },
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            console.print(f"[red]telegram: {resp.status_code} {resp.text}[/red]")
            return False
        return True
    except requests.RequestException as e:
        console.print(f"[red]telegram: {e}[/red]")
        return False


def format_signal(sig: dict) -> str:
    setup = sig.get("setup", "?")
    ticker = sig.get("ticker", "?").replace(".NS", "")
    entry = sig.get("entry")
    sl = sig.get("stoploss")
    tgt = sig.get("target")
    rr = sig.get("rr")
    composite = sig.get("composite_score") or sig.get("score")
    notes = sig.get("notes") or ""
    return (
        f"*{ticker}*  `{setup}`\n"
        f"entry `{entry}`  sl `{sl}`  tgt `{tgt}`  R:R `{rr}`\n"
        f"score `{composite}`\n"
        f"_{notes}_"
    )


def send_signals(signals: list[dict], *, header: str | None = None) -> int:
    """Push a digest of signals. Returns count sent (1 if sent as one digest, 0 if none)."""
    if not signals:
        return 0
    head = header or f"📈 *SwingDesk* — {len(signals)} new signal(s)"
    body = "\n\n".join(format_signal(s) for s in signals[:10])  # cap to 10 to keep msg short
    text = f"{head}\n\n{body}"
    ok = send_message(text)
    return 1 if ok else 0
