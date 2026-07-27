"""Intraday poller for NSE large deals -> Telegram alerts on heavy institutional
SELLs, as close to "as the deal is closed" as public data allows.

What it does
------------
Every ``LIVE_DEALS_POLL_SECS`` during the IST market window it hits NSE's *live*
large-deals feed (``nse.ingest_live_deals``), upserts everything into the same
``deals`` table the app reads, and Telegram-alerts on **new** net-SELL rows whose
value >= ``LIVE_DEALS_MIN_SELL_CR`` crore.

Latency floor is regulatory, not technical: bulk deals surface within ~1h of the
trade, block deals after their window. Nothing public is instant-at-execution.

Run it (from a Terminal with Desktop access, like the app)::

    cd ~/Desktop/StckManipulation
    nohup .venv/bin/python -m swingdesk.live_deals_watch \
        >> ~/Library/Application\\ Support/SwingDesk/deals_watch.log 2>&1 & disown

Stop it::  pkill -f swingdesk.live_deals_watch
"""
from __future__ import annotations

import signal
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from rich.console import Console

from swingdesk.analyze.institutional import _match_marquee
from swingdesk.config import (
    LIVE_DEALS_END_HHMM,
    LIVE_DEALS_MIN_SELL_CR,
    LIVE_DEALS_POLL_SECS,
    LIVE_DEALS_START_HHMM,
    TIMEZONE,
)
from swingdesk.ingest import nse
from swingdesk.notify import telegram

console = Console()
_IST = ZoneInfo(TIMEZONE)
_running = True


def _parse_hhmm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def _in_market_window(now: datetime) -> bool:
    """True on weekdays between the configured start/end (IST)."""
    if now.weekday() > 4:          # Sat/Sun
        return False
    sh, sm = _parse_hhmm(LIVE_DEALS_START_HHMM)
    eh, em = _parse_hhmm(LIVE_DEALS_END_HHMM)
    start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start <= now <= end


def _heavy_sells(new_rows: list[dict]) -> list[dict]:
    """New net-SELL disclosures at/above the ₹cr threshold, biggest first."""
    out = []
    for r in new_rows:
        if (r.get("side") or "").upper() != "SELL":
            continue
        qty = r.get("qty") or 0
        price = r.get("price") or 0
        value_cr = (qty * price) / 1e7
        if value_cr >= LIVE_DEALS_MIN_SELL_CR:
            out.append({**r, "value_cr": round(value_cr, 2)})
    out.sort(key=lambda r: r["value_cr"], reverse=True)
    return out


def _format_alert(sells: list[dict]) -> str:
    lines = [f"🔻 *{len(sells)} heavy institutional SELL(s)* "
             f"(≥ ₹{LIVE_DEALS_MIN_SELL_CR:g} cr, disclosed just now)\n"]
    for r in sells[:15]:
        tk = (r.get("ticker") or "").replace(".NS", "")
        marquee = _match_marquee(r.get("client"))
        star = " ⭐" if marquee else ""
        lines.append(
            f"*{r.get('security') or tk}* (`{tk}`){star}\n"
            f"  {r.get('client') or '?'}\n"
            f"  {r.get('qty', 0):,.0f} @ ₹{r.get('price')} = *₹{r['value_cr']:.2f} cr*"
            f"  · _{r.get('deal_type')} · {r.get('date')}_"
        )
    if len(sells) > 15:
        lines.append(f"…and {len(sells) - 15} more.")
    return "\n".join(lines)


def poll_once(alert: bool = True) -> int:
    """One fetch+upsert cycle. Returns the number of heavy sells alerted on."""
    seen, new_rows = nse.ingest_live_deals()
    sells = _heavy_sells(new_rows)
    console.print(f"[dim]{datetime.now(_IST):%H:%M:%S} · feed={seen} "
                  f"new={len(new_rows)} heavy-sell={len(sells)}[/dim]")
    if alert and sells:
        if telegram.send_message(_format_alert(sells)):
            console.print(f"[green]alerted {len(sells)} heavy sell(s)[/green]")
    return len(sells)


def _stop(signum, frame):
    global _running
    _running = False
    console.print("\n[yellow]deals-watch: stopping…[/yellow]")


def main() -> None:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    tg = "on" if telegram.is_configured() else "OFF (console only — set TELEGRAM_TOKEN/CHAT_ID)"
    console.print(f"[bold]deals-watch[/bold] · poll={LIVE_DEALS_POLL_SECS}s · "
                  f"threshold=₹{LIVE_DEALS_MIN_SELL_CR:g}cr · window={LIVE_DEALS_START_HHMM}-"
                  f"{LIVE_DEALS_END_HHMM} IST · telegram={tg}")

    # Seed once WITHOUT alerting so we don't blast the whole already-disclosed
    # backlog on startup — only deals disclosed after this point trigger alerts.
    seeded, _ = nse.ingest_live_deals()
    console.print(f"[dim]seeded {seeded} deals from current feed (no alerts)[/dim]")

    while _running:
        now = datetime.now(_IST)
        if _in_market_window(now):
            try:
                poll_once(alert=True)
            except Exception as e:                       # never let the loop die
                console.print(f"[red]poll error: {e.__class__.__name__}: {e}[/red]")
            nap = LIVE_DEALS_POLL_SECS
        else:
            console.print(f"[dim]{now:%a %H:%M} outside window — sleeping 30m[/dim]")
            nap = 1800
        # Sleep in 1s slices so SIGTERM is responsive.
        for _ in range(nap):
            if not _running:
                break
            time.sleep(1)

    console.print("[yellow]deals-watch: stopped.[/yellow]")


if __name__ == "__main__":
    main()
