"""APScheduler-backed daemon. Runs the SwingDesk pipeline on Indian market hours.

Jobs (all in Asia/Kolkata):
  08:30  pre-market    — news only (catches overnight/Asia headlines + sentiment)
  16:00  post-close    — full pipeline: prices + news + sentiment + scan + alert
  Mon-Fri only.

Run with:  swingdesk schedule
"""
from __future__ import annotations

import signal
import time

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from rich.console import Console

from swingdesk.analyze import score, sentiment
from swingdesk.analyze.setups import scan_all
from swingdesk.config import DEFAULT_WATCHLIST, MARKET_CLOSE, TIMEZONE
from swingdesk.ingest import news_rss, prices
from swingdesk.notify import telegram
from swingdesk.storage import (
    get_watchlist,
    init_db,
    seed_watchlist_if_empty,
)

console = Console()


def job_news_only() -> None:
    console.rule("[bold cyan]pre-market: news + sentiment")
    seed_watchlist_if_empty(DEFAULT_WATCHLIST)
    wl = get_watchlist()
    n = news_rss.ingest(wl)
    if n:
        sentiment.ingest(max_items=200)


def job_full_pipeline() -> None:
    console.rule("[bold cyan]post-close: full pipeline")
    seed_watchlist_if_empty(DEFAULT_WATCHLIST)
    wl = get_watchlist()
    prices.ingest(wl, period="1y", workers=6)
    news_rss.ingest(wl)
    sentiment.ingest(max_items=200)
    sigs = scan_all(wl, persist=False)  # we re-persist after enrichment
    if not sigs:
        console.print("no setups today")
        return
    enriched = score.enrich(sigs)
    from swingdesk.storage import save_signals
    save_signals(enriched)
    telegram.send_signals(enriched)


def run(once: bool = False) -> None:
    """Start the scheduler. `once=True` runs the full pipeline immediately and exits."""
    init_db()
    if once:
        job_full_pipeline()
        return

    sched = BlockingScheduler(timezone=TIMEZONE)
    sched.add_job(
        job_news_only,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=30),
        id="news_premkt",
        name="pre-market news + sentiment",
        replace_existing=True,
    )
    sched.add_job(
        job_full_pipeline,
        CronTrigger(day_of_week="mon-fri",
                    hour=MARKET_CLOSE[0] + (1 if MARKET_CLOSE[1] >= 30 else 0),
                    minute=0),
        id="full_close",
        name="post-close full pipeline",
        replace_existing=True,
    )

    # graceful shutdown
    def _stop(signum, frame):
        console.print("\n[yellow]shutting down scheduler...[/yellow]")
        sched.shutdown(wait=False)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    console.print(f"[green]scheduler started (tz={TIMEZONE})[/green]")
    for job in sched.get_jobs():
        console.print(f"  - {job.name}: next run {job.next_run_time}")

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        pass
