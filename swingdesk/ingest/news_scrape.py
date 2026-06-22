"""HTML scrapers for news sources that DON'T publish RSS (LiveSquawk,
MarketsMojo). These return the same item dicts as news_rss.fetch_feed so they
flow through insert_news + ticker-matching unchanged.

CAVEAT: unlike RSS, HTML scrapers are brittle — if a site changes its markup the
selector silently returns 0 items (we log it, never crash). Re-check selectors
if a source stops producing items.
"""
from __future__ import annotations

import hashlib
import re

import requests
from bs4 import BeautifulSoup
from rich.console import Console

from swingdesk.ingest.news_rss import _match_tickers

console = Console()
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}


def _get(url: str) -> str:
    r = requests.get(url, headers=_UA, timeout=20)
    r.raise_for_status()
    return r.text


def _stable_link(base: str, title: str) -> str:
    """Synthetic but stable link for headline-only items — satisfies the UNIQUE
    link constraint and dedupes the same headline across fetches."""
    return f"{base}#{hashlib.sha1(title.encode('utf-8')).hexdigest()[:12]}"


def _clean_livesquawk(text: str) -> str:
    """Strip the trailing relative timestamp and 'Show Detail / READ HERE' cruft."""
    t = re.sub(r"\s*\d+\s*(min|mins|minute|minutes|hour|hours|day|days)\s*ago\b.*$",
               "", text, flags=re.I)
    t = re.sub(r"\s*(Show Detail|READ HERE).*$", "", t, flags=re.I)
    return t.strip()


def scrape_livesquawk(watchlist: list[str], html: str | None = None) -> list[dict]:
    url = "https://www.livesquawk.com/latest-news"
    soup = BeautifulSoup(html if html is not None else _get(url), "lxml")
    items, seen = [], set()
    for el in soup.select(".latest_news_each_text"):
        title = _clean_livesquawk(el.get_text(" ", strip=True))
        if len(title) < 20:
            continue
        a = el.find("a", href=True)
        link = a["href"] if a else _stable_link(url, title)
        if link in seen:
            continue
        seen.add(link)
        items.append({"source": "LiveSquawk", "title": title, "link": link,
                      "published": None, "summary": "",
                      "tickers": _match_tickers(title, watchlist)})
    return items


def scrape_marketsmojo(watchlist: list[str], html: str | None = None) -> list[dict]:
    url = "https://www.marketsmojo.com/news/stock-market-news"
    soup = BeautifulSoup(html if html is not None else _get(url), "lxml")
    items, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/news/stock-market-news/" not in href:
            continue
        title = a.get_text(" ", strip=True)
        if len(title) < 35:           # skip nav / category links
            continue
        link = href if href.startswith("http") else "https://www.marketsmojo.com" + href
        if link in seen:
            continue
        seen.add(link)
        items.append({"source": "MarketsMojo", "title": title, "link": link,
                      "published": None, "summary": "",
                      "tickers": _match_tickers(title, watchlist)})
    return items


SCRAPERS = [("LiveSquawk", scrape_livesquawk), ("MarketsMojo", scrape_marketsmojo)]


def ingest(watchlist: list[str]) -> int:
    """Scrape all non-RSS sources and persist via insert_news. Mirrors
    news_rss.ingest's return (count of new rows)."""
    from swingdesk.storage import insert_news
    total = 0
    for name, fn in SCRAPERS:
        try:
            items = fn(watchlist)
            n = insert_news(items)
            total += n
            console.print(f"  scrape: {name:>20} -> {len(items):>3} items ({n} new)")
        except Exception as e:
            console.print(f"[red]scrape failed for {name}: {e}[/red]")
    return total
