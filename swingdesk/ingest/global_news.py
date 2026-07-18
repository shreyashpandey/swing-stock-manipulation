"""Curated global-market RSS ingestion for Indian equity impact mapping."""
from __future__ import annotations

from email.utils import parsedate_to_datetime
from datetime import datetime

import feedparser

from swingdesk.analyze.global_impact import GlobalImpact, map_headline
from swingdesk.storage import insert_global_news


GLOBAL_FEEDS: list[tuple[str, str]] = [
    ("CNBC-World", "https://www.cnbc.com/id/100727362/device/rss/rss.html"),
    ("CNBC-USMarkets", "https://www.cnbc.com/id/15839069/device/rss/rss.html"),
    ("CNBC-AsiaMarkets", "https://www.cnbc.com/id/19832390/device/rss/rss.html"),
    ("MarketWatch-TopStories", "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("MarketWatch-MarketPulse", "https://feeds.marketwatch.com/marketwatch/marketpulse/"),
    ("ET-International", "https://economictimes.indiatimes.com/news/international/rssfeeds/3534138.cms"),
    ("ET-Economy", "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms"),
    ("Moneycontrol-World", "https://www.moneycontrol.com/rss/world-news.xml"),
    ("Moneycontrol-Economy", "https://www.moneycontrol.com/rss/economy.xml"),
]


def _date(entry) -> str | None:
    for key in ("published", "updated", "pubDate"):
        val = entry.get(key)
        if not val:
            continue
        try:
            return parsedate_to_datetime(val).isoformat()
        except Exception:
            try:
                return datetime.fromisoformat(val).isoformat()
            except Exception:
                pass
    return None


def fetch(feed_name: str, url: str, *, limit: int | None = None) -> list[GlobalImpact]:
    parsed = feedparser.parse(url)
    out: list[GlobalImpact] = []
    entries = parsed.entries[:limit] if limit else parsed.entries
    for e in entries:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or f"{url}#{title}").strip()
        summary = (e.get("summary") or e.get("description") or "").strip()
        if not title:
            continue
        mapped = map_headline(feed_name, title, link, published=_date(e), summary=summary)
        if mapped:
            out.append(mapped)
    return out


def ingest(*, per_feed: int | None = 40) -> int:
    """Fetch curated global feeds, map market-moving cues, and persist them."""
    rows = []
    for name, url in GLOBAL_FEEDS:
        for item in fetch(name, url, limit=per_feed):
            rows.append({
                "source": item.source,
                "title": item.title,
                "link": item.link,
                "published": item.published,
                "summary": item.summary,
                "cue": item.cue,
                "direction": item.direction,
                "impact_score": item.impact_score,
                "sectors": item.sectors,
                "tickers": item.tickers,
                "rationale": item.rationale,
            })
    return insert_global_news(rows)

