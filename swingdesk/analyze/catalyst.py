"""News-Catalyst scanner — the news-FIRST view (inverse of discovery.py).

discovery.py is technicals-first over a fixed universe, so it structurally misses
"news/catalyst-driven moves in stocks you don't already track". This module flips
that: it starts from *what's in the news* (any tagged ticker, regardless of
watchlist membership), measures the news pressure, and checks whether price +
volume are reacting. The output is a ranked list of stocks where a catalyst is
actually playing out right now.

SEBI framing: this is descriptive — "these stocks have strong recent news and a
market reaction". It is NOT a recommendation. Levels/▲▼ are observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from swingdesk.analyze.score import BEARISH_WEIGHTS, BULLISH_WEIGHTS
from swingdesk.storage import combined_universe, load_prices, recent_analyzed_news

# A price reaction function: ticker -> (return_pct_over_window, rvol) or (None, None).
PriceFn = Callable[[str], "tuple[float | None, float | None]"]


@dataclass
class CatalystHit:
    ticker: str
    news_pressure: float          # signed: + bullish, - bearish
    direction: str                # "bullish" | "bearish" | "mixed"
    news_count: int
    bullish: int
    bearish: int
    high_impact: int
    ret_pct: float | None         # price move over the window
    rvol: float | None            # latest volume / 20d avg
    top_event: str | None
    top_headline: str | None
    in_universe: bool             # already in watchlist/holdings? (else a NEW idea)
    top_link: str | None = None   # URL of the top headline
    strength: float = 0.0         # ranking score (news + confirming reaction)


def _weight(sentiment: str | None, impact: str | None) -> float:
    if sentiment == "bullish":
        return BULLISH_WEIGHTS.get(impact, 0)
    if sentiment == "bearish":
        return BEARISH_WEIGHTS.get(impact, 0)
    return 0.0


def aggregate_catalysts(news: pd.DataFrame, price_fn: PriceFn,
                        universe: set[str] | None = None,
                        min_news: int = 1, limit: int = 25) -> list[CatalystHit]:
    """Pure core: group already-filtered analyzed news by ticker, score the news
    pressure, attach a price/volume reaction via `price_fn`, and rank. DB-free so
    it unit-tests without fixtures.

    `news` rows need columns: tickers, sentiment, impact, event_type, title.
    """
    if news is None or news.empty:
        return []
    universe = universe or set()

    # Explode the comma-joined tickers column to one (ticker, row) per mention.
    agg: dict[str, dict] = {}
    for r in news.itertuples():
        for tk in str(getattr(r, "tickers", "") or "").split(","):
            tk = tk.strip()
            if not tk:
                continue
            a = agg.setdefault(tk, {"pressure": 0.0, "n": 0, "bull": 0, "bear": 0,
                                    "hi": 0, "top_w": 0.0, "top_event": None,
                                    "top_headline": None, "top_link": None})
            w = _weight(getattr(r, "sentiment", None), getattr(r, "impact", None))
            a["pressure"] += w
            a["n"] += 1
            if getattr(r, "sentiment", None) == "bullish":
                a["bull"] += 1
            elif getattr(r, "sentiment", None) == "bearish":
                a["bear"] += 1
            if getattr(r, "impact", None) == "high":
                a["hi"] += 1
            if abs(w) > abs(a["top_w"]):
                a["top_w"] = w
                a["top_event"] = getattr(r, "event_type", None)
                a["top_headline"] = getattr(r, "title", None)
                a["top_link"] = getattr(r, "link", None)

    hits: list[CatalystHit] = []
    for tk, a in agg.items():
        if a["n"] < min_news:
            continue
        ret, rvol = price_fn(tk)
        pressure = a["pressure"]
        direction = ("bullish" if pressure > 0 else
                     "bearish" if pressure < 0 else "mixed")

        # Strength = news pressure magnitude, boosted when price/volume CONFIRM
        # the news direction (a move on volume in the same direction is the real
        # catalyst, not just chatter).
        strength = abs(pressure)
        if ret is not None and ((pressure > 0 and ret > 0) or (pressure < 0 and ret < 0)):
            strength += min(abs(ret), 15.0)
        if rvol is not None and rvol > 1.5:
            strength += min((rvol - 1.5) * 4, 12.0)

        hits.append(CatalystHit(
            ticker=tk, news_pressure=round(pressure, 1), direction=direction,
            news_count=a["n"], bullish=a["bull"], bearish=a["bear"],
            high_impact=a["hi"],
            ret_pct=round(ret, 2) if ret is not None else None,
            rvol=round(rvol, 2) if rvol is not None else None,
            top_event=a["top_event"], top_headline=a["top_headline"],
            in_universe=tk in universe, top_link=a["top_link"],
            strength=round(strength, 1),
        ))

    hits.sort(key=lambda h: h.strength, reverse=True)
    return hits[:limit]


def _price_reaction(ticker: str, days: int) -> tuple[float | None, float | None]:
    """(% return over the last `days` sessions, latest volume / prior-20d avg)."""
    df = load_prices(ticker)
    if df is None or df.empty or "close" not in df:
        return None, None
    close = df["close"].dropna()                    # stale/holiday rows can be NaN
    if len(close) < days + 1:
        return None, None
    first, last = float(close.iloc[0 if len(close) <= days else -(days + 1)]), float(close.iloc[-1])
    ret = (last / first - 1) * 100 if first else None
    vol = df["volume"].dropna()
    prior20 = vol.iloc[-21:-1].mean() if len(vol) >= 21 else None
    rvol = float(vol.iloc[-1]) / prior20 if prior20 else None
    return ret, rvol


def scan_catalysts(days: int = 3, limit: int = 25, min_news: int = 1) -> list[CatalystHit]:
    """Live scan: analyzed news from the last `days`, ranked by catalyst strength.
    Universe-agnostic — any tagged ticker can surface, flagged `in_universe` so
    you can spot ideas OUTSIDE your watchlist (the Bharat-Forge / Wabag gap)."""
    news = recent_analyzed_news()
    if news.empty:
        return []
    news = news.copy()
    pub = pd.to_datetime(news["published"], errors="coerce", utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    news = news[pub.isna() | (pub >= cutoff)]      # keep in-window + undated rows
    universe = set(combined_universe())            # watchlist + holdings = "known"
    return aggregate_catalysts(news, lambda t: _price_reaction(t, days),
                               universe=universe, min_news=min_news, limit=limit)
