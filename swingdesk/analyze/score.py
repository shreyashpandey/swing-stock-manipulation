"""Combine the technical signal score with recent news sentiment.

The composite score lives in [0, 100]. Weighting:
    composite = 0.7 * technical_score + 0.3 * sentiment_score

Sentiment score is derived from analyzed news in the last `lookback_days`:
    bullish + high   = +20
    bullish + medium = +10
    bullish + low    = +3
    bearish + high   = -25   (asymmetric — bad news weighs more)
    bearish + medium = -12
    bearish + low    = -4
    neutral          =  0
Sum, clip to [-50, +50], then linearly remap to [0, 100] centred at 50.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from swingdesk.storage import get_fundamentals, recent_sentiment_for_ticker

BULLISH_WEIGHTS = {"high": 20, "medium": 10, "low": 3}
BEARISH_WEIGHTS = {"high": -25, "medium": -12, "low": -4}


@dataclass
class SentimentSummary:
    score: float          # 0..100 (50 == neutral)
    raw: float            # signed sum before remap
    bullish: int
    bearish: int
    neutral: int
    top_event: str | None = None  # highest-impact event type seen


def sentiment_score(ticker: str, lookback_days: int = 7) -> SentimentSummary:
    df = recent_sentiment_for_ticker(ticker, days=lookback_days)
    if df.empty:
        return SentimentSummary(score=50.0, raw=0.0, bullish=0, bearish=0, neutral=0)

    raw = 0.0
    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    top_event, top_weight = None, 0.0
    for _, row in df.iterrows():
        s, imp = row["sentiment"], row["impact"]
        counts[s] = counts.get(s, 0) + 1
        if s == "bullish":
            w = BULLISH_WEIGHTS.get(imp, 0)
        elif s == "bearish":
            w = BEARISH_WEIGHTS.get(imp, 0)
        else:
            w = 0
        raw += w
        if abs(w) > abs(top_weight):
            top_weight, top_event = w, row.get("event_type")

    clipped = max(-50.0, min(50.0, raw))
    score = 50.0 + clipped  # maps [-50, +50] → [0, 100]
    return SentimentSummary(
        score=score,
        raw=raw,
        bullish=counts.get("bullish", 0),
        bearish=counts.get("bearish", 0),
        neutral=counts.get("neutral", 0),
        top_event=top_event,
    )


def enrich(signals: list[dict], lookback_days: int = 7,
           use_quality: bool = True) -> list[dict]:
    """Add sentiment_score, quality_score, and composite_score to each signal.

    Composite formula (when quality is available):
        0.55 * technical + 0.20 * sentiment + 0.25 * quality

    When quality is missing (no fundamentals ingested yet):
        0.70 * technical + 0.30 * sentiment
    """
    out = []
    for sig in signals:
        summ = sentiment_score(sig["ticker"], lookback_days=lookback_days)
        technical = float(sig.get("score") or 0.0)
        sent = summ.score
        enriched = dict(sig)
        enriched["technical_score"] = technical
        enriched["sentiment_score"] = round(sent, 1)

        quality = None
        if use_quality:
            f = get_fundamentals(sig["ticker"])
            if f and f.get("quality_score") is not None:
                quality = float(f["quality_score"])
                enriched["quality_score"] = round(quality, 1)

        if quality is not None:
            composite = 0.55 * technical + 0.20 * sent + 0.25 * quality
        else:
            composite = 0.70 * technical + 0.30 * sent
        enriched["composite_score"] = round(composite, 1)
        enriched["score"] = enriched["composite_score"]

        note_parts = [
            f"news: +{summ.bullish}/-{summ.bearish}"
            + (f" ({summ.top_event})" if summ.top_event else ""),
            f"tech={round(technical, 1)}",
            f"sent={round(sent, 1)}",
        ]
        if quality is not None:
            note_parts.append(f"qual={round(quality, 1)}")
        enriched["notes"] = f"{sig.get('notes', '')} | {' | '.join(note_parts)}"
        out.append(enriched)
    out.sort(key=lambda s: s.get("composite_score") or 0, reverse=True)
    return out
