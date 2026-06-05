"""News sentiment analysis via the Claude API.

Design notes:
- One API call per batch of headlines (15 by default) — minimises round trips.
- Structured outputs via Pydantic + `messages.parse()` — no JSON parsing risk.
- Prompt caching on the system prompt: the rubric is stable across all calls,
  so after the first request every subsequent batch gets the rubric for ~0.1x cost.
- Stable content (system prompt + rubric) is rendered FIRST and the volatile
  per-batch payload LAST, so the cache prefix never shifts.
"""
from __future__ import annotations

import os
from typing import Literal

import anthropic
import pandas as pd
from pydantic import BaseModel, Field
from rich.console import Console

from swingdesk.config import CLAUDE_MODEL, SENTIMENT_BATCH_SIZE
from swingdesk.storage import load_unanalyzed_news, update_news_sentiment

console = Console()

Sentiment = Literal["bullish", "bearish", "neutral"]
Impact = Literal["high", "medium", "low"]

# Stable rubric — kept in a constant so the cache prefix is byte-identical
# across runs. Never interpolate dates, request IDs, or per-batch context here.
SYSTEM_PROMPT = """You are a senior equity analyst at an Indian asset manager.
You read market headlines and classify each one along three axes:

1. sentiment: bullish | bearish | neutral
   - bullish: implies upward pressure on the stock(s) mentioned
   - bearish: implies downward pressure
   - neutral: factual or ambiguous, no clear directional implication

2. impact: high | medium | low
   - high: likely to move the stock >2% on the next trading day
     (earnings beats/misses, M&A, regulatory action, large orders, downgrades)
   - medium: notable but not market-moving in isolation
     (broker target changes, sector commentary, minor wins)
   - low: routine coverage, recap, or macro noise

3. event_type: one of
   earnings | guidance | management | mna | regulatory | broker_action |
   product | order_win | macro | sector | global_macro | other

   Use `global_macro` for headlines about US Fed policy, oil prices,
   China economy, geopolitics, global tech sector moves — anything that
   affects Indian markets indirectly. For these:
   - Fed dovish / rate cut → bullish (FII flows in, INR strengthens)
   - Fed hawkish / rate hike → bearish (FII outflows, INR weakens)
   - Oil price spike → bearish for India broadly (importer), bullish for ONGC
   - US tech rally → bullish for Indian IT (TCS, INFY)
   - China stimulus → bullish for metals (TATASTEEL, HINDALCO)

Be conservative — if a headline is ambiguous, prefer neutral/low.
Indian-equity context: NSE/BSE listed stocks. Names like RELIANCE, TCS,
INFOSYS, HDFC Bank, ICICI Bank, SBI, Adani, Tata Motors, etc. are common.
Global news without clear India impact → neutral/low.

For each item provided by the user, return an entry in the same order with
the three classifications plus a one-line rationale (max 100 chars)."""


class HeadlineAnalysis(BaseModel):
    id: int = Field(description="Echoes the input headline id")
    sentiment: Sentiment
    impact: Impact
    event_type: str
    rationale: str = Field(max_length=200)


class BatchResult(BaseModel):
    items: list[HeadlineAnalysis]


def _format_batch(batch: pd.DataFrame) -> str:
    """Render the volatile per-batch payload. Each row is one line, id-prefixed."""
    lines = []
    for _, row in batch.iterrows():
        tickers = row.get("tickers") or ""
        tag = f" [{tickers}]" if tickers else ""
        title = (row.get("title") or "").strip()
        summary = (row.get("summary") or "").strip()
        # Keep the headline+summary compact — sentiment doesn't need the full body.
        text = f"{title}. {summary}"[:400] if summary else title
        lines.append(f"id={int(row['id'])}{tag}: {text}")
    return "\n".join(lines)


def _analyze_batch(client: anthropic.Anthropic, batch: pd.DataFrame) -> list[dict]:
    """Send one batch through Claude with structured outputs."""
    payload = _format_batch(batch)

    response = client.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        # Per-block cache_control on the system prompt — the rubric is stable
        # across all batches, so subsequent calls hit the cache (~0.1x cost).
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": (
                f"Classify these {len(batch)} headlines. "
                f"Return exactly {len(batch)} items, one per id, in the same order.\n\n"
                f"{payload}"
            ),
        }],
        output_format=BatchResult,
    )

    if response.parsed_output is None:
        console.print("[red]sentiment: parse failed[/red]")
        return []

    # Optional: log cache effectiveness so the user can see it working.
    u = response.usage
    if u.cache_read_input_tokens:
        console.print(
            f"  [dim]cache: read {u.cache_read_input_tokens} | "
            f"new {u.input_tokens} | write {u.cache_creation_input_tokens}[/dim]"
        )

    return [item.model_dump() for item in response.parsed_output.items]


def ingest(max_items: int = 200) -> int:
    """Analyze up to N unanalyzed news rows. Returns count classified."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print("[yellow]sentiment: ANTHROPIC_API_KEY not set — skipping[/yellow]")
        return 0

    df = load_unanalyzed_news(limit=max_items)
    if df.empty:
        console.print("  sentiment: no unanalyzed news")
        return 0

    # Longer timeout + more retries — Opus 4.7 can be slow on batched classification.
    client = anthropic.Anthropic(timeout=120.0, max_retries=3)
    total = 0
    n = SENTIMENT_BATCH_SIZE

    console.print(f"[bold]Analyzing {len(df)} headlines (batch={n})[/bold]")
    for start in range(0, len(df), n):
        batch = df.iloc[start:start + n]
        try:
            results = _analyze_batch(client, batch)
        except anthropic.AuthenticationError:
            console.print("[red]sentiment: bad API key — aborting[/red]")
            break
        except anthropic.APIStatusError as e:
            # 4xx other than auth: log and skip this batch
            console.print(f"[red]sentiment: API error {e.status_code} on batch — skipping[/red]")
            continue
        except Exception as e:
            console.print(f"[yellow]sentiment: batch failed ({e.__class__.__name__}) — skipping[/yellow]")
            continue

        # Drop anything where the model returned an id not in this batch.
        valid_ids = set(batch["id"].astype(int))
        clean = [r for r in results if r.get("id") in valid_ids]
        update_news_sentiment(clean)
        total += len(clean)
        console.print(f"  sentiment: batch {start // n + 1} -> {len(clean)} classified")

    console.print(f"[green]{total} headlines classified[/green]")
    return total
