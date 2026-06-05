"""Plain-English methodology + norms behind SwingDesk's suggestions.

Single source of truth for the *explanations* the UI shows so traders can see
exactly which thresholds drive a BUY / WAIT / AVOID call. Keep these in sync
with the actual logic in: summary.py (verdict), quality.py (quality score),
setups.py (entry signals), technicals.py (trend-quality + money-flow reads).

Everything here is markdown, ready to drop into a Streamlit expander.
"""
from __future__ import annotations

# --- The headline: how a verdict is reached ------------------------------------
VERDICT_METHODOLOGY = """\
**How the verdict is decided** — SwingDesk blends four independent lenses and
only calls a BUY when several agree (no single indicator can override the rest):

| Verdict | What has to be true |
|---|---|
| 🟢 **STRONG_BUY** | Quality ≥ 75 **and** in uptrend **and** bullish news > bearish **and** a fresh entry signal firing |
| 🟢 **BUY** | Quality ≥ 65 **and** trend intact (above 200-EMA) **and** ≤ 1 bearish story |
| ⚪ **WAIT** | Mixed picture — no clear edge either way, or data still missing |
| 🔴 **AVOID** | Quality < 50, **or** trend broken with bearish news, **or** overbought & overextended |

The four lenses: **Fundamentals** (quality score) · **Trend** (EMAs) ·
**Money flow** (OBV/MFI/volume) · **Sentiment** (recent news).
"""

# --- Each norm / threshold, grouped -------------------------------------------
NORMS = """\
**Trend state (EMAs)** — exponential moving averages of the closing price:
- **Uptrend** — price above *both* the 50- and 200-day EMA. The trend is your friend.
- **Weakening** — above the 200 but below the 50. Long-term up, short-term cooling.
- **Broken** — below both. Trend is down; suggestions turn cautious.
- 20-EMA crossing *above* the 50-EMA is a classic positional trend-*start*.

**RSI (Relative Strength Index, 0–100)** — speed of recent gains vs losses:
- **> 80** overbought — stretched, pullback risk (we flag *avoid/wait*).
- **45–65** healthy — momentum with room to run (counts *toward* a buy).
- **< 30** oversold — a falling knife until it bases; bounce only *after* it turns.

**MFI (Money Flow Index, 0–100)** — RSI but *volume-weighted*, so it tracks rupees not just price:
- **> 80** buyers exhausted · **< 20** sellers exhausted · **50** neutral line.
- MFI **diverging** from price (price up, MFI down) is the early warning we watch for.

**OBV (On-Balance Volume)** — running tally of up-day minus down-day volume:
- Rising with price = **accumulation** (trustworthy move).
- Falling while price rises = **distribution** (bearish divergence — the crowd is exiting quietly).
- A fresh OBV high often *leads* a price breakout.

**Volume vs 20-day average** — conviction behind a move:
- **≥ 1.5×** average on a breakout = real demand (our breakout setup requires this).
- **≥ 2.5×** on an up-day = a volume *thrust*, often precedes continuation.
- A rally on **< 1× / shrinking** volume is suspect — "running on fumes".

**Volume profile (volume-by-price)** — where shares actually changed hands:
- **POC (Point of Control)** — the single most-traded price; acts as a magnet / S-R.
- **Value Area** — the price band holding ~70% of volume; "fair value".
- Price *above* the value area = extended into thin air; *below* = trading at a discount.
- High-volume nodes become natural **support** (below) and **resistance** (above).

**Trend quality (real vs false uptrend)** — five volume checks on an uptrend
(OBV rising, A/D rising, MFI ≥ 50 & not diverging, >55% of volume on up-days,
volume expanding). Score = checks passed ÷ 5:
- **≥ 4/5 → real** (volume-confirmed) · **2–3 → weak** · **< 2 → false** (distribution risk).

**Quality score (0–100, fundamentals)** — weighted blend, generous to growth and
harsh to red flags: **ROE 25% · growth (earnings+revenue) 20% · margins 15% ·
debt 15%**, plus valuation (P/E) and size. ≥ 65 clears the bar; < 50 is a fail.

**Risk : reward (R:R)** — every entry signal ships with an entry, a stop-loss and
a target sized off **ATR** (Average True Range, i.e. the stock's own volatility),
so a 2:1 means the target is twice the distance of the stop. We prefer ≥ 2:1.
"""

# --- The entry setups themselves ----------------------------------------------
SETUPS = """\
**The entry signals (setups) plotted on the chart** — each fires only when *all*
its conditions are met, and each carries a pre-computed entry / stop / target:

- **breakout_20d** — closes above the prior 20-day high, on **≥ 1.5×** average
  volume, while above the 50-EMA. Stop 1.5×ATR, target 3×ATR (≈ 2:1).
- **pullback_ema20** — in an uptrend (20-EMA > 50-EMA), price dips to tag the
  20-EMA and closes back up with RSI 40–60. Buying the dip in a trend.
- **ema_20_50_cross** — the 20-EMA crosses above the 50-EMA: a positional
  trend-start. Wider stop (2×ATR), bigger target (5×ATR).
- **volume_thrust** — an up-day on **> 2.5×** average volume — a demand spike
  that often precedes continuation.

Markers on the chart are colour-coded by *what happened next* historically:
🟢 up-triangle = would have hit target · 🔴 down-triangle = would have stopped
out · ⭐ star = still active. Hover any marker for the exact reason it fired.
"""

# --- One combined block for a single expander ---------------------------------
def full_methodology() -> str:
    return "\n\n---\n\n".join([VERDICT_METHODOLOGY, NORMS, SETUPS])
