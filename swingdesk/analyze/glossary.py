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

**Supertrend (10, 3)** — an explicit BUY/SELL regime line plotted on price.
Green (below price) = bullish, hold longs; red (above price) = bearish. A flip
from red→green is a buy trigger and the line doubles as a trailing stop.

**ADX + DI (14)** — trend *strength*, not direction. **ADX > 25** = a genuine
trend worth trading; **< 20** = chop where breakouts fizzle. **DI+ above DI−** =
bulls in control. This is the "is it even trending?" gate before any entry.

**Stochastic (14,3,3)** — momentum oscillator for *timing*. %K crossing above
%D from **< 20** = oversold buy timing; **> 80** = overbought. Use with a trend
filter (don't buy oversold in a downtrend).

**Bollinger Bands (20, 2σ)** — a volatility envelope around the 20-SMA. Price at
the **upper** band = strong thrust (or overbought); **lower** = weak (or oversold).
A **squeeze** (bands at their tightest in 60 days) precedes a big expansion move.

**CCI (20)** — deviation from the mean: **> +100** strong up-thrust, **< −100**
washed out. A confirmation gauge, not a standalone trigger.

**Risk : reward (R:R)** — every entry signal ships with an entry, a stop-loss and
a target sized off **ATR** (Average True Range, i.e. the stock's own volatility),
so a 2:1 means the target is twice the distance of the stop. We prefer ≥ 2:1.

**Signal scoreboard** — the one-glance verdict at the top of the chart. Ten
indicators (EMA stack, Supertrend, ADX/DI, MACD, RSI, Stochastic, Bollinger %B,
MFI, OBV) each cast a weighted bull/bear vote; the net is normalised to a
−100…+100 score and labelled STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL.
No single indicator can carry the call — confluence does.
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
- **macd_cross** — MACD line crosses above its signal while above the 50-EMA;
  a cross from below zero scores higher (earlier, stronger).
- **supertrend_flip** — Supertrend flips bearish→bullish with price above the
  200-EMA; the Supertrend line becomes the stop.
- **bollinger_breakout** — a volatility *squeeze* (60-day-tight bands) resolving
  with a close above the upper band on **> 1.3×** volume.
- **rsi_reversal** — oversold bounce *inside* an uptrend: RSI was < 35 and turns
  back up above 40 while price holds the 200-EMA. Buy the dip, not the crash.
- **golden_cross** — the 50-EMA crosses above the 200-EMA; a long-horizon
  positional trend-start (wide stop, large target).
- **adx_trend** — ADX rising through 25 with DI+ above DI− and price above the
  20-EMA: enter only when the market is genuinely trending up.

Markers on the chart are colour-coded by *what happened next* historically:
🟢 up-triangle = would have hit target · 🔴 down-triangle = would have stopped
out · ⭐ star = still active. Hover any marker for the exact reason it fired.
"""

# --- Unusual-activity / manipulation detection --------------------------------
MANIPULATION = """\
**Unusual-activity scorecard** — six lenses on whether a stock is being *pushed*
rather than fairly priced. Each scores 0–100 (higher = more unusual); they blend
into one **risk score** (Low < 30 · Elevated 30–60 · High ≥ 60). Four come from
daily OHLCV + market cap / float (yfinance); two come from NSE (delivery %, bulk/
block deals — click **Refresh NSE data** to pull them).

- **Turnover vs market cap** *("order value vs market cap")* — the rupee value
  traded today (close × volume) as a share of the company's market cap, and how
  far that sits above the 20-day median. Flags at **> 3% of market cap** in a day
  or a **5×+** spike over the norm — money churning out of proportion to size.
- **Volume vs float** — today's volume against the 20-day average (in σ) *and*
  against the tradable float. Flags at **5×+** average volume or when **> 10% of
  free float** changes hands in a single day — the footprint of someone trading
  size. Falls back to shares-outstanding, then market-cap/price, if float is
  unknown (the read says which basis it used).
- **Abnormal return / circuit** — today's move in standard deviations of the
  stock's own daily-return history, the overnight gap, and any *cluster* of
  near-circuit (**+9.5%+**) up-days in the last 10 sessions — the run-up phase of
  a pump, not one isolated jump. Flags a **4σ+** day or **2+** big up-days.
- **Amihud illiquidity** — average |return| ÷ rupee value traded: how far price
  moves per rupee. Paired with median daily turnover; **< ₹50 cr/day** marks a
  thin stock a small order can shove, so any spike above carries more weight.
- **Delivery %** *(NSE)* — of the shares that traded, how many were actually
  *delivered* vs churned intraday. A price run-up on **falling delivery %** is a
  classic pump tell — the move is intraday churn, not real accumulation. Very low
  delivery (**< ~20%**) on a rising stock is the loudest version.
- **Bulk / block deals** *(NSE)* — large single-party trades NSE forces to be
  disclosed. A *repeated* same-party presence around a stock flags an operator;
  a one-off institutional block usually doesn't.

**Still not wired:** **promoter pledge** / holding changes (rising pledge or
shrinking promoter stake alongside a spike) — needs an NSE/BSE filings source;
listed under *data gaps* until then.

This is a *screen, not a verdict* — a high score means "look closer", not "guilty".
Index-heavyweights legitimately trade huge value; thin small-caps spike for real
news too. Use it to decide where to dig.
"""

# --- One combined block for a single expander ---------------------------------
def full_methodology() -> str:
    return "\n\n---\n\n".join([VERDICT_METHODOLOGY, NORMS, SETUPS])
