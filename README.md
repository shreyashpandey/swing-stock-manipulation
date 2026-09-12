# SwingDesk

Local swing-trading signal app for Indian equities (NSE). Runs on your Mac, stores everything in SQLite, no cloud.

**Current state:** local analytics workbench with scanners, charts, backtesting,
portfolio import, risk analysis, fundamentals, news, and saved research workflows.

Open **Discover → Research workspace** for named watchlists, reusable filter screens,
CSV exports, stock comparisons, a sector-grouped return heatmap, and saved daily
price/volume alert conditions. Alert checks are manual and use stored data.

See [the documentation guide](docs/README.md) for product direction, architecture,
and the distinction between implemented features and planned competitor capabilities.
The [business implementation plan](docs/BUSINESS_IMPLEMENTATION_PLAN.md) defines
the proposed path from the local app to a paid web beta.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

(or just `.venv/bin/pip install yfinance pandas pandas-ta-classic feedparser requests streamlit plotly apscheduler python-dotenv sqlalchemy rich beautifulsoup4 lxml`)

## First run

```bash
.venv/bin/python -m swingdesk.cli init        # create SQLite + seed watchlist
.venv/bin/python -m swingdesk.cli run          # fetch prices + news + scan
.venv/bin/python -m swingdesk.cli signals      # print latest signals
```

Then launch the dashboard:

```bash
.venv/bin/streamlit run swingdesk/app.py
```

## CLI

| command | what it does |
|---|---|
| `init` | create SQLite tables + seed default watchlist |
| `prices [--period 2y]` | download OHLCV for watchlist via yfinance |
| `news` | pull RSS (Moneycontrol, ET, Livemint, BS) and tag tickers |
| `sentiment [--max N]` | classify unanalyzed news via Claude (sentiment/impact/event) |
| `scan` | run setup detectors + composite scoring (technical + news) |
| `notify [--limit N]` | push latest signals to Telegram |
| `schedule [--once]` | daemon: pre-market news @ 08:30 IST + full pipeline @ 16:00 IST |
| `run` | prices + news + sentiment + scan + notify (the daily run) |
| `signals [--limit N]` | print latest N signals |
| `watchlist [--set A.NS,B.NS]` | view or replace the watchlist |

## Environment (.env)

```
ANTHROPIC_API_KEY=sk-ant-...
SWINGDESK_CLAUDE_MODEL=claude-opus-4-7    # optional — default is opus-4-7
SWINGDESK_SENTIMENT_BATCH=15              # optional — headlines per Claude call
TELEGRAM_TOKEN=12345:ABC...               # optional, from @BotFather
TELEGRAM_CHAT_ID=12345678                 # optional, from /getUpdates
```

Prompt caching is enabled on the sentiment system prompt — after the first batch of the day every subsequent call pays ~0.1x for the cached rubric.

## Setups detected (Week 1)

- **breakout_20d** — close above 20-day high, on >1.5× avg volume, above 50EMA
- **pullback_ema20** — uptrend pullback touching 20EMA, bullish rebound bar
- **ema_20_50_cross** — 20EMA crosses above 50EMA (positional trend start)
- **volume_thrust** — up-day on >2.5× avg volume

Each signal returns: entry, stoploss (ATR-based), target, R:R, score, notes.

## Watchlist

Defaults to Nifty-50-style names in `swingdesk/config.py`. Edit in the Streamlit sidebar (`Save watchlist`) or via `swingdesk.cli watchlist --set ...`. NSE tickers need the `.NS` suffix, BSE uses `.BO`.

## Product roadmap

The local engine already includes backtesting, portfolio tracking, Groww imports,
and paper trading. Cloud accounts, broker sync, continuous custom alerts, and mobile
apps remain planned. See [competitor feature delivery](docs/COMPETITOR_FEATURE_DELIVERY.md)
for the latest implementation scope and remaining gaps.
