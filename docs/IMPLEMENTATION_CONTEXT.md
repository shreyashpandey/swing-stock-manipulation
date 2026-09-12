# SwingDesk Implementation Context

Last reviewed: 2026-07-07

## Latest implementation note: 2026-09-11

Business planning update: [Business implementation plan](BUSINESS_IMPLEMENTATION_PLAN.md)
now consolidates the proposed paid-beta scope, pricing assumptions, migration,
ordered backlog, and launch gates. It is a plan; the production backend, account
ownership and billing described there have not been implemented.

Review update: [Plan risk review](PLAN_RISK_REVIEW.md) records frontend/backend
gaps and Indian regulatory findings. Local reproductions confirmed that gapped
stored observations can be labelled a one-session return and boolean thresholds
pass numeric validation. These findings are now fixed: returns are labelled as
stored-observation windows with actual dates, and boolean thresholds are rejected.
See [implementation review and fixes](CURRENT_IMPLEMENTATION_REVIEW.md). The plan now estimates
62–97 engineering days, adds privacy/session/recovery work, and requires appropriate
permissions and substantive feature review before external exposure.

The app currently uses grouped primary-section navigation and already includes a
disclaimer gate and Settings & Legal. Navigation comments later in this document
describe an earlier revision.

Added **Discover → Research workspace** with named watchlists, saved numeric
screens and CSV export, stock comparison charts, return heatmaps, and manual
price/volume alert checks. UI lives in `swingdesk/research_ui.py`; analysis and
saved-object persistence live in `swingdesk/analyze/research.py`. An additive
`research_items` table stores local objects without changing the main watchlist.

See [delivery details and remaining scope](COMPETITOR_FEATURE_DELIVERY.md) and
[the documentation guide](README.md). Price loading is read-only in the new page;
users refresh data through existing actions. Custom alerts are not scheduled or
connected to Telegram. New tests live in `tests/test_research.py`.

## What This Project Is

SwingDesk is a local-first Python app for Indian equity swing-trading analytics. It uses:

- SQLite as the central data store (`data/swingdesk.sqlite` by default).
- CLI ingestion and analysis jobs in `swingdesk/cli.py`.
- A Streamlit dashboard in `swingdesk/app.py`.
- Offline-friendly tests with a temporary SQLite DB and mocked price ingestion.

The product direction in `docs/PRODUCT_BLUEPRINT.md` is broader and newer than the README. The README still describes "Week 2", while the implementation already includes backtesting, portfolio import, P&L/tax reporting, scanners, fundamentals, intraday analytics, ML probability, execution/TCA, and productization/compliance planning.

For the current no-advisory launch posture, also read:

- `docs/BUSINESS_MODEL_NO_SEBI_ADVISOR.md`
- `docs/COMPLIANT_FEATURE_WIREFRAMES.md`
- `docs/APP_WORKFLOW_AND_WIREFRAME.md`
- `docs/PRODUCTION_ARCHITECTURE.md`

## Core Architecture

`swingdesk/storage.py` is the persistence hub. It owns schema creation/migration and read/write helpers for:

- Prices, intraday prices, macro indicators.
- News and Claude sentiment annotations.
- Signals and watchlists, including small-cap watchlist separation.
- Positions, holdings, plans, imported trades, autotrader logs.
- Fundamentals, earnings calendar, NSE delivery, bulk/block deals.
- Backtest trades.

`swingdesk/config.py` centralizes runtime settings:

- `DB_PATH`, data directory, default NSE watchlist.
- Anthropic model env vars.
- Telegram env vars.
- Portfolio/risk settings.
- Groww charges and tax assumptions.

`swingdesk/cli.py` is the operational entrypoint. Important commands include:

- `init`, `prices`, `news`, `sentiment`, `scan`, `run`, `signals`, `watchlist`.
- `backtest`, `optimize`, `reconcile`.
- `holdings`, `import`, `enrich`, `positions`, `open`, `close`, `mark`, `sync`, `journal`.
- `discover`, `smallcaps`, `sc-watchlist`, `sc-scan`.
- `fundamentals`, `macro`, `nse`, `catalyst`, `institutional`, `screen`, `earnings`.

`swingdesk/app.py` is a large Streamlit app. It has been changed from eager `st.tabs` to a sidebar `selectbox` page router so only the selected section runs. It also caches expensive analysis paths with `st.cache_data` and clears caches after data refreshes.

## Main Analysis Areas

Technical setup detection:

- `swingdesk/analyze/setups.py`: breakout, EMA pullback, MA cross, volume thrust, MACD cross, supertrend flip, Bollinger breakout, RSI reversal, golden cross, ADX trend.
- `swingdesk/analyze/technicals.py`: indicators, trend quality, money flow, volume profile, signal scoreboard.

Scoring and discovery:

- `swingdesk/analyze/score.py`: technical + sentiment + optional quality composite signal score.
- `swingdesk/analyze/discovery.py`: broader universe opportunity scanner.
- `swingdesk/analyze/smallcaps.py`: small-cap scanner with extra safety filters.
- `swingdesk/analyze/factors.py`, `quality.py`, `screener.py`: factor, fundamental quality, and liquidity-aware screening.

Market context and risk:

- `swingdesk/analyze/spillover.py`: macro/regime and US-to-India spillover.
- `swingdesk/analyze/sectors.py`: sector/micro-sector rotation.
- `swingdesk/analyze/expected_range.py`: volatility, expected range, Monte Carlo, target-vs-stop odds.
- `swingdesk/analyze/risk.py`: position sizing, concentration, correlation, portfolio risk.

Scanners and market structure:

- `swingdesk/analyze/manipulation.py`: unusual activity / operator-footprint style scoring.
- `swingdesk/analyze/liquidity.py`, `market_metrics.py`: shared liquidity and turnover metrics.
- `swingdesk/analyze/catalyst.py`: news catalyst scanner.
- `swingdesk/analyze/institutional.py`: bulk/block deals and brokerage headline extraction.
- `swingdesk/analyze/intraday.py`, `execution.py`, `tca.py`: intraday signals, execution scheduling, transaction cost analysis.

Portfolio and reporting:

- `swingdesk/portfolio/holdings.py`: Groww holdings import and holding analysis.
- `swingdesk/portfolio/import_groww.py`: tradebook and tax P&L import.
- `swingdesk/portfolio/positions.py`: manual/paper positions, mark-to-market, trailing stops.
- `swingdesk/portfolio/paper_trader.py`: autotrader step logic with heat and kill-switch controls.
- `swingdesk/analyze/pnl_report.py`, `charges.py`: realized P&L, taxes, charges.

Unified decision/board:

- `swingdesk/analyze/decision.py`: combines factors, trend, expected range, quality, sentiment, macro, manipulation, liquidity, optional ML, sizing, and holding advice into a `Decision`.
- `swingdesk/analyze/board.py`: categorizes/ranks a decision board.

## Current Product/Compliance Tension

`docs/PRODUCT_BLUEPRINT.md` says the product must be a tool, not advisory. It explicitly calls out replacing BUY/SELL/HOLD/AVOID language with descriptive labels such as setup strength, criteria match, rule-implied levels, and probabilities.

The current implementation still uses advisory-style vocabulary in many places:

- `swingdesk/analyze/decision.py`: `STRONG_BUY`, `BUY`, `ACCUMULATE`, `WAIT`, `AVOID`; holding actions `HOLD`, `ADD`, `TRIM`, `EXIT`.
- `swingdesk/analyze/summary.py` and `glossary.py`: explicit BUY/WAIT/AVOID explanations.
- `swingdesk/portfolio/holdings.py` and `allocate.py`: `BUY_MORE`, `HOLD`, `REDUCE`, `SELL`.
- `swingdesk/app.py`: "Invest", "Invest fresh money", "STRONG BUY", "SELL", "recommend" copy.
- `swingdesk/analyze/thesis.py`: prompt asks for recommended action.

This is the biggest strategic gap if the next task is productization. The analytics code can stay mostly intact, but labels/copy and user flow should be reframed to meet the blueprint.

## Navigation State

Blueprint target: 5 primary sections:

- Home
- Discover
- Scanners
- Portfolio
- Lab

Current Streamlit `PAGES` still contains many flat pages:

- My Holdings, P&L & Taxes, Invest, Discover, Small Caps, Signals, Chart, News, Sectors, Manipulation, Scanners, Calendar, Backtest, Optimize, Portfolio, Reconcile, Fundamentals, Raw data.

The implementation already supports lazy page execution via sidebar selectbox, which is a useful performance improvement. The next UI reorg should probably preserve that lazy routing but group pages under the 5-section IA.

## Test Setup

Tests live in `tests/` and use `pytest`.

`tests/conftest.py`:

- Autouse-mocks price fetching/ingestion to avoid network calls.
- Provides `tmp_db`, which monkeypatches `swingdesk.storage.DB_PATH` to a temporary SQLite file.
- Provides synthetic OHLCV fixtures.

Usual local commands:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m swingdesk.cli init
.venv/bin/python -m swingdesk.cli run
.venv/bin/streamlit run swingdesk/app.py
```

Network-backed commands may need credentials or live internet:

- Prices/fundamentals via yfinance.
- RSS/news scraping.
- NSE delivery/deals.
- Claude sentiment/thesis via Anthropic.
- Telegram notifications.

## Working Tree At Review Time

Observed modified files:

- `docs/PRODUCT_BLUEPRINT.md`
- `swingdesk/analyze/decision.py`
- `swingdesk/app.py`
- `swingdesk/cli.py`
- `swingdesk/config.py`
- `swingdesk/ingest/news_rss.py`
- `swingdesk/ingest/news_scrape.py`
- `swingdesk/portfolio/import_groww.py`
- `swingdesk/storage.py`
- `tests/test_news.py`

Observed untracked files:

- `swingdesk/analyze/board.py`
- `swingdesk/analyze/catalyst.py`
- `swingdesk/analyze/charges.py`
- `swingdesk/analyze/institutional.py`
- `swingdesk/analyze/pnl_report.py`
- `swingdesk/analyze/universe.py`
- `tests/test_board.py`
- `tests/test_catalyst.py`
- `tests/test_charges.py`
- `tests/test_institutional.py`
- `tests/test_pnl_report.py`

I did not revert or clean any of those. Treat them as user/work-in-progress changes unless told otherwise.

## Suggested Next Pass

Highest-value next tasks:

1. Run the full test suite and capture failures.
2. Update README to match the current app phase and command surface.
3. Implement the 5-section IA from the blueprint while preserving lazy routing.
4. Reframe advisory labels and copy into descriptive analytics language.
5. Add onboarding/legal/disclaimer state in storage and Streamlit.
6. Split the very large `app.py` into page modules once behavior is stable.

## Added After Review: Sudden Move + Global Impact

Implemented a descriptive "sudden move" workflow:

- `swingdesk/analyze/sudden_move.py`
  - Scores pressure for outsized upside moves using compression, quiet accumulation, pre-breakout position, relative strength, liquidity/float spark, catalyst/global pressure, optional intraday confirmation, and unusual-activity penalty.
  - Exposes `scan(...)` for live ranking and `backtest(...)` for historical hit-rate checks.
  - Backtest checks whether high radar scores were followed by a next-horizon high/close move above a configurable threshold.

- `swingdesk/analyze/global_impact.py`
  - Maps curated global headlines/cues into likely Indian sectors and stored tickers.
  - Scores signed ticker pressure from recent global impact items.

- `swingdesk/ingest/global_news.py`
  - Fetches curated global RSS feeds and persists mapped impacts.

- `swingdesk/storage.py`
  - Added `global_news` table plus `insert_global_news(...)` and `load_global_news(...)`.

- `swingdesk/cli.py`
  - Added `global-news`, `sudden-radar`, and `sudden-backtest` commands.

- `swingdesk/app.py`
  - Added Streamlit scanner panels under `📡 Scanners`:
    - `⚡ Sudden Move Radar`
    - `🌍 Global Impact`
    - Backtest controls inside the sudden-move panel.

Tests added:

- `tests/test_global_impact.py`
- `tests/test_sudden_move.py`

Verification after this addition:

```bash
.venv/bin/python -m py_compile swingdesk/app.py swingdesk/cli.py swingdesk/analyze/sudden_move.py swingdesk/analyze/global_impact.py swingdesk/ingest/global_news.py swingdesk/storage.py
.venv/bin/python -m pytest
```

Latest result: `479 passed, 1 warning`.

## Added After User Feedback: Market-Flow Confluence

Implemented the missing higher-level layer that answers:

- What is the current market flow?
- Which strategy families historically work in this flow?
- Which stocks have the best combined evidence across tabs?

New module:

- `swingdesk/analyze/market_flow.py`
  - Classifies market flow from NIFTY/VIX macro data:
    - `risk_on_trend`
    - `steady_uptrend`
    - `range_chop`
    - `volatile`
    - `risk_off`
    - `unknown`
  - Maps setup detectors to playbook families:
    - `momentum_breakout`
    - `trend_following`
    - `pullback_continuation`
  - Backtests setup performance grouped by historical market flow via `strategy_flow_backtest(...)`.
  - Builds a live fused `confluence_board(...)` using:
    - Sudden Move Radar
    - Board conviction/gates
    - market-flow strategy fit
    - liquidity/manipulation risk penalties

UI/CLI:

- Streamlit `📡 Scanners` now includes `🧭 Confluence`.
- CLI commands added:
  - `market-flow`
  - `market-flow --backtest`
  - `confluence`

Tests added:

- `tests/test_market_flow.py`

Latest verification after this addition: `484 passed, 1 warning`.

## Added After Clarification: Explicit 10-15% One-Day Move View

The Confluence panel was still too abstract for the user's real question:
"Which stocks can rise 10-15% in a day, and what strategy should I actually
watch for?"

Updated `swingdesk/analyze/market_flow.py`:

- Added `one_day_explosive_profile(...)`
  - Reads a stock's own recent history.
  - Measures how often it reached a selected intraday high move, e.g. +10% or +15%.
  - Reports:
    - `hist_1d_hit_rate_pct`
    - `max_1d_high_move_pct`
    - `p95_1d_high_move_pct`
    - `plausibility_score`

- Added `explosive_score(...)`
  - Blends:
    - sudden-move radar score
    - confluence score
    - strategy fit
    - historical one-day move plausibility
    - risk penalty

- Added visible strategy/playbook details:
  - `entry_trigger`
  - `invalid_if`
  - `works_when`

- Added `explosive_move_backtest(...)`
  - Tests whether high-scoring names reached the selected one-day target on the next session.

Streamlit `📡 Scanners → 🧭 Confluence` now presents an "Explosive Move Confluence"
view with:

- target one-day move selector: 5%, 8%, 10%, 12%, 15%
- `explosive_score`
- historical hit rate / max one-day high move
- entry trigger and invalidation for the playbook
- explosive-filter backtest button

CLI added:

- `confluence --target-move 10`
- `explosive-backtest --target-move 10 --min-score 70`

Latest verification after this addition: `486 passed, 1 warning`.

## Added After Factor Visibility Feedback: Explicit Live Explosive Checks

The user asked why the app was not explicitly checking the factors used by real
traders to identify rare +10/+15% days:

- catalyst/news
- RVOL
- breakout
- VWAP hold
- order value vs market cap
- volume vs float
- liquidity/illiquidity
- manipulation footprint

Added `live_explosive_checks(...)` in `swingdesk/analyze/market_flow.py`.

It now exposes these as first-class Confluence columns:

- `live_confirmation_score`
- `breakout_confirmed`
- `vwap_hold`
- `rvol`
- `catalyst_score`
- `order_value_mcap_pct`
- `order_value_spike_mult`
- `volume_float_pct`
- `liquidity_tier`
- `liquidity_score`
- `amihud`
- `manip_tier`

`explosive_score(...)` now includes live confirmation as part of the blend, not
just historical plausibility and radar/confluence alignment.

Streamlit `📡 Scanners → 🧭 Confluence` now shows those raw factors directly in
the table.

Test added:

- `test_live_explosive_checks_exposes_market_structure`

Latest verification after this addition: `487 passed, 1 warning`.
