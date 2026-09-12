# SwingDesk Production Architecture

The [Business implementation plan](BUSINESS_IMPLEMENTATION_PLAN.md) defines the
smaller first deployment: a modular backend, worker, PostgreSQL and responsive web
client under `backend/` and `clients/web/`. The service/streaming architecture below
remains the longer-term design, not a prerequisite for the paid beta.

Last updated: 2026-08-20
Status: implementation blueprint

## Purpose

This document converts the current local-first SwingDesk codebase into a production-ready platform plan.

It answers:

- whether we should create a separate layer for web, Android, and iPhone apps
- how those apps should connect to the analytics engine
- where streaming fits once data volume grows
- how to organize the repo and services without losing the current Python engine

Short answer:

- yes, we should separate the product into layers
- no, mobile/web should not connect directly to `streamlit` or raw SQLite
- yes, we should add a streaming/event layer for live market feeds, alerts, and scanner updates

## Current State

Today the repo is effectively:

```text
Streamlit app
    ->
Python analysis modules
    ->
SQLite
    ->
CLI ingestion jobs
```

That is good for local R&D, but weak for production because:

- one big app process mixes UI, analytics, ingestion, and storage
- SQLite is fine locally but not for multi-user production workloads
- Android/iPhone/web need a stable backend contract
- live updates should not depend on page refreshes
- heavy analytics and backtests should run asynchronously

## Target Architecture

We should move to 6 layers.

```text
Clients
  Web app
  Android app
  iPhone app
        |
        v
API Gateway / BFF
  auth
  session
  plan enforcement
  response shaping
        |
        v
Product Services
  user service
  watchlist service
  alerts service
  portfolio service
  scanner query service
  admin/compliance service
        |
        +--------------------+
        |                    |
        v                    v
Analytics Engine         Streaming / Event Layer
  factors                market ticks
  scanners               scanner updates
  institutional flow     alert events
  ML scoring             portfolio refresh events
  backtests              ingestion status events
        |                    |
        +---------+----------+
                  |
                  v
Data Platform
  OLTP DB
  market time-series store
  object storage
  cache
  search/index store
```

## Recommended Repo Shape

The cleanest path is a monorepo with shared contracts.

```text
repo/
  apps/
    web/
    mobile/
    ops-admin/
  services/
    api-gateway/
    auth-service/
    portfolio-service/
    alerts-service/
    scanner-service/
    market-data-service/
    ingestion-service/
    notification-service/
  engine/
    swingdesk-core/
    feature-engine/
    ml-models/
    backtest-engine/
  streaming/
    event-schemas/
    consumers/
    producers/
  data/
    migrations/
    warehouse-models/
  infra/
    docker/
    k8s/
    terraform/
  docs/
```

## How To Map The Current Repo Into That Structure

The current `swingdesk` package should become the first engine package, not be thrown away.

### Keep and extract

- `swingdesk/analyze/*` -> `engine/swingdesk-core/analyze/*`
- `swingdesk/backtest/*` -> `engine/backtest-engine/*`
- `swingdesk/ingest/*` -> split between `services/ingestion-service` and `services/market-data-service`
- `swingdesk/portfolio/*` -> `services/portfolio-service` plus some shared engine logic
- `swingdesk/storage.py` -> split into repository layers per service

### Replace over time

- `swingdesk/app.py` -> temporary internal analyst UI only
- direct SQLite access -> service-owned databases
- giant CLI orchestration -> schedulers and workers

## Client Layer

We should not build three fully separate frontends from scratch if avoidable.

Recommended client plan:

- Web: `Next.js`
- Mobile: `React Native` or `Flutter`
- Shared design system: tokens, colors, typography, component rules
- Shared API contract: OpenAPI or typed GraphQL schema

### Why separate client apps from the engine

- web/mobile need auth, account, plan gating, and sync
- app releases should not depend on analytics code deploys
- mobile needs push notifications and offline caching
- clients should receive already-shaped data, not raw dataframes

## Connectivity Layer

We should introduce a backend-for-frontend layer.

### API Gateway / BFF responsibilities

- authenticate users
- validate subscription and entitlements
- merge data from scanner, portfolio, and alerts services
- expose mobile-friendly response payloads
- rate-limit expensive requests
- hide internal service topology from clients

### API style

Use both:

- REST for standard app flows
- WebSocket or Server-Sent Events for live updates

REST examples:

- `GET /v1/home`
- `GET /v1/scanners/sudden-move`
- `GET /v1/scanners/institutional-flow`
- `POST /v1/watchlists`
- `POST /v1/alerts`
- `POST /v1/portfolio/imports`

Realtime examples:

- `scanner.updated`
- `alert.triggered`
- `portfolio.refreshed`
- `institutional_flow.updated`
- `ingestion.status_changed`

## Streaming Layer

Yes, we should add one for production.

But not for every feature on day 1.

### When streaming is truly needed

- live NSE large deals updates
- intraday scanner refreshes
- push alert generation
- market-feed fanout to many consumers
- asynchronous recomputation after ingestion

### When streaming is not needed

- nightly factor ranking
- end-of-day backtests
- static fundamentals refresh
- slow portfolio imports

### Recommended event backbone

Use a message broker such as:

- Kafka for high-throughput, durable event streams
- or RabbitMQ/SQS initially if we want simpler ops early

Practical rollout:

1. Start with a job queue plus event bus abstraction.
2. Move hot paths to Kafka once live feed volume grows.

### Core event topics

```text
market.prices.raw
market.ohlcv.1m
market.deals.disclosed
market.news.ingested
features.stock.updated
scanner.unusual_activity.updated
scanner.sudden_move.updated
scanner.institutional_flow.updated
portfolio.position.changed
alert.rule.triggered
notification.dispatch.requested
notification.dispatch.completed
ml.prediction.generated
system.ingestion.health
```

### Consumers

- feature engine consumer
- scanner materialization consumer
- alert rules consumer
- notification consumer
- audit/compliance consumer
- analytics warehouse loader

## Data Layer

We should split storage by workload.

### 1. OLTP relational database

Use `Postgres`.

Store:

- users
- plans and entitlements
- watchlists
- saved screens
- alerts
- holdings metadata
- imports
- legal acceptance logs
- notification history

### 2. Time-series / market store

Use one of:

- Postgres partitioned tables to start
- TimescaleDB if we want SQL + time-series efficiency
- ClickHouse later for very high-volume analytics

Store:

- OHLCV
- intraday bars
- delivery data
- bulk/block deal rows
- derived market features

### 3. Cache layer

Use `Redis`.

Store:

- hot home/dashboard payloads
- live scanner snapshots
- session state
- rate limits
- websocket fanout state

### 4. Object storage

Use `S3`-style storage.

Store:

- raw vendor dumps
- model artifacts
- backtest reports
- exported CSVs
- logs and audit snapshots

### 5. Search / index layer

Optional, but useful later.

Use `OpenSearch` or similar for:

- news search
- filings/body-text search
- investor-name lookup

## Analytics And Feature Engine

This should become a first-class internal layer.

```text
raw price / deals / news
    ->
normalization
    ->
feature computation
    ->
anomaly detection
    ->
supervised scoring
    ->
materialized scanner outputs
    ->
API responses / alerts
```

### Core sublayers

1. `normalizers`
   - ticker mapping
   - NSE/BSE symbol mapping
   - timezone and session alignment
   - duplicate and correction handling

2. `feature-engine`
   - technicals
   - liquidity
   - market-structure
   - institutional-flow features
   - news sentiment features
   - macro/spillover features

3. `detection-engine`
   - anomaly scores
   - rules engine
   - z-score / isolation forest / clustering jobs

4. `prediction-engine`
   - supervised probability models
   - calibration
   - model versioning
   - offline evaluation

5. `materializers`
   - sudden move table
   - unusual activity table
   - institutional flow table
   - explainer payloads for UI

## Where The Two-Model Setup Lives

Your earlier structure fits here cleanly:

```text
Market Data + F&O + News + Institutional Flow
    ->
Feature Engine
    ->
Model 1: Unusual activity / anomaly engine
    ->
Model 2: Historical move probability engine
    ->
Ranking + explanation + alert eligibility
```

### Model 1

Purpose:

- detect things that look unusual even if we did not define every pattern manually

Examples:

- isolation forest
- autoencoder later
- robust z-scores
- peer-relative abnormal turnover

### Model 2

Purpose:

- estimate the probability of a defined outcome such as `+10% within 5 days`

Examples:

- XGBoost
- LightGBM
- Random Forest
- calibrated logistic regression baseline

### Evaluation layer

These metrics should be stored and versioned per scanner/model:

- `Precision@K`
- `Recall@K`
- `ROC-AUC`
- `PR-AUC`
- `Average Return of Top Ranked`
- `Hit Rate`
- `Sharpe Ratio`
- `Max Drawdown`
- calibration error
- coverage and staleness

## Institutional Flow And Major Players

This deserves its own service boundary even if the logic starts inside the engine.

### What we can realistically know

Publicly:

- bulk deals
- block deals
- shareholding pattern filings
- mutual fund disclosures
- insider trades
- pledge and promoter changes
- brokerage actions

Not fully:

- complete hidden institutional order flow
- all proprietary desk activity
- intra-broker beneficial owner identity

### Recommended architecture

```text
Institutional data ingestors
  NSE deals
  BSE deals
  shareholding pattern
  MF portfolios
  insider trades
      ->
entity resolution
  "Morgan Stanley Asia" == same group?
      ->
investor graph / registry
      ->
stock-investor activity features
      ->
institutional flow scanner
      ->
API + alerts
```

### Why bulk deals seem incomplete today

Usually because of:

- exchange-specific coverage gaps
- archive format changes
- intraday feed vs end-of-day feed mismatch
- symbol resolution problems
- partial public disclosure limits
- client-name normalization issues
- historical backfill missing for some windows

So production should have:

- raw source capture
- parsed row store
- deduplication rules
- reconciliation dashboard
- coverage metrics by date/source/exchange

## Service Breakdown

### 1. Market Data Service

Responsibilities:

- vendor connectors
- normalization
- instrument master
- historical bar serving
- live bar serving

### 2. Ingestion Orchestrator

Responsibilities:

- schedule jobs
- trigger backfills
- retry failures
- publish ingestion events
- maintain source freshness status

### 3. Scanner Service

Responsibilities:

- read materialized scanner outputs
- expose scanner query APIs
- support filters, sorts, saved screens
- provide human-readable explanations

### 4. Portfolio Service

Responsibilities:

- holdings, imports, snapshots
- portfolio analytics
- P&L, risk, exposure summaries
- user-owned metadata only

### 5. Alert Service

Responsibilities:

- rule creation
- debounce and cooldown logic
- evaluate against event streams
- emit notification requests

### 6. Notification Service

Responsibilities:

- FCM/APNs push
- email
- Telegram or WhatsApp later
- delivery tracking

### 7. Compliance / Audit Service

Responsibilities:

- disclaimer acceptance
- plan entitlements
- content/audit trail
- AI usage disclosures
- incident review logs

## Wireframe Of Runtime Flow

```text
                ┌────────────────────────────┐
                │        Web / Mobile        │
                │  Next.js / RN / Flutter    │
                └─────────────┬──────────────┘
                              │
                    HTTPS / WS / SSE
                              │
                ┌─────────────▼──────────────┐
                │      API Gateway / BFF     │
                └──────┬──────────┬──────────┘
                       │          │
                       │          │
          ┌────────────▼───┐   ┌──▼────────────────┐
          │ Product APIs   │   │ Realtime Gateway  │
          │ users/watchlist│   │ subscriptions     │
          │ alerts/portfolio│  │ fanout            │
          └──────┬─────────┘   └──┬────────────────┘
                 │                │
                 │                │
       ┌─────────▼────────────────▼─────────┐
       │        Streaming / Event Bus       │
       └──────┬──────────┬──────────┬───────┘
              │          │          │
      ┌───────▼───┐ ┌────▼─────┐ ┌──▼────────────┐
      │ Ingestion │ │ Features │ │ Alerts/Notif  │
      │ workers   │ │ + Models │ │ consumers     │
      └───────┬───┘ └────┬─────┘ └──┬────────────┘
              │          │           │
              └────┬─────┴─────┬─────┘
                   │           │
             ┌─────▼─────┐ ┌───▼────────────┐
             │ Postgres  │ │ Time-series DB │
             │ users/app │ │ prices/features│
             └───────────┘ └────────────────┘
```

## UI Wireframe By Platform

### Web

Best for:

- full research workflows
- backtests
- portfolio diagnostics
- saved screens

### Mobile

Best for:

- quick scanner checks
- alerts
- watchlist monitoring
- lightweight portfolio summaries

### Shared screen families

```text
Home
Discover
Scanners
Portfolio
Lab
Account
```

### Mobile-specific rule

Do not port the full current Streamlit density into mobile.

Mobile should prioritize:

- alerts
- watchlist
- scanner cards
- institutional flow highlights
- one-tap save/follow

Leave power workflows like optimizer/backtests mainly on web.

## Suggested Folder Plan For The Current Repo

If we want an immediate in-repo transition without a new monorepo today:

```text
StckManipulation/
  swingdesk/
    analyze/
    backtest/
    ingest/
    portfolio/
  platform/
    api/
    services/
    contracts/
    workers/
  clients/
    web/
    mobile/
  docs/
```

That lets us keep iterating locally while starting the real product layer beside it.

## Rollout Plan

### Phase 0: stabilize current engine

- split `app.py` into modules
- clean advisory language
- make scanner outputs API-friendly
- version schemas and models

### Phase 1: internal backend

- add `platform/api`
- add Postgres
- expose read-only scanner APIs
- keep Streamlit as internal ops console

### Phase 2: production web app

- auth
- onboarding
- watchlists
- saved screens
- alerts
- plan gating

### Phase 3: streaming and live alerts

- add event bus
- realtime gateway
- push notifications
- ingestion monitoring

### Phase 4: mobile apps

- scanner feeds
- alerts
- portfolio sync
- institutional flow views

### Phase 5: advanced data graph

- investor identity graph
- shareholding trend engine
- better institutional-flow coverage
- model registry and experiment tracking

## Recommended First Build Order

If we start now, the first practical sequence should be:

1. Create a separate `platform/` folder in this repo.
2. Build a small FastAPI backend around existing scanners.
3. Move scanner responses from dataframe-shaped outputs to explicit response schemas.
4. Add Postgres for users, watchlists, alerts, and legal logs.
5. Keep market data in the current local pipeline until API contracts settle.
6. Add Redis for cache and job coordination.
7. Add a worker process for ingestion and heavy recompute.
8. Add websocket/SSE only for live scanners and alerts.
9. Build web first, mobile after the API stabilizes.

## Final Recommendation

Yes, we should create a separate layer for production.

The right design is:

- current `swingdesk` code becomes the analytics engine
- a new `platform/` layer handles auth, APIs, plans, alerts, and connectivity
- web/mobile connect only to the platform APIs
- a streaming layer is added for live data and alerts, not for every workload
- Streamlit remains a fast internal workbench until the product UI fully replaces it

This gives us:

- cleaner scaling
- easier mobile/web development
- safer production boundaries
- better data freshness
- room for institutional-flow and ML expansion without turning the UI into a bottleneck
