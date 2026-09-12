# SwingDesk Competitor Gap Analysis

Last updated: 2026-08-20
Status: product comparison working doc

Implementation update, 2026-09-11: see [Competitor feature delivery](COMPETITOR_FEATURE_DELIVERY.md)
for the newly implemented named watchlists, saved screens, exports, comparisons,
heatmap, and manual alert conditions, including TradingView and Screener references.
The matrix below is the earlier planning snapshot, not the latest delivery status.

## Purpose

This document compares the current SwingDesk product to:

- Tickertape
- smallcase
- Univest
- Liquide

The goal is not to copy everything.

The goal is to identify:

- what they have that we do not
- what we already do well
- what is safe to add under the current non-advisory posture
- what should only be added if we become or partner with a SEBI-registered advisory layer

## Assumption

The user referenced "liquidity". For this analysis, that is treated as **Liquide**.

## Current SwingDesk Strengths

Compared with these competitors, SwingDesk already has unusually strong depth in:

- unusual-activity / manipulation-style scanner
- sudden-move and confluence-style scanners
- backtesting and optimizer workflows
- institutional flow from disclosed deals
- portfolio risk, journaling, reconcile, and paper-trading workflows
- local-first research flow
- technical + liquidity + catalyst + macro combination logic

This means the biggest gaps are mostly **product-layer gaps**, not raw analytics gaps.

## Comparison Matrix

Legend:

- `Yes` = present in product today
- `Partial` = present but weaker, local-only, or not productized
- `No` = largely missing

| Capability | SwingDesk | Tickertape | smallcase | Univest | Liquide | Notes for SwingDesk |
|---|---|---|---|---|---|---|
| Web app | Yes | Yes | Yes | Yes | Yes | SwingDesk has Streamlit, not production web product yet |
| Mobile app | No | Yes | Yes | Yes | Yes | Big product gap |
| User login / account system | No | Yes | Yes | Yes | Yes | Big product gap |
| Cloud sync across devices | No | Yes | Yes | Yes | Yes | Big product gap |
| Multiple watchlists | Partial | Yes | Yes | Yes | Yes | SwingDesk has watchlists but not full multi-device product flow |
| Saved screens | Partial | Yes | No clear core emphasis | Yes | Yes | SwingDesk needs proper saved-screen product layer |
| Shareable screens / screen library | No | Yes | Limited relevance | Partial | Partial | Good growth feature for analytics product |
| Alerts / notifications | Partial | Yes | Yes | Yes | Yes | SwingDesk has Telegram-style alert plumbing, but not full push/email/app alerts |
| Portfolio import from brokers | Partial | Yes | Yes | Yes | Yes | SwingDesk supports imports, but not linked-broker cloud sync |
| Multi-account portfolio aggregation | No | Yes | Yes | Yes | Yes | Clear user-value gap |
| Portfolio health / diversification score | Partial | Yes | Yes | Yes | Yes | SwingDesk has risk analytics but needs cleaner productization |
| Mutual fund support | No | Yes | Yes | Yes | Yes | Missing if we want mass-market breadth |
| ETF support | Partial | Yes | Yes | Partial | Yes | SwingDesk can likely analyze if data exists, but it is not packaged |
| Ready-made prebuilt screens | Partial | Yes | N/A | Yes | Yes | We need curated starter workflows |
| Advanced screener filters and universes | Partial | Yes | Limited relative to stock screener | Yes | Yes | SwingDesk has power internally but UI/product layer is behind |
| Forecast / analyst estimates | No | Yes | Limited in product story | Partial | Partial | Could add later via data vendor |
| Broker ratings / analyst consensus | No | Yes | No core emphasis | Yes | Partial | Missing |
| Shareholding / ownership trend views | Partial | Yes | Limited | Partial | Partial | Institutional graph can grow into this |
| Deals insights with filters and party search | Partial | Yes | No core emphasis | Partial | Partial | We now have deal data, but not rich search/history UX |
| Market news and sentiment | Yes | Yes | Partial | Yes | Yes | SwingDesk already has strong base here |
| AI assistant / conversational copilot | No | No clear core consumer bot | No | Partial | Yes | Optional product differentiator later |
| Recommendation engine / buy-sell-hold | No by design | No by design | Via registered managers, not as generic tool | Yes | Yes | This is intentionally excluded for current compliance posture |
| One-click execution / broker trading rail | No | No core emphasis | Yes | Yes | Via broker rails | Only add with clear strategic reason |
| Curated portfolios / baskets | No | Partial via smallcase links | Yes | Partial | Yes | Missing, but risky unless structured carefully |
| IPO module | Partial | No clear core emphasis | No clear core emphasis | Yes | Yes | Nice-to-have, not core wedge |
| Education / academy / webinars | No | Partial | Partial | Yes | Yes | Missing growth/retention layer |
| Pricing / paywall / plan gating | No | Yes | Yes | Yes | Yes | Foundational product gap |
| Export / downloads | Partial | Yes | Partial | Partial | Partial | SwingDesk has raw data but not polished export UX |
| Admin / compliance / disclaimer tracking | Partial | Yes | Yes | Yes | Yes | We started disclaimer gating, but not full product ops layer |

## Product-By-Product Notes

## 1. Tickertape

Tickertape's biggest advantages over SwingDesk are product polish and breadth in self-directed investing tools.

What Tickertape has that we do not fully have:

- polished multi-asset portfolio tracking
- linked account portfolio analysis
- customizable saved screens and user-created screens
- prebuilt screens and a stronger screener onboarding flow
- alerts as a first-class product feature
- forecasts, analyst ratings, and richer stock fundamentals packaging
- deals insights with date/category/party-name exploration
- exports and pro-tier gating
- mature consumer UX across web and app

What SwingDesk is already stronger at:

- deeper scanner logic for unusual activity and sudden moves
- more research-lab style workflows
- better backtesting orientation

Strategic reading:

- Tickertape is the closest safe benchmark for our current posture.
- We should copy the **tool/product layer**, not their exact feature copy.

## 2. smallcase

smallcase is a different category in one important sense: it combines discovery with execution rails and managed portfolios.

What smallcase has that we do not:

- curated model portfolios / baskets
- one-click investing rails through brokers
- rebalance workflows
- SIP-style investing flows
- broad wealth surface including mutual funds and fixed deposits
- net-worth style portfolio management
- expert-managed offerings through registered creators/managers

What matters for SwingDesk right now:

- the main gap is not analytics
- the main gap is portfolio productization, user accounts, and rails

What to avoid for now:

- manager-led advisory behavior unless we intentionally partner with or become a registered layer

## 3. Univest

Univest is much closer to a recommendation-plus-execution app than SwingDesk.

What Univest has that we do not:

- advisory-led stock recommendations
- buy/sell/hold portfolio guidance
- one-click execution and brokerage integration
- 25+ screeners packaged for retail users
- shark portfolios / top investor tracking packaging
- live market content and results packaging
- mutual fund advisory
- strong mobile-first flows

What we should learn without copying the advisory model:

- retail-friendly onboarding
- better portfolio review packaging
- stronger beginner UX
- better distribution via mobile notifications and habit loops

What we should not copy under current posture:

- direct buy/sell recommendations
- personalized action guidance
- "stocks to buy today" framing

## 4. Liquide

Liquide combines portfolio analytics, AI assistant behavior, and advisory framing.

What Liquide has that we do not:

- mobile-native product
- stock alerts as a polished feature
- portfolio health score and red-flag packaging
- mutual fund analytics
- AI bot / conversational assistant
- IPO verdicts
- curated baskets
- academy / education layer

What is worth learning from Liquide:

- portfolio-health packaging
- AI assistant UX for explanation and discovery
- push-driven engagement

What is risky for us to copy now:

- buy/sell/hold advice
- advisory-style AI assistant behavior

## The Real Gap List

If we strip away recommendation-only features, the biggest missing parts in SwingDesk are:

### Tier 1: core product gaps

- authentication and user accounts
- production web app
- mobile app
- cloud sync
- plan/paywall/billing
- saved screens and saved alerts
- push/email/in-app notification system
- multi-account portfolio linking and refresh
- better onboarding and empty states

### Tier 2: research-tool gaps

- richer screener UX and screen library
- shareable/public screen links
- watchlist history and scanner history
- richer institutional-flow history, filters, and party search
- ownership/shareholding trend views
- better portfolio health scoring and diversification visuals
- cleaner exports and downloadable reports

### Tier 3: optional breadth expansion

- mutual funds
- ETF-first workflows
- IPO workflow
- academy / education content
- conversational AI assistant

### Tier 4: strategy-dependent gaps

Only pursue these if strategy changes:

- baskets / curated portfolios
- broker execution rails
- advisory-style ratings
- personalized buy/sell/hold guidance

## What We Should Build First

For the current product direction, the best next additions are:

1. Auth, accounts, and cloud sync.
2. Saved screens, saved watchlists, and saved alerts.
3. Push/email/in-app alert system.
4. Productized portfolio health and diversification dashboard.
5. Screener library with beginner-friendly prebuilt screens.
6. Institutional-flow history with investor/entity search.
7. Proper web app, then mobile app.

## Priority Table

This table is the practical decision layer on top of the comparison.

| Priority | Capability | Why it matters | Safe under current posture? | Where it should live | Suggested phase |
|---|---|---|---|---|---|
| Must build now | Auth and user accounts | Required for multi-user product, saved work, plans, alerts, and sync | Yes | `platform/api`, `platform/services/auth`, `Postgres` | Phase 1 |
| Must build now | Cloud sync | Needed for web + mobile continuity | Yes | `platform/services/user`, `platform/services/watchlist` | Phase 1 |
| Must build now | Saved watchlists | Core daily habit feature | Yes | `platform/services/watchlist` | Phase 1 |
| Must build now | Saved screens | One of the clearest Tickertape-style tool gaps | Yes | `platform/services/scanner`, `platform/contracts` | Phase 1 |
| Must build now | Saved alerts | Critical retention and monetization feature | Yes | `platform/services/alerts`, `workers` | Phase 1 |
| Must build now | Plan gating / billing | Needed for paid product, freemium, and pro tiering | Yes | `platform/services/billing`, `platform/api` | Phase 1 |
| Must build now | Production web app | Needed to replace Streamlit as public product UI | Yes | `clients/web` | Phase 2 |
| Must build now | Portfolio health dashboard | Strong user-facing packaging of existing risk engine | Yes | `clients/web`, `platform/services/portfolio`, `engine` | Phase 2 |
| Must build now | Screener library / prebuilt screens | Helps onboarding and retention, especially for non-expert users | Yes | `clients/web`, `platform/services/scanner` | Phase 2 |
| Must build now | Notification layer | Push/email/in-app alerts make the app habit-forming | Yes | `platform/services/notifications`, `streaming` | Phase 2 |
| Build next | Mobile app | Needed for distribution, alerts, and daily engagement | Yes | `clients/mobile` | Phase 3 |
| Build next | Shareable screens / links | Good growth loop and community acquisition feature | Yes | `clients/web`, `platform/api` | Phase 3 |
| Build next | Scanner history | Improves trust, monitoring, and product stickiness | Yes | `platform/services/scanner`, `time-series store` | Phase 3 |
| Build next | Institutional-flow history and party search | Strong wedge if we package it better than generic tools | Yes | `platform/services/scanner`, `search/index`, `engine` | Phase 3 |
| Build next | Ownership / shareholding trend views | Helps close the gap with Tickertape-style research depth | Yes | `market-data service`, `engine`, `clients/web` | Phase 3 |
| Build next | Exports and polished downloads | Important for pro users and desk workflows | Yes | `platform/api`, `object storage` | Phase 3 |
| Good to have | ETF workflows | Broader surface without changing product posture much | Yes | `engine`, `clients/web` | Phase 4 |
| Good to have | Mutual fund analytics | Useful for broader retail product expansion | Yes | `market-data service`, `clients/web/mobile` | Phase 4 |
| Good to have | IPO workflow | Broadens app usage but is not the wedge | Yes | `clients/web/mobile`, `market-data service` | Phase 4 |
| Good to have | Education / academy layer | Helps acquisition and retention | Yes | `clients/web/mobile`, content CMS | Phase 4 |
| Good to have | Conversational AI explainer | Useful if kept descriptive, not advisory | Yes, with guardrails | `platform/api`, `clients/web/mobile` | Phase 4 |
| Avoid for now | Buy/sell/hold recommendation engine | Crosses into advisory-style behavior | No | Do not build in public product | Only with registered layer |
| Avoid for now | Personalized action guidance on holdings | High regulatory risk | No | Do not build in public product | Only with registered layer |
| Avoid for now | Curated advisory baskets | Feels like managed advice unless structured carefully | Not safely | Separate advisory track only | Only with registered layer |
| Avoid for now | One-click broker execution rails | Operationally heavy and changes business posture | Not needed now | Separate execution program later | Only if strategy changes |
| Avoid for now | Advisory AI copilot | Too easy to drift into direct recommendations | No | Do not build in current form | Only with registered layer |

## Priority Groups By Team

### Product layer

- auth and accounts
- cloud sync
- saved watchlists
- saved screens
- saved alerts
- plan gating
- onboarding and empty states
- production web app
- mobile app

### Analytics packaging layer

- portfolio health dashboard
- screener library
- scanner history
- institutional-flow search/history
- ownership trend views
- exports

### Platform / infra layer

- notification service
- worker jobs
- search/index layer
- time-series store
- event streaming for live scanners and alerts

### Deferred strategy layer

- baskets
- broker rails
- recommendation engine
- advisory AI

## Feature Adoption Plan

This section answers the narrow question:

**Which features do these products have today that SwingDesk does not, and what should we do about each one?**

| Competitor feature seen publicly | Seen on | SwingDesk today | Decision | Plan |
|---|---|---|---|---|
| Saved screeners and better screener UX | Tickertape, Univest, Liquide | Partial | Build | Add saved screens, presets, beginner templates, and better filter UX in the new product layer |
| Price / technical / event alerts | Tickertape, Liquide, Univest | Partial | Build | Build unified alerts service for price, technical, scanner, institutional-flow, and event alerts |
| Portfolio diversification score | Tickertape | Partial | Build | Package existing risk engine into a simple 0-100 portfolio health and diversification module |
| Portfolio red flags | Tickertape, Liquide | Partial | Build | Turn existing portfolio analytics into plain-language red flags: concentration, liquidity risk, earnings risk, sector crowding |
| Watchlists with richer metrics | Tickertape | Partial | Build | Add multi-watchlist support, sortable columns, saved views, and watchlist snapshots/history |
| Forecasts / analyst estimates | Tickertape | No | Build later | Add only after core product layer is live; requires data vendor and clean source attribution |
| Mutual fund tracking / analysis | Tickertape, smallcase, Liquide, Univest | No | Build later | Add as a separate expansion track after core equities product stabilizes |
| Linked portfolio aggregation / net worth view | Tickertape, smallcase | No | Build | Add account sync and portfolio aggregation in platform layer, start with manual import + broker integrations later |
| Alerts tied to watchlist and portfolio | Tickertape | Partial | Build | Reuse alert engine, but make watchlist and portfolio alert rules first-class product objects |
| Smallcase / basket integration | Tickertape, smallcase, Liquide | No | Skip for now | Only revisit if we intentionally add partner rails or regulated basket distribution |
| Curated stock / ETF / MF portfolios | smallcase | No | Skip for now | Too close to managed/recommended product strategy unless we choose that route |
| Rebalance updates | smallcase | No | Skip for now | Relevant only if we launch baskets or managed portfolios |
| SIP flows | smallcase | No | Skip for now | Not core to SwingDesk wedge |
| Direct mutual fund investing | smallcase | No | Build later | Optional breadth expansion, not phase 1 |
| Fixed deposits / loans / wealth office | smallcase | No | Skip | Outside current product wedge |
| Create your own baskets / portfolios | smallcase | No | Build selectively | Add only as user-created tracking baskets, not managed recommendation baskets |
| 100+ filter retail screener with guided discovery | Univest | Partial | Build | Improve screener onboarding, presets, and discoverability rather than copying advisory wrapper |
| Shark / ace investor portfolios | Univest | Partial | Build | Expand institutional/shareholding layer into investor tracking, ownership change, and public-portfolio monitoring |
| Results / IPO packaging | Univest, Liquide | Partial | Build later | Keep as research modules, not recommendation modules |
| Buy/sell/hold portfolio review | Univest, Liquide | No by design | Skip | Do not add under current compliance posture |
| Trade ideas with entry / target / stop | Univest, Liquide | No by design | Skip | Do not add in public tool product |
| One-click execution and broker rails | Univest, smallcase | No | Skip for now | Operationally heavy and strategy-changing; not phase 1 |
| Push notifications and mobile-first habit loop | Univest, Liquide | Partial | Build | Add notifications before mobile, then mobile apps on top of same alert/event system |
| AI stock copilot / assistant | Liquide | No | Build carefully later | Add only as descriptive explainer over our analytics, never advisory verdicts |
| Education / academy / webinars | Liquide, Univest | No | Build later | Useful for acquisition and retention after core workflow is solid |
| Exclusive screeners behind subscription | Liquide | Partial | Build | Natural fit for paywall once saved screens and alerts exist |
| F&O tools / ideas | Liquide, Univest | Partial | Build selectively | Add analytics and scanner layers only, not trade-call layer |
| IPO insights | Liquide, Univest | Partial | Build later | Good breadth feature, not the main wedge |

## Recommended Build Plan From The Feature Gap

### Phase 1: features we should definitely add

- saved screens
- multi-watchlist productization
- alert engine with price, technical, event, and scanner alerts
- portfolio health score
- portfolio red flags
- linked portfolio/account model
- screener presets and guided discovery
- investor / ownership tracking built on public disclosures

### Phase 2: features we should add after the base product works

- watchlist history
- scanner history
- shareholding trend views
- richer institutional-flow search
- exports and reports
- descriptive AI explainer
- mobile apps

### Phase 3: optional expansions

- mutual funds
- IPO module
- ETF-first workflows
- education / academy
- user-created tracking baskets

### Explicit no-for-now list

- buy/sell/hold engine
- personalized portfolio action advice
- trade calls with stop/target
- advisory AI copilot
- broker execution rails
- managed or expert baskets

## What We Probably Do Better Than Them If We Package It Right

This is important.

SwingDesk does not need to win by becoming a weaker Tickertape or a pseudo-Univest.

It can win by owning:

- unusual-activity detection
- research validation
- scanner depth
- institutional-flow interpretation
- trader-grade confluence workflows

That is the wedge.

## Source Snapshot

These references were checked on or around **2026-08-20**.

- Tickertape analytical tools and screener terms:
  - https://www.tickertape.in/meta/analytical-tools
- Tickertape membership / features:
  - https://www.tickertape.in/membership
- Tickertape announcement archive:
  - https://www.tickertape.in/blog/announcements/
- smallcase investment tools:
  - https://www.smallcase.com/meta/investment-tools/
- smallcase platform / product overview:
  - https://www.smallcase.com/
  - https://www.smallcase.com/learn/what-is-smallcase/
- Univest product pages:
  - https://univest.in/
  - https://univest.in/pricing
- Liquide product pages / store listings:
  - https://apps.apple.com/in/app/liquide-stocks-f-o/id1624726081
  - https://play.google.com/store/apps/details?hl=en-IN&id=life.liquide.app
  - https://liquidelife.smallcase.com/
