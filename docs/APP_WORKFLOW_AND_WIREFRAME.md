# SwingDesk App Workflow And Wireframe

Last updated: 2026-08-06
Status: build blueprint

## Purpose

This document is the working blueprint for converting SwingDesk from a local analytics tool into a proper user-facing product with:

- web app
- Android app
- iPhone app
- authentication
- premium plans
- alerts
- cloud data
- launch-ready onboarding and support flows

It combines:

- product workflow
- user journey
- feature packaging
- competitor-inspired additions
- build order
- low-fidelity wireframes

Use this alongside:

- `docs/BUSINESS_MODEL_NO_SEBI_ADVISOR.md`
- `docs/COMPLIANT_FEATURE_WIREFRAMES.md`
- `docs/PRODUCT_BLUEPRINT.md`

## Product Definition

SwingDesk should launch as:

**A subscription analytics workbench for Indian self-directed investors, centered on unusual-activity detection, screening, portfolio diagnostics, alerts, and research validation.**

It should not launch as:

- an advisory app
- a calls/tips app
- a broker
- a portfolio manager

## What We Keep From The Current Engine

These are the strongest reusable parts of the current repo:

- unusual-activity / manipulation scanner
- discovery and screening engine
- factor and quality ranking
- sector rotation
- portfolio analytics
- paper trader and journal
- expected range and scenario analysis
- backtests and optimizer
- macro / spillover / event context
- execution and TCA research

## What Competitors Have That We Need To Add

These are product-layer capabilities users now expect because competitors like Tickertape, Trendlyne, Screener, StockEdge, and Chartink offer them in some form.

### Must-have additions

- sign-up / login
- onboarding flow
- multiple watchlists
- saved screens
- saved alerts
- mobile push notifications
- plan gating and upgrade flow
- cloud sync across devices
- exports / downloads
- public pricing page
- legal / disclaimer acceptance tracking
- support / feedback flow

### Strong additions

- daily digest email / push
- scanner history
- portfolio snapshots over time
- onboarding checklist
- empty-state education
- referral flow
- simple in-app explainers and glossary

### Avoid copying directly

Do not adopt competitor features that cross into recommendation behavior unless regulatory posture changes:

- expert calls
- recommended trades
- top picks
- target-price advisories
- personalized action recommendations

## Target Product Structure

The proper app should have 3 layers:

### Layer 1: Engine

Owned by the current repo or extracted package later.

Responsibilities:

- analytics
- scanners
- scoring
- backtesting
- risk and range calculations
- data ingestion and processing

### Layer 2: Product API

New backend.

Responsibilities:

- auth and user profiles
- watchlists
- saved screens
- alerts
- plan enforcement
- portfolio import metadata
- notification jobs
- legal acceptance logs
- app analytics and admin tools

### Layer 3: User App

New web + mobile interface.

Responsibilities:

- onboarding
- screen navigation
- pricing / paywall
- portfolio experience
- alert setup
- mobile UX
- billing and account management

## End-To-End Workflow

This is the intended product workflow from acquisition to retention.

```text
Traffic / Referral / Creator / SEO
    ->
Landing Page
    ->
Signup / Login
    ->
Disclaimer Acceptance
    ->
Onboarding
    ->
Choose watchlist + interests + style
    ->
Home dashboard
    ->
Discover / Scanners / Portfolio / Lab
    ->
Save screen / create alert / import holdings
    ->
Receive digest / push / reminder
    ->
Return to app
    ->
Upgrade to paid plan
    ->
Retention loop
```

## Core User Flows

### 1. First-time user flow

Goal:

- reach value quickly
- avoid overwhelming the user
- establish compliance posture early

Flow:

```text
Open app
    ->
Welcome screen
    ->
Sign up / log in
    ->
Accept disclaimer
    ->
Choose user type:
       Active investor / swing trader / learner
    ->
Choose focus:
       Unusual activity / screeners / portfolio / backtests
    ->
Create first watchlist
    ->
See Home
    ->
Prompt:
       Save first screen or create first alert
```

### 2. Daily returning user flow

Goal:

- give a reason to come back every day

Flow:

```text
Open app
    ->
Home dashboard
    ->
Check market pulse
    ->
Review unusual activity cards
    ->
Open screen results
    ->
Review watchlist changes
    ->
Save / update alerts
    ->
Exit
```

### 3. Portfolio user flow

Goal:

- make the portfolio section useful without drifting into advice

Flow:

```text
Import holdings
    ->
Normalize symbols
    ->
Compute exposure / P&L / concentration / event risk
    ->
Show diagnostics
    ->
Optional:
       journal note
       allocation calculator
       risk report
```

### 4. Paid conversion flow

Goal:

- convert from useful teaser to paid habit

Flow:

```text
User sees partial scanner data
    ->
User hits saved-alert or full-results limit
    ->
Upgrade sheet / page opens
    ->
Show concrete value:
       full unusual-activity feed
       custom alerts
       saved screens
       backtests
       exports
    ->
Checkout
    ->
Plan active
    ->
Confirmation + first paid action prompt
```

## Information Architecture

### Primary sections

- Home
- Discover
- Scanners
- Portfolio
- Lab

### Utility areas

- Watchlists
- Alerts
- Pricing / Plans
- Settings
- Legal
- Help / Feedback

## Screen-Level Workflow

### Home

Purpose:

- orient the user
- provide a daily habit surface
- surface teaser value

Key modules:

- market pulse
- watchlist activity
- unusual-activity highlights
- upcoming events
- portfolio snapshot
- quick actions

### Discover

Purpose:

- help users actively search for matches

Key modules:

- screen builder
- saved screens
- factor ranking
- quality ranking
- sector rotation
- small-cap desk
- calendar explorer

### Scanners

Purpose:

- show what is unusual right now

Key modules:

- unusual activity
- catalysts
- range
- intraday
- global impact
- liquidity / market structure

### Portfolio

Purpose:

- provide diagnostics, not advice

Key modules:

- import holdings
- portfolio snapshot
- concentration
- event proximity
- risk report
- P&L and charges
- journal
- allocation calculator

### Lab

Purpose:

- validate strategies and ideas

Key modules:

- backtester
- optimizer
- confluence board
- chart lab
- execution / TCA
- ML probability
- raw data

## Recommended Plans

### Free

- 1 watchlist
- limited scanner preview
- limited saved items
- delayed or capped alerts
- no exports

### Pro

- full unusual-activity scanner
- saved screens
- custom alerts
- portfolio diagnostics
- backtests
- exports

### Trader

- intraday tools
- scanner history
- advanced lab
- more alerts
- priority notifications

## Build Workflow

### Phase 0: Product freeze

Deliverables:

- feature list
- plan packaging
- legal posture
- naming cleanup

Exit criteria:

- no unclear advisory features remain in scope

### Phase 1: Architecture setup

Deliverables:

- new app repo
- auth system
- app database
- core API structure
- billing integration scaffolding

Exit criteria:

- user can sign up and persist watchlists

### Phase 2: MVP product shell

Deliverables:

- Home
- Discover
- Scanners
- basic Portfolio
- pricing and upgrade flow

Exit criteria:

- closed beta users can onboard and use core value loop

### Phase 3: Retention and monetization

Deliverables:

- alerts
- email/push digests
- saved screens
- portfolio imports
- scanner history

Exit criteria:

- users have repeat-return hooks

### Phase 4: Mobile polish

Deliverables:

- React Native app
- push notifications
- mobile-friendly cards and tables
- App Store / Play Store assets

Exit criteria:

- stable beta on Android and iOS

### Phase 5: Launch hardening

Deliverables:

- analytics
- crash reporting
- support flow
- legal pages
- billing edge cases
- app review readiness

Exit criteria:

- ready for public launch

## Suggested Tech Workflow

### Web

- marketing site
- app shell
- pricing
- onboarding

### API

- auth
- engine orchestration
- user state
- alerts and jobs
- billing

### Mobile

- dashboard
- scanner cards
- watchlists
- alerts
- portfolio summaries

### Jobs

- daily scanner refresh
- unusual-activity generation
- digest generation
- push and email alert delivery

## Wireframes

These are low-fidelity and meant to guide implementation.

### 1. Landing Page

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ SwingDesk                                                                    │
│ Analytics workbench for Indian self-directed investors                       │
│                                                                              │
│ [ See unusual activity ] [ Join waitlist / Sign up ]                         │
│                                                                              │
│ Why it matters:                                                              │
│ - Track unusual activity                                                     │
│ - Build screens                                                              │
│ - Monitor portfolio risk                                                     │
│ - Backtest ideas                                                             │
│                                                                              │
│ Pricing teaser: Free | Pro | Trader                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2. Signup + Disclaimer

```text
┌────────────────────────────── Welcome to SwingDesk ──────────────────────────┐
│ Create your account                                                          │
│ [ Email ] [ Continue ]                                                       │
│                                                                              │
│ Disclaimer:                                                                  │
│ SwingDesk is an analytics and research tool. It is not investment advice.    │
│ We are not a SEBI-registered Investment Adviser or Research Analyst.         │
│                                                                              │
│ [ ] I understand and accept                                                  │
│                                                  [ Accept and continue ]     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3. Onboarding

```text
┌──────────────────────────── Set up your workspace ───────────────────────────┐
│ What best describes you?                                                     │
│ ( ) Active investor   ( ) Swing trader   ( ) Learning / exploring            │
│                                                                              │
│ What do you want first?                                                      │
│ [ Unusual activity ] [ Screeners ] [ Portfolio ] [ Backtests ]              │
│                                                                              │
│ Create your first watchlist                                                  │
│ [ Add symbols ]                                                              │
│                                                                              │
│ [ Continue ]                                                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4. Home

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Home                                                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│ Market pulse                                                                 │
│ Regime: Risk-on   Breadth: Improving   Risk tone: Neutral-positive          │
├──────────────────────────────────────────────────────────────────────────────┤
│ Watchlist activity              Unusual-activity highlights                  │
│ ABC  +2.1%  Strong setup        XYZ  Turnover 6x avg                         │
│ DEF  -0.8%  Mixed setup         PQR  Delivery spike                          │
│                                 [ View full scanner ]                        │
├──────────────────────────────────────────────────────────────────────────────┤
│ Upcoming events                  Portfolio snapshot                          │
│ RBI policy                       Holdings value                              │
│ Earnings calendar                Concentration risk                          │
├──────────────────────────────────────────────────────────────────────────────┤
│ Quick actions: Refresh | Save screen | Create alert                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5. Discover

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Discover                                                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ Filters: Setup | Universe | Quality | Liquidity | Timeframe                 │
│ [ Run screen ]  [ Save screen ]                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Rule-based analytics only. These stocks match your criteria.                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ Ticker   Setup strength   Entry band   Risk level   Scenario upside          │
│ ABC      Strong           2890-2910    2850         +5.2%                    │
│ GHI      Moderate         540-545      528          +4.1%                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ Saved screens | Factor ranking | Quality ranking | Sector rotation           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6. Scanners

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Scanners                                                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ Tabs: Unusual activity | Catalyst | Range | Intraday | Global               │
├──────────────────────────────────────────────────────────────────────────────┤
│ XYZ LTD  flags: 4                                                            │
│ - Turnover 6.2x 30-day average                                               │
│ - Delivery spike vs recent history                                           │
│ - Abnormal return vs realized volatility                                     │
│ - No major public catalyst matched                                           │
│                                                  [ View evidence ] [ Alert ] │
├──────────────────────────────────────────────────────────────────────────────┤
│ PQR LTD  flags: 3                                                            │
│ ...                                                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7. Portfolio

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Portfolio                                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ [ Import holdings ] [ Refresh ] [ Export ]                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ Holdings value   Day P&L   Exposure   Concentration                          │
├──────────────────────────────────────────────────────────────────────────────┤
│ Ticker   Weight   P&L   Trend quality   Earnings proximity   Risk note       │
│ ABC      18%      6.2%  Improving       4 days              High weight      │
│ DEF      11%     -2.4%  Weakening       22 days             Normal           │
├──────────────────────────────────────────────────────────────────────────────┤
│ Allocation calculator                                                        │
│ Selected symbol | Capital | Max risk % | Estimated shares | Risk amount     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 8. Lab

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Lab                                                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│ Inputs: Setup | Universe | Max hold | Costs | Run                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ Hypothetical historical analysis only. Past performance does not guarantee  │
│ future results.                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Trades   Win rate   Expectancy   Profit factor   Max drawdown               │
│ 132      47%        0.21R        1.29            -6.8R                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ Optimizer | Confluence board | Charts | Execution / TCA                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 9. Pricing

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Choose your plan                                                             │
├──────────────────────────────────────────────────────────────────────────────┤
│ Free                  Pro                           Trader                   │
│ 1 watchlist           Full scanner                  Intraday + advanced lab   │
│ Limited preview       Alerts + saved screens        More alerts              │
│ No exports            Portfolio diagnostics         Scanner history          │
│ [ Start free ]        [ Upgrade to Pro ]            [ Upgrade to Trader ]   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 10. Mobile-first dashboard

```text
┌───────────────────────────────┐
│ Home                          │
│ Market pulse                  │
│ Risk-on                       │
│                               │
│ Unusual activity              │
│ XYZ  flags: 4                 │
│ [ View ]                      │
│                               │
│ Watchlist                     │
│ ABC +2.1%                     │
│ DEF -0.8%                     │
│                               │
│ Upcoming events               │
│ RBI policy                    │
│ Earnings                      │
│                               │
│ Bottom nav:                   │
│ Home Discover Scan Portfolio  │
└───────────────────────────────┘
```

## Launch Checklist

Before public launch, confirm:

- auth works across web and mobile
- disclaimer acceptance is stored
- premium gating works
- alerts work reliably
- exports are plan-restricted
- support and legal pages exist
- no advisory-style copy remains
- onboarding drives users to first value quickly
- mobile UX is usable for daily check-ins

## Immediate Next Build Steps

1. Create the separate app repo.
2. Build auth, onboarding, and disclaimer acceptance.
3. Build Home, Discover, and Scanners first.
4. Add saved screens and alerts.
5. Add Portfolio diagnostics and pricing.
6. Add Lab after the daily-value loop is stable.

## Outcome

If we follow this workflow and wireframe, SwingDesk becomes:

**a proper analytics subscription app with a clear user journey, strong daily habit surfaces, compliant positioning, and enough product structure to launch on web, Android, and iPhone.**

