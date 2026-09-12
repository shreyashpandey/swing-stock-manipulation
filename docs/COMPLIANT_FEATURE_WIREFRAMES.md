# SwingDesk Compliant Feature Set And Wireframes

Last updated: 2026-08-03
Status: launch-planning

## Purpose

This document proposes a bounded analytics interface for public-scope review.
It does not establish that these features can be offered without registration.
Read the [Plan risk review](PLAN_RISK_REVIEW.md), including the stock-specific
research and educational-data findings, before implementing the launch wireframes.

The focus is:

- what we can ship now
- what must be reframed
- what should be hidden or removed
- how the current app maps into a compliant first product

This document is intentionally narrower than `docs/PRODUCT_BLUEPRINT.md`. It is the implementation-facing version of that strategy.

## Current App Inventory: Launch Decision

### Candidates for substantive public-scope review

These are analytical/tooling candidates. Their actual outputs, defaults, data
permissions and marketing need review; copy cleanup alone is insufficient.

| Current app area | Keep? | Launch home | Notes |
|---|---|---|---|
| Discovery | Yes | Discover | User-driven filtering and ranking |
| Small Caps | Yes | Discover | Keep as research workbench |
| Signals | Yes, but rename | Discover | Reframe to screener output / criteria matches |
| News | Yes | Discover / Scanners | Descriptive only |
| Sectors | Yes | Discover | Rotation and sector analytics are safe |
| Manipulation | Yes | Scanners | Primary paid wedge |
| Scanners | Yes | Scanners | Good fit if language stays descriptive |
| Calendar | Yes | Home / Discover | Events and seasonality are informational |
| Backtest | Yes | Lab | Add hypothetical-performance disclaimer |
| Optimize | Yes | Lab | Research tooling, not advice |
| Fundamentals | Yes | Discover | Quality/ranking framing |
| Raw data | Yes | Lab / Settings | Power-user feature |
| Global | Yes | Scanners | Contextual analytics |
| Range | Yes | Scanners | Must be statistical, not prescriptive |
| Risk | Yes | Portfolio | Calculator framing only |
| Rank | Yes | Discover | Rankings and factor analytics |
| ML | Yes, with restraint | Lab / Scanners | Probability language only |
| Intraday | Yes | Scanners | Descriptive scan output only |
| Execution | Yes | Lab | Execution research / TCA only |
| P&L & Taxes | Yes | Portfolio | Clear "not tax advice" notice |
| Reconcile | Yes | Portfolio | Operational/accounting tool |
| Paper Trader | Yes | Portfolio / Lab | Simulation and journaling |

### Keep only after reframe

These areas are useful but currently too advisory in language or flow.

| Current app area | Problem today | Safe reframing |
|---|---|---|
| Invest | Suggests deployment decisions | Convert into allocation calculator where user chooses stock and amount |
| My Holdings | Can drift into action recommendations | Limit to portfolio diagnostics, exposure, concentration, P&L, and risk |
| Board | Uses action/conviction framing | Convert to confluence board with descriptive tags only |
| Signals details | Uses entry/stop/target and verdicts | Rename to rule-implied levels and statistical scenarios |
| ML direction | Sounds predictive/advisory | Use probability bands and historical calibration |
| Thesis-style outputs | Too close to recommendation | Convert to explainability / evidence summary only |

### Remove, disable, or postpone for public launch

These are highest-risk without a registered advisory layer.

| Current behavior | Reason |
|---|---|
| BUY / SELL / HOLD / AVOID labels | Looks like direct recommendation language |
| BUY MORE / REDUCE / EXIT holding actions | Personalized advice on existing portfolio |
| "Invest fresh money" positioning | Direct capital deployment framing |
| Recommendation summaries for holdings | Personalized portfolio action guidance |
| Target/stop as instructions | Feels like trade advice instead of analytics |
| "Today is the day to buy" style copy | Prescriptive trade timing |
| Any public "top picks" framing | Recommendation-style marketing |

## Launch Navigation

The app should ship with 5 sections and a legal/settings utility area.

### Primary nav

- Home
- Discover
- Scanners
- Portfolio
- Lab

### Utilities

- Watchlist
- Alerts
- Settings
- Legal
- Plan / Billing later

## Launch Feature Bundle

### Home

Purpose:

- orient the user
- show market context
- surface useful things without telling them what to do

Should include:

- market regime summary
- watchlist movers
- unusual-activity teaser
- portfolio snapshot
- next 7-day event calendar
- quick actions: refresh prices, fetch news, refresh fundamentals

Should not include:

- top stock to buy now
- recommended trade of the day
- personalized action summary

### Discover

Purpose:

- help the user find ideas through filters they control

Should include:

- screener builder
- discovery universe ranking
- factor ranking
- quality / fundamentals ranking
- sector rotation
- small-cap research desk
- calendar and seasonality explorer

Language rules:

- "matches your criteria"
- "screen results"
- "rule strength"
- "historical context"

Avoid:

- "buy signal"
- "recommended trade"

### Scanners

Purpose:

- show what is unusual or noteworthy right now

Should include:

- unusual-activity / manipulation scanner
- catalyst scanner
- global impact
- expected range
- intraday monitor
- liquidity / market-structure context

This is the best paid conversion area because it feels high-value while remaining descriptive.

### Portfolio

Purpose:

- help users understand what they already own and how exposed they are

Should include:

- holdings import
- P&L and charges
- tax estimation
- concentration and exposure
- risk report
- journal
- paper trades
- allocation calculator
- reconcile

Should not include:

- direct buy/sell advice on holdings
- "sell this holding now"
- auto-generated replacement ideas for the user

### Lab

Purpose:

- let users validate ideas and do research

Should include:

- backtester
- optimizer
- confluence board
- chart analysis
- execution / TCA
- raw data and diagnostics
- ML probability research

Required notice:

- hypothetical performance disclaimer on every performance-heavy view

## Current App Copy That Must Change

These are concrete examples from the current app that should be rewritten before launch.

| Current wording | Replace with |
|---|---|
| STRONG BUY / BUY / WAIT / AVOID | Strong setup / Moderate setup / Mixed setup / Weak setup |
| HOLD / ADD / TRIM / EXIT | Monitor / Strengthening / Extended / Deteriorating |
| Invest fresh money | Allocation calculator |
| Latest signals | Screen results |
| Suggestions, not advice | Rule-based analytics only |
| Entry / Stoploss / Target | Rule-implied entry band / risk level / scenario exit level |
| Recommendation | Evidence summary |
| Top picks | Highlighted matches |

## Wireframes

These are intentionally low-fidelity and implementation-friendly.

### 1. Global shell

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ SwingDesk    Home   Discover   Scanners   Portfolio   Lab        Settings   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                            active section content                            │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Analytical tools only · Not investment advice · Not SEBI-registered · AI-assisted │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2. Onboarding disclaimer gate

```text
┌──────────────────────────── Welcome to SwingDesk ────────────────────────────┐
│                                                                              │
│  SwingDesk is a market analytics and research tool.                          │
│  We are not a SEBI-registered Investment Adviser or Research Analyst.        │
│  Nothing here is a buy/sell/hold recommendation or personalized advice.      │
│  Markets involve risk. Some outputs are AI-assisted.                         │
│                                                                              │
│  [ ] I understand and accept the disclaimer                                  │
│                                                                              │
│                         [ View full legal ] [ Accept & continue ]            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3. Home

```text
┌─ Market pulse ───────────────────────────────────────────────────────────────┐
│ Regime: Risk-on   Breadth: Improving   Risk tone: Neutral-positive           │
├──────────────────────────────────────────────────────────────────────────────┤
│ ┌─ Watchlist activity ─────────────────┐ ┌─ Unusual activity teaser ───────┐ │
│ │ Ticker   Move   Setup strength       │ │ XYZ  vol 6x avg  float-turn high │ │
│ │ ABC      +2.1%  Strong               │ │ PQR  gap + delivery spike        │ │
│ │ DEF      -0.9%  Mixed                │ │ [ See full scanner ]             │ │
│ └──────────────────────────────────────┘ └───────────────────────────────────┘ │
│ ┌─ Portfolio snapshot ────────────────┐ ┌─ Next 7 days ────────────────────┐ │
│ │ Holdings value   Day P&L            │ │ RBI policy                        │ │
│ │ Open positions   Concentration      │ │ Earnings / macro events           │ │
│ └─────────────────────────────────────┘ └───────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4. Discover screener

```text
┌─ Build your screen ──────────────────────────────────────────────────────────┐
│ Setup: [Breakout]  Universe: [Nifty 200]  Min quality: [60]  RR: [2.0]     │
│                                                     [ Run screen ]          │
├──────────────────────────────────────────────────────────────────────────────┤
│ Rule-based analytics only. These stocks match your criteria.                │
├──────────────────────────────────────────────────────────────────────────────┤
│ Ticker   Setup strength   Rule entry band   Risk level   Scenario upside    │
│ ABC      Strong           2890–2910         2850         +5.2%             │
│ GHI      Moderate         540–545           528          +4.1%             │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5. Unusual-activity scanner

```text
┌─ Unusual activity ───────────────────────────────────────────────────────────┐
│ Filters: Universe [All]  Lookback [20d]  Min flags [2]                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ Descriptive heuristics only. Review evidence before acting.                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ XYZ LTD   flags: 4                                                          │
│  • Turnover 6.2x 30-day average                                             │
│  • Delivery spike vs 90-day history                                         │
│  • Return abnormal vs recent realized volatility                            │
│  • No matching major public catalyst                                        │
│                               [ View evidence ] [ Add alert ]               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6. Portfolio diagnostics

```text
┌─ Holdings diagnostics ───────────────────────────────────────────────────────┐
│ Import Groww   Refresh prices   Export                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│ Value   Day P&L   Unrealized P&L   Concentration risk   Sector exposure      │
├──────────────────────────────────────────────────────────────────────────────┤
│ Ticker   Weight   P&L %   Trend quality   Earnings proximity   Risk note     │
│ ABC      18%      +6.2%   Improving       4 days              High weight    │
│ DEF      11%      -2.4%   Weakening       22 days             Normal         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7. Allocation calculator

```text
┌─ Allocation calculator ──────────────────────────────────────────────────────┐
│ User-selected stock: [ABC.NS]  Capital: [50000]  Max risk %: [1.0]          │
├──────────────────────────────────────────────────────────────────────────────┤
│ This calculator is user-directed and informational only.                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ Estimated shares    Estimated risk amount    Position concentration          │
│ 17                  ₹490                     8.5% of portfolio               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 8. Lab backtester

```text
┌─ Strategy lab ───────────────────────────────────────────────────────────────┐
│ Setup [Breakout]  Universe [Watchlist]  Max hold [12]  Costs [On]           │
│                                                   [ Run backtest ]          │
├──────────────────────────────────────────────────────────────────────────────┤
│ Hypothetical historical analysis only. Past performance does not guarantee  │
│ future results.                                                             │
├──────────────────────────────────────────────────────────────────────────────┤
│ Trades   Win rate   Expectancy   Profit factor   Max drawdown               │
│ 132      47%        0.21R        1.29            -6.8R                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Suggested Build Order

### Step 1

Make these pages launch-safe first:

- Home
- Discover
- Scanners
- Portfolio diagnostics
- Lab backtest

### Step 2

Reframe the currently risky pages:

- Invest -> Allocation calculator
- Signals -> Screen results
- Board -> Confluence board
- Holdings recommendations -> Diagnostics only

### Step 3

Remove advisory vocabulary from:

- `swingdesk/app.py`
- `swingdesk/analyze/summary.py`
- `swingdesk/analyze/glossary.py`
- `swingdesk/analyze/decision.py`
- `swingdesk/portfolio/allocate.py`
- `swingdesk/portfolio/holdings.py`
- `swingdesk/analyze/thesis.py`

## Immediate Implementation Checklist

- Add legal pages and linked disclaimers
- Remove or rename advisory page titles
- Replace BUY/SELL/HOLD/AVOID vocabulary in UI
- Convert "Invest" into calculator-only workflow
- Convert "Latest signals" into screener output
- Remove holdings action recommendations from default UI
- Add hypothetical-performance disclaimer to all backtest/performance surfaces
- Ensure marketing copy matches the tool-only model

## Outcome

If we follow this document, the first public SwingDesk product becomes:

**A user-driven analytics workbench with scanners, rankings, diagnostics, alerts, and backtesting for Indian active investors, without crossing into recommendation-led advisory behavior.**
