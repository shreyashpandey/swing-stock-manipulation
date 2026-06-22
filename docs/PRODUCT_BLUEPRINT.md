# SwingDesk — Product Blueprint

> Reference document for turning the local SwingDesk app into a sellable consumer product.
> Status: planning. Last updated: 2026-06-19.
> Read this top-to-bottom once; after that use it as the map for "how we operate."

---

## 0. TL;DR — the one-paragraph version

We are turning a single-user local Streamlit app into **"a quant analytics & screening workbench for
Indian swing traders"** — a *tool*, not an advisory. We make money on **subscriptions** (freemium,
~₹4,999/yr Pro). To stay clear of SEBI we **never tell anyone what to buy/sell**; we show data,
rankings, scanners, backtests and let the user decide. The headline paid feature and primary
marketing weapon is the **Unusual-Activity (manipulation) scanner**. The current 22 scattered tabs
collapse into **5 clean sections**: Home · Discover · Scanners · Portfolio · Lab.

---

## 1. The operating model — how the market leaders work, and which one we copy

| App | What it is | SEBI status | Execution | Our takeaway |
|---|---|---|---|---|
| **smallcase** | Platform hosting curated stock *baskets* | **Tech platform** — baskets built by registered RA/RIA/PMS managers; smallcase itself is rails | One-click via 15+ linked brokers; user owns stocks in own demat | The *platform* can be unregistered **if** the advice comes from a registered party. Separation of "rails" vs "advice." |
| **Univest** | Stock advisory + broker | **Registered RA (INH000013776) + IA (INA000017639)** | One-click, GTT orders | Full Path A: AI screens 5,000 stocks → registered analysts approve → calls with entry/SL/target. Requires registration. |
| **Liquide** | AI research assistant ("LiMo") + ideas | **Registered RA** | Via smallcase rails + brokers | Even the "AI gives Buy/Sell/Hold" model is done as a *registered RA*. AI doesn't exempt you. |
| **Tickertape** *(smallcase-owned)* | Pure analytics / screener / scores | **No registration — it's a tool** | None (read-only) | **★ This is our model.** Data, scores, screeners, portfolio tracking. Never says "buy." Sells subscriptions. |

**The decisive insight:** every app that gives *recommendations* is a registered RA/RIA. The only way to
sell **without** registration is to be the **analytics-tool layer (Tickertape)**, not the advice layer.
So:

- **Phase 1 (now → launch): be Tickertape.** Analytics workbench, Path B, no registration.
- **Phase 2 (optional, later): add an advice layer** either by (a) getting RA-registered yourself, or
  (b) becoming/partnering with a **smallcase manager** so the registered party supplies the calls and
  we stay the tech rails. Design the app now so this can bolt on later without a rewrite.

> SEBI RA rules tightened on **16 Dec 2024**: deposit-with-bank instead of net-worth cert (scaled by
> client count), mandatory NISM certification, a **mandatory disclosure when AI tools are used in
> research**, and a required public website. We design for these now (see §4) so Phase 2 is cheap.

---

## 2. Product principles — the "tool, not advice" rules (baked into every screen)

A feature stays a sellable tool when it passes all three tests. Fail one → it's regulated advice.

| Test | ✅ Tool (we do this) | 🛑 Advice (we never do this) |
|---|---|---|
| **Who decides?** | User sets the filter/rule → app shows matches | App picks the stock and tells you to act |
| **Descriptive or prescriptive?** | "Volume 5× avg, float turnover high" | "Buy this" / "AVOID" |
| **Generic or personalized?** | Same ranking shown to everyone | "Given *your* holdings, do X with *this* much" |

### Language rules (enforce in code + copy review)

| ❌ Never use | ✅ Use instead |
|---|---|
| BUY / SELL / AVOID / HOLD | Strong / Neutral / Weak setup · Meets / Fails criteria |
| "Target" / "Stoploss" (as instruction) | "Rule-implied level" / "Backtested exit level" |
| "You should…" / "Recommended" | "Stocks matching your screen" / "Historically…" |
| "Invest ₹X here" | A *calculator* where the user inputs the stock & amount |
| "Will go up" | "Model probability" / "Expected range (statistical)" |

> Code touchpoints: replace BUY/WAIT/AVOID vocabulary in `analyze/summary.py` + `analyze/glossary.py`;
> reframe the "Latest signals" tab as a screener; cut/recalculate the "Invest fresh money" tab into a
> pure calculator.

---

## 3. Information architecture — fixing "too many jumps"

**Problem today:** 22 flat tabs, no hierarchy, no obvious starting point. **Fix:** 5 primary sections
+ global utilities. Every current tab has exactly one home.

### New navigation

```
┌─ Primary nav (always visible) ───────────────────────────────────────────┐
│  🏠 Home   🔍 Discover   📡 Scanners   💼 Portfolio   🧪 Lab              │
└───────────────────────────────────────────────────────────────────────────┘
   Top-right utilities:  🔭 Watchlist   🔔 Alerts   👤 Account/Plan   ⚙️ Settings
```

1. **🏠 Home** — daily command center. *"What do I look at first?"*
   Market-regime banner · watchlist movers · today's flagged unusual activity (teaser) ·
   portfolio snapshot · upcoming events (7 days) · quick actions (refresh data).

2. **🔍 Discover** — find ideas (USER-DRIVEN screening). *"What fits my criteria?"*
   Screener (the reframed "signals") · Factor Ranking · Quality Ranking (fundamentals) ·
   Sector & micro-sector Rotation · New-universe Discovery + Smallcap workbench · Calendar/Seasonality.

3. **📡 Scanners** — what's happening now (DESCRIPTIVE analytics). *"What's unusual today?"*
   **Unusual-Activity / Manipulation scanner ★** · Intraday monitor (VWAP/ORB/RVOL) ·
   Expected Range · Market Regime / US→India Spillover detail.

4. **💼 Portfolio** — track & manage. *"How am I doing & how much should I size?"*
   Holdings (Groww import) + analysis · Paper trades · Journal · Risk & Position Sizing ·
   Reconcile/Drift · (Allocation **calculator** — reframed, user picks the stocks).

5. **🧪 Lab** — research & validate (POWER tools). *"Does my idea actually work?"*
   Backtester (walk-forward) · Parameter Optimizer · ML Probability model · Backtest-vs-Paper drift.

**Utilities (not primary tabs):** Watchlist (global, reachable anywhere) · Alerts/Telegram config ·
Account & Plan · Settings (data sources, coverage) · Legal & Disclaimers.

### Mapping — every current tab → new home

| Current tab (app.py) | New location | Note |
|---|---|---|
| Actions | Home (quick actions) + Settings | split |
| Watchlist | Global utility (top bar) | reachable anywhere |
| Discovery (find new stocks) | Discover › Discovery | |
| **Invest fresh money** | Portfolio › Allocation **calculator** | ⚠️ reframe (user picks stocks) or cut |
| Small-cap workbench | Discover › Smallcap | |
| **Latest signals** | Discover › Screener | ⚠️ reframe as criteria matches |
| Sector rotation | Discover › Sector Rotation | |
| **Unusual-activity scanner** | Scanners › Unusual Activity ★ | headline paid feature |
| Seasonality & event calendar | Home + Discover › Calendar | |
| Walk-forward backtest | Lab › Backtester | |
| Analyze Groww holdings | Portfolio › Holdings | strip buy/sell conclusions |
| Parameter optimizer | Lab › Optimizer | |
| Paper + real portfolio | Portfolio › Paper / Holdings | |
| Paper vs Backtest drift | Lab › Drift | |
| Fundamental quality ranking | Discover › Quality Ranking | |
| Watchlist price coverage | Settings › Data Coverage | |
| Spillover & regime | Home banner + Scanners › Regime | |
| Expected price range | Scanners › Expected Range | |
| Position sizing & risk | Portfolio › Risk & Sizing | |
| Factor ranking | Discover › Factor Ranking | |
| ML direction | Lab › ML Probability | keep as probability only |
| Intraday monitor | Scanners › Intraday | |

---

## 4. Compliance & disclaimers — where every notice lives

Five layers. Disclaimers are not decoration — they are the legal boundary that keeps us on Path B.

1. **Onboarding gate (first launch / signup) — BLOCKING.**
   Modal the user must actively accept before entering. Store `disclaimer_accepted_at` + version
   against the `user_id`. Re-prompt if the disclaimer version changes.

2. **Persistent footer (every screen).**
   > *SwingDesk provides analytical tools and information for educational purposes only. It is not
   > investment advice and we are not a SEBI-registered Investment Adviser or Research Analyst.
   > Markets carry risk; do your own research. Some analysis is generated using automated / AI tools.*

3. **Contextual banner (analytics screens: Screener, Scanners, Expected Range, ML).**
   Slim inline note: *"Rule-based analytics, not a recommendation."*

4. **Backtest / performance disclaimer (Lab + anywhere returns are shown).**
   *"Hypothetical, past performance. Not indicative of future results."* — SEBI is strict on
   performance claims; never show a win-rate without this.

5. **Settings › Legal hub.** Full T&C · Privacy Policy · Risk Disclosure · Data-source attribution ·
   Refund/Cancellation policy · AI-usage disclosure.

> **AI disclosure:** because we use Claude/AI in sentiment & analysis, we disclose it everywhere (point
> 2 above). This is good practice now and a hard requirement if we ever go Phase 2 (registered RA).

---

## 5. Wireframes

Low-fidelity, structure-only. Web/desktop layout shown; collapses to mobile bottom-nav later.

### 5.0 Global shell

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ SwingDesk        🏠 Home  🔍 Discover  📡 Scanners  💼 Portfolio  🧪 Lab        │
│                                   🔭 Watchlist  🔔  👤 Pro ▾  ⚙️                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│                          << ACTIVE SECTION CONTENT >>                          │
│                                                                                │
├──────────────────────────────────────────────────────────────────────────────┤
│ ⓘ Educational tools only · Not investment advice · Not SEBI-registered · AI-assisted │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Onboarding / disclaimer gate (blocking, first run)

```
┌───────────────────────── Welcome to SwingDesk ──────────────────────────┐
│                                                                          │
│  Before you start, please read and accept:                              │
│                                                                          │
│  • SwingDesk is an ANALYTICS & RESEARCH TOOL, not an advisory.          │
│  • We do NOT provide buy/sell recommendations or investment advice.     │
│  • We are NOT a SEBI-registered Investment Adviser or Research Analyst.  │
│  • All outputs are informational; markets carry risk of loss.           │
│  • Some analysis is generated using automated / AI tools.               │
│                                                                          │
│   [ ] I have read and understood the above and the full Disclaimer.      │
│                                                                          │
│                      [ View full Disclaimer ]   [  Accept & Continue  ]  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Home (dashboard)

```
┌─ MARKET REGIME ────────────────────────────────────────────────────────┐
│ 🟢 Risk-On  ·  US→India spillover: +0.6% bias  ·  Nifty above 50/200 EMA │  ← Scanners›Regime
├─────────────────────────────────────────────────────────────────────────┤
│ ┌─ Watchlist movers ───────────┐  ┌─ Today's flags (Unusual activity) ─┐ │
│ │ TICKER   %chg   setup-strength│  │ 🚨 XYZ  vol 6× · float-turn high  │ │  ← teaser;
│ │ ABC      +2.1%  ● Strong      │  │ 🚨 PQR  gap+OBV spike             │ │    full list = Pro
│ │ DEF      -0.8%  ○ Neutral     │  │ [ See all in Scanners → ]         │ │
│ └───────────────────────────────┘  └────────────────────────────────────┘ │
│ ┌─ Portfolio snapshot ─────────┐  ┌─ Next 7 days (events) ────────────┐ │
│ │ Value ₹4.2L  Day P&L +₹3,100 │  │ Tue  RELIANCE results             │ │
│ │ Open positions: 6            │  │ Thu  RBI policy                   │ │
│ │ [ Go to Portfolio → ]        │  │ [ Calendar → ]                    │ │
│ └───────────────────────────────┘  └────────────────────────────────────┘ │
│  Quick actions:  [↻ Refresh prices] [↻ Fetch news] [+ Add to watchlist]   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Discover › Screener (the reframed "signals")

```
┌─ Build your screen ───────────────────────────────────────────────────────┐
│ Setup: [Breakout ▾]  Universe:[Nifty 200 ▾]  Min quality:[60]  RR≥:[2.0]   │
│ [ Run screen ]                                          [ Save this screen ]│
├─────────────────────────────────────────────────────────────────────────┤
│ ⓘ These stocks MATCH YOUR CRITERIA. This is not a buy list. Rule-implied   │  ← contextual
│   levels are screen outputs — backtest before acting.                      │    disclaimer
├─────────────────────────────────────────────────────────────────────────┤
│ Ticker  Setup-strength  Rule entry  Rule exit  Rule-implied tgt  RR  →    │
│ ABC     ● Strong        2,900       2,850      3,050             2.4  [▸]  │
│ GHI     ◐ Medium        540         528        575              2.1  [▸]  │
│ ...                                                       [ Backtest these ]│
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Scanners › Unusual Activity (★ headline feature — purely descriptive)

```
┌─ Unusual-activity scanner ─────────────────────────────────────────────────┐
│ Filters: [All universe ▾]  Lookback:[20d ▾]  Min flags:[2 ▾]               │
├─────────────────────────────────────────────────────────────────────────┤
│ ⓘ Descriptive heuristics — operator-footprint signals. NOT a recommendation.│  ← disclaimer
├─────────────────────────────────────────────────────────────────────────┤
│ 🚨 XYZ LTD                                                       flags: 4  │
│    • Turnover 6.2× 30-day avg          • Day vol = 18% of free float       │
│    • +14% on no news                   • Delivery% spike vs 90-day         │
│    [ View chart + evidence ▸ ]                          [ 🔔 Alert me ]     │
│ ─────────────────────────────────────────────────────────────────────────│
│ 🚨 PQR LTD                                                       flags: 3  │
│    ...                                                                     │
└─────────────────────────────────────────────────────────────────────────┘
  (Free tier: 1 flag/week shown + count blurred · Pro: full list + alerts)
```

### 5.5 Portfolio

```
┌─ Holdings (imported from Groww) ─────────────────  [ ↻ Re-import ]  [+ Manual]┐
│ Ticker  Qty  Avg   LTP    P&L      Setup-state   Risk flag                   │
│ ABC     50   2,710 2,901  +₹9,550  ● Strong       —                          │
│ DEF     30   1,200 1,150  -₹1,500  ○ Weak         ⚠ below initial stop       │  ← descriptive,
├──────────────────────────────────────────────────────────────────────────┤    no "sell" verb
│ [ Holdings ] [ Paper trades ] [ Journal ] [ Risk & sizing ] [ Reconcile ] │
├─ Risk & position sizing (calculator — YOU pick the inputs) ────────────────┤
│ Capital ₹[5,00,000]  Risk/trade [1%]  Entry [2,900]  Stop [2,850]          │
│ → Suggested qty: 100   |  Portfolio heat: 3.2%  |  Sector concentration: ok │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.6 Lab › Backtester

```
┌─ Walk-forward backtest ────────────────────────────────────────────────────┐
│ Strategy:[Breakout ▾]  Universe:[Nifty 200 ▾]  Period:[2019–2026]          │
│ Params: SL[ATR×1.5]  Target[ATR×3]  Hold[≤15d]            [ Run backtest ]  │
├─────────────────────────────────────────────────────────────────────────┤
│ ⚠ HYPOTHETICAL past performance. NOT indicative of future results.        │  ← mandatory
├─────────────────────────────────────────────────────────────────────────┤
│ CAGR 18%  ·  Win 54%  ·  MaxDD -12%  ·  Sharpe 1.1  ·  Trades 240         │
│  [equity curve chart]                                                      │
│                                       [ Save as screen ]  [ Optimize → Lab]│
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.7 Account & Plan (upgrade)

```
┌─ Your plan: FREE ──────────────────────────────────────────────────────────┐
│  Free                    Pro ₹4,999/yr  ★ best     Quant ₹11,999/yr        │
│  • 15-stock watchlist    • Full universe           • Everything in Pro     │
│  • Basic screener        • Unusual-activity (full) • ML probability        │
│  • Portfolio tracker     • Factor/sector/backtest  • Optimizer · Intraday  │
│  • 1 flag/week (teaser)  • 10 alerts/day           • Unlimited alerts · API│
│       [current]              [ Upgrade — annual ]      [ Upgrade ]          │
│                          Pay via Razorpay · GST invoice · cancel anytime    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Monetization overlay (mapped to the IA)

| Section | Free | Pro (₹4,999/yr) | Quant (₹11,999/yr) |
|---|---|---|---|
| Home | ✅ | ✅ | ✅ |
| Discover › Screener | 1 basic screen | All + custom + save | + smallcap |
| Discover › Factor/Quality/Sector | — | ✅ | ✅ |
| Scanners › Unusual Activity ★ | teaser (1/wk) | ✅ full + alerts | ✅ |
| Scanners › Intraday / Expected Range | — | Expected Range | + Intraday |
| Portfolio (tracker + Groww import) | ✅ (the hook) | ✅ | ✅ |
| Portfolio › Risk & sizing | basic | ✅ | ✅ |
| Lab › Backtester | — | ✅ | ✅ + Optimizer |
| Lab › ML Probability | — | — | ✅ |
| Alerts (Telegram) | — | 10/day | unlimited |
| API / export | — | — | ✅ |

**Secondary revenue:** broker partner/affiliate links (Upstox/Angel One/Dhan — verify current terms),
later a B2B API for the manipulation/factor data. **Avoid:** any "tips"/calls channel (→ SEBI advice).

**Rough unit economics:** ~₹3,900 net/Pro user/yr after GST+gateway; ~₹50k/mo fixed early →
**break-even ≈ 155 paying users**. Margins climb steeply because analytics are **computed once
server-side** and served to all users (see §7).

---

## 7. Technical architecture (single-tenant → multi-tenant SaaS)

| Concern | Today | Target |
|---|---|---|
| Tenancy | One SQLite file, global watchlist/holdings | Postgres + `user_id` on every user-scoped table |
| Auth | none | Email/OTP/Google login; `plan` field per user for feature gating |
| Compute | recomputes ~30 indicators **per session** (lag) | **Batch once server-side** (Celery/RQ + Redis), UI just reads |
| Data | yfinance + NSE scrape (not licensable) | Licensed feed (broker API / TrueData / GDFL) |
| Frontend | Streamlit | MVP: Streamlit + auth. Scale: FastAPI backend + React/Next.js |
| Payments | none | Razorpay subscriptions + GST invoicing |
| Secrets | `.env` | Secrets manager |
| AI cost | per call | batch news sentiment once for all users; cap per-user thesis |

The single most important architectural change is **batch-compute-once**: market data is identical for
all users, so signals/scanners must be computed by a scheduler and stored, not recomputed per click.
This is what makes margins work and the app fast.

---

## 8. Build roadmap

| Phase | Goal | Key work |
|---|---|---|
| **0. De-risk (1–2 wk)** | Don't build on sand | Confirm Path B with a SEBI lawyer; license a data feed for trial; lock copy/disclaimer language |
| **1. Reframe (1–2 wk)** | Make it legally a tool | Kill BUY/WAIT/AVOID → setup-strength; "signals"→screener; "invest fresh money"→calculator; add 5 disclaimer layers |
| **2. Re-IA (1–2 wk)** | Kill the confusion | Implement 5-section nav + mapping (§3); add Home dashboard |
| **3. Multi-tenant MVP (4–8 wk)** | First paying users | Postgres + `user_id`; auth; plan-gating (§6); Razorpay; batch-compute |
| **4. Productize (8–12 wk)** | Real SaaS | FastAPI + React; Celery/Redis batch engine; onboarding; licensed feed at scale |
| **5. (optional) Advice layer** | Premium | RA registration **or** smallcase-manager partnership for an advisory tier |

---

## 9. Open decisions (non-code — you own these)

- [ ] Confirm **Path B** with a SEBI-compliance lawyer (and what disclaimer wording they want).
- [ ] Choose a **licensed data vendor** (broker API vs TrueData/GDFL) — drives recurring cost & pricing.
- [ ] Decide **web-first (Streamlit MVP) vs mobile-first** (competitors are mobile; consumers are mobile).
- [ ] Confirm current **broker affiliate** terms (Upstox/Angel One/Dhan).
- [ ] Pricing: launch **1 paid tier** or 2? (recommend: Free + Pro only at launch.)
- [ ] Brand/name + domain (current "SwingDesk" — check trademark/availability).

---

## 10. References

- smallcase — model & broker integration: https://www.smallcase.com/learn/what-is-smallcase/ ,
  who manages: https://tejimandi.com/blogs/unboxing-smallcases/who-manages-smallcase-role-of-sebi-ria-registered-investment-advisor
- Univest — SEBI-registered RA/IA advisory: https://univest.in/
- Liquide — SEBI-registered RA + AI: https://liquide.life/
- SEBI Research Analyst amendments (16 Dec 2024): https://www.compliancecalendar.in/learn/sebi-research-analyst-third-amendment-regulations-2024
</content>
</invoke>
