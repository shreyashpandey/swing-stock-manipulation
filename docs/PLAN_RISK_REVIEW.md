# Business plan review: product, engineering and Indian regulatory risks

Reviewed: 2026-09-11. Applies to the proposed Indian customer launch.

## Review conclusion

Continue internal development, but do not treat the current app or the original
35-engineering-day estimate as ready for a paid public launch. The direction is
viable to investigate; the registration position, commercial data rights, customer
demand and production safeguards are not yet established.

The most consequential flaws are:

1. Earlier docs describe analytics labels and disclaimers as a regulatory boundary.
   That is not a reliable legal test. The code still emits recommendations.
2. The end-of-day product is inconsistent with parts of the app's explosive-move
   positioning and may not solve the problem that initially motivated the product.
3. Global local storage and caches cannot safely become a multi-user service just
   by adding login.
4. Missing sessions and unadjusted corporate actions can produce misleading
   returns, rankings and alerts even when every request succeeds.
5. The plan leaves privacy lifecycle, incident reporting, consumer grievance
   handling and future effective dates under-specified.
6. The calendar, annual pricing economics and activation metric are too optimistic
   to support a spending or launch commitment.

Severity: **P0** must be resolved before the affected public capability is exposed;
**P1** must be resolved before charging or expanding the beta; **P2** is later-phase
work. A P0 engineering gap is not automatically an allegation of illegality.
Finding IDs in this review are separate from implementation-backlog IDs in the plan.

Evidence labels below distinguish **observed code**, **reproduced behavior**, and
**plan gap**. This was a source/design review and limited local reproduction, not
a penetration test, certification, production load test, or legal opinion.

## 1. Frontend and app findings

| ID / priority | Failure and evidence | User consequence | Required change and acceptance test |
|---|---|---|---|
| F01 / P0 | **Observed:** `app.py` disclaimer says no recommendations, while `_ACTION_BADGE`, holdings and board views expose STRONG BUY and holding actions. `decision.py` and `portfolio/holdings.py` generate them | Contradictory representation; actionable outputs escape the planned scope | Reviewed public response allowlist, not only hidden navigation or changed labels. Inspect UI, APIs, exports, email and error payloads for prohibited outputs; preserve factual trade-side records |
| F02 / P0 | **Reproduced:** `research.snapshot()` calls two stored observations a 1-session return even when ten calendar days apart | A stale/gapped series can look like today's mover; misleading comparisons and rule matches | Match observations to expected exchange sessions. Show missing/suspended status and last-traded change separately. Test holidays, missing sessions, suspensions and timezone boundaries |
| F03 / P1 | **Observed:** screens use technical metric keys, editable rows and an empty default rule set; invalid conditions stop rendering downstream actions | Beginners unintentionally run an unfiltered universe or abandon editing | Plain-language builder with units, inline errors, explicit all-symbols state, draft saving and preview counts. Keyboard-only user can complete/edit/save a rule |
| F04 / P1 | **Observed:** same-name saves overwrite; deletes act immediately; there is no revision conflict control | Mistyped names or two open tabs destroy/update the wrong research | Stable IDs, distinct Create/Update/Save copy actions, revision/ETag conflict response, undo or focused deletion confirmation. Exercise simultaneous edits and failed-save retries |
| F05 / P1 | **Plan gap:** 25 Pro rules with a 500 rule-symbol cap, but no explanation of capacity; Free has no aggregate cap | A user expects 25 rules over 100 symbols and instead needs 2,500 evaluations; silent skipping would break the offer | Show exact usage before saving; cap both tiers; reject or pause explicitly, never silently truncate; show which symbols/rules are affected |
| F06 / P1 | **Observed/planned:** saved screens and alerts retain ticker snapshots, while users can edit or activate a watchlist separately | User thinks new members are monitored when they are not | Show saved universe/version, count, and a deliberate Refresh membership action. Test rename/deletion of the originating list and unsupported symbols |
| F07 / P1 | **Plan gap:** responsive web, narrow-screen tables, keyboard access and screen-reader chart alternatives have no acceptance criteria | Core research is unusable on a phone or without colour/hover | Test 360px and desktop layouts, 200% zoom, labelled inputs, visible focus, non-colour statuses and chart data tables. Document supported browsers and slow-network behavior |
| F08 / P1 | **Plan gap:** page response target excludes rendering and network; no loading/error/expired-session state matrix | Duplicate saves, lost drafts, empty pages and paid-but-locked confusion | Define loading/empty/no-match/missing/stale/error/unauthorized/over-limit states. Keep drafts through sign-in recovery; payment-pending view reconciles without prompting a second purchase |
| F09 / P1 | **Observed:** comparison requires every symbol to have common positive-close dates; a single missing symbol blanks the chart | Adding an IPO or unavailable symbol removes useful context for all names | Identify the limiting symbol and overlapping dates, offer an explicit exclusion, and show requested versus actual coverage. Never silently switch the time interval |
| F10 / P0 | **Plan gap:** score semantics are not a public contract; explosive-move wording appears in the internal app | User reads an ordinal score of 80 as an 80% probability or treats a retrospective flag as a prediction | State score meaning, components and known limits. Hide uncalibrated future-move probabilities from public routes; comprehension-test evidence cards and marketing |

Code references: [app](../swingdesk/app.py),
[research UI](../swingdesk/research_ui.py),
[research calculations](../swingdesk/analyze/research.py).
The existing new-workspace captions already disclose manual alerts and stored
data; those are useful safeguards, but do not fix the missing-session calculation.

## 2. Backend, data and operations findings

| ID / priority | Failure and evidence | Required design and verification |
|---|---|---|
| B01 / P0 | **Observed:** `research_items`, watchlist, portfolio and acceptance settings have global local ownership. Private risk/portfolio caches have no authenticated owner context | Owner-scoped repositories, explicit admin role, per-user legal acceptance, authorization in jobs/downloads, owner-aware caches. Account B cannot retrieve A's objects via guessed IDs, exports, cached pages, background tasks or support endpoints |
| B02 / P0 | **Plan gap:** hosted identity is named, but session expiry, recovery, cookie/CSRF policy, token validation and administrator security are unspecified | Threat model and managed identity integration with issuer/audience/expiry checks, secure session storage, origin/CSRF controls where relevant, redirect allowlists and admin MFA. Test logout/revocation, recovery and cross-origin writes |
| B03 / P0 | **Observed:** `fetch_one()` requests `auto_adjust=False`; `upsert_prices()` drops adjusted close and retains only raw OHLCV | Preserve raw and adjustment data, version corporate actions, select a documented basis for every indicator. Test splits, bonuses, dividends and corrected histories. Raw price changes must not be marketed as total return |
| B04 / P0 | **Observed:** snapshot loads latest rows per ticker independently; prices overwrite by ticker/date and fundamentals by ticker | Immutable/versioned observation batches, field-level timestamps, observed-at versus available-at, published batch pointer and correction lineage. All results in one response use the same permitted snapshot; historical tests cannot see future fundamentals |
| B05 / P1 | **Observed:** snapshot reads full fundamentals and opens a price query per ticker; each alert repeats snapshot work; `load_prices(days=...)` fetches all history before trimming | Load a bounded batch once and evaluate rules against it. Use indexed/bounded queries, query budgets and worker profiling. Measure cold-cache database and queue behavior under Free-user load plus ingestion |
| B06 / P0 | **Plan gap:** one worker/outbox without leases, failed-job handling and transaction boundaries is not a recovery design | Atomic job claiming, lease expiry, idempotent evaluation, unique event keys, retry budget, dead-letter state and replay tool. Kill the worker before/after event commit and provider send; verify recovery without duplicate logical events |
| B07 / P0 | **Plan gap:** an edited rule or corrected batch may be re-evaluated against a different universe/threshold; no correction notification policy | Pin rule/universe/formula/batch versions at scheduling. Same-session edits apply next session unless explicitly previewed. Preserve the original event; expose a correction/retraction referencing it; skip/quarantine unsupported data |
| B08 / P0 | **Plan gap:** provider reconciliation alone cannot restore the lost internal order-to-user mapping after a database rollback | Persist immutable internal order IDs mapped to account, provider IDs, price, currency, access period and refund state. Use point-in-time recovery and durable webhook receipt. Reconcile to a provider watermark before resuming access mutations; test paid/refunded events spanning restore |
| B09 / P0 | **Plan gap:** deletion has no propagation to jobs, replicas, exports, email processors or restored backups; retention is unspecified | Data inventory and purpose/retention matrix; stop jobs, revoke sessions/links, delete/anonymize eligible records across processors, isolate legally retained records. Replay deletion tombstones after restore; test an account deleted while a digest is queued |
| B10 / P1 | **Observed:** ticker validation is lexical, not instrument validation; numeric rules accept booleans; there are no API-sized payload budgets | Stable instrument master with exchange/series/currency/status, strict typed finite values excluding booleans, max rule/ticker/body/history limits and schema versions. Reject invalid/unsupported securities before accepting payment for their coverage |
| B11 / P1 | **Observed/plan gap:** CSV output is direct `to_csv`; public news/AI rendering and future broker uploads lack a documented trust boundary | Escape spreadsheet formula prefixes in text cells while preserving real numeric types; sanitize external text/links, constrain fetch destinations, isolate uploads with size/row/type limits, and treat article text as untrusted model input. Do not expose arbitrary fetch or webhook URLs |
| B12 / P0 | **Plan gap:** next-business-day support and generic backups are the only operational commitments | Separate security incident response from ordinary support; monitored alerting, named backup contact, applicable reporting clocks, secure logs and tested restore. Provider outages and a sick founder must have a defined response |
| B13 / P1 | **Observed:** dependencies have lower bounds without a locked deployment set; migration/rollback order and secrets rotation are not specified | Reproducible locked builds, migration compatibility tests, software/dependency inventory, secret scanning and rotation, separated environments, staged release and rollback rehearsal |

Code references: [storage](../swingdesk/storage.py),
[price ingestion](../swingdesk/ingest/prices.py),
[research engine](../swingdesk/analyze/research.py),
[cached views](../swingdesk/app.py).
These are production-transition findings; global state is expected in a local
single-user application, but must not be presented as multi-user-ready.

### Local reproduction performed

Mock `load_prices()` with closes 100 on 2026-09-01 and 110 on 2026-09-11, and mock
fundamentals empty. `snapshot(["TEST.NS"])` returns `change_1d ≈ 10%` with
`price_as_of = 2026-09-11`. The old observation is not checked against the prior
exchange session. No customer database was used for this reproduction.

`validate_rules([{"metric": "close", "op": ">=", "value": True}])` also succeeds.
This is a strict-validation gap for a future API, not a demonstrated remote exploit.

## 3. Business and delivery-plan flaws

| ID / priority | Flaw | Correction |
|---|---|---|
| P01 / P0 | Regulatory and data-rights checks are described early, but the milestone table places confirmation after the external closed beta | Gate every external exposure, including free demos and trial accounts, on the permissions/review applicable to that exposure. Use synthetic data and internal-only access until then |
| P02 / P1 | 35 engineering days allocates 5 days to auth/DB/migration, 3 to billing, 3 to operations and 2 to release testing | Replace with a provisional 62–97 engineering-day range plus business work and contingency. A 12-week prototype remains possible; a solo paid product needs a planning range around 18–26 weeks, subject to re-estimation and external lead times |
| P03 / P1 | The assumed target is after-close research, but earlier user needs and existing copy focus on rare next-day/intraday 10–15% moves; a liquid 200-symbol universe may exclude the desired names | Interview around the actual decision and required data frequency before fixing coverage. Do not market the beta as an explosive-move predictor. Measure whether the EOD evidence workflow itself is valuable |
| P04 / P1 | ₹2,999 is both a checkout proposal and the model's net revenue; free-user costs, variable licence costs and annual refund exposure are uncertain | Separate customer payment, indirect taxes, recognized revenue, fees, refunds and serving cost. Model base/downside licensing and Free:Paid ratios. The original break-even figures are illustrative only, not a budget |
| P05 / P1 | Activation requires a matching result and willingness-to-pay interviews substitute for purchases | A zero-match screen can be successful. Define activation as saved universe/rule plus understood evaluated outcome and timestamps. Separately measure evidence engagement and actual paid conversion with cohort denominators |
| P06 / P1 | 95% overall coverage and 99% evaluations can pass while every missing symbol belongs to one customer | Show per-symbol/per-rule status, coverage for each account, and causes of every skip. No silent gaps. Track all entitled scheduled pairs as well as the valid-data subset so excluding failures cannot improve the score |
| P07 / P1 | Broad Free access and scanner-heavy acquisition can overload fixed-price subscriptions; support and safety have no cost owner | Add hard budgets for aggregate rule-symbol slots, query volume, exports and email. Assign founder/support/incident ownership with explicit backup and budget; do not promise unlimited use |
| P08 / P1 | Conflicting older docs remain authoritative-sounding and can reintroduce deferred or unreviewed features | Mark the reviewed plan as governing beta scope; treat older competitor claims and legal conclusions as historical assumptions. Every future feature changes the rights/security/regulatory checklist |

## 4. What Indian rules restrict

There is no single government approval for a stock analytics website, and not every
restriction below is a ban on the feature. Distinguish regulated activity, conduct
prohibitions, contractual permissions, and future rules. An India-focused launch
also needs entity-specific professional review; these findings do not determine
SwingDesk's legal classification.

### G01 — Securities research and advice: registration-dependent, P0

Paid research reports, security-specific recommendations, price targets and stop
loss targets can fall within the RA framework. Personalized portfolio advice can
raise IA requirements. Relevant exemptions and definitions must be assessed on
the actual service; merely attaching an individually registered person does not
establish that the selling entity and all of its activities are covered.
[RA Regulations, definitions and registration](https://www.sebi.gov.in/sebi_data/attachdocs/aug-2026/1785912002778.pdf),
[IA Regulations](https://www.sebi.gov.in/sebi_data/attachdocs/aug-2026/1785912187595.pdf).

SEBI's February 2026 RA FAQs explicitly reject a blanket exemption for
security-specific technical research. Exclusions such as company financial
statistics and sector/index technical analysis do not automatically cover stock
rankings, probability views or confluence-generated price levels.
[RA Master Circular, Annexure I, FAQs 4–8](https://www.sebi.gov.in/sebi_data/attachdocs/feb-2026/1770375507051.pdf).

**Application to SwingDesk:** block current personalized holding actions and
trade-instruction outputs from the customer surface. Submit representative
scanner cards, defaults, rankings, alerts and marketing for written scope review.
If the intended paid value depends on regulated research/advice, choose the
appropriate registered operating model and re-plan; do not disguise it as software.

### G02 — Educational use is not an EOD data workaround, P0

SEBI's 8 May 2026 circular applies from 1 July 2026. It prescribes a 30-day lag for
the covered educational price-data sharing/use arrangements, with a narrow NISM
simulation exception. It also updates how sole educators may discuss securities
using recent price data. This is not a blanket ban on properly permitted commercial
charting, but defeats an assumption that yesterday's prices are automatically
permitted because the app says educational.
[SEBI educational price-data circular](https://www.sebi.gov.in/sebi_data/attachdocs/may-2026/1778242522289.pdf).

**Required:** document the actual commercial supply/use route separately from any
education-only route; review demos, videos and educator partnerships on that basis.

### G03 — Exchange/vendor rights: contractual restrictions, P0

NSE's policy makes commercial use and redistribution subject to the relevant
agreement. A feed purchase does not necessarily grant exports, derived-data,
notification, historical retention, or unlimited end-user rights.
[NSE data policy](https://www.nseindia.com/static/market-data/nse-data-policy).

This is an exchange/licensor obligation, not a statement that the government bans
CSV export. Verify NSE and BSE separately, plus news and fundamentals permissions.
Store entitlement and source lineage with data; enforce export rights server-side.
The current yfinance/NSE/BSE connectors are not evidence of licensed SaaS rights.

### G04 — Deceptive interfaces and misleading offers: conduct restrictions, P0

The existing dark-pattern framework addresses misleading urgency, hidden charges,
subscription traps and other deceptive choices. Fake scarcity, a false original
price, forced marketing consent, or a misleading guaranteed-gain claim cannot be
fixed by a disclaimer. Ordinary paywalls and clearly agreed prepaid access are
not inherently prohibited.
[Government explanation of the 2023 dark-pattern guidelines](https://www.pib.gov.in/Pressreleaseshare.aspx?PRID=2146813&lang=2&reg=48).

**Required:** accurate coverage/frequency before checkout, explicit total price,
clear access period and renewal behavior, a usable grievance/refund path, and
truthful marketing. Apply the relevant e-commerce duties to our own-service model;
do not copy marketplace-only obligations indiscriminately.
[E-Commerce Rules and amendments reference](https://www.wipo.int/wipolex/en/legislation/details/23198).

### G05 — Consumer rule changes during our launch horizon, P1 with effective date

A government announcement dated 10 September 2026 states that the 2026
E-Commerce Amendment Rules take effect on **1 January 2027**. It describes NCH
convergence participation, complaint copies, sponsored-listing disclosures,
30-day prior-price history for reductions, and annual dark-pattern self-audit
requirements. Marketplace-specific changes have a different scope.
[Official announcement](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2308759&lang=1&reg=48).

The announcement was reviewed; a consolidated amended Gazette text was not
verified in this pass. Obtain it and confirm exact applicability before committing
the release checklist. Treat these as announced future obligations, not rules
already effective on the review date. Preserve offer history now so January
compliance does not depend on reconstructing prices later.

### G06 — Privacy: existing obligations plus phased DPDP commencement, P0/P1

Do not describe the entire DPDP regime as already effective on 11 September 2026.
The November 2025 commencement notification phases provisions; many core
processing obligations commence eighteen months after publication, during May
2027. Some institutional provisions commenced earlier. Recheck for amendments
before launch.
[DPDP commencement notification](https://www.meity.gov.in/static/uploads/2025/11/c56ceae6c383460ca69577428d36828b.pdf).

The transition is not permission to neglect current privacy/security duties.
Assess applicable IT Act/SPDI duties, particularly where passwords or financial
information are handled.
[SPDI Rules, official Gazette](https://upload.indiacode.nic.in/showfile?actid=AC_CEN_45_76_00001_200021_1517807324077&filename=GSR313E_10511%281%29_0.pdf&type=rule).

Design now for purpose-specific notices, necessary collection, appropriate lawful
processing, withdrawal, correction/erasure requests, processors and a contact
person. The DPDP Act also defines children as under 18; subject to applicable
exceptions, the child-data regime involves parental consent and restrictions on
tracking/targeted advertising once operative. An adults-only beta is a scope
choice, not proof that a checkbox resolves age-assurance duties.
[DPDP Act, sections 4–9 and 11–14](https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf).

Rules 6–8 include safeguards, breach notifications and one-year retention
requirements for specified processing/log records when operative. Rule 7 includes
initial notice without delay and more detail within 72 hours, subject to its terms;
do not confuse that with CERT-In's separate clock. Do not promise instant deletion
of all backups and legally retained transaction records.
[DPDP Rules, commencement and rules 6–8](https://www.meity.gov.in/static/uploads/2025/11/53450e6e5dc0bfa85ebd78686cadad39.pdf).

**Required:** effective-date calendar, processor inventory, retention/legal-hold
matrix and deletion workflow before account rollout. Choose locations for each
data class based on actual rules/contracts, not an unsupported claim that all
Indian SaaS data must always be hosted in India.

### G07 — Cyber incident handling: current operational duties, P0

CERT-In's 28 April 2022 Directions cover specified entities including service
providers and bodies corporate. They require reporting listed cyber incidents
within six hours of noticing/being informed, designate a contact, prescribe clock
synchronization, and require secure ICT logs for a rolling 180 days within Indian
jurisdiction. The separate five-year subscriber-record requirement for listed
infrastructure providers must not automatically be applied to ordinary SaaS users.
[CERT-In Directions, clauses (i)–(v)](https://www.cert-in.org.in/PDF/CERT-In_Directions_70B_28.04.2022.pdf).

**Required:** applicable-incident runbook, security alerting and a backup human
contact. Waiting until the next working day is unsuitable for this obligation.
Specify India log retention with the hosting/observability vendors, access controls,
and reconciliation with later DPDP retention duties.

### G08 — Partnerships, execution and future expansion: separate review gates

SEBI's association rules restrict regulated entities/agents from associating with
persons engaged in specified unregistered advice/recommendation or prohibited
return-claim activities. A broker referral or educator arrangement is not an
automatic revenue channel; examine activities and consideration on both sides.
[SEBI association circular](https://www.sebi.gov.in/legal/circulars/jan-2025/details-clarifications-on-provisions-related-to-association-of-persons-regulated-by-the-board-miis-and-their-agents-with-persons-engaged-in-prohibited-activities_91356.html).

Broker execution, handling trading funds, paid signal distribution, native app
billing, public community content and sponsored rankings each change obligations
and operational risk. Keep them outside this beta and perform a dedicated review
before adding them. Taxes, invoicing and entity structure also need accountant/legal
assessment; this review does not assert a blanket GST threshold or registration
exemption. App-store and payment-provider policies are distinct from government law.

## 5. What can proceed now

Internal engineering, synthetic-data demos, factual data presentation, user-defined
filters, saved research and a calculator limited to arithmetic are reasonable
development directions. None is a categorical legal safe harbour: public scope,
data rights, presentation and actual outputs still determine the applicable duties.

No need to abandon the business or register for every possible financial activity.
Choose between a bounded software product and an appropriately regulated
research/advice offering based on the value customers need. That decision precedes
final scanner packaging and marketing.

## 6. Corrected sequence and release blockers

1. Resolve product scope and source rights for the first external exposure; review
   scanner examples and recent-price education/marketing use. Owner: founder with
   securities counsel and data provider. Record whether a changed regulated model
   is required, rather than treating review as automatic approval.
2. Stabilize calculation dates, corporate actions, baseline tests and immutable
   source batches. Owner: engineering. Exit: deterministic fixtures cover gaps,
   corrections, splits, missing news and historical-data availability.
3. Build identity, ownership, session security, retention and migration together.
   Owner: engineering. Exit: two-account abuse tests, delete/restore rehearsal,
   versioned terms and independent security review of the public surface.
4. Deliver one narrow research journey with explicit coverage, units, evidence,
   accessibility and error states. Owner: product/engineering. Exit: five observed
   user sessions including a no-match outcome and a failed network request.
5. Add replayable alerts and verified billing with bounded quotas and durable
   account mappings. Owner: engineering. Exit: worker-crash, expiry, refund,
   late-webhook, data-correction and backup-restore scenarios pass.
6. Prove reporting/support operations, retention costs, price economics and
   date-specific legal readiness before charging. Owner: founder plus relevant
   specialists. Expand only on measured usage, reliability and contribution.

The amended business plan carries the revised estimates, quotas, activation
definition, and additional launch gates. These are documentation changes;
the code findings in this review remain open until separately implemented and tested.
