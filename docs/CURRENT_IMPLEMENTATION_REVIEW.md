# Current implementation review — 2026-09-11

## Remediation update

Validation after fixes: **519 tests passed** in 27.89 seconds. Python compilation
of the changed application modules and `git diff --check` passed. The sole test
warning is joblib falling back to logical-core detection. Changes remain local;
no commit or push was performed.

The nine findings below have been addressed locally following the user's fix request:

- Exchange-aware deal primary keys now apply to new databases and transactional migrations of both legacy schemas. Existing rows are preserved and repeated ingestion updates the matching exchange only.
- Signal backtests no longer calculate or present portfolio Sharpe/drawdown. Their compatibility fields return `None` with an explanation. This removes both the initial-loss error and the unsupported compounding of overlapping forward returns; a daily portfolio simulator remains future work.
- ROC-AUC uses average ranks for ties; average precision groups tied thresholds. The UI labels average precision accurately and states the threshold-filtered evaluation scope.
- Constant-baseline anomaly detection distinguishes floating-point noise from material departures, using the existing saturated score limit for departures.
- Home renders actual NIFTY 50, NIFTY BANK and INDIA VIX values from the macro response. Missing metrics display a dash; measured zeroes remain zero.
- Research labels returns and relative volume in stored observations, exposes each window's start date, and includes return dates in heatmap hovers and exports. No exchange-calendar completeness is claimed. Boolean rule thresholds are also rejected.
- The news test uses recent/expired timestamps relative to UTC and verifies both outcomes.

Regression coverage lives in `tests/test_review_regressions.py` and the updated news test. Existing research UI, anomaly, ingestion and institutional tests also exercise the changes. Previously overwritten exchange data cannot be reconstructed by a schema migration and would require re-ingestion. No live exchange fetch or production database migration was run during this fix.

## Original review (historical)

The findings and failing-test results below describe the pre-fix implementation. No commit or push was performed in the review.

## Verification

- `.venv/bin/python -m pytest -o addopts= -q`: **504 passed, 2 failed**, 27.46 seconds.
- `git diff --check`: passed before this report was added.
- Isolated Python reproductions confirmed exchange overwrite, tied-score metric errors, and incorrect initial drawdown. These used synthetic inputs and a temporary SQLite database, not the user's database.
- Reviewed changed ingestion, storage, analytics, research, and UI paths. This was source inspection and local execution, not a browser interaction audit, live exchange integration test, security audit, or legal clearance.

## Findings in priority order

### 1. High: cross-exchange deals silently overwrite each other

Locations: `swingdesk/storage.py:204`, `swingdesk/storage.py:339`, `swingdesk/storage.py:951`.

The deal primary key omits `exchange`, while BSE ingestion can map a security to its existing `.NS` ticker. Inserting otherwise identical NSE and BSE disclosures with `INSERT OR REPLACE` leaves only the BSE row. The migration adds the exchange column but does not rebuild the primary key. `existing_deal_keys()` misleadingly describes a key that includes exchange.

Reproduction: insert two rows with ticker `TEST.NS`, date `2026-09-01`, client `CLIENT`, type `bulk`, side `BUY`, quantity 100, and different exchanges. `load_deals()` returns one row with exchange BSE.

Required fix: migrate both existing and fresh databases to an exchange-aware identity; verify cross-exchange coexistence, same-exchange idempotency, and migration preservation. Previously overwritten records require re-ingestion.

### 2. High: drawdown excludes the initial capital

Location: `swingdesk/analyze/sudden_move.py:403`.

The equity curve starts after the first return and its running maximum omits starting equity. A single selected trade returning -10% produces `max_drawdown_top_5_pct = 0.0`, understating risk.

Required fix: include starting capital in the running peak and test initial losses, consecutive losses, and recovery.

### 3. High: tied scores corrupt ranking metrics

Locations: `swingdesk/analyze/sudden_move.py:334`, `swingdesk/analyze/sudden_move.py:348`.

ROC-AUC assigns distinct ranks to identical scores. For scores `[1, 1]`, labels `[0, 1]` yield AUC 1.0; reversing labels yields 0.0. Both should be 0.5. Average precision similarly changes between 0.5 and 1.0; a grouped-threshold calculation should give 0.5 for this example. Rounded scanner scores make ties a normal input.

Required fix: use tie-aware ROC-AUC and grouped-threshold average precision, with permutation-invariance tests. Label average precision accurately rather than implying a different PR-area convention.

### 4. High: multi-session returns are treated as daily portfolio returns

Locations: `swingdesk/analyze/sudden_move.py:397`, `swingdesk/analyze/sudden_move.py:439`.

The backtest accepts a configurable holding horizon, but compounds each signal date's full forward return and annualizes its standard deviation with sqrt(252). For horizons greater than one, positions overlap; there is no capital allocation or daily mark-to-market series. Dates without qualifying signals are also absent. The displayed Sharpe and drawdown therefore are not reliable portfolio statistics.

Required fix: construct an explicit daily portfolio with position overlap and idle cash, or withhold portfolio Sharpe/drawdown and report clearly scoped signal outcome statistics. Also disclose that recall/AUC currently cover candidates passing `min_score`, not the complete universe.

### 5. Medium: anomaly detector misses changes from constant history

Location: `swingdesk/analyze/sudden_move.py:83`.

When historical MAD and standard deviation are both near zero, `_robust_z` returns zero even for a large new deviation. The existing `test_score_frame_surfaces_anomaly_spike` fails because `anomaly_range_z` is 0.0 where the test expects greater than one.

Required fix: define a defensible zero-variance fallback that distinguishes unchanged values from genuine departures, and keep the existing spike test meaningful.

### 6. Medium: home market cards request nonexistent fields

Locations: `swingdesk/app.py:639`, `swingdesk/ingest/macro.py:113`.

Home requests top-level `regime`, `breadth_label`, and `risk_tone`. `market_pulse()` returns a dictionary keyed by instrument name containing close, changes, and category. Consequently, populated macro data still yields dashes for all three cards.

Required fix: connect these cards to a defined analytics response or render the actual instrument metrics. Verify the populated-data UI state.

### 7. Medium: unavailable backtest metrics appear as valid zeroes

Locations: `swingdesk/app.py:2753`, `swingdesk/app.py:2816`.

The UI uses `summary.get(...) or 0`. Empty backtests, one-class AUC, or undefined Sharpe become numeric zeroes, making unavailable results appear measured.

Required fix: preserve `None` as unavailable and show the reason; retain legitimate numeric zeroes.

### 8. Medium: missing price sessions change the meaning of research returns

Location: `swingdesk/analyze/research.py:97`.

Return windows count stored rows without establishing session completeness. Two closes of 100 on September 1 and 110 on September 11 become a 10% "1-session return". Freshness of the latest price does not establish completeness of the preceding window. This can affect screen and alert matches.

Required fix: check expected sessions using an exchange calendar and expose incomplete windows, or explicitly label these values as returns across stored observations with their actual dates. Do not present missing-session windows as known one-session returns.

### 9. Medium: news test depends on wall-clock date

Location: `tests/test_global_impact.py:47`.

`test_global_news_storage_roundtrip` inserts July 7, 2026 news and then requires a positive score within a trailing 30-day window. On September 11 it correctly falls outside that window, so the assertion fails. This is a test fixture defect, not evidence that the expiry behavior is wrong.

Required fix: freeze the analysis clock or construct the fixture relative to a controlled time, and separately test expired news.

## Remaining release scope

Keep the production and regulatory release gates in `PLAN_RISK_REVIEW.md`; passing local tests does not clear those gates. UI coverage here includes populated and missing-data states through Streamlit AppTest, not a full browser or live-network audit.
