# Competitor feature delivery

Updated: 2026-09-11

## Implemented local research workflows

Open **Discover → Research workspace**. Existing scanners, chart indicators,
backtests, portfolio tools, and news pages remain available in their current homes.

| Capability | Delivered behavior | Limits |
|---|---|---|
| Named watchlists | Create, update, delete, switch research universe, explicitly activate a saved list as the main watchlist | Local persistence; no cloud sync. Lists do not automatically fetch new symbols |
| Saved screens | AND conditions on 11 price, volume, valuation, growth, and quality metrics; persist rules and ticker universe; select sort direction | Fixed supported metric set; no free-form formula execution or public sharing |
| Export | Download displayed screen results as CSV | No company financial-statement workbook |
| Stock comparison | Up to six symbols, fundamental comparison table, rebased performance chart across 21/63/126/252 stored observations | Uses common dates without forward filling; not total return including dividends |
| Heatmap | Equal-size tiles grouped by sector and coloured by returns across 1/5/21 stored observations, with actual window dates | Selected universe only, not an exchange-wide or market-cap-weighted map |
| Saved alert conditions | Named rules for daily close, stored-observation return, or relative volume; manual evaluation across saved symbols | No continuous service, push delivery, crossings, or event alerts; no automatic messages |

All research objects survive restarts in SQLite's `research_items` table. Saving the
same kind/name updates that object; deleting it removes only that saved object.
Watchlists and screens store their ticker membership at save time. Later list edits
do not silently change the universe of an existing saved screen or alert.

Price and fundamentals timestamps are shown. Active filters reject unknown values.
Relative volume compares the current stored observation against the prior 20 observations,
excluding the current observation. Alerts require available data and a price date no
more than four calendar days old; the threshold accommodates weekends but is not
an exchange-calendar freshness guarantee. Fundamental metrics are screenable but
are not supported as alert conditions in this version.

## Competitor references

Checked against official public pages on 2026-09-11. These describe capability
families, not an assertion of full parity, plan availability, or licensed access.

- [Tickertape screener guide](https://www.tickertape.in/blog/how-to-use-tickertape-stock-screener-to-discover-stocks-a-complete-guide/): user-controlled filtering and watchlist universes informed saved screening workflows.
- [TradingView features](https://www.tradingview.com/features/): chart research, watchlists, screening, and alerts informed the research-tool direction. SwingDesk does not embed TradingView or implement Pine Script.
- [Screener CSV export](https://support.screener.in/article/28-export-screen-results): exporting research results informed downloadable screen tables.
- [Screener getting started](https://www.screener.in/docs/guides/getting-started/): watchlists and screen alerts inform future announcement/event workflows.

The older Tickertape/smallcase/Univest/Liquide matrix remains a planning snapshot.
Its broad Yes/No labels have not all been independently reverified in this pass.

## Remaining work in order

1. Extend the saved-rule model to scanner and event metrics; add a background
   evaluator, deduplication, alert history, and delivery preferences before enabling
   continuous notifications.
2. Package existing portfolio risk analytics into an understandable health view;
   add account-aware imports and historical portfolio snapshots.
3. Add richer research templates, financial-statement histories, ownership changes,
   and event timelines with source attribution and appropriate data providers.
4. Expand chart layouts, indicator presets, and historical replay. Pine-compatible
   scripting and TradingView-scale charting are substantial separate projects.
5. Implement identity, per-user storage, API contracts, cloud sync, entitlements,
   and mobile clients according to the production architecture.

Broker execution, managed baskets, recommendation-led features, mutual funds,
and options terminals remain outside this delivery's scope.

## Implementation map

- `swingdesk/analyze/research.py`: persistence, validation, snapshots, filters,
  normalized comparison series, and manual alert evaluation.
- `swingdesk/research_ui.py`: lazily loaded research page.
- `swingdesk/storage.py`: additive, idempotent table creation.
- `tests/test_research.py`: persistence, metric units, incomplete data, date
  alignment, alert freshness, and Streamlit mode smoke tests.

## Verification

`python -m pytest tests/test_research.py -o addopts= -q`: 11 passed, including
Streamlit navigation and saving watchlists/alerts. Compilation and diff whitespace
checks passed.

The full suite also exposed two independent failures in existing tests:
`test_global_news_storage_roundtrip` (fixed July news date outside the current
30-day window) and `test_score_frame_surfaces_anomaly_spike` (range anomaly score
is zero). Both reproduce with the new research schema removed at runtime and
without importing either new research module. Their existing implementation and
fixtures were left intact.
