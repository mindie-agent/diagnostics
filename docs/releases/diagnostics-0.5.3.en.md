# mindie-diagnostics 0.5.3

Updater failures recorded with `category="update"` now share one classification policy based on typed facts. Missing modules and broken artifacts or protocols can enter the consented reporting queue. Transport, credentials, permissions and local prerequisites remain visible locally. Classification never reads exception messages, commands or raw output.

Read-only recent failures include `error_code` and `reporting_eligibility`, distinguishing local-only records, missing consent at capture, another profile and current consent. `consent_allowed` describes eligibility; delivery is shown by the existing queue state. The bounded view reads at most eight 16 KiB tails and returns five failures, with `truncated` marking an incomplete window.

Existing deduplication, consent withdrawal and uncertain-write reconciliation remain authoritative. Regression tests exercise an actual isolated Python helper failure, sanitized delivery, aggregation and consent boundaries without publishing test issues. Plugins must also instrument their updater failures; upgrading diagnostics cannot reconstruct events an older plugin never recorded.
