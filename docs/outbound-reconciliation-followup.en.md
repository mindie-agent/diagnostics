# Bounding the Outbound Reconciliation Read

[中文](./outbound-reconciliation-followup.md) · **English** · [Back to Documentation](./README.en.md)

**Implementation status (October 29, 2026):** attempt timestamps, positive-only search with bounded listing, window-exhaustion classification, and prune budget admission are implemented. Sections 1–7 retain the original design rationale; their line references describe `838480a`, not current source locations. Sections 8–9 describe the implementation and local verification scope.

---

## 1. The reproducible contradiction before implementation

`GitHub.find_issue` ([`reporter.py:468-497`](../mindie_diagnostics/reporter.py)) uses a single "list everything and download every issue body" read for two different purposes: cross-session deduplication, and reconciling this worker's own uncertain POST. It pages up to 20 times at 100 issues per page (`reporter.py:481-482`), while one whole publication cycle (one reconciliation plus one POST) shares an 8 MiB byte budget (`reporter.py:457-458`; `self._bytes` is reset only in `begin_cycle`).

Meanwhile the issues this component itself publishes are capped at 60000 bytes of body (`reporter.py:416-417`). Feeding a local fake `gh` script (a real subprocess through the real `_parse_response`, no network) a listing of 250 issues with 52,698-byte bodies: page 1 reads 6,074,551 bytes and page 2 raises `github_output_limit` because the remaining budget is too small.

That is **local fixture evidence about budget arithmetic only**. It is not a retrieval-quality, NPU, or live-acceptance result.

The consequence follows from the source: `github_output_limit` is not in `publish_one`'s `recoverable` set and is not `permanent` (`reporter.py:574-577`), so it is recorded as retryable and replayed under backoff — yet a repository only grows, so the reconciliation window never actually recovers. `pending` items never reach their POST and `uncertain` items never finish reconciling, while the surface reports a transient fault.

## 2. Invariants any solution must keep

1. **An uncertain POST is never replayed**: the `begin_post` lease fence (`outbox.py:428-434`) accepts only `pending`/`retry`, so an `uncertain` row can only take the read-only reconciliation path (`reporter.py:543-547`).
2. **"Could not read" is never "does not exist"**: an incomplete window, an exhausted byte budget and search-index lag must all stay visible failures, never collapse into "the remote has no such issue".
3. **A local receipt failure never overrides a confirmed remote result** (`_record_publication_outcome`, `reporter.py:509-517`).
4. **Public identity stays the `<!-- mindie-incident:<sha256> -->` marker in the issue body**, independent of the read shape.

## 3. Why "just use the search endpoint" is not sufficient on its own

`GET search/issues` does return a payload independent of repository size, but GitHub's search index is **eventually consistent**: a freshly created issue can be missing from it for a while. Therefore:

- search may only be used to **confirm presence** (a hit can safely drive `reconciled`);
- a search **miss must never count as proof of absence**, and in particular must never let a `pending` row POST — that is exactly how duplicate issues appear.

So search can cut the byte cost of deduplication, but only paired with a time-bounded, lag-free listing read, and both must report an inconclusive window honestly.

## 4. `first_seen` is not the POST instant

`first_seen`, as returned by `Outbox.claim` (`outbox.py:307-329`), is when the fault was **first seen locally**, not when a POST was attempted:

- it is genuinely `<= ` the real POST instant, so `[first_seen, now]` is a **correct** lower-bound window (it cannot miss an issue this worker created);
- but its **width is unbounded**: a fault first seen months ago that only now reached the head of the queue degrades the window into a full scan;
- the revival branch resets `first_seen` to the current time (`outbox.py:196-201`). That branch does exclude rows carrying `submission_uncertain*` (so rows holding unknown-write evidence are never reset), which only underlines that the field's owner-meaning is "first local sighting", not "external intent".

Conclusion: `first_seen` may be used as a **correctness lower bound** only; it must not be presented as yielding a small window.

## 5. A bounded window needs single-owner intent metadata

No column in `incidents` currently records a stable "this process attempted a POST at T": `lease_until` is zeroed by any state write (`outbox.py:409`) and `next_attempt` is rewritten on every reconciliation cycle, so neither can reconstruct the original attempt instant.

The minimal shape is **one nullable attempt-instant column on the existing `incidents` row**, written by `begin_post` inside the **same `BEGIN IMMEDIATE` transaction** that sets `state='uncertain'` (single owner, living and dying with the uncertain state), migrated additively in the style of the current schema validation (`PRAGMA table_info` plus `ALTER TABLE ... ADD COLUMN`, falling back to the `first_seen` lower bound when an older database reads `NULL`).

Do **not** add a second receipt/audit ledger or a second state machine for this: the input to the reconciliation window must come from the same row that owns the state.

## 6. Bounded reads and honest window reporting

- Each purpose declares its own page, byte and per-body limits, all charged to the same `begin_cycle` budget;
- when pages or bytes run out, use the existing `github_reconciliation_window_exhausted` (`reporter.py:493`, today nearly unreachable because the byte budget is spent first) as distinct from the transport-level `github_output_limit`, so `reporting status` can tell "window too small" from "network fault";
- while the window is incomplete, stay `uncertain`/`retry` and never POST on that basis.

## 7. What this design explicitly does not claim

- **No cross-profile or cross-host exactly-once**: a stable marker only guarantees "no duplicate once a reader can already see that marker". It cannot rule out two profiles being mid-POST simultaneously.
- **No change to `publications` / rate-limit schema**: `publish_one` does not pass `hourly_limit` today (`reporter.py:550`), so the table stays empty; that is recorded here, but no table or branch is removed before a real defect and a validated migration exist.
- **No merging of the health evaluators**: the duplicated judgement in `health.read_health` and `integration._worker_view` is the same kind of item, left to a separate change with migration validation.

## 8. Implemented: aligning the `prune` and `cursor_snapshot` budgets

`maintenance.prune(timeout=1.0)` keeps its default deadline and reserves 50ms for registry release and the retention marker. It passes the actual remaining time to `cursor_snapshot(..., remaining_budget=...)`. Each reader requires at least `3.0s + 50ms` before opening SQLite; otherwise the snapshot returns `None`. Schema inspection and cursor selection share one read transaction to avoid separate database lock waits.

With the default budget and a registered reader, prune returns `reader_unknown=True` and `limited=True` without deleting segments. No reader means no SQLite wait is needed. Callers may explicitly provide a larger `timeout`; `reporting_commands.maintain` passes its own remaining budget minus the settlement reserve so explicit maintenance can still reclaim consumed logs. SQLite retains `busy_timeout=3000`, and prune's default deadline is unchanged.

The existing log recycling caller passes only `timeout` and retains its cooperative execution deadline. This change does not claim to close every latency boundary in that hot path.

## 9. Implemented: outbound window and verification scope

- `outbox.py` (this repository has no `storage.py`) adds nullable `attempt_timestamp REAL` to `incidents`. `begin_post` writes the timestamp and `uncertain` in the same `BEGIN IMMEDIATE` transaction. A legacy uncertain row receives its `first_seen` lower bound inside its first claim transaction; later reconciliation never advances it.
- Public identity remains `<!-- mindie-incident:<sha256> -->`. Search is restricted by `repo:`, `is:issue`, `in:body`, and the fingerprint, with at most five results on one page. Only a complete marker on a non-PR result confirms presence; a miss or `incomplete_results` does not establish absence.
- Listing uses `state=all`, creation order descending, 30 items per page and at most five pages. Its lower bound is the attempt timestamp (or `first_seen` before an attempt), minus 300 seconds of clock skew, rounded down to a second. `since` moves one more second backward to include boundary timestamps. Because `since` filters update time, listing stops when the page's final creation time precedes the lower bound; a short page also completes the window.
- Search, listing and POST share the default 8 MiB response budget of one `begin_cycle`; individual read bodies are capped at 256 KiB. `max_response_bytes` allows a smaller budget. Reconciliation byte, body or page exhaustion yields `github_reconciliation_window_exhausted`, backs off at least 3600 seconds, retains `retry`/`uncertain`, and never authorizes POST. Other transport calls retain `github_output_limit`.
- Verification uses pure in-memory marker matching, real SQLite migration/rollback/leases, and a local HTTP server reached by a real gh substitute subprocess through the production byte counter. It covers positive search, delayed indexing, timestamp boundaries, cumulative bytes, exhausted pages, unknown POST non-replay and prune admission. Tests are in `tests/test_reporter.py`, `tests/test_outbox.py`, `tests/test_reconciliation_http.py`, `tests/test_reader_registry.py` and `tests/test_maintenance.py`.

This is local Tier 0/Tier 1 evidence, not live GitHub retrieval-quality, NPU or cross-host exactly-once acceptance.
