# Bounding the Outbound Reconciliation Read (Design Note, Not Implemented)

[中文](./outbound-reconciliation-followup.md) · **English** · [Back to Documentation](./README.en.md)

This note records intent and constraints only. **No production transport or API was changed**: the request shape in `reporter.py`, the `Outbox` schema and its state machine are untouched. Line references are against `838480a`.

---

## 1. The reproducible contradiction today

`GitHub.find_issue` ([`reporter.py:470-499`](../mindie_diagnostics/reporter.py)) uses a single "list everything and download every issue body" read for two different purposes: cross-session deduplication, and reconciling this worker's own uncertain POST. It pages up to 20 times at 100 issues per page (`reporter.py:483-484`), while one whole publication cycle (one reconciliation plus one POST) shares an 8 MiB byte budget (`reporter.py:459-460`; `self._bytes` is reset only in `begin_cycle`).

Meanwhile the issues this component itself publishes are capped at 60000 bytes of body (`reporter.py:418-419`). Feeding a local fake `gh` script (a real subprocess through the real `_parse_response`, no network) a listing of 250 issues with 52,698-byte bodies: page 1 reads 6,074,551 bytes and page 2 raises `github_output_limit` because the remaining budget is too small.

That is **local fixture evidence about budget arithmetic only**. It is not a retrieval-quality, NPU, or live-acceptance result.

The consequence follows from the source: `github_output_limit` is not in `publish_one`'s `recoverable` set and is not `permanent` (`reporter.py:576-579`), so it is recorded as retryable and replayed under backoff — yet a repository only grows, so the reconciliation window never actually recovers. `pending` items never reach their POST and `uncertain` items never finish reconciling, while the surface reports a transient fault.

## 2. Invariants any solution must keep

1. **An uncertain POST is never replayed**: the `begin_post` lease fence (`outbox.py:428-434`) accepts only `pending`/`retry`, so an `uncertain` row can only take the read-only reconciliation path (`reporter.py:545-549`).
2. **"Could not read" is never "does not exist"**: an incomplete window, an exhausted byte budget and search-index lag must all stay visible failures, never collapse into "the remote has no such issue".
3. **A local receipt failure never overrides a confirmed remote result** (`_record_publication_outcome`, `reporter.py:511-519`).
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
- when pages or bytes run out, use the existing `github_reconciliation_window_exhausted` (`reporter.py:495`, today nearly unreachable because the byte budget is spent first) as distinct from the transport-level `github_output_limit`, so `reporting status` can tell "window too small" from "network fault";
- while the window is incomplete, stay `uncertain`/`retry` and never POST on that basis.

## 7. What this design explicitly does not claim

- **No cross-profile or cross-host exactly-once**: a stable marker only guarantees "no duplicate once a reader can already see that marker". It cannot rule out two profiles being mid-POST simultaneously.
- **No change to `publications` / rate-limit schema**: `publish_one` does not pass `hourly_limit` today (`reporter.py:552`), so the table stays empty; that is recorded here, but no table or branch is removed before a real defect and a validated migration exist.
- **No merging of the health evaluators**: the duplicated judgement in `health.read_health` and `integration._worker_view` is the same kind of item, left to a separate change with migration validation.

## 8. Adjacent item: aligning the `prune` and `cursor_snapshot` budgets

`maintenance.prune`'s 1.0s deadline is inconsistent with the 3.0s minimum database wait kept by `reader_registry.cursor_snapshot` (see the corrected notes in both module headers). With no demonstrable real lock-wait path, this pass only corrected the documented promise. A real alignment should be **admission based on the remaining budget** (when too little time remains to host one lawful 3.0s wait, skip and record cursors as unknown, which conservatively deletes nothing) rather than simply raising the outer deadline or pushing SQLite waits below a second and reintroducing hot-path false kills.
