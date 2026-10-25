# mindie-diagnostics 0.5.4

This release fixes local failure records being rejected by reader-metadata coordination on a cold start or when a free segment slot is available. A writer now claims a never-existing file with `O_EXCL` inside the fixed `BACKUP_COUNT + 1` window. A new file has no unread bytes, so this path does not touch the reader registry and does not open SQLite. Only recycling an existing segment owned by this writer enters reader coordination, and it still requires every registered reader to have consumed the segment, respects backpressure, uses atomic replacement and reports `logging_failed` truthfully.

The recycling path keeps the existing 100ms cooperative work cutoff: at most 50ms waiting for the registry lock, with the remaining request handed to `cursor_snapshot`. That cutoff is a best-effort admission boundary, not a hard latency ceiling, and it does not change `cursor_snapshot`'s existing 3.0s SQLite wait floor; a real recycle attempt can still exceed 100ms. This is a known recycle-budget boundary, not something this fix closes or advertises as solved. A timeout or unknown cursor fails closed and drops the record instead of recycling data that is not confirmed consumed.

This release introduces no asynchronous collection, resident daemon or IPC. A short-lived Hook still writes local evidence synchronously, and the existing opt-in reporter still scans only logs already on disk. Asynchronous or queued collection needs a separate deliberate design; a daemon thread that cannot be guaranteed to finish before exit cannot replace the local write contract.

Regression tests use the real filesystem, reader registry, Outbox/ingestion cursor schema and real cross-process locks to cover: a 101ms cold-start delay still admits the first write; a free rotation slot is used while the reader-registry lock is held; unread cursors still protect a full window; and real backpressure still forbids growth. The reader protections, atomic replacement and status semantics from 0.5.3 are unchanged.

The package `__version__` is synchronized to `0.5.4`.
