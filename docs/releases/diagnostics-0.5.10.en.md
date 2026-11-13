# mindie-diagnostics 0.5.10: Windows paths and enterprise transport

[中文](diagnostics-0.5.10.md)

Fixes mindie-agent-codex#113 by accepting local Windows extended paths, encoding SQLite filenames correctly, and sharing reader/log/cursor identity between ordinary and extended spellings. The default retention budget increases from one to five seconds to cover SQLite's three-second wait and cleanup, avoiding a permanently unknown idle reader.

Generated Windows virtualenvs use `%LOCALAPPDATA%/mindie/diagnostics/<profile-hash>/<source-hash>/venv` to stay within native Python launcher limits. The policy-side `current.json` and lock remain the sole activation authority, with legacy generation lookup for rollback. Existing queues, cursors, consent revisions and uncertain publication receipts are retained. Verified live degraded workers can receive the fix; replacement generations still require healthy readback.

Background network capture preserves caller Git HTTP proxy, CA and TLS options while excluding credential helpers and authentication headers. Applying a new environment clears revoked settings. DNS/TCP/TLS failures retain their actual error meaning and do not invalidate accounts or bypass enterprise policy.

Tests exercise real long-path SQLite, virtualenv preparation, generation handoff and queue retention. CI adds Python 3.12. The formal workflow verifies platform contracts, exact-source benchmarks, release assets and real old-client upgrades. The affected user's TCP timeout still requires verification on that network; this release does not change its firewall or proxy policy.
