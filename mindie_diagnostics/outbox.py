"""Bounded SQLite publishing queue for diagnostics reporting.

[Role]
Manages the local SQLite incident queue (`reporter.sqlite3`), tracking active
(`pending`, `retry`, `uncertain`) and terminal (`published`, `withdrawn`,
`expired`, `exhausted`, `blocked`, `permanent-failed`) incident states,
operation deduplication (`seen`), and hourly publication rate limits.

[Invariants]
- Never acts as an execution or resource ownership ledger.
- Active capacity (`capacity`, default 1000) counts only `pending`, `retry`,
  and `uncertain` rows; unpublished active evidence is never evicted by terminal
  history pruning.
- Transient failures and uncertain writes retain evidence until publication or withdrawal.
- `begin_post` transitions leased rows to `uncertain` BEFORE any mutating GitHub
  POST so uncertain external writes are never blindly retried without
  reconciling the GitHub marker first.

[Timeouts & Locks]
- Cross-process initialization is serialized by an `flock`/`msvcrt` lock on
  `<db>.init.lock`.
- SQLite connections use `timeout=3.0`, `PRAGMA busy_timeout=3000`, and explicit
  `BEGIN IMMEDIATE` transactions for state mutations.

[Side-Effect Policy]
- Terminal rows are compacted into minimal receipts (`public_fingerprint`,
  `incident_ids`, `component`, `operation`, `issue_url`) and pruned at 1000 rows
  or 30 days.
"""
from __future__ import annotations

import errno
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from .fallback import _private_mkdir


TERMINAL_HISTORY_LIMIT = 1000
TERMINAL_HISTORY_SECONDS = 30 * 86400
_ACTIVE_STATES = ("pending", "retry", "uncertain")
_TERMINAL_STATES = ("published", "withdrawn", "expired", "exhausted", "blocked", "permanent-failed")
_RECEIPT_TOKEN_LIMIT = 128


_SCHEMA = """
                CREATE TABLE incidents (
                    fingerprint TEXT PRIMARY KEY, payload TEXT NOT NULL,
                    first_seen REAL NOT NULL, last_seen REAL NOT NULL,
                    occurrences INTEGER NOT NULL DEFAULT 1,
                    state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt REAL NOT NULL DEFAULT 0, lease_until REAL NOT NULL DEFAULT 0,
                    lease_token TEXT, issue_number INTEGER, issue_url TEXT, last_error TEXT,
                    consent TEXT);
                CREATE TABLE seen (
                    operation_id TEXT PRIMARY KEY, observed REAL NOT NULL);
                CREATE TABLE publications (at REAL NOT NULL);
            """


class QueueFull(RuntimeError):
    pass


@contextmanager
def _initialization_lock(path):
    lock = path.with_name(path.name + '.init.lock')
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        if os.fstat(fd).st_size == 0:
            os.write(fd, b'0')
        if os.name == 'nt':
            import msvcrt
            while True:
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    time.sleep(.01)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class Outbox:
    def __init__(self, path: str | Path, *, capacity: int = 1000, clock=time.time):
        if type(capacity) is not int or not 1 <= capacity <= 1000:
            raise ValueError("diagnostic queue capacity must be between 1 and 1000")
        self.path = Path(path).absolute()
        self._prepare_storage()
        self.clock = clock
        self.capacity = capacity
        with _initialization_lock(self.path):
            fresh = not self.path.exists()
            if fresh:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
                os.close(descriptor)
            elif self.path.stat().st_size == 0:
                raise ValueError('diagnostic queue schema is incomplete; prior outcomes cannot be reconstructed')
            with self.connect() as db:
                if fresh:
                    db.executescript(_SCHEMA)
                self._validate_schema(db)
                # Older releases stopped reconciliation after three reads.
                # Their compact receipt still identifies the remote marker.
                db.execute(
                    "UPDATE incidents SET state='uncertain',next_attempt=0,lease_until=0,lease_token=NULL "
                    "WHERE state IN ('exhausted','expired','permanent-failed') "
                    "AND last_error LIKE 'submission_uncertain%'"
                )
        if os.name != "nt" and self.path.exists():
            os.chmod(self.path, 0o600)

    def _validate_schema(self, db):
        required = {
            'incidents': {'fingerprint', 'payload', 'first_seen', 'last_seen', 'occurrences', 'state', 'attempts', 'next_attempt', 'lease_until', 'lease_token', 'issue_number', 'issue_url', 'last_error', 'consent'},
            'seen': {'operation_id', 'observed'},
            'publications': {'at'},
        }
        for table, cols in required.items():
            actual = {row[1] for row in db.execute(f'PRAGMA table_info({table})')}
            if not cols <= actual:
                raise ValueError('diagnostic queue schema is incomplete; prior outcomes cannot be reconstructed')

    def _prepare_storage(self) -> None:
        """Create missing parent directories and ensure queue path is a regular file."""
        path = self.path
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("diagnostic queue path must be a regular SQLite file")
        parent = path.parent
        if parent.is_symlink():
            raise ValueError("diagnostic queue directory must not be a symlink")
        _private_mkdir(parent, parents=True, exist_ok=True)
        if not parent.is_dir():
            raise NotADirectoryError("diagnostic queue parent is not a directory")

    @contextmanager
    def connect(self):
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise ValueError("diagnostic queue path must be a regular SQLite file")
        db = sqlite3.connect(self.path.as_uri() + '?mode=rw', uri=True, timeout=3.0)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA busy_timeout=3000")
        except BaseException:
            db.close()
            raise
        try:
            with db:
                yield db
        finally:
            db.close()

    def enqueue(self, fingerprint: str, operation_id: str, payload: dict[str, Any], *, consent=None) -> bool:
        payload = dict(payload)
        if self._incident_id(operation_id):
            payload["incident_ids"] = [operation_id]
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        if len(body.encode()) > 48000:
            raise ValueError("diagnostic issue payload exceeds 48000 bytes")
        reference = json.dumps(consent, ensure_ascii=True, separators=(",", ":")) if consent is not None else None
        if reference is not None and len(reference.encode()) > 8192:
            raise ValueError("reporting reference exceeds limit")
        now = self.clock()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM seen WHERE observed < ?", (now - 30 * 86400,))
            db.execute("DELETE FROM publications WHERE at < ?", (now - 86400,))
            self._prune_published(db, now)
            if db.execute("SELECT 1 FROM seen WHERE operation_id=?", (operation_id,)).fetchone():
                return False
            existing = db.execute("SELECT payload,state,last_error FROM incidents WHERE fingerprint=?", (fingerprint,)).fetchone()
            if existing:
                if (existing["state"] in ("exhausted", "expired", "permanent-failed")
                        and not (existing["last_error"] or "").startswith("submission_uncertain")):
                    # A fresh authorized event restores evidence discarded by
                    # the old read-attempt budget. Never replay unknown writes.
                    db.execute(
                        "UPDATE incidents SET payload=?,state='pending',attempts=0,next_attempt=0,"
                        "first_seen=?,last_seen=?,occurrences=occurrences+1 WHERE fingerprint=?",
                        (body, now, now, fingerprint),
                    )
                    db.execute("INSERT INTO seen VALUES (?,?)", (operation_id, now))
                    return True
                # Same consent+fault does not reset attempts, state, or payload.
                kept = json.loads(existing["payload"])
                if isinstance(kept, dict) and self._incident_id(operation_id):
                    ids = kept.get("incident_ids", [])
                    ids = ids if isinstance(ids, list) else []
                    kept["incident_ids"] = list(dict.fromkeys([value for value in ids if self._incident_id(value)] + [operation_id]))[-80:]
                db.execute("UPDATE incidents SET last_seen=?, occurrences=occurrences+1,payload=? WHERE fingerprint=?", (now, json.dumps(kept, separators=(",", ":")), fingerprint))
            else:
                if db.execute(
                    "SELECT COUNT(*) FROM incidents WHERE state IN ('pending','retry','uncertain')"
                ).fetchone()[0] >= self.capacity:
                    raise QueueFull("diagnostic outbox is full; retained incidents were not discarded")
                db.execute("INSERT INTO incidents (fingerprint,payload,first_seen,last_seen,consent) VALUES (?,?,?,?,?)", (fingerprint, body, now, now, reference))
            db.execute("INSERT INTO seen VALUES (?,?)", (operation_id, now))
            db.execute("DELETE FROM seen WHERE operation_id IN "
                       "(SELECT operation_id FROM seen ORDER BY observed DESC, operation_id DESC LIMIT -1 OFFSET ?)",
                       (self.capacity * 20,))
        return True

    def maintain(self) -> None:
        """Compact completed history without discarding unpublished evidence."""
        now = self.clock()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._prune_published(db, now)

    @staticmethod
    def _incident_id(value):
        return isinstance(value, str) and len(value) == 32 and all(c in "0123456789abcdef" for c in value)

    @staticmethod
    def _safe_token(value: Any) -> str | None:
        if not isinstance(value, str) or not value or len(value) > _RECEIPT_TOKEN_LIMIT:
            return None
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            return None
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:/-")
        if any(char not in allowed for char in value):
            return None
        return value

    @classmethod
    def _receipt(cls, fingerprint: str, payload: str, issue_url: str | None) -> str:
        component = operation = None
        try:
            data = json.loads(payload) if isinstance(payload, str) else None
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict):
            events = data.get("events")
            event = next((row for row in reversed(events[-80:]) if isinstance(row, dict) and row.get("status") == "error"), {}) if isinstance(events, list) else data
            component = cls._safe_token(event.get("component"))
            operation = cls._safe_token(event.get("operation"))
        body: dict[str, Any] = {}
        identifiers = data.get("incident_ids", []) if isinstance(data, dict) else []
        identifiers = identifiers if isinstance(identifiers, list) else []
        if isinstance(data, dict) and isinstance(data.get("events"), list):
            identifiers = [row.get("operation_id") for row in data["events"][-80:] if isinstance(row, dict)] + identifiers
        if isinstance(identifiers, list):
            safe_ids = list(dict.fromkeys(value for value in identifiers[-80:] if cls._incident_id(value)))
            if safe_ids:
                body["incident_ids"] = safe_ids
        if component is not None:
            body["component"] = component
        if operation is not None:
            body["operation"] = operation
        public = data.get("public_fingerprint") if isinstance(data, dict) else None
        if not isinstance(public, str) or len(public) != 64 or any(c not in "0123456789abcdef" for c in public):
            from .reporter import fingerprint as public_fingerprint
            public = public_fingerprint(data or {})
        body["public_fingerprint"] = public
        if isinstance(issue_url, str) and issue_url and len(issue_url) <= 512 and "\n" not in issue_url and "\r" not in issue_url:
            body["issue_url"] = issue_url
        return json.dumps(body, ensure_ascii=True, separators=(",", ":"), sort_keys=True)

    def _compact_terminal(self, db) -> None:
        rows = db.execute(
            "SELECT fingerprint, payload, issue_url FROM incidents WHERE state IN "
            "('published','withdrawn','expired','exhausted','blocked','permanent-failed') "
            "AND (payload LIKE '%\"events\":%' OR payload NOT LIKE '%\"public_fingerprint\":%')"
        ).fetchall()
        for row in rows:
            receipt = self._receipt(row["fingerprint"], row["payload"], row["issue_url"])
            if receipt != row["payload"]:
                db.execute("UPDATE incidents SET payload=? WHERE fingerprint=?", (receipt, row["fingerprint"]))

    def _prune_published(self, db, now):
        # Terminal markers only. Active unpublished evidence is never evicted.
        self._compact_terminal(db)
        db.execute(
            "DELETE FROM incidents WHERE state IN "
            "('published','withdrawn','expired','exhausted','blocked','permanent-failed') AND last_seen < ?",
            (now - TERMINAL_HISTORY_SECONDS,),
        )
        db.execute(
            "DELETE FROM incidents WHERE fingerprint IN "
            "(SELECT fingerprint FROM incidents WHERE state IN "
            "('published','withdrawn','expired','exhausted','blocked','permanent-failed') "
            "ORDER BY last_seen DESC, fingerprint DESC LIMIT -1 OFFSET ?)",
            (TERMINAL_HISTORY_LIMIT,),
        )

    def claim(self, *, lease_seconds: float = 300) -> dict[str, Any] | None:
        now, token = self.clock(), uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._prune_published(db, now)
            row = db.execute(
                """SELECT * FROM incidents WHERE state IN ('pending','retry','uncertain')
                AND next_attempt <= ? AND lease_until <= ?
                ORDER BY first_seen LIMIT 1""",
                (now, now),
            ).fetchone()
            if row is None:
                return None
            # Validate persisted authority before spending a cycle or leasing
            # the row. Broken saved consent is not an unscoped old event.
            consent = self._stored_consent(row["consent"])
            payload = json.loads(row["payload"])
            attempts = int(row["attempts"]) + 1
            db.execute(
                "UPDATE incidents SET attempts=?, lease_until=?, lease_token=? WHERE fingerprint=?",
                (attempts, now + lease_seconds, token, row["fingerprint"]),
            )
        return {**dict(row), "attempts": attempts, "lease_token": token, "payload": payload,
                "consent": consent}

    @staticmethod
    def _stored_consent(raw):
        if raw is None or raw == "":
            return None
        from . import fallback as f
        from .reporting import ConsentUnavailable
        try:
            reference = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise ConsentUnavailable() from exc
        if (not f.scope_key("saved-reference-validation", reference)
                or f._as_local_absolute(reference.get("config_file")) is None):
            raise ConsentUnavailable()
        return reference

    @staticmethod
    def _consent(raw):
        try:
            return json.loads(raw) if raw else None
        except (ValueError, TypeError):
            return None

    def withdraw_unconsented(self) -> int:
        """Revoke queued work before intake; retain bounded local history.

        This includes entries with no explicit scope. Leases are fenced so an
        overlapping worker cannot publish a revoked item.
        """
        from .reporting import consent_status, ConsentUnavailable
        with self.connect() as db:
            rows = db.execute(
                "SELECT fingerprint,consent,state FROM incidents WHERE state NOT IN ('withdrawn','published')"
            ).fetchall()
        checked, withdrawn = {}, 0
        for row in rows:
            raw = row["consent"]
            if raw not in checked:
                checked[raw] = consent_status(self._stored_consent(raw))
            if checked[raw] == "allowed":
                continue
            if checked[raw] == "unavailable":
                # A damaged/missing policy is not a user revocation. Preserve
                # pending/unknown evidence and its processing budget.
                raise ConsentUnavailable()
            with self.connect() as db:
                changed = db.execute(
                    "UPDATE incidents SET state='withdrawn',"
                    "last_error=CASE WHEN state='uncertain' OR last_error LIKE 'submission_uncertain%' "
                    "THEN 'submission_uncertain_reporting_consent_withdrawn' "
                    "ELSE 'reporting_consent_unavailable_or_withdrawn' END,lease_until=0,lease_token=NULL "
                    "WHERE fingerprint=? AND state NOT IN ('withdrawn','published')",
                    (row["fingerprint"],),
                )
                withdrawn += changed.rowcount
                if changed.rowcount:
                    self._compact_one(db, row["fingerprint"])
        with self.connect() as db:
            self._prune_published(db, self.clock())
        return withdrawn

    def _compact_one(self, db, fingerprint: str) -> None:
        row = db.execute(
            "SELECT fingerprint, payload, issue_url, state FROM incidents WHERE fingerprint=?",
            (fingerprint,),
        ).fetchone()
        if row is None or row["state"] not in _TERMINAL_STATES:
            return
        receipt = self._receipt(row["fingerprint"], row["payload"], row["issue_url"])
        if receipt != row["payload"]:
            db.execute("UPDATE incidents SET payload=? WHERE fingerprint=?", (receipt, fingerprint))

    def update(self, item: dict[str, Any], **fields: Any) -> None:
        allowed = {"state", "next_attempt", "issue_number", "issue_url", "last_error"}
        if not fields or set(fields) - allowed:
            raise ValueError("invalid outbox update")
        if "state" in fields and fields["state"] not in _ACTIVE_STATES + _TERMINAL_STATES:
            raise ValueError("invalid outbox state")
        with self.connect() as db:
            cursor = db.execute("UPDATE incidents SET " + ",".join(f"{key}=?" for key in fields) + ",lease_until=0,lease_token=NULL WHERE fingerprint=? AND lease_token=?", (*fields.values(), item["fingerprint"], item["lease_token"]))
            if cursor.rowcount != 1:
                raise RuntimeError("diagnostic worker lease lost")
            if fields.get("state") in _TERMINAL_STATES:
                self._compact_one(db, item["fingerprint"])
            self._prune_published(db, self.clock())

    def begin_post(self, item: dict[str, Any], *, hourly_limit: int | None = None) -> bool:
        """Persist uncertainty BEFORE a mutating request, including process death.

        The claim count is diagnostic, never a permanent delivery cutoff.
        """
        now = self.clock()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if hourly_limit is not None:
                count = db.execute("SELECT COUNT(*) FROM publications WHERE at >= ?", (now - 3600,)).fetchone()[0]
                if count >= hourly_limit:
                    return False
            cursor = db.execute(
                "UPDATE incidents SET state='uncertain' WHERE fingerprint=? AND lease_token=? "
                "AND lease_until>? AND state IN ('pending','retry')",
                (item["fingerprint"], item["lease_token"], now),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("diagnostic worker lease expired before publication")
            if hourly_limit is not None:
                db.execute("INSERT INTO publications VALUES (?)", (now,))
        return True

    def rows(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = [dict(row) for row in db.execute("SELECT fingerprint,first_seen,last_seen,occurrences,state,attempts,next_attempt,issue_number,issue_url,last_error,json_extract(payload,'$.incident_ids') AS incident_ids FROM incidents ORDER BY (state='published'),last_seen DESC LIMIT ?", (self.capacity * 2,))]
        for row in rows:
            value = self._consent(row.get("incident_ids"))
            row["incident_ids"] = [ident for ident in value[-80:] if self._incident_id(ident)] if isinstance(value, list) else []
        return rows
