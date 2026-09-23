"""Bounded publishing queue. Never an execution or resource ownership ledger.

Pending evidence and already-public history each have their own capacity.
Active capacity counts only pending, retry, and uncertain rows. Terminal
history is capped at 1000 rows or 30 days, whichever is less, and stores a
receipt rather than the original evidence. Operation dedup retains at most
30 days / 20 times capacity, whichever is less; after that bounded lookback,
publication still reconciles the GitHub marker.
"""
from __future__ import annotations

import json
import os
import sqlite3
import stat
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any


MAX_AUTOMATIC_CYCLES = 3
UNSENT_TTL_SECONDS = 7 * 86400
TERMINAL_HISTORY_LIMIT = 1000
TERMINAL_HISTORY_SECONDS = 30 * 86400
_ACTIVE_STATES = ("pending", "retry", "uncertain")
_TERMINAL_STATES = ("published", "withdrawn", "expired", "exhausted", "blocked", "permanent-failed")
_RECEIPT_TOKEN_LIMIT = 128


class QueueFull(RuntimeError):
    pass


class Outbox:
    def __init__(self, path: str | Path, *, capacity: int = 1000, clock=time.time):
        if type(capacity) is not int or not 1 <= capacity <= 1000:
            raise ValueError("diagnostic queue capacity must be between 1 and 1000")
        self.path = Path(path)
        self._prepare_storage()
        self.clock = clock
        self.capacity = capacity
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS incidents (
                    fingerprint TEXT PRIMARY KEY, payload TEXT NOT NULL,
                    first_seen REAL NOT NULL, last_seen REAL NOT NULL,
                    occurrences INTEGER NOT NULL DEFAULT 1,
                    state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt REAL NOT NULL DEFAULT 0, lease_until REAL NOT NULL DEFAULT 0,
                    lease_token TEXT, issue_number INTEGER, issue_url TEXT, last_error TEXT,
                    consent TEXT);
                CREATE TABLE IF NOT EXISTS seen (
                    operation_id TEXT PRIMARY KEY, observed REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS publications (at REAL NOT NULL);
            """)
            if 'consent' not in {row[1] for row in db.execute('PRAGMA table_info(incidents)')}:
                raise ValueError('unsupported diagnostic queue schema; use a fresh state directory')
        self._restrict_database_file()

    def _prepare_storage(self) -> None:
        """Create only missing parents as owner-only dirs. Never chmod an existing parent."""
        path = self.path
        if path.is_symlink():
            raise ValueError("diagnostic queue path must be a regular SQLite file")
        parent = path.parent
        missing = []
        probe = parent
        while not probe.exists():
            missing.append(probe)
            if probe.parent == probe:
                break
            probe = probe.parent
        if probe.is_symlink():
            raise ValueError("diagnostic queue directory must not be a symlink")
        for directory in reversed(missing):
            os.mkdir(directory, 0o700)
            os.chmod(directory, 0o700)
        if parent.is_symlink():
            raise ValueError("diagnostic queue directory must not be a symlink")
        if not parent.is_dir():
            raise NotADirectoryError("diagnostic queue parent is not a directory")
        if os.name != "nt":
            info = parent.stat()
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise ValueError("diagnostic queue requires an owner-only directory")
        self._reject_unsafe_database_path()
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except FileExistsError:
            self._reject_unsafe_database_path()
        else:
            os.close(descriptor)

    def _reject_unsafe_database_path(self) -> None:
        path = self.path
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("diagnostic queue path must be a regular SQLite file")
        if os.name != "nt" and info.st_uid != os.getuid():
            raise ValueError("diagnostic queue owner mismatch")

    def _restrict_database_file(self) -> None:
        self._reject_unsafe_database_path()
        if self.path.exists():
            os.chmod(self.path, 0o600)

    @contextmanager
    def connect(self):
        self._reject_unsafe_database_path()
        db = sqlite3.connect(self.path, timeout=0.1)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=100")
        try:
            with db:
                yield db
        finally:
            db.close()
        self._restrict_database_file()

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
            self._terminalize(db, now)
            self._prune_published(db, now)
            if db.execute("SELECT 1 FROM seen WHERE operation_id=?", (operation_id,)).fetchone():
                return False
            existing = db.execute("SELECT payload FROM incidents WHERE fingerprint=?", (fingerprint,)).fetchone()
            if existing:
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
        """Expire over-budget or over-age unsent rows and compact terminal history. No timer."""
        now = self.clock()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._terminalize(db, now)
            self._prune_published(db, now)

    def _terminalize(self, db, now: float) -> None:
        """Close expired leases at the cycle budget, then age out unsent work."""
        leased_out = db.execute(
            "SELECT fingerprint, payload, state, issue_url FROM incidents "
            "WHERE state IN ('pending','retry','uncertain') AND attempts>=? AND lease_until<=?",
            (MAX_AUTOMATIC_CYCLES, now),
        ).fetchall()
        for row in leased_out:
            fields = ["state='exhausted'", "payload=?", "lease_until=0", "lease_token=NULL"]
            values = [self._receipt(row["fingerprint"], row["payload"], row["issue_url"])]
            if row["state"] == "uncertain":
                fields.append("last_error=?")
                values.append("submission_uncertain_budget_exhausted")
            values.append(row["fingerprint"])
            db.execute(
                "UPDATE incidents SET " + ",".join(fields) + " WHERE fingerprint=?",
                values,
            )
        aged = db.execute(
            "SELECT fingerprint, payload, state, issue_url FROM incidents "
            "WHERE state IN ('pending','retry','uncertain') AND first_seen<? AND lease_until<=?",
            (now - UNSENT_TTL_SECONDS, now),
        ).fetchall()
        for row in aged:
            fields = ["state='expired'", "payload=?", "lease_until=0", "lease_token=NULL"]
            values = [self._receipt(row["fingerprint"], row["payload"], row["issue_url"])]
            if row["state"] == "uncertain":
                fields.append("last_error=?")
                values.append("submission_uncertain_expired")
            values.append(row["fingerprint"])
            db.execute(
                "UPDATE incidents SET " + ",".join(fields) + " WHERE fingerprint=?",
                values,
            )

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
            "('published','withdrawn','expired','exhausted','blocked','permanent-failed')"
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
            self._terminalize(db, now)
            self._prune_published(db, now)
            row = db.execute(
                """SELECT * FROM incidents WHERE state IN ('pending','retry','uncertain')
                AND attempts < ? AND next_attempt <= ? AND lease_until <= ?
                ORDER BY first_seen LIMIT 1""",
                (MAX_AUTOMATIC_CYCLES, now, now),
            ).fetchone()
            if row is None:
                return None
            attempts = int(row["attempts"]) + 1
            db.execute(
                "UPDATE incidents SET attempts=?, lease_until=?, lease_token=? WHERE fingerprint=?",
                (attempts, now + lease_seconds, token, row["fingerprint"]),
            )
        return {**dict(row), "attempts": attempts, "lease_token": token, "payload": json.loads(row["payload"]),
                "consent": self._consent(row["consent"])}

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
        from .reporting import consent_allowed
        with self.connect() as db:
            rows = db.execute(
                "SELECT fingerprint,consent,state FROM incidents WHERE state NOT IN ('withdrawn','published')"
            ).fetchall()
        checked, withdrawn = {}, 0
        for row in rows:
            raw = row["consent"]
            if raw not in checked:
                checked[raw] = consent_allowed(self._consent(raw))
            if checked[raw]:
                continue
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

        The automatic cycle was already consumed by claim. This does not spend
        another attempt, and an hourly allowance cannot open a fourth cycle.
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
                "AND lease_until>? AND attempts BETWEEN 1 AND ? AND state IN ('pending','retry')",
                (item["fingerprint"], item["lease_token"], now, MAX_AUTOMATIC_CYCLES),
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
