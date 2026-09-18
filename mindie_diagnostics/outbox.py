"""Bounded publishing queue. Never an execution or resource ownership ledger.

Pending evidence and already-public history each have their own capacity.
Operation dedup retains at most 30 days / 20 times capacity, whichever is less;
after that bounded lookback, publication still reconciles the GitHub marker.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class QueueFull(RuntimeError):
    pass


class Outbox:
    def __init__(self, path: str | Path, *, capacity: int = 1000, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
                    diagnosis TEXT, diagnosis_state TEXT NOT NULL DEFAULT 'pending', consent TEXT);
                CREATE TABLE IF NOT EXISTS seen (
                    operation_id TEXT PRIMARY KEY, observed REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS publications (at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS generations (at REAL NOT NULL);
            """)
            if 'consent' not in {row[1] for row in db.execute('PRAGMA table_info(incidents)')}:
                raise ValueError('unsupported diagnostic queue schema; use a fresh state directory')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=5000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def enqueue(self, fingerprint: str, operation_id: str, payload: dict[str, Any], *, consent=None) -> bool:
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        if len(body.encode()) > 48000:
            raise ValueError("diagnostic issue payload exceeds 48000 bytes")
        reference = json.dumps(consent, ensure_ascii=True, separators=(",", ":")) if consent is not None else None
        if reference is not None and len(reference.encode()) > 8192:
            raise ValueError("community reference exceeds limit")
        now = self.clock()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM seen WHERE observed < ?", (now - 30 * 86400,))
            db.execute("DELETE FROM publications WHERE at < ?", (now - 86400,))
            db.execute("DELETE FROM generations WHERE at < ?", (now - 86400,))
            self._prune_published(db, now)
            if db.execute("SELECT 1 FROM seen WHERE operation_id=?", (operation_id,)).fetchone():
                return False
            existing = db.execute("SELECT 1 FROM incidents WHERE fingerprint=?", (fingerprint,)).fetchone()
            if existing:
                db.execute("UPDATE incidents SET last_seen=?, occurrences=occurrences+1 WHERE fingerprint=?", (now, fingerprint))
            else:
                if db.execute("SELECT COUNT(*) FROM incidents WHERE state NOT IN ('published','withdrawn')").fetchone()[0] >= self.capacity:
                    raise QueueFull("diagnostic outbox is full; retained incidents were not discarded")
                db.execute("INSERT INTO incidents (fingerprint,payload,first_seen,last_seen,consent) VALUES (?,?,?,?,?)", (fingerprint, body, now, now, reference))
            db.execute("INSERT INTO seen VALUES (?,?)", (operation_id, now))
            db.execute("DELETE FROM seen WHERE operation_id IN "
                       "(SELECT operation_id FROM seen ORDER BY observed DESC, operation_id DESC LIMIT -1 OFFSET ?)",
                       (self.capacity * 20,))
        return True

    def _prune_published(self, db, now):
        # Public markers are recoverable remotely. Withdrawn items can never
        # resume; bounded local history prevents revocation filling the queue.
        # Active unpublished evidence is never evicted to make intake space.
        db.execute("DELETE FROM incidents WHERE state IN ('published','withdrawn') AND last_seen < ?", (now - 30 * 86400,))
        db.execute("DELETE FROM incidents WHERE fingerprint IN "
                   "(SELECT fingerprint FROM incidents WHERE state IN ('published','withdrawn') "
                   "ORDER BY last_seen DESC, fingerprint DESC LIMIT -1 OFFSET ?)", (self.capacity,))

    def claim(self, *, lease_seconds: float = 300) -> dict[str, Any] | None:
        now, token = self.clock(), uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT * FROM incidents WHERE state IN ('pending','retry','uncertain')
                AND next_attempt <= ? AND lease_until <= ? ORDER BY first_seen LIMIT 1""", (now, now)).fetchone()
            if row is None:
                return None
            db.execute("UPDATE incidents SET lease_until=?, lease_token=? WHERE fingerprint=?", (now + lease_seconds, token, row['fingerprint']))
        return {**dict(row), "lease_token": token, "payload": json.loads(row["payload"]),
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
        overlapping worker cannot publish a revoked item after generation.
        """
        from .community import consent_allowed
        with self.connect() as db:
            rows = db.execute("SELECT fingerprint,consent,state FROM incidents WHERE state!='withdrawn'").fetchall()
        checked, withdrawn = {}, 0
        for row in rows:
            raw = row['consent']
            if raw not in checked:
                checked[raw] = consent_allowed(self._consent(raw))
            if checked[raw]:
                continue
            with self.connect() as db:
                if row['state'] == 'published':
                    changed = db.execute("UPDATE incidents SET diagnosis_state='withdrawn',lease_until=0,lease_token=NULL "
                                         "WHERE fingerprint=? AND diagnosis_state!='withdrawn'", (row['fingerprint'],))
                else:
                    changed = db.execute("UPDATE incidents SET state='withdrawn',diagnosis_state='withdrawn',"
                                         "last_error='community_consent_unavailable_or_withdrawn',lease_until=0,lease_token=NULL "
                                         "WHERE fingerprint=? AND state!='withdrawn'", (row['fingerprint'],))
                withdrawn += changed.rowcount
        with self.connect() as db:
            self._prune_published(db, self.clock())
        return withdrawn

    def update(self, item: dict[str, Any], **fields: Any) -> None:
        allowed = {"state", "attempts", "next_attempt", "issue_number", "issue_url", "last_error", "diagnosis", "diagnosis_state"}
        if not fields or set(fields) - allowed:
            raise ValueError("invalid outbox update")
        with self.connect() as db:
            cursor = db.execute("UPDATE incidents SET " + ",".join(f"{key}=?" for key in fields) + ",lease_until=0,lease_token=NULL WHERE fingerprint=? AND lease_token=?", (*fields.values(), item["fingerprint"], item["lease_token"]))
            if cursor.rowcount != 1:
                raise RuntimeError("diagnostic worker lease lost")
            self._prune_published(db, self.clock())

    def begin_post(self, item: dict[str, Any], *, hourly_limit: int = 10) -> bool:
        """Persist uncertainty BEFORE a mutating request, including process death."""
        now = self.clock()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM publications WHERE at >= ?", (now - 3600,)).fetchone()[0]
            if count >= hourly_limit:
                return False
            cursor = db.execute("UPDATE incidents SET state='uncertain',attempts=attempts+1 WHERE fingerprint=? AND lease_token=? AND lease_until>?", (item["fingerprint"], item["lease_token"], now))
            if cursor.rowcount != 1:
                raise RuntimeError("diagnostic worker lease expired before publication")
            db.execute("INSERT INTO publications VALUES (?)", (now,))
        return True

    def begin_generation(self, item: dict[str, Any], *, hourly_limit: int = 10) -> bool:
        """Bound paid model requests independently of comment publication."""
        now = self.clock()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT 1 FROM incidents WHERE fingerprint=? AND lease_token=? AND lease_until>?',
                              (item['fingerprint'], item['lease_token'], now)).fetchone():
                raise RuntimeError('bot lease expired before generation')
            db.execute('DELETE FROM generations WHERE at < ?', (now - 86400,))
            if db.execute('SELECT COUNT(*) FROM generations WHERE at >= ?', (now - 3600,)).fetchone()[0] >= hourly_limit:
                return False
            db.execute('INSERT INTO generations VALUES (?)', (now,))
        return True

    def rows(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT fingerprint,first_seen,last_seen,occurrences,state,attempts,next_attempt,issue_number,issue_url,last_error,diagnosis_state FROM incidents ORDER BY (state='published'),last_seen DESC LIMIT ?", (self.capacity * 2,))]

    def published(self, *, limit: int = 10) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [{**dict(row), 'consent': self._consent(row['consent'])}
                    for row in db.execute("SELECT * FROM incidents WHERE state='published' AND diagnosis_state='pending' ORDER BY first_seen LIMIT ?", (min(limit, 100),))]
