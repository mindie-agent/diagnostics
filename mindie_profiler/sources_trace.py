"""Streaming Chrome Trace JSON reader with a rebuildable sidecar index.

[Role]
  Read huge ``trace_view.json`` captures without ``json.load`` of the whole
  file.  Both Chrome roots are accepted: a bare event array ``[...]`` and an
  object root ``{"traceEvents": [...]}``.  Optionally materialise a local
  SQLite sidecar index for repeated window/name queries.

[Invariants]
  * Parsing is incremental: one top-level event object is decoded at a time and
    the consumed prefix of the read buffer is discarded.  No code path reads
    the entire file into memory.
  * The index is disposable and rebuildable.  Its validity key is
    ``(resolved_path, size_bytes, mtime_ns, schema_version)``; content SHA256
    is never computed here.
  * Timestamps are kept as the raw JSON value (Chrome microseconds) plus an
    integer ``ts_ns`` derived once with ``Decimal``; no float subtraction is
    used for intervals.
  * A malformed trace raises ``ProfilerSchemaError`` with a byte/char offset
    instead of silently yielding a partial event set.

[Timeouts & Locks]
  Index builds use WAL plus ``busy_timeout=5000``.  The read stream itself has
  no wall-clock deadline; callers bound query work with SQLite progress
  handlers in ``sql.py``.

[Side-Effect Policy]
  Reads the supplied trace.  The sidecar index is the only file written and is
  explicitly disposable; no network or source mutation.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from .events import ProfilerQueryError, ProfilerSchemaError

TRACE_SCHEMA = "mindie-profiler-trace-index/1"
TRACE_INDEX_SUFFIX = ".mindie-trace-index.sqlite3"
_DEFAULT_CHUNK = 1024 * 1024


def index_path_for(trace_path: str | Path) -> Path:
    path = Path(trace_path)
    return path.parent / (path.name + TRACE_INDEX_SUFFIX)


class _JsonStream:
    """Small incremental JSON reader; only one value is materialised at a time."""

    def __init__(self, path: Path, *, chunk_bytes: int = _DEFAULT_CHUNK) -> None:
        self.path = path
        self.chunk_bytes = max(4096, int(chunk_bytes))
        self._handle = path.open("r", encoding="utf-8")
        self._buffer = ""
        self._pos = 0
        self._eof = False
        self._decoder = json.JSONDecoder()
        self._consumed_chars = 0
        self._fill()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "_JsonStream":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _fill(self) -> bool:
        if self._eof:
            return False
        chunk = self._handle.read(self.chunk_bytes)
        if not chunk:
            self._eof = True
            return False
        if self._pos:
            self._consumed_chars += self._pos
            self._buffer = self._buffer[self._pos:] + chunk
            self._pos = 0
        else:
            self._buffer += chunk
        return True

    def _compact(self) -> None:
        if self._pos > 65536 or self._pos == len(self._buffer):
            self._consumed_chars += self._pos
            self._buffer = self._buffer[self._pos:]
            self._pos = 0

    def peek(self) -> str:
        self._skip_ws()
        return self._buffer[self._pos] if self._pos < len(self._buffer) else ""

    def _skip_ws(self) -> None:
        while True:
            size = len(self._buffer)
            while self._pos < size and self._buffer[self._pos] in " \t\r\n":
                self._pos += 1
            if self._pos < size or not self._fill():
                return

    def expect(self, char: str) -> None:
        if self.peek() != char:
            raise self._error(f"expected {char!r}")
        self._pos += 1

    def _error(self, detail: str) -> ProfilerSchemaError:
        offset = self._consumed_chars + self._pos
        return ProfilerSchemaError(f"{self.path.name}: invalid JSON trace at char {offset}: {detail}")

    def value(self) -> Any:
        self._skip_ws()
        while True:
            try:
                value, end = self._decoder.raw_decode(self._buffer, self._pos)
            except json.JSONDecodeError as exc:
                if self._fill():
                    continue
                raise self._error(exc.msg) from exc
            if end == len(self._buffer) and not self._eof:
                # A scalar may have been cut at the chunk edge.  Refill once
                # and re-decode; complete objects simply decode identically.
                if self._fill():
                    continue
            self._pos = end
            return value

    def iter_array(self) -> Iterator[Any]:
        self.expect("[")
        if self.peek() == "]":
            self._pos += 1
            return
        while True:
            yield self.value()
            char = self.peek()
            if char == ",":
                self._pos += 1
                continue
            if char == "]":
                self._pos += 1
                return
            raise self._error("expected ',' or ']' in trace array")


def _root_kind(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                return "empty"
            for char in chunk:
                if char in " \t\r\n":
                    continue
                if char == "[":
                    return "array"
                if char == "{":
                    return "object"
                return "unknown"


def iter_trace_events(trace_path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield Chrome trace event objects one at a time (array or object root)."""

    path = Path(trace_path)
    if not path.is_file():
        raise ProfilerSchemaError(f"trace file not found: {path}")
    with _JsonStream(path) as stream:
        char = stream.peek()
        if char == "[":
            for item in stream.iter_array():
                if isinstance(item, dict):
                    yield item
            return
        if char != "{":
            raise stream._error("root must be an array or an object with traceEvents")
        stream.expect("{")
        found = False
        if stream.peek() == "}":
            stream._pos += 1
            raise ProfilerSchemaError(f"{path.name}: object root has no traceEvents array")
        while True:
            key = stream.value()
            stream.expect(":")
            if key == "traceEvents":
                found = True
                if stream.peek() != "[":
                    raise stream._error("traceEvents must be an array")
                for item in stream.iter_array():
                    if isinstance(item, dict):
                        yield item
            else:
                stream.value()  # bounded metadata value; never the event list
            char = stream.peek()
            if char == ",":
                stream._pos += 1
                continue
            if char == "}":
                stream._pos += 1
                break
            raise stream._error("expected ',' or '}' in trace object")
        if not found:
            raise ProfilerSchemaError(f"{path.name}: object root has no traceEvents array")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _ns_from_us(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int((Decimal(str(value)) * Decimal(1000)).to_integral_value())
    except (InvalidOperation, ValueError):
        return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _create_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS trace_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS trace_events(
            seq INTEGER PRIMARY KEY,
            name TEXT,
            cat TEXT,
            ph TEXT,
            ts REAL,
            dur REAL,
            ts_ns INTEGER,
            end_ns INTEGER,
            pid TEXT,
            tid TEXT,
            args_json TEXT
        );
        CREATE INDEX IF NOT EXISTS trace_events_ts_ns ON trace_events(ts_ns);
        CREATE INDEX IF NOT EXISTS trace_events_name ON trace_events(name);
        CREATE INDEX IF NOT EXISTS trace_events_tid ON trace_events(tid, ts_ns);
        """
    )


def _meta(con: sqlite3.Connection) -> dict[str, str]:
    try:
        return {str(row[0]): str(row[1]) for row in con.execute("SELECT key, value FROM trace_meta")}
    except sqlite3.Error:
        return {}


def _identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source_path": str(path.resolve()),
        "source_size_bytes": str(int(stat.st_size)),
        "source_mtime_ns": str(int(stat.st_mtime_ns)),
        "schema_version": TRACE_SCHEMA,
    }


def index_is_valid(index_path: str | Path, trace_path: str | Path) -> bool:
    """Return True only when the sidecar matches the current source identity."""

    index = Path(index_path)
    trace = Path(trace_path)
    if not index.is_file() or not trace.is_file():
        return False
    try:
        con = sqlite3.connect(_readonly_uri(index), uri=True, timeout=1.0)
    except sqlite3.Error:
        return False
    try:
        meta = _meta(con)
        expected = _identity(trace)
        return all(meta.get(key) == value for key, value in expected.items()) and meta.get("event_count") is not None
    finally:
        con.close()


def _readonly_uri(path: Path) -> str:
    return "file:" + quote(str(path.resolve()), safe="/") + "?mode=ro"


def _open_index(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path), timeout=5.0)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = NORMAL")
    con.execute("PRAGMA busy_timeout = 5000")
    return con


def build_trace_index(
    trace_path: str | Path,
    index_path: str | Path | None = None,
    *,
    chunk_bytes: int = _DEFAULT_CHUNK,
    batch_size: int = 2000,
    force: bool = False,
) -> dict[str, Any]:
    """Build (or reuse) a sidecar SQLite index and return its identity facts."""

    trace = Path(trace_path).expanduser().resolve()
    if not trace.is_file():
        raise ProfilerSchemaError(f"trace file not found: {trace}")
    index = Path(index_path).expanduser().resolve() if index_path else index_path_for(trace)
    identity = _identity(trace)
    if not force and index_is_valid(index, trace):
        return {
            "index_path": str(index),
            "rebuilt": False,
            "source": identity,
            "event_count": _index_event_count(index),
        }
    index.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    con = _open_index(index)
    try:
        con.execute("DROP TABLE IF EXISTS trace_events")
        con.execute("DROP TABLE IF EXISTS trace_meta")
        _create_schema(con)
        con.execute("DELETE FROM trace_meta")
        event_count = 0
        batch: list[tuple[Any, ...]] = []
        with con:
            for event in iter_trace_events(trace):
                event_count += 1
                ts = _number(event.get("ts"))
                dur = _number(event.get("dur"))
                ts_ns = _ns_from_us(event.get("ts"))
                end_ns = ts_ns + _ns_from_us(event.get("dur")) if ts_ns is not None and dur is not None else None
                args = event.get("args")
                args_json = json.dumps(args, ensure_ascii=False, sort_keys=True) if args is not None else None
                batch.append(
                    (
                        event_count,
                        _text(event.get("name")),
                        _text(event.get("cat")),
                        _text(event.get("ph")),
                        ts,
                        dur,
                        ts_ns,
                        end_ns,
                        _text(event.get("pid")),
                        _text(event.get("tid")),
                        args_json,
                    )
                )
                if len(batch) >= max(1, batch_size):
                    con.executemany(
                        "INSERT INTO trace_events(seq,name,cat,ph,ts,dur,ts_ns,end_ns,pid,tid,args_json) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        batch,
                    )
                    batch.clear()
            if batch:
                con.executemany(
                    "INSERT INTO trace_events(seq,name,cat,ph,ts,dur,ts_ns,end_ns,pid,tid,args_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    batch,
                )
            con.executemany(
                "INSERT OR REPLACE INTO trace_meta(key, value) VALUES (?, ?)",
                [
                    *identity.items(),
                    ("event_count", str(event_count)),
                    ("root_kind", _root_kind(trace)),
                    ("built_at_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                    ("build_elapsed_ms", f"{(time.perf_counter() - started) * 1000.0:.6f}"),
                ],
            )
    except sqlite3.Error as exc:
        raise ProfilerQueryError(f"cannot build trace index {index}: {exc}") from exc
    finally:
        con.close()
    return {
        "index_path": str(index),
        "rebuilt": True,
        "source": identity,
        "event_count": event_count,
    }


def _index_event_count(index: Path) -> int | None:
    try:
        con = sqlite3.connect(_readonly_uri(index), uri=True, timeout=1.0)
    except sqlite3.Error:
        return None
    try:
        row = con.execute("SELECT value FROM trace_meta WHERE key = 'event_count'").fetchone()
        return int(row[0]) if row and row[0] is not None else None
    except (sqlite3.Error, ValueError):
        return None
    finally:
        con.close()


@dataclass(frozen=True, slots=True)
class TraceIndex:
    path: str
    source_path: str
    event_count: int | None
    rebuilt: bool

    @classmethod
    def ensure(cls, trace_path: str | Path, index_path: str | Path | None = None, **kwargs: Any) -> "TraceIndex":
        facts = build_trace_index(trace_path, index_path, **kwargs)
        return cls(
            path=str(facts["index_path"]),
            source_path=str(facts["source"]["source_path"]),
            event_count=facts.get("event_count"),
            rebuilt=bool(facts["rebuilt"]),
        )


def query_trace_events(
    index_path: str | Path,
    *,
    start_ns: int | None = None,
    end_ns: int | None = None,
    name_pattern: str | None = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Bounded name/window query over an already-built sidecar index."""

    index = Path(index_path)
    if not index.is_file():
        raise ProfilerQueryError(f"trace index not found: {index}")
    clauses: list[str] = []
    params: list[Any] = []
    if start_ns is not None and end_ns is not None:
        clauses.append("ts_ns IS NOT NULL AND end_ns IS NOT NULL AND end_ns > ? AND ts_ns < ?")
        params.extend([int(start_ns), int(end_ns)])
    if name_pattern:
        clauses.append("name LIKE ?")
        params.append(f"%{name_pattern}%")
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    sql = (
        "SELECT seq,name,cat,ph,ts,dur,ts_ns,end_ns,pid,tid,args_json FROM trace_events"
        f"{where} ORDER BY ts_ns, seq LIMIT ?"
    )
    params.append(max(1, int(limit)))
    con = sqlite3.connect(_readonly_uri(index), uri=True, timeout=5.0)
    con.execute("PRAGMA query_only = ON")
    con.execute("PRAGMA busy_timeout = 5000")
    con.set_progress_handler(lambda: 0, 10000)
    try:
        rows = con.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        raise ProfilerQueryError(f"{index.name}: trace query failed: {exc}") from exc
    finally:
        con.close()
    keys = ("seq", "name", "cat", "ph", "ts", "dur", "ts_ns", "end_ns", "pid", "tid", "args_json")
    return [dict(zip(keys, row)) for row in rows]


def iter_indexed_events(index_path: str | Path, *, limit: int = 100_000) -> Iterator[dict[str, Any]]:
    index = Path(index_path)
    con = sqlite3.connect(_readonly_uri(index), uri=True, timeout=5.0)
    con.execute("PRAGMA busy_timeout = 5000")
    try:
        cursor = con.execute(
            "SELECT seq,name,cat,ph,ts,dur,ts_ns,end_ns,pid,tid,args_json "
            "FROM trace_events ORDER BY seq LIMIT ?",
            (int(limit),),
        )
        keys = ("seq", "name", "cat", "ph", "ts", "dur", "ts_ns", "end_ns", "pid", "tid", "args_json")
        for row in cursor:
            yield dict(zip(keys, row))
    finally:
        con.close()


__all__ = [
    "TRACE_SCHEMA",
    "TRACE_INDEX_SUFFIX",
    "TraceIndex",
    "build_trace_index",
    "index_is_valid",
    "index_path_for",
    "iter_indexed_events",
    "iter_trace_events",
    "query_trace_events",
]
