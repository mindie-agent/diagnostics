"""Read-only bounded SQL access to an Ascend profiler SQLite capture.

[Role]
  Provide the ``mindie-profiler sql`` primitive: open an existing DB with
  ``mode=ro``, force ``PRAGMA query_only=ON``, deny write/DDL/ATTACH/PRAGMA/
  extension-loading through a sqlite authorizer, and execute one bounded
  query with a progress-handler deadline and an explicit row limit.

[Invariants]
  * The database file is opened URI read-only and never modified.
  * Only one statement is executed; multiple statements raise.
  * A query exceeding ``limit`` rows raises ``ProfilerQueryError``.  It is
    never silently truncated and presented as a complete result.
  * A query exceeding ``timeout_s`` is interrupted by the sqlite progress
    handler and raises ``ProfilerQueryError``.
  * ``load_extension``/``writefile``/``readfile`` and other side-effecting
    functions are denied even if a future sqlite build exposes them.
  * STRING_IDS resolution is additive: the raw id column is preserved and a
    ``<column>_resolved`` companion is added only for string-id-like columns.
    Missing ids stay ``<unresolved:id>``.

[Timeouts & Locks]
  ``timeout_s`` defaults to 10s and is enforced by a progress handler checked
  every 20k VM instructions.  ``busy_timeout`` mirrors the deadline.  The
  connection is closed in ``finally``; no lock survives the call.

[Side-Effect Policy]
  Read-only filesystem access; no writes, no network, no subprocesses.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from .events import ProfilerQueryError, ProfilerSchemaError

SQL_SCHEMA = "mindie-profiler-sql/1"
_DEFAULT_TIMEOUT_S = 10.0
_DEFAULT_LIMIT = 50
_ALLOWED_START = re.compile(r"^\s*(select|with|explain|values)\b", re.IGNORECASE)
_COMMENT = re.compile(r"^\s*(?:--[^\n]*\n|/\*.*?\*/)\s*", re.DOTALL)
_DENIED_FUNCTIONS = {
    "load_extension",
    "writefile",
    "readfile",
    "edit",
    "fsdir",
    "zipfile",
    "sqlite_compileoption_used",
}
_STRING_ID_COLUMN = re.compile(
    r"(?:name|type|core|shapes|dtypes|formats|group)$", re.IGNORECASE
)

_DENIED_ACTIONS = {
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_CREATE_VTABLE,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_INDEX,
    sqlite3.SQLITE_DROP_TEMP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_TRIGGER,
    sqlite3.SQLITE_DROP_TEMP_VIEW,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_DROP_VTABLE,
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_REINDEX,
    sqlite3.SQLITE_ANALYZE,
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_PRAGMA,
    sqlite3.SQLITE_TRANSACTION,
    sqlite3.SQLITE_SAVEPOINT,
}


def _authorizer(
    action: int,
    arg1: str | None,
    _arg2: str | None,
    _db: str | None,
    _trigger: str | None,
    *,
    allow_pragma: bool = False,
) -> int:
    if action == sqlite3.SQLITE_PRAGMA:
        return sqlite3.SQLITE_OK if allow_pragma else sqlite3.SQLITE_DENY
    if action in _DENIED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and str(arg1 or "").lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def open_readonly(
    db_path: str | Path,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    allow_pragma: bool = False,
) -> sqlite3.Connection:
    """Open a DB URI read-only with query_only and a side-effect authorizer.

    ``allow_pragma`` is for internal schema introspection (``PRAGMA
    table_info``) only; the user-facing ``sql`` command never enables it.
    """

    path = Path(db_path).expanduser()
    if not path.is_file():
        raise ProfilerSchemaError(f"profiler db not found: {path}")
    timeout = max(0.1, float(timeout_s))
    uri = "file:" + quote(str(path.resolve()), safe="/") + "?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True, timeout=timeout)
    except sqlite3.Error as exc:
        raise ProfilerQueryError(f"cannot open profiler db read-only: {path}: {exc}") from exc

    def authorizer(action: int, arg1: str | None, arg2: str | None, db: str | None, trigger: str | None) -> int:
        return _authorizer(action, arg1, arg2, db, trigger, allow_pragma=allow_pragma)

    try:
        con.execute("PRAGMA query_only = ON")
        con.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
        try:
            con.enable_load_extension(False)
        except (AttributeError, sqlite3.Error):
            pass
        con.set_authorizer(authorizer)
    except sqlite3.Error:
        con.close()
        raise
    return con


def _strip_leading_comments(query: str) -> str:
    text = query
    while True:
        match = _COMMENT.match(text)
        if not match:
            return text
        text = text[match.end():]


def _validate_statement(query: str) -> str:
    text = query.strip()
    if not text:
        raise ProfilerSchemaError("empty SQL query")
    if not _ALLOWED_START.match(_strip_leading_comments(text)):
        raise ProfilerQueryError(
            "only read-only SELECT / WITH / EXPLAIN / VALUES queries are allowed"
        )
    if "\x00" in text:
        raise ProfilerQueryError("SQL query contains a NUL byte")
    return text


def string_ids_map(con: sqlite3.Connection) -> dict[int, str]:
    """Load STRING_IDS once; an absent table is reported as unavailable."""

    try:
        tables = {str(row[0]) for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    except sqlite3.Error:
        return {}
    if "STRING_IDS" not in tables:
        return {}
    try:
        return {
            int(row[0]): str(row[1]) if row[1] is not None else ""
            for row in con.execute("SELECT id, value FROM STRING_IDS")
        }
    except sqlite3.Error:
        return {}


def _resolve(value: Any, strings: dict[int, str]) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in strings:
            return strings[value]
        return f"<unresolved:{value}>"
    if isinstance(value, str):
        try:
            key = int(value)
        except ValueError:
            return value
        if key in strings:
            return strings[key]
    return value


def _row_dict(cursor: sqlite3.Cursor, row: Iterable[Any], strings: dict[int, str], resolve: bool) -> dict[str, Any]:
    columns = [str(item[0]) for item in cursor.description or ()]
    payload: dict[str, Any] = {}
    for column, value in zip(columns, row):
        payload[column] = value
        if resolve and _STRING_ID_COLUMN.search(column):
            payload[f"{column}_resolved"] = _resolve(value, strings)
    return payload


def run_sql(
    db_path: str | Path,
    query: str,
    *,
    limit: int = _DEFAULT_LIMIT,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    resolve_strings: bool = True,
) -> dict[str, Any]:
    """Execute one bounded read-only query and return rows plus provenance."""

    if limit <= 0:
        raise ProfilerQueryError("limit must be positive")
    if timeout_s <= 0:
        raise ProfilerQueryError("timeout_s must be positive")
    text = _validate_statement(query)
    con = open_readonly(db_path, timeout_s=timeout_s)
    started = time.perf_counter()
    deadline = started + max(0.1, float(timeout_s))
    con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 20_000)
    try:
        strings = string_ids_map(con) if resolve_strings else {}
        cursor = con.execute(text)
        rows = cursor.fetchmany(int(limit) + 1)
        if len(rows) > limit:
            raise ProfilerQueryError(
                f"query returned more than the {limit}-row limit; add a narrower WHERE or raise --limit"
            )
        payload = [_row_dict(cursor, row, strings, resolve_strings) for row in rows]
        columns = [str(item[0]) for item in cursor.description or ()]
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise ProfilerQueryError(f"query exceeded the {timeout_s:.1f}s deadline") from exc
        raise ProfilerQueryError(f"query failed: {exc}") from exc
    except sqlite3.DatabaseError as exc:
        raise ProfilerQueryError(f"query rejected: {exc}") from exc
    finally:
        con.set_progress_handler(None, 0)
        con.set_authorizer(None)
        con.close()
    resolved_columns = [
        str(item[0]) for item in payload[0].keys() if str(item).endswith("_resolved")
    ] if payload else []
    return {
        "schema_version": SQL_SCHEMA,
        "db": str(Path(db_path).expanduser().resolve()),
        "query": text,
        "row_count": len(payload),
        "row_limit": int(limit),
        "truncated": False,
        "columns": columns,
        "rows": payload,
        "string_ids": {
            "available": bool(strings),
            "count": len(strings),
            "resolved_columns": resolved_columns,
            "resolution_policy": "raw id preserved; <column>_resolved added for string-id-like columns",
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 6),
    }


__all__ = ["SQL_SCHEMA", "open_readonly", "run_sql", "string_ids_map"]
