"""Fast, read-only inventory of an Ascend profiling directory.

[Role]
  Answer "what is here and can it answer the question?" without analyzing
  windows: enumerate ranks, every DB/CSV/JSON candidate (including ambiguous
  multiple-DB directories), sizes, available tables, native ``ProfilerStep#``
  and MSTX phase windows, and the captured time span.

[Invariants]
  * Multiple DBs in one rank directory are all listed; no newest-mtime source
    is chosen.  The caller must pass an explicit DB before ``analyze``.
  * Missing tables are reported as absent, never fabricated as empty results.
  * Native step/MSTX windows come from real name rows joined through
    STRING_IDS; no window is inferred from kernel periodicity.
  * Every file size/mtime is read from stat; content SHA256 is not computed.

[Timeouts & Locks]
  SQLite connections use ``mode=ro`` + ``query_only=ON`` with a short bounded
  progress deadline.  Directory discovery is bounded by ``max_dirs``.

[Side-Effect Policy]
  Read-only filesystem/sqlite access.  No sidecar index, write, or network.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from .events import ProfilerError
from .segment import (
    _OUTPUT_DIR,
    find_csv_candidates,
    find_db_candidates,
    find_trace_candidates,
    infer_rank_id,
    infer_rank_index,
)
from .sources_db import probe_db
from .sql import open_readonly

INVENTORY_SCHEMA = "mindie-profiler-inventory/1"
_CORE_TABLES = (
    "STRING_IDS",
    "TASK",
    "COMPUTE_TASK_INFO",
    "PYTORCH_API",
    "CANN_API",
    "CONNECTION_IDS",
    "MSTX_EVENTS",
    "COMMUNICATION_OP",
    "COMMUNICATION_SCHEDULE_TASK_INFO",
    "TASK_PMU_INFO",
)
_STEP_RE = re.compile(r"ProfilerStep#(\d+)", re.IGNORECASE)
_PHASE_RE = re.compile(r"(warmup|prefill|decode|draft|verify|graph_capture)", re.IGNORECASE)
_TIME_TABLES = ("TASK", "PYTORCH_API", "MSTX_EVENTS", "CANN_API", "COMMUNICATION_OP")


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _pick(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    available = set(columns)
    for alias in aliases:
        if alias in available:
            return alias
    return None


def _file_row(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "basename": path.name, "size_bytes": None, "mtime_ns": None}
    return {
        "path": str(path.resolve()),
        "basename": path.name,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _discover_rank_dirs(root: Path, *, max_dirs: int) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    walked = 0
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        walked += 1
        if walked > max_dirs:
            raise ProfilerError(f"inventory discovery exceeded {max_dirs} directories under {root}")
        current_path = Path(current)
        has_db = any(name.startswith("ascend_pytorch_profiler") and name.endswith(".db") for name in filenames)
        has_csv = "kernel_details.csv" in filenames
        has_trace = any(name.startswith("trace_view") and name.endswith(".json") for name in filenames)
        has_output = _OUTPUT_DIR in dirnames
        if has_db or has_csv or has_trace:
            if current_path not in seen:
                discovered.append(current_path)
                seen.add(current_path)
            dirnames[:] = []
            continue
        if has_output:
            output = current_path / _OUTPUT_DIR
            try:
                names = {entry.name for entry in output.iterdir()}
            except OSError:
                names = set()
            if (
                any(name.startswith("ascend_pytorch_profiler") and name.endswith(".db") for name in names)
                or "kernel_details.csv" in names
                or any(name.startswith("trace_view") and name.endswith(".json") for name in names)
            ):
                if current_path not in seen:
                    discovered.append(current_path)
                    seen.add(current_path)
                dirnames[:] = []
    return discovered


def _string_ids(con: sqlite3.Connection) -> dict[int, str]:
    try:
        return {
            int(row[0]): str(row[1]) if row[1] is not None else ""
            for row in con.execute("SELECT id, value FROM STRING_IDS")
        }
    except sqlite3.Error:
        return {}


def _native_windows(
    con: sqlite3.Connection,
    tables: set[str],
    strings: dict[int, str],
    *,
    limit: int = 200,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    mstx: list[dict[str, Any]] = []
    for table in ("PYTORCH_API", "MSTX_EVENTS"):
        if table not in tables:
            continue
        columns = _table_columns(con, table)
        name_col = _pick(columns, ("name", "eventName", "message"))
        start_col = _pick(columns, ("startNs", "start_ns", "timestamp"))
        end_col = _pick(columns, ("endNs", "end_ns"))
        tid_col = _pick(columns, ("globalTid", "tid", "threadId"))
        if name_col is None or start_col is None or end_col is None:
            continue
        tid_expr = f", {tid_col}" if tid_col else ", NULL"
        sql = (
            f"SELECT {name_col}, {start_col}, {end_col}{tid_expr} FROM {table} "
            f"WHERE {end_col} > {start_col} ORDER BY {start_col} LIMIT ?"
        )
        try:
            rows = con.execute(sql, (max(1, int(limit)) * 50,)).fetchall()
        except sqlite3.Error:
            continue
        for row in rows:
            raw_name = row[0]
            if isinstance(raw_name, int):
                name = strings.get(raw_name, f"<unresolved:{raw_name}>")
            else:
                name = str(raw_name or "")
            start_ns, end_ns = int(row[1]), int(row[2])
            entry = {
                "source_table": table,
                "name": name,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ms": round((end_ns - start_ns) / 1_000_000.0, 6),
                "global_tid": None if row[3] is None else str(row[3]),
            }
            step_match = _STEP_RE.search(name)
            if step_match:
                entry["step"] = int(step_match.group(1))
                steps.append(entry)
                continue
            phase_match = _PHASE_RE.search(name)
            if phase_match:
                entry["phase"] = phase_match.group(1).lower()
                mstx.append(entry)
    steps.sort(key=lambda item: (item["start_ns"], item.get("step", 0)))
    mstx.sort(key=lambda item: item["start_ns"])
    return steps[:limit], mstx[:limit]


def _capture_span(con: sqlite3.Connection, tables: set[str]) -> dict[str, Any] | None:
    lows: list[int] = []
    highs: list[int] = []
    for table in _TIME_TABLES:
        if table not in tables:
            continue
        columns = _table_columns(con, table)
        start_col = _pick(columns, ("startNs", "start_ns", "timestamp"))
        end_col = _pick(columns, ("endNs", "end_ns"))
        if start_col is None or end_col is None:
            continue
        try:
            row = con.execute(f"SELECT MIN({start_col}), MAX({end_col}) FROM {table}").fetchone()
        except sqlite3.Error:
            continue
        if row and row[0] is not None and row[1] is not None:
            lows.append(int(row[0]))
            highs.append(int(row[1]))
    if not lows or not highs:
        return None
    start_ns, end_ns = min(lows), max(highs)
    return {
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ms": round((end_ns - start_ns) / 1_000_000.0, 6),
        "basis": "min(startNs)..max(endNs) over available time tables",
    }


def _db_facts(path: Path, *, timeout_s: float) -> dict[str, Any]:
    row = _file_row(path)
    facts: dict[str, Any] = {
        **row,
        "kind": "db",
        "schema_version": None,
        "schema_variant": None,
        "tables": [],
        "core_tables": {name: False for name in _CORE_TABLES},
        "native_step_windows": [],
        "mstx_windows": [],
        "capture_span": None,
        "limitations": [],
    }
    try:
        schema = probe_db(path, timeout_s=timeout_s)
        facts["schema_version"] = schema.schema_version
        facts["schema_variant"] = schema.schema_variant
        facts["tables"] = list(schema.tables)
    except ProfilerError as exc:
        facts["limitations"].append({"code": "db_schema_probe_failed", "detail": str(exc)})
    try:
        con = open_readonly(path, timeout_s=timeout_s, allow_pragma=True)
    except (ProfilerError, sqlite3.Error) as exc:
        facts["limitations"].append({"code": "db_open_failed", "detail": str(exc)})
        return facts
    try:
        deadline = time.perf_counter() + max(0.1, timeout_s)
        con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 20_000)
        table_rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        tables = {str(item[0]) for item in table_rows}
        facts["tables"] = sorted(tables)
        facts["core_tables"] = {name: name in tables for name in _CORE_TABLES}
        strings = _string_ids(con) if "STRING_IDS" in tables else {}
        steps, mstx = _native_windows(con, tables, strings)
        facts["native_step_windows"] = steps
        facts["mstx_windows"] = mstx
        facts["capture_span"] = _capture_span(con, tables)
        if not steps and not mstx:
            facts["limitations"].append(
                {"code": "native_windows_unavailable", "detail": "no ProfilerStep# or MSTX phase rows were found"}
            )
    except sqlite3.Error as exc:
        facts["limitations"].append({"code": "db_query_failed", "detail": str(exc)})
    finally:
        con.close()
    return facts


def _rank_facts(rank_dir: Path, *, ordinal: int, timeout_s: float) -> dict[str, Any]:
    rank_id = infer_rank_id(rank_dir, ordinal)
    db_candidates = find_db_candidates(rank_dir)
    csv_candidates = find_csv_candidates(rank_dir)
    trace_candidates = find_trace_candidates(rank_dir)
    dbs = [_db_facts(path, timeout_s=timeout_s) for path in db_candidates]
    files = (
        [*dbs]
        + [{**_file_row(path), "kind": "csv"} for path in csv_candidates]
        + [{**_file_row(path), "kind": "json_trace"} for path in trace_candidates]
    )
    limitations: list[dict[str, Any]] = []
    if len(db_candidates) > 1:
        limitations.append(
            {
                "code": "multiple_db_candidates",
                "detail": f"{len(db_candidates)} DB candidates; pass one explicit .db path",
                "candidates": [str(item.resolve()) for item in db_candidates],
            }
        )
    if not db_candidates and not csv_candidates and not trace_candidates:
        limitations.append({"code": "no_source_files", "detail": "rank directory has no DB/CSV/trace source"})
    return {
        "rank_id": rank_id,
        "rank_index": infer_rank_index(rank_dir.name),
        "rank_dir": str(rank_dir.resolve()),
        "db_candidates": [str(item.resolve()) for item in db_candidates],
        "csv_candidates": [str(item.resolve()) for item in csv_candidates],
        "trace_candidates": [str(item.resolve()) for item in trace_candidates],
        "files": files,
        "limitations": limitations,
    }


def inventory_profile(
    root: str | Path,
    *,
    max_dirs: int = 4096,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Return a bounded inventory object for a directory or a single source file."""

    path = Path(root).expanduser()
    if not path.exists():
        raise ProfilerError(f"profiling root not found: {path}")
    ranks: list[dict[str, Any]] = []
    if path.is_file():
        rank_dir = path.parent.parent if path.parent.name == _OUTPUT_DIR else path.parent
        ranks.append(_rank_facts(rank_dir, ordinal=0, timeout_s=timeout_s))
    else:
        resolved = path.resolve()
        if resolved.name == _OUTPUT_DIR:
            resolved = resolved.parent
        own = find_db_candidates(resolved) + find_csv_candidates(resolved) + find_trace_candidates(resolved)
        if own:
            ranks.append(_rank_facts(resolved, ordinal=0, timeout_s=timeout_s))
        else:
            for ordinal, rank_dir in enumerate(sorted(_discover_rank_dirs(resolved, max_dirs=max_dirs), key=str)):
                ranks.append(_rank_facts(rank_dir, ordinal=ordinal, timeout_s=timeout_s))
    if not ranks:
        raise ProfilerError(f"no rank/source candidates found under {path}")
    limitations: list[dict[str, Any]] = []
    for rank in ranks:
        limitations.extend(
            {"rank_id": rank["rank_id"], **item} for item in rank.get("limitations", [])
        )
    return {
        "schema_version": INVENTORY_SCHEMA,
        "root": str(path.resolve()),
        "rank_count": len(ranks),
        "ranks": ranks,
        "source_priority": "DB > CSV > JSON Trace; only one authority per metric",
        "identity_policy": "path+size_bytes+mtime_ns+schema_version; no SHA256 by default",
        "limitations": limitations,
    }


__all__ = ["INVENTORY_SCHEMA", "inventory_profile"]
