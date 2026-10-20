"""Host/Runtime to device correlation for one explicit profiler window.

[Role]
  Drill into a device gap, hotspot kernel name, or explicit window by joining
  ``PYTORCH_API`` / ``MSTX_EVENTS`` (same ``globalTid`` nanosecond containment)
  through ``CONNECTION_IDS`` to ``CANN_API`` and ``TASK``.  Output a bounded
  host call stack, host dispatch duration/launch delay, unmatched ratio, and
  the ambiguous ``globalTaskId`` groups that must never be bound 1:1.

[Invariants]
  * Host parent/child edges require exact integer-nanosecond containment on the
    same ``globalTid``; they are never guessed from display names.
  * TASK association prefers ``CONNECTION_IDS``/``connectionId``; a same-thread
    CANN containment fallback is tagged with its basis.
  * ACLGraph protection: ``globalTaskId`` rows repeated anywhere in the full
    ``TASK`` table are marked ambiguous.  No wait/PMU value is summed, averaged,
    or attached to one replay.
  * Device gaps are device-uncovered intervals, not evidence of a slow host by
    themselves.  ``host_dispatch_gap_ms`` is emitted only for associated host
    calls and carries its measurement basis.
  * Missing optional tables produce limitations, never fabricated zero rows.

[Timeouts & Locks]
  ``timeout_s`` (default 20s) is enforced through a sqlite progress handler.
  Every table read has an explicit row cap; the connection closes in finally.

[Side-Effect Policy]
  Read-only sqlite access to one DB.  No writes, index files, or network.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from .accounting import merge_intervals
from .events import ProfilerQueryError, ProfilerUsageError, Window
from .segment import discover_rank_sources, resolve_ranks
from .sources_db import _text
from .sql import open_readonly

CORRELATION_SCHEMA = "mindie-profiler-correlation/1"
_HOST_TABLES = ("PYTORCH_API", "MSTX_EVENTS")
_DEVICE_TABLE = "TASK"
_COMPUTE_TABLE = "COMPUTE_TASK_INFO"
_CONNECTION_TABLE = "CONNECTION_IDS"


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _tables(con: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _pick(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    available = set(columns)
    for alias in aliases:
        if alias in available:
            return alias
    return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ms(value_ns: int | float | None) -> float | None:
    return None if value_ns is None else round(float(value_ns) / 1_000_000.0, 6)


def _strings(con: sqlite3.Connection) -> dict[int, str]:
    try:
        return {
            int(row[0]): str(row[1]) if row[1] is not None else ""
            for row in con.execute("SELECT id, value FROM STRING_IDS")
        }
    except sqlite3.Error:
        return {}


def _host_rows(
    con: sqlite3.Connection,
    table: str,
    strings: dict[int, str],
    *,
    start_ns: int,
    end_ns: int,
    limit: int,
) -> list[dict[str, Any]]:
    columns = _columns(con, table)
    name_col = _pick(columns, ("name", "eventName", "message"))
    start_col = _pick(columns, ("startNs", "start_ns"))
    end_col = _pick(columns, ("endNs", "end_ns"))
    tid_col = _pick(columns, ("globalTid", "tid", "threadId"))
    if name_col is None or start_col is None or end_col is None:
        return []
    id_col = _pick(columns, ("id",))
    conn_col = _pick(columns, ("connectionId", "connection_id"))
    select = [name_col, start_col, end_col]
    select.append(tid_col or "NULL")
    select.append(id_col or "NULL")
    select.append(conn_col or "NULL")
    sql = (
        f"SELECT {', '.join(select)} FROM {table} "
        f"WHERE {end_col} > ? AND {start_col} < ? ORDER BY {start_col}, rowid LIMIT ?"
    )
    try:
        rows = con.execute(sql, (start_ns, end_ns, int(limit))).fetchall()
    except sqlite3.Error:
        return []
    out: list[dict[str, Any]] = []
    unresolved: list[int] = []
    for row in rows:
        start, end = _int(row[1]), _int(row[2])
        if start is None or end is None or end <= start:
            continue
        out.append(
            {
                "table": table,
                "name": _text(strings, row[0], unresolved),
                "start_ns": start,
                "end_ns": end,
                "global_tid": None if row[3] is None else str(row[3]),
                "id": _int(row[4]),
                "connection_id": _int(row[5]),
            }
        )
    return out


def _task_rows(
    con: sqlite3.Connection,
    strings: dict[int, str],
    *,
    start_ns: int,
    end_ns: int,
    limit: int,
) -> list[dict[str, Any]]:
    columns = _columns(con, _DEVICE_TABLE)
    if not columns:
        return []
    start_col = _pick(columns, ("startNs", "start_ns"))
    end_col = _pick(columns, ("endNs", "end_ns"))
    global_col = _pick(columns, ("globalTaskId",))
    task_col = _pick(columns, ("taskId",))
    stream_col = _pick(columns, ("streamId",))
    device_col = _pick(columns, ("deviceId",))
    conn_col = _pick(columns, ("connectionId", "connection_id"))
    model_col = _pick(columns, ("modelId",))
    if start_col is None or end_col is None or global_col is None:
        return []
    select = [global_col, start_col, end_col, task_col or "NULL", stream_col or "NULL",
              device_col or "NULL", conn_col or "NULL", model_col or "NULL"]
    sql = (
        f"SELECT {', '.join(select)} FROM {_DEVICE_TABLE} "
        f"WHERE {end_col} > ? AND {start_col} < ? ORDER BY {start_col}, {global_col} LIMIT ?"
    )
    try:
        rows = con.execute(sql, (start_ns, end_ns, int(limit))).fetchall()
    except sqlite3.Error as exc:
        raise ProfilerQueryError(f"cannot read TASK rows: {exc}") from exc
    out: list[dict[str, Any]] = []
    for row in rows:
        start, end = _int(row[1]), _int(row[2])
        if start is None or end is None or end <= start:
            continue
        out.append(
            {
                "global_task_id": _int(row[0]),
                "start_ns": start,
                "end_ns": end,
                "task_id": None if row[3] is None else str(row[3]),
                "stream_id": None if row[4] is None else str(row[4]),
                "device_id": None if row[5] is None else str(row[5]),
                "connection_id": _int(row[6]),
                "model_id": _int(row[7]),
            }
        )
    return out


def _compute_names(
    con: sqlite3.Connection,
    strings: dict[int, str],
    global_ids: Sequence[int],
    *,
    limit: int = 5000,
) -> dict[int, str]:
    if _COMPUTE_TABLE not in _tables(con) or not global_ids:
        return {}
    columns = _columns(con, _COMPUTE_TABLE)
    name_col = _pick(columns, ("name",))
    global_col = _pick(columns, ("globalTaskId",))
    if name_col is None or global_col is None:
        return {}
    out: dict[int, str] = {}
    unresolved: list[int] = []
    ids = list(dict.fromkeys(global_ids))[:limit]
    for offset in range(0, len(ids), 500):
        chunk = ids[offset: offset + 500]
        placeholders = ",".join("?" for _ in chunk)
        try:
            rows = con.execute(
                f"SELECT {global_col}, {name_col} FROM {_COMPUTE_TABLE} "
                f"WHERE {global_col} IN ({placeholders})",
                chunk,
            ).fetchall()
        except sqlite3.Error:
            return out
        for global_id, name_id in rows:
            out[int(global_id)] = _text(strings, name_id, unresolved)
    return out


def _connections(con: sqlite3.Connection) -> dict[int, int]:
    if _CONNECTION_TABLE not in _tables(con):
        return {}
    columns = _columns(con, _CONNECTION_TABLE)
    id_col = _pick(columns, ("id",))
    conn_col = _pick(columns, ("connectionId", "connection_id"))
    if id_col is None or conn_col is None:
        return {}
    out: dict[int, int] = {}
    try:
        for left, right in con.execute(f"SELECT {id_col}, {conn_col} FROM {_CONNECTION_TABLE}"):
            left_int, right_int = _int(left), _int(right)
            if left_int is not None and right_int is not None:
                out[left_int] = right_int
                out.setdefault(right_int, left_int)
    except sqlite3.Error:
        return {}
    return out


def _task_occurrences(con: sqlite3.Connection, global_ids: Sequence[int], *, limit: int = 2000) -> dict[int, int]:
    columns = _columns(con, _DEVICE_TABLE)
    global_col = _pick(columns, ("globalTaskId",))
    if global_col is None:
        return {}
    counts: dict[int, int] = {}
    ids = list(dict.fromkeys(global_ids))[:limit]
    for offset in range(0, len(ids), 500):
        chunk = ids[offset: offset + 500]
        placeholders = ",".join("?" for _ in chunk)
        try:
            rows = con.execute(
                f"SELECT {global_col}, COUNT(*) FROM {_DEVICE_TABLE} "
                f"WHERE {global_col} IN ({placeholders}) GROUP BY {global_col}",
                chunk,
            ).fetchall()
        except sqlite3.Error:
            return counts
        for global_id, count in rows:
            counts[int(global_id)] = int(count)
    return counts


def _connection_for(event: dict[str, Any], links: dict[int, int]) -> int | None:
    direct = event.get("connection_id")
    if direct is not None:
        return links.get(direct, direct)
    event_id = event.get("id")
    if event_id is not None:
        return links.get(event_id)
    return None


_STEP_MARKER = re.compile(r"^ProfilerStep#", re.IGNORECASE)


def _is_step_marker(event: dict[str, Any]) -> bool:
    return bool(_STEP_MARKER.match(str(event.get("name") or "")))


def _attach_stacks(events: list[dict[str, Any]]) -> None:
    by_thread: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_thread.setdefault(str(event.get("global_tid") or ""), []).append(event)
    for thread_events in by_thread.values():
        thread_events.sort(key=lambda item: (item["start_ns"], -item["end_ns"]))
        stack: list[dict[str, Any]] = []
        for event in thread_events:
            if not _is_step_marker(event):
                # Step markers are native window evidence, not module frames.
                while stack and _is_step_marker(stack[-1]):
                    stack.pop()
            while stack and not (
                stack[-1]["start_ns"] <= event["start_ns"] and stack[-1]["end_ns"] >= event["end_ns"]
            ):
                stack.pop()
            event["parent"] = stack[-1] if stack else None
            event["depth"] = len(stack)
            stack.append(event)


def _stack_path(event: dict[str, Any] | None, *, limit: int = 8) -> list[dict[str, Any]]:
    path: list[dict[str, Any]] = []
    cursor = event
    while cursor is not None and len(path) < limit:
        path.append(
            {
                "table": cursor["table"],
                "name": cursor["name"],
                "start_ns": cursor["start_ns"],
                "end_ns": cursor["end_ns"],
                "duration_ms": _ms(cursor["end_ns"] - cursor["start_ns"]),
                "global_tid": cursor.get("global_tid"),
            }
        )
        cursor = cursor.get("parent")
    return list(reversed(path))


def _find_cann(
    task: dict[str, Any],
    cann_events: Sequence[dict[str, Any]],
    links: dict[int, int],
) -> tuple[dict[str, Any] | None, str | None]:
    connection = task.get("connection_id")
    if connection is not None:
        for event in cann_events:
            if _connection_for(event, links) == connection:
                return event, "connection_ids"
    for event in cann_events:
        if event["start_ns"] <= task["start_ns"] and event["end_ns"] >= task["end_ns"]:
            return event, "cann_interval_contains_task"
    return None, None


def _find_pytorch(
    cann: dict[str, Any] | None,
    pytorch_events: Sequence[dict[str, Any]],
    links: dict[int, int],
    task: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if cann is not None:
        connection = _connection_for(cann, links)
        if connection is not None:
            for event in pytorch_events:
                if _connection_for(event, links) == connection:
                    return event, "connection_ids"
        tid = cann.get("global_tid")
        candidates = [
            event
            for event in pytorch_events
            if tid is not None
            and event.get("global_tid") == tid
            and event["start_ns"] <= cann["start_ns"]
            and event["end_ns"] >= cann["end_ns"]
        ]
        if candidates:
            # Tightest containing parent: latest start, then earliest end.
            return max(candidates, key=lambda event: (event["start_ns"], -event["end_ns"])), "same_global_tid_containment"
        if cann.get("parent") is not None:
            # Defensive fallback for a same-table stack if one is ever supplied.
            return cann["parent"], "same_global_tid_containment"
    connection = task.get("connection_id")
    if connection is not None:
        for event in pytorch_events:
            if _connection_for(event, links) == connection:
                return event, "connection_ids"
    return None, None


def _gap_intervals(intervals: Sequence[tuple[int, int]], start_ns: int, end_ns: int) -> list[tuple[int, int]]:
    merged = [(max(left, start_ns), min(right, end_ns)) for left, right in merge_intervals(intervals)]
    merged = [(left, right) for left, right in merged if right > left]
    if not merged:
        return [(start_ns, end_ns)]
    gaps: list[tuple[int, int]] = []
    if merged[0][0] > start_ns:
        gaps.append((start_ns, merged[0][0]))
    for left, right in zip(merged[:-1], merged[1:]):
        if right[0] > left[1]:
            gaps.append((left[1], right[0]))
    if merged[-1][1] < end_ns:
        gaps.append((merged[-1][1], end_ns))
    return gaps


def correlate_profile(
    root: str | Path,
    *,
    window: Window | None = None,
    full_range: bool = False,
    ranks: Iterable[str] | None = None,
    kernel_name: str | None = None,
    min_gap_ms: float | None = None,
    max_rows: int = 5000,
    timeout_s: float = 20.0,
    call_limit: int = 200,
) -> dict[str, Any]:
    """Correlate host stacks and device tasks for one rank/window."""

    if window is None and not full_range:
        raise ProfilerUsageError("correlate requires an explicit --window or --full-range")
    if window is not None and full_range:
        raise ProfilerUsageError("--full-range cannot be combined with --window")
    if max_rows <= 0 or call_limit <= 0:
        raise ProfilerUsageError("max_rows and call_limit must be positive")
    sources = resolve_ranks(discover_rank_sources(root, source="db"), ranks)
    if not sources:
        raise ProfilerUsageError("no rank selected")
    item = sources[0]
    if item.db_path is None:
        raise ProfilerUsageError(f"rank {item.rank_id!r} has no DB source")
    db_path = Path(item.db_path)
    con = open_readonly(db_path, timeout_s=timeout_s, allow_pragma=True)
    started = time.perf_counter()
    deadline = started + max(0.1, timeout_s)
    con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 20_000)
    limitations: list[dict[str, Any]] = []
    try:
        tables = _tables(con)
        strings = _strings(con) if "STRING_IDS" in tables else {}
        if window is None:
            row = con.execute(
                f"SELECT MIN(startNs), MAX(endNs) FROM {_DEVICE_TABLE}"
            ).fetchone() if _DEVICE_TABLE in tables else None
            if not row or row[0] is None or row[1] is None or int(row[1]) <= int(row[0]):
                raise ProfilerUsageError(f"{db_path.name}: no TASK rows for --full-range")
            window = Window(int(row[0]), int(row[1]), label="full_range", source="full_range")
        host_events: list[dict[str, Any]] = []
        for table in _HOST_TABLES:
            if table not in tables:
                limitations.append({"code": "optional_table_missing", "detail": f"{table} is absent"})
                continue
            host_events.extend(
                _host_rows(con, table, strings, start_ns=window.start_ns, end_ns=window.end_ns, limit=max_rows)
            )
        pytorch_events = [event for event in host_events if event["table"] == "PYTORCH_API"]
        mstx_events = [event for event in host_events if event["table"] == "MSTX_EVENTS"]
        _attach_stacks(pytorch_events)
        _attach_stacks(mstx_events)
        cann_events = (
            _host_rows(con, "CANN_API", strings, start_ns=window.start_ns, end_ns=window.end_ns, limit=max_rows)
            if "CANN_API" in tables
            else []
        )
        _attach_stacks(cann_events)
        links = _connections(con)
        tasks = _task_rows(con, strings, start_ns=window.start_ns, end_ns=window.end_ns, limit=max_rows)
        compute_names = _compute_names(con, strings, [task["global_task_id"] for task in tasks if task["global_task_id"] is not None])
        occurrences = _task_occurrences(con, [task["global_task_id"] for task in tasks if task["global_task_id"] is not None])
        if kernel_name:
            try:
                pattern = re.compile(kernel_name, re.IGNORECASE)
            except re.error as exc:
                raise ProfilerUsageError(f"invalid --kernel-name pattern: {exc}") from exc
        else:
            pattern = None
        calls: list[dict[str, Any]] = []
        matched = 0
        selected = 0
        ambiguous: set[int] = set()
        dispatch_gaps: list[int] = []
        for task in tasks:
            name = compute_names.get(task["global_task_id"], "") if task["global_task_id"] is not None else ""
            task["kernel_name"] = name
            if pattern is not None and not pattern.search(name or ""):
                continue
            selected += 1
            cann, cann_basis = _find_cann(task, cann_events, links)
            pytorch, pytorch_basis = _find_pytorch(cann, pytorch_events, links, task)
            global_id = task["global_task_id"]
            count = occurrences.get(global_id, 0) if global_id is not None else 0
            task["global_task_id_occurrences"] = count
            task["ambiguous_global_task_id"] = count > 1
            if count > 1 and global_id is not None:
                ambiguous.add(global_id)
            if pytorch is not None or cann is not None:
                matched += 1
            # Runtime launch latency is measured from the CANN API end when it
            # exists; otherwise the enclosing PYTORCH_API end is the anchor.
            dispatch_host = cann or pytorch
            if dispatch_host is not None:
                gap_ns = max(0, task["start_ns"] - dispatch_host["end_ns"])
                dispatch_gaps.append(gap_ns)
            else:
                gap_ns = None
            if len(calls) < call_limit:
                calls.append(
                    {
                        "kernel_name": name or None,
                        "device_task": {key: value for key, value in task.items() if key != "kernel_name"},
                        "host_stack": _stack_path(pytorch),
                        "mstx_stack": _stack_path(
                            next(
                                (
                                    event
                                    for event in mstx_events
                                    if event["start_ns"] <= task["start_ns"] and event["end_ns"] >= task["end_ns"]
                                ),
                                None,
                            )
                        ),
                        "cann_api": None if cann is None else {
                            "name": cann["name"], "start_ns": cann["start_ns"], "end_ns": cann["end_ns"],
                            "duration_ms": _ms(cann["end_ns"] - cann["start_ns"]),
                        },
                        "association_basis": {
                            "cann": cann_basis,
                            "pytorch": pytorch_basis,
                        },
                        "dispatch": {
                            "host_dispatch_gap_ms": _ms(gap_ns),
                            "basis": "max(0, task.startNs - associated_host.endNs)" if gap_ns is not None else None,
                            "host_duration_ms": (
                                None
                                if dispatch_host is None
                                else _ms(dispatch_host["end_ns"] - dispatch_host["start_ns"])
                            ),
                            "anchor_table": None if dispatch_host is None else dispatch_host["table"],
                        },
                        "global_task_id_unique": count == 1,
                        "ambiguous_global_task_id": count > 1,
                    }
                )
        task_intervals = [(task["start_ns"], task["end_ns"]) for task in tasks]
        gaps = _gap_intervals(task_intervals, window.start_ns, window.end_ns)
        if min_gap_ms is not None:
            threshold = max(0.0, float(min_gap_ms)) * 1_000_000.0
            gaps = [gap for gap in gaps if (gap[1] - gap[0]) >= threshold]
        gap_rows = [
            {
                "start_ns": left,
                "end_ns": right,
                "device_uncovered_ms": _ms(right - left),
                "host_evidence": [
                    {
                        "table": event["table"],
                        "name": event["name"],
                        "depth": event.get("depth"),
                        "start_ns": event["start_ns"],
                        "end_ns": event["end_ns"],
                    }
                    for event in host_events
                    if event["end_ns"] >= left and event["start_ns"] <= right
                ][:8],
            }
            for left, right in gaps
        ]
        total = len(tasks)
        return {
            "schema_version": CORRELATION_SCHEMA,
            "root": str(Path(root).expanduser().resolve()),
            "rank_id": item.rank_id,
            "db": str(db_path),
            "window": {
                "start_ns": window.start_ns,
                "end_ns": window.end_ns,
                "window_duration_ms": _ms(window.span_ns),
                "label": window.label,
            },
            "filters": {"kernel_name": kernel_name, "min_gap_ms": min_gap_ms},
            "host": {
                "tables": {
                    "PYTORCH_API": len(pytorch_events),
                    "MSTX_EVENTS": len(mstx_events),
                    "CANN_API": len(cann_events),
                    "CONNECTION_IDS": len(links),
                    "TASK": total,
                },
                "host_event_count": len(host_events),
                "tasks_read": total,
                "selected_task_count": selected,
                "matched_task_count": matched,
                "unmatched_task_count": selected - matched,
                "unmatched_ratio": round((selected - matched) / selected, 6) if selected else None,
                "dispatch_gap_ms_p50": _ms(median(dispatch_gaps)) if dispatch_gaps else None,
                "dispatch_gap_ms_max": _ms(max(dispatch_gaps)) if dispatch_gaps else None,
                "dispatch_gap_basis": "associated host endNs to device TASK startNs; None when unmatched",
            },
            "correlated_calls": calls,
            "selected_task_count": selected,
            "calls_truncated": selected > len(calls),
            "device_gaps": gap_rows,
            "device_gap_scope": "explicit window edges included",
            "aclgraph": {
                "global_task_id_unique_required": True,
                "ambiguous_global_task_ids": sorted(ambiguous),
                "ambiguous_count": len(ambiguous),
                "policy": "globalTaskId is bound only when the full TASK table occurrence count is exactly 1",
            },
            "limitations": limitations,
            "query_elapsed_ms": round((time.perf_counter() - started) * 1000.0, 6),
        }
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise ProfilerQueryError(f"correlate exceeded the {timeout_s:.1f}s deadline") from exc
        raise ProfilerQueryError(f"correlate query failed: {exc}") from exc
    finally:
        con.set_progress_handler(None, 0)
        con.close()


__all__ = ["CORRELATION_SCHEMA", "correlate_profile"]
