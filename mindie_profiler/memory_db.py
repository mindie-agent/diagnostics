"""Read-only msprof memory DB loaders for device/app NPU_MEM evidence.

[Role]
  Load the exporter's ``NPU_MEM`` app/device series without interpreting a
  magnitude as a unit or merging app-reserved and device-observed scopes.

[Invariants]
  * SQLite is opened read-only/query-only with a busy timeout.
  * ``NPU_MEM.hbm``/``ddr`` are byte values as exported.  The unit basis is
    recorded and cross-checked against the matching ``memory_record.csv`` MB
    conversion; it is never inferred from the numeric magnitude.
  * ``type`` joins the capture's ``STRING_IDS`` app/device names; numeric
    dictionary ids are local to each export, not a fixed enum.
  * App and device series stay separate.  APP is observed profiler app-scope
    reserved, not proven identical to the PyTorch allocator.  Device is observed
    profiler device scope, not proven physical whole-device.  A sampled maximum
    is not an absolute/continuous peak; no cadence is invented from min/max.
  * Unknown type ids or invalid rows make ``complete=false``; rows are not
    silently relabeled.

[Timeouts & Locks]
  Read-only connection with busy timeout; bounded row count.

[Side-Effect Policy]
  No filesystem/network writes.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from .memory_sampler import MemoryEvidenceError

NPU_MEM_SCHEMA_VERSION = "mindie-msprof-npu-mem-sqlite/1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")]


def _p50(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def load_npu_mem_sqlite(
    path: str | Path,
    *,
    max_rows: int = 1_000_000,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Load read-only NPU_MEM app/device byte series from a captured DB."""
    db_path = Path(path)
    if not db_path.is_file():
        raise MemoryEvidenceError("db_missing", f"NPU_MEM DB not found: {db_path}")
    if not isinstance(max_rows, int) or max_rows <= 0:
        raise MemoryEvidenceError("invalid_max_rows", "max_rows must be a positive integer")
    uri = f"file:{db_path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=max(0.1, float(timeout_s)))
    try:
        connection.execute(f"PRAGMA busy_timeout={int(max(0.1, float(timeout_s)) * 1000)}")
        connection.execute("PRAGMA query_only=ON")
        tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "NPU_MEM" not in tables or "STRING_IDS" not in tables:
            raise MemoryEvidenceError("npu_mem_schema_missing", "NPU_MEM and STRING_IDS tables are required")
        npu_columns = set(_columns(connection, "NPU_MEM"))
        required = {"type", "ddr", "hbm", "timestampNs", "deviceId"}
        if not required.issubset(npu_columns):
            raise MemoryEvidenceError("npu_mem_schema_invalid", f"NPU_MEM columns must include {sorted(required)}")
        string_columns = set(_columns(connection, "STRING_IDS"))
        if not {"id", "value"}.issubset(string_columns):
            raise MemoryEvidenceError("string_ids_schema_invalid", "STRING_IDS columns must include id/value")
        type_names = {int(row[0]): str(row[1]) for row in connection.execute("SELECT id, value FROM STRING_IDS")}
        rows = connection.execute(
            "SELECT type, ddr, hbm, timestampNs, deviceId FROM NPU_MEM ORDER BY timestampNs, deviceId LIMIT ?",
            (int(max_rows) + 1,),
        ).fetchall()
    finally:
        connection.close()
    if len(rows) > max_rows:
        raise MemoryEvidenceError("npu_mem_row_limit", f"NPU_MEM exceeds max_rows={max_rows}")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    invalid_rows: list[dict[str, Any]] = []
    unknown_type_ids: set[int] = set()
    for row_number, (type_id, ddr, hbm, timestamp_ns, device_id) in enumerate(rows, start=1):
        try:
            type_int = int(type_id)
            ddr_value = float(ddr)
            hbm_value = float(hbm)
            timestamp_int = int(timestamp_ns)
            device_int = int(device_id)
        except (TypeError, ValueError):
            invalid_rows.append({"row": row_number, "reason": "non_numeric_required_field"})
            continue
        if ddr_value < 0 or hbm_value < 0 or timestamp_int < 0 or device_int < 0:
            invalid_rows.append({"row": row_number, "reason": "negative_required_field"})
            continue
        scope = type_names.get(type_int)
        if scope not in {"app", "device"}:
            unknown_type_ids.add(type_int)
            invalid_rows.append({"row": row_number, "reason": "unknown_type_id", "type": type_int})
            continue
        grouped.setdefault(str(device_int), {}).setdefault(scope, []).append({
            "type_id": type_int,
            "ddr_bytes": ddr_value,
            "hbm_bytes": hbm_value,
            "timestamp_ns": timestamp_int,
            "device_id": device_int,
        })

    devices: dict[str, Any] = {}
    for device_key, scopes in sorted(grouped.items()):
        device_entry: dict[str, Any] = {}
        for scope in ("app", "device"):
            series = sorted(scopes.get(scope, []), key=lambda item: item["timestamp_ns"])
            if not series:
                continue
            hbm = [item["hbm_bytes"] for item in series]
            ddr = [item["ddr_bytes"] for item in series]
            timestamps = [item["timestamp_ns"] for item in series]
            intervals = [right - left for left, right in zip(timestamps, timestamps[1:]) if right > left]
            device_entry[scope] = {
                "type_id": series[0]["type_id"],
                "scope": "app_reserved" if scope == "app" else "device_observed",
                "scope_claim": "profiler_app_scope_reserved" if scope == "app" else "profiler_device_scope_observed",
                "pytorch_allocator_identity_proven": False if scope == "app" else None,
                "physical_whole_device_proven": None if scope == "app" else False,
                "unit": "byte",
                "measurement_kind": "sampled_max",
                "is_peak": scope == "device",
                "peak_scope": "sampled" if scope == "device" else "allocator_sample",
                "continuous_peak": False,
                "absolute_peak_limited": True,
                "coverage_status": "unknown",
                "schedule_known": False,
                "max_bytes": max(hbm),
                "min_bytes": min(hbm),
                "last_bytes": hbm[-1],
                "samples": len(series),
                "first_timestamp_ns": timestamps[0],
                "last_timestamp_ns": timestamps[-1],
                "ddr_max_bytes": max(ddr),
                "ddr_min_bytes": min(ddr),
                "observed_intervals_ns": {
                    "count": len(intervals),
                    "min": min(intervals) if intervals else None,
                    "p50": _p50(intervals),
                    "max": max(intervals) if intervals else None,
                },
            }
        if device_entry:
            devices[device_key] = device_entry

    all_timestamps = [item["timestamp_ns"] for scopes in grouped.values() for series in scopes.values() for item in series]
    complete = bool(rows) and not invalid_rows and not unknown_type_ids
    return {
        "schema_version": NPU_MEM_SCHEMA_VERSION,
        "source_kind": "msprof_npu_mem_sqlite",
        "source_path": str(db_path),
        "source_sha256": _sha256_file(db_path),
        "rows": len(rows),
        "complete": complete,
        "unit": "byte",
        "unit_basis": (
            "NPU_MEM.hbm/ddr exporter bytes; actual memory_record.csv APP "
            "Total Reserved(MB) x 1048576 exactly matches NPU_MEM APP hbm at the same timestamp"
        ),
        "type_names": {str(key): value for key, value in type_names.items() if value in {"app", "device"}},
        "scope_names": {str(key): value for key, value in type_names.items() if value in {"app", "device"}},
        "unknown_type_ids": sorted(unknown_type_ids),
        "invalid_rows": invalid_rows,
        "timestamp_range_ns": [min(all_timestamps), max(all_timestamps)] if all_timestamps else None,
        "devices": devices,
        "limitations": [
            {"code": "npu_mem_schedule_unknown", "message": "NPU_MEM declares no intended cadence/slot schedule; observed intervals are evidence, not coverage", "required": True},
            {"code": "npu_mem_scope_separation", "message": "app is observed profiler app-scope reserved (not proven PyTorch allocator); device is observed profiler device scope (not proven physical whole-device); neither is an additive baseline/component", "required": True},
        ] if rows else [],
    }


def load_memory_record_sqlite(path: str | Path, *, max_rows: int = 1_000_000, timeout_s: float = 5.0) -> dict[str, Any]:
    """Load MEMORY_RECORD byte totals with exact device/component partitions."""
    db_path = Path(path)
    if not db_path.is_file():
        raise MemoryEvidenceError("db_missing", f"MEMORY_RECORD DB not found: {db_path}")
    if not isinstance(max_rows, int) or max_rows <= 0:
        raise MemoryEvidenceError("invalid_max_rows", "max_rows must be a positive integer")
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True, timeout=max(0.1, float(timeout_s)))
    try:
        connection.execute(f"PRAGMA busy_timeout={int(max(0.1, float(timeout_s)) * 1000)}")
        connection.execute("PRAGMA query_only=ON")
        tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "MEMORY_RECORD" not in tables or "STRING_IDS" not in tables:
            raise MemoryEvidenceError("memory_record_schema_missing", "MEMORY_RECORD and STRING_IDS are required")
        names = {int(row[0]): str(row[1]) for row in connection.execute("SELECT id, value FROM STRING_IDS")}
        rows = connection.execute(
            "SELECT component, timestamp, totalAllocated, totalReserved, totalActive, streamPtr, deviceId "
            "FROM MEMORY_RECORD ORDER BY deviceId, timestamp, component LIMIT ?",
            (int(max_rows) + 1,),
        ).fetchall()
    finally:
        connection.close()
    if len(rows) > max_rows:
        raise MemoryEvidenceError("memory_record_row_limit", f"MEMORY_RECORD exceeds max_rows={max_rows}")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    unknown: set[int] = set()
    invalid: list[dict[str, Any]] = []
    for row_number, (component, timestamp, allocated, reserved, active, stream, device) in enumerate(rows, start=1):
        try:
            component_id = int(component)
            item = {
                "component_id": component_id,
                "timestamp_ns": int(timestamp), "total_allocated_bytes": int(allocated),
                "total_reserved_bytes": int(reserved), "total_active_bytes": int(active),
                "stream_ptr": int(stream) if stream is not None else None, "device_id": int(device),
            }
        except (TypeError, ValueError):
            invalid.append({"row": row_number, "reason": "non_numeric_required_field"})
            continue
        name = names.get(component_id)
        if name is None:
            unknown.add(component_id)
            invalid.append({"row": row_number, "reason": "unknown_component_id", "component": component_id})
            continue
        grouped.setdefault(str(item["device_id"]), {}).setdefault(name, []).append(item)

    devices: dict[str, Any] = {}
    rows_by_device: dict[str, int] = {}
    for device_key, component_map in sorted(grouped.items()):
        components: dict[str, Any] = {}
        device_timestamps: list[int] = []
        for name, series in sorted(component_map.items()):
            series.sort(key=lambda item: item["timestamp_ns"])
            alloc = [item["total_allocated_bytes"] for item in series]
            res = [item["total_reserved_bytes"] for item in series]
            active = [item["total_active_bytes"] for item in series]
            timestamps = [item["timestamp_ns"] for item in series]
            device_timestamps.extend(timestamps)
            components[name] = {
                "component_id": series[0]["component_id"], "samples": len(series),
                "first_timestamp_ns": timestamps[0], "last_timestamp_ns": timestamps[-1],
                "min_total_allocated_bytes": min(alloc), "max_total_allocated_bytes": max(alloc),
                "min_total_reserved_bytes": min(res), "max_total_reserved_bytes": max(res),
                "min_total_active_bytes": min(active), "max_total_active_bytes": max(active),
                "total_allocated_bytes": alloc, "total_reserved_bytes": res,
                "total_active_bytes": active,
                "timestamps_ns": timestamps,
            }
        devices[device_key] = {
            "components": components,
            "rows": sum(item["samples"] for item in components.values()),
            "timestamp_range_ns": [min(device_timestamps), max(device_timestamps)] if device_timestamps else None,
        }
        rows_by_device[device_key] = devices[device_key]["rows"]
    return {
        "schema_version": "mindie-msprof-memory-record-sqlite/2",
        "source_kind": "msprof_memory_record_sqlite", "source_path": str(db_path),
        "source_sha256": _sha256_file(db_path), "rows": len(rows), "unit": "byte",
        "unit_basis": "MEMORY_RECORD byte totals; actual memory_record.csv Total Reserved(MB) x 1048576 matches",
        "complete": bool(rows) and not invalid,
        "unknown_component_ids": sorted(unknown), "invalid_rows": invalid,
        "device_ids": sorted(devices), "rows_by_device": rows_by_device,
        "devices": devices,
        "scope_note": "Each device is returned independently; PTA/PTA+GE duplicate views must not be added across devices/components.",
    }


def load_op_memory_sqlite(path: str | Path, *, max_rows: int = 1_000_000, timeout_s: float = 5.0,
                          min_large_bytes: int = 1 << 30) -> dict[str, Any]:
    """Load OP_MEMORY rows with exact device-local identities and lifetimes."""
    db_path = Path(path)
    if not db_path.is_file():
        raise MemoryEvidenceError("db_missing", f"OP_MEMORY DB not found: {db_path}")
    if not isinstance(max_rows, int) or max_rows <= 0:
        raise MemoryEvidenceError("invalid_max_rows", "max_rows must be a positive integer")
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True, timeout=max(0.1, float(timeout_s)))
    try:
        connection.execute(f"PRAGMA busy_timeout={int(max(0.1, float(timeout_s)) * 1000)}")
        connection.execute("PRAGMA query_only=ON")
        tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "OP_MEMORY" not in tables or "STRING_IDS" not in tables:
            raise MemoryEvidenceError("op_memory_schema_missing", "OP_MEMORY and STRING_IDS are required")
        names = {int(row[0]): str(row[1]) for row in connection.execute("SELECT id, value FROM STRING_IDS")}
        rows = connection.execute(
            "SELECT name, size, allocationTime, releaseTime, activeReleaseTime, duration, activeDuration, "
            "allocationTotalAllocated, allocationTotalReserved, allocationTotalActive, "
            "releaseTotalAllocated, releaseTotalReserved, releaseTotalActive, streamPtr, deviceId "
            "FROM OP_MEMORY ORDER BY deviceId, allocationTime, size DESC LIMIT ?",
            (int(max_rows) + 1,),
        ).fetchall()
    finally:
        connection.close()
    if len(rows) > max_rows:
        raise MemoryEvidenceError("op_memory_row_limit", f"OP_MEMORY exceeds max_rows={max_rows}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    unknown: set[int] = set()
    invalid: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        try:
            (name_id, size, alloc, release, active_release, duration, active_duration,
             alloc_total, alloc_res, alloc_active, release_total, release_res, release_active,
             stream, device) = row
            name = names.get(int(name_id))
            if name is None:
                unknown.add(int(name_id))
                invalid.append({"row": row_number, "reason": "unknown_operator_id", "operator": int(name_id)})
                continue
            observations = grouped.setdefault(str(int(device)), [])
            observations.append({
                "name": name, "name_id": int(name_id), "size_bytes": int(size),
                "allocation_time_ns": int(alloc),
                "release_time_ns": int(release) if release is not None else None,
                "active_release_time_ns": int(active_release) if active_release is not None else None,
                "duration_ns": int(duration) if duration is not None else None,
                "active_duration_ns": int(active_duration) if active_duration is not None else None,
                "allocation_total_allocated_bytes": int(alloc_total) if alloc_total is not None else None,
                "allocation_total_reserved_bytes": int(alloc_res) if alloc_res is not None else None,
                "allocation_total_active_bytes": int(alloc_active) if alloc_active is not None else None,
                "release_total_allocated_bytes": int(release_total) if release_total is not None else None,
                "release_total_reserved_bytes": int(release_res) if release_res is not None else None,
                "release_total_active_bytes": int(release_active) if release_active is not None else None,
                "stream_ptr": int(stream) if stream is not None else None, "device_id": int(device),
            })
        except (TypeError, ValueError):
            invalid.append({"row": row_number, "reason": "non_numeric_required_field"})

    devices: dict[str, Any] = {}
    rows_by_device: dict[str, int] = {}
    for device_key, observations in sorted(grouped.items()):
        by_name: dict[str, Any] = {}
        for item in observations:
            entry = by_name.setdefault(item["name"], {
                "name_id": item["name_id"], "count": 0, "max_size_bytes": 0, "sum_size_bytes": 0,
                "max_allocation_total_allocated_bytes": None,
            })
            entry["count"] += 1
            entry["max_size_bytes"] = max(entry["max_size_bytes"], item["size_bytes"])
            entry["sum_size_bytes"] += item["size_bytes"]
            if item["allocation_total_allocated_bytes"] is not None:
                entry["max_allocation_total_allocated_bytes"] = max(
                    entry["max_allocation_total_allocated_bytes"] or 0, item["allocation_total_allocated_bytes"])
        large = [item for item in observations if item["size_bytes"] >= min_large_bytes]
        largest = max(observations, key=lambda item: (item["size_bytes"], item["allocation_time_ns"]), default=None)
        timestamps = [t for item in observations for t in (item["allocation_time_ns"], item["release_time_ns"]) if t is not None]
        devices[device_key] = {
            "rows": len(observations), "largest_allocation": largest,
            "large_allocations": large, "large_allocation_count": len(large), "by_name": by_name,
            "timestamp_range_ns": [min(timestamps), max(timestamps)] if timestamps else None,
        }
        rows_by_device[device_key] = len(observations)
    return {
        "schema_version": "mindie-msprof-op-memory-sqlite/2",
        "source_kind": "msprof_op_memory_sqlite", "source_path": str(db_path),
        "source_sha256": _sha256_file(db_path), "rows": len(rows), "unit": "byte",
        "unit_basis": "OP_MEMORY size/cumulative totals are bytes; actual operator_memory.csv Size(KB) x 1024 matches",
        "complete": bool(rows) and not invalid,
        "unknown_operator_ids": sorted(unknown), "invalid_rows": invalid,
        "device_ids": sorted(devices), "rows_by_device": rows_by_device, "devices": devices,
        "semantics": "transient operator allocation; attention/workspace semantics are not proven by the operator name alone",
        "workspace_semantics_proven": False,
    }


__all__ = ["NPU_MEM_SCHEMA_VERSION", "load_npu_mem_sqlite", "load_memory_record_sqlite", "load_op_memory_sqlite"]
