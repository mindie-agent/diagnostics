"""Real memory evidence collection and msprof/msmemscope parsing.

[Role]
Capture allocator and whole-device HBM values from the installed torch_npu
runtime, and parse the documented msprof ``npu_module_mem`` / ``npu_mem`` and
msmemscope snapshot CSV artifacts.  ``memory_profile`` consumes the resulting
values; this module never estimates a missing measurement.

[Invariants]
- Allocator bytes, reserved bytes, peak bytes, device free/total bytes, and
  the derived device-used-minus-allocator value are kept as separate fields.
  A derived difference is never called a measured driver value.
- Runtime API point observations and msprof sampled module maxima are
  distinct source kinds/measurement kinds.  A P1 priority never relabels a
  torch_npu API observation as npu-smi.
- SQLite ``NPU_MODULE_MEM`` resolves ``moduleId`` through ``ENUM_MODULE`` and
  preserves event timestamps, device scope, byte units, and sampled-max
  semantics.  Unknown module ids are explicit unmapped evidence, not guessed.
- A missing phase, column, API, or snapshot stays missing.  No zero is
  substituted and no residual is clamped.
- msprof ``(KB)``/``(MB)`` columns are read as binary units, matching the
  artifact's established arithmetic; the chosen column is recorded.
- torch/torch_npu are imported by the caller and passed in, so importing this
  module never requires NPU hardware or a torch installation.

[Timeouts & Locks]
No subprocess, network, lock, sleep, or unbounded read.  CSV reads are bounded
by ``max_rows``; runtime probe calls are direct and synchronous.

[Side-Effect Policy]
CSV and snapshot parsers are read-only.  ``collect_runtime_memory_snapshot``
calls only allocator/query APIs and an optional synchronize.  The optional
msmemscope call writes to msmemscope's already-configured output directory.
"""

from __future__ import annotations

import csv
import hashlib
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from .memory_db import load_memory_record_sqlite, load_npu_mem_sqlite, load_op_memory_sqlite
from .memory_sampler import (
    BoundedMemorySampler,
    MemoryEvidenceError,
    SAMPLED_PEAK_SCHEMA,
    SNAPSHOT_SCHEMA,
    sampled_peak_to_snapshot,
)

SCHEMA_VERSION = "mindie-memory-evidence/1"
MODULE_MEM_SCHEMA = "mindie-msprof-module-mem/1"
MODULE_MEM_SQLITE_SCHEMA = "mindie-msprof-module-mem-sqlite/1"
NPU_MEM_SCHEMA = "mindie-msprof-npu-mem/1"
MSMEMSCOPE_SCHEMA = "mindie-msmemscope-snapshot/1"
DEFAULT_MAX_CSV_ROWS = 2_000_000
_ATTR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^,}]+)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _required_npu_attr(npu: Any, name: str) -> Any:
    value = getattr(npu, name, None)
    if not callable(value):
        raise MemoryEvidenceError(
            "torch_npu_api_missing",
            f"torch_npu.npu.{name} is required for allocator/device evidence",
        )
    return value


def collect_runtime_memory_snapshot(
    torch_npu: Any,
    *,
    device: int = 0,
    phase: str = "unspecified",
    synchronize: bool = True,
) -> dict[str, Any]:
    """Read real allocator and driver-reported device memory at one moment.

    ``torch_npu.npu.mem_get_info(device)`` is called with the installed API's
    ``(free, total)`` return order.  The derived device-used value is preserved
    as-is even if it is negative.
    """
    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        raise MemoryEvidenceError("invalid_device", f"device must be a non-negative integer: {device!r}")
    npu = getattr(torch_npu, "npu", None)
    if npu is None:
        raise MemoryEvidenceError("torch_npu_missing", "module has no torch_npu.npu attribute")
    if synchronize:
        _required_npu_attr(npu, "synchronize")()
    allocated = int(_required_npu_attr(npu, "memory_allocated")(device))
    reserved = int(_required_npu_attr(npu, "memory_reserved")(device))
    max_allocated = int(_required_npu_attr(npu, "max_memory_allocated")(device))
    max_reserved = int(_required_npu_attr(npu, "max_memory_reserved")(device))
    free_bytes, total_bytes = _required_npu_attr(npu, "mem_get_info")(device)
    free_bytes, total_bytes = int(free_bytes), int(total_bytes)
    device_used = total_bytes - free_bytes
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "phase": str(phase),
        "device": device,
        "unit": "byte",
        "source": "torch_npu.npu.memory_allocated/memory_reserved/max_*/mem_get_info",
        "source_kind": "torch_npu_runtime_api",
        "source_priority": "P1",
        "measurement_kind": "point_observation",
        "scope": f"device:{device}",
        "is_peak": False,
        "allocator_max_kind": "allocator_max_since_process_start",
        "point_semantics": "single runtime API observation; allocator max_* fields are allocator maxima, not device peaks",
        "observed_at": _utc_now(),
        "allocator": {
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "max_allocated_bytes": max_allocated,
            "max_reserved_bytes": max_reserved,
        },
        "device_memory": {
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "used_bytes": device_used,
        },
        "derived": {
            "reserved_minus_allocated_bytes": reserved - allocated,
            "device_used_minus_allocator_allocated_bytes": device_used - allocated,
            "device_used_minus_allocator_reserved_bytes": device_used - reserved,
            "note": "derived difference; not a direct driver measurement",
        },
    }


def _read_csv_rows(path: str | Path, *, max_rows: int) -> tuple[list[str], list[dict[str, str]]]:
    if not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows: list[dict[str, str]] = []
        for index, row in enumerate(reader):
            if index >= max_rows:
                raise MemoryEvidenceError("csv_row_bound", f"CSV exceeds {max_rows} rows: {path}")
            rows.append({
                str(key): ("" if value is None else str(value).strip())
                for key, value in row.items() if key is not None
            })
    return fields, rows


def _device_key(row: Mapping[str, str]) -> str:
    value = (row.get("Device_id") or row.get("Device ID") or row.get("Device Id") or "unknown").strip()
    return value or "unknown"


def parse_npu_module_mem_csv(
    path: str | Path, *, max_rows: int = DEFAULT_MAX_CSV_ROWS
) -> dict[str, Any]:
    """Parse msprof ``npu_module_mem_*.csv`` component peaks per device."""
    fields, rows = _read_csv_rows(path, max_rows=max_rows)
    if "Component" not in fields:
        raise MemoryEvidenceError("csv_column_missing", "npu_module_mem CSV has no Component column")
    if "Total Reserved(MB)" in fields:
        column, factor, unit = "Total Reserved(MB)", 1024 ** 2, "MB"
    elif "Total Reserved(KB)" in fields:
        column, factor, unit = "Total Reserved(KB)", 1024, "KB"
    else:
        raise MemoryEvidenceError(
            "csv_column_missing",
            "npu_module_mem CSV has neither Total Reserved(MB) nor Total Reserved(KB)",
        )
    devices: dict[str, dict[str, dict[str, Any]]] = {}
    invalid: list[dict[str, Any]] = []
    for line, row in enumerate(rows, start=2):
        component = (row.get("Component") or "").strip().upper()
        if not component:
            continue
        raw = row.get(column, "")
        if raw == "":
            invalid.append({"line": line, "component": component, "reason": "empty_value"})
            continue
        try:
            value = float(raw) * factor
        except ValueError:
            invalid.append({"line": line, "component": component, "raw": raw, "reason": "invalid_value"})
            continue
        if value < 0:
            invalid.append({"line": line, "component": component, "raw": raw, "reason": "negative_value"})
            continue
        device = _device_key(row)
        bucket = devices.setdefault(device, {}).setdefault(
            component, {"max_bytes": 0.0, "samples": 0, "last_timestamp_us": None}
        )
        bucket["max_bytes"] = max(bucket["max_bytes"], value)
        bucket["samples"] += 1
        timestamp = row.get("Timestamp(us)")
        if timestamp:
            try:
                bucket["last_timestamp_us"] = float(timestamp)
            except ValueError:
                invalid.append({"line": line, "component": component, "raw": timestamp, "reason": "invalid_timestamp"})
    return {
        "schema_version": MODULE_MEM_SCHEMA,
        "source_path": str(path),
        "source": "msprof npu_module_mem CSV",
        "source_kind": "msprof_npu_module_mem_csv",
        "measurement_kind": "sampled_max",
        "unit": "byte",
        "source_unit": unit,
        "unit_basis": "binary",
        "devices": devices,
        "invalid_rows": invalid,
        "complete": not invalid,
        "rows": len(rows),
    }


def parse_npu_mem_csv(
    path: str | Path, *, max_rows: int = DEFAULT_MAX_CSV_ROWS
) -> dict[str, Any]:
    """Parse msprof ``npu_mem_*.csv`` device/APP HBM peaks per device."""
    fields, rows = _read_csv_rows(path, max_rows=max_rows)
    if "event" not in fields or "hbm(KB)" not in fields:
        raise MemoryEvidenceError("csv_column_missing", "npu_mem CSV needs event and hbm(KB) columns")
    devices: dict[str, dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    for line, row in enumerate(rows, start=2):
        event = (row.get("event") or "").strip().upper()
        raw = row.get("hbm(KB)", "")
        if not event:
            continue
        try:
            value = float(raw) * 1024
        except ValueError:
            invalid.append({"line": line, "event": event, "raw": raw, "reason": "invalid_value"})
            continue
        if value < 0:
            invalid.append({"line": line, "event": event, "raw": raw, "reason": "negative_value"})
            continue
        device = _device_key(row)
        bucket = devices.setdefault(device, {"device_bytes": None, "app_bytes": None, "samples": 0})
        bucket["samples"] += 1
        if event == "DEVICE":
            bucket["device_bytes"] = max(bucket["device_bytes"] or 0.0, value)
        elif event == "APP":
            bucket["app_bytes"] = max(bucket["app_bytes"] or 0.0, value)
    for bucket in devices.values():
        if bucket["device_bytes"] is not None and bucket["app_bytes"] is not None:
            bucket["device_minus_app_bytes"] = bucket["device_bytes"] - bucket["app_bytes"]
        else:
            bucket["device_minus_app_bytes"] = None
    return {
        "schema_version": NPU_MEM_SCHEMA,
        "source_path": str(path),
        "source": "msprof npu_mem CSV",
        "unit": "byte",
        "unit_basis": "binary",
        "devices": devices,
        "invalid_rows": invalid,
        "complete": not invalid,
        "rows": len(rows),
    }


def _parse_attr(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in _ATTR_RE.findall(text or ""):
        result[key] = value.strip()
    return result


def parse_msmemscope_snapshot_csv(
    path: str | Path, *, max_rows: int = DEFAULT_MAX_CSV_ROWS
) -> dict[str, Any]:
    """Parse documented msmemscope ``SNAPSHOT`` rows (allocator and device)."""
    fields, rows = _read_csv_rows(path, max_rows=max_rows)
    if "Event" not in fields or "Attr" not in fields:
        raise MemoryEvidenceError("csv_column_missing", "msmemscope CSV needs Event and Attr columns")
    snapshots: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for line, row in enumerate(rows, start=2):
        if (row.get("Event") or "").strip().upper() != "SNAPSHOT":
            continue
        attr = _parse_attr(row.get("Attr") or "")
        if not attr:
            invalid.append({"line": line, "reason": "empty_attr"})
            continue
        converted: dict[str, Any] = {}
        for key in ("total_mem", "free_mem", "reserved", "peak_reserved",
                    "allocated", "peak_allocated", "device_utilization", "pt_utilization"):
            if key not in attr:
                continue
            try:
                converted[key] = float(attr[key])
            except ValueError:
                invalid.append({"line": line, "key": key, "raw": attr[key], "reason": "invalid_value"})
        device = _device_key(row)
        snapshot: dict[str, Any] = {
            "device": device,
            "timestamp_ns": row.get("Timestamp(ns)"),
            "name": row.get("Name") or "",
            "attr_raw": attr,
            "values": converted,
        }
        if "total_mem" in converted and "free_mem" in converted:
            snapshot["device_used_bytes"] = converted["total_mem"] - converted["free_mem"]
        if "allocated" in converted:
            snapshot["allocator_allocated_bytes"] = converted["allocated"]
        if "reserved" in converted:
            snapshot["allocator_reserved_bytes"] = converted["reserved"]
        snapshots.append(snapshot)
    return {
        "schema_version": MSMEMSCOPE_SCHEMA,
        "source_path": str(path),
        "source": "msmemscope SNAPSHOT CSV",
        "unit": "byte",
        "unit_basis": "documented Attr byte values",
        "snapshots": snapshots,
        "invalid_rows": invalid,
        "complete": not invalid,
        "rows": len(rows),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_only_connection(path: Path, timeout_s: float) -> sqlite3.Connection:
    if not path.is_file():
        raise MemoryEvidenceError("sqlite_file_missing", f"DB not found: {path}")
    uri = "file:" + quote(str(path.resolve()), safe="/") + "?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True, timeout=max(0.1, timeout_s))
    except sqlite3.Error as exc:
        raise MemoryEvidenceError("sqlite_open_failed", f"cannot open DB read-only: {exc}") from exc
    con.execute("PRAGMA query_only = ON")
    con.execute(f"PRAGMA busy_timeout = {int(max(1, timeout_s) * 1000)}")
    return con


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})")}


def load_npu_module_mem_sqlite(
    path: str | Path,
    *,
    max_rows: int = DEFAULT_MAX_CSV_ROWS,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    """Load real ``NPU_MODULE_MEM`` component samples from an Ascend DB.

    The installed 1.2.0 schema stores ``moduleId`` as an ``ENUM_MODULE`` FK,
    ``timestampNs`` in nanoseconds, and ``totalReserved`` in bytes (validated
    against the same-run APP/RUNTIME allocator values).  The values are
    sampled maxima over the capture window, not whole-workload peaks.
    """

    if not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")
    db_path = Path(path)
    started = time.perf_counter()
    con = _read_only_connection(db_path, timeout_s)
    deadline = started + max(0.1, timeout_s)
    con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 20_000)
    try:
        present = {str(row[0]) for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"NPU_MODULE_MEM": {"moduleId", "timestampNs", "totalReserved", "deviceId"},
                    "ENUM_MODULE": {"id", "name"}}
        missing: dict[str, list[str]] = {}
        for table, columns in required.items():
            if table not in present:
                missing[table] = sorted(columns)
                continue
            absent = sorted(columns - _table_columns(con, table))
            if absent:
                missing[table] = absent
        if missing:
            detail = ", ".join(f"{table}({','.join(cols)})" for table, cols in sorted(missing.items()))
            raise MemoryEvidenceError("sqlite_schema_missing", f"NPU_MODULE_MEM schema missing: {detail}")
        modules = {int(row[0]): str(row[1]) for row in con.execute("SELECT id, name FROM ENUM_MODULE")}
        rows = con.execute(
            "SELECT moduleId, timestampNs, totalReserved, deviceId FROM NPU_MODULE_MEM "
            "ORDER BY deviceId, timestampNs, moduleId LIMIT ?",
            (max_rows + 1,),
        ).fetchall()
        if len(rows) > max_rows:
            raise MemoryEvidenceError("sqlite_row_bound", f"NPU_MODULE_MEM exceeds {max_rows} rows: {db_path}")
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise MemoryEvidenceError("sqlite_timeout", f"NPU_MODULE_MEM read exceeded {timeout_s:.1f}s") from exc
        raise MemoryEvidenceError("sqlite_read_failed", f"NPU_MODULE_MEM read failed: {exc}") from exc
    except sqlite3.Error as exc:
        raise MemoryEvidenceError("sqlite_read_failed", f"NPU_MODULE_MEM read failed: {exc}") from exc
    finally:
        con.set_progress_handler(None, 0)
        con.close()

    devices: dict[str, dict[str, dict[str, Any]]] = {}
    invalid: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []
    timestamps: list[int] = []
    for line, (module_id, timestamp_ns, total_reserved, device_id) in enumerate(rows, start=2):
        if module_id is None or total_reserved is None or timestamp_ns is None or device_id is None:
            invalid.append({"line": line, "reason": "missing_required_value"})
            continue
        try:
            module_id_int = int(module_id)
            timestamp_int = int(timestamp_ns)
            value = float(total_reserved)
            device_key = str(int(device_id))
        except (TypeError, ValueError):
            invalid.append({"line": line, "reason": "invalid_numeric_value"})
            continue
        if value < 0:
            invalid.append({"line": line, "reason": "negative_value", "raw": total_reserved})
            continue
        component = modules.get(module_id_int)
        if component is None:
            component = f"MODULE_{module_id_int}"
            unmapped.append({"line": line, "module_id": module_id_int})
        bucket = devices.setdefault(device_key, {}).setdefault(
            component,
            {
                "max_bytes": 0.0,
                "min_bytes": None,
                "samples": 0,
                "first_timestamp_ns": timestamp_int,
                "last_timestamp_ns": timestamp_int,
                "measurement_kind": "sampled_max",
                "module_id": module_id_int,
                "component": component,
            },
        )
        bucket["max_bytes"] = max(float(bucket["max_bytes"]), value)
        bucket["min_bytes"] = value if bucket["min_bytes"] is None else min(float(bucket["min_bytes"]), value)
        bucket["samples"] += 1
        bucket["first_timestamp_ns"] = min(int(bucket["first_timestamp_ns"]), timestamp_int)
        bucket["last_timestamp_ns"] = max(int(bucket["last_timestamp_ns"]), timestamp_int)
        timestamps.append(timestamp_int)
    return {
        "schema_version": MODULE_MEM_SQLITE_SCHEMA,
        "source_path": str(db_path),
        "source_sha256": _sha256_file(db_path),
        "source": "msprof NPU_MODULE_MEM SQLite",
        "source_kind": "msprof_npu_module_mem_sqlite",
        "unit": "byte",
        "source_unit": "database numeric",
        "unit_basis": "database values are bytes; the actual APP sample matches same-run allocator reserved bytes",
        "measurement_kind": "sampled_max",
        "devices": devices,
        "invalid_rows": invalid,
        "unmapped_modules": unmapped,
        "unmapped_module_ids": sorted({int(item["module_id"]) for item in unmapped}),
        "complete": not invalid and not unmapped,
        "rows": len(rows),
        "timestamp_range_ns": [min(timestamps), max(timestamps)] if timestamps else None,
        "query_elapsed_ms": round((time.perf_counter() - started) * 1000.0, 6),
    }


def msprof_components_to_manifest(module_mem: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Convert parsed module-memory samples to the attribution manifest shape.

    ``measurement_kind`` and sample/timestamp provenance are preserved so a
    sampled maximum is never silently promoted to a measured whole-run peak.
    """
    result: dict[str, dict[str, dict[str, Any]]] = {}
    devices = module_mem.get("devices")
    if not isinstance(devices, Mapping):
        return result
    for device, components in devices.items():
        if not isinstance(components, Mapping):
            continue
        result[str(device)] = {}
        for component, values in components.items():
            if not isinstance(values, Mapping) or "max_bytes" not in values:
                continue
            payload: dict[str, Any] = {
                "bytes": float(values["max_bytes"]),
                "measurement_kind": str(values.get("measurement_kind") or "sampled_max"),
            }
            for key in ("samples", "first_timestamp_ns", "last_timestamp_ns",
                        "last_timestamp_us", "module_id", "component"):
                if key in values:
                    payload[key] = values[key]
            result[str(device)][str(component)] = payload
    return result


def _snapshot_source_kind(snapshot: Mapping[str, Any]) -> str:
    explicit = snapshot.get("source_kind")
    if explicit:
        return str(explicit)
    source = str(snapshot.get("source") or "").lower()
    if "torch_npu" in source or "mem_get_info" in source:
        return "torch_npu_runtime_api"
    if "npu-smi" in source or "npu_smi" in source:
        return "npu_smi"
    return "unknown_runtime_source"


def _snapshot_metadata(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    device = snapshot.get("device", "0")
    return {
        "source_kind": _snapshot_source_kind(snapshot),
        "source_priority": str(snapshot.get("source_priority") or "P1"),
        "measurement_kind": str(snapshot.get("measurement_kind") or "point_observation"),
        "is_peak": bool(snapshot.get("is_peak", False)),
        "peak_scope": snapshot.get("peak_scope"),
        "continuous_peak": snapshot.get("continuous_peak"),
        "absolute_peak_limited": snapshot.get("absolute_peak_limited"),
        "sample_count": snapshot.get("sample_count"),
        "sample_window_ns": snapshot.get("sample_window_ns"),
        "sample_window_clock": snapshot.get("sample_window_clock"),
        "peak_sample": snapshot.get("peak_sample"),
        "peak_sample_selection": snapshot.get("peak_sample_selection"),
        "allocator": snapshot.get("allocator"),
        "query_bounds": snapshot.get("query_bounds"),
        "target_coverage_complete": snapshot.get("target_coverage_complete"),
        "coverage_policy": snapshot.get("coverage_policy"),
        "coverage_ratio": snapshot.get("coverage_ratio"),
        "cadence_ms": snapshot.get("cadence_ms"),
        "scope": str(snapshot.get("scope") or f"device:{device}"),
        "allocator_max_kind": (snapshot.get("allocator_max_kind")
                               or ("allocator_max_since_process_start"
                                   if _snapshot_source_kind(snapshot) == "torch_npu_runtime_api" else None)),
        "point_semantics": (snapshot.get("point_semantics")
                            or "single point observation; not a measured whole-device peak"),
    }


def manifest_from_snapshots(
    snapshots: Mapping[str, Mapping[str, Any]],
    *,
    tp: int = 1,
    dp: int = 1,
    enable_expert_parallel: bool = False,
    msprof_components_by_device: Mapping[str, Any] | None = None,
    npu_mem_by_device: Mapping[str, Any] | None = None,
    op_memory_by_device: Mapping[str, Any] | None = None,
    vllm_info: Mapping[str, Any] | None = None,
    weight_manifest: Mapping[str, Any] | None = None,
    model_config: Mapping[str, Any] | None = None,
    kv_cache_applicable: bool | None = None,
) -> dict[str, Any]:
    """Build an attribution manifest from real phase snapshots.

    Phases are explicit point observations.  A phase without a device-used
    value is omitted rather than filled with zero.  The original runtime API
    source/measurement kind is preserved; it is never relabeled as npu-smi.
    """
    phase_map = {
        "baseline": "baseline_hbm",
        "ready": "after_ready_hbm",
        "after_ready": "after_ready_hbm",
        "inference_peak": "after_infer_hbm",
        "after_infer": "after_infer_hbm",
        "sampled_peak": "sampled_peak_hbm",
    }
    manifest: dict[str, Any] = {
        "schema_version": "mindie-memory-observation/1",
        "tp": tp,
        "dp": dp,
        "enable_expert_parallel": bool(enable_expert_parallel),
        "kv_cache_applicable": kv_cache_applicable,
        "architecture": {
            "kv_cache_applicable": kv_cache_applicable,
            "source": "caller_explicit" if kv_cache_applicable is not None else "unknown",
        },
        "source_snapshots": {
            str(phase): {
                "device": snapshot.get("device"),
                "observed_at": snapshot.get("observed_at"),
                "source": snapshot.get("source"),
                **_snapshot_metadata(snapshot),
            }
            for phase, snapshot in snapshots.items() if isinstance(snapshot, Mapping)
        },
        "limitations": [],
    }
    for phase, key in phase_map.items():
        snapshot = snapshots.get(phase)
        if not isinstance(snapshot, Mapping):
            continue
        device_memory = snapshot.get("device_memory")
        used = device_memory.get("used_bytes") if isinstance(device_memory, Mapping) else None
        if not isinstance(used, (int, float)) or isinstance(used, bool):
            manifest["limitations"].append({
                "code": "device_used_missing",
                "phase": phase,
                "message": "snapshot has no device_memory.used_bytes; phase omitted",
            })
            continue
        device = str(snapshot.get("device", "0"))
        manifest.setdefault(key, {})[device] = {
            "used_bytes": float(used),
            "observed_at": snapshot.get("observed_at"),
            "source": snapshot.get("source"),
            "phase": phase,
            **_snapshot_metadata(snapshot),
        }
    if msprof_components_by_device:
        manifest["msprof_components_by_device"] = dict(msprof_components_by_device)
    if npu_mem_by_device:
        manifest["npu_mem_by_device"] = dict(npu_mem_by_device)
    if op_memory_by_device:
        manifest["op_memory_by_device"] = dict(op_memory_by_device)
    if vllm_info:
        manifest["vllm_info"] = dict(vllm_info)
    if weight_manifest:
        manifest["weight_manifest"] = dict(weight_manifest)
    if model_config:
        manifest["model_config"] = dict(model_config)
    return manifest


def take_msmemscope_snapshot(device: int = 0, name: str = "mindie", *, module: Any = None) -> dict[str, Any]:
    """Call msmemscope's documented snapshot API if it is installed.

    The caller owns msmemscope configuration/output.  This function returns an
    explicit unavailable error instead of silently skipping an enabled probe.
    """
    if module is None:
        try:
            import msmemscope  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MemoryEvidenceError("msmemscope_missing", "msmemscope is not installed") from exc
        module = msmemscope
    take = getattr(module, "take_snapshot", None)
    if not callable(take):
        raise MemoryEvidenceError("msmemscope_api_missing", "msmemscope.take_snapshot is missing")
    take(device, name=name)
    return {"ok": True, "device": device, "name": name, "observed_at": _utc_now()}


__all__ = [
    "MemoryEvidenceError",
    "BoundedMemorySampler",
    "SAMPLED_PEAK_SCHEMA",
    "collect_runtime_memory_snapshot",
    "load_memory_record_sqlite",
    "load_npu_mem_sqlite",
    "load_npu_module_mem_sqlite",
    "load_op_memory_sqlite",
    "manifest_from_snapshots",
    "msprof_components_to_manifest",
    "parse_msmemscope_snapshot_csv",
    "parse_npu_mem_csv",
    "parse_npu_module_mem_csv",
    "sampled_peak_to_snapshot",
    "take_msmemscope_snapshot",
]
