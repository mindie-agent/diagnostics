"""Steps-file and DB-clock evidence derived only from real capture records.

[Role]
Turn ``framework_capture`` per-step wall/monotonic records into the explicit
caller step file consumed by ``segment.load_steps_file`` and the compare
acceptance driver.  Also inspect the exported kernel DB's actual timestamp
range and declare whether it aligns with the capture's wall interval.

[Invariants]
- Step IDs come only from the caller-declared capture range and the active
  step records; they are never inferred from kernel rows, CSV order, or names.
- Every step file timestamp is an integer ``time.time_ns`` value with a
  matching epoch-float ``*_ts`` field.  ``time.perf_counter_ns`` bounds are
  retained separately and are never used as epoch time.
- Kernel DB alignment is checked against the actual DB ``TASK.startNs/endNs``
  range.  No offset is guessed, added, or fabricated; unknown clocks are
  reported as unknown instead of being forced to match.
- Only a successful capture with active records can produce a step file.

[Timeouts & Locks]
The DB read is a bounded read-only SQLite query with ``busy_timeout``.  No
network, subprocess, lock, or sleep.

[Side-Effect Policy]
Reads the exported DB and writes only the caller-requested steps JSON.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "mindie-steps-evidence/1"
DB_CLOCK_SCHEMA = "mindie-kernel-db-clock/1"
ALIGNMENT_SCHEMA = "mindie-capture-db-alignment/1"
EPOCH_NS_MIN = 1_500_000_000_000_000_000
EPOCH_NS_MAX = 4_000_000_000_000_000_000


class CollectorEvidenceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _active_records(capture: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if capture.get("status") != "ok":
        raise CollectorEvidenceError("capture_not_ok",
                                     f"capture status is {capture.get('status')!r}; export is not valid")
    records = capture.get("step_records")
    if not isinstance(records, list):
        raise CollectorEvidenceError("step_records_missing", "capture has no step_records list")
    active = [item for item in records if isinstance(item, Mapping) and item.get("segment") == "active"]
    if not active:
        raise CollectorEvidenceError("active_records_missing", "capture has no active step records")
    plan = capture.get("plan")
    capture_plan = plan.get("capture") if isinstance(plan, Mapping) else None
    expected = capture_plan.get("captured_step_ids") if isinstance(capture_plan, Mapping) else None
    if not isinstance(expected, list) or not all(isinstance(item, int) for item in expected):
        raise CollectorEvidenceError("captured_ids_missing", "capture plan has no explicit captured_step_ids")
    actual = [item.get("step") for item in active]
    if actual != expected:
        raise CollectorEvidenceError("captured_ids_mismatch",
                                     f"active records {actual!r} do not equal declared capture {expected!r}")
    for item in active:
        for key in ("wall_start_ns", "wall_end_ns", "wall_start_ts", "wall_end_ts",
                    "perf_start_ns", "perf_end_ns", "duration_ms"):
            if key not in item:
                raise CollectorEvidenceError("step_field_missing", f"step {item.get('step')} lacks {key}")
        if int(item["wall_end_ns"]) <= int(item["wall_start_ns"]):
            raise CollectorEvidenceError("step_interval_invalid", f"step {item.get('step')} has non-positive wall bounds")
        if int(item["perf_end_ns"]) <= int(item["perf_start_ns"]):
            raise CollectorEvidenceError("step_interval_invalid", f"step {item.get('step')} has non-positive perf bounds")
    return sorted(active, key=lambda item: item["step"])


def build_steps_document(capture: Mapping[str, Any], *, clock_domain: str = "unix_wall") -> dict[str, Any]:
    """Build ``mindie-steps`` JSON with explicit IDs and actual invocation bounds."""
    active = _active_records(capture)
    if not isinstance(clock_domain, str) or not clock_domain.strip():
        raise ValueError("clock_domain must be non-empty")
    steps = []
    for item in active:
        steps.append({
            "step": int(item["step"]),
            "start": int(item["wall_start_ns"]),
            "end": int(item["wall_end_ns"]),
            "start_ns": int(item["wall_start_ns"]),
            "end_ns": int(item["wall_end_ns"]),
            "start_ts": float(item["wall_start_ts"]),
            "end_ts": float(item["wall_end_ts"]),
            "perf_start_ns": int(item["perf_start_ns"]),
            "perf_end_ns": int(item["perf_end_ns"]),
            "duration_ms": float(item["duration_ms"]),
            "clock_domain": clock_domain,
            "identity_source": "framework_capture.step_records",
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "time_unit": "ns",
        "clock_domain": clock_domain,
        "capture_wall": {
            "start_ns": capture.get("capture_wall_start_ns"),
            "end_ns": capture.get("capture_wall_end_ns"),
            "start_ts": capture.get("capture_wall_start_ts"),
            "end_ts": capture.get("capture_wall_end_ts"),
        },
        "capture_active_wall": {
            "start_ns": capture.get("active_wall_start_ns"),
            "end_ns": capture.get("active_wall_end_ns"),
            "start_ts": capture.get("active_wall_start_ts"),
            "end_ts": capture.get("active_wall_end_ts"),
        },
        "source": "caller_declared_capture_active_records",
        "steps": steps,
    }


def write_steps_file(path: str | Path, capture: Mapping[str, Any], *, clock_domain: str = "unix_wall") -> dict[str, Any]:
    document = build_steps_document(capture, clock_domain=clock_domain)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
    return document


def find_capture_db(capture: Mapping[str, Any]) -> Path | None:
    artifacts = capture.get("artifacts")
    entries = artifacts.get("artifacts") if isinstance(artifacts, Mapping) else None
    if not isinstance(entries, list):
        return None
    candidates = [
        item for item in entries
        if isinstance(item, Mapping) and item.get("kind") == "db"
        and item.get("non_empty") and item.get("path")
    ]
    if not candidates:
        return None
    chosen = max(candidates, key=lambda item: item.get("modified_ns") or 0)
    return Path(str(chosen["path"]))


def inspect_kernel_db_clock_range(
    db_path: str | Path, *, busy_timeout_ms: int = 5000,
    active_steps: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read the exported DB's actual TASK timestamp range without guessing."""
    path = Path(db_path)
    if not path.is_file():
        raise CollectorEvidenceError("db_missing", f"kernel DB not found: {path}")
    uri = f"file:{path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=busy_timeout_ms / 1000.0)
    try:
        connection.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        connection.execute("PRAGMA query_only=ON")
        try:
            row = connection.execute("SELECT MIN(startNs), MAX(endNs), COUNT(*) FROM TASK").fetchone()
            step_observations = []
            for step in active_steps or []:
                start_ns, end_ns = step["wall_start_ns"], step["wall_end_ns"]
                count = connection.execute(
                    "SELECT COUNT(*) FROM TASK WHERE startNs < ? AND endNs > ?",
                    (end_ns, start_ns),
                ).fetchone()[0]
                step_observations.append({
                    "step": step["step"], "start_ns": start_ns, "end_ns": end_ns,
                    "task_overlap_count": int(count),
                })
        except sqlite3.Error as exc:
            raise CollectorEvidenceError("db_clock_query_failed", f"TASK timestamp query failed: {exc}") from exc
    finally:
        connection.close()
    if not row or row[0] is None or row[1] is None:
        return {
            "schema_version": DB_CLOCK_SCHEMA, "db_path": str(path), "table": "TASK",
            "min_ns": None, "max_ns": None, "row_count": 0,
            "clock_domain": "unknown", "status": "unavailable",
        }
    minimum, maximum, count = int(row[0]), int(row[1]), int(row[2])
    domain = "unix_epoch_ns" if EPOCH_NS_MIN <= minimum <= EPOCH_NS_MAX and EPOCH_NS_MIN <= maximum <= EPOCH_NS_MAX else "unknown_non_epoch_ns"
    return {
        "schema_version": DB_CLOCK_SCHEMA,
        "db_path": str(path),
        "table": "TASK",
        "min_ns": minimum,
        "max_ns": maximum,
        "row_count": count,
        "active_step_observations": step_observations,
        "clock_domain": domain,
        "status": "ok" if domain == "unix_epoch_ns" else "unknown_clock_domain",
    }


def align_capture_to_db(
    capture: Mapping[str, Any], db_clock: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Declare overlap/containment between capture wall bounds and DB timestamps."""
    active_start = capture.get("active_wall_start_ns")
    active_end = capture.get("active_wall_end_ns")
    capture_start = capture.get("capture_wall_start_ns")
    capture_end = capture.get("capture_wall_end_ns")
    base = {
        "schema_version": ALIGNMENT_SCHEMA,
        "capture_clock_domain": "unix_wall",
        "capture_wall_start_ns": capture_start,
        "capture_wall_end_ns": capture_end,
        "active_wall_start_ns": active_start,
        "active_wall_end_ns": active_end,
        "db_clock_domain": None,
        "db_min_ns": None,
        "db_max_ns": None,
        "offset_applied_ns": 0,
        "offset_guessed": False,
        "status": "unavailable",
    }
    if not isinstance(db_clock, Mapping) or db_clock.get("status") != "ok":
        base["reason"] = "kernel DB clock is unavailable or non-epoch"
        return base
    if not all(isinstance(value, int) and not isinstance(value, bool)
               for value in (active_start, active_end, capture_start, capture_end)):
        base["reason"] = "capture wall bounds are incomplete"
        return base
    db_min, db_max = int(db_clock["min_ns"]), int(db_clock["max_ns"])
    base.update({
        "db_clock_domain": db_clock.get("clock_domain"),
        "db_min_ns": db_min,
        "db_max_ns": db_max,
        "db_contains_active": db_min <= active_start and db_max >= active_end,
        "db_contains_total_capture": db_min <= capture_start and db_max >= capture_end,
        "capture_contains_db": capture_start <= db_min and db_max <= capture_end,
        "overlap_total_capture_ns": max(0, min(db_max, capture_end) - max(db_min, capture_start)),
        "overlap_active_ns": max(0, min(db_max, active_end) - max(db_min, active_start)),
    })
    observations = db_clock.get("active_step_observations") or []
    active_records = [row for row in capture.get("step_records", []) if row.get("segment") == "active"]
    observed_windows = [(row["step"], row["start_ns"], row["end_ns"]) for row in observations]
    actual_windows = [(row["step"], row["wall_start_ns"], row["wall_end_ns"]) for row in active_records]
    base["active_step_observations"] = observations
    base["every_active_step_has_device_tasks"] = bool(
        actual_windows and observed_windows == actual_windows
        and all(row["task_overlap_count"] > 0 for row in observations)
    )
    if base["capture_contains_db"] and base["every_active_step_has_device_tasks"]:
        base["status"] = "aligned"
        base["reason"] = "epoch DB is inside capture and has tasks in every caller step; no offset applied"
    elif base["overlap_total_capture_ns"] > 0:
        base["status"] = "partial_overlap"
        base["reason"] = "capture overlaps DB but per-step task or total capture containment evidence is incomplete"
    else:
        base["status"] = "no_overlap"
        base["reason"] = "DB TASK range does not overlap capture wall bounds"
    return base


__all__ = [
    "CollectorEvidenceError",
    "align_capture_to_db",
    "build_steps_document",
    "find_capture_db",
    "inspect_kernel_db_clock_range",
    "write_steps_file",
]
