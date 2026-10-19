"""Bounded sampled device-memory peak collection.

[Role]
  Sample torch_npu device/allocator memory with monotonic query intervals,
  explicit temporal/cadence coverage, and finite sampled peak scope.

[Invariants]
  * Only ``mem_get_info`` device used bytes form the device maximum; allocator
    ``max_*`` values remain allocator maxima and are never called device peaks.
  * Scope arithmetic uses one monotonic clock.  Absolute wall timestamps are
    recorded as evidence only, never used for cadence/coverage arithmetic.
  * The declared scope ends at the stop request.  A query ending after that
    boundary is recorded but excluded from target/cadence/max.  No final sample
    is taken after stop.
  * Slot assignment is the raw sample's declared ``slot_index``; summary
    never re-derives it from query start/elapsed time.  Missing/outside/
    duplicate indices and late queries remain visible.
  * Temporal span coverage and scheduled cadence completeness are separate.
    The reported ``coverage_ratio`` is their conservative minimum, also
    penalised by failed device/optional queries. Cadence punctuality is
    reported separately; a late query does not erase an observed slot.
  * A partial sampled maximum remains visible. Target acceptance requires all
    expected slots, no duplicate/failed slots, and all device/optional queries.
[Timeouts & Locks]
  Sampler thread uses a stop event, hard sample cap, and bounded join timeout.
  Optional allocator calls are synchronous, device-argument only, and
  error-captured.  No unit-sleep is injected by tests.

[Side-Effect Policy]
  Read-only runtime API calls.  ``reset_peak_memory_stats`` resets allocator
  peaks only; no filesystem/network writes.
"""

from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

SNAPSHOT_SCHEMA = "mindie-runtime-memory-snapshot/1"
SAMPLED_PEAK_SCHEMA = "mindie-sampled-device-memory-peak/1"
CLOCK_NAME = "time.monotonic_ns"
WALL_CLOCK_NAME = "time.time_ns"
COVERAGE_POLICY = "mindie-sampled-memory-coverage/2"


class MemoryEvidenceError(RuntimeError):
    """An explicitly unsupported or malformed evidence source."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_finite_positive(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 1.0 if numerator <= 0 else 0.0
    return max(0.0, min(1.0, numerator / denominator))


def _slot_indices(indices: list[int], limit: int = 32) -> list[int]:
    return sorted(set(indices))[:limit]


def summarize_sampling(
    samples: Sequence[Mapping[str, Any]],
    *,
    scope_start_ns: int,
    scope_end_ns: int,
    stop_requested_ns: int,
    interval_ms: float,
    max_samples: int,
    overflow: bool,
    thread_alive_at_stop: bool,
    device: int,
    reset: Mapping[str, Any],
    errors: Sequence[Mapping[str, Any]],
    observed_at: str,
) -> dict[str, Any]:
    """Summarize raw query observations against an explicit monotonic scope.

    This is pure and is the canonical contract used by the live sampler.  It
    intentionally has no sleeps or hardware access so temporal edge cases can
    be tested deterministically.
    """
    if not _is_finite_positive(interval_ms):
        raise MemoryEvidenceError("invalid_interval", f"interval_ms must be positive: {interval_ms!r}")
    if not isinstance(scope_start_ns, int) or not isinstance(scope_end_ns, int):
        raise MemoryEvidenceError("invalid_scope", "scope bounds must be integer monotonic ns")
    if scope_end_ns < scope_start_ns:
        raise MemoryEvidenceError("invalid_scope", "scope_end_ns must be >= scope_start_ns")
    interval_ns = max(1, int(round(float(interval_ms) * 1_000_000.0)))
    duration_ns = scope_end_ns - scope_start_ns
    expected_slots = ((duration_ns - 1) // interval_ns + 1) if duration_ns > 0 else 0
    expected_slot_set = set(range(expected_slots))
    late_tolerance_ns = max(1, interval_ns // 2)

    rows: list[dict[str, Any]] = []
    completed_slots: set[int] = set()
    late_slots: set[int] = set()
    duplicate_slot_indices: set[int] = set()
    outside_slot_indices: set[int] = set()
    duplicate_slots = 0
    missing_slot_index_sample_count = 0
    outside_slot_sample_count = 0
    successful_in_scope: list[dict[str, Any]] = []
    failed_in_scope = 0
    out_of_scope = 0
    for raw in samples:
        row = dict(raw)
        start = row.get("query_start_ns")
        end = row.get("query_end_ns")
        if not isinstance(start, int) or not isinstance(end, int):
            row["in_scope"] = False
            row["excluded_reason"] = "missing_query_interval"
            out_of_scope += 1
            rows.append(row)
            continue
        in_scope = start >= scope_start_ns and end <= scope_end_ns
        if not in_scope:
            out_of_scope += 1
            if start < scope_start_ns:
                row["excluded_reason"] = "query_started_before_scope"
            elif start >= scope_end_ns:
                row["excluded_reason"] = "query_started_after_scope_end"
            else:
                row["excluded_reason"] = "query_ended_after_scope_end"
            row["in_scope"] = False
            row["slot_target_ns"] = None
            row["late_ns"] = None
            row["duplicate_slot"] = False
            row["slot_index_status"] = "out_of_scope"
            rows.append(row)
            continue
        row["in_scope"] = True
        row["excluded_reason"] = None
        declared_slot = row.get("slot_index")
        if not isinstance(declared_slot, int) or isinstance(declared_slot, bool):
            row["declared_slot_index"] = declared_slot
            row["slot_target_ns"] = None
            row["late_ns"] = None
            row["duplicate_slot"] = False
            row["slot_index_status"] = "missing_slot_index"
            missing_slot_index_sample_count += 1
            if row.get("status") == "ok" and row.get("used_bytes") is not None:
                successful_in_scope.append(row)
            else:
                failed_in_scope += 1
            rows.append(row)
            continue
        row["declared_slot_index"] = declared_slot
        row["slot_index"] = declared_slot
        target = scope_start_ns + declared_slot * interval_ns
        row["slot_target_ns"] = target
        row["late_ns"] = max(0, start - target)
        if declared_slot < 0 or declared_slot >= expected_slots:
            row["slot_index_status"] = "outside_expected_slots"
            row["duplicate_slot"] = False
            outside_slot_indices.add(declared_slot)
            outside_slot_sample_count += 1
            if row.get("status") == "ok" and row.get("used_bytes") is not None:
                successful_in_scope.append(row)
            else:
                failed_in_scope += 1
            rows.append(row)
            continue
        row["slot_index_status"] = "declared"
        if row.get("status") == "ok" and row.get("used_bytes") is not None:
            duplicate = declared_slot in completed_slots
            row["duplicate_slot"] = duplicate
            if duplicate:
                duplicate_slots += 1
                duplicate_slot_indices.add(declared_slot)
            else:
                completed_slots.add(declared_slot)
                if row["late_ns"] > late_tolerance_ns:
                    late_slots.add(declared_slot)
            successful_in_scope.append(row)
        else:
            row["duplicate_slot"] = False
            failed_in_scope += 1
        rows.append(row)

    successful_in_scope.sort(key=lambda item: (int(item["query_start_ns"]), int(item["query_end_ns"])))
    used_values = [float(item["used_bytes"]) for item in successful_in_scope]
    peak_sample = max(successful_in_scope, key=lambda item: item["used_bytes"], default=None)
    query_starts = [int(item["query_start_ns"]) for item in successful_in_scope]
    query_ends = [int(item["query_end_ns"]) for item in successful_in_scope]
    if query_starts and query_ends:
        first_ns, last_ns = min(query_starts), max(query_ends)
        sample_window_ns = [first_ns, last_ns]
        temporal_span = max(0, last_ns - first_ns)
        temporal_ratio = _ratio(temporal_span, duration_ns) if duration_ns > 0 else 0.0
    else:
        sample_window_ns = [scope_start_ns, scope_end_ns]
        temporal_ratio = 0.0
    completed_count = len(completed_slots)
    cadence_ratio = _ratio(completed_count, expected_slots) if expected_slots else 0.0
    device_total = successful_in_scope.__len__() + failed_in_scope
    device_completeness = _ratio(successful_in_scope.__len__(), device_total) if device_total else 0.0
    allocator_ok = sum(int(item.get("allocator_ok_count") or 0) for item in successful_in_scope)
    allocator_attempted = sum(int(item.get("allocator_attempted_count") or 0) for item in successful_in_scope)
    allocator_completeness = _ratio(allocator_ok, allocator_attempted) if allocator_attempted else 1.0
    missing_slots = sorted(expected_slot_set - completed_slots)
    coverage_ratio = round(min(temporal_ratio, cadence_ratio, device_completeness, allocator_completeness), 6)
    target_coverage_complete = bool(
        expected_slots > 0
        and not missing_slots
        and duplicate_slots == 0
        and failed_in_scope == 0
        and temporal_ratio >= 0.99
        and cadence_ratio >= 0.99
        and allocator_completeness >= 1.0
        and missing_slot_index_sample_count == 0
        and outside_slot_sample_count == 0
        and not overflow
        and not thread_alive_at_stop
    )
    is_peak = bool(
        len(successful_in_scope) >= 2
        and temporal_ratio > 0
        and not thread_alive_at_stop
    )
    cadence_ms_values = [
        (right - left) / 1e6 for left, right in zip(query_starts, query_starts[1:]) if right > left
    ]
    max_allocated = [
        int(item["max_allocated_bytes"]) for item in successful_in_scope
        if item.get("max_allocated_bytes") is not None
    ]
    max_reserved = [
        int(item["max_reserved_bytes"]) for item in successful_in_scope
        if item.get("max_reserved_bytes") is not None
    ]
    cadence = {
        "requested": float(interval_ms),
        "count": len(cadence_ms_values),
        "min": min(cadence_ms_values) if cadence_ms_values else None,
        "p50": _percentile(cadence_ms_values, 0.5),
        "p95": _percentile(cadence_ms_values, 0.95),
        "max": max(cadence_ms_values) if cadence_ms_values else None,
        "expected_slots": expected_slots,
        "completed_slots": completed_count,
        "missing_slots": len(missing_slots),
        "late_slots": len(late_slots),
        "failed_samples": failed_in_scope,
        "successful_samples": len(successful_in_scope),
        "missing_slot_indices": _slot_indices(missing_slots),
        "late_slot_indices": _slot_indices(sorted(late_slots)),
        "outside_slot_indices": _slot_indices(sorted(outside_slot_indices)),
        "duplicate_slot_indices": _slot_indices(sorted(duplicate_slot_indices)),
        "missing_slot_index_sample_count": missing_slot_index_sample_count,
        "outside_slot_sample_count": outside_slot_sample_count,
        "duplicate_slot_sample_count": duplicate_slots,
        "target_coverage_complete": target_coverage_complete,
        "punctuality_ratio": _ratio(completed_count - len(late_slots), completed_count),
        "punctuality_complete": bool(completed_count and not late_slots),
    }
    summary = {
        "schema_version": SAMPLED_PEAK_SCHEMA,
        "coverage_policy": COVERAGE_POLICY,
        "status": "ok" if is_peak else "failed",
        "coverage_status": "complete" if target_coverage_complete else "partial",
        "phase": "sampled_peak",
        "device": device,
        "unit": "byte",
        "source": "bounded background sampler: torch_npu.npu.mem_get_info/memory_*",
        "source_kind": "torch_npu_runtime_api",
        "source_priority": "P1",
        "measurement_kind": "sampled_device_peak" if is_peak else "point_observation",
        "peak_scope": "sampled" if is_peak else "insufficient_samples",
        "continuous_peak": False,
        "absolute_peak_limited": True,
        "is_peak": is_peak,
        "scope": f"device:{device}",
        "observed_at": observed_at,
        "clock": CLOCK_NAME,
        "wall_clock": WALL_CLOCK_NAME,
        "scope_start_ns": scope_start_ns,
        "scope_end_ns": scope_end_ns,
        "stop_requested_ns": stop_requested_ns,
        "scope_clock": CLOCK_NAME,
        "sample_window_ns": sample_window_ns,
        "sample_window_clock": CLOCK_NAME,
        "peak_sample": dict(peak_sample) if peak_sample is not None else None,
        "peak_sample_selection": "first successful in-scope query with maximum device used bytes",
        "interval_ms_requested": float(interval_ms),
        "max_samples": int(max_samples),
        "sample_count": len(rows),
        "successful_sample_count": len(successful_in_scope),
        "failed_sample_count": failed_in_scope,
        "in_scope_sample_count": len(successful_in_scope) + failed_in_scope,
        "out_of_scope_sample_count": out_of_scope,
        "duplicate_slot_sample_count": duplicate_slots,
        "missing_slot_index_sample_count": missing_slot_index_sample_count,
        "outside_slot_sample_count": outside_slot_sample_count,
        "slot_assignment": "raw_sample.slot_index",
        "samples_truncated": bool(overflow),
        "thread_alive_at_stop": bool(thread_alive_at_stop),
        "expected_slot_count": expected_slots,
        "completed_slot_count": completed_count,
        "missing_slot_count": len(missing_slots),
        "late_slot_count": len(late_slots),
        "missing_slot_indices": _slot_indices(missing_slots),
        "late_slot_indices": _slot_indices(sorted(late_slots)),
        "outside_slot_indices": _slot_indices(sorted(outside_slot_indices)),
        "duplicate_slot_indices": _slot_indices(sorted(duplicate_slot_indices)),
        "temporal_span_coverage_ratio": round(temporal_ratio, 6),
        "cadence_completeness_ratio": round(cadence_ratio, 6),
        "device_sample_completeness_ratio": round(device_completeness, 6),
        "optional_getter_completeness_ratio": round(allocator_completeness, 6),
        "coverage_ratio": coverage_ratio,
        "coverage_ratio_definition": "min(temporal_span, cadence_completeness, device_sample_completeness, optional_getter_completeness)",
        "target_coverage_complete": target_coverage_complete,
        "cadence_ms": cadence,
        "device_peak": {
            "used_bytes_max": max(used_values) if used_values else None,
            "used_bytes_min": min(used_values) if used_values else None,
            "used_bytes_last": used_values[-1] if used_values else None,
            "sample_count": len(used_values),
            "slot_count": completed_count,
            "absolute_peak_limited": True,
            "sampling_kind": "bounded_background_samples",
        },
        "allocator_peak": {
            "max_allocated_bytes_observed": max(max_allocated) if max_allocated else None,
            "max_reserved_bytes_observed": max(max_reserved) if max_reserved else None,
            "allocator_peak_reset": dict(reset),
            "allocator_max_scope": "since_reset" if reset.get("ok") else "since_process_start_or_last_reset",
            "failed_allocator_query_count": max(0, allocator_attempted - allocator_ok),
            "allocator_ok_query_count": allocator_ok,
            "allocator_attempted_query_count": allocator_attempted,
            "not_device_peak": True,
        },
        "samples": rows,
        "errors": [dict(item) for item in errors],
    }
    return summary


class BoundedMemorySampler:
    """Bounded background sampler for whole-device memory observations.

    Scope arithmetic, cadence slots, and coverage all use ``clock_ns``
    (default ``time.monotonic_ns``).  ``wall_clock_ns`` is evidence-only.
    """

    def __init__(
        self,
        torch_npu: Any,
        *,
        device: int = 0,
        interval_ms: float = 2.0,
        max_samples: int = 512,
        stop_timeout_s: float = 2.0,
        clock_ns: Any = time.monotonic_ns,
        wall_clock_ns: Any = time.time_ns,
    ) -> None:
        if not isinstance(device, int) or isinstance(device, bool) or device < 0:
            raise MemoryEvidenceError("invalid_device", f"device must be a non-negative integer: {device!r}")
        if not _is_finite_positive(interval_ms):
            raise MemoryEvidenceError("invalid_interval", f"interval_ms must be positive: {interval_ms!r}")
        if not isinstance(max_samples, int) or max_samples <= 1:
            raise MemoryEvidenceError("invalid_max_samples", "max_samples must be an integer > 1")
        if not callable(clock_ns) or not callable(wall_clock_ns):
            raise MemoryEvidenceError("invalid_clock", "clock functions must be callable")
        self.torch_npu = torch_npu
        self.device = device
        self.interval_ms = float(interval_ms)
        self.max_samples = int(max_samples)
        self.stop_timeout_s = float(stop_timeout_s)
        self.clock_ns = clock_ns
        self.wall_clock_ns = wall_clock_ns
        self._interval_ns = max(1, int(round(self.interval_ms * 1_000_000.0)))
        self._samples: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._scope_start_ns: int | None = None
        self._scope_end_ns: int | None = None
        self._stop_requested_ns: int | None = None
        self._overflow = False
        self._thread_alive = False
        self._reset: dict[str, Any] = {"attempted": False, "ok": False, "api": "torch_npu.npu.reset_peak_memory_stats"}
        self.summary: dict[str, Any] | None = None

    def _npu(self) -> Any:
        npu = getattr(self.torch_npu, "npu", None)
        if npu is None:
            raise MemoryEvidenceError("torch_npu_missing", "module has no torch_npu.npu attribute")
        return npu

    def _record_error(self, code: str, message: str) -> None:
        if len(self._errors) < 16:
            self._errors.append({"code": code, "message": message, "clock_ns": int(self.clock_ns())})

    def _call_optional(self, npu: Any, name: str) -> tuple[int | None, bool, bool]:
        fn = getattr(npu, name, None)
        if not callable(fn):
            return None, False, False
        try:
            value = fn(self.device)
            return int(value), True, True
        except Exception as exc:
            self._record_error("allocator_api_failed", f"{name}: {type(exc).__name__}: {exc}")
            return None, True, False

    def _sample(self, slot_index: int) -> None:
        if len(self._samples) >= self.max_samples:
            self._overflow = True
            self._stop.set()
            return
        query_start_ns = int(self.clock_ns())
        wall_ns = int(self.wall_clock_ns())
        npu = self._npu()
        error = None
        used_bytes = None
        free_bytes = None
        total_bytes = None
        try:
            free_bytes, total_bytes = npu.mem_get_info(self.device)
            free_bytes, total_bytes = int(free_bytes), int(total_bytes)
            used_bytes = total_bytes - free_bytes
        except Exception as exc:
            error = {"code": "mem_get_info_failed", "message": f"{type(exc).__name__}: {exc}"}
            self._record_error(error["code"], error["message"])
        values: dict[str, tuple[int | None, bool, bool]] = {}
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
            values[name] = self._call_optional(npu, name)
        wall_end_ns = int(self.wall_clock_ns())
        query_end_ns = int(self.clock_ns())
        sample = {
            "slot_index": int(slot_index),
            "query_start_ns": query_start_ns,
            "query_end_ns": query_end_ns,
            "wall_ns": wall_ns,
            "wall_end_ns": wall_end_ns,
            "status": "ok" if error is None and used_bytes is not None else "error",
            "error": error,
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
            "allocated_bytes": values["memory_allocated"][0],
            "reserved_bytes": values["memory_reserved"][0],
            "max_allocated_bytes": values["max_memory_allocated"][0],
            "max_reserved_bytes": values["max_memory_reserved"][0],
            "allocator_ok_count": sum(1 for _value, _attempted, ok in values.values() if ok),
            "allocator_attempted_count": sum(1 for _value, attempted, _ok in values.values() if attempted),
        }
        self._samples.append(sample)

    def _run(self) -> None:
        scope_start_ns = int(self._scope_start_ns or self.clock_ns())
        next_slot_ns = scope_start_ns
        last_slot = -1
        while not self._stop.is_set():
            now_ns = int(self.clock_ns())
            if now_ns < next_slot_ns:
                if self._stop.wait((next_slot_ns - now_ns) / 1e9):
                    break
                continue
            slot_index = max(last_slot + 1, int((now_ns - scope_start_ns) // self._interval_ns))
            last_slot = slot_index
            self._sample(slot_index)
            if self._overflow:
                break
            next_slot_ns = scope_start_ns + (slot_index + 1) * self._interval_ns

    def start(self) -> "BoundedMemorySampler":
        if self._thread is not None:
            raise MemoryEvidenceError("sampler_already_started", "sampler.start() called twice")
        npu = self._npu()
        reset = getattr(npu, "reset_peak_memory_stats", None)
        if callable(reset):
            self._reset["attempted"] = True
            try:
                reset(self.device)
                self._reset["ok"] = True
            except Exception as exc:
                self._reset["error"] = f"{type(exc).__name__}: {exc}"
                self._record_error("allocator_peak_reset_failed", self._reset["error"])
        else:
            self._reset["error"] = "reset_peak_memory_stats is not callable"
        self._scope_start_ns = int(self.clock_ns())
        self._thread = threading.Thread(target=self._run, name="mindie-memory-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        if self._thread is None or self._scope_start_ns is None:
            raise MemoryEvidenceError("sampler_not_started", "sampler.stop() called before start()")
        self._stop_requested_ns = int(self.clock_ns())
        self._scope_end_ns = self._stop_requested_ns
        self._stop.set()
        self._thread.join(timeout=max(0.1, self.stop_timeout_s))
        self._thread_alive = self._thread.is_alive()
        summary = summarize_sampling(
            list(self._samples),
            scope_start_ns=self._scope_start_ns,
            scope_end_ns=self._scope_end_ns,
            stop_requested_ns=self._stop_requested_ns,
            interval_ms=self.interval_ms,
            max_samples=self.max_samples,
            overflow=self._overflow,
            thread_alive_at_stop=self._thread_alive,
            device=self.device,
            reset=self._reset,
            errors=self._errors,
            observed_at=_utc_now(),
        )
        self.summary = summary
        return summary

    def __enter__(self) -> "BoundedMemorySampler":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.stop()
        return False


def sampled_peak_to_snapshot(peak: Mapping[str, Any]) -> dict[str, Any]:
    """Convert sampler output into a manifest snapshot; no continuous claim."""
    used = peak.get("device_peak", {}).get("used_bytes_max") if isinstance(peak.get("device_peak"), Mapping) else None
    is_peak = bool(peak.get("is_peak")) and peak.get("measurement_kind") == "sampled_device_peak"
    if not _finite_number(used) or not is_peak:
        return {
            "schema_version": SNAPSHOT_SCHEMA,
            "phase": "sampled_peak",
            "device": peak.get("device"),
            "error": "sampled device peak missing",
            "measurement_kind": "point_observation",
            "is_peak": False,
            "peak_scope": "insufficient_samples",
            "source_kind": "torch_npu_runtime_api",
        }
    cadence = peak.get("cadence_ms") if isinstance(peak.get("cadence_ms"), Mapping) else {}
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "phase": "sampled_peak",
        "device": peak.get("device"),
        "unit": "byte",
        "source": peak.get("source"),
        "source_kind": "torch_npu_runtime_api",
        "source_priority": "P1",
        "measurement_kind": "sampled_device_peak",
        "is_peak": True,
        "peak_scope": "sampled",
        "continuous_peak": False,
        "absolute_peak_limited": True,
        "scope": peak.get("scope") or f"device:{peak.get('device')}",
        "observed_at": peak.get("observed_at"),
        "device_memory": {"used_bytes": float(used)},
        "sample_count": peak.get("sample_count"),
        "sample_window_ns": peak.get("sample_window_ns"),
        "sample_window_clock": peak.get("sample_window_clock"),
        "coverage_policy": peak.get("coverage_policy"),
        "peak_sample": peak.get("peak_sample"),
        "peak_sample_selection": peak.get("peak_sample_selection"),
        "coverage_ratio": peak.get("coverage_ratio"),
        "temporal_span_coverage_ratio": peak.get("temporal_span_coverage_ratio"),
        "cadence_completeness_ratio": peak.get("cadence_completeness_ratio"),
        "optional_getter_completeness_ratio": peak.get("optional_getter_completeness_ratio"),
        "target_coverage_complete": peak.get("target_coverage_complete"),
        "coverage_status": peak.get("coverage_status"),
        "expected_slot_count": peak.get("expected_slot_count"),
        "completed_slot_count": peak.get("completed_slot_count"),
        "missing_slot_count": peak.get("missing_slot_count"),
        "late_slot_count": peak.get("late_slot_count"),
        "failed_sample_count": peak.get("failed_sample_count"),
        "missing_slot_index_sample_count": peak.get("missing_slot_index_sample_count"),
        "outside_slot_sample_count": peak.get("outside_slot_sample_count"),
        "slot_assignment": peak.get("slot_assignment"),
        "cadence_ms": dict(cadence),
        "allocator_max_scope": peak.get("allocator_peak", {}).get("allocator_max_scope") if isinstance(peak.get("allocator_peak"), Mapping) else None,
    }


__all__ = [
    "BoundedMemorySampler",
    "CLOCK_NAME",
    "MemoryEvidenceError",
    "SAMPLED_PEAK_SCHEMA",
    "WALL_CLOCK_NAME",
    "sampled_peak_to_snapshot",
    "summarize_sampling",
]
