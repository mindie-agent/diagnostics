"""Memory decomposition anchored to the selected device-peak query.

[Role]
  Keep allocator observations paired with their device observation and exclude
  the baseline allocator reservation before adding a later reservation.
[Invariants]
  Only the selected query's allocated/reserved values are used. Allocator
  maxima and unrelated profiler maxima cannot replace that query. The
  baseline difference is observed non-allocator memory, not driver identity.
  Runtime API calls are sequential within a query interval, not atomic.
[Timeouts & Locks]
  Pure bounded mapping arithmetic; no locks or clocks are read.
[Side-Effect Policy]
  No I/O or mutation of inputs.
"""

from __future__ import annotations

from typing import Any, Mapping

from .memory_units import _is_number


def peak_query(stats: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(stats, Mapping):
        return None
    row = stats.get("peak_sample")
    if not isinstance(row, Mapping) or row.get("status") != "ok" or not row.get("in_scope"):
        return None
    if row.get("used_bytes") != stats.get("used_bytes"):
        return None
    return row


def peak_wall_interval(stats: Mapping[str, Any] | None) -> tuple[int, int] | None:
    """Return only measured wall-clock bounds, never convert monotonic values."""
    row = peak_query(stats)
    if row is None:
        return None
    start, end = row.get("wall_ns"), row.get("wall_end_ns")
    return (start, end) if isinstance(start, int) and isinstance(end, int) and end >= start else None


def baseline_and_allocator_candidates(
    baseline: Mapping[str, Any] | None, target: Mapping[str, Any] | None,
    *, baseline_bytes: float | None, device_id: int, candidate: Any,
    source_kind: str, measurement_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if baseline_bytes is None or not isinstance(baseline, Mapping):
        return [], []
    base_allocator = baseline.get("allocator")
    base_reserved = base_allocator.get("reserved_bytes") if isinstance(base_allocator, Mapping) else None
    row = peak_query(target)
    allocated = row.get("allocated_bytes") if row else None
    reserved = row.get("reserved_bytes") if row else None
    paired = bool(
        source_kind == "torch_npu_runtime_api"
        and all(_is_number(value) for value in (base_reserved, allocated, reserved))
        and 0 <= base_reserved <= baseline_bytes and 0 <= allocated <= reserved
        and row.get("allocator_ok_count") == row.get("allocator_attempted_count")
        and row.get("allocator_ok_count", 0) >= 2
    )
    value = baseline_bytes - base_reserved if paired else baseline_bytes
    base = candidate(
        key="baseline_observed", label=("Observed baseline outside the allocator reservation" if paired
                                        else "Observed runtime-initialized HBM baseline"),
        byte_value=value, priority=str(baseline.get("source_priority") or "P1"),
        kind=source_kind, ref=str(baseline.get("source") or "baseline snapshot"),
        phase=str(baseline.get("phase") or "baseline"), device_id=device_id,
        scope_kind="device", unit="B", raw=baseline,
        evidence=("baseline device used minus baseline allocator reserved; avoids re-counting baseline allocator storage"
                  if paired else "single baseline observation; runtime and process already initialized, not proof of driver-only overhead"),
        peak_kind=measurement_kind, measurement_kind=measurement_kind,
        overlap_group="device_baseline",
    )
    items = [base] if base else []
    if not paired:
        limitations = ([{"code": "allocator_pair_unavailable", "required": False,
                         "message": "selected peak and baseline allocator observations are unavailable or incomplete; no allocator subtraction or inactive reservation is included"}]
                       if isinstance(target, Mapping) and target.get("measurement_kind") == "sampled_device_peak" else [])
        return items, limitations
    inactive = candidate(
        key="allocator_inactive_reserved", label="Allocator reserved minus allocated at the selected device peak",
        byte_value=reserved - allocated, priority="P1", kind="torch_npu_runtime_api",
        ref="peak_sample.reserved_bytes - peak_sample.allocated_bytes", phase="sampled_peak",
        device_id=device_id, scope_kind="device", unit="B", raw=row,
        evidence="same query as the selected device maximum; retained reservation is not a new active tensor or a fragmentation measurement",
        peak_kind="paired_query_difference", measurement_kind="paired_query_difference",
        overlap_group="allocator_inactive_reserved",
    )
    if inactive:
        items.append(inactive)
    return items, [{"code": "memory_query_nonatomic", "required": False,
                    "message": "device and allocator APIs were read sequentially within the recorded peak query; their readings are not atomic"}]


__all__ = ["baseline_and_allocator_candidates", "peak_query", "peak_wall_interval"]
