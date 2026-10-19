"""P0 NPU_MEM and OP_MEMORY candidate adapters.

[Role]
  Build observed NPU_MEM scope references and a conservative transient
  OP_MEMORY operator-allocation candidate for one selected device.

[Invariants]
  * NPU_MEM app/device references are never additive.
  * APP is observed profiler app-scope reserved, not proven PyTorch allocator.
  * Device is observed profiler device scope, not proven physical whole-device.
  * OP_MEMORY allocations are references: interval overlap does not prove
    coexistence or disjointness from model/KV/allocator quantities. Operator
    name alone never proves scratch/workspace semantics.
  * No cross-device largest-allocation selection.

[Timeouts & Locks]
  Pure mapping adaptation.

[Side-Effect Policy]
  No I/O or mutation.
"""

from __future__ import annotations

from typing import Any, Mapping

from .memory_units import _is_number


def npu_mem_candidates(npu_mem: Mapping[str, Any] | None, *, device_id: int, candidate: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not isinstance(npu_mem, Mapping):
        return [], []
    candidates: list[dict[str, Any]] = []
    limitations: list[dict[str, str]] = []
    for scope, key, label, overlap in (
        ("device", "npu_device_sampled", "NPU_MEM observed device-scope HBM", "npu_mem_device"),
        ("app", "npu_app_sampled", "NPU_MEM observed app-scope reserved", "npu_mem_app"),
    ):
        fact = npu_mem.get(scope)
        if not isinstance(fact, Mapping) or not _is_number(fact.get("max_bytes")):
            continue
        item = candidate(
            key=key, label=label, byte_value=float(fact["max_bytes"]), priority="P0",
            kind="msprof_npu_mem_sqlite", ref=f"NPU_MEM {scope}:hbm sampled max",
            phase="capture", device_id=device_id,
            scope_kind="device_observed" if scope == "device" else "app_reserved",
            unit="B", raw=fact, evidence=(
                "read-only NPU_MEM hbm bytes; app is observed profiler app-scope reserved "
                "(not proven PyTorch allocator); device is observed profiler device scope "
                "(not proven physical whole-device); sampled max is not absolute/continuous"
            ), peak_kind="sampled_max", measurement_kind="sampled_max",
            included_in_sum=False, overlap_group=overlap,
        )
        if item:
            candidates.append(item)
        if int(fact.get("samples") or 0) < 2:
            limitations.append({"code": "npu_mem_samples_insufficient",
                                "message": f"NPU_MEM {scope} has fewer than two samples; sampled max is weak", "required": True})
    if candidates:
        limitations.append({"code": "npu_mem_schedule_unknown",
                            "message": "NPU_MEM declares no intended cadence/slot schedule; observed intervals are evidence, not coverage", "required": True})
    return candidates, limitations


def _contains_target(allocation: Mapping[str, Any], target_window_ns: tuple[int, int] | None) -> bool:
    if target_window_ns is None:
        return False
    start, end = target_window_ns
    alloc = allocation.get("allocation_time_ns")
    release = allocation.get("release_time_ns")
    if not isinstance(alloc, int):
        return False
    return isinstance(release, int) and alloc <= start and release >= end


def op_memory_candidates(
    op_memory: Mapping[str, Any] | None,
    *,
    device_id: int,
    candidate: Any,
    target_window_ns: tuple[int, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not isinstance(op_memory, Mapping):
        return [], []
    largest = op_memory.get("largest_allocation")
    if not isinstance(largest, Mapping) or not _is_number(largest.get("size_bytes")):
        return [], []
    aligned = _contains_target(largest, target_window_ns)
    item = candidate(
        key="transient_operator_allocation", label="Transient OP_MEMORY operator allocation",
        byte_value=float(largest["size_bytes"]), priority="P0", kind="msprof_op_memory_sqlite",
        ref=f"OP_MEMORY {largest.get('name')}:size", phase="capture", device_id=device_id,
        scope_kind="device_observed", unit="B", raw=largest,
        evidence="operator allocation identity is exact; alignment and disjointness from model/KV/allocator quantities are separate requirements; reference only",
        peak_kind="transient_allocation", measurement_kind="transient_allocation",
        included_in_sum=False, overlap_group="transient_scope_unverified",
    )
    limitations: list[dict[str, str]] = [
        {"code": "transient_operator_workspace_unproven",
         "message": "operator name alone does not prove scratch/workspace semantics", "required": False},
    ]
    if not aligned:
        limitations.append({"code": "transient_operator_window_unaligned",
                            "message": "allocation lifetime does not contain the selected peak query's measured wall-clock interval; reference-only", "required": False})
    limitations.append({"code": "transient_operator_disjointness_unproven",
                        "message": "allocation lifetime alone does not prove disjointness from model/KV/allocator quantities; reference-only", "required": False})
    return ([item] if item else []), limitations


__all__ = ["npu_mem_candidates", "op_memory_candidates"]
