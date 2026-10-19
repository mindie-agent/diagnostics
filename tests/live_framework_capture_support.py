"""Workload benchmark and module-trace support for the live framework capture fixture.

[Role]
Keep unprofiled step timing and opt-in module-trace mapping out of the main live
workload script so that file stays cohesive and bounded.

[Invariants]
- Unprofiled timing uses only caller step ids and the same model/input path.
- Module trace is written only when the caller opted in and supplied a capture
  DB plus the steps document; no metrics are fabricated on missing inputs.
- The trace file contains the actual structure evidence and mapping coverage.

[Timeouts & Locks]
Pure in-process timing and bounded DB reads through ``module_trace``.

[Side-Effect Policy]
Writes only the caller-requested module trace JSON; no NPU launch.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from mindie_profiler.module_trace import (
    build_structure_evidence,
    map_module_trace,
    write_module_trace,
)


def timing(records: list[dict]) -> dict:
    values = [float(item["duration_ms"]) for item in records]
    return {
        "count": len(values),
        "total_ms": sum(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": sum(values) / len(values),
    }


def execute_step(torch, torch_npu, model, inputs, iterations: int) -> None:
    with torch.no_grad():
        for _ in range(iterations):
            model(inputs)
    torch_npu.npu.synchronize()


def run_unprofiled(torch, torch_npu, model, inputs, step_ids: list[int], iterations: int) -> dict:
    records: list[dict] = []
    window_start_ns = time.time_ns()
    window_perf_start_ns = time.perf_counter_ns()
    for step in step_ids:
        wall_start_ns = time.time_ns()
        perf_start_ns = time.perf_counter_ns()
        execute_step(torch, torch_npu, model, inputs, iterations)
        wall_end_ns = time.time_ns()
        perf_end_ns = time.perf_counter_ns()
        records.append({
            "step": step,
            "wall_start_ns": wall_start_ns,
            "wall_end_ns": wall_end_ns,
            "wall_start_ts": wall_start_ns / 1_000_000_000,
            "wall_end_ts": wall_end_ns / 1_000_000_000,
            "perf_start_ns": perf_start_ns,
            "perf_end_ns": perf_end_ns,
            "duration_ms": (perf_end_ns - perf_start_ns) / 1_000_000,
            "iterations_per_step": iterations,
            "clock_domain": "unix_wall",
        })
    window_end_ns = time.time_ns()
    window_perf_end_ns = time.perf_counter_ns()
    return {
        "measurement_window": {
            "start_ts": window_start_ns / 1_000_000_000,
            "end_ts": window_end_ns / 1_000_000_000,
            "start_ns": window_start_ns,
            "end_ns": window_end_ns,
        },
        "duration_ms": (window_perf_end_ns - window_perf_start_ns) / 1_000_000,
        "sample_count": len(records),
        "timing": timing(records),
        "steps": records,
        "perf_window_ns": [window_perf_start_ns, window_perf_end_ns],
    }


def write_module_trace_evidence(
    raw_trace: Mapping[str, Any] | None,
    db_path: Path | None,
    steps_document: Mapping[str, Any] | None,
    output_dir: Path,
    requested_path: str | None,
) -> dict[str, Any]:
    if raw_trace is None:
        return {"status": "not_requested", "structure": None, "path": None, "data": None}
    if db_path is None or steps_document is None:
        return {
            "status": "unmapped",
            "structure": None,
            "path": None,
            "data": None,
            "error": {"code": "module_trace_missing_inputs",
                      "message": "module trace needs a successful DB export and steps document"},
        }
    try:
        mapping = map_module_trace(raw_trace, db_path, windows=steps_document["steps"])
        structure = build_structure_evidence(mapping)
        target = Path(requested_path) if requested_path else output_dir / "module_trace.json"
        write_module_trace(target, {
            **raw_trace,
            "mapping_result": mapping,
            "structure": structure,
            "source_identity": mapping["source_identity"],
        })
        return {
            "status": "mapped" if mapping["mapping"]["events_unmapped"] == 0 else "partial",
            "structure": structure,
            "path": str(target),
            "data": {
                "path": str(target),
                "schema_version": raw_trace["schema_version"],
                "scope": raw_trace["scope"],
                "instrumentation": raw_trace["instrumentation"],
                "mapping": mapping["mapping"],
                "structure_schema": structure["schema_version"],
            },
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "structure": None,
            "path": None,
            "data": None,
            "error": {"code": getattr(exc, "code", "module_trace_failed"),
                      "message": f"{type(exc).__name__}: {exc}"},
        }
