"""Explicit caller module trace collection and device-event containment mapping.

[Role]
Opt-in PyTorch leaf-module hooks for a profiled tiny MLP.  The collector records
actual module identity/order/invocation, caller step id, wall/monotonic bounds,
tensor shapes/dtypes, then maps real DB kernel events inside each synchronized
invocation and aggregates per module identity.

[Invariants]
- Hooks apply only while the caller holds the collector; ``remove()`` is
  mandatory and is exercised in ``finally`` blocks by the live workload.
- Module names/types come from the actual PyTorch tree.  A tiny MLP has modules,
  never Transformer/LLM layers or fabricated aggregate identities.
- The opt-in mode synchronizes NPU at module boundaries; its overhead scope is
  recorded and the result is not application throughput evidence.
- Kernel events are mapped only when fully inside a synchronized invocation;
  overlapping/ambiguous events stay unmapped and visible.
- Missing step ids, input/output tensors, DB events, or partial mappings stay
  explicit; nothing is silently zero-filled.

[Timeouts & Locks]
Hooks are in-process timing/synchronize calls.  DB mapping uses the existing
bounded ``DbSource`` reads.  No network, subprocess, or lock is introduced.

[Side-Effect Policy]
Registers/removes module hooks on the caller-provided model; reads the named DB
and steps only.  Writes only caller-requested JSON.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import hashlib
from itertools import accumulate
import json
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence

TRACE_SCHEMA = "mindie-module-trace/1"
MAPPING_SCHEMA = "mindie-module-event-mapping/1"
STRUCTURE_SCHEMA = "mindie-module-structure/1"
SCOPE = "tiny_mlp_leaf_modules"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_specs(value: Any, limit: int = 4) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if len(specs) >= limit:
            return
        shape = getattr(item, "shape", None)
        dtype = getattr(item, "dtype", None)
        if shape is not None and dtype is not None:
            try:
                specs.append({"shape": [int(dim) for dim in shape], "dtype": str(dtype)})
            except (TypeError, ValueError):
                specs.append({"shape": None, "dtype": str(dtype)})
            return
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return specs


def _ms(start_ns: int, end_ns: int) -> float:
    return (int(end_ns) - int(start_ns)) / 1_000_000.0


def _union_ms(intervals: Sequence[tuple[int, int]]) -> float:
    ordered = sorted((int(start), int(end)) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    total += current_end - current_start
    return total / 1_000_000.0


class ModuleTraceCollector:
    """Hook actual leaf modules for one opt-in profiled run."""

    def __init__(self, model: Any, *, sync: Callable[[], Any] | None = None,
                 max_tensor_specs: int = 4) -> None:
        self.model = model
        self.sync = sync
        self.max_tensor_specs = max_tensor_specs
        self._leaf: list[tuple[str, Any]] = [
            (name, module) for name, module in model.named_modules()
            if name and not list(module.children())
        ]
        if not self._leaf:
            raise ValueError("model exposes no leaf modules to trace")
        self._ordinals = {name: index for index, (name, _module) in enumerate(self._leaf)}
        self._handles: list[Any] = []
        self._pending: dict[str, dict[str, Any]] = {}
        self._order = 0
        self._step: int | None = None
        self.records: list[dict[str, Any]] = []

    def set_step(self, step: int | None) -> None:
        if step is not None and not isinstance(step, int):
            raise TypeError("step id must be an int or None")
        self._step = step

    def _synchronize(self) -> None:
        if self.sync is not None:
            self.sync()

    def _pre(self, name: str, module: Any, args: Any, kwargs: Any) -> None:
        self._synchronize()
        self._pending[name] = {
            "order": self._order,
            "step_id": self._step,
            "wall_start_ns": time.time_ns(),
            "monotonic_start_ns": time.perf_counter_ns(),
            "input_tensors": _tensor_specs((args, kwargs), self.max_tensor_specs),
        }
        self._order += 1

    def _post(self, name: str, module: Any, args: Any, kwargs: Any, output: Any) -> None:
        self._synchronize()
        pending = self._pending.pop(name, None)
        if pending is None:
            return
        self.records.append({
            "module_id": name,
            "module_name": name,
            "module_type": type(module).__name__,
            "ordinal": self._ordinals[name],
            "invocation": pending["order"],
            "step_id": pending["step_id"],
            "wall_start_ns": pending["wall_start_ns"],
            "wall_end_ns": time.time_ns(),
            "monotonic_start_ns": pending["monotonic_start_ns"],
            "monotonic_end_ns": time.perf_counter_ns(),
            "input_tensors": pending["input_tensors"],
            "output_tensors": _tensor_specs(output, self.max_tensor_specs),
        })

    def attach(self) -> "ModuleTraceCollector":
        for name, module in self._leaf:
            self._handles.append(module.register_forward_pre_hook(
                lambda _module, args, kwargs, name=name: self._pre(name, _module, args, kwargs),
                with_kwargs=True,
            ))
            self._handles.append(module.register_forward_hook(
                lambda _module, args, kwargs, output, name=name: self._post(
                    name, _module, args, kwargs, output
                ),
                with_kwargs=True,
            ))
        return self

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending.clear()

    def __enter__(self) -> "ModuleTraceCollector":
        return self.attach()

    def __exit__(self, *_exc: Any) -> bool:
        self.remove()
        return False

    def trace(self) -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCHEMA,
            "scope": SCOPE,
            "module_scope": "actual_pytorch_leaf_modules",
            "not_application_throughput": True,
            "instrumentation": {
                "npu_synchronize": self.sync is not None,
                "scope": "profiled_run_only",
                "overhead_scope": "includes per-leaf-module NPU synchronization and hook timing",
                "throughput_evidence": False,
            },
            "modules": [
                {"module_id": name, "module_name": name,
                 "module_type": type(module).__name__, "ordinal": index}
                for index, (name, module) in enumerate(self._leaf)
            ],
            "steps": sorted({record["step_id"] for record in self.records
                             if record["step_id"] is not None}),
            "records": self.records,
        }


def _coerce_windows(steps_file: str | Path | None, windows: Sequence[Any] | None) -> list[Any]:
    from .events import Window
    from .segment import load_steps_file

    if steps_file is not None:
        return load_steps_file(steps_file, unit="ns")
    rows = list(windows or [])
    if not rows:
        raise ValueError("module trace mapping needs explicit step windows")
    result = []
    for item in rows:
        if isinstance(item, Window):
            result.append(item)
        elif isinstance(item, Mapping) and "start_ns" in item and "end_ns" in item:
            result.append(Window(int(item["start_ns"]), int(item["end_ns"]),
                                 step=item.get("step"), label=item.get("label")))
        else:
            raise ValueError(f"unsupported step window: {item!r}")
    return result


def map_module_trace(
    trace: Mapping[str, Any],
    db_path: str | Path,
    *,
    steps_file: str | Path | None = None,
    windows: Sequence[Any] | None = None,
    timeout_s: float = 60.0,
    max_rows: int = 2_000_000,
) -> dict[str, Any]:
    """Map fully-contained DB kernel events into synchronized module invocations."""
    from .classify import apply as classify_events
    from .events import PARSER_VERSION
    from .sources_db import DbSource, probe_db

    selected = _coerce_windows(steps_file, windows)
    selected_steps = {window.step for window in selected}
    source = DbSource(probe_db(Path(db_path), timeout_s=timeout_s))
    events_by_step: dict[int | None, list[Any]] = {}
    step_stats = {}
    for window in selected:
        events, stats = source.iter_window(window, max_rows=max_rows, timeout_s=timeout_s)
        classify_events(events)
        events_by_step[window.step] = sorted(events, key=lambda event: (
            event.start_ns, event.end_ns,
            int(event.task_id) if str(event.task_id).isdigit() else 0,
        ))
        step_stats[str(window.step)] = {"events": len(events), "rows_read": stats.rows_read}

    invocations: list[dict[str, Any]] = []
    indices = {
        step: ([event.start_ns for event in events],
               list(accumulate((event.end_ns for event in events), max)))
        for step, events in events_by_step.items()
    }
    candidates = []
    owner_counts: dict[int, int] = {}
    for record in trace.get("records", []):
        step = record.get("step_id")
        if step not in selected_steps:
            continue
        start, end = int(record["wall_start_ns"]), int(record["wall_end_ns"])
        events = events_by_step.get(step, [])
        starts, prefix_ends = indices.get(step, ([], []))
        # The prefix end index retains long overlapping events while skipping
        # completed earlier kernels; normal leaf traces avoid an N-by-M scan.
        nearby = events[bisect_right(prefix_ends, start):bisect_left(starts, end)]
        contained = [event for event in nearby if start <= event.start_ns and event.end_ns <= end]
        crossing = [event for event in nearby
                    if event.end_ns > start and event.start_ns < end
                    and not (start <= event.start_ns and event.end_ns <= end)]
        candidates.append((record, start, end, contained, crossing))
        for event in contained:
            owner_counts[id(event)] = owner_counts.get(id(event), 0) + 1

    used_event_ids: set[int] = set()
    ambiguous_event_ids: set[int] = set()
    for record, start, end, contained, crossing in candidates:
        mapped = [event for event in contained if owner_counts[id(event)] == 1]
        ambiguous = crossing + [event for event in contained if owner_counts[id(event)] > 1]
        used_event_ids.update(id(event) for event in mapped)
        ambiguous_event_ids.update(id(event) for event in ambiguous)
        busy = _union_ms([(event.start_ns, event.end_ns) for event in mapped])
        wall = _ms(start, end)
        invocations.append({
            **{key: record.get(key) for key in (
                "module_id", "module_name", "module_type", "ordinal", "invocation",
                "step_id", "wall_start_ns", "wall_end_ns", "monotonic_start_ns",
                "monotonic_end_ns", "input_tensors", "output_tensors",
            )},
            "wall_ms": wall,
            "busy_union_ms": busy,
            "host_gap_ms": max(0.0, wall - busy),
            "mapped_event_count": len(mapped),
            "ambiguous_event_count": len(ambiguous),
            "kernel_names": sorted({event.name for event in mapped}),
            "mapping_status": "mapped" if mapped and not ambiguous else (
                "partial" if mapped else "unmapped"
            ),
        })

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for item in invocations:
        key = (str(item["module_id"]), str(item["module_type"]), int(item["ordinal"]))
        grouped.setdefault(key, []).append(item)
    modules = []
    for (module_id, module_type, ordinal), rows in sorted(grouped.items(), key=lambda item: item[0][2]):
        mapped_rows = [row for row in rows if row["mapped_event_count"] > 0]
        modules.append({
            "module_id": module_id,
            "module_name": rows[0]["module_name"],
            "module_type": module_type,
            "ordinal": ordinal,
            "invocation_count": len(rows),
            "step_ids": sorted({row["step_id"] for row in rows if row["step_id"] is not None}),
            "wall_ms_p50": median(row["wall_ms"] for row in rows),
            "busy_union_ms_p50": median(row["busy_union_ms"] for row in mapped_rows) if mapped_rows else None,
            "host_gap_ms_p50": median(row["host_gap_ms"] for row in mapped_rows) if mapped_rows else None,
            "mapped_invocations": len(mapped_rows),
            "unmapped_invocations": len(rows) - len(mapped_rows),
            "mapping_coverage": len(mapped_rows) / len(rows) if rows else None,
        })
    total_events = sum(len(value) for value in events_by_step.values())
    mapped_events = len(used_event_ids)
    return {
        "schema_version": MAPPING_SCHEMA,
        "scope": SCOPE,
        "module_scope": "actual_pytorch_leaf_modules",
        "not_application_throughput": True,
        "source_identity": {
            "db_path": str(Path(db_path)),
            "db_sha256": _sha256(Path(db_path)),
            "steps_file": str(steps_file) if steps_file is not None else None,
            "steps_sha256": _sha256(Path(steps_file)) if steps_file is not None else None,
            "parser_version": PARSER_VERSION,
            "step_stats": step_stats,
        },
        "mapping": {
            "events_total": total_events,
            "events_mapped": mapped_events,
            "events_unmapped": total_events - mapped_events,
            "events_ambiguous": len(ambiguous_event_ids),
            "partial_invocations": sum(1 for row in invocations if row["mapping_status"] != "mapped"),
        },
        "modules": modules,
        "invocations": invocations,
    }


def build_structure_evidence(mapping: Mapping[str, Any]) -> dict[str, Any]:
    invocations = list(mapping.get("invocations", []))
    sequence = [f"{item['module_name']}:{item['module_type']}" for item in invocations]
    per_step: dict[str, int] = {}
    for item in invocations:
        key = str(item.get("step_id"))
        per_step[key] = per_step.get(key, 0) + 1
    shape = {
        "token_count": len(sequence),
        "per_step_counts": per_step,
        "unique_tokens": sorted(set(sequence)),
        "prefix": sequence[:64],
        "suffix": sequence[-64:],
    }
    return {
        "schema_version": STRUCTURE_SCHEMA,
        "scope": SCOPE,
        "module_scope": "actual_pytorch_leaf_modules",
        "not_application_throughput": True,
        "source_identity": mapping.get("source_identity"),
        "mapping": mapping.get("mapping"),
        "operator_sequence": sequence,
        "event_sequence_shape": shape,
        "modules": mapping.get("modules", []),
        "layers": [],
        "layer_mapping": {
            "status": "not_applicable",
            "reason": "tiny MLP actual leaf modules; these are not Transformer/LLM layers",
        },
    }


def build_module_table(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    def modules(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        structure = value.get("structure") if isinstance(value.get("structure"), Mapping) else value
        rows = structure.get("modules") if isinstance(structure, Mapping) else None
        return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []

    b_modules, c_modules = modules(baseline), modules(candidate)
    b_map = {(row.get("module_id"), row.get("module_type"), row.get("ordinal")): row for row in b_modules}
    c_map = {(row.get("module_id"), row.get("module_type"), row.get("ordinal")): row for row in c_modules}
    rows = []
    limitations = []
    for key in sorted(set(b_map) | set(c_map), key=lambda item: item[2] if isinstance(item[2], int) else 0):
        b_row, c_row = b_map.get(key), c_map.get(key)
        if b_row is None or c_row is None:
            limitations.append({"code": "module_missing", "module": key})
            rows.append({"module_id": key[0], "module_type": key[1], "ordinal": key[2],
                         "status": "not_comparable"})
            continue
        rows.append({
            "module_id": key[0],
            "module_name": b_row.get("module_name"),
            "module_type": key[1],
            "ordinal": key[2],
            "status": "compared",
            "baseline_invocation_count": b_row.get("invocation_count"),
            "candidate_invocation_count": c_row.get("invocation_count"),
            "baseline_wall_ms_p50": b_row.get("wall_ms_p50"),
            "candidate_wall_ms_p50": c_row.get("wall_ms_p50"),
            "wall_delta_ms": (c_row.get("wall_ms_p50") - b_row.get("wall_ms_p50"))
            if isinstance(b_row.get("wall_ms_p50"), (int, float))
            and isinstance(c_row.get("wall_ms_p50"), (int, float)) else None,
            "baseline_busy_union_ms_p50": b_row.get("busy_union_ms_p50"),
            "candidate_busy_union_ms_p50": c_row.get("busy_union_ms_p50"),
            "baseline_host_gap_ms_p50": b_row.get("host_gap_ms_p50"),
            "candidate_host_gap_ms_p50": c_row.get("host_gap_ms_p50"),
            "baseline_mapping_coverage": b_row.get("mapping_coverage"),
            "candidate_mapping_coverage": c_row.get("mapping_coverage"),
        })
    status = "comparable" if rows and all(row["status"] == "compared" for row in rows) else (
        "comparable_with_limitations" if any(row["status"] == "compared" for row in rows) else "not_comparable"
    )
    return {"status": status, "scope": SCOPE, "modules": rows, "limitations": limitations}


def write_module_trace(path: str | Path, trace: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


__all__ = [
    "MAPPING_SCHEMA",
    "ModuleTraceCollector",
    "SCOPE",
    "STRUCTURE_SCHEMA",
    "TRACE_SCHEMA",
    "build_module_table",
    "build_structure_evidence",
    "map_module_trace",
    "write_module_trace",
]
