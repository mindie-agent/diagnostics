"""Actual-capture C7 three-time acceptance for pinned TensorCast.

[Role]
  Read one completed live-workload result and its read-only profiling SQLite,
  select observed MatMul/ReLU kernel keys, compute the measured p50 duration
  from TASK/COMPUTE_TASK_INFO, and pair it with the raw local ideal roofline
  bound and the actual pinned TensorCast analytic prediction.

[Invariants]
  * Raw capture, profile DB, steps and manifests are read-only.  No NPU or
    remote workload is launched and no source path is installed or modified.
  * Kernel keys and shape/dtype/format values come from the observed DB rows.
    Fused MatMul+bias mapping is accepted only when DB input shapes and the
    safetensors weight/bias shapes agree; otherwise the kernel is unmodeled.
  * ``ideal_hardware_lower_bound_ms`` uses raw physical profile peaks;
    ``model_predicted_ms`` is the upstream efficiency-adjusted result and
    ``measured_device_ms`` is the DB p50.  The three values are never mixed.
  * Unsupported or unproven keys remain ``unmodeled`` with a reason; no shape
    or op mapping is invented.
  * PMU counters, when present, are reported as raw source-value medians and
    are not interpreted as a hardware bottleneck.

[Timeouts & Locks]
  One bounded read-only SQLite session and one provider construction per
  artifact.  No network, git-per-kernel scan, or NPU execution.

[Side-Effect Policy]
  Reads explicit result/profile paths.  Writes only the requested JSON output,
  atomically through a temporary file and ``os.replace``.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sqlite3
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .modeling_inputs import HardwareSpec, KernelInputs, estimate_ideal_hardware_lower_bound
from .tensor_cast_provider import TensorCastAnalyticProvider
from .theoretical_model import ModelingRequest, align_modeling

SCHEMA_VERSION = "mindie-profiler-provider-c7-three-time/1"
_KERNEL_TABLES = "TASK t JOIN COMPUTE_TASK_INFO c ON c.globalTaskId = t.globalTaskId"
_PMU_COUNTERS = (
    "aic_total_time",
    "aic_mac_ratio",
    "aic_mte1_ratio",
    "aic_mte2_ratio",
    "aic_scalar_ratio",
    "aiv_total_time",
    "aiv_vec_ratio",
    "aiv_mte2_ratio",
    "aiv_mte3_ratio",
)


def _add_limitation(
    limitations: list[dict[str, Any]],
    code: str,
    scope: str,
    detail: str,
    severity: str = "medium",
) -> None:
    for item in limitations:
        if item.get("code") == code and item.get("scope") == scope:
            return
    limitations.append({"code": code, "scope": scope, "detail": detail, "severity": severity})


def _coerce_hardware(value: Any) -> tuple[HardwareSpec | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, HardwareSpec):
        return value, None
    if isinstance(value, Mapping):
        try:
            return HardwareSpec(**dict(value)), None
        except (TypeError, ValueError) as exc:
            return None, f"model_hardware_invalid:{type(exc).__name__}"
    return None, f"model_hardware_invalid:{type(value).__name__}"


def _coerce_inputs(value: Any, hardware: HardwareSpec | None) -> KernelInputs | None:
    candidate: KernelInputs | None = None
    if isinstance(value, KernelInputs):
        candidate = value
    elif isinstance(value, Mapping):
        if "inputs" in value:
            return _coerce_inputs(value.get("inputs"), hardware)
        try:
            candidate = KernelInputs.from_mapping(value)
        except (TypeError, ValueError):
            return None
    if candidate is not None and hardware is not None and candidate.hardware is None:
        candidate = dataclasses.replace(candidate, hardware=hardware)
    return candidate


def _looks_like_inputs(value: Any) -> bool:
    if isinstance(value, KernelInputs):
        return True
    return isinstance(value, Mapping) and any(
        key in value for key in ("shape", "dtype", "layout", "hardware", "op_kind", "flops", "memory_bytes")
    )


def _normalize_registry(
    model_inputs: Any,
    hardware: HardwareSpec | None,
) -> tuple[dict[Any, Any] | None, list[dict[str, Any]]]:
    limitations: list[dict[str, Any]] = []
    if model_inputs is None:
        return None, limitations
    if isinstance(model_inputs, Mapping):
        if _looks_like_inputs(model_inputs):
            _add_limitation(
                limitations, "model_inputs_must_be_keyed", "global",
                "model_inputs is a single kernel input record; key it by kernel_name or (rank_id, kernel_name)",
                "high",
            )
            return None, limitations
        return dict(model_inputs), limitations
    if isinstance(model_inputs, Sequence) and not isinstance(model_inputs, (str, bytes)):
        registry: dict[Any, Any] = {}
        for index, item in enumerate(model_inputs):
            if not isinstance(item, Mapping) or "inputs" not in item:
                _add_limitation(
                    limitations, "model_inputs_entry_invalid", f"entry_{index}",
                    "each model_inputs sequence entry needs kernel_name and inputs",
                    "high",
                )
                continue
            name = item.get("kernel_name") or item.get("name")
            rank_id = item.get("rank_id")
            block = item.get("semantic_block")
            if name is None:
                _add_limitation(
                    limitations, "model_inputs_entry_invalid", f"entry_{index}",
                    "model_inputs entry has no kernel_name",
                    "high",
                )
                continue
            if rank_id is not None and block is not None:
                key = (str(rank_id), str(block), str(name))
            elif rank_id is not None:
                key = (str(rank_id), str(name))
            else:
                key = str(name)
            registry[key] = item["inputs"]
        return registry, limitations
    _add_limitation(
        limitations, "model_inputs_invalid", "global",
        "model_inputs must be a keyed mapping or a sequence of {kernel_name, inputs} records",
        "high",
    )
    return None, limitations


def _lookup_inputs(
    registry: Mapping[Any, Any] | None,
    rank_id: str,
    block: str,
    kernel_name: str,
    hardware: HardwareSpec | None,
) -> KernelInputs | None:
    if registry is None:
        return None
    keys = (
        (rank_id, block, kernel_name),
        (rank_id, kernel_name),
        (block, kernel_name),
        kernel_name,
        f"{rank_id}:{kernel_name}",
        f"{rank_id}:{block}:{kernel_name}",
    )
    for key in keys:
        if key in registry:
            return _coerce_inputs(registry[key], hardware)
    return None


def _hotspot_scope(row: Mapping[str, Any]) -> dict[str, Any]:
    locator = row.get("evidence_locator") or {}
    return {
        **locator,
        "rank_id": row.get("rank_id"),
        "rank_index": row.get("rank_index"),
        "semantic_block": row.get("semantic_block"),
        "kernel_name": row.get("kernel_name"),
        "measured_device_ms_p50": row.get("measured_device_ms_p50"),
        "measured_basis": "core hotspot measured_device_ms_p50 from clipped event durations",
    }


def _model_input_scope(inputs: KernelInputs) -> dict[str, Any]:
    hardware = inputs.hardware
    return {
        "shape": list(inputs.shape) if inputs.shape is not None else None,
        "dtype": inputs.dtype,
        "layout": inputs.layout,
        "op_kind": inputs.op_kind,
        "flops": inputs.flops,
        "memory_bytes": inputs.memory_bytes,
        "hardware": hardware.name if hardware is not None else None,
        "hardware_peak_flops_per_s": hardware.peak_flops_per_s if hardware is not None else None,
        "hardware_memory_bandwidth_bytes_per_s": (
            hardware.memory_bandwidth_bytes_per_s if hardware is not None else None
        ),
        "input_signature": inputs.input_signature(),
    }


def _unmodeled_alignment(row: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "status": "unmodeled",
        "ideal_hardware_lower_bound_ms": None,
        "model_predicted_ms": None,
        "measured_device_ms": row.get("measured_device_ms_p50"),
        "model_provider": None,
        "model_assumptions": None,
        "unmodeled_reason": reason,
        "input_signature": None,
        "model_evidence": [],
        "measurement_limitation": None,
        "ideal_assumptions": None,
        "ideal_bottleneck": None,
        "ideal_formula_version": None,
        "scope": _hotspot_scope(row),
        "model_input_scope": None,
    }


def apply_modeling(
    rows: Sequence[dict[str, Any]],
    *,
    provider: Any,
    registry: Mapping[Any, Any] | None,
    hardware: HardwareSpec | None,
    hardware_error: str | None,
) -> dict[str, Any]:
    """Attach provider alignment to hotspot rows without changing measured values."""
    modeled = 0
    unmodeled = 0
    reasons: set[str] = set()
    limitations: list[dict[str, Any]] = []
    if hardware_error:
        _add_limitation(limitations, hardware_error, "global", "model_hardware could not be parsed", "high")
    if provider is not None and registry is None:
        _add_limitation(
            limitations, "model_inputs_missing", "global",
            "a model provider was supplied but no keyed kernel inputs were supplied",
            "high",
        )
    if provider is None and registry is not None:
        _add_limitation(
            limitations, "model_provider_not_configured", "global",
            "kernel inputs were supplied but no model provider was configured",
            "medium",
        )
    for row in rows:
        inputs = _lookup_inputs(
            registry,
            str(row.get("rank_id")),
            str(row.get("semantic_block")),
            str(row.get("kernel_name")),
            hardware,
        )
        if inputs is None:
            if provider is not None or registry is not None:
                reason = "no model inputs supplied for this hotspot"
            else:
                reason = str(
                    row.get("modeling_alignment", {}).get("unmodeled_reason")
                    or "model provider not configured"
                )
            payload = _unmodeled_alignment(row, reason)
            reasons.add(reason)
        else:
            request = ModelingRequest(
                inputs=inputs,
                measured_device_ms=row.get("measured_device_ms_p50"),
                measured_source="core hotspot measured_device_ms_p50",
            )
            alignment = align_modeling(request, provider)
            payload = alignment.to_c7_alignment()
            lower = estimate_ideal_hardware_lower_bound(inputs)
            payload["ideal_assumptions"] = list(lower.assumptions) if lower.status == "modeled" else None
            payload["ideal_bottleneck"] = lower.bottleneck if lower.status == "modeled" else None
            payload["ideal_formula_version"] = lower.formula_version
            payload["scope"] = _hotspot_scope(row)
            payload["model_input_scope"] = _model_input_scope(inputs)
            if alignment.unmodeled_reason:
                reasons.add(alignment.unmodeled_reason)
        if payload["status"] == "modeled":
            modeled += 1
        else:
            unmodeled += 1
        row["modeling_alignment"] = payload
    return {
        "provider_supplied": provider is not None,
        "model_inputs_supplied": registry is not None,
        "hardware_supplied": hardware is not None,
        "hotspots_total": len(rows),
        "hotspots_modeled": modeled,
        "hotspots_unmodeled": unmodeled,
        "unmodeled_reasons": sorted(reasons),
        "limitations": limitations,
    }



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _db_path(result_path: Path) -> Path:
    result = _load_json(result_path)
    profile_root = result_path.parent / "profile"
    artifacts = result.get("capture", {}).get("artifacts", {}).get("artifacts", [])
    for item in artifacts:
        if item.get("kind") != "db":
            continue
        relative = item.get("relative_path")
        if relative:
            candidate = profile_root / relative
            if candidate.is_file():
                return candidate
    matches = sorted(profile_root.rglob("ascend_pytorch_profiler.db"))
    if not matches:
        raise FileNotFoundError(f"no ascend_pytorch_profiler.db under {profile_root}")
    return matches[0]


def _steps(result_path: Path) -> list[dict[str, Any]]:
    steps_path = result_path.parent / "steps.json"
    if steps_path.is_file():
        payload = _load_json(steps_path)
        return [
            {"step": int(item["step"]), "start_ns": int(item["start_ns"]), "end_ns": int(item["end_ns"])}
            for item in payload["steps"]
        ]
    result = _load_json(result_path)
    records = result.get("capture", {}).get("step_records", [])
    return [
        {"step": int(item["step"]), "start_ns": int(item["wall_start_ns"]), "end_ns": int(item["wall_end_ns"])}
        for item in records
        if item.get("segment") == "active"
    ]


def _strings(con: sqlite3.Connection) -> dict[int, str]:
    return {int(row[0]): str(row[1]) for row in con.execute("SELECT id, value FROM STRING_IDS")}


def _schema_version(con: sqlite3.Connection) -> str | None:
    row = con.execute("SELECT value FROM META_DATA WHERE name='SCHEMA_VERSION' ORDER BY rowid DESC LIMIT 1").fetchone()
    return None if row is None else str(row[0])


def read_actual_kernels(result_path: str | Path) -> dict[str, Any]:
    """Read observed kernel keys and p50/p95 durations from the profile DB."""
    result_file = Path(result_path).expanduser().resolve()
    db_path = _db_path(result_file)
    steps = _steps(result_file)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        strings = _strings(con)
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        rows = con.execute(
            "SELECT t.rowid AS task_rowid, t.globalTaskId, t.startNs, t.endNs, t.streamId, t.taskId, "
            "c.name, c.opType, c.taskType, c.inputShapes, c.inputDataTypes, c.inputFormats, "
            "c.blockNum, c.mixBlockNum "
            f"FROM {_KERNEL_TABLES} ORDER BY t.startNs"
        )
        for row in rows:
            start_ns, end_ns = int(row["startNs"]), int(row["endNs"])
            matched = next((step for step in steps if step["start_ns"] <= start_ns < step["end_ns"]), None)
            if matched is None:
                continue
            key = (
                strings.get(row["name"], ""),
                strings.get(row["opType"], ""),
                strings.get(row["taskType"], ""),
                strings.get(row["inputShapes"], ""),
                strings.get(row["inputDataTypes"], ""),
                strings.get(row["inputFormats"], ""),
                int(row["blockNum"]),
                int(row["mixBlockNum"]),
            )
            left, right = max(start_ns, matched["start_ns"]), min(end_ns, matched["end_ns"])
            groups.setdefault(key, []).append(
                {
                    "step": matched["step"],
                    "task_rowid": int(row["task_rowid"]),
                    "global_task_id": int(row["globalTaskId"]),
                    "stream_id": int(row["streamId"]),
                    "task_id": int(row["taskId"]),
                    "duration_ns": max(0, right - left),
                }
            )
        pmu = _read_pmu(con, strings, groups)
        schema_version = _schema_version(con)
    finally:
        con.close()
    kernels = []
    for key, items in groups.items():
        durations_ms = [item["duration_ns"] / 1_000_000.0 for item in items]
        by_step: dict[int, list[float]] = {}
        for item in items:
            by_step.setdefault(item["step"], []).append(item["duration_ns"] / 1_000_000.0)
        kernels.append(
            {
                "kernel_key": {
                    "name": key[0],
                    "op_type": key[1],
                    "task_type": key[2],
                    "input_shapes": key[3],
                    "input_dtypes": key[4],
                    "input_formats": key[5],
                    "block_num": key[6],
                    "mix_block_num": key[7],
                },
                "source_evidence": {
                    "table": "TASK JOIN COMPUTE_TASK_INFO",
                    "row_count": len(items),
                    "step_ids": sorted(by_step),
                    "step_p50_ms": {str(step): statistics.median(values) for step, values in sorted(by_step.items())},
                    "global_task_id_range": [min(i["global_task_id"] for i in items), max(i["global_task_id"] for i in items)],
                    "sample_global_task_ids": [i["global_task_id"] for i in items[:5]],
                    "task_rowid_range": [min(i["task_rowid"] for i in items), max(i["task_rowid"] for i in items)],
                    "stream_ids": sorted({i["stream_id"] for i in items}),
                    "task_ids_sample": [i["task_id"] for i in items[:5]],
                    "duration_ns_source": "TASK.endNs - TASK.startNs clipped to explicit step window",
                },
                "measured": {
                    "statistic": "p50",
                    "unit": "ms",
                    "value": statistics.median(durations_ms),
                    "p50_ms": statistics.median(durations_ms),
                    "p95_ms": _percentile(durations_ms, 0.95),
                    "mean_ms": statistics.mean(durations_ms),
                    "min_ms": min(durations_ms),
                    "max_ms": max(durations_ms),
                    "samples": len(durations_ms),
                },
                "pmu": pmu.get(key, {"status": "absent"}),
            }
        )
    kernels.sort(key=lambda item: (item["kernel_key"]["op_type"], item["kernel_key"]["input_shapes"]))
    return {
        "result_path": str(result_file),
        "db_path": str(db_path),
        "db_sha256": _sha256(db_path),
        "db_size_bytes": db_path.stat().st_size,
        "db_schema_version": schema_version,
        "steps": steps,
        "kernels": kernels,
    }


def _read_pmu(
    con: sqlite3.Connection,
    strings: Mapping[int, str],
    groups: Mapping[tuple[Any, ...], Sequence[dict[str, Any]]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    task_to_key = {
        item["global_task_id"]: key
        for key, items in groups.items()
        for item in items
    }
    values: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    if not task_to_key:
        return {}
    task_ids = sorted(task_to_key)
    for offset in range(0, len(task_ids), 800):
        chunk = task_ids[offset : offset + 800]
        placeholders = ",".join("?" for _ in chunk)
        query = (
            "SELECT p.globalTaskId, s.value AS name, p.value "
            "FROM TASK_PMU_INFO p LEFT JOIN STRING_IDS s ON s.id = p.name "
            f"WHERE p.globalTaskId IN ({placeholders})"
        )
        for global_task_id, name, value in con.execute(query, chunk):
            if name not in _PMU_COUNTERS or not isinstance(value, (int, float)):
                continue
            key = task_to_key.get(int(global_task_id))
            if key is None:
                continue
            values.setdefault(key, {}).setdefault(str(name), []).append(float(value))
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, counters in values.items():
        result[key] = {
            "status": "present",
            "counter_source": "TASK_PMU_INFO",
            "counter_unit": "source_value_uninterpreted",
            "interpretation": "raw_source_counter_medians_not_a_hardware_bottleneck",
            "median_by_counter": {name: statistics.median(items) for name, items in sorted(counters.items())},
            "samples_by_counter": {name: len(items) for name, items in sorted(counters.items())},
        }
    return result


def _parse_tensor_shapes(text: str) -> list[list[int]]:
    shapes: list[list[int]] = []
    cleaned = text.strip().strip('"').strip("'")
    for part in cleaned.split(";"):
        part = part.strip().strip('"').strip("'")
        dims = [int(value.strip()) for value in part.split(",") if value.strip()]
        shapes.append(dims)
    return shapes


def _dtype_from_profile(text: str) -> str | None:
    names = [part.strip().upper() for part in text.split(";") if part.strip()]
    mapping = {"FLOAT": "float32", "FLOAT16": "float16", "BFLOAT16": "bfloat16", "DT_BF16": "bfloat16"}
    normalized = {mapping.get(name) for name in names}
    if len(normalized) != 1 or None in normalized:
        return None
    return normalized.pop()


def _weight_match(tensors: Sequence[Mapping[str, Any]], n: int, k: int) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    for tensor in tensors:
        shape = [int(dim) for dim in tensor.get("shape", [])]
        if shape != [n, k]:
            continue
        stem = str(tensor.get("name", "")).rsplit(".", 1)[0]
        bias_name = f"{stem}.bias"
        bias = next((item for item in tensors if item.get("name") == bias_name and [int(dim) for dim in item.get("shape", [])] == [n]), None)
        if bias is not None:
            return tensor, bias
    return None


def _kernel_inputs(
    entry: Mapping[str, Any],
    provider: TensorCastAnalyticProvider,
    weight_tensors: Sequence[Mapping[str, Any]],
) -> tuple[KernelInputs | None, dict[str, Any] | None, str | None]:
    key = entry["kernel_key"]
    name, op_type, task_type = key["name"], key["op_type"], key["task_type"]
    try:
        shapes = _parse_tensor_shapes(key["input_shapes"])
    except ValueError:
        return None, None, "profile_shape_unparseable"
    dtype = _dtype_from_profile(key["input_dtypes"])
    if dtype is None:
        return None, None, f"unsupported_profile_dtype:{key['input_dtypes']}"
    formats = {part.strip().upper() for part in key["input_formats"].split(";") if part.strip()}
    if formats != {"ND"}:
        return None, None, f"unsupported_profile_layout:{key['input_formats']}"
    annotations = (
        ("profile_name", name),
        ("profile_op_type", op_type),
        ("profile_task_type", task_type),
        ("profile_input_shapes", key["input_shapes"]),
        ("profile_input_dtypes", key["input_dtypes"]),
        ("profile_input_formats", key["input_formats"]),
    )
    if op_type == "Relu" and len(shapes) == 1:
        inputs = KernelInputs(
            shape=tuple(shapes[0]),
            dtype=dtype,
            layout="ND",
            hardware=provider.hardware_spec(dtype, "relu"),
            op_kind="relu",
            kernel_name=name,
            annotations=annotations,
        )
        return inputs, {
            "shape_source": "COMPUTE_TASK_INFO.inputShapes",
            "input_shape": shapes[0],
            "activation": "relu",
        }, None
    if op_type == "MatMulV2" and len(shapes) == 3:
        a_shape, b_shape, bias_shape = shapes
        if len(a_shape) != 2 or len(b_shape) != 2 or len(bias_shape) != 1:
            return None, None, "profile_matmul_shape_rank_unsupported"
        m, k = a_shape
        n, b_k = b_shape
        if b_k != k or bias_shape[0] != n:
            return None, None, "profile_matmul_shape_inconsistent"
        match = _weight_match(weight_tensors, n, k)
        if match is None:
            return None, None, "weight_manifest_shape_unverified"
        weight, bias = match
        inputs = KernelInputs(
            shape=(m, k, n),
            dtype=dtype,
            layout="ND",
            hardware=provider.hardware_spec(dtype, "addmm"),
            op_kind="addmm",
            kernel_name=name,
            annotations=annotations + (("b_layout", "transposed"), ("fused_bias", "true")),
        )
        return inputs, {
            "shape_source": "COMPUTE_TASK_INFO.inputShapes + safetensors weight/bias headers",
            "a_shape": a_shape,
            "b_stored_shape": b_shape,
            "bias_shape": bias_shape,
            "effective_matmul": "A[m,k] @ B_stored[n,k]^T + bias[n]",
            "weight_tensor": {"name": weight["name"], "shape": weight["shape"]},
            "bias_tensor": {"name": bias["name"], "shape": bias["shape"]},
        }, None
    return None, None, f"unsupported_op_or_shape_mapping:{op_type}"


def build_c7_three_time_artifact(
    result_path: str | Path,
    *,
    source_path: str | Path,
    device_name: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the actual measured/ideal/predicted artifact for observed kernels."""
    result_file = Path(result_path).expanduser().resolve()
    result = _load_json(result_file)
    actual = read_actual_kernels(result_file)
    provider = TensorCastAnalyticProvider(source_path, device_name=device_name)
    weight_tensors = result.get("manifest", {}).get("weight_manifest", {}).get("tensors", [])
    entries: list[dict[str, Any]] = []
    for detected in actual["kernels"]:
        inputs, shape_proof, reason = _kernel_inputs(detected, provider, weight_tensors)
        entry = {
            "kernel_key": detected["kernel_key"],
            "source_evidence": detected["source_evidence"],
            "measured": detected["measured"],
            "pmu": detected["pmu"],
            "shape_proof": shape_proof,
            "model_inputs": None if inputs is None else inputs.canonical_payload(),
        }
        if inputs is None:
            entry["three_time"] = {
                "status": "unmodeled",
                "measured_device_ms": detected["measured"]["value"],
                "ideal_hardware_lower_bound_ms": None,
                "model_predicted_ms": None,
                "unmodeled_reason": reason,
            }
            entry["alignment"] = None
        else:
            alignment = align_modeling(
                ModelingRequest(
                    inputs=inputs,
                    measured_device_ms=detected["measured"]["value"],
                    measured_source="TASK/COMPUTE_TASK_INFO p50 clipped duration",
                ),
                provider,
            )
            payload = alignment.to_c7_alignment()
            lower = estimate_ideal_hardware_lower_bound(inputs)
            payload["ideal_assumptions"] = list(lower.assumptions) if lower.status == "modeled" else None
            payload["ideal_bottleneck"] = lower.bottleneck if lower.status == "modeled" else None
            payload["ideal_formula_version"] = lower.formula_version
            payload["scope"] = {
                "kernel_name": detected["kernel_key"]["name"],
                "op_type": detected["kernel_key"]["op_type"],
                "task_type": detected["kernel_key"]["task_type"],
                "profile_input_shapes": detected["kernel_key"]["input_shapes"],
                "step_ids": detected["source_evidence"]["step_ids"],
                "stream_ids": detected["source_evidence"]["stream_ids"],
            }
            entry["alignment"] = payload
            entry["three_time"] = {
                "status": alignment.status,
                "measured_device_ms": alignment.measured_device_ms,
                "ideal_hardware_lower_bound_ms": alignment.ideal_hardware_lower_bound_ms,
                "model_predicted_ms": alignment.model_predicted_ms,
                "unmodeled_reason": alignment.unmodeled_reason,
            }
        entries.append(entry)
    raw_snapshot = provider.probe_status().as_dict()
    steps_path = result_file.parent / "steps.json"
    source_identity = {
        "msmodeling_source_path": str(Path(source_path).expanduser().resolve()),
        "msmodeling_source_version": provider.source_version,
        "msmodeling_source_commit": provider.source_commit,
        "probe": raw_snapshot,
        "result_json_path": str(result_file),
        "result_json_sha256": _sha256(result_file),
        "steps_json_path": str(steps_path),
        "steps_json_sha256": _sha256(steps_path) if steps_path.is_file() else None,
        "profile_db_path": actual["db_path"],
        "profile_db_sha256": actual["db_sha256"],
        "profile_db_size_bytes": actual["db_size_bytes"],
        "profile_db_schema_version": actual["db_schema_version"],
    }
    environment = result.get("environment", {})
    hardware_assumptions = provider.profile_summary("float32")
    hardware_assumptions["observed_device_name"] = environment.get("device_name")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_identity": {
            "run_id": result.get("run_id"),
            "acceptance_ready": result.get("acceptance_ready"),
            "label": result.get("label"),
            "host": result.get("host"),
            "physical_card": result.get("physical_card"),
            "input_shape": result.get("input_shape"),
            "input_dtype": result.get("input_dtype"),
            "input_sha256": result.get("input_sha256"),
            "model_state_sha256": result.get("model_state_sha256"),
            "environment": environment,
            "correctness": result.get("correctness"),
            "clock_domain": result.get("clock_domain"),
            "weight_tensors": weight_tensors,
        },
        "source_identity": source_identity,
        "provider": provider.status(),
        "hardware_assumptions": hardware_assumptions,
        "steps": actual["steps"],
        "kernels": entries,
        "limitations": [
            "measured_device_ms is the p50 of observed TASK durations clipped to explicit step windows, not a per-iteration benchmark mean",
            "model_predicted_ms is a pinned TensorCast analytic prediction on CPU meta tensors; no NPU execution was launched; this artifact is kernel-level three-time validation, not a full-run acceptance claim",
            "device profile mapping is caller-explicit and unverified against exact Ascend910B4 hardware specifications",
            "PMU values are raw source-value medians and are not interpreted as a hardware bottleneck",
            "unsupported or shape-unproven kernel keys remain unmodeled",
        ],
    }
    if output_path is not None:
        _write_json_atomic(Path(output_path).expanduser(), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mindie_profiler.enrichment_model",
        description="Pair actual capture DB durations with pinned TensorCast predictions.",
    )
    parser.add_argument("--result", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    artifact = build_c7_three_time_artifact(
        args.result,
        source_path=args.source,
        device_name=args.device,
        output_path=args.output,
    )
    if args.output:
        print(json.dumps({"output": args.output, "kernels": len(artifact["kernels"])}, sort_keys=True))
    else:
        print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if any(item["three_time"]["status"] == "modeled" for item in artifact["kernels"]) else 1

if __name__ == "__main__":
    raise SystemExit(main())
__all__ = ["SCHEMA_VERSION", "read_actual_kernels", "build_c7_three_time_artifact", "_coerce_hardware", "_normalize_registry", "apply_modeling", "main"]
