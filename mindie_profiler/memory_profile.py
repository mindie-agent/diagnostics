"""P0-P4 memory attribution with explicit sources, units, ranges, and peaks.

[Role]
Turn HBM observations, msprof components, framework logs, safetensors
manifests, and config theory into deterministic per-device attribution.  The
residual is reported, never invented to make a run pass.

[Invariants]
- P0 msprof, P1 whole-device runtime observation, P2 framework logs, P3
  safetensors, P4 config theory; priority never rewrites actual source kind.
- A point observation is never promoted to a measured peak.  Baseline,
  overlapping P0 samples, and device point deltas are non-additive references.
- Container totals are never summed with child components.  Only
  ``included_in_sum`` claims enter ``known_sum_bytes``.
- KV absence is missing unless the caller explicitly marks the architecture
  not-applicable; not-applicable is never zero bytes.
- MB/GB are decimal, MiB/GiB binary; negative residuals and missing sources
  stay explicit.  The residual target only gates ``target_met``.

[Timeouts & Locks]
No subprocess/network/lock/I/O; pure over bounded input mappings.

[Side-Effect Policy]
No filesystem/environment/process/global mutation; inputs are not modified.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .memory_compare import compare_peaks
from .memory_logs import parse_vllm_logs
from .memory_p0 import npu_mem_candidates, op_memory_candidates
from .memory_peak import baseline_and_allocator_candidates, peak_wall_interval
from .memory_units import UNIT_FACTORS, _is_number, to_bytes
from .memory_weights import estimate_weights_from_config

SCHEMA_VERSION = "mindie-memory-attribution/1"
DEFAULT_RESIDUAL_TARGET_BYTES = 200_000_000
PRIORITIES = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}
PRIORITY_LABELS = {
    "P0": "msprof component observation (npu_module_mem)",
    "P1": "whole-device runtime observation",
    "P2": "framework startup log",
    "P3": "safetensors file headers",
    "P4": "config.json theoretical estimate",
}
def _source(priority: str, kind: str, ref: str, scope: str, unit: str, raw: Any = None) -> dict[str, Any]:
    result = {
        "priority": priority, "priority_label": PRIORITY_LABELS.get(priority, "unknown"),
        "kind": kind, "ref": ref, "scope": scope, "unit": unit, "normalized_unit": "byte",
    }
    if raw is not None:
        result["raw"] = raw
    return result

def _scope_range(device_id: int | None, kind: str) -> dict[str, Any]:
    return {"kind": kind, "devices": "all" if kind != "device" else [device_id]}

def _component(*, key: str, label: str, byte_value: float, source: Mapping[str, Any],
               phase: str, device_id: int | None, scope_kind: str,
               applies_to_phase: str, evidence: str = "", peak_kind: str = "measured",
               measurement_kind: str | None = None, included_in_sum: bool = True,
               overlap_group: str | None = None) -> dict[str, Any]:
    scope = f"{scope_kind}:{'all' if scope_kind != 'device' else device_id}"
    kind = measurement_kind or peak_kind
    return {
        "key": key, "label": label, "bytes": byte_value, "unit": "byte",
        "display_gib": byte_value / (1024 ** 3), "source": dict(source),
        "phase": phase, "applies_to_phase": applies_to_phase, "scope": scope,
        "range": _scope_range(device_id, scope_kind),
        "overlap_group": overlap_group or key, "evidence": evidence,
        "included_in_sum": bool(included_in_sum),
        "measurement_kind": kind,
        "peak": {"bytes": byte_value, "unit": "byte", "phase": phase, "scope": scope,
                 "kind": kind, "is_peak": kind == "measured" and phase == applies_to_phase},
    }

def _candidate(*, key: str, label: str, byte_value: float | None, priority: str, kind: str,
               ref: str, phase: str, device_id: int | None, scope_kind: str, unit: str,
               raw: Any, evidence: str = "", exact: bool = False,
               peak_kind: str = "measured", measurement_kind: str | None = None,
               included_in_sum: bool = True, overlap_group: str | None = None) -> dict[str, Any] | None:
    if byte_value is None:
        return None
    return {
        "key": key, "label": label, "bytes": float(byte_value), "priority": priority,
        "phase": phase, "device_id": device_id, "scope_kind": scope_kind, "unit": unit,
        "raw": raw, "evidence": evidence, "exact": exact, "peak_kind": peak_kind,
        "measurement_kind": measurement_kind or peak_kind,
        "included_in_sum": bool(included_in_sum),
        "overlap_group": overlap_group or key,
        "source": _source(priority, kind, ref, f"{scope_kind}:all", unit, raw),
    }

def _used_bytes(stats: Mapping[str, Any] | None) -> float | None:
    if not isinstance(stats, Mapping):
        return None
    for key, factor in (("used_bytes", 1.0), ("used_mb", UNIT_FACTORS["MB"]),
                        ("used_mib", UNIT_FACTORS["MIB"]), ("used_gb", UNIT_FACTORS["GB"]),
                        ("used_gib", UNIT_FACTORS["GIB"]), ("hbm_used_mb", UNIT_FACTORS["MB"]),
                        ("hbm_used_bytes", 1.0)):
        if key in stats and stats[key] is not None:
            if not _is_number(stats[key]):
                raise ValueError(f"invalid {key} value {stats[key]!r}")
            return float(stats[key]) * factor
    if "used" in stats and stats["used"] is not None:
        return to_bytes(stats["used"], str(stats.get("unit") or "MB"))
    return None

def _observed_at(stats: Mapping[str, Any] | None) -> str | None:
    if isinstance(stats, Mapping):
        value = stats.get("observed_at") or stats.get("timestamp")
        return str(value) if value is not None else None
    return None

def _stats_source_kind(stats: Mapping[str, Any] | None) -> str:
    if isinstance(stats, Mapping):
        explicit = stats.get("source_kind")
        if explicit:
            return str(explicit)
        source = str(stats.get("source") or "").lower()
        if "torch_npu" in source or "mem_get_info" in source:
            return "torch_npu_runtime_api"
        if "npu-smi" in source or "npu_smi" in source:
            return "npu_smi"
    return "unknown_runtime_source"

def _stats_measurement_kind(stats: Mapping[str, Any] | None) -> str:
    if isinstance(stats, Mapping):
        return str(stats.get("measurement_kind") or "point_observation")
    return "point_observation"

def _stats_is_peak(stats: Mapping[str, Any] | None) -> bool:
    return bool(isinstance(stats, Mapping) and stats.get("is_peak", False))

def _msprof_value(value: Any, default_unit: str) -> tuple[float | None, str]:
    if isinstance(value, Mapping):
        if "bytes" in value:
            return to_bytes(value["bytes"], "B"), "B"
        if "byte_size" in value:
            return to_bytes(value["byte_size"], "B"), "B"
        if "value" in value:
            unit = str(value.get("unit") or default_unit)
            return to_bytes(value["value"], unit), unit
        return None, default_unit
    if isinstance(value, str):
        try:
            return to_bytes(float(value), default_unit), default_unit
        except ValueError:
            return None, default_unit
    return (to_bytes(value, default_unit), default_unit) if _is_number(value) else (None, default_unit)

def _msprof_candidates(msprof: Mapping[str, Any] | None, *, device_id: int, default_unit: str,
                       applies_to_phase: str, baseline_present: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    if not isinstance(msprof, Mapping) or not msprof:
        return [], [], [{"code": "msprof_missing", "message": "no msprof npu_module_mem components; HCCL/RUNTIME/SLOG are untraceable", "required": True}]
    values = {str(name).strip().upper(): _msprof_value(value, default_unit)
              for name, value in msprof.items() if str(name).strip()}
    candidates: list[dict[str, Any]] = []
    containers: list[dict[str, Any]] = []
    limitations: list[dict[str, str]] = []
    app_bytes, app_unit = values.get("APP", (None, default_unit))
    if app_bytes is not None:
        containers.append({
            "key": "process_total", "label": "msprof APP process total", "bytes": app_bytes,
            "unit": app_unit, "included_in_sum": False, "overlap_group": "process_total",
            "measurement_kind": "sampled_max",
            "source": _source("P0", "msprof", "npu_module_mem:APP", "process:all", app_unit, msprof.get("APP")),
        })
    core = {"HCCL": ("hccl", "HCCL buffers"), "RUNTIME": ("runtime", "CANN runtime"),
            "SLOG": ("slog", "SLOG system log")}
    for name, (key, label) in core.items():
        byte_value, unit = values.get(name, (None, default_unit))
        if byte_value is None or byte_value == 0:
            continue
        if byte_value < 0:
            limitations.append({"code": "negative_msprof_component", "message": f"msprof {name} is negative ({byte_value} bytes); not used", "required": False})
            continue
        raw = msprof.get(name)
        meta = raw if isinstance(raw, Mapping) else {}
        measurement_kind = str(meta.get("measurement_kind") or "sampled_max")
        included = False
        item = _candidate(key=key, label=label if included else f"{label} (scope unproven; reference only)",
                          byte_value=byte_value, priority="P0", kind="msprof",
                          ref=f"npu_module_mem:Component={name}", phase="capture_window", device_id=device_id,
                          scope_kind="process", unit=unit, raw=raw,
                          evidence=f"msprof {name} sampled max = {byte_value / UNIT_FACTORS['MB']:.3f} MB; "
                                   + ("added" if included else "reference only; overlap with baseline/other P0 scope is not proven"),
                          exact=False, peak_kind=measurement_kind, measurement_kind=measurement_kind,
                          included_in_sum=included,
                          overlap_group=("baseline_runtime" if baseline_present else "p0_scope_unknown"))
        if item:
            candidates.append(item)
    minor = {name: byte_value for name, (byte_value, _unit) in values.items()
             if name not in ("APP", "HCCL", "RUNTIME", "SLOG") and byte_value is not None and byte_value > 0}
    if minor:
        included = False
        item = _candidate(key="other_msprof", label="Other msprof components (scope unproven; reference only)",
                          byte_value=sum(minor.values()), priority="P0", kind="msprof",
                          ref="npu_module_mem:minor components", phase="capture_window",
                          device_id=device_id, scope_kind="process", unit="B", raw=minor,
                          evidence="sum of distinct minor msprof components (APP/HCCL/RUNTIME/SLOG excluded); reference only because overlap is not proven",
                          peak_kind="sampled_max", measurement_kind="sampled_max",
                          included_in_sum=included, overlap_group=("baseline_runtime" if baseline_present else "other_msprof"))
        if item:
            candidates.append(item)
    if not candidates:
        limitations.append({"code": "msprof_components_unusable", "message": "msprof data is present but has no usable HCCL/RUNTIME/SLOG values", "required": True})
    elif not baseline_present:
        limitations.append({"code": "p0_component_scope_unknown",
                            "message": "no baseline proves P0 component overlap; HCCL/RUNTIME/SLOG are observed references, not additive",
                            "required": True})
    return candidates, containers, limitations

def _per_device_from_weight_manifest(manifest: Mapping[str, Any], *, tp: int, dp: int, enable_ep: bool) -> float | None:
    per_device = manifest.get("per_device_bytes")
    same = (manifest.get("tp") in (None, tp) and manifest.get("dp") in (None, dp)
            and bool(manifest.get("enable_expert_parallel")) == bool(enable_ep))
    if _is_number(per_device) and same:
        return float(per_device)
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        return None
    ep = tp * dp if enable_ep else tp
    total = 0.0
    for item in tensors:
        if not isinstance(item, Mapping) or not _is_number(item.get("byte_size")):
            continue
        strategy = str(item.get("shard_strategy") or "replicated")
        if strategy in ("col_parallel", "row_parallel"):
            total += float(item["byte_size"]) / tp
        elif strategy == "expert_parallel":
            total += float(item["byte_size"]) / ep
        else:
            total += float(item["byte_size"])
    return total

def _weight_candidates(*, vllm: Mapping[str, Any], weight_manifest: Mapping[str, Any] | None,
                       model_config: Mapping[str, Any] | None, tp: int, dp: int,
                       enable_ep: bool, device_id: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    candidates: list[dict[str, Any]] = []
    limitations: list[dict[str, str]] = []
    if _is_number(vllm.get("weights_bytes")):
        candidates.append(_candidate(key="weights", label="Model weights", byte_value=float(vllm["weights_bytes"]),
                                     priority="P2", kind="vllm_log", ref="vllm log: weights_bytes", phase="ready",
                                     device_id=device_id, scope_kind="device", unit="B", raw=vllm["weights_bytes"],
                                     evidence="explicit weights_bytes from framework log", exact=True))
    elif _is_number(vllm.get("weights_gib")):
        candidates.append(_candidate(key="weights", label="Model weights", byte_value=to_bytes(vllm["weights_gib"], "GiB"),
                                     priority="P2", kind="vllm_log", ref="vllm log: weights_gib", phase="ready",
                                     device_id=device_id, scope_kind="device", unit="GiB", raw=vllm["weights_gib"],
                                     evidence="framework log GiB value", exact=True))
    elif _is_number(vllm.get("weights_gb")):
        # The log says "GB" explicitly here; GiB is parsed into weights_gib.
        unit = str(vllm.get("weights_unit") or "GB")
        candidates.append(_candidate(key="weights", label="Model weights", byte_value=to_bytes(vllm["weights_gb"], unit),
                                     priority="P2", kind="vllm_log", ref="vllm log: Loading model weights took",
                                     phase="ready", device_id=device_id, scope_kind="device", unit=unit,
                                     raw=vllm["weights_gb"], evidence=f"framework log value in {unit}"))
    if isinstance(weight_manifest, Mapping) and weight_manifest.get("status") != "error":
        per_device = _per_device_from_weight_manifest(weight_manifest, tp=tp, dp=dp, enable_ep=enable_ep)
        if per_device is not None:
            candidates.append(_candidate(key="weights", label="Model weights", byte_value=per_device, priority="P3",
                                         kind="safetensors", ref="safetensors headers: per-device bytes",
                                         phase="static", device_id=device_id, scope_kind="model_files", unit="B",
                                         raw=per_device, evidence="byte-exact header sizes with rule-based sharding", exact=True))
    if isinstance(model_config, Mapping) and model_config:
        estimate = estimate_weights_from_config(model_config, tp=tp, dp=dp, enable_expert_parallel=enable_ep)
        if estimate.get("per_device_bytes") is not None:
            candidates.append(_candidate(key="weights", label="Model weights (theoretical)",
                                         byte_value=estimate["per_device_bytes"], priority="P4", kind="config",
                                         ref="config.json theoretical estimate", phase="theoretical",
                                         device_id=device_id, scope_kind="theory", unit="B",
                                         raw=estimate.get("total_params"), evidence="P4 reference only; never a measurement",
                                         peak_kind="reference"))
            if not any(item["priority"] in ("P2", "P3") for item in candidates if item["key"] == "weights"):
                limitations.append({"code": "weights_reference_only", "message": "model weights have no P2/P3 measurement; P4 config estimate is reference only", "required": True})
    return candidates, limitations

def _kv_candidate(vllm: Mapping[str, Any], *, device_id: int) -> dict[str, Any] | None:
    if _is_number(vllm.get("kv_cache_current_bytes")):
        value, unit, raw = float(vllm["kv_cache_current_bytes"]), "B", vllm["kv_cache_current_bytes"]
        ref = "vllm log: Current KV cache memory"
    elif _is_number(vllm.get("kv_cache_current_gib")):
        value, unit, raw = to_bytes(vllm["kv_cache_current_gib"], "GiB"), "GiB", vllm["kv_cache_current_gib"]
        ref = "vllm log: Current KV cache memory"
    elif _is_number(vllm.get("kv_cache_current_gb")):
        value, unit, raw = to_bytes(vllm["kv_cache_current_gb"], "GB"), "GB", vllm["kv_cache_current_gb"]
        ref = "vllm log: Current KV cache memory"
    elif _is_number(vllm.get("kv_cache_bytes")):
        value, unit, raw = float(vllm["kv_cache_bytes"]), "B", vllm["kv_cache_bytes"]
        ref = "vllm log: kv_cache_bytes"
    else:
        return None
    return _candidate(key="kv_cache", label="KV cache reservation", byte_value=value, priority="P2",
                      kind="vllm_log", ref=ref, phase="ready",
                      device_id=device_id, scope_kind="device", unit=unit, raw=raw,
                      evidence="explicit current KV cache memory; Available KV cache memory is only a budget", exact=unit == "B")

def _graph_candidate(vllm: Mapping[str, Any], *, device_id: int) -> dict[str, Any] | None:
    if _is_number(vllm.get("graph_capture_bytes")):
        value, unit, raw = float(vllm["graph_capture_bytes"]), "B", vllm["graph_capture_bytes"]
    elif _is_number(vllm.get("graph_capture_gib")):
        value, unit, raw = to_bytes(vllm["graph_capture_gib"], "GiB"), "GiB", vllm["graph_capture_gib"]
    else:
        return None
    return _candidate(key="graph", label="ACL graph capture", byte_value=value, priority="P2",
                      kind="vllm_log", ref="vllm log: Graph capturing finished", phase="ready",
                      device_id=device_id, scope_kind="device", unit=unit, raw=raw,
                      evidence="framework-reported graph capture memory", exact=unit == "B")

def _select_candidates(candidates: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["key"], []).append(candidate)
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    for key, items in grouped.items():
        items.sort(key=lambda item: (PRIORITIES[item["priority"]], 0 if item["exact"] else 1))
        selected.append(items[0])
        if len(items) > 1:
            alternatives = [{"key": key, "selected_priority": items[0]["priority"],
                             "excluded_priority": alternate["priority"], "excluded_bytes": alternate["bytes"],
                             "excluded_source": alternate["source"]["kind"],
                             "reason": "lower_priority_same_quantity"} for alternate in items[1:]]
            excluded.extend(alternatives)
            overlaps.append({"key": key, "selected": items[0]["priority"], "alternatives": alternatives})
    order = {"baseline_observed": 0, "weights": 1, "kv_cache": 2, "graph": 3, "hccl": 4,
             "runtime": 5, "slog": 6, "other_msprof": 7, "transient_operator_allocation": 8,
             "npu_device_sampled": 9, "npu_app_sampled": 10, "activation": 11}
    selected.sort(key=lambda item: order.get(item["key"], 99))
    return selected, excluded, overlaps

def attribute_device(device_id: int, *, baseline: Mapping[str, Any] | None = None,
                     ready: Mapping[str, Any] | None = None, inference: Mapping[str, Any] | None = None,
                     sampled_peak: Mapping[str, Any] | None = None,
                     npu_mem: Mapping[str, Any] | None = None,
                     op_memory: Mapping[str, Any] | None = None,
                     msprof: Mapping[str, Any] | None = None, vllm: Mapping[str, Any] | None = None,
                     weight_manifest: Mapping[str, Any] | None = None,
                     model_config: Mapping[str, Any] | None = None, tp: int = 1, dp: int = 1,
                     enable_expert_parallel: bool = False, target_phase: str | None = None,
                     residual_target_bytes: float = DEFAULT_RESIDUAL_TARGET_BYTES,
                     msprof_unit: str = "MB", kv_cache_applicable: bool | None = None) -> dict[str, Any]:
    """Attribute one device's HBM use without filling, clamping, or double counting."""
    if not isinstance(device_id, int) or isinstance(device_id, bool) or device_id < 0:
        raise ValueError("device_id must be a non-negative integer")
    if not _is_number(residual_target_bytes) or residual_target_bytes < 0:
        raise ValueError("residual_target_bytes must be a non-negative number")
    if target_phase not in (None, "ready", "inference_peak", "sampled_peak", "npu_mem_device"):
        raise ValueError("target_phase must be ready, inference_peak, sampled_peak, npu_mem_device, or None")
    ready_bytes, infer_bytes, baseline_bytes = _used_bytes(ready), _used_bytes(inference), _used_bytes(baseline)
    sampled_bytes = _used_bytes(sampled_peak)
    npu_device = npu_mem.get("device") if isinstance(npu_mem, Mapping) and isinstance(npu_mem.get("device"), Mapping) else None
    npu_device_bytes = float(npu_device["max_bytes"]) if isinstance(npu_device, Mapping) and _is_number(npu_device.get("max_bytes")) else None
    limitations: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    not_applicable: list[dict[str, Any]] = []
    requested_phase = target_phase or ("inference_peak" if infer_bytes is not None else "ready")
    actual_phase = requested_phase
    target_bytes: float | None = None
    target_stats: Mapping[str, Any] | None = None
    if requested_phase == "npu_mem_device":
        if npu_device_bytes is None:
            actual_phase = "inference_peak" if infer_bytes is not None else "ready"
            missing.append({"code": "npu_mem_device_unavailable", "message": "npu_mem_device was requested but NPU_MEM device hbm is missing", "required": True})
            limitations.append({"code": "npu_mem_device_unavailable", "message": "no measured NPU_MEM device sampled max", "required": True})
        else:
            target_bytes = npu_device_bytes
            target_stats = {
                "used_bytes": npu_device_bytes,
                "phase": "npu_mem_device",
                "source_kind": "msprof_npu_mem_sqlite",
                "source_priority": "P0",
                "measurement_kind": "sampled_max",
                "is_peak": True,
                "peak_scope": "sampled",
                "continuous_peak": False,
                "absolute_peak_limited": True,
                "sample_count": npu_device.get("samples"),
                "coverage_ratio": None,
                "cadence_ms": {"target_coverage_complete": False},
            }
            limitations.append({"code": "npu_mem_schedule_unknown", "message": "NPU_MEM device sampled max has no intended cadence/slot schedule; target coverage is partial/unknown", "required": True})
    if requested_phase == "sampled_peak":
        if sampled_bytes is None:
            actual_phase = "inference_peak" if infer_bytes is not None else "ready"
            missing.append({"code": "sampled_peak_unavailable", "message": "sampled_peak was requested but no real sampled device peak exists; no point observation is relabeled as a peak", "required": True})
            limitations.append({"code": "sampled_peak_unavailable", "message": "bounded memory sampler did not produce a usable peak", "required": True})
        else:
            target_bytes = sampled_bytes
            target_stats = sampled_peak
    if requested_phase == "inference_peak" and infer_bytes is None:
        actual_phase = "ready"
        limitations.append({"code": "target_phase_unavailable", "message": "inference peak was requested but after-inference HBM is missing; using ready phase", "required": True})
        missing.append({"code": "target_phase_unavailable", "message": "requested inference peak could not be evaluated; ready phase is reported instead", "required": True})
    if target_stats is None:
        if actual_phase == "inference_peak":
            target_bytes, target_stats = infer_bytes, inference
        elif actual_phase == "ready":
            target_bytes, target_stats = ready_bytes, ready
    target_source_kind = _stats_source_kind(target_stats)
    target_measurement_kind = _stats_measurement_kind(target_stats)
    target_is_peak = _stats_is_peak(target_stats)
    if not target_is_peak:
        target_peak_scope = "point_observation"
    else:
        target_peak_scope = (target_stats.get("peak_scope") if isinstance(target_stats, Mapping) else None) or (
            "sampled" if target_measurement_kind == "sampled_device_peak" else "measured"
        )
    target_continuous_peak = bool(target_stats.get("continuous_peak", False)) if isinstance(target_stats, Mapping) else False
    target_absolute_limited = (
        True if not target_is_peak
        else bool(target_stats.get("absolute_peak_limited", not target_continuous_peak))
        if isinstance(target_stats, Mapping) else True
    )
    target_scope = f"device:{device_id}"
    target_ref = str(target_stats.get("phase") or actual_phase) if isinstance(target_stats, Mapping) else actual_phase
    target_cadence = target_stats.get("cadence_ms") if isinstance(target_stats, Mapping) and isinstance(target_stats.get("cadence_ms"), Mapping) else {}
    target_coverage_ratio = target_stats.get("coverage_ratio") if isinstance(target_stats, Mapping) else None
    target_coverage_complete = bool(target_is_peak)
    if target_is_peak and target_peak_scope == "sampled":
        target_coverage_complete = bool(target_cadence.get("target_coverage_complete")) and (
            not _is_number(target_coverage_ratio) or float(target_coverage_ratio) >= 0.99
        )
        if not target_coverage_complete:
            missing.append({"code": "sampled_peak_coverage_incomplete",
                            "message": "sampled peak has missing/duplicate/failed slots or incomplete getter coverage; target_met cannot pass",
                            "required": True})
            limitations.append({"code": "sampled_peak_coverage_incomplete",
                                "message": "sampled maximum is valid but target coverage is not complete",
                                "required": True})
    candidates: list[dict[str, Any]] = []
    containers: list[dict[str, Any]] = []
    if not target_is_peak:
        missing.append({"code": "target_not_measured_peak", "message": "target is a point observation, not a measured whole-device peak; residual target cannot pass as a peak comparison", "required": True})
        limitations.append({"code": "target_point_observation", "message": "target phase is one point observation; no peak claim is made", "required": True})
    elif target_peak_scope == "sampled":
        limitations.append({"code": "sampled_peak_not_absolute", "message": "target is a bounded sampled device maximum; absolute continuous peak is not measured", "required": False})

    baseline_candidates, baseline_limitations = baseline_and_allocator_candidates(
        baseline, target_stats, baseline_bytes=baseline_bytes, device_id=device_id,
        candidate=_candidate, source_kind=_stats_source_kind(baseline),
        measurement_kind=_stats_measurement_kind(baseline))
    candidates.extend(baseline_candidates)
    limitations.extend(baseline_limitations)
    if baseline_bytes is None:
        missing.append({"code": "baseline_snapshot_missing", "message": "no baseline runtime/device snapshot; baseline overhead is unknown, not zero", "required": True})

    weight_candidates, weight_limitations = _weight_candidates(
        vllm=vllm or {}, weight_manifest=weight_manifest, model_config=model_config,
        tp=tp, dp=dp, enable_ep=enable_expert_parallel, device_id=device_id)
    candidates.extend(weight_candidates)
    limitations.extend(weight_limitations)
    if not any(item["key"] == "weights" for item in weight_candidates):
        missing.append({"code": "model_weights_missing", "message": "no P2/P3/P4 weight source", "required": True})
    elif all(item["priority"] == "P4" for item in weight_candidates if item["key"] == "weights"):
        missing.append({"code": "weights_measured_missing", "message": "weights are a P4 theoretical estimate; no measured P2/P3 source", "required": True})

    kv_candidate = _kv_candidate(vllm or {}, device_id=device_id)
    if kv_cache_applicable is False:
        not_applicable.append({"code": "kv_cache_not_applicable",
                               "message": "caller explicitly annotated this architecture as not using KV cache; no byte value is claimed or substituted",
                               "required": False})
        if kv_candidate is not None:
            limitations.append({"code": "kv_cache_value_conflicts_with_not_applicable",
                                "message": "a framework KV value was present but the caller explicitly marked KV not applicable; the value was not used",
                                "required": False})
    elif kv_candidate is not None:
        candidates.append(kv_candidate)
    else:
        missing.append({"code": "kv_cache_missing", "message": "no framework KV cache value and no explicit not-applicable annotation", "required": True})
        if _is_number((vllm or {}).get("kv_cache_available_gib")) or _is_number((vllm or {}).get("kv_cache_available_gb")):
            limitations.append({"code": "kv_cache_available_budget_only",
                                "message": "Available KV cache memory is a budget, not actual KV allocation; no current KV value was used",
                                "required": False})
    graph = _graph_candidate(vllm or {}, device_id=device_id)
    if graph:
        candidates.append(graph)

    msprof_candidates, msprof_containers, msprof_limitations = _msprof_candidates(
        msprof, device_id=device_id, default_unit=msprof_unit,
        applies_to_phase=actual_phase, baseline_present=baseline_bytes is not None)
    target_window_ns = peak_wall_interval(target_stats)
    npu_candidates, npu_mem_limitations = npu_mem_candidates(npu_mem, device_id=device_id, candidate=_candidate)
    op_candidates, op_memory_limitations = op_memory_candidates(
        op_memory, device_id=device_id, candidate=_candidate,
        target_window_ns=target_window_ns,
    )
    candidates.extend(msprof_candidates)
    candidates.extend(npu_candidates)
    candidates.extend(op_candidates)
    containers.extend(msprof_containers)
    limitations.extend(msprof_limitations)
    limitations.extend({**item, "source_priority": "P0", "source_table": "NPU_MEM",
                        "required_for_selected_target": target_source_kind == "msprof_npu_mem_sqlite"}
                       for item in npu_mem_limitations)
    limitations.extend(op_memory_limitations)
    if not msprof_candidates:
        missing.append({"code": "msprof_components_missing", "message": "HCCL/RUNTIME/SLOG are not traceable without P0 msprof components", "required": True})

    if actual_phase == "inference_peak":
        if ready_bytes is None or infer_bytes is None:
            missing.append({"code": "activation_delta_missing", "message": "post-workload device delta needs both ready and after-inference snapshots", "required": True})
        else:
            point_delta = infer_bytes - ready_bytes
            label = ("Activation peak delta reference (non-additive)" if target_is_peak
                     else "Post-workload device point delta (not an activation peak)")
            item = _candidate(key="activation", label=label, byte_value=point_delta, priority="P1",
                              kind=target_source_kind, ref="after_infer_hbm - after_ready_hbm",
                              phase=actual_phase, device_id=device_id, scope_kind="device", unit="B",
                              raw=point_delta,
                              evidence="whole-device delta between ready and the post-workload point; it is not exclusive to activations and may include profiler allocations",
                              exact=False,
                              peak_kind=("measured" if target_is_peak else "point_delta"),
                              measurement_kind=("measured_peak" if target_is_peak else "point_delta"),
                              included_in_sum=False, overlap_group="device_point_delta")
            if item:
                candidates.append(item)

    selected, excluded, overlaps = _select_candidates(candidates)
    components: list[dict[str, Any]] = []
    for item in selected:
        source = item["source"]
        source["scope"] = f"{item['scope_kind']}:{'all' if item['scope_kind'] != 'device' else device_id}"
        components.append(_component(key=item["key"], label=item["label"], byte_value=item["bytes"],
                                     source=source, phase=item["phase"], device_id=item["device_id"],
                                     scope_kind=item["scope_kind"], applies_to_phase=actual_phase,
                                     evidence=item["evidence"], peak_kind=item["peak_kind"],
                                     measurement_kind=item["measurement_kind"],
                                     included_in_sum=item["included_in_sum"],
                                     overlap_group=item["overlap_group"]))
    baseline_component = next((item for item in components if item["key"] == "baseline_observed"), None)
    p0_references = [
        item for item in components
        if item["key"] in {"hccl", "runtime", "slog", "other_msprof", "npu_device_sampled", "npu_app_sampled"}
        and not item["included_in_sum"]
    ]
    if baseline_component is not None and p0_references:
        overlaps.append({
            "key": "baseline_observed",
            "selected": baseline_component["source"]["priority"],
            "alternatives": [
                {"key": item["key"], "selected_priority": item["source"]["priority"],
                 "excluded_priority": item["source"]["priority"], "excluded_bytes": item["bytes"],
                 "excluded_source": item["source"]["kind"],
                 "reason": "P0 component overlap with baseline or allocator quantities is unproven; reported as reference only"}
                for item in p0_references
            ],
        })
    references = [item for item in components if not item["included_in_sum"]]
    rounded_logs = any(item["included_in_sum"] and item["source"]["priority"] == "P2"
                       and item["source"]["unit"] not in {"B", "byte", "bytes"} for item in components)
    if rounded_logs:
        limitations.append({"code": "p2_rounded_values", "required": False,
                            "message": "additive startup-log amounts are rounded and are not same-query allocation measurements; the residual is approximate"})
    source_applicability: dict[str, Any] = {}
    for priority in sorted(PRIORITIES, key=PRIORITIES.get):
        keys = sorted({item["key"] for item in components if item["source"]["priority"] == priority})
        if keys:
            status = "present"
        elif priority == "P2" and not vllm:
            status = "not_provided"
        elif priority == "P4" and not model_config:
            status = "not_provided"
        else:
            status = "missing"
        source_applicability[priority] = {
            "status": status,
            "keys": keys,
            "priority_label": PRIORITY_LABELS[priority],
        }
    known_sum = sum(item["bytes"] for item in components if item["included_in_sum"])
    reference_sum = sum(item["bytes"] for item in references)
    if target_bytes is None:
        residual_value, residual_status = None, "unavailable"
        missing.append({"code": "target_total_missing", "message": "no used-HBM total for the target phase", "required": True})
    else:
        residual_value = target_bytes - known_sum
        residual_status = "negative" if residual_value < 0 else ("above_target" if residual_value >= residual_target_bytes else "within_target")
    coverage = "unavailable" if target_bytes is None else ("complete" if not missing and target_is_peak else "partial")
    target_met = bool(
        target_bytes is not None
        and target_is_peak
        and target_coverage_complete
        and residual_status == "within_target"
        and coverage == "complete"
    )
    target_peak = None
    if target_bytes is not None:
        target_peak = {"bytes": target_bytes, "unit": "byte", "phase": actual_phase, "scope": target_scope,
                       "observed_at": _observed_at(target_stats),
                       "measurement_kind": target_measurement_kind,
                       "peak_scope": target_peak_scope,
                       "continuous_peak": target_continuous_peak,
                       "absolute_peak_limited": target_absolute_limited,
                       "sample_count": target_stats.get("sample_count") if isinstance(target_stats, Mapping) else None,
                       "sample_window_ns": target_stats.get("sample_window_ns") if isinstance(target_stats, Mapping) else None,
                       "sample_window_clock": target_stats.get("sample_window_clock") if isinstance(target_stats, Mapping) else None,
                       "peak_query_wall_interval_ns": target_window_ns,
                       "peak_sample": target_stats.get("peak_sample") if isinstance(target_stats, Mapping) else None,
                       "coverage_ratio": target_coverage_ratio,
                       "coverage_complete": target_coverage_complete,
                       "expected_slot_count": target_cadence.get("expected_slots"),
                       "completed_slot_count": target_cadence.get("completed_slots"),
                       "missing_slot_count": target_cadence.get("missing_slots"),
                       "late_slot_count": target_cadence.get("late_slots"),
                       "failed_sample_count": target_cadence.get("failed_samples"),
                       "missing_slot_indices": target_cadence.get("missing_slot_indices"),
                       "late_slot_indices": target_cadence.get("late_slot_indices"),
                       "outside_slot_indices": target_cadence.get("outside_slot_indices"),
                       "duplicate_slot_indices": target_cadence.get("duplicate_slot_indices"),
                       "missing_slot_index_sample_count": target_cadence.get("missing_slot_index_sample_count"),
                       "outside_slot_sample_count": target_cadence.get("outside_slot_sample_count"),
                       "duplicate_slot_sample_count": target_cadence.get("duplicate_slot_sample_count"),
                       "slot_assignment": "raw_sample.slot_index",
                       "source": _source(str(target_stats.get("source_priority") or "P1") if isinstance(target_stats, Mapping) else "P1",
                                         target_source_kind, target_ref, target_scope, "B", target_stats),
                       "kind": target_measurement_kind,
                       "is_peak": target_is_peak}
    activation_reference = next((item for item in references if item["key"] == "activation"), None)
    cross_validation: dict[str, Any] = {
        "target_bytes": target_bytes,
        "additive_component_sum_bytes": known_sum,
        "non_additive_reference_bytes": reference_sum,
        "residual_bytes": residual_value,
        "baseline_observed_bytes": baseline_bytes,
        "activation_point_delta_bytes": activation_reference["bytes"] if activation_reference else None,
        "p0_overlap_bytes": sum(item["bytes"] for item in p0_references),
    }
    app = next((item for item in containers if item["key"] == "process_total"), None)
    if app is not None:
        cross_validation["msprof_app_bytes"] = app["bytes"]
        cross_validation["app_minus_additive_component_sum_bytes"] = app["bytes"] - known_sum
        weight_kv = sum(item["bytes"] for item in components if item["key"] in ("weights", "kv_cache") and item["included_in_sum"])
        if weight_kv:
            cross_validation["app_minus_weight_kv_bytes"] = app["bytes"] - weight_kv
    return {
        "schema_version": SCHEMA_VERSION, "device_id": device_id, "unit": "byte",
        "target_phase_requested": requested_phase, "target_phase": actual_phase,
        "target_measurement_kind": target_measurement_kind, "target_is_peak": target_is_peak,
        "target_peak_scope": target_peak_scope,
        "target_continuous_peak": target_continuous_peak,
        "target_absolute_peak_limited": target_absolute_limited,
        "target_coverage_ratio": target_coverage_ratio,
        "target_coverage_complete": target_coverage_complete,
        "target": {"bytes": target_bytes, "unit": "byte",
                   "measurement_kind": target_measurement_kind, "is_peak": target_is_peak,
                   "peak_scope": target_peak_scope,
                   "continuous_peak": target_continuous_peak,
                   "absolute_peak_limited": target_absolute_limited,
                   "coverage_ratio": target_coverage_ratio,
                   "coverage_complete": target_coverage_complete,
                   "source": _source(str(target_stats.get("source_priority") or "P1") if isinstance(target_stats, Mapping) else "P1",
                                     target_source_kind, target_ref, target_scope, "B", target_stats),
                   "peak": target_peak},
        "components": components, "references": references, "containers": containers,
        "source_applicability": source_applicability,
        "sum_rule": "sum only components with included_in_sum=true; references/containers are not additive",
        "known_sum_bytes": known_sum, "reference_sum_bytes": reference_sum,
        "residual": {"bytes": residual_value, "unit": "byte", "status": residual_status,
                     "approximate": rounded_logs,
                     "target_bytes": residual_target_bytes, "target_met": target_met,
                     "negative_overage_bytes": max(0.0, -residual_value) if residual_value is not None else None,
                     "coverage": coverage, "missing_sources": missing,
                     "not_applicable_sources": not_applicable,
                     "target_source": "configurable residual target, not a measurement",
                     "status_scope": "residual magnitude only; target_met additionally requires a measured target peak and complete coverage",
                     "additive_scope": "only included_in_sum components; baseline/P0 overlap/device point deltas are not double counted"},
        "missing_sources": missing, "not_applicable_sources": not_applicable,
        "limitations": list({(str(item.get("code")), str(item.get("message"))): item for item in limitations}.values()), "overlaps": overlaps,
        "excluded_claims": excluded, "cross_validation": cross_validation, "peak": target_peak,
    }

def _device_map(value: Any) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            try:
                device_id = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(item, Mapping):
                result[device_id] = item
    return result

def attribute_memory(manifest: Mapping[str, Any], *, residual_target_bytes: float = DEFAULT_RESIDUAL_TARGET_BYTES,
                     target_phase: str | None = None, tp: int | None = None, dp: int | None = None,
                     enable_expert_parallel: bool | None = None,
                     vllm_info: Mapping[str, Any] | None = None,
                     weight_manifest: Mapping[str, Any] | None = None,
                     model_config: Mapping[str, Any] | None = None,
                     msprof_components: Mapping[str, Any] | None = None,
                     msprof_components_by_device: Mapping[str, Any] | None = None,
                     npu_mem_by_device: Mapping[str, Any] | None = None,
                     op_memory_by_device: Mapping[str, Any] | None = None,
                     msprof_unit: str = "MB",
                     kv_cache_applicable: bool | None = None) -> dict[str, Any]:
    """Attribute every device in a collected memory manifest."""
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    vllm: dict[str, Any] = {}
    log_text = manifest.get("vllm_log") or manifest.get("log_text")
    if isinstance(log_text, str):
        vllm.update(parse_vllm_logs(log_text))
    for mapping in (manifest.get("vllm_info"), vllm_info):
        if isinstance(mapping, Mapping):
            vllm.update(dict(mapping))
    weights = weight_manifest if weight_manifest is not None else manifest.get("weight_manifest")
    weights = weights if isinstance(weights, Mapping) else None
    config = model_config if model_config is not None else manifest.get("model_config")
    config = config if isinstance(config, Mapping) else None
    baseline, ready, inference, sampled_peak = (_device_map(manifest.get("baseline_hbm")),
                                               _device_map(manifest.get("after_ready_hbm")),
                                               _device_map(manifest.get("after_infer_hbm")),
                                               _device_map(manifest.get("sampled_peak_hbm")))
    per_device = _device_map(msprof_components_by_device) or _device_map(manifest.get("msprof_components_by_device"))
    npu_mem_per_device = _device_map(npu_mem_by_device) or _device_map(manifest.get("npu_mem_by_device"))
    op_memory_value = op_memory_by_device if op_memory_by_device is not None else manifest.get("op_memory_by_device")
    if isinstance(op_memory_value, Mapping) and isinstance(op_memory_value.get("devices"), Mapping):
        op_memory_value = op_memory_value["devices"]
    op_memory_per_device = _device_map(op_memory_value)
    global_msprof = msprof_components if msprof_components is not None else manifest.get("msprof_components")
    global_msprof = global_msprof if isinstance(global_msprof, Mapping) else None
    device_ids = sorted(set(baseline) | set(ready) | set(inference) | set(per_device) | set(npu_mem_per_device) | set(op_memory_per_device))
    actual_tp = int(tp if tp is not None else manifest.get("tp") or 1)
    actual_dp = int(dp if dp is not None else manifest.get("dp") or 1)
    actual_ep = bool(enable_expert_parallel if enable_expert_parallel is not None else manifest.get("enable_expert_parallel", False))
    if kv_cache_applicable is None:
        kv_cache_applicable = manifest.get("kv_cache_applicable")
    if not isinstance(kv_cache_applicable, bool):
        kv_cache_applicable = None
    device_ids = sorted(set(device_ids) | set(sampled_peak))
    devices = [attribute_device(device_id, baseline=baseline.get(device_id), ready=ready.get(device_id),
                                inference=inference.get(device_id), sampled_peak=sampled_peak.get(device_id),
                                npu_mem=npu_mem_per_device.get(device_id),
                                op_memory=op_memory_per_device.get(device_id),
                                msprof=per_device.get(device_id, global_msprof),
                                vllm=vllm, weight_manifest=weights, model_config=config, tp=actual_tp,
                                dp=actual_dp, enable_expert_parallel=actual_ep, target_phase=target_phase,
                                residual_target_bytes=residual_target_bytes, msprof_unit=msprof_unit,
                                kv_cache_applicable=kv_cache_applicable)
               for device_id in device_ids]
    if not devices:
        status = "failed"
    elif all(item["residual"]["target_met"] for item in devices):
        status = "ok"
    elif any(item["target"]["bytes"] is not None for item in devices):
        status = "partial"
    else:
        status = "failed"
    applicability: dict[str, Any] = {}
    for priority in sorted(PRIORITIES, key=PRIORITIES.get):
        per_device = {
            str(item["device_id"]): item["source_applicability"][priority]["status"]
            for item in devices
        }
        statuses = set(per_device.values())
        applicability[priority] = {
            "status": next(iter(statuses)) if len(statuses) == 1 else "mixed",
            "devices": per_device,
            "priority_label": PRIORITY_LABELS[priority],
        }
    return {
        "schema_version": SCHEMA_VERSION, "status": status, "unit": "byte",
        "target_residual_bytes": residual_target_bytes, "source_priorities": dict(PRIORITY_LABELS),
        "source_applicability": applicability,
        "p2_facts": dict(vllm) if vllm else {},
        "tp": actual_tp, "dp": actual_dp, "enable_expert_parallel": actual_ep,
        "kv_cache_applicable": kv_cache_applicable, "devices": devices,
        "target_met": bool(devices) and all(item["residual"]["target_met"] for item in devices),
        "negative_residual_devices": [item["device_id"] for item in devices if item["residual"]["status"] == "negative"],
        "above_target_devices": [item["device_id"] for item in devices if item["residual"]["status"] == "above_target"],
        "missing_source_devices": [item["device_id"] for item in devices if item["missing_sources"]],
    }

__all__ = ["DEFAULT_RESIDUAL_TARGET_BYTES", "PRIORITIES", "SCHEMA_VERSION", "attribute_device",
           "attribute_memory", "compare_peaks", "estimate_weights_from_config", "parse_vllm_logs", "to_bytes"]
