"""Matched-run binding/evidence helpers and standalone cross-run acceptance CLI.

[Role]
Provide explicit run-binding checks, operator-sequence structural classification,
median comparison tables, and a two-run acceptance driver that consumes exported
framework-profiler roots plus C2 telemetry evidence.  The driver never launches
workloads; a supervisor supplies run roots and telemetry.

[Invariants]
- Binding comparisons require runtime, device, model, shape, dtype and clock
  identity; missing facts are never treated as equal because both are null.
- Operator-sequence classes come only from two explicit event-order
  ``structure.operator_sequence`` lists; ranked hotspots are never a substitute.
- The 5 Hz health check accepts only an explicit 0.2 s sample interval.
- Correctness and profiled/unprofiled benchmark separation are explicit inputs,
  never inferred from profile timing.
- Unknown or failed C2 telemetry keeps acceptance inconclusive/failed; a false
  throttle Boolean alone is not a positive no-throttle claim.

[Timeouts & Locks]
Pure in-memory helpers plus bounded, caller-supplied exported reads through the
existing analyzer.  No network, subprocess, lock, sleep, or NPU scheduling.

[Side-Effect Policy]
Reads only caller-named files; writes only the requested acceptance JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .compare_acceptance_contract import (
    _close,
    _valid_measurement,
    _valid_timing,
    _valid_window,
    _verify_provenance,
    validate_module_table,
)
from .events import ProfilerError
from .module_trace import build_module_table

BINDING_FIELDS = ("runtime", "device", "model", "shape", "dtype")
ACCEPTANCE_SCHEMA = "mindie-profile-matched-comparison/1"
STRUCTURE_BASIS = "structure.operator_sequence"


def _canon(value: Any) -> str | None:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError):
        return None


def _hash(value: Any) -> str | None:
    text = _canon(value)
    return hashlib.sha256(text.encode()).hexdigest() if text is not None else None


def _scene(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    value = summary.get("scene")
    return value if isinstance(value, Mapping) else {}


def _lim(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **{k: v for k, v in extra.items() if v is not None}}


def _binding_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in {"unknown", "none", "null"}
    if isinstance(value, (Mapping, list, tuple, set)):
        return bool(value)
    return True


def compare_run_binding(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    required: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare explicit runtime/device/model/shape/dtype facts."""
    b_scene, c_scene = _scene(baseline), _scene(candidate)
    entry: dict[str, Any] = {"required": required, "fields": {}, "status": "match"}
    hard: list[dict[str, Any]] = []
    limits: list[dict[str, Any]] = []
    for field in BINDING_FIELDS:
        b_value, c_value = b_scene.get(field), c_scene.get(field)
        field_entry = {
            "baseline_sha256": _hash(b_value) if b_value is not None else None,
            "candidate_sha256": _hash(c_value) if c_value is not None else None,
            "status": "match",
        }
        if not _binding_present(b_value) or not _binding_present(c_value):
            field_entry["status"] = "unavailable"
            message = f"scene.{field} is missing from at least one report"
            if required:
                hard.append(_lim("run_binding_field_unavailable", message, field=f"scene.{field}"))
            else:
                limits.append(_lim("run_binding_field_unavailable", message, field=f"scene.{field}"))
        elif field_entry["baseline_sha256"] is None or field_entry["candidate_sha256"] is None:
            hard.append(_lim("run_binding_field_unhashable", f"scene.{field} cannot be canonicalized",
                             field=f"scene.{field}"))
        elif field_entry["baseline_sha256"] != field_entry["candidate_sha256"]:
            field_entry["status"] = "mismatch"
            hard.append(_lim("run_binding_field_mismatch", f"scene.{field} differs",
                             field=f"scene.{field}"))
        entry["fields"][field] = field_entry
    if any(item["status"] != "match" for item in entry["fields"].values()):
        entry["status"] = "incomplete" if not hard else "not_comparable"
    return entry, hard, limits


def _structure_container(summary: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = summary.get("structure")
    return value if isinstance(value, Mapping) else None


def _explicit_sequence(summary: Mapping[str, Any]) -> list[str] | None:
    container = _structure_container(summary)
    raw = container.get("operator_sequence") if container else None
    if not isinstance(raw, list) or not raw:
        return None
    if not all(isinstance(item, str) and item for item in raw):
        return None
    return list(raw)


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (0 if left_item == right_item else 1),
            ))
        previous = current
    return previous[-1]


def classify_operator_sequence(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Classify operator-sequence change as identical/small/medium/large."""
    b_sequence, c_sequence = _explicit_sequence(baseline), _explicit_sequence(candidate)
    if b_sequence is None or c_sequence is None:
        return {
            "status": "not_available",
            "basis": None,
            "classification": "unknown",
            "reuse_recommendation": "not_available",
            "limitations": [_lim(
                "operator_sequence_unavailable",
                "both reports need an actual event-order structure.operator_sequence; "
                "ranked hotspots are not execution structure",
            )],
        }
    basis = STRUCTURE_BASIS
    longest = max(len(b_sequence), len(c_sequence))
    if b_sequence == c_sequence:
        distance = 0
        prefix, suffix = len(b_sequence), 0
        b_middle_len = c_middle_len = 0
    else:
        prefix = 0
        while prefix < min(len(b_sequence), len(c_sequence)) and b_sequence[prefix] == c_sequence[prefix]:
            prefix += 1
        suffix = 0
        while (
            suffix < min(len(b_sequence), len(c_sequence)) - prefix
            and b_sequence[-1 - suffix] == c_sequence[-1 - suffix]
        ):
            suffix += 1
        b_middle = b_sequence[prefix:len(b_sequence) - suffix]
        c_middle = c_sequence[prefix:len(c_sequence) - suffix]
        b_middle_len, c_middle_len = len(b_middle), len(c_middle)
        distance = _edit_distance(b_middle, c_middle)
    similarity = 1.0 - distance / longest if longest else 1.0
    if b_sequence == c_sequence:
        classification = "identical"
        recommendation = "full_structure_reuse"
    elif distance <= max(1, int(0.10 * longest)):
        classification = "small"
        recommendation = "layer_identity_reuse_with_event_shape"
    elif distance <= max(2, int(0.35 * longest)):
        classification = "medium"
        recommendation = "block_level_only"
    else:
        classification = "large"
        recommendation = "none"
    return {
        "status": "classified",
        "basis": basis,
        "classification": classification,
        "reuse_recommendation": recommendation,
        "baseline_count": len(b_sequence),
        "candidate_count": len(c_sequence),
        "edit_distance": distance,
        "similarity": round(similarity, 6),
        "common_prefix": prefix,
        "common_suffix": suffix,
        "middle_length_baseline": b_middle_len,
        "middle_length_candidate": c_middle_len,
        "baseline_sha256": _hash(b_sequence),
        "candidate_sha256": _hash(c_sequence),
        "limitations": [],
    }


def _metric_row(name: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "unit": entry.get("unit"),
        "status": entry.get("status"),
        "baseline_p50": entry.get("baseline_p50"),
        "candidate_p50": entry.get("candidate_p50"),
        "delta": entry.get("delta"),
        "delta_pp": entry.get("delta_pp"),
        "relative_delta_pct": entry.get("relative_delta_pct"),
        "confidence_interval": entry.get("confidence_interval"),
    }


def build_median_tables(
    metrics: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
    hierarchy: Mapping[str, Any],
) -> dict[str, Any]:
    """Build Layer / seven-semantic-block / bubble median tables."""
    semantic = [
        {
            "block": entry.get("block"),
            "status": entry.get("status"),
            "baseline_busy_union_ms_p50": entry.get("baseline_busy_union_ms_p50"),
            "candidate_busy_union_ms_p50": entry.get("candidate_busy_union_ms_p50"),
            "delta_ms": entry.get("delta_ms"),
            "relative_delta_pct": entry.get("relative_delta_pct"),
            "baseline_share_of_busy_pct": entry.get("baseline_share_of_busy_pct"),
            "candidate_share_of_busy_pct": entry.get("candidate_share_of_busy_pct"),
            "share_delta_pp": entry.get("share_delta_pp"),
        }
        for entry in blocks
    ]
    bubble_names = (
        "host_gap_ms_p50", "underfeed_bubble_ratio",
        "aicpu_exposed_ratio", "exposed_comm_ms_p50",
    )
    bubble = [
        _metric_row(name, metrics[name])
        for name in bubble_names if name in metrics
    ]
    layers = [
        {
            "layer": layer.get("layer"),
            "layer_id": layer.get("layer_id"),
            "ordinal": layer.get("ordinal"),
            "status": layer.get("status"),
            "match_basis": layer.get("match_basis"),
            "metrics": {
                name: _metric_row(name, entry)
                for name, entry in (layer.get("metrics") or {}).items()
            },
        }
        for layer in hierarchy.get("layers", [])
    ]
    return {"layers": layers, "semantic_blocks": semantic, "bubble": bubble}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = _load_json(manifest_path)
    manifest["_base_dir"] = str(manifest_path.parent)
    return manifest


def _resolve(base: str, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(base) / path


def _telemetry_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = manifest.get("telemetry")
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        return _load_json(_resolve(str(manifest.get("_base_dir") or "."), value))
    raise ValueError("run manifest needs telemetry object or telemetry file path")


def _telemetry_raw(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    value = manifest.get("telemetry_raw")
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        return _load_json(_resolve(str(manifest.get("_base_dir") or "."), value))
    raise ValueError("run manifest telemetry_raw must be an object or file path")


def _scene_device_card(scene: Mapping[str, Any]) -> int | None:
    device = scene.get("device")
    if isinstance(device, int) and not isinstance(device, bool):
        return device
    if isinstance(device, Mapping):
        card = device.get("card")
        return card if isinstance(card, int) and not isinstance(card, bool) else None
    return None


def _telemetry_cards(payload: Mapping[str, Any]) -> list[int]:
    cards: set[int] = set()
    raw = payload.get("devices")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, int) and not isinstance(item, bool):
                cards.add(item)
            elif isinstance(item, Mapping):
                card = item.get("npu_id", item.get("device_id"))
                if isinstance(card, int) and not isinstance(card, bool):
                    cards.add(card)
    window = payload.get("window")
    if isinstance(window, Mapping):
        requested = window.get("requested_devices")
        if isinstance(requested, list):
            cards.update(item for item in requested if isinstance(item, int) and not isinstance(item, bool))
    return sorted(cards)


def _payload_host(payload: Mapping[str, Any]) -> str | None:
    host = payload.get("host")
    if isinstance(host, str) and host:
        return host
    server = payload.get("server")
    if isinstance(server, Mapping) and isinstance(server.get("host"), str):
        return server["host"]
    return None


def _window_bounds(payload: Mapping[str, Any]) -> tuple[float, float] | None:
    window = payload.get("window")
    if isinstance(window, Mapping):
        start, end = window.get("start_ts"), window.get("end_ts")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
            return float(start), float(end)
    start, end = payload.get("start_ts"), payload.get("end_ts")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
        return float(start), float(end)
    return None


def _sample_interval(manifest: Mapping[str, Any], raw: Mapping[str, Any] | None) -> float | None:
    if raw is not None:
        metadata = raw.get("collector_metadata")
        if isinstance(metadata, Mapping) and isinstance(metadata.get("interval_s"), (int, float)):
            return float(metadata["interval_s"])
    value = manifest.get("sample_interval_s")
    return float(value) if isinstance(value, (int, float)) else None


def _capture_matches(manifest: Mapping[str, Any], review: Mapping[str, Any],
                     profiled: Mapping[str, Any]) -> tuple[bool, list[str]]:
    capture = manifest.get("capture")
    reasons: list[str] = []
    if not isinstance(capture, Mapping):
        return False, ["capture_receipt_missing"]
    if capture.get("status") != "ok":
        reasons.append("capture_status_not_ok")
    scene = manifest.get("scene") if isinstance(manifest.get("scene"), Mapping) else {}
    expected_card = manifest.get("physical_card")
    if not isinstance(expected_card, int) or isinstance(expected_card, bool):
        expected_card = _scene_device_card(scene)
    if not isinstance(capture.get("device"), int) or capture.get("device") != expected_card:
        reasons.append("capture_device_does_not_match_card")
    plan = capture.get("plan")
    capture_plan = plan.get("capture") if isinstance(plan, Mapping) else None
    captured = capture_plan.get("captured_step_ids") if isinstance(capture_plan, Mapping) else None
    if not isinstance(captured, list) or not all(isinstance(item, int) for item in captured):
        reasons.append("capture_step_ids_missing")
    else:
        if any(step not in captured for step in review.get("_step_ids", [])):
            reasons.append("capture_steps_do_not_cover_callers")
    active = capture.get("steps", {}).get("active") if isinstance(capture.get("steps"), Mapping) else None
    caller = plan.get("caller_window") if isinstance(plan, Mapping) else None
    if not _valid_timing(active):
        reasons.append("capture_active_timing_missing")
    else:
        if isinstance(caller, Mapping) and all(
            isinstance(caller.get(key), int) for key in ("start_step", "end_step")
        ) and caller["end_step"] - caller["start_step"] != active.get("count"):
            reasons.append("capture_plan_count_disagrees_with_active")
        if active.get("count") != len(review.get("_step_ids", [])):
            reasons.append("capture_active_count_disagrees_with_callers")
        if not all(_close(active.get(key), profiled.get("timing", {}).get(key))
                   for key in ("count", "total_ms", "min_ms", "max_ms", "mean_ms")):
            reasons.append("profiled_timing_disagrees_with_capture")
    phases = capture.get("phases")
    total = phases.get("total") if isinstance(phases, Mapping) else None
    if not _close(total, profiled.get("duration_ms")):
        reasons.append("profiled_duration_disagrees_with_capture")
    return not reasons, reasons


def _benchmark_checks(manifest: Mapping[str, Any], review: Mapping[str, Any]) -> tuple[bool, list[str]]:
    benchmark = manifest.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return False, ["benchmark_missing"]
    profiled, unprofiled = benchmark.get("profiled"), benchmark.get("unprofiled")
    reasons: list[str] = []
    if not _valid_measurement(profiled):
        reasons.append("profiled_measurement_missing")
    if not _valid_measurement(unprofiled):
        reasons.append("unprofiled_measurement_missing")
    if reasons:
        return False, reasons
    task_window = _valid_window(manifest.get("profile_measurement_window"))
    capture_window = _valid_window(manifest.get("capture_wall_window"))
    profiled_window = _valid_window(profiled.get("measurement_window"))
    unprofiled_window = _valid_window(unprofiled.get("measurement_window"))
    if profiled_window is None or capture_window is None or profiled_window != capture_window:
        reasons.append("profiled_window_does_not_match_capture_wall")
    if (
        task_window is None or capture_window is None
        or capture_window[0] > task_window[0]
        or capture_window[1] < task_window[1]
    ):
        reasons.append("capture_wall_does_not_contain_task_interval")
    if profiled_window is None or unprofiled_window is None or (
        profiled_window[0] < unprofiled_window[1] and unprofiled_window[0] < profiled_window[1]
    ):
        reasons.append("profiled_and_unprofiled_windows_overlap")
    capture_ok, capture_reasons = _capture_matches(manifest, review, profiled)
    reasons.extend(capture_reasons)
    return not reasons, reasons

def _check_acceptance(
    baseline_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    baseline_review: Mapping[str, Any],
    candidate_review: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    limitations: list[dict[str, Any]] = []
    b_scene = baseline_manifest.get("scene") if isinstance(baseline_manifest.get("scene"), Mapping) else {}
    c_scene = candidate_manifest.get("scene") if isinstance(candidate_manifest.get("scene"), Mapping) else {}
    b_card, c_card = baseline_manifest.get("physical_card"), candidate_manifest.get("physical_card")
    b_scene_card, c_scene_card = _scene_device_card(b_scene), _scene_device_card(c_scene)
    b_cards = _telemetry_cards(baseline_review.get("_payload", {}))
    c_cards = _telemetry_cards(candidate_review.get("_payload", {}))
    b_raw_cards, c_raw_cards = baseline_review.get("_raw_cards"), candidate_review.get("_raw_cards")
    cards_match = (
        isinstance(b_card, int) and isinstance(c_card, int) and b_card == c_card
        and b_scene_card == b_card and c_scene_card == c_card
        and b_cards == [b_card] and c_cards == [c_card]
        and b_raw_cards == [b_card] and c_raw_cards == [c_card]
    )
    checks["same_physical_card"] = cards_match
    if not cards_match:
        limitations.append(_lim("physical_card_mismatch",
                                "both runs, scene and raw telemetry must select the same single card",
                                details={"baseline": {"manifest": b_card, "scene": b_scene_card, "telemetry": b_cards, "raw": b_raw_cards},
                                         "candidate": {"manifest": c_card, "scene": c_scene_card, "telemetry": c_cards, "raw": c_raw_cards}}))

    b_host, c_host = baseline_manifest.get("host"), candidate_manifest.get("host")
    b_payload_host = _payload_host(baseline_review.get("_payload", {}))
    c_payload_host = _payload_host(candidate_review.get("_payload", {}))
    host_ok = (
        isinstance(b_host, str) and isinstance(c_host, str) and b_host and b_host == c_host
        and b_payload_host == b_host and c_payload_host == c_host
    )
    checks["same_host"] = host_ok
    if not host_ok:
        limitations.append(_lim("host_mismatch",
                                "both runs must bind the same non-empty host in manifest and telemetry",
                                details={"baseline": b_host, "candidate": c_host,
                                         "baseline_telemetry": b_payload_host,
                                         "candidate_telemetry": c_payload_host}))

    b_id, c_id = baseline_manifest.get("run_id"), candidate_manifest.get("run_id")
    run_ids_ok = (
        isinstance(b_id, str) and isinstance(c_id, str)
        and bool(b_id.strip()) and bool(c_id.strip()) and b_id != c_id
    )
    checks["distinct_run_ids"] = run_ids_ok
    if not run_ids_ok:
        limitations.append(_lim("run_ids_not_distinct",
                                "baseline and candidate need distinct non-empty run_id values",
                                details={"baseline": b_id, "candidate": c_id}))

    b_interval, c_interval = baseline_review.get("_sample_interval"), candidate_review.get("_sample_interval")
    interval_ok = all(
        isinstance(value, (int, float)) and abs(float(value) - 0.2) <= 1e-9
        for value in (b_interval, c_interval)
    ) and baseline_review.get("_raw_present") is True and candidate_review.get("_raw_present") is True
    checks["sample_interval_5hz"] = interval_ok
    if not interval_ok:
        limitations.append(_lim("sample_interval_not_5hz",
                                "health telemetry must carry raw evidence with an explicit 0.2 s (5 Hz) interval",
                                details={"baseline": b_interval, "candidate": c_interval,
                                         "baseline_raw": baseline_review.get("_raw_present"),
                                         "candidate_raw": candidate_review.get("_raw_present")}))

    coverage_ok = all(
        review.get("coverage", {}).get("complete") is True
        and isinstance(review.get("coverage", {}).get("ratio"), (int, float))
        and float(review["coverage"]["ratio"]) >= 1.0 - 1e-12
        for review in (baseline_review, candidate_review)
    )
    checks["telemetry_coverage_complete"] = coverage_ok
    if not coverage_ok:
        limitations.append(_lim("telemetry_coverage_incomplete",
                                "C2 coverage must include all scheduled sample points for both runs",
                                details={"baseline": baseline_review.get("coverage"),
                                         "candidate": candidate_review.get("coverage")}))

    b_clock, c_clock = b_scene.get("clock_domain"), c_scene.get("clock_domain")
    b_tclock, c_tclock = baseline_manifest.get("telemetry_clock_domain"), candidate_manifest.get("telemetry_clock_domain")
    payload_clocks = []
    for review in (baseline_review, candidate_review):
        payload = review.get("_payload", {})
        window = payload.get("window") if isinstance(payload.get("window"), Mapping) else {}
        domains = window.get("clock_domains")
        payload_clocks.append(sorted(str(item) for item in domains) if isinstance(domains, list) else None)
    clock_ok = (
        all(isinstance(item, str) and item and item.lower() != "unknown"
            for item in (b_clock, c_clock, b_tclock, c_tclock))
        and b_clock == c_clock == b_tclock == c_tclock
        and all(item is None or item == [b_clock] for item in payload_clocks)
    )
    checks["telemetry_clock_comparable"] = clock_ok
    if not clock_ok:
        limitations.append(_lim("telemetry_clock_not_comparable",
                                "scene and declared C2 telemetry clock domains must match and be known",
                                details={"baseline_scene": b_clock, "candidate_scene": c_clock,
                                         "baseline_telemetry": b_tclock, "candidate_telemetry": c_tclock,
                                         "payload": payload_clocks}))
    if any(item is None for item in payload_clocks):
        limitations.append(_lim("c2_clock_domain_not_embedded",
                                "C2 payload does not embed window.clock_domains; manifest declaration is used"))

    b_task = _valid_window(baseline_manifest.get("profile_measurement_window"))
    c_task = _valid_window(candidate_manifest.get("profile_measurement_window"))
    b_capture = _valid_window(baseline_manifest.get("capture_wall_window"))
    c_capture = _valid_window(candidate_manifest.get("capture_wall_window"))
    b_telemetry = _window_bounds(baseline_review.get("_payload", {}))
    c_telemetry = _window_bounds(candidate_review.get("_payload", {}))
    capture_contains_ok = all(
        task is not None and capture is not None
        and capture[0] <= task[0] and capture[1] >= task[1]
        for task, capture in ((b_task, b_capture), (c_task, c_capture))
    )
    checks["capture_wall_contains_profile"] = capture_contains_ok
    if not capture_contains_ok:
        limitations.append(_lim("capture_wall_does_not_contain_profile",
                                "capture wall must contain the caller-explicit task profile interval",
                                details={"baseline": {"profile": b_task, "capture": b_capture},
                                         "candidate": {"profile": c_task, "capture": c_capture}}))
    contains_ok = all(
        capture is not None and telemetry is not None
        and telemetry[0] <= capture[0] and telemetry[1] >= capture[1]
        for capture, telemetry in ((b_capture, b_telemetry), (c_capture, c_telemetry))
    )
    checks["telemetry_window_contains_capture"] = contains_ok
    if not contains_ok:
        limitations.append(_lim("telemetry_window_does_not_contain_capture",
                                "C2 window must contain the full actual capture wall interval",
                                details={"baseline": {"capture": b_capture, "telemetry": b_telemetry},
                                         "candidate": {"capture": c_capture, "telemetry": c_telemetry}}))

    b_correct, c_correct = baseline_manifest.get("correctness"), candidate_manifest.get("correctness")
    b_status = b_correct.get("status") if isinstance(b_correct, Mapping) else None
    c_status = c_correct.get("status") if isinstance(c_correct, Mapping) else None
    correctness_ok = b_status == "passed" and c_status == "passed"
    checks["correctness_preserved"] = correctness_ok
    if not correctness_ok:
        limitations.append(_lim("correctness_not_preserved",
                                "both runs must provide explicit correctness.status=passed",
                                details={"baseline": b_status, "candidate": c_status}))

    b_provenance_ok, b_provenance_reasons = _verify_provenance(baseline_manifest)
    c_provenance_ok, c_provenance_reasons = _verify_provenance(candidate_manifest)
    provenance_ok = b_provenance_ok and c_provenance_ok
    checks["provenance_integrity"] = provenance_ok
    if not provenance_ok:
        limitations.append(_lim("provenance_integrity_failed",
                                "manifest source_files must match the immutable raw files by SHA256",
                                details={"baseline": b_provenance_reasons,
                                         "candidate": c_provenance_reasons}))

    b_separated, b_reasons = _benchmark_checks(baseline_manifest, baseline_review)
    c_separated, c_reasons = _benchmark_checks(candidate_manifest, candidate_review)
    separation_ok = b_separated and c_separated
    checks["profile_benchmark_separated"] = separation_ok
    if not separation_ok:
        limitations.append(_lim("profile_benchmark_not_separated",
                                "profiled and unprofiled sections need explicit windows, durations, timing summaries and capture-consistent measures",
                                details={"baseline": b_reasons, "candidate": c_reasons}))

    module_table = comparison.get("tables", {}).get("modules")
    if isinstance(module_table, Mapping) and module_table.get("modules"):
        module_ok, module_reasons = validate_module_table(module_table)
        checks["module_trace_complete"] = module_ok
        if not module_ok:
            limitations.append(_lim("module_trace_incomplete",
                                    "actual module trace/table needs full invocation mapping on both sides",
                                    details={"reasons": module_reasons}))
    binding_ok = comparison.get("matching", {}).get("run_binding", {}).get("status") == "match"
    checks["run_binding_complete"] = binding_ok
    if not binding_ok:
        limitations.append(_lim("run_binding_incomplete",
                                "runtime/device/model/shape/dtype binding is not complete"))
    allowed_comparison_limitations = {"structure_reuse_not_established"}
    step_accounting = comparison.get("step_accounting") or {}
    semantic_blocks = comparison.get("semantic_blocks") or {}
    step_ok = step_accounting.get("status") == "comparable"
    blocks_ok = semantic_blocks.get("status") == "comparable"
    disallowed = [
        item for item in comparison.get("limitations", [])
        if item.get("code") not in allowed_comparison_limitations
    ]
    comparison_complete = step_ok and blocks_ok and not disallowed
    checks["comparison_complete"] = comparison_complete
    if not comparison_complete:
        limitations.append(_lim("comparison_not_complete",
                                "step/block comparison is incomplete or carries a non-scope limitation",
                                details={"status": comparison.get("status"),
                                         "step_status": step_accounting.get("status"),
                                         "block_status": semantic_blocks.get("status"),
                                         "disallowed": [item.get("code") for item in disallowed]}))
    no_throttle = (
        baseline_review.get("status") == "no_throttling_observed"
        and candidate_review.get("status") == "no_throttling_observed"
    )
    checks["telemetry_no_throttling_observed"] = no_throttle
    if not no_throttle:
        limitations.append(_lim("telemetry_throttling_not_cleared",
                                "both telemetry windows must support a no-throttling observation"))
    if not comparison.get("comparable") or any(
        not value for key, value in checks.items() if key != "telemetry_no_throttling_observed"
    ):
        status = "not_passed"
    elif not no_throttle:
        status = "inconclusive"
    else:
        status = "passed"
    return {"status": status, "checks": checks, "limitations": limitations}

def _prepare_run(manifest: Mapping[str, Any], *, rank: Sequence[str] | None, unit: str, top_n: int) -> dict[str, Any]:
    from .analyze import analyze_profile
    from .segment import load_steps_file
    from .thermal_guard import review_telemetry_window

    base_dir = str(manifest.get("_base_dir") or ".")
    root = _resolve(base_dir, manifest["root"])
    steps_file = _resolve(base_dir, manifest["steps_file"])
    scene = dict(manifest.get("scene") or {})
    windows = load_steps_file(steps_file, unit=unit)
    report = analyze_profile(
        root,
        ranks=list(rank) if rank is not None else manifest.get("ranks"),
        windows=windows,
        scene=scene,
        top_n=top_n,
    )
    structure = manifest.get("structure")
    structure_file = manifest.get("structure_file")
    if structure_file is not None:
        loaded = _load_json(_resolve(base_dir, structure_file))
        structure = loaded.get("structure") if isinstance(loaded.get("structure"), Mapping) else loaded
    if isinstance(structure, Mapping):
        report["structure"] = dict(structure)
    payload = _telemetry_payload(manifest)
    raw = _telemetry_raw(manifest)
    card = manifest.get("physical_card")
    if not isinstance(card, int) or isinstance(card, bool):
        card = _scene_device_card(scene)
    review = review_telemetry_window(
        payload,
        expected_host=manifest.get("host"),
        expected_devices=[card] if isinstance(card, int) else None,
    )
    review["_payload"] = payload
    review["_sample_interval"] = _sample_interval(manifest, raw)
    review["_raw_present"] = raw is not None
    review["_raw_cards"] = (
        raw.get("collector_metadata", {}).get("cards")
        if isinstance(raw, Mapping) and isinstance(raw.get("collector_metadata"), Mapping)
        else None
    )
    review["_step_ids"] = [window.step for window in windows if window.step is not None]
    return {"report": report, "review": review, "scene": scene, "payload": payload,
            "raw_sha256": _hash(raw) if raw is not None else None}

def run_acceptance(
    baseline_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    *,
    rank: Sequence[str] | None = None,
    unit: str = "us",
    top_n: int = 8,
) -> dict[str, Any]:
    """Run the matched two-run acceptance without launching any workload."""
    from .compare import compare_profiles

    baseline = _prepare_run(baseline_manifest, rank=rank, unit=unit, top_n=top_n)
    candidate = _prepare_run(candidate_manifest, rank=rank, unit=unit, top_n=top_n)
    comparison = compare_profiles(
        baseline["report"], candidate["report"], require_binding=True
    )
    baseline_structure = baseline["report"].get("structure")
    candidate_structure = candidate["report"].get("structure")
    if isinstance(baseline_structure, Mapping) and isinstance(candidate_structure, Mapping):
        module_table = build_module_table(baseline_structure, candidate_structure)
        if module_table["modules"]:
            comparison.setdefault("tables", {})["modules"] = module_table
            comparison["module_scope"] = module_table["scope"]
    acceptance = _check_acceptance(
        baseline_manifest, candidate_manifest,
        baseline["review"], candidate["review"], comparison,
    )
    return {
        "schema_version": ACCEPTANCE_SCHEMA,
        "status": acceptance["status"],
        "baseline_run_id": baseline_manifest.get("run_id"),
        "candidate_run_id": candidate_manifest.get("run_id"),
        "inputs": {
            "baseline_root": str(baseline_manifest.get("root")),
            "candidate_root": str(candidate_manifest.get("root")),
            "baseline_steps_file": str(baseline_manifest.get("steps_file")),
            "candidate_steps_file": str(candidate_manifest.get("steps_file")),
            "baseline_telemetry_sha256": _hash(baseline["payload"]),
            "candidate_telemetry_sha256": _hash(candidate["payload"]),
            "baseline_raw_telemetry_sha256": baseline["raw_sha256"],
            "candidate_raw_telemetry_sha256": candidate["raw_sha256"],
        },
        "telemetry": {
            "baseline": {key: value for key, value in baseline["review"].items() if not key.startswith("_")},
            "candidate": {key: value for key, value in candidate["review"].items() if not key.startswith("_")},
        },
        "comparison": comparison,
        "acceptance": acceptance,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mindie_profiler.compare_acceptance",
        description="Matched two-run C7/C2 acceptance; consumes exported roots and telemetry only.",
        allow_abbrev=False,
    )
    parser.add_argument("--baseline-run", required=True, help="baseline run manifest JSON")
    parser.add_argument("--candidate-run", required=True, help="candidate run manifest JSON")
    parser.add_argument("--rank", action="append", default=None, help="rank filter for both roots")
    parser.add_argument("--unit", choices=("us", "ns", "ms"), default="us")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--output", help="atomically write acceptance JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_acceptance(
            _load_manifest(args.baseline_run),
            _load_manifest(args.candidate_run),
            rank=args.rank,
            unit=args.unit,
            top_n=args.top,
        )
    except (OSError, ValueError, KeyError, ProfilerError) as exc:
        print(f"mindie-profiler acceptance: {exc}", file=sys.stderr)
        return 2
    if args.output:
        from .report import write_report
        write_report(result, args.output)
        print(json.dumps({"output": args.output, "status": result["status"]}))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
