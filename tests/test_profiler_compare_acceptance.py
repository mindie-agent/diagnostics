"""Matched cross-run acceptance driver tests using real analyzer/C4 evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from unittest.mock import patch

from mindie_profiler.compare_acceptance import (
    ACCEPTANCE_SCHEMA,
    classify_operator_sequence,
    main,
    run_acceptance,
)
from mindie_profiler.framework_capture import build_capture_plan
from tests.test_profiler_core import RANK0_ID, build_run

C4_ROOT = Path(__file__).resolve().parent / "fixtures" / "profiler_core" / "c2_a2"
SOURCE_TELEMETRY = C4_ROOT / "source.telemetry.json"
SOURCE_RAW = C4_ROOT / "source.raw-window.json"

SCENE = {
    "framework": "vllm-ascend",
    "phase": "decode",
    "tp": 1,
    "dp": 1,
    "graph_mode": "eager",
    "clock_domain": "aligned_syscnt",
    "input_signature": "same-inputs",
    "runtime": {
        "soc": "Ascend910B4",
        "framework": {"torch_npu": "2.10.0.post4"},
        "cann": "9.1.0",
    },
    "device": {"card": 0, "host": "173.125.1.2"},
    "model": "tiny-llm",
    "shape": [8, 1024],
    "dtype": "bf16",
}

STRUCTURE = {
    "event_sequence_shape": ["attention", "moe", "norm"],
    "operator_sequence": ["Attention:flash", "MoE:grouped_matmul", "Norm:rmsnorm"],
    "layers": [
        {
            "layer_id": "layer_0",
            "ordinal": 0,
            "name": "block_0",
            "busy_union_ms_p50": 0.010,
            "wall_ms_p50": 0.020,
            "host_gap_ms_p50": 0.001,
        }
    ],
}


def _steps_file(tmp_path):
    path = tmp_path / "steps.json"
    path.write_text(json.dumps({"steps": [
        {"step": 4, "start_us": 0, "end_us": 60},
        {"step": 5, "start_us": 100, "end_us": 160},
    ]}))
    return path


def _manifest(tmp_path, payload, *, run_id="run-1"):
    root = build_run(tmp_path / "run")
    steps_path = _steps_file(tmp_path)
    db_path = next((root / RANK0_ID / "ASCEND_PROFILER_OUTPUT").glob("ascend_pytorch_profiler*.db"))
    telemetry_window = payload["window"]
    start, end = float(telemetry_window["start_ts"]), float(telemetry_window["end_ts"])
    capture_wall = {"start_ts": start, "end_ts": end}
    profile_window = {"start_ts": start + 0.05, "end_ts": end - 0.05}
    active = {
        "count": 2,
        "total_ms": 40.0,
        "min_ms": 18.0,
        "max_ms": 22.0,
        "mean_ms": 20.0,
    }
    return {
        "run_id": run_id,
        "root": str(root),
        "steps_file": str(steps_path),
        "scene": copy.deepcopy(SCENE),
        "structure": copy.deepcopy(STRUCTURE),
        "telemetry": str(SOURCE_TELEMETRY),
        "telemetry_raw": str(SOURCE_RAW),
        "profile_measurement_window": profile_window,
        "capture_wall_window": capture_wall,
        "telemetry_clock_domain": SCENE["clock_domain"],
        "physical_card": 0,
        "host": payload["host"],
        "correctness": {"status": "passed"},
        "capture": {
            "status": "ok",
            "device": 0,
            "plan": build_capture_plan(4, 6),
            "steps": {"active": active},
            "phases": {"total": 100.0},
        },
        "provenance": {
            "schema_version": "mindie-live-derived-manifest/1",
            "source_files": {
                name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name, path in {
                    "steps": steps_path,
                    "db": db_path,
                    "telemetry": SOURCE_TELEMETRY,
                    "telemetry_raw": SOURCE_RAW,
                }.items()
            },
        },
        "benchmark": {
            "profiled": {
                "measurement_window": dict(capture_wall),
                "duration_ms": 100.0,
                "sample_count": 2,
                "timing": dict(active),
            },
            "unprofiled": {
                "measurement_window": {"start_ts": end + 1.0, "end_ts": end + 2.0},
                "duration_ms": 300.0,
                "sample_count": 3,
                "timing": {"count": 3, "total_ms": 300.0, "min_ms": 90.0, "max_ms": 110.0, "mean_ms": 100.0},
            },
        },
    }


def test_matched_acceptance_passes_with_real_c2_and_exported_profile(tmp_path):
    payload = json.loads(SOURCE_TELEMETRY.read_text(encoding="utf-8"))
    baseline = _manifest(tmp_path / "baseline", payload, run_id="run-1")
    candidate = _manifest(tmp_path / "candidate", payload, run_id="run-2")

    result = run_acceptance(baseline, candidate)

    assert result["schema_version"] == ACCEPTANCE_SCHEMA
    assert result["status"] == "passed"
    assert all(result["acceptance"]["checks"].values())
    assert {item["code"] for item in result["acceptance"]["limitations"]} == {
        "c2_clock_domain_not_embedded"
    }
    assert result["comparison"]["comparable"] is True
    assert result["comparison"]["matching"]["run_binding"]["status"] == "match"
    assert result["comparison"]["operator_sequence"]["classification"] == "identical"
    assert result["comparison"]["tables"]["layers"][0]["layer_id"] == "layer_0"
    assert len(result["comparison"]["tables"]["semantic_blocks"]) == 7
    assert result["telemetry"]["baseline"]["status"] == "no_throttling_observed"
    assert result["telemetry"]["baseline"]["temperature"]["used_as_throttling_proof"] is False


def test_matched_acceptance_refuses_missing_binding(tmp_path):
    payload = json.loads(SOURCE_TELEMETRY.read_text(encoding="utf-8"))
    baseline = _manifest(tmp_path / "baseline", payload, run_id="run-1")
    candidate = copy.deepcopy(baseline)
    candidate["run_id"] = "run-2"
    candidate["scene"].pop("runtime")
    candidate["scene"].pop("dtype")

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "not_passed"
    assert result["acceptance"]["checks"]["run_binding_complete"] is False
    assert result["comparison"]["status"] == "not_comparable"
    assert any(
        item["code"] == "run_binding_field_unavailable"
        for item in result["comparison"]["reasons"]
    )


def test_matched_acceptance_refuses_non_5hz_health(tmp_path):
    payload = json.loads(SOURCE_TELEMETRY.read_text(encoding="utf-8"))
    baseline = _manifest(tmp_path / "baseline", payload, run_id="run-1")
    candidate = copy.deepcopy(baseline)
    candidate["run_id"] = "run-2"
    candidate["sample_interval_s"] = 0.1
    candidate.pop("telemetry_raw")

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "not_passed"
    assert result["acceptance"]["checks"]["sample_interval_5hz"] is False
    assert any(item["code"] == "sample_interval_not_5hz" for item in result["acceptance"]["limitations"])


def test_matched_acceptance_keeps_unknown_throttle_inconclusive(tmp_path):
    payload = json.loads(SOURCE_TELEMETRY.read_text(encoding="utf-8"))
    unknown = copy.deepcopy(payload)
    unknown["throttling_detected"] = None
    unknown["devices"] = [0]
    baseline = _manifest(tmp_path / "baseline", payload, run_id="run-1")
    candidate = _manifest(tmp_path / "candidate", payload, run_id="run-2")
    candidate["telemetry"] = unknown

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "inconclusive"
    assert result["acceptance"]["checks"]["telemetry_no_throttling_observed"] is False
    assert any(item["code"] == "telemetry_throttling_not_cleared" for item in result["acceptance"]["limitations"])


def test_acceptance_cli_writes_result_and_returns_passed(tmp_path):
    payload = json.loads(SOURCE_TELEMETRY.read_text(encoding="utf-8"))
    baseline = _manifest(tmp_path, payload, run_id="run-1")
    candidate = copy.deepcopy(baseline)
    candidate["run_id"] = "run-2"
    baseline_path = tmp_path / "baseline-run.json"
    candidate_path = tmp_path / "candidate-run.json"
    baseline_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))
    output = tmp_path / "acceptance.json"

    code = main([
        "--baseline-run", str(baseline_path),
        "--candidate-run", str(candidate_path),
        "--output", str(output),
    ])

    assert code == 0
    result = json.loads(output.read_text())
    assert result["status"] == "passed"
    assert result["comparison"]["operator_sequence"]["classification"] == "identical"


def test_matched_acceptance_requires_same_scene_and_telemetry_card(tmp_path):
    payload = json.loads(SOURCE_TELEMETRY.read_text(encoding="utf-8"))
    baseline = _manifest(tmp_path / "baseline", payload, run_id="run-1")
    candidate = copy.deepcopy(baseline)
    candidate["run_id"] = "run-2"
    candidate["physical_card"] = 1

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "not_passed"
    assert result["acceptance"]["checks"]["same_physical_card"] is False
    assert any(item["code"] == "physical_card_mismatch" for item in result["acceptance"]["limitations"])


def test_matched_acceptance_requires_raw_health_evidence(tmp_path):
    payload = json.loads(SOURCE_TELEMETRY.read_text(encoding="utf-8"))
    baseline = _manifest(tmp_path / "baseline", payload, run_id="run-1")
    candidate = copy.deepcopy(baseline)
    candidate["run_id"] = "run-2"
    candidate.pop("telemetry_raw")
    candidate["sample_interval_s"] = 0.2

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "not_passed"
    assert result["acceptance"]["checks"]["sample_interval_5hz"] is False
    limitation = next(
        item for item in result["acceptance"]["limitations"]
        if item["code"] == "sample_interval_not_5hz"
    )
    assert limitation["details"]["candidate_raw"] is False


def test_matched_acceptance_rejects_profile_window_outside_c2_window(tmp_path):
    payload = json.loads(SOURCE_TELEMETRY.read_text(encoding="utf-8"))
    baseline = _manifest(tmp_path / "baseline", payload, run_id="run-1")
    candidate = copy.deepcopy(baseline)
    candidate["run_id"] = "run-2"
    candidate["profile_measurement_window"]["start_ts"] = payload["window"]["start_ts"] - 0.1

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "not_passed"
    assert result["acceptance"]["checks"]["capture_wall_contains_profile"] is False
    assert result["acceptance"]["checks"]["telemetry_window_contains_capture"] is True
    assert any(
        item["code"] == "capture_wall_does_not_contain_profile"
        for item in result["acceptance"]["limitations"]
    )


def test_matched_acceptance_rejects_host_clock_and_run_id_mismatch(tmp_path):
    payload = json.loads(SOURCE_TELEMETRY.read_text(encoding="utf-8"))
    baseline = _manifest(tmp_path / "baseline", payload, run_id="run-1")

    same_id = copy.deepcopy(baseline)
    same_id["host"] = "other-host"
    same_id["telemetry_clock_domain"] = "other-clock"
    result = run_acceptance(baseline, same_id)

    assert result["status"] == "not_passed"
    assert result["acceptance"]["checks"]["distinct_run_ids"] is False
    assert result["acceptance"]["checks"]["same_host"] is False
    assert result["acceptance"]["checks"]["telemetry_clock_comparable"] is False


def test_matched_acceptance_rejects_missing_benchmark_measure(tmp_path):
    payload = json.loads(SOURCE_TELEMETRY.read_text(encoding="utf-8"))
    baseline = _manifest(tmp_path / "baseline", payload, run_id="run-1")
    candidate = copy.deepcopy(baseline)
    candidate["run_id"] = "run-2"
    candidate["benchmark"]["profiled"]["timing"]["mean_ms"] = None

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "not_passed"
    assert result["acceptance"]["checks"]["profile_benchmark_separated"] is False
    assert any(
        "profiled_measurement_missing" in item.get("details", {}).get("candidate", [])
        for item in result["acceptance"]["limitations"]
        if item["code"] == "profile_benchmark_not_separated"
    )


def test_matched_acceptance_rejects_capture_not_covering_callers(tmp_path):
    payload = json.loads(SOURCE_TELEMETRY.read_text(encoding="utf-8"))
    baseline = _manifest(tmp_path / "baseline", payload, run_id="run-1")
    candidate = copy.deepcopy(baseline)
    candidate["run_id"] = "run-2"
    candidate["capture"]["plan"]["capture"]["captured_step_ids"] = [0, 1]

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "not_passed"
    assert result["acceptance"]["checks"]["profile_benchmark_separated"] is False
    limitation = next(
        item for item in result["acceptance"]["limitations"]
        if item["code"] == "profile_benchmark_not_separated"
    )
    assert "capture_steps_do_not_cover_callers" in limitation["details"]["candidate"]


def test_matched_acceptance_allows_explicit_mlp_without_layer_hierarchy(tmp_path):
    payload = json.loads(SOURCE_TELEMETRY.read_text(encoding="utf-8"))
    baseline = _manifest(tmp_path / "baseline", payload, run_id="run-1")
    candidate = _manifest(tmp_path / "candidate", payload, run_id="run-2")
    baseline["structure"]["layers"] = []
    candidate["structure"]["layers"] = []

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "passed"
    assert result["comparison"]["status"] == "comparable_with_limitations"
    assert result["comparison"]["step_accounting"]["status"] == "comparable"
    assert result["comparison"]["hierarchy"]["status"] == "not_comparable"
    assert result["comparison"]["operator_sequence"]["classification"] == "identical"
    assert result["comparison"]["operator_sequence"]["reuse_recommendation"] == (
        "unavailable_without_explicit_layers"
    )
    assert result["comparison"]["tables"]["layers"] == []
    assert any(
        item["code"] == "structure_reuse_not_established"
        for item in result["comparison"]["limitations"]
    )


LIVE_ACCEPTANCE = Path(os.environ["MINDIE_TEST_CAPTURE_ACCEPTANCE_ROOT"]) if os.environ.get("MINDIE_TEST_CAPTURE_ACCEPTANCE_ROOT") else None
requires_live_manifests = pytest.mark.skipif(
    LIVE_ACCEPTANCE is None or not (LIVE_ACCEPTANCE / "baseline" / "run-manifest.json").exists()
    or not (LIVE_ACCEPTANCE / "candidate" / "run-manifest.json").exists(),
    reason="derived live acceptance manifests are unavailable",
)


def _live_manifests():
    baseline = json.loads((LIVE_ACCEPTANCE / "baseline" / "run-manifest.json").read_text())
    candidate = json.loads((LIVE_ACCEPTANCE / "candidate" / "run-manifest.json").read_text())
    return baseline, candidate


@requires_live_manifests
def test_actual_live_manifest_rejects_input_identity_mismatch():
    baseline, candidate = _live_manifests()
    candidate["scene"]["input_signature"] = candidate["scene"]["input_signature"] + "-mismatch"

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "not_passed"
    assert result["comparison"]["status"] == "not_comparable"
    assert any(
        item["code"] == "input_identity_mismatch"
        for item in result["comparison"]["reasons"]
    )


@requires_live_manifests
def test_actual_live_manifest_rejects_source_hash_mismatch():
    baseline, candidate = _live_manifests()
    candidate["provenance"]["source_files"]["dcmi"]["sha256"] = "0" * 64

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "not_passed"
    assert result["acceptance"]["checks"]["provenance_integrity"] is False
    limitation = next(
        item for item in result["acceptance"]["limitations"]
        if item["code"] == "provenance_integrity_failed"
    )
    assert any(
        "provenance_hash_mismatch:dcmi" in item
        for item in limitation["details"]["candidate"]
    )


@requires_live_manifests
def test_actual_live_manifest_rejects_profile_window_outside_c2():
    baseline, candidate = _live_manifests()
    candidate["profile_measurement_window"]["start_ts"] -= 1.0

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "not_passed"
    assert result["acceptance"]["checks"]["capture_wall_contains_profile"] is False
    assert result["acceptance"]["checks"]["telemetry_window_contains_capture"] is True
    assert any(
        item["code"] == "capture_wall_does_not_contain_profile"
        for item in result["acceptance"]["limitations"]
    )


@requires_live_manifests
def test_actual_sequence_identical_skips_dp_and_real_one_token_change_is_small():
    baseline, _candidate = _live_manifests()
    sequence = baseline["structure"]["operator_sequence"]
    assert len(sequence) >= 1000

    with patch(
        "mindie_profiler.compare_acceptance._edit_distance",
        side_effect=AssertionError("DP must not run for identical sequences"),
    ):
        identical = classify_operator_sequence(
            {"structure": {"operator_sequence": sequence}},
            {"structure": {"operator_sequence": sequence}},
        )

    assert identical["classification"] == "identical"
    assert identical["edit_distance"] == 0
    assert identical["common_prefix"] == len(sequence)
    assert identical["middle_length_baseline"] == 0

    changed = list(sequence)
    changed[len(changed) // 2] = "MUTATED:actual_sequence:test"
    result = classify_operator_sequence(
        {"structure": {"operator_sequence": sequence}},
        {"structure": {"operator_sequence": changed}},
    )
    assert result["classification"] == "small"
    assert result["edit_distance"] == 1
    assert result["common_prefix"] + result["common_suffix"] == len(sequence) - 1


@requires_live_manifests
def test_actual_full_capture_manifests_record_real_health_gap():
    baseline, candidate = _live_manifests()
    assert baseline["capture_wall_window"] == baseline["benchmark"]["profiled"]["measurement_window"]
    assert candidate["capture_wall_window"] == candidate["benchmark"]["profiled"]["measurement_window"]
    assert (
        baseline["capture_wall_window"]["start_ts"]
        <= baseline["profile_measurement_window"]["start_ts"]
        and baseline["capture_wall_window"]["end_ts"]
        >= baseline["profile_measurement_window"]["end_ts"]
    )
    baseline_payload = json.loads(Path(baseline["telemetry"]).read_text())
    candidate_payload = json.loads(Path(candidate["telemetry"]).read_text())
    assert baseline_payload["window"]["scheduled_sample_count"] == baseline_payload["window"]["observed_sample_count"]
    assert candidate_payload["window"]["scheduled_sample_count"] == candidate_payload["window"]["observed_sample_count"]
    assert baseline_payload["window"]["start_ts"] <= baseline["capture_wall_window"]["start_ts"]
    assert baseline_payload["window"]["end_ts"] >= baseline["capture_wall_window"]["end_ts"]

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "inconclusive"
    assert result["comparison"]["step_accounting"]["status"] == "comparable"
    assert result["acceptance"]["checks"]["telemetry_window_contains_capture"] is True
    assert result["acceptance"]["checks"]["telemetry_no_throttling_observed"] is False
    assert any(
        item["code"] == "telemetry_throttling_not_cleared"
        for item in result["acceptance"]["limitations"]
    )


def test_acceptance_consumes_explicit_module_trace_structure_file(tmp_path):
    payload = json.loads(SOURCE_TELEMETRY.read_text(encoding="utf-8"))
    baseline = _manifest(tmp_path / "baseline", payload, run_id="run-1")
    candidate = _manifest(tmp_path / "candidate", payload, run_id="run-2")
    structure = {
        "schema_version": "mindie-module-structure/1",
        "scope": "tiny_mlp_leaf_modules",
        "module_scope": "actual_pytorch_leaf_modules",
        "operator_sequence": ["0:Linear", "1:ReLU", "2:Linear"],
        "event_sequence_shape": {
            "token_count": 3,
            "per_step_counts": {"4": 3},
            "unique_tokens": ["0:Linear", "1:ReLU", "2:Linear"],
            "prefix": ["0:Linear", "1:ReLU", "2:Linear"],
            "suffix": ["0:Linear", "1:ReLU", "2:Linear"],
        },
        "modules": [
            {"module_id": "0", "module_name": "0", "module_type": "Linear", "ordinal": 0,
             "invocation_count": 1, "wall_ms_p50": 0.10, "busy_union_ms_p50": 0.08,
             "host_gap_ms_p50": 0.02, "mapping_coverage": 1.0},
            {"module_id": "1", "module_name": "1", "module_type": "ReLU", "ordinal": 1,
             "invocation_count": 1, "wall_ms_p50": 0.05, "busy_union_ms_p50": 0.04,
             "host_gap_ms_p50": 0.01, "mapping_coverage": 1.0},
        ],
        "layers": [],
    }
    structure_path = tmp_path / "module-trace.json"
    structure_path.write_text(json.dumps({"structure": structure}))
    for manifest in (baseline, candidate):
        manifest.pop("structure", None)
        manifest["structure_file"] = str(structure_path)

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "passed"
    assert result["acceptance"]["checks"]["module_trace_complete"] is True
    assert result["comparison"]["tables"]["modules"]["status"] == "comparable"
    assert result["comparison"]["tables"]["modules"]["scope"] == "tiny_mlp_leaf_modules"
    assert result["comparison"]["tables"]["layers"] == []


LIVE_MARGIN = Path(os.environ["MINDIE_TEST_CAPTURE_MARGIN_ROOT"]) if os.environ.get("MINDIE_TEST_CAPTURE_MARGIN_ROOT") else None
requires_live_margin = pytest.mark.skipif(
    LIVE_MARGIN is None or not (LIVE_MARGIN / "baseline" / "run-manifest.json").exists()
    or not (LIVE_MARGIN / "candidate" / "run-manifest.json").exists(),
    reason="margin-corrected live acceptance manifests are unavailable",
)


@requires_live_margin
def test_actual_margin_corrected_capture_recovers_late_getter_health():
    baseline = json.loads((LIVE_MARGIN / "baseline" / "run-manifest.json").read_text())
    candidate = json.loads((LIVE_MARGIN / "candidate" / "run-manifest.json").read_text())
    assert baseline["telemetry_query_margin_s"] == 0.5
    assert candidate["telemetry_query_margin_s"] == 0.5
    for manifest in (baseline, candidate):
        assert manifest["telemetry_query_window"]["start_ts"] <= manifest["capture_wall_window"]["start_ts"]
        assert manifest["telemetry_query_window"]["end_ts"] >= manifest["capture_wall_window"]["end_ts"]
        payload = json.loads(Path(manifest["telemetry"]).read_text())
        assert payload["window"]["scheduled_sample_count"] == payload["window"]["observed_sample_count"]
        assert payload["window"]["missing_sample_count"] == 0
        assert payload["coverage_ratio"] == 1.0

    result = run_acceptance(baseline, candidate)

    assert result["status"] == "passed"
    assert result["acceptance"]["checks"]["telemetry_no_throttling_observed"] is True
    assert result["acceptance"]["checks"]["telemetry_window_contains_capture"] is True
    assert result["comparison"]["step_accounting"]["status"] == "comparable"
    assert result["comparison"]["hierarchy"]["status"] == "not_comparable"
