"""Real safetensors headers and P0-P4 memory attribution invariants."""

from __future__ import annotations

import json
import stat
import struct
from pathlib import Path

import pytest

from mindie_profiler.memory_profile import (
    attribute_memory,
    compare_peaks,
    parse_vllm_logs,
    to_bytes,
)
from mindie_profiler.weight_inspector import (
    SafetensorsError,
    inspect_weights,
    parse_tensor_info,
    read_safetensors_header,
)

GB = 1_000_000_000


def _write_safetensors(path: Path, entries: list[tuple[str, str, list[int], int]]) -> int:
    header: dict[str, object] = {}
    offset = 0
    for name, dtype, shape, size in entries:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * offset)
    return offset


def _manifest_full() -> dict:
    return {
        "tp": 1,
        "dp": 1,
        "baseline_hbm": {"0": {"used_bytes": 1 * GB, "source_kind": "npu_smi",
                               "measurement_kind": "measured", "is_peak": False}},
        "after_ready_hbm": {"0": {"used_bytes": 4.2 * GB, "source_kind": "npu_smi",
                                  "measurement_kind": "point_observation", "is_peak": False}},
        "after_infer_hbm": {"0": {"used_bytes": 4.25 * GB, "source_kind": "npu_smi",
                                  "measurement_kind": "measured_peak", "is_peak": True}},
        "msprof_components": {
            "APP": {"bytes": 4.3 * GB},
            "HCCL": {"bytes": 100_000_000},
            "RUNTIME": {"bytes": 50_000_000},
            "SLOG": {"bytes": 10_000_000},
        },
        "vllm_info": {
            "weights_bytes": 2 * GB,
            "kv_cache_bytes": 1 * GB,
            "graph_capture_bytes": 100_000_000,
        },
        "weight_manifest": {"status": "ok", "tp": 1, "per_device_bytes": 3 * GB},
        "model_config": {
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "vocab_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "intermediate_size": 16,
        },
    }


def _components(device: dict) -> dict[str, dict]:
    return {item["key"]: item for item in device["components"]}


def test_safetensors_headers_are_read_only_and_sharded_per_device(tmp_path: Path):
    first = tmp_path / "model-00001.safetensors"
    second = tmp_path / "model-00002.safetensors"
    _write_safetensors(first, [
        ("model.embed_tokens.weight", "F16", [4, 2], 16),
        ("model.layers.0.self_attn.q_proj.weight", "F16", [4, 2], 16),
        ("model.layers.0.mlp.experts.0.w1.weight", "F16", [4, 2], 16),
    ])
    _write_safetensors(second, [("model.norm.weight", "F16", [4], 8)])
    first.chmod(0o444)

    header, header_bytes, payload_bytes = read_safetensors_header(first, with_sizes=True)
    assert header["model.embed_tokens.weight"]["data_offsets"] == [0, 16]
    assert header_bytes > 8
    assert payload_bytes == 48

    result = inspect_weights(tmp_path, tp=2, dp=1, enable_expert_parallel=True)
    assert result["status"] == "ok"
    assert result["source"] == "safetensors_file_headers"
    assert result["read_only"] is True
    assert result["total_tensors"] == 4
    assert result["total_bytes"] == 56
    # embedding 8 + q_proj 8 + expert 8 + replicated norm 8
    assert result["per_device_bytes"] == 32
    assert stat.S_IMODE(first.stat().st_mode) == 0o444
    categories = {item["category"] for item in result["categories"]}
    assert {"embedding", "attn_q", "moe_expert", "norm"} <= categories


def test_safetensors_invalid_ranges_and_truncation_are_explicit(tmp_path: Path):
    overlapping = {
        "a": {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]},
        "b": {"dtype": "F16", "shape": [4], "data_offsets": [4, 12]},
    }
    with pytest.raises(SafetensorsError):
        parse_tensor_info(overlapping, payload_bytes=12)

    with pytest.raises(SafetensorsError):
        parse_tensor_info(
            {"a": {"dtype": "F16", "shape": [4], "data_offsets": [0, 80]}},
            payload_bytes=8,
        )

    truncated = tmp_path / "truncated.safetensors"
    raw = json.dumps({"a": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    truncated.write_bytes(struct.pack("<Q", len(raw) + 10) + raw)
    with pytest.raises(SafetensorsError):
        read_safetensors_header(truncated)

    empty = tmp_path / "empty"
    empty.mkdir()
    assert inspect_weights(empty)["status"] == "error"


def test_p0_p4_priority_containers_overlap_and_residual_target():
    result = attribute_memory(_manifest_full(), residual_target_bytes=200 * 1024 * 1024)
    assert result["status"] == "ok"
    assert result["target_met"] is True
    device = result["devices"][0]
    components = _components(device)
    assert set(components) == {
        "baseline_observed", "weights", "kv_cache", "graph", "hccl", "runtime", "slog", "activation",
    }
    assert components["weights"]["source"]["priority"] == "P2"
    assert components["baseline_observed"]["source"]["priority"] == "P1"
    assert components["hccl"]["source"]["priority"] == "P0"
    assert components["kv_cache"]["source"]["priority"] == "P2"
    assert components["activation"]["phase"] == "inference_peak"
    assert components["baseline_observed"]["included_in_sum"] is True
    assert components["hccl"]["included_in_sum"] is False
    assert components["activation"]["included_in_sum"] is False
    assert device["known_sum_bytes"] == 4_100_000_000
    assert device["residual"]["bytes"] == 150_000_000
    assert device["residual"]["status"] == "within_target"
    assert device["residual"]["target_met"] is True
    assert device["residual"]["coverage"] == "complete"
    assert {item["key"] for item in device["references"]} == {"hccl", "runtime", "slog", "activation"}

    # P3/P4 are alternatives for the same logical weights, never added.
    assert [item["key"] for item in device["containers"]] == ["process_total"]
    assert device["containers"][0]["bytes"] == 4_300_000_000
    assert device["containers"][0]["included_in_sum"] is False
    excluded = {(item["key"], item["excluded_priority"]) for item in device["excluded_claims"]}
    assert ("weights", "P3") in excluded
    assert ("weights", "P4") in excluded
    assert device["cross_validation"]["msprof_app_bytes"] == 4_300_000_000
    assert device["cross_validation"]["app_minus_additive_component_sum_bytes"] == 4_300_000_000 - 4_100_000_000

    strict = attribute_memory(_manifest_full(), residual_target_bytes=1)
    strict_device = strict["devices"][0]
    assert strict_device["residual"]["bytes"] == 150_000_000
    assert strict_device["residual"]["status"] == "above_target"
    assert strict_device["residual"]["target_met"] is False
    assert strict["status"] == "partial"


def test_negative_residual_is_reported_not_clamped_or_rounded_to_pass():
    B = 1_000_000_000
    manifest = {
        "baseline_hbm": {"0": {"used_bytes": 1 * B, "source_kind": "npu_smi", "is_peak": False}},
        "after_ready_hbm": {"0": {"used_bytes": 4.0 * B, "source_kind": "npu_smi", "is_peak": False}},
        "after_infer_hbm": {"0": {"used_bytes": 6.0 * B, "source_kind": "npu_smi",
                                  "measurement_kind": "measured_peak", "is_peak": True}},
        "msprof_components": {
            "APP": {"bytes": 4.0 * B},
            "HCCL": {"bytes": 100_000_000},
            "RUNTIME": {"bytes": 50_000_000},
            "SLOG": {"bytes": 10_000_000},
        },
        "vllm_info": {
            "weights_bytes": 6 * B,
            "kv_cache_bytes": 1 * B,
            "graph_capture_bytes": 100_000_000,
        },
    }
    device = attribute_memory(manifest, residual_target_bytes=200 * 1024 * 1024)["devices"][0]
    assert device["known_sum_bytes"] == 8_100_000_000
    assert device["residual"]["bytes"] == -2_100_000_000
    assert device["residual"]["status"] == "negative"
    assert device["residual"]["negative_overage_bytes"] == 2_100_000_000
    assert device["residual"]["target_met"] is False


def test_missing_sources_stay_missing_and_do_not_fill_zero():
    B = 1_000_000_000
    manifest = {
        "after_ready_hbm": {"0": {"used_bytes": 1 * B}},
        "weight_manifest": {"status": "ok", "tp": 1, "per_device_bytes": 500_000_000},
        "vllm_info": {"kv_cache_bytes": 100_000_000},
    }
    device = attribute_memory(manifest, residual_target_bytes=1 * 1024 ** 3)["devices"][0]
    assert "baseline_observed" not in _components(device)
    assert device["residual"]["bytes"] == 400_000_000
    assert device["residual"]["status"] == "within_target"
    assert device["residual"]["coverage"] == "partial"
    assert device["residual"]["target_met"] is False
    codes = {item["code"] for item in device["missing_sources"]}
    assert {"baseline_snapshot_missing", "msprof_components_missing", "target_not_measured_peak"} <= codes
    assert device["components"][0]["bytes"] > 0


def test_p4_config_estimate_is_reference_only_and_cannot_pass():
    B = 1_000_000_000
    manifest = {
        "baseline_hbm": {"0": {"used_bytes": 1 * B}},
        "after_ready_hbm": {"0": {"used_bytes": 1 * B}},
        "vllm_info": {"kv_cache_bytes": 100_000_000},
        "model_config": {
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "vocab_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "intermediate_size": 16,
        },
    }
    device = attribute_memory(manifest, residual_target_bytes=1 * 1024 ** 3)["devices"][0]
    weights = _components(device)["weights"]
    assert weights["source"]["priority"] == "P4"
    assert weights["peak"]["kind"] == "reference"
    assert "weights_measured_missing" in {item["code"] for item in device["missing_sources"]}
    assert device["residual"]["target_met"] is False


def test_requested_inference_peak_without_inference_snapshot_cannot_pass():
    manifest = _manifest_full()
    manifest.pop("after_infer_hbm")
    device = attribute_memory(manifest, target_phase="inference_peak")["devices"][0]
    assert device["target_phase_requested"] == "inference_peak"
    assert device["target_phase"] == "ready"
    assert device["residual"]["coverage"] == "partial"
    assert device["residual"]["target_met"] is False
    assert "target_phase_unavailable" in {item["code"] for item in device["missing_sources"]}


def test_sampled_peak_is_explicit_and_can_pass_only_with_complete_sources():
    manifest = {
        "baseline_hbm": {"0": {"used_bytes": 1_000_000_000, "source_kind": "torch_npu_runtime_api",
                               "measurement_kind": "point_observation", "is_peak": False}},
        "after_ready_hbm": {"0": {"used_bytes": 1_200_000_000, "source_kind": "torch_npu_runtime_api",
                                  "measurement_kind": "point_observation", "is_peak": False}},
        "sampled_peak_hbm": {"0": {"used_bytes": 1_200_000_000, "source_kind": "torch_npu_runtime_api",
                                   "measurement_kind": "sampled_device_peak", "is_peak": True,
                                   "peak_scope": "sampled", "continuous_peak": False,
                                   "absolute_peak_limited": True, "sample_count": 25,
                                   "sample_window_ns": [10, 20], "coverage_ratio": 1.0,
                                   "cadence_ms": {"target_coverage_complete": True,
                                                  "expected_slots": 25, "completed_slots": 25,
                                                  "missing_slots": 0, "late_slots": 0,
                                                  "failed_samples": 0}}},
        "msprof_components": {"HCCL": {"bytes": 10_000_000}, "RUNTIME": {"bytes": 20_000_000}},
        "weight_manifest": {"status": "ok", "tp": 1, "per_device_bytes": 50_000_000},
    }
    result = attribute_memory(manifest, residual_target_bytes=200 * 1024 * 1024,
                              target_phase="sampled_peak", kv_cache_applicable=False)
    device = result["devices"][0]
    assert device["target_phase"] == "sampled_peak"
    assert device["target_is_peak"] is True
    assert device["target_measurement_kind"] == "sampled_device_peak"
    assert device["target_peak_scope"] == "sampled"
    assert device["target_continuous_peak"] is False
    assert device["target_absolute_peak_limited"] is True
    assert device["target"]["peak"]["is_peak"] is True
    assert device["known_sum_bytes"] == 1_050_000_000
    assert device["residual"]["bytes"] == 150_000_000
    assert device["residual"]["status"] == "within_target"
    assert device["residual"]["coverage"] == "complete"
    assert result["target_met"] is True
    assert "sampled_peak_not_absolute" in {item["code"] for item in device["limitations"]}
    comparison = compare_peaks(device["peak"], device["peak"])
    assert comparison["comparable"] is True


def test_partial_sampled_coverage_keeps_max_but_blocks_target_met():
    manifest = {
        "baseline_hbm": {"0": {"used_bytes": 1_000_000_000, "source_kind": "torch_npu_runtime_api",
                               "measurement_kind": "point_observation", "is_peak": False}},
        "sampled_peak_hbm": {"0": {"used_bytes": 1_200_000_000, "source_kind": "torch_npu_runtime_api",
                                   "measurement_kind": "sampled_device_peak", "is_peak": True,
                                   "peak_scope": "sampled", "continuous_peak": False,
                                   "absolute_peak_limited": True, "sample_count": 3,
                                   "sample_window_ns": [10, 20], "coverage_ratio": 0.2,
                                   "cadence_ms": {"target_coverage_complete": False,
                                                  "expected_slots": 10, "completed_slots": 3,
                                                  "missing_slots": 7, "late_slots": 0,
                                                  "failed_samples": 0}}},
        "msprof_components": {"RUNTIME": {"bytes": 20_000_000}},
        "weight_manifest": {"status": "ok", "tp": 1, "per_device_bytes": 50_000_000},
    }
    result = attribute_memory(manifest, target_phase="sampled_peak", kv_cache_applicable=False)
    device = result["devices"][0]
    assert device["target_is_peak"] is True
    assert device["target_coverage_complete"] is False
    assert device["residual"]["target_met"] is False
    assert device["residual"]["coverage"] == "partial"
    assert "sampled_peak_coverage_incomplete" in {item["code"] for item in device["missing_sources"]}
    assert device["target"]["peak"]["is_peak"] is True
    assert device["target"]["coverage_complete"] is False


def test_peak_comparison_requires_same_phase_and_scope():
    baseline = {"bytes": 100, "phase": "inference_peak", "scope": "device:0", "is_peak": True,
                "observed_at": "2026-01-01T00:00:00Z"}
    candidate = {"bytes": 150, "phase": "inference_peak", "scope": "device:0", "is_peak": True,
                 "observed_at": "2026-01-01T00:00:01Z"}
    same_phase = compare_peaks(baseline, candidate)
    assert same_phase["comparable"] is True
    assert same_phase["delta_bytes"] == 50
    assert same_phase["exact_moment"] is False

    other_phase = dict(candidate, phase="ready")
    mismatch = compare_peaks(baseline, other_phase)
    assert mismatch["comparable"] is False
    assert mismatch["reasons"][0]["code"] == "peak_phase_mismatch"

    important = compare_peaks(baseline, dict(candidate, is_peak=False))
    assert important["comparable"] is False
    assert {item["code"] for item in important["reasons"]} == {"not_a_peak"}


def test_units_and_framework_log_parsing_are_explicit():
    assert to_bytes(1, "MiB") == 1024 ** 2
    assert to_bytes(1, "MB") == 1_000_000
    assert to_bytes(1, "GiB") == 1024 ** 3
    with pytest.raises(ValueError):
        to_bytes(1, "furlong")

    parsed = parse_vllm_logs(
        "Loading model weights took 9.5 GB\n"
        "Available KV cache memory: 13.75 GiB\n"
        "GPU KV cache size: 1,234 tokens\n"
        "Graph capturing finished in 3 secs, took 0.49 GiB\n"
        "gpu_memory_utilization=0.85\n"
    )
    assert parsed["weights_gb"] == 9.5
    assert parsed["kv_cache_available_gib"] == 13.75
    assert parsed["kv_cache_tokens"] == 1234
    assert parsed["graph_capture_gib"] == 0.49
    assert parsed["gpu_memory_utilization"] == 0.85


@pytest.mark.parametrize("residual,accepted", [(199_999_999, True), (200_000_000, False), (205_000_000, False)])
def test_default_residual_target_uses_strict_decimal_megabytes(residual, accepted):
    manifest = _manifest_full()
    manifest["after_infer_hbm"]["0"]["used_bytes"] = 4_100_000_000 + residual
    result = attribute_memory(manifest)
    assert result["devices"][0]["residual"]["bytes"] == residual
    assert result["target_met"] is accepted
