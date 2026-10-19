"""Real first-A2 memory capture regression: source, point, and overlap honesty.

Fixtures are read-only slices of the actual capture:
DB SHA256 adb931778fd1759dd2bf39e9b0ec725602536402363ce274da7d1fb3ede6ef22
and the result JSON's memory_snapshots/weight_manifest.  No raw artifact is
modified and no NPU workload is launched here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from mindie_profiler.memory_evidence import (
    MemoryEvidenceError,
    load_npu_module_mem_sqlite,
    manifest_from_snapshots,
    msprof_components_to_manifest,
)
from mindie_profiler.memory_profile import attribute_memory, compare_peaks

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "profiler_core" / "memory_real_capture"
DB_PATH = FIXTURE / "npu_module_mem_slice.db"
INPUTS_PATH = FIXTURE / "actual_memory_inputs.json"
PROVENANCE_PATH = FIXTURE / "provenance.json"


def _inputs() -> dict:
    return json.loads(INPUTS_PATH.read_text(encoding="utf-8"))


def _corrected():
    data = _inputs()
    module = load_npu_module_mem_sqlite(DB_PATH)
    components = msprof_components_to_manifest(module)
    manifest = manifest_from_snapshots(
        data["memory_snapshots"],
        msprof_components_by_device=components,
        weight_manifest=data["weight_manifest"],
        kv_cache_applicable=data["kv_cache_applicable"],
    )
    attribution = attribute_memory(
        manifest,
        residual_target_bytes=200 * 1024 * 1024,
        target_phase="inference_peak",
    )
    return module, manifest, attribution


def test_real_module_mem_sqlite_slice_preserves_enum_timestamps_and_scope():
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    module = load_npu_module_mem_sqlite(DB_PATH)
    assert module["source_kind"] == "msprof_npu_module_mem_sqlite"
    assert module["measurement_kind"] == "sampled_max"
    assert module["unit"] == "byte"
    assert module["rows"] == provenance["selection"]["npu_module_mem_rows"] == 847
    assert module["complete"] is False
    assert module["unmapped_module_ids"] == [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 46, 59]
    assert module["timestamp_range_ns"] == provenance["selection"]["timestamps_ns"]
    device = module["devices"]["0"]
    assert device["APP"]["max_bytes"] == 2_097_152.0
    assert device["RUNTIME"]["max_bytes"] == 33_660_928.0
    assert device["APP"]["samples"] == 11
    assert device["APP"]["first_timestamp_ns"] < device["APP"]["last_timestamp_ns"]
    assert device["RUNTIME"]["module_id"] == 7
    manifest_components = msprof_components_to_manifest(module)["0"]
    assert manifest_components["RUNTIME"]["measurement_kind"] == "sampled_max"
    assert manifest_components["RUNTIME"]["first_timestamp_ns"] == device["RUNTIME"]["first_timestamp_ns"]


def test_actual_manifest_preserves_runtime_api_source_and_point_semantics():
    _module, manifest, _attribution = _corrected()
    assert manifest["architecture"]["kv_cache_applicable"] is False
    baseline = manifest["baseline_hbm"]["0"]
    assert baseline["source_kind"] == "torch_npu_runtime_api"
    assert baseline["measurement_kind"] == "point_observation"
    assert baseline["is_peak"] is False
    assert "npu-smi" not in baseline["source"]
    assert manifest["source_snapshots"]["inference_peak"]["is_peak"] is False


def test_actual_attribution_no_source_relabel_no_double_count_no_fake_peak():
    _module, _manifest, result = _corrected()
    assert result["status"] == "partial"
    assert result["target_met"] is False
    assert result["kv_cache_applicable"] is False
    device = result["devices"][0]
    components = {item["key"]: item for item in device["components"]}
    references = {item["key"]: item for item in device["references"]}

    baseline = components["baseline_observed"]
    assert baseline["label"] == "Observed runtime-initialized HBM baseline"
    assert baseline["source"]["kind"] == "torch_npu_runtime_api"
    assert "npu-smi" not in baseline["source"]["priority_label"]
    assert baseline["included_in_sum"] is True
    assert baseline["peak"]["is_peak"] is False
    assert "not proof of driver-only overhead" in baseline["evidence"]

    target = device["target"]
    assert target["is_peak"] is False
    assert target["measurement_kind"] == "point_observation"
    assert target["peak"]["is_peak"] is False
    assert target["source"]["kind"] == "torch_npu_runtime_api"
    assert "target_not_measured_peak" in {item["code"] for item in device["missing_sources"]}
    assert device["residual"]["coverage"] == "partial"
    assert device["residual"]["target_met"] is False

    runtime = references["runtime"]
    assert runtime["source"]["priority"] == "P0"
    assert runtime["measurement_kind"] == "sampled_max"
    assert runtime["peak"]["is_peak"] is False
    assert runtime["included_in_sum"] is False
    assert runtime["source"]["raw"]["first_timestamp_ns"] == 1792441933207981905
    assert runtime["source"]["raw"]["last_timestamp_ns"] == 1792441933395243115
    assert "hccl" not in components and "slog" not in components

    activation = references["activation"]
    assert "Activation peak" not in activation["label"]
    assert activation["measurement_kind"] == "point_delta"
    assert activation["peak"]["is_peak"] is False
    assert activation["included_in_sum"] is False
    assert "profiler allocations" in activation["evidence"]

    # baseline 377,053,184 + P3 weights 1,051,648; P0 runtime and the point
    # delta stay reference-only to avoid double counting the device baseline.
    assert device["known_sum_bytes"] == 378_104_832.0
    assert device["reference_sum_bytes"] == 43_098_112.0
    assert device["residual"]["bytes"] == 10_482_688.0
    assert device["residual"]["status"] == "within_target"
    assert device["cross_validation"]["p0_overlap_bytes"] == 33_660_928.0
    assert device["cross_validation"]["activation_point_delta_bytes"] == 9_437_184.0
    assert "runtime" not in {item for item in components if components[item]["included_in_sum"]}
    assert any(item["code"] == "kv_cache_not_applicable" for item in device["not_applicable_sources"])
    assert result["source_applicability"]["P0"]["status"] == "present"
    assert result["source_applicability"]["P1"]["status"] == "present"
    assert result["source_applicability"]["P2"]["status"] == "not_provided"
    assert result["source_applicability"]["P3"]["status"] == "present"
    assert result["source_applicability"]["P4"]["status"] == "not_provided"

    peak = device["peak"]
    comparison = compare_peaks(peak, peak)
    assert comparison["comparable"] is False
    assert {item["code"] for item in comparison["reasons"]} == {"not_a_peak"}


def test_kv_unknown_remains_missing_not_zero():
    data = _inputs()
    module = load_npu_module_mem_sqlite(DB_PATH)
    manifest = manifest_from_snapshots(
        data["memory_snapshots"],
        msprof_components_by_device=msprof_components_to_manifest(module),
        weight_manifest=data["weight_manifest"],
    )
    result = attribute_memory(manifest, target_phase="inference_peak")
    device = result["devices"][0]
    assert "kv_cache" not in {item["key"] for item in device["components"]}
    assert "kv_cache_missing" in {item["code"] for item in device["missing_sources"]}
    assert not any(item["code"] == "kv_cache_not_applicable" for item in device["not_applicable_sources"])


def test_unknown_module_mem_schema_fails_explicitly(tmp_path):
    broken = tmp_path / "broken.db"
    con = sqlite3.connect(broken)
    con.execute("CREATE TABLE NPU_MODULE_MEM(moduleId INTEGER, timestampNs INTEGER, totalReserved NUMERIC, deviceId INTEGER)")
    con.execute("INSERT INTO NPU_MODULE_MEM VALUES (7, 1, 33660928, 0)")
    con.commit()
    con.close()
    with pytest.raises(MemoryEvidenceError) as caught:
        load_npu_module_mem_sqlite(broken)
    assert caught.value.code == "sqlite_schema_missing"


def test_live_harness_prefers_actual_capture_db_module_mem():
    script = Path(__file__).resolve().parent / "live_framework_capture_workload.py"
    spec = importlib.util.spec_from_file_location("live_framework_capture_workload_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    errors: list[dict] = []
    capture = {
        "artifacts": {
            "artifacts": [{
                "kind": "db",
                "non_empty": True,
                "path": str(DB_PATH),
                "modified_ns": 1,
            }]
        }
    }
    components, source = module._module_mem_evidence(capture, None, errors)
    assert errors == []
    assert source["kind"] == "npu_module_mem_sqlite"
    assert source["sha256"] == hashlib.sha256(DB_PATH.read_bytes()).hexdigest()
    assert source["complete"] is False
    assert source["nonzero_components"]["0"]["RUNTIME"] == 33_660_928.0
    assert components["0"]["RUNTIME"]["measurement_kind"] == "sampled_max"


def test_source_priorities_do_not_rename_runtime_api_to_npu_smi():
    _module, _manifest, result = _corrected()
    assert result["source_priorities"]["P1"].startswith("whole-device runtime observation")
    device = result["devices"][0]
    for item in device["components"]:
        if item["source"]["kind"] == "torch_npu_runtime_api":
            assert item["source"]["priority_label"] == result["source_priorities"]["P1"]
