"""Actual A3 NPU_MEM and P2 memory-evidence regression tests.

Fixtures are exact slices of the immutable online-v4-a3 result; no NPU,
server, or remote call is performed.  Source hashes are pinned in provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from mindie_profiler.memory_db import load_memory_record_sqlite, load_npu_mem_sqlite, load_op_memory_sqlite
from mindie_profiler.memory_evidence import MemoryEvidenceError, manifest_from_snapshots
from mindie_profiler.memory_p0 import op_memory_candidates
from mindie_profiler.memory_logs import parse_vllm_logs
from mindie_profiler.memory_profile import _candidate, attribute_memory, estimate_weights_from_config

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "profiler_core" / "memory_online_a3"
DB = FIXTURES / "npu_mem_slice.db"
RECORD_DB = FIXTURES / "memory_record_op_slice.db"
OP_CSV = FIXTURES / "operator_memory_large.csv"
LOG = FIXTURES / "server_stdout_memory_lines.log"
CONFIG = FIXTURES / "config.json"
MEMORY_RECORD = FIXTURES / "memory_record_first_app.csv"
PROVENANCE = json.loads((FIXTURES / "provenance.json").read_text())


def test_npu_mem_loader_uses_actual_bytes_and_keeps_app_device_separate():
    loaded = load_npu_mem_sqlite(DB)
    assert loaded["source_sha256"] == PROVENANCE["slice_db_sha256"]
    assert loaded["rows"] == 40
    assert loaded["complete"] is True
    assert loaded["unit"] == "byte"
    assert loaded["type_names"] == {"36": "app", "37": "device"}
    assert loaded["unknown_type_ids"] == []
    assert loaded["timestamp_range_ns"] == [1792445431714962944, 1792445432089321984]
    device = loaded["devices"]["0"]
    assert device["app"]["max_bytes"] == 26_626_957_312
    assert device["app"]["min_bytes"] == 26_622_468_096
    assert device["app"]["samples"] == 20
    assert device["device"]["max_bytes"] == 26_932_236_288
    assert device["device"]["min_bytes"] == 26_920_824_832
    assert device["device"]["samples"] == 20
    assert device["app"]["ddr_max_bytes"] == 0
    assert device["device"]["ddr_max_bytes"] == 0
    assert device["app"]["scope"] == "app_reserved"
    assert device["app"]["pytorch_allocator_identity_proven"] is False
    assert device["device"]["scope"] == "device_observed"
    assert device["device"]["physical_whole_device_proven"] is False
    assert device["app"]["measurement_kind"] == "sampled_max"
    assert device["app"]["is_peak"] is False
    assert device["device"]["is_peak"] is True
    assert device["device"]["peak_scope"] == "sampled"
    assert device["device"]["continuous_peak"] is False
    assert device["device"]["absolute_peak_limited"] is True
    assert device["device"]["schedule_known"] is False
    assert device["device"]["coverage_status"] == "unknown"
    assert device["device"]["observed_intervals_ns"]["p50"] == 19_988_992


def test_unit_basis_matches_actual_memory_record_mb_conversion():
    with MEMORY_RECORD.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    reserved_mb = float(rows[0]["Total Reserved(MB)"])
    app_min = load_npu_mem_sqlite(DB)["devices"]["0"]["app"]["min_bytes"]
    assert reserved_mb * 1024 * 1024 == app_min
    assert PROVENANCE["slice_memory_record_sha256"] == hashlib.sha256(MEMORY_RECORD.read_bytes()).hexdigest()


def test_op_memory_csv_kb_matches_db_bytes():
    with OP_CSV.open(newline="") as handle:
        row = list(csv.DictReader(handle))[0]
    assert float(row["Size(KB)"]) * 1024 == 13_197_641_216
    assert float(row["Allocation Total Allocated(MB)"]) * 1024 * 1024 == 26_319_992_320
    assert PROVENANCE["slice_operator_memory_sha256"] == hashlib.sha256(OP_CSV.read_bytes()).hexdigest()


def test_npu_mem_scope_follows_string_ids_after_dictionary_reindex(tmp_path):
    """String ids are local to each export; identical names keep their scope."""
    target = tmp_path / "reindexed.db"
    shutil.copyfile(DB, target)
    with sqlite3.connect(target) as connection:
        connection.execute("UPDATE STRING_IDS SET id = id + 1000")
        connection.execute("UPDATE NPU_MEM SET type = type + 1000")
    original = load_npu_mem_sqlite(DB)
    loaded = load_npu_mem_sqlite(target)
    assert loaded["complete"] is True
    assert loaded["devices"]["0"]["app"]["max_bytes"] == original["devices"]["0"]["app"]["max_bytes"]
    assert loaded["devices"]["0"]["device"]["max_bytes"] == original["devices"]["0"]["device"]["max_bytes"]
    assert loaded["devices"]["0"]["app"]["type_id"] == 1036


def test_memory_record_and_op_memory_reconcile_the_13_2gb_active_spike():
    record = load_memory_record_sqlite(RECORD_DB)
    ops = load_op_memory_sqlite(RECORD_DB)
    pta = record["devices"]["0"]["components"]["PTA"]
    assert pta["max_total_allocated_bytes"] == 26_319_992_320
    assert pta["min_total_allocated_bytes"] == 13_122_348_032
    assert pta["max_total_reserved_bytes"] == 26_342_326_272
    assert pta["max_total_active_bytes"] == 26_319_992_320
    assert record["unit"] == "byte"
    assert record["complete"] is True
    assert ops["device_ids"] == ["0"]
    assert ops["workspace_semantics_proven"] is False
    largest = ops["devices"]["0"]["largest_allocation"]
    assert largest["name"] == "npu::npu_fused_infer_attention_score"
    assert largest["size_bytes"] == 13_197_641_216
    assert largest["allocation_total_allocated_bytes"] == 26_319_992_320
    assert largest["release_total_allocated_bytes"] == 13_122_351_104
    assert ops["devices"]["0"]["by_name"]["npu::npu_fused_infer_attention_score"]["count"] == 64
    assert len(ops["devices"]["0"]["large_allocations"]) == 14
    assert ops["unit"] == "byte"


def test_actual_p2_log_parses_npu_kv_current_budget_and_startup_facts():
    parsed = parse_vllm_logs(LOG.read_text())
    assert parsed["weights_gb"] == 0.0001
    assert parsed["weights_unit"] == "GB"
    assert parsed["kv_cache_available_gib"] == 12.22
    assert parsed["kv_cache_available_is_budget"] is True
    assert parsed["kv_cache_current_gib"] == 12.22
    assert parsed["kv_cache_current_is_actual"] is True
    assert parsed["kv_cache_tokens"] == 102_483_968
    assert parsed["startup_free_gib"] == 60.88
    assert parsed["startup_total_gib"] == 61.27
    assert parsed["desired_gpu_memory_utilization"] == 0.2
    assert parsed["desired_budget_gib"] == 12.25
    assert parsed["weights_usage_gib"] == 0.0
    assert parsed["peak_activation_gib"] == 0.02
    assert parsed["non_torch_memory_gib"] == 0.02
    assert parsed["npu_graph_memory_gib"] == 0.0
    assert parsed["torch_reserved_gib"] == 12.23
    assert parsed["torch_allocated_gib"] == 12.22
    assert parsed["kv_cache_memory_fit_bytes"] == 12_960_670_720
    assert parsed["kv_cache_memory_full_bytes"] == 65_173_402_624
    assert parsed["raw"]["kv_cache_available_gib"] == "12.22 GiB"
    assert parsed["raw"]["kv_cache_current_gib"] == "12.22 GiB"
    assert parsed["raw"]["weights_gb"] == "0.0001 GB"


def test_actual_a3_attribution_keeps_p0_references_separate_and_target_truthful():
    loaded = load_npu_mem_sqlite(DB)
    p2 = parse_vllm_logs(LOG.read_text())
    config = json.loads(CONFIG.read_text())
    manifest = manifest_from_snapshots(
        {},
        npu_mem_by_device=loaded["devices"],
        msprof_components_by_device={"0": {"RUNTIME": {"bytes": 87_941_120}}},
        vllm_info=p2,
        model_config=config,
        kv_cache_applicable=True,
    )
    ops = load_op_memory_sqlite(RECORD_DB)
    manifest["op_memory_by_device"] = ops
    result = attribute_memory(manifest, target_phase="npu_mem_device")
    device = result["devices"][0]
    assert result["status"] == "partial"
    assert result["target_met"] is False
    assert device["target_phase"] == "npu_mem_device"
    assert device["target"]["bytes"] == 26_932_236_288
    assert device["target_is_peak"] is True
    assert device["target_peak_scope"] == "sampled"
    assert device["target_continuous_peak"] is False
    assert device["target_absolute_peak_limited"] is True
    assert device["target_coverage_complete"] is False
    assert device["residual"]["status"] == "above_target"
    assert device["residual"]["bytes"] == pytest.approx(13_811_011_198.72, rel=1e-9)
    assert device["residual"]["target_met"] is False
    assert device["residual"]["approximate"] is True
    assert "p2_rounded_values" in {item["code"] for item in device["limitations"]}
    assert device["known_sum_bytes"] == 13_121_225_089.28
    components = {item["key"]: item for item in device["components"]}
    assert components["transient_operator_allocation"]["bytes"] == 13_197_641_216
    assert components["transient_operator_allocation"]["included_in_sum"] is False
    assert components["transient_operator_allocation"]["overlap_group"] == "transient_scope_unverified"
    assert components["runtime"]["included_in_sum"] is False
    assert components["runtime"]["overlap_group"] == "p0_scope_unknown"
    assert "p0_component_scope_unknown" in {item["code"] for item in device["limitations"]}
    assert components["weights"]["bytes"] == 100_000
    assert components["weights"]["source"]["unit"] == "GB"
    assert components["kv_cache"]["bytes"] == 13_121_125_089.28
    assert components["kv_cache"]["source"]["kind"] == "vllm_log"
    assert components["kv_cache"]["source"]["raw"] == 12.22
    assert components["npu_device_sampled"]["included_in_sum"] is False
    assert components["npu_app_sampled"]["included_in_sum"] is False
    assert components["npu_device_sampled"]["source"]["scope"].startswith("device_observed:")
    assert components["npu_app_sampled"]["source"]["scope"].startswith("app_reserved:")
    assert "baseline_snapshot_missing" in {item["code"] for item in device["missing_sources"]}
    assert "transient_operator_disjointness_unproven" in {item["code"] for item in device["limitations"]}
    assert "transient_operator_workspace_unproven" in {item["code"] for item in device["limitations"]}
    assert "sampled_peak_coverage_incomplete" in {item["code"] for item in device["missing_sources"]}
    assert result["source_applicability"]["P0"]["status"] == "present"
    assert result["source_applicability"]["P1"]["status"] == "missing"
    assert result["source_applicability"]["P2"]["status"] == "present"
    assert result["source_applicability"]["P3"]["status"] == "missing"
    assert result["p2_facts"]["kv_cache_current_gib"] == 12.22
    assert result["p2_facts"]["kv_cache_available_gib"] == 12.22
    estimate = estimate_weights_from_config(config)
    assert estimate["per_device_bytes"] == 107_008
    assert estimate["confidence"] == "reference_only"


def test_actual_transient_allocation_never_sums_an_unrelated_window_maximum():
    ops = load_op_memory_sqlite(RECORD_DB)["devices"]["0"]
    allocation = ops["largest_allocation"]
    for window in (
        (1792445431790000000, 1792445432090000000),
        (allocation["allocation_time_ns"], allocation["release_time_ns"]),
        (1792445432090000001, 1792445432095000000),
        None,
    ):
        references, limitations = op_memory_candidates(
            ops, device_id=0, candidate=_candidate, target_window_ns=window)
        assert references[0]["included_in_sum"] is False
        assert references[0]["bytes"] == 13_197_641_216
        assert "transient_operator_disjointness_unproven" in {item["code"] for item in limitations}


def test_device_partition_and_attribution_use_selected_device_op_memory(tmp_path):
    target = tmp_path / "two_devices.db"
    shutil.copyfile(RECORD_DB, target)
    with sqlite3.connect(target) as connection:
        connection.execute(
            "INSERT INTO OP_MEMORY VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (536870927, 20_000_000_000, 1792445431799000000, 1792445431799010000,
             1792445431799009000, 10000, 9000, 26_319_992_320, 26_342_326_272, 26_319_992_320,
             13_122_351_104, 26_342_326_272, 13_122_351_104, 47, 1),
        )
    loaded = load_op_memory_sqlite(target)
    assert loaded["device_ids"] == ["0", "1"]
    assert loaded["devices"]["0"]["largest_allocation"]["size_bytes"] == 13_197_641_216
    assert loaded["devices"]["1"]["largest_allocation"]["size_bytes"] == 20_000_000_000
    manifest = manifest_from_snapshots({}, op_memory_by_device=loaded)
    result = attribute_memory(manifest)
    by_device = {item["device_id"]: item for item in result["devices"]}
    assert by_device[0]["components"][0]["key"] == "transient_operator_allocation"
    assert by_device[0]["components"][0]["bytes"] == 13_197_641_216
    assert by_device[1]["components"][0]["bytes"] == 20_000_000_000


def test_row_caps_are_bounded_for_both_loaders():
    with pytest.raises(MemoryEvidenceError):
        load_memory_record_sqlite(RECORD_DB, max_rows=10)
    with pytest.raises(MemoryEvidenceError):
        load_op_memory_sqlite(RECORD_DB, max_rows=10)


def test_available_kv_budget_is_not_used_as_actual_kv_allocation():
    manifest = manifest_from_snapshots(
        {},
        npu_mem_by_device=load_npu_mem_sqlite(DB)["devices"],
        vllm_info={"kv_cache_available_gib": 12.22},
        kv_cache_applicable=True,
    )
    result = attribute_memory(manifest, target_phase="npu_mem_device")
    device = result["devices"][0]
    assert not any(item["key"] == "kv_cache" for item in device["components"])
    assert "kv_cache_missing" in {item["code"] for item in device["missing_sources"]}
    assert "kv_cache_available_budget_only" in {item["code"] for item in device["limitations"]}
