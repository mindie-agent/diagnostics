"""Real-format memory evidence parsers and missing-value invariants."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_profiler.memory_evidence import (
    BoundedMemorySampler,
    MemoryEvidenceError,
    collect_runtime_memory_snapshot,
    manifest_from_snapshots,
    msprof_components_to_manifest,
    sampled_peak_to_snapshot,
    parse_msmemscope_snapshot_csv,
    parse_npu_mem_csv,
    parse_npu_module_mem_csv,
    take_msmemscope_snapshot,
)
from mindie_profiler.memory_sampler import summarize_sampling

MODULE_HEADER = "Device_id,Component,Timestamp(us),Total Reserved(KB),Device\n"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_npu_module_mem_kb_peak_and_invalid_row(tmp_path: Path):
    path = _write(tmp_path, "npu_module_mem_0.csv", MODULE_HEADER +
                  "0,APP,10,100,NPU:0\n"
                  "0,APP,20,150,NPU:0\n"
                  "0,HCCL,20,25,NPU:0\n"
                  "0,RUNTIME,20,not-a-number,NPU:0\n")
    parsed = parse_npu_module_mem_csv(path)
    assert parsed["unit"] == "byte"
    assert parsed["unit_basis"] == "binary"
    assert parsed["source_unit"] == "KB"
    assert parsed["devices"]["0"]["APP"]["max_bytes"] == 150 * 1024
    assert parsed["devices"]["0"]["APP"]["samples"] == 2
    assert parsed["devices"]["0"]["HCCL"]["max_bytes"] == 25 * 1024
    assert parsed["complete"] is False
    assert parsed["invalid_rows"][0]["reason"] == "invalid_value"

    manifest = msprof_components_to_manifest(parsed)
    assert manifest["0"]["APP"]["bytes"] == 150 * 1024


def test_npu_module_mem_mb_column_is_recorded_explicitly(tmp_path: Path):
    path = _write(tmp_path, "npu_module_mem_1.csv",
                  "Device_id,Component,Timestamp(us),Total Reserved(MB),Device\n"
                  "1,SLOG,5,2.5,NPU:1\n")
    parsed = parse_npu_module_mem_csv(path)
    assert parsed["source_unit"] == "MB"
    assert parsed["devices"]["1"]["SLOG"]["max_bytes"] == 2.5 * 1024 ** 2


def test_npu_mem_device_and_app_are_kept_separate(tmp_path: Path):
    path = _write(tmp_path, "npu_mem_0.csv",
                  "Device_id,event,hbm(KB),timestamp(us)\n"
                  "0,Device,1000,10\n"
                  "0,Device,1200,20\n"
                  "0,APP,400,20\n")
    parsed = parse_npu_mem_csv(path)
    assert parsed["devices"]["0"]["device_bytes"] == 1200 * 1024
    assert parsed["devices"]["0"]["app_bytes"] == 400 * 1024
    assert parsed["devices"]["0"]["device_minus_app_bytes"] == 800 * 1024
    assert parsed["complete"] is True


def test_msmemscope_snapshot_attr_maps_device_and_allocator(tmp_path: Path):
    path = _write(tmp_path, "memscope_dump.csv",
                  "ID,Event,Event Type,Name,Timestamp(ns),Process Id,Thread Id,Device Id,Ptr,Attr\n"
                  "1,SNAPSHOT,N/A,ready,123,7,8,0,N/A,"
                  "\"{total_mem:1000,free_mem:400,reserved:500,peak_reserved:550,allocated:300,peak_allocated:350}\"\n")
    parsed = parse_msmemscope_snapshot_csv(path)
    assert parsed["complete"] is True
    snapshot = parsed["snapshots"][0]
    assert snapshot["device"] == "0"
    assert snapshot["device_used_bytes"] == 600
    assert snapshot["allocator_reserved_bytes"] == 500
    assert snapshot["allocator_allocated_bytes"] == 300
    assert snapshot["values"]["peak_allocated"] == 350


def test_manifest_from_snapshots_omits_missing_device_phase():
    snapshots = {
        "baseline": {"device": 0, "device_memory": {"used_bytes": 1000}, "source": "probe"},
        "ready": {"device": 0, "device_memory": {"used_bytes": 3000}, "source": "probe"},
        "inference_peak": {"device": 0, "device_memory": {"free_bytes": 1}, "source": "probe"},
    }
    manifest = manifest_from_snapshots(
        snapshots,
        weight_manifest={"status": "ok", "per_device_bytes": 500},
        model_config={"hidden_size": 8, "num_hidden_layers": 1, "vocab_size": 8},
    )
    assert manifest["baseline_hbm"]["0"]["used_bytes"] == 1000
    assert manifest["after_ready_hbm"]["0"]["used_bytes"] == 3000
    assert "after_infer_hbm" not in manifest
    assert manifest["limitations"][0]["code"] == "device_used_missing"
    assert manifest["weight_manifest"]["per_device_bytes"] == 500


def test_runtime_snapshot_requires_exact_npu_api_without_zero_fill():
    with pytest.raises(MemoryEvidenceError) as caught:
        collect_runtime_memory_snapshot(SimpleNamespace(npu=None), phase="ready")
    assert caught.value.code == "torch_npu_missing"

    with pytest.raises(MemoryEvidenceError) as caught:
        collect_runtime_memory_snapshot(SimpleNamespace(npu=SimpleNamespace()), phase="ready")
    assert caught.value.code == "torch_npu_api_missing"


def test_runtime_snapshot_preserves_torch_npu_source_and_point_kind():
    class FakeNpu:
        def synchronize(self):
            return None

        def memory_allocated(self, device):
            return 100

        def memory_reserved(self, device):
            return 200

        def max_memory_allocated(self, device):
            return 150

        def max_memory_reserved(self, device):
            return 250

        def mem_get_info(self, device):
            return 800, 1000

    snapshot = collect_runtime_memory_snapshot(SimpleNamespace(npu=FakeNpu()), phase="baseline")
    assert snapshot["source_kind"] == "torch_npu_runtime_api"
    assert snapshot["source_priority"] == "P1"
    assert snapshot["measurement_kind"] == "point_observation"
    assert snapshot["is_peak"] is False
    assert snapshot["scope"] == "device:0"
    assert snapshot["allocator_max_kind"] == "allocator_max_since_process_start"
    assert snapshot["device_memory"]["used_bytes"] == 200
    assert snapshot["derived"]["device_used_minus_allocator_reserved_bytes"] == 0


def test_bounded_sampler_core_records_raw_slot_schedule_without_sleep():
    class FakeNpu:
        def __init__(self):
            self.used = 1000

        def mem_get_info(self, device):
            self.used += 100
            return 10_000 - self.used, 10_000

        def memory_allocated(self, device):
            return 100

        def memory_reserved(self, device):
            return 200

        def max_memory_allocated(self, device):
            return 150

        def max_memory_reserved(self, device):
            return 250

    class Clock:
        def __init__(self):
            self.value = 0

        def __call__(self):
            return self.value

    clock = Clock()
    npu = FakeNpu()
    sampler = BoundedMemorySampler(
        SimpleNamespace(npu=npu),
        interval_ms=1.0,
        max_samples=20,
        clock_ns=clock,
        wall_clock_ns=clock,
    )
    sampler._scope_start_ns = 0
    sampler._sample(0)
    clock.value = 1_000_000
    sampler._sample(1)
    summary = summarize_sampling(
        sampler._samples,
        scope_start_ns=0,
        scope_end_ns=2_000_000,
        stop_requested_ns=2_000_000,
        interval_ms=1.0,
        max_samples=20,
        overflow=False,
        thread_alive_at_stop=False,
        device=0,
        reset=sampler._reset,
        errors=sampler._errors,
        observed_at="2026-01-01T00:00:00Z",
    )
    assert summary["status"] == "ok"
    assert summary["measurement_kind"] == "sampled_device_peak"
    assert summary["is_peak"] is True
    assert summary["peak_scope"] == "sampled"
    assert summary["continuous_peak"] is False
    assert summary["absolute_peak_limited"] is True
    assert summary["sample_count"] == 2
    assert summary["expected_slot_count"] == 2
    assert summary["completed_slot_count"] == 2
    assert summary["missing_slot_count"] == 0
    assert summary["samples"][0]["slot_index"] == 0
    assert summary["samples"][1]["slot_index"] == 1
    assert all("wall_ns" in item and "query_start_ns" in item and "query_end_ns" in item for item in summary["samples"])
    assert summary["scope_start_ns"] <= summary["sample_window_ns"][0]
    assert summary["sample_window_ns"][1] <= summary["scope_end_ns"]
    assert summary["temporal_span_coverage_ratio"] == 0.5
    assert summary["cadence_completeness_ratio"] == 1.0
    assert summary["target_coverage_complete"] is False
    snapshot = sampled_peak_to_snapshot(summary)
    assert snapshot["phase"] == "sampled_peak"
    assert snapshot["is_peak"] is True
    assert snapshot["measurement_kind"] == "sampled_device_peak"
    assert snapshot["device_memory"]["used_bytes"] == summary["device_peak"]["used_bytes_max"]
    assert snapshot["continuous_peak"] is False
    assert snapshot["target_coverage_complete"] is False


def test_msmemscope_opt_in_failure_is_explicit():
    with pytest.raises(MemoryEvidenceError) as caught:
        take_msmemscope_snapshot(0, "probe", module=SimpleNamespace())
    assert caught.value.code == "msmemscope_api_missing"
