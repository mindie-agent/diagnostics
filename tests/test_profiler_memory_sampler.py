"""Pure sampling-contract tests for bounded device-memory evidence.

These tests use synthetic monotonic timestamps, not hardware sleeps or NPU
launches.  They pin post-stop exclusion, missing cadence, duplicate slots,
failed getters, and the two-endpoints coverage failure.
"""

from __future__ import annotations

from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from mindie_profiler.memory_evidence import manifest_from_snapshots
from mindie_profiler.memory_profile import attribute_memory
from mindie_profiler.memory_sampler import (
    BoundedMemorySampler,
    sampled_peak_to_snapshot,
    summarize_sampling,
)

MS = 1_000_000


def _obs(start_ms, end_ms, used=1000, *, slot_index=None, status="ok", alloc_ok=4, alloc_attempted=4):
    return {
        "slot_index": int(start_ms // 10) if slot_index is None else slot_index,
        "query_start_ns": int(start_ms * MS),
        "query_end_ns": int(end_ms * MS),
        "wall_ns": int(start_ms * MS),
        "status": status,
        "used_bytes": used if status == "ok" else None,
        "allocated_bytes": 10,
        "reserved_bytes": 20,
        "max_allocated_bytes": 30,
        "max_reserved_bytes": 40,
        "allocator_ok_count": alloc_ok,
        "allocator_attempted_count": alloc_attempted,
    }


def _summary(samples, *, start_ms=0, end_ms=100, interval_ms=10):
    return summarize_sampling(
        samples,
        scope_start_ns=start_ms * MS,
        scope_end_ns=end_ms * MS,
        stop_requested_ns=end_ms * MS,
        interval_ms=interval_ms,
        max_samples=512,
        overflow=False,
        thread_alive_at_stop=False,
        device=0,
        reset={"attempted": True, "ok": True},
        errors=[],
        observed_at="2026-01-01T00:00:00Z",
    )


def test_post_stop_samples_are_excluded_from_peak_and_scope():
    summary = _summary([
        _obs(0, 1, 100),
        _obs(50, 51, 500),
        _obs(100, 101, 900),   # ends after scope_end
        _obs(120, 121, 9999),  # starts after stop request
    ])
    assert summary["scope_start_ns"] == 0
    assert summary["scope_end_ns"] == 100 * MS
    assert summary["out_of_scope_sample_count"] == 2
    assert summary["device_peak"]["used_bytes_max"] == 500
    assert summary["sample_window_ns"][1] == 51 * MS
    assert summary["temporal_span_coverage_ratio"] == 0.51
    assert summary["expected_slot_count"] == 10
    assert summary["completed_slot_count"] == 2
    assert summary["missing_slot_count"] == 8
    assert summary["cadence_completeness_ratio"] == 0.2
    assert summary["coverage_ratio"] == 0.2
    assert summary["target_coverage_complete"] is False
    assert summary["is_peak"] is True  # valid sampled max, but partial scope


def test_two_endpoints_over_long_gap_do_not_look_complete():
    summary = _summary([_obs(0, 1, 100), _obs(99, 100, 200)])
    assert summary["temporal_span_coverage_ratio"] == 1.0
    assert summary["expected_slot_count"] == 10
    assert summary["completed_slot_count"] == 2
    assert summary["missing_slot_count"] == 8
    assert summary["cadence_completeness_ratio"] == 0.2
    assert summary["coverage_ratio"] == 0.2
    assert summary["target_coverage_complete"] is False
    assert summary["coverage_status"] == "partial"
    assert summary["is_peak"] is True


def test_declared_slot_is_not_reassigned_from_late_query_start():
    summary = _summary([
        _obs(15, 16, 900, slot_index=0),   # declared slot 0, query starts 15 ms late
        _obs(20, 21, 100, slot_index=2),
    ], end_ms=30, interval_ms=10)
    assert summary["expected_slot_count"] == 3
    assert summary["completed_slot_count"] == 2
    assert summary["missing_slot_indices"] == [1]
    assert summary["late_slot_indices"] == [0]
    assert summary["samples"][0]["slot_index"] == 0
    assert summary["samples"][0]["slot_index_status"] == "declared"
    assert summary["samples"][0]["slot_target_ns"] == 0
    assert summary["samples"][0]["late_ns"] == 15 * MS
    assert summary["target_coverage_complete"] is False
    assert summary["is_peak"] is True


def test_missing_slot_index_stays_partial_and_visible():
    missing = _obs(10, 11, 200)
    missing.pop("slot_index")
    summary = _summary([
        _obs(0, 1, 100, slot_index=0),
        missing,
    ], end_ms=20, interval_ms=10)
    assert summary["expected_slot_count"] == 2
    assert summary["completed_slot_count"] == 1
    assert summary["missing_slot_index_sample_count"] == 1
    assert summary["missing_slot_indices"] == [1]
    assert summary["samples"][1]["slot_index_status"] == "missing_slot_index"
    assert summary["samples"][1]["slot_target_ns"] is None
    assert summary["target_coverage_complete"] is False
    assert summary["is_peak"] is True


def test_outside_and_duplicate_indices_are_reported():
    summary = _summary([
        _obs(0, 1, 100, slot_index=0),
        _obs(1, 2, 200, slot_index=0),
        _obs(99, 100, 900, slot_index=99),
    ])
    assert summary["expected_slot_count"] == 10
    assert summary["completed_slot_count"] == 1
    assert summary["duplicate_slot_sample_count"] == 1
    assert summary["duplicate_slot_indices"] == [0]
    assert summary["outside_slot_sample_count"] == 1
    assert summary["outside_slot_indices"] == [99]
    assert summary["missing_slot_count"] == 9
    assert summary["device_peak"]["used_bytes_max"] == 900
    assert summary["target_coverage_complete"] is False


def test_duplicate_calls_in_one_slot_do_not_inflate_cadence():
    summary = _summary([
        _obs(0, 1, 100),
        _obs(1, 2, 900),
        _obs(20, 21, 300),
        _obs(90, 91, 400),
    ])
    assert summary["expected_slot_count"] == 10
    assert summary["completed_slot_count"] == 3  # slots 0, 2, 9
    assert summary["duplicate_slot_sample_count"] == 1
    assert summary["cadence_completeness_ratio"] == 0.3
    assert summary["device_peak"]["used_bytes_max"] == 900
    assert summary["target_coverage_complete"] is False


def test_failed_allocator_getter_is_visible_and_blocks_complete_target():
    summary = _summary([
        _obs(0, 1, 100, alloc_ok=3, alloc_attempted=4),
        _obs(10, 11, 100, alloc_ok=4, alloc_attempted=4),
        _obs(20, 21, 100, alloc_ok=4, alloc_attempted=4),
        _obs(30, 31, 100, alloc_ok=4, alloc_attempted=4),
        _obs(40, 41, 100, alloc_ok=4, alloc_attempted=4),
        _obs(50, 51, 100, alloc_ok=4, alloc_attempted=4),
        _obs(60, 61, 100, alloc_ok=4, alloc_attempted=4),
        _obs(70, 71, 100, alloc_ok=4, alloc_attempted=4),
        _obs(80, 81, 100, alloc_ok=4, alloc_attempted=4),
        _obs(90, 91, 100, alloc_ok=4, alloc_attempted=4),
    ])
    assert summary["temporal_span_coverage_ratio"] == 0.91
    assert summary["cadence_completeness_ratio"] == 1.0
    assert summary["optional_getter_completeness_ratio"] == pytest.approx(39 / 40, rel=1e-6)
    assert summary["coverage_ratio"] == pytest.approx(0.91, rel=1e-6)
    assert summary["allocator_peak"]["failed_allocator_query_count"] == 1
    assert summary["target_coverage_complete"] is False


def test_failed_device_sample_is_recorded_as_failed_slot():
    summary = _summary([
        _obs(0, 1, status="error"),
        _obs(10, 11, 100),
    ])
    assert summary["failed_sample_count"] == 1
    assert summary["completed_slot_count"] == 1
    assert summary["missing_slot_count"] == 9
    assert summary["device_sample_completeness_ratio"] == 0.5
    assert summary["is_peak"] is False


def test_cap_truncation_keeps_sampled_max_but_blocks_complete_target():
    summary = summarize_sampling(
        [_obs(0, 1, 100), _obs(10, 11, 200), _obs(20, 21, 300)],
        scope_start_ns=0,
        scope_end_ns=100 * MS,
        stop_requested_ns=100 * MS,
        interval_ms=10,
        max_samples=3,
        overflow=True,
        thread_alive_at_stop=False,
        device=0,
        reset={"attempted": True, "ok": True},
        errors=[],
        observed_at="2026-01-01T00:00:00Z",
    )
    assert summary["is_peak"] is True
    assert summary["samples_truncated"] is True
    assert summary["device_peak"]["used_bytes_max"] == 300
    assert summary["target_coverage_complete"] is False
    assert summary["coverage_status"] == "partial"


def test_snapshot_preserves_cadence_contract_fields():
    summary = _summary([_obs(0, 1, 100), _obs(10, 11, 200)])
    snapshot = sampled_peak_to_snapshot(summary)
    assert snapshot["phase"] == "sampled_peak"
    assert snapshot["is_peak"] is True
    assert snapshot["coverage_ratio"] == summary["coverage_ratio"]
    assert snapshot["temporal_span_coverage_ratio"] == summary["temporal_span_coverage_ratio"]
    assert snapshot["cadence_completeness_ratio"] == summary["cadence_completeness_ratio"]
    assert snapshot["target_coverage_complete"] is False
    assert snapshot["cadence_ms"]["missing_slots"] == summary["missing_slot_count"]
    assert snapshot["cadence_ms"]["missing_slot_indices"] == summary["missing_slot_indices"]
    assert snapshot["slot_assignment"] == "raw_sample.slot_index"


def test_complete_sampled_summary_survives_manifest_and_allows_met_target():
    samples = [_obs(float(start), float(start + 1), 1_200_000_000) for start in range(0, 91, 10)]
    summary = _summary(samples, start_ms=0, end_ms=91, interval_ms=10)
    assert summary["target_coverage_complete"] is True
    assert summary["cadence_completeness_ratio"] == 1.0
    snapshot = sampled_peak_to_snapshot(summary)
    manifest = manifest_from_snapshots(
        {
            "baseline": {
                "device": 0,
                "device_memory": {"used_bytes": 1_000_000_000},
                "source": "torch_npu.npu.memory_allocated/.../mem_get_info",
                "source_kind": "torch_npu_runtime_api",
                "measurement_kind": "point_observation",
                "is_peak": False,
            },
            "sampled_peak": snapshot,
        },
        msprof_components_by_device={"0": {"RUNTIME": {"bytes": 10_000_000}}},
        weight_manifest={"status": "ok", "tp": 1, "per_device_bytes": 50_000_000},
        kv_cache_applicable=False,
    )
    result = attribute_memory(manifest, target_phase="sampled_peak")
    device = result["devices"][0]
    assert device["target_is_peak"] is True
    assert device["target_coverage_complete"] is True
    assert device["target"]["coverage_complete"] is True
    assert device["residual"]["coverage"] == "complete"
    assert device["residual"]["target_met"] is True


def test_allocator_query_does_not_retry_with_different_signature():
    class FakeNpu:
        def memory_allocated(self, device):
            raise TypeError("device argument required; no zero-arg retry allowed")

    sampler = BoundedMemorySampler(
        SimpleNamespace(npu=FakeNpu()),
        clock_ns=lambda: 0,
        wall_clock_ns=lambda: 0,
    )
    value, attempted, ok = sampler._call_optional(FakeNpu(), "memory_allocated")
    assert value is None
    assert attempted is True
    assert ok is False
    assert sampler._errors[0]["code"] == "allocator_api_failed"


def test_selected_device_peak_keeps_its_allocator_query_through_manifest():
    early = _obs(0, 1, 200)
    early.update(allocated_bytes=20, reserved_bytes=40, wall_ns=1000, wall_end_ns=1010)
    later = _obs(10, 11, 100)
    later.update(allocated_bytes=90, reserved_bytes=190, wall_ns=2000, wall_end_ns=2010)
    summary = _summary([early, later])
    snapshot = sampled_peak_to_snapshot(summary)
    manifest = manifest_from_snapshots({"sampled_peak": snapshot})
    selected = manifest["sampled_peak_hbm"]["0"]
    assert selected["peak_sample"]["allocated_bytes"] == 20
    assert selected["peak_sample"]["reserved_bytes"] == 40
    assert selected["peak_sample"]["wall_end_ns"] == 1010
    assert selected["sample_window_clock"] == "time.monotonic_ns"


def test_all_observed_slots_are_complete_despite_one_late_query():
    rows = [_obs(index * 10, index * 10 + 1, 100, slot_index=index) for index in range(100)]
    rows[43] = _obs(437, 438, 100, slot_index=43)
    summary = _summary(rows, end_ms=1000)
    assert summary["target_coverage_complete"] is True
    assert summary["cadence_ms"]["punctuality_complete"] is False
    assert summary["cadence_ms"]["late_slot_indices"] == [43]
    assert summary["continuous_peak"] is False
    assert _summary(rows + [dict(rows[0])], end_ms=1000)["target_coverage_complete"] is False
    assert _summary(rows[:-1], end_ms=1000)["target_coverage_complete"] is False


def test_actual_online_peak_subtracts_baseline_allocator_before_adding_inactive_reservation():
    from mindie_profiler.memory_peak import peak_wall_interval
    fixture = json.loads((Path(__file__).parent / "fixtures/profiler_core/memory_online_a3/v6_peak_query.json").read_text())
    summary = {**fixture["summary"], "peak_sample": fixture["selected_peak_sample"]}
    snapshot = sampled_peak_to_snapshot(summary)
    manifest = manifest_from_snapshots({
        "baseline": fixture["baseline"]["snapshot"], "sampled_peak": snapshot},
        vllm_info={"weights_gb": 0.0001, "kv_cache_current_gib": 12.22},
        msprof_components_by_device={"0": {"RUNTIME": {"bytes": 87_941_120}}},
        kv_cache_applicable=True)
    device = attribute_memory(manifest, target_phase="sampled_peak", residual_target_bytes=200_000_000)["devices"][0]
    components = {item["key"]: item for item in device["components"]}
    assert components["baseline_observed"]["bytes"] == 433_963_008 - 2_097_152
    assert components["allocator_inactive_reserved"]["bytes"] == 26_321_354_752 - 13_120_824_832
    assert device["residual"]["bytes"] == pytest.approx(158_770_302.72)
    assert device["residual"]["target_met"] is False  # original missing slot remains partial
    assert peak_wall_interval(manifest["sampled_peak_hbm"]["0"]) is None  # v6 did not collect wall end
