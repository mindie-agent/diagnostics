"""Tier-0 contract tests for the P0 metric/identity corrections.

All cases are pure in-memory or tiny SQLite fixtures and run in well under 5s:
metric naming, engine/bottleneck separation, variant grouping, clipped-event
distribution hygiene, default identity, multi-DB refusal, and request wall time.
"""

from __future__ import annotations

import hashlib

import pytest

from mindie_profiler import ProfilerUsageError, Window, analyze_profile
from mindie_profiler.accounting import account_window, hotspot_aggregates
from mindie_profiler.classify import classify_event
from mindie_profiler.events import Event
from mindie_profiler.inventory import inventory_profile
from mindie_profiler.segment import discover_rank_sources
from tests.test_profiler_core import RANK0_ID, STEP4, build_db, build_run


def _event(start, end, name="kernel", *, shapes="", dtypes="", formats="", block="Unclassified"):
    return Event(
        rank_id="rank_0",
        rank_index=0,
        name=name,
        op_type="Add",
        task_type="AI_CORE",
        core="AI_CORE",
        stream_id="1",
        task_id="1",
        device_id="0",
        start_ns=start,
        end_ns=end,
        source_kind="test",
        table="COMPUTE_TASK_INFO",
        block=block,
        input_shapes=shapes,
        input_dtypes=dtypes,
        input_formats=formats,
    )


def test_request_window_span_and_device_uncovered_are_distinct():
    window = Window(0, 1000, label="w")
    events = [_event(100, 300), _event(800, 1200)]
    accounting = account_window(events, window, request_wall_ns=5_000)

    assert accounting.window_span_ns == 1000
    assert accounting.request_wall_ns == 5_000
    assert accounting.observed_event_span_ns == 900
    assert accounting.busy_union_ns == 400
    assert accounting.device_activity_sum_ns == 400
    assert accounting.device_uncovered_ns == 500
    assert accounting.host_dispatch_gap_ns is None
    assert accounting.complete_event_count == 1
    assert accounting.clipped_event_count == 1
    # Deprecated aliases preserve the same measured values.
    assert accounting.wall_ns == accounting.observed_event_span_ns
    assert accounting.host_gap_ns == accounting.device_uncovered_ns


def test_engine_is_not_a_bottleneck_verdict():
    cube = classify_event(name="aclnnMatMul", core="AI_CORE")
    vector = classify_event(name="aclnnRmsNorm", core="AI_VECTOR_CORE")
    mix = classify_event(name="fused", core="MIX_AIC")
    comm = classify_event(name="hcom_allReduce", op_type="hcom_allReduce_", table="COMMUNICATION_OP")
    host = classify_event(name="ProfilerStep#1", table="PYTORCH_API")

    assert cube.execution_engine == "aic"
    assert vector.execution_engine == "aiv"
    assert mix.execution_engine == "mix"
    assert comm.execution_engine == "hccl"
    assert host.execution_engine == "host"
    assert cube.bound_family == vector.bound_family == "unknown"


def test_hotspots_group_by_variant_not_first_shape(tmp_path):
    events = [
        {"start": 0, "end": 10_000, "name": "GroupedMatmul", "op_type": "GroupedMatmul",
         "core": "AI_CORE", "task_type": "AI_CORE", "stream": 1, "task": 1, "shapes": '"8,16"',
         "emit_comm": False},
        {"start": 10_000, "end": 20_000, "name": "GroupedMatmul", "op_type": "GroupedMatmul",
         "core": "AI_CORE", "task_type": "AI_CORE", "stream": 1, "task": 2, "shapes": '"16,32"',
         "emit_comm": False},
    ]
    source = build_db(tmp_path / "rank_0" / "ascend_pytorch_profiler_0.db", events)
    window = [Window(0, 20_000)]

    variants = analyze_profile(source.parent, windows=window)
    rows = [row for row in variants["top_hotspot_kernels"] if row["kernel_name"] == "GroupedMatmul"]
    assert len(rows) == 2
    assert {row["input_shapes"] for row in rows} == {'"8,16"', '"16,32"'}
    assert all(row["variant_count"] == 1 for row in rows)

    by_name = analyze_profile(source.parent, windows=window, group_by="kernel")
    row = next(row for row in by_name["top_hotspot_kernels"] if row["kernel_name"] == "GroupedMatmul")
    assert row["variant_count"] == 2
    assert row["input_shapes"] is None
    assert len(row["variant_inputs"]) == 2


def test_clipped_events_stay_in_sums_but_not_in_per_call_distribution():
    window = Window(5000, 15_000)
    clipped = _event(0, 10_000, "same_kernel", shapes='"1"')
    complete = _event(10_000, 12_000, "same_kernel", shapes='"1"')
    accounting = account_window([clipped, complete], window)

    assert accounting.event_count == 2
    assert accounting.complete_event_count == 1
    assert accounting.clipped_event_count == 1
    assert accounting.busy_union_ns == 7000
    assert accounting.device_activity_sum_ns == 7000

    (row,) = hotspot_aggregates([clipped, complete], window)
    assert row["duration_ns"] == 7000
    assert row["count"] == 2
    assert row["clipped_event_count"] == 1
    assert row["durations"] == [2000]


def test_default_identity_skips_sha_and_multiple_dbs_are_not_mtime_selected(tmp_path):
    root = build_run(tmp_path)
    db_path = root / RANK0_ID / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_0.db"

    report = analyze_profile(root, ranks=[RANK0_ID], windows=[STEP4])
    source = report["source_identity"]["sources"][0]
    assert source["sha256"] is None
    assert source["size_bytes"] == db_path.stat().st_size
    assert source["mtime_ns"] == db_path.stat().st_mtime_ns
    assert report["source_identity"]["source_fingerprints"][RANK0_ID]["mtime_ns"] == db_path.stat().st_mtime_ns
    assert any(item["code"] == "sha256_disabled" for item in report["limitations"])

    hashed = analyze_profile(root, ranks=[RANK0_ID], windows=[STEP4], hash_sources=True)
    assert hashed["source_identity"]["db_files_sha256"][RANK0_ID] == hashlib.sha256(
        db_path.read_bytes()
    ).hexdigest()

    second = db_path.with_name("ascend_pytorch_profiler_1.db")
    second.write_bytes(db_path.read_bytes())
    with pytest.raises(ProfilerUsageError, match="candidates"):
        discover_rank_sources(root)
    inventory = inventory_profile(root)
    rank = next(item for item in inventory["ranks"] if item["rank_id"] == RANK0_ID)
    assert len(rank["db_candidates"]) == 2
    assert any(item["code"] == "multiple_db_candidates" for item in rank["limitations"])


def test_request_wall_is_measured_not_derived(tmp_path):
    root = build_run(tmp_path)
    report = analyze_profile(root, ranks=[RANK0_ID], windows=[STEP4], request_wall_ms=12.5)
    row = next(item for item in report["window_accounting"] if item["step"] == 4)
    assert row["request_wall_ms"] == 12.5
    assert row["window_duration_ms"] == 0.06
    assert row["observed_event_span_ms"] == 0.045
    assert row["device_uncovered_ms"] == 0.005
    assert row["host_dispatch_gap_ms"] is None
