"""Discriminating Wait-Anchor and PMU provenance regressions.

The public CSV is a two-row exact slice of the actual A3 online-v7
kernel_details.csv.  The GLM45 DB is the existing real repeated-globalTaskId
fixture.  Synthetic KernelEvent cases cover pure interval arithmetic only; they
do not invent hardware metrics.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from decimal import Decimal
from pathlib import Path

import pytest

from mindie_profiler import Window, analyze_profile
from mindie_profiler.anomaly_discovery import KernelEvent, assess_wait_anchor
from mindie_profiler.enrichment_anomaly import wait_anchor_summary
from mindie_profiler.events import Event, Observation, QueryStats, SourceRef
from mindie_profiler.sources_csv import CsvSource, probe_csv
from mindie_profiler.wait_provenance import attach_wait_anchor

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "profiler_core"
A3_FIXTURE = FIXTURE_ROOT / "real_a3_wait_pmu"
A3_CSV = A3_FIXTURE / "kernel_details.csv"
A3_PROVENANCE = A3_FIXTURE / "provenance.json"
GLM45_DB = FIXTURE_ROOT / "real_glm45" / "rank_0" / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_0.db"
GLM45_WINDOW = Window(1763456164890045000, 1763456164895045000, label="real_5ms")
A2_DB = Path(os.environ["MINDIE_TEST_A2_CAPTURE_DB"]) if os.environ.get("MINDIE_TEST_A2_CAPTURE_DB") else None
REAL_A3_CSV = Path(os.environ["MINDIE_TEST_A3_CAPTURE_CSV"]) if os.environ.get("MINDIE_TEST_A3_CAPTURE_CSV") else None


def _event(
    name: str,
    *,
    event_id: str,
    observation_id: str,
    start_us: float,
    duration_us: float,
    wait_us: float | None,
    semantic_block: str = "Norm",
    stream_id: int = 0,
    task_id: str | None = None,
    window_start_ns: int = 0,
    window_end_ns: int = 2_000_000,
) -> KernelEvent:
    start_ns = window_start_ns + int(round(start_us * 1000.0))
    end_ns = start_ns + int(round(duration_us * 1000.0))
    return KernelEvent(
        name=name,
        step=1,
        start_us=start_us,
        duration_us=duration_us,
        wait_us=wait_us,
        task_type="AI_CORE",
        accelerator_core="AI_CORE",
        stream_id=stream_id,
        rank=0,
        event_id=event_id,
        observation_id=observation_id,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        window_label="synthetic-scope",
        source_kind="csv",
        source_sha256="sha256:fixture",
        source_table="kernel_details.csv",
        device_id="0",
        task_id=task_id or event_id,
        semantic_block=semantic_block,
        event_start_ns=start_ns,
        event_end_ns=end_ns,
        wait_ns=None if wait_us is None else int(round(wait_us * 1000.0)),
        wait_start_ns=None if wait_us is None else start_ns - int(round(wait_us * 1000.0)),
        wait_end_ns=None if wait_us is None else start_ns,
    )


def _row(name: str, event_ids: list[str], *, semantic_block: str = "Norm", window=(0, 2_000_000)) -> dict:
    return {
        "rank": 0,
        "rank_id": "rank_0",
        "semantic_block": semantic_block,
        "kernel_name": name,
        "evidence_locator": {
            "event_ids": event_ids,
            "source_sha256": "sha256:fixture",
            "window_bounds_ns": [list(window)],
        },
    }


def test_public_a3_fixture_is_exact_row_subset_with_provenance():
    provenance = json.loads(A3_PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["source"]["sha256"] == "7380711ab7b67df88d58b40be128ba16d649b74b4c430ddfc70d8e68abb82b62"
    assert provenance["selection"]["row_count"] == 2
    assert provenance["selection"]["task_ids"] == ["23606", "23612"]
    assert provenance["transformation"]["values_modified"] is False
    assert provenance["privacy"] == {
        "contains_host": False,
        "contains_credentials": False,
        "contains_private_path": False,
    }
    with A3_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["Task ID"] for row in rows] == ["23606", "23612"]
    assert rows[1]["Wait Time(us)"].strip() == "17290.690000"
    assert rows[1]["Duration(us)"].strip() == "1.620"


def test_public_a3_fixture_wait_group_does_not_become_individual_confirmation():
    report = analyze_profile(A3_CSV, full_range=True, source="csv", top_n=8)
    wait = report["enrichment"]["anomaly_discovery"]["wait_anchor"]
    assert wait["status"] == "available"
    assert wait["measurement_scope"]["wait_usable_event_count"] == 1
    assert len(wait["findings"]) == 1
    finding = wait["findings"][0]
    assert finding["op_name"] == "aclnnInplaceFillScalar_FillAiCore_Fill"
    assert finding["evidence_status"] == "insufficient"
    assert finding["is_false_hotspot_risk"] is False
    assert finding["confirmed_event_ids"] == []
    assert finding["candidate_events"]
    assert finding["candidate_events"][0]["window"]["label"] == "full_range"
    assert finding["candidate_events"][0]["hardware_scope"]["status"] == "not_available"
    assert all(row["wait_anchor_false_hotspot"] is None for row in report["top_hotspot_kernels"])
    assert all(row["wait_anchor_false_hotspot"] is not False for row in report["top_hotspot_kernels"])
    assert all(
        row["wait_anchor_assessment"] in {"not_evaluated", "unverified"}
        for row in report["top_hotspot_kernels"]
    )


def test_public_a3_fixture_pmu_units_and_derived_are_full_event_non_causal():
    report = analyze_profile(A3_CSV, full_range=True, source="csv", top_n=8)
    pmu = report["enrichment"]["pmu_observations"]
    assert pmu["status"] == "observed"
    assert pmu["derived_attribution"] == "rule_candidate_not_root_cause"
    assert pmu["metric_units"]["aiv_total_time"] == "ns"
    assert pmu["metric_units"]["cube_utilization_pct"] == "percent"
    fill = next(item for item in pmu["bound_events"] if "Fill" in item["kernel_name"])
    assert fill["metric_units"]["aiv_total_time"] == "ns"
    assert fill["metric_units"]["aiv_vec_ratio"] == "ratio"
    assert fill["derived"]["task_role"] == "aiv"
    assert fill["derived"]["scope"] == {
        "kind": "full_event",
        "source": "kernel_details.csv",
        "complete_in_window": True,
        "usable_for_clipped_window": True,
        "derived_for_clipped_window": True,
    }
    assert fill["derived"]["vector_rule"]["status"] == "available"
    assert fill["derived"]["vector_rule"]["candidate_bound"] is None
    assert fill["derived"]["vector_rule"]["causal"] is False
    assert fill["derived"]["vector_rule"]["source_commit"] == "639eb8702c1884813dfab59ddb371f7838caab6d"
    cube = fill["derived"]["cube_utilization_pct"]
    assert cube["value"] == 0.0
    assert cube["unit"] == "percent"
    assert cube["source"] == "kernel_details.csv:cube_utilization(%)"
    assert cube["not_equivalent_to"] == "aic_mac_ratio"
    assert fill["derived"]["causal"] is False
    consistency = {item["pipe"]: item for item in fill["derived"]["aiv"]["ratio_time_consistency"]}
    assert consistency["vec"]["residual_ns"] == pytest.approx(-0.59)
    assert consistency["vec"]["status"] == "compatible_with_export_precision"
    assert consistency["vec"]["export_precision_envelope_ns"] > abs(consistency["vec"]["residual_ns"])
    assert fill["derived"]["aiv"]["ratio_time_all_consistent"] is None


def test_public_a3_prefers_accelerator_core_over_operation_type():
    with A3_CSV.open(encoding="utf-8-sig", newline="") as handle:
        row = next(item for item in csv.DictReader(handle) if item["Task ID"] == "23612")
    start_ns = int(Decimal(row["Start Time(us)"].strip()) * 1000)
    end_ns = start_ns + int(Decimal(row["Duration(us)"]) * 1000)
    source = CsvSource(probe_csv(A3_CSV), rank_id="rank_0", rank_index=0)
    events, _ = source.read_windows([Window(start_ns, end_ns)])[0]
    assert len(events) == 1
    assert events[0].core == row["Accelerator Core"] == "AI_VECTOR_CORE"
    assert events[0].op_type == row["Type"] == "Fill"


def test_public_a3_fixture_clipped_pmu_keeps_full_counters_but_marks_derived_scope():
    with A3_CSV.open(encoding="utf-8-sig", newline="") as handle:
        fill_row = next(row for row in csv.DictReader(handle) if row["Task ID"] == "23612")
    start_ns = int(Decimal(fill_row["Start Time(us)"].strip()) * 1000)
    report = analyze_profile(
        A3_CSV,
        windows=[Window(start_ns + 500, start_ns + 2_000, label="clipped_fill")],
        source="csv",
        top_n=4,
    )
    wait = report["enrichment"]["anomaly_discovery"]["wait_anchor"]
    assert wait["status"] == "unverified"
    assert wait["measurement_scope"]["wait_clipped_event_count"] >= 1
    pmu = report["enrichment"]["pmu_observations"]
    clipped = next(item for item in pmu["bound_events"] if "Fill" in item["kernel_name"])
    assert clipped["complete_in_window"] is False
    assert clipped["counts_are_full_event"] is True
    assert clipped["metric_count"] == 25
    assert clipped["derived"]["scope"]["kind"] == "full_event"
    assert clipped["derived"]["scope"]["usable_for_clipped_window"] is False
    assert clipped["derived"]["scope"]["derived_for_clipped_window"] is False
    assert clipped["derived"]["aiv"]["pipe_time_fractions"]


def test_cross_window_relative_timelines_are_not_pooled_for_overlap():
    anchor = _event(
        "anchor",
        event_id="event:anchor",
        observation_id="observation:one",
        start_us=1_000.0,
        duration_us=2.0,
        wait_us=1_000.0,
    )
    other = _event(
        "other",
        event_id="event:other",
        observation_id="observation:two",
        start_us=0.0,
        duration_us=1_000.0,
        wait_us=0.0,
    )
    report = assess_wait_anchor([anchor, other])
    finding = report.findings[0]
    assert finding["evidence_status"] == "insufficient"
    assert finding["is_false_hotspot_risk"] is False
    assert finding["confirmed_event_ids"] == []
    assert finding["overlap_pairs"] == []


def test_exact_attachment_matches_only_event_window_source_and_semantic_scope():
    anchor = _event(
        "anchor",
        event_id="event:anchor",
        observation_id="observation:one",
        start_us=1_000.0,
        duration_us=2.0,
        wait_us=1_000.0,
        semantic_block="Norm",
    )
    overlap = _event(
        "upstream",
        event_id="event:upstream",
        observation_id="observation:one",
        start_us=0.0,
        duration_us=1_000.0,
        wait_us=0.0,
        semantic_block="Norm",
    )
    finding = assess_wait_anchor([anchor, overlap]).as_c7_list()[0]
    assert finding["is_false_hotspot_risk"] is True
    rows = [
        _row("anchor", ["event:anchor"]),
        _row("anchor", ["event:upstream"]),
        _row("anchor", []),
        _row("anchor", ["event:anchor"], semantic_block="Copy_Layout"),
    ]
    attach_wait_anchor(rows, {"status": "available", "findings": [finding], "reason": None})
    assert rows[0]["wait_anchor_false_hotspot"] is True
    assert rows[0]["wait_anchor_assessment"] == "assessed"
    assert rows[0]["wait_anchor_scope"]["event_scope"]["event_id"] == "event:anchor"
    assert rows[0]["wait_anchor_scope"]["overlap_scope"]["overlap"]["event_id"] == "event:upstream"
    for sibling in rows[1:]:
        assert sibling["wait_anchor_false_hotspot"] is None
        assert sibling["wait_anchor_assessment"] == "not_evaluated"
        assert sibling["wait_anchor_reason"]


def test_event_identity_includes_interval_stream_and_locator_not_just_task_id():
    base = Event(
        rank_id="rank_0",
        rank_index=0,
        name="op",
        op_type="Op",
        task_type="AI_CORE",
        core="AI_CORE",
        stream_id="1",
        task_id="7",
        device_id="0",
        start_ns=100,
        end_ns=110,
        source_kind="db",
        table="COMPUTE_TASK_INFO",
        locator={"task_global_id": 4, "task_id": "7", "stream_id": "1", "device_id": "0"},
    )
    same = copy.deepcopy(base)
    other_stream = copy.deepcopy(base)
    other_stream.stream_id = "2"
    other_time = copy.deepcopy(base)
    other_time.start_ns += 1
    other_time.end_ns += 1
    assert base.event_id == same.event_id
    assert base.event_id != other_stream.event_id
    assert base.event_id != other_time.event_id
    assert base.identity()["payload"]["start_ns"] == 100
    assert base.identity()["payload"]["locator"]["task_global_id"] == 4


def test_actual_a3_group_findings_never_attach_without_exact_event_scope():
    if REAL_A3_CSV is None:
        pytest.skip("set MINDIE_TEST_A3_CAPTURE_CSV for full-capture acceptance")
    with REAL_A3_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    starts = [Decimal(row["Start Time(us)"].strip()) for row in rows]
    end_ns = int(max(start + Decimal(row["Duration(us)"]) for start, row in zip(starts, rows)) * 1000)
    report = analyze_profile(
        REAL_A3_CSV,
        windows=[Window(int(min(starts) * 1000), end_ns, label="a3_full")],
        source="csv",
        top_n=200,
    )
    wait = report["enrichment"]["anomaly_discovery"]["wait_anchor"]
    assert wait["status"] == "available"
    assert wait["findings"]
    for finding in wait["findings"]:
        if finding["is_false_hotspot_risk"]:
            assert finding["confirmed_event_ids"]
            assert finding["overlap_pairs"]
            for pair in finding["overlap_pairs"]:
                assert pair["candidate"]["window"] == pair["overlap"]["window"]
                assert pair["candidate"]["observation_id"] == pair["overlap"]["observation_id"]
    for row in report["top_hotspot_kernels"]:
        assert row["wait_anchor_false_hotspot"] is not False
        if row["wait_anchor_false_hotspot"] is True:
            scope = row["wait_anchor_scope"]
            assert scope
            assert scope["event_scope"]["event_id"] in row["evidence_locator"]["event_ids"]
            assert scope["overlap_scope"]
            assert scope["event_scope"]["window"] == scope["overlap_scope"]["candidate"]["window"]


def test_glm45_repeated_groups_keep_exact_candidate_scope_and_absent_hardware():
    if not GLM45_DB.is_file():
        pytest.skip("GLM45 fixture is unavailable")
    report = analyze_profile(GLM45_DB, windows=[GLM45_WINDOW], source="db", top_n=50)
    wait = report["enrichment"]["anomaly_discovery"]["wait_anchor"]
    assert wait["status"] == "available"
    assert wait["measurement_scope"]["wait_ambiguous_event_count"] == 35
    for finding in wait["findings"]:
        assert finding["is_false_hotspot_risk"] is False
        assert finding["confirmed_event_ids"] == []
        assert finding["candidate_events"]
        event_ids = [item["event_id"] for item in finding["candidate_events"]]
        assert len(event_ids) == len(set(event_ids))
        assert finding["candidate_events"][0]["hardware_scope"]["status"] == "not_available"
    assert all(row["wait_anchor_false_hotspot"] is not False for row in report["top_hotspot_kernels"])
    pmu = report["enrichment"]["pmu_observations"]
    assert pmu["status"] == "partial"
    assert pmu["ambiguous_groups"]
    assert pmu["ambiguous_groups"][0]["metric_units"]
    assert "per_metric" in pmu["ambiguous_groups"][0]
    assert pmu["bound_events"]
    assert pmu["bound_events"][0]["metric_units"]
    assert pmu["bound_events"][0]["derived"]["attribution"] == "rule_candidate_not_root_cause"

def test_actual_a2_pmu_derivation_keeps_fractions_and_official_rules_distinct():
    if A2_DB is None:
        pytest.skip("set MINDIE_TEST_A2_CAPTURE_DB for full-capture acceptance")
    steps = json.loads(
        (A2_DB.parents[3] / "steps.json").read_text(encoding="utf-8")
    )["steps"]
    report = analyze_profile(
        A2_DB,
        windows=[
            Window(item["start_ns"], item["end_ns"], step=item["step"], label=f"step{item['step']}")
            for item in steps
        ],
        source="db",
        top_n=8,
    )
    pmu = report["enrichment"]["pmu_observations"]
    matmul = next(item for item in pmu["bound_events"] if item["kernel_name"].startswith("aclnnAddmm"))
    relu = next(item for item in pmu["bound_events"] if item["kernel_name"].startswith("aclnnRelu"))
    assert pmu["metric_units"]["aic_total_time"] == "ns"
    assert pmu["metric_units"]["cube_utilization_pct"] == "percent"
    assert matmul["metric_units"]["aic_total_time"] == "ns"
    assert matmul["metric_units"]["aic_mac_ratio"] == "ratio"
    assert matmul["derived"]["aic"]["ratio_time_all_consistent"] is True
    assert matmul["derived"]["aic"]["fractions_overlap"] is True
    assert matmul["derived"]["aic"]["dominant_pipe_by_time"]["pipe"] == "scalar"
    assert matmul["derived"]["cube_rule"]["matched"] is False
    assert matmul["derived"]["cube_rule"]["candidate_bound"] is None
    assert matmul["derived"]["cube_utilization_pct"]["status"] == "absent"
    assert relu["derived"]["aiv"]["ratio_time_all_consistent"] is True
    assert relu["derived"]["vector_rule"]["matched"] is False
    assert relu["derived"]["vector_rule"]["candidate_bound"] is None
    assert relu["derived"]["aiv"]["dominant_pipe_by_time"]["pipe"] == "scalar"
    assert relu["derived"]["causal"] is False

def test_wait_summary_does_not_pool_relative_times_from_two_windows():
    source = SourceRef(
        rank_id="rank_0",
        rank_index=0,
        kind="csv",
        path="fixture/kernel_details.csv",
        basename="kernel_details.csv",
        sha256="sha256:fixture",
        size_bytes=1,
        schema_version="1.0",
        schema_variant="csv",
    )

    def observation(window: Window, event: Event) -> Observation:
        return Observation(
            rank_id="rank_0",
            rank_index=0,
            window=window,
            accounting=None,
            events=[event],
            source=source,
            stats=QueryStats(rank_id="rank_0", window_label=window.label or "window"),
        )

    def event(name: str, task_id: str, start_ns: int, end_ns: int, wait_ns: int) -> Event:
        return Event(
            rank_id="rank_0",
            rank_index=0,
            name=name,
            op_type=name,
            task_type="AI_CORE",
            core="AI_CORE",
            stream_id="0",
            task_id=task_id,
            device_id="0",
            start_ns=start_ns,
            end_ns=end_ns,
            source_kind="csv",
            table="kernel_details.csv",
            wait_ns=wait_ns,
            wait_status="bound",
            wait_source="Wait Time(us)",
            wait_provenance={"status": "bound", "raw_wait_ns": wait_ns},
            locator={"row_index": int(task_id), "task_id": task_id, "stream_id": "0"},
        )

    # The two relative timelines would overlap if pooled: anchor's wait is
    # 0..1000 us and the other event is 0..1000 us in its own window.
    first = observation(Window(0, 2_000_000, label="first"), event("anchor", "1", 1_000_000, 1_002_000, 1_000_000))
    second = observation(Window(100_000_000, 102_000_000, label="second"), event("upstream", "2", 100_000_000, 101_000_000, 0))
    summary = wait_anchor_summary([first, second])
    assert summary["status"] == "available"
    assert summary["findings"]
    finding = summary["findings"][0]
    assert finding["is_false_hotspot_risk"] is False
    assert finding["confirmed_event_ids"] == []
    assert finding["overlap_pairs"] == []
    assert finding["candidate_events"][0]["window"]["label"] == "first"
