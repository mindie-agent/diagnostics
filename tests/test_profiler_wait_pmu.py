"""Real-slice and measured wait/PMU integration tests.

The DB/CSV fixtures are read-only slices of the real GLM45 capture (source
SHA256 df4b5df...), not fabricated hardware results.  The tests cover the
previous parser failure (STRING_IDS emitted as integers), exact DB/CSV wait
units, PMU non-multiplication under repeated globalTaskIds, clipped-event
scope, and AICPU counter honesty.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from mindie_profiler import Window, analyze_profile
from mindie_profiler.sources_csv import CsvSource, probe_csv
from mindie_profiler.sources_db import DbSource, probe_db

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "profiler_core" / "real_glm45"
DB_PATH = FIXTURE / "rank_0" / "ASCEND_PROFILER_OUTPUT" / "ascend_pytorch_profiler_0.db"
CSV_PATH = FIXTURE / "kernel_details_slice.csv"
PROVENANCE = FIXTURE / "slice_provenance.json"
START_NS = 1763456164890045000
END_NS = 1763456164895045000
WINDOW = Window(START_NS, END_NS, step=None, label="real_5ms")
CLIP_WINDOW = Window(1763456164891087000, 1763456164891089000, step=None, label="clipped_1063")


def _provenance() -> dict:
    return json.loads(PROVENANCE.read_text(encoding="utf-8"))


def _events(window=WINDOW):
    schema = probe_db(DB_PATH)
    source = DbSource(schema)
    return schema, source, *source.iter_window(window, max_rows=1_000_000)


def _by_global(events, global_id):
    return [event for event in events if event.locator.get("task_global_id") == global_id]


def test_real_slice_provenance_preserves_full_capture_ambiguity():
    data = _provenance()
    assert data["source"]["db_sha256"] == "df4b5dfbfca3bda520d7d5cb92010ba70692b871937ecc5c1bcad3fc1f790373"
    assert data["window_ns"] == [START_NS, END_NS]
    assert hashlib.sha256(DB_PATH.read_bytes()).hexdigest() == data["slice"]["db_sha256"]
    assert data["selection"]["selected_task_rows"] == 102
    assert data["selection"]["selected_distinct_global_task_ids"] == 102
    assert data["selection"]["selected_compute_global_task_ids"] == 59
    assert data["selection"]["task_rows_copied"] > data["selection"]["selected_task_rows"]
    assert data["selection"]["ambiguous_selected_global_task_ids"]["0"] == 26
    assert data["selection"]["ambiguous_selected_global_task_ids"]["1"] == 26
    assert data["selection"]["ambiguous_selected_global_task_ids"]["466"] == 26
    assert data["selection"]["pmu_rows_copied"] == 26152
    assert "HOST_INFO" in data["excluded_tables"]
    assert "ENV_VARIABLES" not in json.dumps(data["tables_copied"])


def test_real_db_decodes_string_ids_and_binds_unique_wait_pmu():
    schema, _source, events, stats = _events()
    assert schema.wait_column == "waitNs"
    assert schema.pmu_table_present is True
    event = _by_global(events, 1063)[0]
    # Old failure: these were the raw STRING_IDS integers 370/379/368.
    assert event.input_shapes == '"2;1"'
    assert event.input_dtypes == "INT32;INT64"
    assert event.input_formats == "ND;ND"
    assert event.input_shapes not in {"370", "379", "368"}
    assert event.wait_ns == 177104
    assert event.wait_status == "bound"
    assert event.wait_source == "COMPUTE_TASK_INFO.waitNs"
    assert event.task_occurrences_full_capture == 1
    assert event.pmu_status == "bound"
    metrics = dict(event.pmu)
    assert len(metrics) == 28
    assert metrics["aiv_total_time"] == pytest.approx(1678.1818181818182)
    assert metrics["aiv_total_cycles"] == 2769.0
    assert event.pmu_provenance["binding"] == "unique_task_occurrence"
    assert event.pmu_provenance["raw_rows"] == 28


def test_repeated_global_task_id_is_group_scoped_not_multiplied_or_picked():
    _schema, _source, events, stats = _events()
    repeated = _by_global(events, 0)[0]
    assert repeated.task_occurrences_full_capture == 26
    assert repeated.wait_ns is None
    assert repeated.wait_status == "ambiguous_task_occurrence"
    assert repeated.pmu == ()
    assert repeated.pmu_status == "ambiguous_task_occurrence"
    provenance = repeated.pmu_provenance
    assert provenance["raw_rows"] == 728
    assert provenance["metric_count"] == 28
    assert provenance["binding"] == "group_scoped_only"
    assert provenance["per_metric"]["aiv_total_time"]["rows"] == 26
    assert provenance["per_metric"]["aiv_total_time"]["all_zero"] is True
    assert provenance["per_metric"]["aiv_total_cycles"]["rows"] == 26
    assert provenance["per_metric"]["aiv_total_cycles"]["distinct_values"] == 26
    assert stats.compute_rows == 59
    assert stats.pmu_rows == 26152
    assert stats.pmu_bound_rows == 24
    assert stats.pmu_ambiguous_rows == 35
    assert stats.ambiguous_groups == 35
    # A naive TASK x PMU join would produce 26 copies of each group value.
    assert sum(len(event.pmu) for event in events) == 24 * 28


def test_full_capture_occurrences_of_repeated_id_remain_unavailable():
    schema = probe_db(DB_PATH)
    source = DbSource(schema)
    events, stats = source.iter_window(Window(0, 2**63 - 1), max_rows=1_000_000)
    repeated = _by_global(events, 0)
    assert len(repeated) == 26
    assert all(event.wait_ns is None for event in repeated)
    assert all(event.pmu == () for event in repeated)
    assert all(event.pmu_status == "ambiguous_task_occurrence" for event in repeated)
    assert stats.pmu_rows == 26152


def test_clipped_event_does_not_use_full_wait_or_relabel_pmu():
    report = analyze_profile(DB_PATH, windows=[CLIP_WINDOW], source="db")
    wait = report["enrichment"]["anomaly_discovery"]["wait_anchor"]
    assert wait["status"] == "unverified"
    assert wait["wait_anchor_false_hotspot"] is None
    assert wait["measurement_scope"]["wait_clipped_event_count"] >= 1
    assert wait["measurement_scope"]["wait_usable_event_count"] == 0
    codes = {item["code"] for item in report["limitations"]}
    assert "wait_time_clipped_event_not_used_for_wait_anchor" in codes
    pmu = report["enrichment"]["pmu_observations"]
    assert pmu["counts"]["clipped_bound_event_count"] >= 1
    assert pmu["counts_are_full_event"] is True
    clipped = next(item for item in pmu["bound_events"] if item["task_global_id"] == 1063)
    assert clipped["complete_in_window"] is False
    assert clipped["counts_are_full_event"] is True
    assert clipped["usable_for_clipped_window"] is False
    assert 0.0 < clipped["window_coverage"] < 1.0
    assert clipped["metric_count"] == 28
    assert "pmu_full_event_values_not_clipped_window" in {item["code"] for item in pmu["limitations"]}
    assert all(item["wait_anchor_false_hotspot"] is None for item in report["top_hotspot_kernels"])


def test_real_csv_exact_wait_and_pmu_units():
    schema = probe_csv(CSV_PATH)
    assert schema.wait_column == "Wait Time(us)"
    assert len(schema.pmu_columns) == 25
    source = CsvSource(schema, rank_id="rank_0", rank_index=0)
    events, stats = source.read_windows([Window(1763456164891087000, 1763456164891090000)])[0]
    event = next(item for item in events if item.task_id == "44303")
    assert event.wait_ns == 177310  # 177.310 us -> ns
    assert event.wait_status == "bound"
    assert event.pmu_status == "bound"
    metrics = dict(event.pmu)
    assert len(metrics) == 25
    assert metrics["aiv_total_time"] == 1678.0  # 1.678 us -> ns
    assert metrics["aiv_total_cycles"] == 2769.0
    assert metrics["cube_utilization_pct"] == 0.0
    assert "2;1" in event.input_shapes
    assert event.pmu_provenance["binding"] == "per_row"
    assert event.pmu_provenance["raw_rows"] == 25
    assert stats.wait_bound_rows == 1
    assert stats.pmu_bound_rows == 1


def test_unresolved_string_id_is_visible_not_numeric(tmp_path):
    copied = tmp_path / "slice.db"
    shutil.copyfile(DB_PATH, copied)
    con = sqlite3.connect(copied)
    con.execute("UPDATE COMPUTE_TASK_INFO SET inputShapes=999999 WHERE globalTaskId=1063")
    con.commit()
    con.close()
    schema = probe_db(copied)
    events, stats = DbSource(schema).iter_window(WINDOW, max_rows=1_000_000)
    event = _by_global(events, 1063)[0]
    assert event.input_shapes == "<unresolved:999999>"
    assert stats.unresolved_string_ids == 1


def test_real_pmu_observations_and_repeated_groups_are_explicit():
    report = analyze_profile(DB_PATH, windows=[WINDOW], source="db")
    pmu = report["enrichment"]["pmu_observations"]
    assert pmu["schema_version"] == "mindie-pmu-observation/1"
    assert pmu["status"] == "partial"
    assert pmu["attribution"] == "measurement_scope_only"
    assert pmu["counts_are_full_event"] is True
    assert "non-AICPU" in pmu["scope"]
    assert pmu["counts"]["bound_event_count"] == 21
    assert pmu["counts"]["ambiguous_event_count"] == 35
    assert pmu["counts"]["raw_pmu_rows"] == 26068
    assert pmu["source_stats"] == {"pmu_rows": 26152, "ambiguous_groups": 35}
    groups = {item["global_task_id"]: item for item in pmu["ambiguous_groups"]}
    assert set(groups) >= {0, 1, 466}
    assert groups[0]["raw_rows"] == 728
    assert groups[0]["task_occurrences_full_capture"] == 26
    assert groups[0]["per_metric"]["aiv_total_time"]["rows"] == 26
    assert pmu["units"]["ns"].startswith("nanoseconds")
    assert any(item["code"] == "pmu_ambiguous_task_occurrence" for item in pmu["limitations"])
    wait = report["enrichment"]["anomaly_discovery"]["wait_anchor"]
    assert wait["measurement_scope"]["wait_bound_event_count"] == 24
    assert wait["measurement_scope"]["wait_ambiguous_event_count"] == 35


def test_aicpu_core_pmu_does_not_establish_aicpu_counters(tmp_path):
    csv_path = tmp_path / "rank_0" / "kernel_details.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(
        "Name,Start Time(us),Duration(us),Accelerator Core,Wait Time(us),"
        "aicore_time(us),aiv_time(us),aiv_total_cycles\n"
        "ai_core_op,0.000,10.000,AI_CORE,0.000,5.000,5.000,100\n"
        "ai_cpu_op,20.000,10.000,AI_CPU,0.000,5.000,5.000,100\n",
        encoding="utf-8",
    )
    report = analyze_profile(csv_path, windows=[Window(0, 60_000, step=None, label="aicpu")])
    window = report["enrichment"]["anomaly_discovery"]["aicpu_exposure"]["by_window"][0]
    assert window["interval_assessment"]["counter_status"] == "missing"
    assert report["enrichment"]["pmu_observations"]["aicpu_counters"]["counter_status"] == "missing"
    assert "aicpu_pmu_counters_missing" in {item["code"] for item in report["limitations"]}
    # The AI-core row may carry PMU; the AICPU row must not inherit it.
    parsed_schema = probe_csv(csv_path)
    parsed = CsvSource(parsed_schema, rank_id="rank_0", rank_index=0).read_windows(
        [Window(0, 60_000)]
    )[0][0]
    aicpu = next(item for item in parsed if item.name == "ai_cpu_op")
    assert aicpu.pmu == ()
    assert aicpu.pmu_status == "not_applicable"


def test_real_pmu_derived_evidence_is_full_event_and_wait_scope_is_observation_local():
    report = analyze_profile(DB_PATH, windows=[WINDOW], source="db", top_n=50)
    pmu = report["enrichment"]["pmu_observations"]
    bound = next(
        item for item in pmu["bound_events"]
        if item["derived"]["aic"]["total_time_ns"] or item["derived"]["aiv"]["total_time_ns"]
    )
    assert bound["metric_units"]
    assert bound["derived"]["scope"]["kind"] == "full_event"
    assert bound["derived"]["scope"]["usable_for_clipped_window"] is True
    assert bound["derived"]["causal"] is False
    assert bound["derived"]["attribution"] == "rule_candidate_not_root_cause"
    assert "pipe_time_fractions" in bound["derived"]["aic"]
    assert "ratio_time_consistency" in bound["derived"]["aic"]
    assert bound["derived"]["cube_utilization_pct"]["status"] in {"present", "absent"}
    wait = report["enrichment"]["anomaly_discovery"]["wait_anchor"]
    assert wait["findings"]
    for finding in wait["findings"]:
        assert finding["candidate_events"]
        for candidate in finding["candidate_events"]:
            assert candidate["window"]["start_ns"] == WINDOW.start_ns
            assert candidate["window"]["end_ns"] == WINDOW.end_ns
            assert candidate["observation_id"]
            assert candidate["hardware_scope"]["status"] == "not_available"
